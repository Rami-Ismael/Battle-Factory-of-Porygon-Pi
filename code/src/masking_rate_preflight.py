"""Battle-free probes for the four masking-rate rows, run before any of their trainings or grids
(docs/masking-rate-rows.md; issue #294 on the masking-rate map #291).

  hidden_share      with every exponent 1 (the linear masking rate) the share of fields hidden at t is t, on every field
  corrected_t       the expected-share rule of the corrected-t ticket (#293): the t each of the 84 passes gets, checked
                    against the values that ticket published; under the linear masking rate it gives 1 - filled/84
  t_sensitivity     the ceiling for the corrected-t row: with the same sampling seed, how far the default model's drawing
                    distributions and teams move when a pass is told the corrected t and not 1 - filled/84
  cross_rate_loss   the default model's validation loss under its own per-field masking rate and under the linear one,
                    so the linear model's loss is read against the right number
  training_smoke    the linear masking rate through the real training code: per-field cross-entropy at the start against
                    ln K, then 32 teams overfit for 300 epochs, beside the per-field rate on the same teams and seed
  recipe_replay     one epoch of the default model's own recipe (per-field rate) with today's code and table must land on
                    the losses that model logged at epoch 1, so the new training differs from it in the masking rate alone

Nothing here changes diffusion84.py; the sampler below copies Model84.sample's loop and only swaps the t of each pass.
Results: ~/vgc-data/results/masking_rate_rows_preflight.json

    /tmp/vgc-pilot/.venv/bin/python src/masking_rate_preflight.py [probe ...]      (from code/; default all six)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train84 as TR                                    # puts src/ and /tmp/vgc-pilot/src on sys.path, as when training
import asked_vs_got as X
import diffusion84 as M
import lossdown as L
import position_split_preflight as PP
import team84 as T
from checkpoints import load_checkpoint

LINEAR = "1,1,1,1,1"                                    # every field hidden with chance t; Stat Points already use exponent 1
DEFAULT_MODEL = L.CK / "stat84_labelled_noise_byepoch_es3_cap80" / "stat84_labelled_noise_byepoch_es3_cap80_ep80.pt"
DEFAULT_MODEL_SHA256 = "c057b8ef47c9bb7446d0155524794568ba6d39f532d695113eeafc8ee17aaa7b"   # recorded by its 23-ask grid
OUT = Path.home() / "vgc-data/results/masking_rate_rows_preflight.json"
GUIDANCE, SAMPLING_SEED, TEAMS_PER_ASK = 2.0, 7301, 256                  # the ask grid's guidance and its first sampling seed
ASK_NUMERATORS = (6, 12, 17)
# The corrected-t ticket's table (code/docs/research/corrected-t-for-per-field-masking-rate.md, branch research/corrected-t):
# fields filled -> t under the expected-share rule.
PUBLISHED_CORRECTED_T = {0: 1.0, 6: 0.9701, 12: 0.9338, 18: 0.8901, 24: 0.8391, 42: 0.6497, 48: 0.5759, 60: 0.4120, 78: 0.1193, 83: 0.0220}
MIN_TOTAL_VARIATION = 0.01          # set before the run: below this on every field type, the corrected t cannot change a result
REPLAY_TOLERANCE = 0.01             # set before the run: epoch-1 losses of today's code against the default model's log
PASS_GROUPS = dict(species=range(0, 6), ability_and_item=range(6, 18), moves=range(18, 42), stat_alignment=range(42, 48),
                   stat_points=range(48, 84))           # passes of the species-first order, by the field type they fill


def corrected_t(exponents, filled):
    """The t whose expected hidden share in training, mean(t ** exponent), equals the share of fields still hidden."""
    still_hidden = 1 - filled / len(exponents)
    if still_hidden >= 1: return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if float((middle ** exponents).mean()) < still_hidden: low = middle
        else: high = middle
    return (low + high) / 2


def hidden_share():
    """Mask draws exactly as batch_loss makes them, at seven values of t."""
    torch.manual_seed(0)
    linear, per_field = M.sched84(LINEAR), M.sched84(M.SCHED)
    rows, worst = {}, 0.0
    for t in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        hidden = (torch.rand(40000, M.COLS) < torch.full((40000, 1), t) ** linear.view(1, -1)).float().mean(0)
        worst = max(worst, float((hidden - t).abs().max()))
        rows[str(t)] = dict(linear_hidden_share=float(hidden.mean()), per_field_expected_hidden_share=float((t ** per_field).mean()))
    return dict(every_exponent_is_1=bool((linear == 1).all()), loss_weight="exponent / t = 1 / t", by_t=rows,
                largest_gap_any_field=worst, passed=bool((linear == 1).all()) and worst < 0.01)


def corrected_t_values():
    per_field, linear = M.sched84(M.SCHED).double().numpy(), M.sched84(LINEAR).double().numpy()
    per_pass = [corrected_t(per_field, filled) for filled in range(M.COLS)]
    gap_from_ticket = max(abs(per_pass[k] - v) for k, v in PUBLISHED_CORRECTED_T.items())
    gap_linear = max(abs(corrected_t(linear, k) - (1 - k / M.COLS)) for k in range(M.COLS))
    return dict(rule="t solves mean(t ** exponent) = 1 - filled/84", per_pass=per_pass, today=[1 - k / M.COLS for k in range(M.COLS)],
                at_half_filled=per_pass[42], largest_gap_from_ticket_table=gap_from_ticket, largest_gap_under_linear_rate=gap_linear,
                passed=gap_from_ticket < 5e-4 and gap_linear < 1e-9)


@torch.no_grad()
def sample_with_t(model, n, ask, t_of_pass, seed, other_rules=None):
    """Model84.sample's loop (species-first order, guidance 2, temperature 1) with pass k told t_of_pass[k]. other_rules:
    {name: a second t per pass}; then also returns, per rule and pass, the mean total variation between that rule's
    drawing distribution and this one's on the same partly filled teams."""
    torch.manual_seed(seed)
    ask_w, null_w = torch.full((n,), ask, device=M.DEV), torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV); total_variation = {name: [] for name in other_rules or {}}

    def drawing_distribution(t_value, c):
        t = torch.full((n,), t_value, device=M.DEV)
        log_odds = GUIDANCE * model._column(model._logprobs(x, t, ask_w), c) + (1 - GUIDANCE) * model._column(model._logprobs(x, t, null_w), c)
        return torch.softmax(log_odds.masked_fill(~model._legal(x, c), -float("inf")), -1)

    for step, c in enumerate(M.ORDER):
        p = drawing_distribution(t_of_pass[step], c)
        for name, other_t in (other_rules or {}).items():
            total_variation[name].append(float(0.5 * (p - drawing_distribution(other_t[step], c)).abs().sum(-1).mean()))
        v = torch.multinomial(p, 1).squeeze(1)
        x[:, c] = v + 1 if c >= M.STAT0 else v
    return x.cpu().numpy(), total_variation


def t_sensitivity(model):
    """passed judges the corrected t alone. The two extreme rules (every pass told t = 1, or t = 1/84) show how much
    any t could move the model."""
    today = [1.0 - step / M.COLS for step in range(M.COLS)]
    corrected = [corrected_t(M.sched84(model.info.get("sched", M.SCHED)).double().numpy(), k) for k in range(M.COLS)]
    rules = dict(corrected=corrected, always_1=[1.0] * M.COLS, always_1_over_84=[1 / M.COLS] * M.COLS)
    species_columns = [slot * 8 for slot in range(6)]
    by_ask, largest = {}, 0.0
    for k in ASK_NUMERATORS:
        reference = model.sample(TEAMS_PER_ASK, ask=k / 24, guidance=GUIDANCE, seed=SAMPLING_SEED).cpu().numpy()
        same, total_variation = sample_with_t(model, TEAMS_PER_ASK, k / 24, today, SAMPLING_SEED, other_rules=rules)
        moved, _ = sample_with_t(model, TEAMS_PER_ASK, k / 24, corrected, SAMPLING_SEED)
        by_type = {rule: {name: float(np.mean([per_pass[i] for i in passes])) for name, passes in PASS_GROUPS.items()}
                   for rule, per_pass in total_variation.items()}
        largest = max(largest, *by_type["corrected"].values())
        by_ask[f"{k}/24"] = dict(
            copy_matches_model84_sample=bool((same == reference).all()), total_variation_by_rule_and_field_type=by_type,
            share_of_teams_changed=float((moved != same).any(1).mean()), share_of_fields_changed=float((moved != same).mean()),
            share_of_line_ups_changed=float(np.mean([sorted(a[species_columns]) != sorted(b[species_columns]) for a, b in zip(moved, same)])))
    copies = all(v["copy_matches_model84_sample"] for v in by_ask.values())
    return dict(teams_per_ask=TEAMS_PER_ASK, sampling_seed=SAMPLING_SEED, guidance=GUIDANCE, bar=MIN_TOTAL_VARIATION, by_ask=by_ask,
                largest_total_variation_corrected=largest, passed=copies and largest >= MIN_TOTAL_VARIATION)


def cross_rate_loss(model, d):
    labelled = ~np.isnan(d["all_W"]); rows = d["all_val"] & labelled
    vx, vw = d["all_X"][rows], d["all_W"][rows]
    own = M.val_loss(model.net, model.tables, vx, vw, M.sched84(M.SCHED).to(M.DEV))
    linear = M.val_loss(model.net, model.tables, vx, vw, M.sched84(LINEAR).to(M.DEV))
    logged = load_checkpoint(DEFAULT_MODEL.parent.with_suffix(".pt"), map_location="cpu")["info"]["best_val_loss"]   # its best epoch is 80
    return dict(validation_teams=int(rows.sum()), teams_the_tables_reject=int(model.tables.check(vx[:, :M.STAT0]).sum()),
                per_field_rate=own, linear_rate=linear, logged_at_epoch_80=logged, passed=abs(own - logged) < 1e-3)


def training_smoke(d, V, corpus):
    labelled = np.flatnonzero(~d["all_val"] & ~np.isnan(d["all_W"]))[:256]
    x, w = torch.as_tensor(d["all_X"][labelled]), d["all_W"][labelled]
    torch.manual_seed(TR.RECIPE["seed"]); start = PP.start_loss(V, M.Net84(V, **TR.NET).eval(), x)
    out = dict(start=start, calibration=PP.CALIBRATION, device=M.DEV)
    for name, exponents in (("per_field", M.SCHED), ("linear", LINEAR)):
        t0 = time.perf_counter()
        m = M.Model84.train_new(V, corpus, d["all_X"][labelled[:32]], w[:32], epochs=PP.OVERFIT_EPOCHS, batch=32, warmup=10,
                                val=(d["all_X"][labelled[:32]], w[:32]), label_noise=1 / 48, seed=TR.RECIPE["seed"], lr=TR.RECIPE["lr"],
                                sched=exponents, **TR.NET)
        c = m.info["curves"]
        out[name] = dict(exponents=m.info["sched"], first_validation_loss=c[0]["val_loss"], last_validation_loss=c[-1]["val_loss"],
                         first_training_loss=c[0]["train_loss"], last_training_loss=c[-1]["train_loss"],
                         minutes=round((time.perf_counter() - t0) / 60, 2),
                         passed=c[-1]["val_loss"] <= (1 - PP.OVERFIT_FALL) * c[0]["val_loss"])
    out["passed"] = all(v["passed"] for v in start.values()) and out["per_field"]["passed"] and out["linear"]["passed"]
    return out


def recipe_replay(d, V, corpus):
    """The default model's training call (train84.py, arm byepoch) stopped after one epoch. early_stopping_threshold=0 is
    its stopping rule: it trained before the 0.001 threshold existed, and any new lowest validation loss reset patience."""
    logged = load_checkpoint(DEFAULT_MODEL.parent.with_suffix(".pt"), map_location="cpu")["info"]
    labelled = ~np.isnan(d["all_W"]); train, validation = ~d["all_val"] & labelled, d["all_val"] & labelled
    m = M.Model84.train_new(V, corpus, d["all_X"][train], d["all_W"][train], epochs=80, val=(d["all_X"][validation], d["all_W"][validation]),
                            patience=3, early_stopping_threshold=0.0, label_noise=1 / 48, stop_after=1, **TR.RECIPE, **TR.NET)
    now, then = m.info["curves"][0], logged["curves"][0]
    gaps = dict(training_loss=abs(now["train_loss"] - then["train_loss"]), validation_loss=abs(now["val_loss"] - then["val_loss"]))
    same_table = d["sha256"] == logged["data_sha256"]
    return dict(epoch_1_now=dict(training_loss=now["train_loss"], validation_loss=now["val_loss"], minutes=now["minutes"]),
                epoch_1_logged=dict(training_loss=then["train_loss"], validation_loss=then["val_loss"], minutes=then["minutes"]),
                teams_trained_now=m.info["n_train"], teams_trained_logged=logged["n_train"], same_table=same_table, gaps=gaps,
                bar=REPLAY_TOLERANCE, passed=same_table and m.info["n_train"] == logged["n_train"] and max(gaps.values()) < REPLAY_TOLERANCE)


def main(names):
    results = json.load(open(OUT)) if OUT.exists() else {}
    need_model = {"t_sensitivity", "cross_rate_loss"} & set(names)
    need_data = {"cross_rate_loss", "training_smoke", "recipe_replay"} & set(names)
    model = d = None
    if need_model:
        sha = hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
        assert sha == DEFAULT_MODEL_SHA256, f"{DEFAULT_MODEL} is not the checkpoint the default model's grid recorded"
        model = M.Model84.load(DEFAULT_MODEL); results["default_model"] = dict(checkpoint=DEFAULT_MODEL.name, sha256=sha)
    if need_data: d = TR.load(); results["table"] = dict(name=d["table"], sha256=d["sha256"])
    probes = dict(hidden_share=hidden_share, corrected_t=corrected_t_values, t_sensitivity=lambda: t_sensitivity(model),
                  cross_rate_loss=lambda: cross_rate_loss(model, d),
                  training_smoke=lambda: training_smoke(d, T.regulation_vocab(), X.setup()[0]),
                  recipe_replay=lambda: recipe_replay(d, T.regulation_vocab(), X.setup()[0]))
    for name in names:
        results[name] = probes[name]()
        json.dump(results, open(OUT.with_suffix(".tmp"), "w"), indent=1, default=float); OUT.with_suffix(".tmp").replace(OUT)
        shown = {k: v for k, v in results[name].items() if k not in ("per_pass", "today", "calibration")}
        print(f"[{name}] {'PASS' if results[name]['passed'] else 'FAIL'}\n{json.dumps(shown, indent=1, default=float)}", flush=True)
    print("MASKING_RATE_PREFLIGHT_DONE", "PASS" if all(results[n]["passed"] for n in names) else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1:] or ["hidden_share", "corrected_t", "t_sensitivity", "cross_rate_loss", "training_smoke", "recipe_replay"])
