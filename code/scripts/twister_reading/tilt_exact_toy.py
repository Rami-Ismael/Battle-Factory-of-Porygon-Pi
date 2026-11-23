"""Exact check that the lookahead tilts the 84-field model's draws, on a sub-problem small enough to enumerate.

One Pokémon (slot 0) shows Fake Out as its first move; every other field of the team stays masked. Three fields are
drawn in random-order fashion: ability, then second move, then species. Every path is enumerated with the model's own
probabilities (tables mask applied, guidance as in the grid), so both laws below are exact:

    target    p(a) p(m|a) p(s|a,m) on paths that end with a legal species, renormalised
              (the model's own law restricted to legal outcomes; draw-then-reject gives this)
    sampler   the lookahead law: each draw drops values that leave no legal species and renormalises

Then each law is also drawn N times by sampling, to show the gap with real draws.

    AG84_MODEL=sched_linear_random python tilt_exact_toy.py out.json
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import ask_grid84 as A
import diffusion84 as M

ASK, GUIDANCE, N_DRAWS, SEED = 15 / 24, 2.0, 200_000, 7300
C_SPECIES, C_ABILITY, C_MOVE2 = 0, 1, 4


@torch.no_grad()
def column_logprobs(w, x, c):
    """Guided log-probabilities for column c at states x, as Model84.sample combines them."""
    n = len(x); t = (1.0 - (x != 0).sum(1).float() / M.COLS).to(M.DEV)
    ask = torch.full((n,), ASK, device=M.DEV); none = torch.full((n,), M.NAN, device=M.DEV)
    out = []
    for i in range(0, n, 512):
        sl = slice(i, i + 512)
        lc = w._column(w._logprobs(x[sl], t[sl], ask[sl]), c)
        lc = GUIDANCE * lc + (1 - GUIDANCE) * w._column(w._logprobs(x[sl], t[sl], none[sl]), c)
        out.append(lc)
    return torch.cat(out)


def masked_probs(lc, mask):
    return torch.softmax(lc.masked_fill(~mask, -float("inf")), -1)


def main(path):
    w = A.load_model(); V = w.tables.V
    x0 = torch.zeros(1, M.COLS, dtype=torch.long, device=M.DEV); x0[0, 3] = V.stoi["move"]["fakeout"]

    # step 1: ability
    lc = column_logprobs(w, x0, C_ABILITY)
    _, plain, kept, _ = w._legal_parts(x0, C_ABILITY, lookahead=True)
    p_a = masked_probs(lc, plain)[0]                       # target's first draw: tables only
    q_a = masked_probs(lc, kept)[0]                        # sampler's first draw: lookahead
    abilities = kept[0].nonzero().squeeze(1)               # only these can end legal

    # step 2: second move, one state per kept ability
    x1 = x0.repeat(len(abilities), 1); x1[:, C_ABILITY] = abilities
    lc = column_logprobs(w, x1, C_MOVE2)
    _, plain, kept, _ = w._legal_parts(x1, C_MOVE2, lookahead=True)
    p_m = masked_probs(lc, plain); q_m = masked_probs(lc, kept)
    keep_share = (p_m * kept).sum(1)                       # Z_m(a): the share of p(m|a) the lookahead keeps

    # step 3: species, one state per (ability, kept move)
    ai, mi = kept.nonzero(as_tuple=True)
    x2 = x1[ai].clone(); x2[:, C_MOVE2] = mi
    lc = column_logprobs(w, x2, C_SPECIES)
    _, plain, _, fallback = w._legal_parts(x2, C_SPECIES, lookahead=True)
    assert not fallback.any(), "the lookahead let a path reach a dead end"
    p_s = masked_probs(lc, plain)                          # [paths, species]; identical for both laws

    # joint over (ability, move, species), then marginals
    path_target = p_a[abilities[ai]] * p_m[ai, mi]         # target, before renormalising
    path_sampler = q_a[abilities[ai]] * q_m[ai, mi]
    p_valid = float(path_target.sum())                     # P(legal) when drawing then rejecting
    path_target = path_target / path_target.sum()
    joint_t = path_target.unsqueeze(1) * p_s; joint_q = path_sampler.unsqueeze(1) * p_s
    sp_t, sp_q = joint_t.sum(0), joint_q.sum(0)
    ab_t = torch.zeros_like(p_a).index_add_(0, abilities[ai], path_target)
    ab_q = torch.zeros_like(p_a).index_add_(0, abilities[ai], path_sampler)

    tv = lambda a, b: 0.5 * float((a - b).abs().sum())
    kl = lambda a, b: float((a * (a.clamp_min(1e-30).log() - b.clamp_min(1e-30).log())).sum())

    # real draws: draw-then-reject from the target's tables-only law; the lookahead sampler from its own law
    g = torch.Generator(device="cpu").manual_seed(SEED)
    pa_cpu, pm_cpu, ps_cpu = p_a.cpu(), p_m.cpu(), p_s.cpu()
    idx_of = {(int(a), int(m)): k for k, (a, m) in enumerate(zip(ai.tolist(), mi.tolist(), strict=True))}
    pos_of_ability = {int(a): k for k, a in enumerate(abilities.tolist())}
    a_draw = torch.multinomial(pa_cpu, N_DRAWS, replacement=True, generator=g)
    accepted = []
    for a in a_draw.tolist():
        k = pos_of_ability.get(a)
        if k is None: continue                            # dead end: no Fake Out user has this ability
        m = int(torch.multinomial(pm_cpu[k], 1, generator=g))
        j = idx_of.get((k, m))
        if j is None: continue                            # dead end: no species has Fake Out, a and m
        accepted.append(int(torch.multinomial(ps_cpu[j], 1, generator=g)))
    rej_sp = torch.bincount(torch.tensor(accepted), minlength=len(sp_t)).float() / len(accepted)
    jq = joint_q.reshape(-1).cpu()
    flat = torch.multinomial(jq / jq.sum(), N_DRAWS, replacement=True, generator=g)
    look_sp = torch.bincount(flat % joint_q.shape[1], minlength=len(sp_t)).float() / N_DRAWS

    names_sp, names_ab = V.itos["species"], V.itos["ability"]
    top = torch.argsort(torch.maximum(sp_t, sp_q).cpu(), descending=True)[:12].tolist()
    weights = keep_share.cpu()
    out = dict(
        model=os.environ.get("AG84_MODEL"), ask=ASK, guidance=GUIDANCE, n_draws=N_DRAWS,
        abilities_tables=int((p_a > 0).sum()), abilities_kept=len(abilities), paths=len(ai),
        p_valid_draw_then_reject=p_valid, accepted=len(accepted),
        tv_species=tv(sp_t, sp_q), kl_species=kl(sp_t, sp_q), tv_ability=tv(ab_t, ab_q),
        tv_joint=tv(joint_t.reshape(-1), joint_q.reshape(-1)),
        tv_species_draws=tv(rej_sp, look_sp),
        keep_share_min=float(weights.min()), keep_share_max=float(weights.max()),
        log_weight_spread=float(weights.log().max() - weights.log().min()),
        species=[dict(name=names_sp[s], target=float(sp_t[s]), sampler=float(sp_q[s]),
                      target_draws=float(rej_sp[s]), sampler_draws=float(look_sp[s])) for s in top],
        abilities=sorted([dict(name=names_ab[int(a)], target=float(ab_t[a]), sampler=float(ab_q[a]),
                               keep_share=float(weights[k]), ratio=float(ab_q[a] / ab_t[a]))
                          for k, a in enumerate(abilities.tolist())], key=lambda r: -max(r["target"], r["sampler"])),
    )
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("species", "abilities")}, indent=1))
    for r in out["species"]: print(f"{r['name']:18s} target {r['target']:.3f} ({r['target_draws']:.3f})  sampler {r['sampler']:.3f} ({r['sampler_draws']:.3f})")
    for r in out["abilities"][:10]: print(f"{r['name']:16s} target {r['target']:.3f} sampler {r['sampler']:.3f} kept share of move 2 {r['keep_share']:.2f}")
    assert math.isclose(float(joint_q.sum()), 1.0, rel_tol=1e-4)


if __name__ == "__main__":
    main(sys.argv[1])
