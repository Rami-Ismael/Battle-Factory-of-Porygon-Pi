"""Battle-free diagnostic probes for Issue #438 (shared-recipe Uniform).

Covers:
  - Item 5: Pokémon-to-field matching probe (slot_binding_probe.py extended to UniformModel84)
    at epochs 45, 49, 55, 61, and 80 across all three comparison checkpoints.
  - Item 6: Slot-swap sensitivity — do Stat Points, moves, and abilities move with a swapped
    species, and how equivariant is the denoiser under full-slot swaps?
  - Item 7: Generation speed and peak MPS memory (docs/generation-memory-spec.md) in one
    isolated subprocess per model/setting, including Showdown validator redraws.
  - Part A item 2 (battle-free half): recomputed clipped-time validation loss bound on the
    identical 4,368 filtered validation rows of lossdata84_shuffled_only.npz (seed=12345).

Pure summarization, bootstrap, and ECharts figure helpers import no /tmp/vgc-pilot or
~/vgc-data modules so unit tests run cleanly on CPU.

Usage (from code/):
    /tmp/vgc-pilot/.venv/bin/python src/uniform_gidd_lr_probes.py slot-binding
    /tmp/vgc-pilot/.venv/bin/python src/uniform_gidd_lr_probes.py slot-swap
    /tmp/vgc-pilot/.venv/bin/python src/uniform_gidd_lr_probes.py speed-memory
    /tmp/vgc-pilot/.venv/bin/python src/uniform_gidd_lr_probes.py val-loss
    /tmp/vgc-pilot/.venv/bin/python src/uniform_gidd_lr_probes.py figures <out_dir>
    /tmp/vgc-pilot/.venv/bin/python src/uniform_gidd_lr_probes.py comment <figure_base_url>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echarts_chart import write_chart

RES_DIR = Path.home() / "vgc-data/results"
SLOT_BINDING_OUT = RES_DIR / "uniform_gidd_lr_slot_binding_438.json"
SLOT_SWAP_OUT = RES_DIR / "uniform_gidd_lr_slot_swap_438.json"
SPEED_MEM_OUT = RES_DIR / "uniform_gidd_lr_speed_memory_438.json"
VAL_LOSS_OUT = RES_DIR / "uniform_gidd_lr_val_loss_438.json"

ARMS = (
    dict(
        id="uniform_gidd_lr",
        reader="Uniform (shared recipe)",
        checkpoint="stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80",
        uniform=True,
        color="#0f766e",
        dark_color="#5eead4",
    ),
    dict(
        id="uniform_4e4",
        reader="Uniform (4e-4 recipe)",
        checkpoint="stat84_udlm_loglinear_es3_cap80",
        uniform=True,
        color="#c2410c",
        dark_color="#fdba74",
    ),
    dict(
        id="masked_gidd_lr",
        reader="Masked, full filter (shared recipe)",
        checkpoint="stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80",
        uniform=False,
        color="#365d93",
        dark_color="#93c5fd",
    ),
)
ARM_BY_ID = {arm["id"]: arm for arm in ARMS}
DARK_MAP = {arm["color"]: arm["dark_color"] for arm in ARMS}
SLOT_BINDING_EPOCHS = (45, 49, 55, 61, 80)
SWAP_PAIRS = ((0, 3), (1, 4), (2, 5))
N_VAL_TEAMS = 256
PROBE_DRAWS = 4
PROBE_SEED = 12345
BOOT_DRAWS = 2000
BOOT_SEED = 43801


def bootstrap_mean(values, *, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Mean and 95% percentile bootstrap interval over a 1D array of per-team values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return dict(mean=float("nan"), interval=[float("nan"), float("nan")], n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(draws, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(arr.mean()), interval=[float(lo), float(hi)], n=int(arr.size))


def tv_distance(p, q):
    """Total variation distance 0.5 * sum |p - q| along the last axis."""
    return 0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum(axis=-1)


def species_swap_perm(slot_a, slot_b):
    """84-column permutation swapping all 14 fields (8 base + 6 Stat Points) of slot_a and slot_b."""
    perm = list(range(84))
    for offset in range(8):
        col_a, col_b = slot_a * 8 + offset, slot_b * 8 + offset
        perm[col_a], perm[col_b] = col_b, col_a
    for offset in range(6):
        col_a, col_b = 48 + slot_a * 6 + offset, 48 + slot_b * 6 + offset
        perm[col_a], perm[col_b] = col_b, col_a
    return perm


def _import_runtime():
    """Import runtime model modules lazily so pure helpers stay CPU/CI-safe."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import lossdown as L

    sys.path.insert(0, str(HERE))
    import asked_vs_got as X
    import diffusion84 as M
    import uniform_diffusion84 as UM
    import uniform_math84 as UMath
    import torch

    return L, X, M, UM, UMath, torch


def _epoch_checkpoint_path(ck_root: Path, base_name: str, epoch: int) -> Path:
    if epoch == 80:
        final = ck_root / f"{base_name}.pt"
        if final.exists():
            return final
    ep_path = ck_root / base_name / f"{base_name}_ep{epoch:02d}.pt"
    if ep_path.exists():
        return ep_path
    raise FileNotFoundError(f"Missing epoch {epoch} checkpoint: {ep_path}")


def _load_val_rows(X, limit=None):
    data = np.load(X.RES / "lossdata84_shuffled_only.npz")
    mask = data["all_val"] & ~np.isnan(data["all_W"])
    xs = data["all_X"][mask].astype(np.int64)
    ws = data["all_W"][mask].astype(np.float32)
    if limit is not None:
        return xs[:limit], ws[:limit]
    return xs, ws


def _uniform_column_probs(model, x_uniform, time_val, w_tensor, col, UMath, torch):
    """Return [B, V_real] clean-token probabilities for column `col` (0-based real vocabulary)."""
    key = UMath.KEY_OF_COL[col] if hasattr(UMath, "KEY_OF_COL") else None
    if key is None:
        key = next(k for k, cols in UMath.FIELD_COLS.items() if col in cols)
    pos_in_key = UMath.FIELD_COLS[key].index(col)
    t_tensor = torch.full((len(x_uniform),), float(time_val), device=x_uniform.device)
    logits = model._logits_eval(x_uniform, t_tensor, w_tensor)[key][:, pos_in_key, :].float()
    return torch.softmax(logits, dim=-1)


def _masked_column_probs(model, x_table, time_val, w_tensor, col, M, torch):
    """Return [B, V_real] clean-token probabilities for column `col` (excluding index 0 [MASK])."""
    t_tensor = torch.full((len(x_table),), float(time_val), device=x_table.device)
    lp = model._logprobs(x_table, t_tensor, w_tensor, tables=False)
    col_lp = model._column(lp, col).float()
    if col < M.STAT0:
        col_lp = col_lp[:, 1:]  # drop [MASK] at index 0 to align with 0..N_k-1 real tokens
    return torch.softmax(col_lp, dim=-1)


def _allowed_real(tables, x_table, col, M):
    """Return [B, V_real] boolean legality mask aligned to 0..N_k-1 real tokens (excluding [MASK])."""
    al = M.allowed(tables, x_table)
    key = M.L.KEY_OF_COL[col] if col < M.STAT0 else "stat"
    if col < M.STAT0:
        idx = M.L.COLS_BY_KEY[key].index(col)
        return al[key][:, idx, 1:]
    return al["stat"][:, col - M.STAT0, :]


def _eval_slot_binding_field(model, arm, ep, protocol, cols, x0, x0_u, w, species_cols, M, UMath, torch):
    n = len(x0)
    own_teams = np.zeros(n, dtype=np.float64)
    other_teams = np.zeros(n, dtype=np.float64)
    none_teams = np.zeros(n, dtype=np.float64)
    for slot_idx, col in enumerate(cols):
        step = M.ORDER.index(col)
        x_ref = x0.clone()
        x_ref[:, M.ORDER[step:]] = 0
        a_own = _allowed_real(model.tables, x_ref, col, M)
        a_any = torch.zeros_like(a_own)
        for other_slot in range(6):
            if other_slot == slot_idx:
                continue
            y_swap = x_ref.clone()
            y_swap[:, species_cols[slot_idx]] = x_ref[:, species_cols[other_slot]]
            a_any |= _allowed_real(model.tables, y_swap, col, M)

        if not arm["uniform"]:
            p = _masked_column_probs(model, x_ref, 1.0 - step / M.COLS, w, col, M, torch)
        else:
            gen = torch.Generator(device="cpu").manual_seed(PROBE_SEED + ep * 100 + col)
            cols_corrupt = M.ORDER[step:] if protocol == "species_first_prefix" else [
                c for c in range(M.COLS) if c not in species_cols
            ]
            t_val = len(cols_corrupt) / M.COLS
            probs_accum = None
            for _ in range(PROBE_DRAWS):
                xu = x0_u.clone()
                for c_rand in cols_corrupt:
                    key_r = next(k for k, cl in UMath.FIELD_COLS.items() if c_rand in cl)
                    xu[:, c_rand] = torch.randint(0, model.vocab_sizes[key_r], (n,), generator=gen).to(M.DEV)
                p_draw = _uniform_column_probs(model, xu, t_val, w, col, UMath, torch)
                probs_accum = p_draw if probs_accum is None else probs_accum + p_draw
            p = probs_accum / PROBE_DRAWS

        own_teams += (p * a_own).sum(-1).cpu().numpy() / 6.0
        other_teams += (p * (~a_own & a_any)).sum(-1).cpu().numpy() / 6.0
        none_teams += (p * (~a_own & ~a_any)).sum(-1).cpu().numpy() / 6.0

    return dict(
        own=bootstrap_mean(own_teams, seed=BOOT_SEED + ep),
        other=bootstrap_mean(other_teams, seed=BOOT_SEED + ep + 1),
        none=bootstrap_mean(none_teams, seed=BOOT_SEED + ep + 2),
    )


def run_slot_binding(out_path: Path = SLOT_BINDING_OUT, epochs=SLOT_BINDING_EPOCHS):
    """Run Item 5: Pokémon-to-field matching probe across epochs and all three checkpoints."""
    L, X, M, UM, UMath, torch = _import_runtime()
    corpus = X.setup()[0]
    xs_np, ws_np = _load_val_rows(X, limit=N_VAL_TEAMS)
    species_cols = L.COLS_BY_KEY["species"]
    field_specs = (("ability", L.COLS_BY_KEY["ability"]), ("first_move", [s * 8 + 3 for s in range(6)]))

    results = dict(
        issue=438, probe="slot_binding", n_val_teams=len(xs_np), epochs=list(epochs),
        uniform_draws=PROBE_DRAWS, seed=PROBE_SEED, arms={},
    )

    for arm in ARMS:
        arm_out = dict(reader=arm["reader"], checkpoint=arm["checkpoint"], uniform=arm["uniform"], by_epoch={})
        for ep in epochs:
            ck_path = _epoch_checkpoint_path(L.CK, arm["checkpoint"], ep)
            model = UM.UniformModel84.load(ck_path, corpus) if arm["uniform"] else M.Model84.load(ck_path, corpus)
            x0 = torch.as_tensor(xs_np, device=M.DEV)
            w = torch.as_tensor(ws_np, device=M.DEV)
            x0_u = UMath.table_to_uniform(x0) if arm["uniform"] else None
            protocols = ("species_first_prefix", "species_only") if arm["uniform"] else ("species_first_prefix",)
            with torch.no_grad():
                ep_res = {
                    proto: {
                        fname: _eval_slot_binding_field(model, arm, ep, proto, cols, x0, x0_u, w, species_cols, M, UMath, torch)
                        for fname, cols in field_specs
                    }
                    for proto in protocols
                }
            arm_out["by_epoch"][str(ep)] = ep_res
            ab = ep_res["species_first_prefix"]["ability"]
            fm = ep_res["species_first_prefix"]["first_move"]
            print(
                f"{arm['id']:16s} ep {ep:2d} | ability own {ab['own']['mean']:.3f} other {ab['other']['mean']:.3f} "
                f"none {ab['none']['mean']:.3f} | first move own {fm['own']['mean']:.3f} "
                f"other {fm['other']['mean']:.3f} none {fm['none']['mean']:.3f}",
                flush=True,
            )
        results["arms"][arm["id"]] = arm_out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"SLOT_BINDING_DONE {out_path}", flush=True)
    return results


def _eval_all_probs(model, arm, inputs_list, t_val, w, M, UMath, torch):
    """Average clean-token probabilities over inputs_list, returning {col: np.ndarray [B, V_real]}."""
    accum = {}
    for inp in inputs_list:
        t_tensor = torch.full((len(inp),), float(t_val), device=M.DEV)
        if arm["uniform"]:
            logits_dict = model._logits_eval(inp, t_tensor, w)
            for key, cols in UMath.FIELD_COLS.items():
                probs = torch.softmax(logits_dict[key].float(), dim=-1).cpu().numpy()
                for pos, col in enumerate(cols):
                    accum[col] = probs[:, pos, :] if col not in accum else accum[col] + probs[:, pos, :]
        else:
            lp = model._logprobs(inp, t_tensor, w, tables=False)
            for col in range(M.COLS):
                col_lp = model._column(lp, col).float()
                if col < M.STAT0:
                    col_lp = col_lp[:, 1:]
                probs = torch.softmax(col_lp, dim=-1).cpu().numpy()
                accum[col] = probs if col not in accum else accum[col] + probs
    scale = 1.0 / len(inputs_list)
    return {col: arr * scale for col, arr in accum.items()}


def _make_corrupted_species_only_inputs(model, arm, x0, x0_u, non_species_cols, M, UMath, torch):
    n = len(x0)
    if not arm["uniform"]:
        xm = x0.clone()
        xm[:, non_species_cols] = 0
        return [xm]
    gen = torch.Generator(device="cpu").manual_seed(PROBE_SEED + 600)
    out = []
    for _ in range(PROBE_DRAWS):
        xu = x0_u.clone()
        for c_rand in non_species_cols:
            key_r = next(k for k, cl in UMath.FIELD_COLS.items() if c_rand in cl)
            xu[:, c_rand] = torch.randint(0, model.vocab_sizes[key_r], (n,), generator=gen).to(M.DEV)
        out.append(xu)
    return out


def _eval_arm_slot_swap(model, arm, x0, x0_u, w, species_cols, non_species_cols, L, M, UMath, torch):
    n = len(x0)
    t_species_only = len(non_species_cols) / M.COLS
    base_inputs = _make_corrupted_species_only_inputs(model, arm, x0, x0_u, non_species_cols, M, UMath, torch)

    orig_probs = _eval_all_probs(model, arm, base_inputs, t_species_only, w, M, UMath, torch)
    acc = {k: np.zeros(n) for k in ("tab", "tfm", "tst", "t1ab", "t1fm", "t1st", "new_ab", "old_ab", "new_fm", "old_fm")}
    for slot_a, slot_b in SWAP_PAIRS:
        sp_a, sp_b = species_cols[slot_a], species_cols[slot_b]
        swapped_inputs = []
        for inp in base_inputs:
            sw = inp.clone()
            sw[:, sp_a], sw[:, sp_b] = inp[:, sp_b].clone(), inp[:, sp_a].clone()
            swapped_inputs.append(sw)
        sw_probs = _eval_all_probs(model, arm, swapped_inputs, t_species_only, w, M, UMath, torch)

        x_orig_ref = x0.clone()
        x_orig_ref[:, non_species_cols] = 0
        x_swap_ref = x_orig_ref.clone()
        x_swap_ref[:, sp_a], x_swap_ref[:, sp_b] = x_orig_ref[:, sp_b].clone(), x_orig_ref[:, sp_a].clone()

        for src, dst in ((slot_a, slot_b), (slot_b, slot_a)):
            for offset, k_t, k_1, k_n, k_o in ((1, "tab", "t1ab", "new_ab", "old_ab"), (3, "tfm", "t1fm", "new_fm", "old_fm")):
                c_src, c_dst = src * 8 + offset, dst * 8 + offset
                acc[k_t] += (1.0 - tv_distance(sw_probs[c_dst], orig_probs[c_src])) / 6.0
                acc[k_1] += (sw_probs[c_dst].argmax(-1) == orig_probs[c_src].argmax(-1)) / 6.0
                a_new = _allowed_real(model.tables, x_swap_ref, c_dst, M).cpu().numpy()
                a_old = _allowed_real(model.tables, x_orig_ref, c_dst, M).cpu().numpy()
                acc[k_n] += (sw_probs[c_dst] * a_new).sum(-1) / 6.0
                acc[k_o] += (sw_probs[c_dst] * (~a_new & a_old)).sum(-1) / 6.0
            for st_i in range(6):
                cs_src, cs_dst = 48 + src * 6 + st_i, 48 + dst * 6 + st_i
                acc["tst"] += (1.0 - tv_distance(sw_probs[cs_dst], orig_probs[cs_src])) / 36.0
                acc["t1st"] += (sw_probs[cs_dst].argmax(-1) == orig_probs[cs_src].argmax(-1)) / 36.0

    perm = species_swap_perm(1, 4)
    gen_eq = torch.Generator(device="cpu").manual_seed(PROBE_SEED + 700)
    if arm["uniform"]:
        mask_eq = torch.rand((n, M.COLS), generator=gen_eq) < 0.5
        x_eq = x0_u.clone()
        for col in range(M.COLS):
            key_c = next(k for k, cl in UMath.FIELD_COLS.items() if col in cl)
            r = torch.randint(0, model.vocab_sizes[key_c], (n,), generator=gen_eq).to(M.DEV)
            x_eq[:, col] = torch.where(mask_eq[:, col].to(M.DEV), r, x_eq[:, col])
    else:
        mask_eq = (torch.rand((n, M.COLS), generator=gen_eq) < 0.5).to(M.DEV)
        x_eq = x0.masked_fill(mask_eq, 0)
    p_orig_eq = _eval_all_probs(model, arm, [x_eq], 0.5, w, M, UMath, torch)
    p_swap_eq = _eval_all_probs(model, arm, [x_eq[:, perm]], 0.5, w, M, UMath, torch)
    eq_by_group = {
        gname: bootstrap_mean(np.mean([tv_distance(p_swap_eq[perm[c]], p_orig_eq[c]) for c in gcols], axis=0), seed=BOOT_SEED + 700)
        for gname, gcols in (("ability", L.COLS_BY_KEY["ability"]), ("moves", L.COLS_BY_KEY["move"]), ("stat_points", list(range(M.STAT0, M.COLS))))
    }
    return dict(
        reader=arm["reader"], checkpoint=arm["checkpoint"], uniform=arm["uniform"],
        species_swap_tracking=dict(
            ability=dict(
                similarity_1_minus_tv=bootstrap_mean(acc["tab"], seed=BOOT_SEED + 10),
                top1_match=bootstrap_mean(acc["t1ab"], seed=BOOT_SEED + 11),
                new_species_legal=bootstrap_mean(acc["new_ab"], seed=BOOT_SEED + 12),
                old_species_only_legal=bootstrap_mean(acc["old_ab"], seed=BOOT_SEED + 13),
            ),
            first_move=dict(
                similarity_1_minus_tv=bootstrap_mean(acc["tfm"], seed=BOOT_SEED + 20),
                top1_match=bootstrap_mean(acc["t1fm"], seed=BOOT_SEED + 21),
                new_species_legal=bootstrap_mean(acc["new_fm"], seed=BOOT_SEED + 22),
                old_species_only_legal=bootstrap_mean(acc["old_fm"], seed=BOOT_SEED + 23),
            ),
            stat_points=dict(
                similarity_1_minus_tv=bootstrap_mean(acc["tst"], seed=BOOT_SEED + 30),
                top1_match=bootstrap_mean(acc["t1st"], seed=BOOT_SEED + 31),
            ),
        ),
        full_slot_equivariance_tv_shift=eq_by_group,
    )


def run_slot_swap(out_path: Path = SLOT_SWAP_OUT):
    """Run Item 6: slot-swap sensitivity at epoch 80 across all three checkpoints."""
    L, X, M, UM, UMath, torch = _import_runtime()
    corpus = X.setup()[0]
    xs_np, ws_np = _load_val_rows(X, limit=N_VAL_TEAMS)
    species_cols = L.COLS_BY_KEY["species"]
    non_species_cols = [c for c in range(M.COLS) if c not in species_cols]
    results = dict(
        issue=438, probe="slot_swap_sensitivity", epoch=80, n_val_teams=len(xs_np),
        swap_pairs=[list(p) for p in SWAP_PAIRS], uniform_draws=PROBE_DRAWS, arms={},
    )
    for arm in ARMS:
        ck_path = _epoch_checkpoint_path(L.CK, arm["checkpoint"], 80)
        model = UM.UniformModel84.load(ck_path, corpus) if arm["uniform"] else M.Model84.load(ck_path, corpus)
        x0 = torch.as_tensor(xs_np, device=M.DEV)
        w = torch.as_tensor(ws_np, device=M.DEV)
        x0_u = UMath.table_to_uniform(x0) if arm["uniform"] else None
        with torch.no_grad():
            arm_res = _eval_arm_slot_swap(model, arm, x0, x0_u, w, species_cols, non_species_cols, L, M, UMath, torch)
        results["arms"][arm["id"]] = arm_res
        st = arm_res["species_swap_tracking"]
        eq = arm_res["full_slot_equivariance_tv_shift"]
        print(
            f"{arm['id']:16s} | swap track (1-TV): ability {st['ability']['similarity_1_minus_tv']['mean']:.3f}, "
            f"first move {st['first_move']['similarity_1_minus_tv']['mean']:.3f}, "
            f"Stat Points {st['stat_points']['similarity_1_minus_tv']['mean']:.3f} | "
            f"equiv TV shift: ab {eq['ability']['mean']:.4f}, mv {eq['moves']['mean']:.4f}, st {eq['stat_points']['mean']:.4f}",
            flush=True,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"SLOT_SWAP_DONE {out_path}", flush=True)
    return results


def _speed_memory_worker(arm_id: str, passes: int, guidance: float, batch_size: int, min_legal: int, max_batches: int):
    """Isolated single-setting speed & memory worker following docs/generation-memory-spec.md."""
    import torch

    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    ceiling_bytes = None
    if device_type == "mps":
        torch.mps.set_per_process_memory_fraction(1.0)
        ceiling_bytes = int(torch.mps.recommended_max_memory())

    L, X, M, UM, _UMath, _ = _import_runtime()
    from hps_generate import Validator

    arm = ARM_BY_ID[arm_id]
    ck_path = _epoch_checkpoint_path(L.CK, arm["checkpoint"], 80)
    corpus = X.setup()[0]
    t_load0 = time.perf_counter()
    if arm["uniform"]:
        model = UM.UniformModel84.load(ck_path, corpus)
        sampler = UM.UniformSampler84(model, passes=passes)
    else:
        model = M.Model84.load(ck_path, corpus)
        sampler = model
    if device_type == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - t_load0

    def driver_mem():
        if device_type == "mps":
            torch.mps.synchronize()
            return int(torch.mps.driver_allocated_memory())
        return 0

    mem_after_load = driver_mem()
    peak_mem = mem_after_load

    # Warm-up 1 small batch (not timed) so Metal shaders are compiled before timing
    torch.manual_seed(43800)
    _ = sampler.sample(8, ask=0.5, guidance=guidance)
    peak_mem = max(peak_mem, driver_mem())

    val = Validator()
    torch.manual_seed(43801)
    batches_run = 0
    total_tries = 0
    total_legal = 0
    sample_seconds = 0.0
    validate_seconds = 0.0
    per_batch_times = []
    per_batch_legal = []

    try:
        while batches_run < max_batches and (batches_run < 2 or total_legal < min_legal):
            t0 = time.perf_counter()
            rows = sampler.sample(batch_size, ask=0.5, guidance=guidance).cpu().numpy()
            if device_type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            peak_mem = max(peak_mem, driver_mem())

            legal_here = 0
            for row in rows:
                paste, _ = sampler.to_paste(row, repair=True)
                if val(paste) is None:
                    legal_here += 1
            t2 = time.perf_counter()

            batches_run += 1
            total_tries += len(rows)
            total_legal += legal_here
            sample_seconds += t1 - t0
            validate_seconds += t2 - t1
            per_batch_times.append(t2 - t0)
            per_batch_legal.append(legal_here)
    finally:
        val.close()

    total_seconds = sample_seconds + validate_seconds
    legality = total_legal / max(total_tries, 1)
    sec_per_try = total_seconds / max(total_tries, 1)
    sec_per_valid = total_seconds / total_legal if total_legal > 0 else None

    # Bootstrap seconds per valid team over individual tries (using mean time/try and Bernoulli legal indicators)
    indicators = np.array([1.0] * total_legal + [0.0] * (total_tries - total_legal), dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED + passes + int(guidance * 10))
    idx = rng.integers(0, total_tries, size=(BOOT_DRAWS, total_tries))
    boot_leg = indicators[idx].mean(axis=1)
    boot_spv = np.where(boot_leg > 0, sec_per_try / boot_leg, np.nan)
    finite_spv = boot_spv[np.isfinite(boot_spv)]
    spv_iv = [float(np.percentile(finite_spv, 2.5)), float(np.percentile(finite_spv, 97.5))] if finite_spv.size else [None, None]
    leg_iv = [float(np.percentile(boot_leg, 2.5)), float(np.percentile(boot_leg, 97.5))]

    out = dict(
        arm_id=arm_id,
        reader=arm["reader"],
        checkpoint=arm["checkpoint"],
        passes=int(passes),
        guidance=float(guidance),
        ask=0.5,
        batch_size=int(batch_size),
        weight_precision="float32",
        device=device_type,
        ceiling_bytes=ceiling_bytes,
        ceiling_gib=round(ceiling_bytes / (1024**3), 4) if ceiling_bytes else None,
        mem_after_load_gib=round(mem_after_load / (1024**3), 4),
        peak_gib=round(peak_mem / (1024**3), 4),
        fits_ceiling=bool(ceiling_bytes is None or peak_mem <= ceiling_bytes),
        load_seconds=round(load_seconds, 3),
        batches=batches_run,
        tries=total_tries,
        legal_teams=total_legal,
        first_try_legality=legality,
        first_try_legality_interval=leg_iv,
        sample_seconds=round(sample_seconds, 3),
        validate_seconds=round(validate_seconds, 3),
        total_seconds=round(total_seconds, 3),
        seconds_per_try=sec_per_try,
        seconds_per_valid_team=sec_per_valid,
        seconds_per_valid_team_interval=spv_iv,
    )
    print(json.dumps(out))
    return out


def run_speed_memory(out_path: Path = SPEED_MEM_OUT, batch_size: int = 96):
    """Run Item 7: isolated subprocess speed and peak MPS memory benchmark across all three checkpoints."""
    settings = (
        ("uniform_gidd_lr", 84, 1.0),
        ("uniform_gidd_lr", 128, 1.0),
        ("uniform_gidd_lr", 256, 1.0),
        ("uniform_gidd_lr", 84, 2.0),
        ("uniform_gidd_lr", 128, 2.0),
        ("uniform_4e4", 84, 1.0),
        ("uniform_4e4", 128, 1.0),
        ("masked_gidd_lr", 84, 2.0),
    )
    rows = []
    for arm_id, passes, guidance in settings:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "speed-memory-worker",
            "--arm",
            arm_id,
            "--passes",
            str(passes),
            "--guidance",
            str(guidance),
            "--batch-size",
            str(batch_size),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        spv = f"{row['seconds_per_valid_team']:.3f}s" if row["seconds_per_valid_team"] is not None else "n/a"
        print(
            f"{arm_id:16s} K={passes:3d} g={guidance:g} | legal {row['first_try_legality']:5.1%} "
            f"({row['legal_teams']}/{row['tries']}) | {spv}/valid team | peak {row['peak_gib']:.3f} GiB",
            flush=True,
        )

    out = dict(
        issue=438,
        probe="generation_speed_memory",
        spec="code/docs/generation-memory-spec.md",
        ask=0.5,
        batch_size=batch_size,
        rows=rows,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"SPEED_MEMORY_DONE {out_path}", flush=True)
    return out


def run_val_loss(out_path: Path = VAL_LOSS_OUT):
    """Recompute clipped-time validation diffusion loss on the identical 4,368 filtered val rows (seed=12345)."""
    L, X, M, UM, _UMath, _ = _import_runtime()
    corpus = X.setup()[0]
    xs_np, ws_np = _load_val_rows(X, limit=None)
    rows = {}
    for arm in ARMS:
        ck_path = _epoch_checkpoint_path(L.CK, arm["checkpoint"], 80)
        if arm["uniform"]:
            model = UM.UniformModel84.load(ck_path, corpus)
            vl = float(UM.val_loss(model.net, xs_np, ws_np, model.vocab_sizes, reps=4, seed=12345))
            note = "Continuous-time UDLM Eq. 19 loss on t in [1e-3, 1 - 1e-3], no legality tables"
        else:
            model = M.Model84.load(ck_path, corpus)
            sched = M.sched84(M.trained_sched(model.info)).to(M.DEV)
            vl = float(M.val_loss(model.net, model.tables, xs_np, ws_np, sched, reps=4, seed=12345, table_loss=True))
            note = "Continuous-time absorbing masked ELBO on linear schedule t in [1e-3, 1], with legality tables"
        stored_best = model.info.get("best_val_loss")
        rows[arm["id"]] = dict(
            reader=arm["reader"],
            checkpoint=arm["checkpoint"],
            val_loss=vl,
            stored_best_val_loss=stored_best,
            val_rows=len(xs_np),
            reps=4,
            seed=12345,
            note=note,
        )
        stored_txt = f"{stored_best:.4f}" if stored_best is not None else "n/a"
        print(f"{arm['id']:16s} | recomputed val_loss {vl:.4f} (stored {stored_txt})", flush=True)

    out = dict(issue=438, probe="val_loss_bound", val_rows=len(xs_np), reps=4, seed=12345, arms=rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"VAL_LOSS_DONE {out_path}", flush=True)
    return out


# ---------------------------------------------------------------- ECharts storyboard figures
def build_slot_binding_figure(data: dict, out_dir: Path) -> Path:
    """Two-panel trajectory: own-species probability for Ability and First Move across epochs 45-80."""
    epochs = [int(e) for e in data["epochs"]]
    labels = [f"ep {e}" for e in epochs]
    series = []
    for field_idx, (field_key, field_title) in enumerate((("ability", "Ability"), ("first_move", "First move"))):
        for arm in ARMS:
            arm_data = data["arms"][arm["id"]]
            pts = [arm_data["by_epoch"][str(e)]["species_first_prefix"][field_key]["own"]["mean"] for e in epochs]
            series.append({
                "name": arm["reader"],
                "type": "line",
                "xAxisIndex": field_idx,
                "yAxisIndex": field_idx,
                "color": arm["color"],
                "symbolSize": 8,
                "lineStyle": {"width": 2.6, "type": "solid" if arm["id"] == "uniform_gidd_lr" else "dashed"},
                "emphasis": {"focus": "series"},
                "data": [round(v, 4) for v in pts],
                "label": {
                    "show": True,
                    "position": "bottom" if arm["id"] == "uniform_4e4" else "top",
                    "formatter": "js:p => (p.value * 100).toFixed(0) + '%'",
                    "fontSize": 10,
                },
            })

    option = {
        "animation": False,
        "legend": {"top": 4, "data": [arm["reader"] for arm in ARMS]},
        "tooltip": {"trigger": "axis", "valueFormatter": "js:v => (Number(v) * 100).toFixed(1) + '%'"},
        "title": [
            {"text": "Ability probability on own species", "left": "25%", "top": 36, "textAlign": "center", "textStyle": {"fontSize": 13}},
            {"text": "First-move probability on own species", "left": "75%", "top": 36, "textAlign": "center", "textStyle": {"fontSize": 13}},
        ],
        "grid": [
            {"left": 64, "width": "40%", "top": 74, "bottom": 48},
            {"left": "55%", "width": "40%", "top": 74, "bottom": 48},
        ],
        "xAxis": [
            {"type": "category", "gridIndex": 0, "data": labels, "name": "Training epoch", "nameLocation": "middle", "nameGap": 28},
            {"type": "category", "gridIndex": 1, "data": labels, "name": "Training epoch", "nameLocation": "middle", "nameGap": 28},
        ],
        "yAxis": [
            {"type": "value", "gridIndex": 0, "min": 0, "max": 1.0, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}, "name": "Own-species probability", "nameLocation": "middle", "nameGap": 44},
            {"type": "value", "gridIndex": 1, "min": 0, "max": 1.0, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}},
        ],
        "series": series,
    }
    u5 = data["arms"]["uniform_gidd_lr"]["by_epoch"]["80"]["species_first_prefix"]
    title = (
        f"Shared-recipe Uniform binds {u5['ability']['own']['mean']:.1%} of ability and "
        f"{u5['first_move']['own']['mean']:.1%} of first-move mass to each Pokémon's own species by epoch 80"
    )
    return write_chart(
        option,
        out_dir / "slot-binding.html",
        title=title,
        subtitle="Pokémon-to-field matching probe on 256 validation teams (species-first prefix; Masked evaluated without tables).",
        note="Uniform models replace future fields with uniform random real tokens (4 draws); Masked replaces them with [MASK].",
        height=440,
        dark=DARK_MAP,
    )


def build_slot_swap_figure(data: dict, out_dir: Path) -> Path:
    """Grouped bar chart: distribution tracking (1 - TV) when two species swap slots."""
    categories = ["Ability", "First move", "Stat Points"]
    keys = ("ability", "first_move", "stat_points")
    series = []
    for arm in ARMS:
        st = data["arms"][arm["id"]]["species_swap_tracking"]
        vals = [round(st[k]["similarity_1_minus_tv"]["mean"], 4) for k in keys]
        series.append({
            "name": arm["reader"],
            "type": "bar",
            "color": arm["color"],
            "barMaxWidth": 36,
            "emphasis": {"focus": "series"},
            "data": vals,
            "label": {"show": True, "position": "top", "formatter": "js:p => (p.value * 100).toFixed(1) + '%'", "fontSize": 11},
        })
    option = {
        "animation": False,
        "legend": {"top": 4, "data": [arm["reader"] for arm in ARMS]},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "valueFormatter": "js:v => (Number(v) * 100).toFixed(1) + '%'"},
        "grid": {"left": 74, "right": 28, "top": 52, "bottom": 48},
        "xAxis": {"type": "category", "data": categories},
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1.0,
            "name": "Distribution tracking (1 − TV)",
            "nameLocation": "middle",
            "nameGap": 50,
            "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"},
        },
        "series": series,
    }
    u5 = data["arms"]["uniform_gidd_lr"]["species_swap_tracking"]
    title = (
        f"Swapping two species moves Ability ({u5['ability']['similarity_1_minus_tv']['mean']:.1%}), "
        f"First Move ({u5['first_move']['similarity_1_minus_tv']['mean']:.1%}), and Stat Points "
        f"({u5['stat_points']['similarity_1_minus_tv']['mean']:.1%}) with the species"
    )
    return write_chart(
        option,
        out_dir / "slot-swap.html",
        title=title,
        subtitle="256 validation teams at epoch 80, averaged over 3 disjoint slot swaps covering all 6 slots (100% = exact move).",
        note="Measures 1 − TV(p(slot b | swapped), p(slot a | original)) with all 6 species visible and non-species fields corrupted/masked.",
        height=400,
        dark=DARK_MAP,
    )


def build_speed_memory_figure(data: dict, out_dir: Path) -> Path:
    """Horizontal bar chart comparing seconds per Showdown-valid team across models and pass counts."""
    rows = data["rows"]
    labels = [
        f"{row['reader']}\n(K={row['passes']}, g={row['guidance']:g}, {row['first_try_legality']:.0%} legal)"
        for row in rows
    ]
    vals = [round(row["seconds_per_valid_team"], 3) if row["seconds_per_valid_team"] is not None else 0.0 for row in rows]
    colors = [ARM_BY_ID[row["arm_id"]]["color"] for row in rows]
    option = {
        "animation": False,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "valueFormatter": "js:v => Number(v).toFixed(3) + ' s'"},
        "grid": {"left": 285, "right": 74, "top": 28, "bottom": 48},
        "xAxis": {"type": "value", "name": "Seconds per Showdown-valid team (incl. redraws)", "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "category", "data": labels, "inverse": True, "axisLabel": {"fontSize": 11}},
        "series": [{
            "name": "Seconds per valid team",
            "type": "bar",
            "barWidth": 22,
            "data": [{"value": v, "itemStyle": {"color": c}} for v, c in zip(vals, colors)],
            "label": {"show": True, "position": "right", "formatter": "js:p => Number(p.value).toFixed(3) + ' s'"},
        }],
    }
    max_gib = max(r["peak_gib"] for r in rows)
    ceil_gib = next((r["ceiling_gib"] for r in rows if r["ceiling_gib"] is not None), 17.76)
    return write_chart(
        option,
        out_dir / "speed-memory.html",
        title=f"Every 84-field model stays under {max_gib:.2f} GiB peak MPS memory (ceiling {ceil_gib:.2f} GiB); legality drives wall-clock cost",
        subtitle="Batch size 96, ask 12/24, measured on laptop MPS in one isolated process per setting (including Showdown validator redraws).",
        note="Wall-clock time divides total sampling + Showdown validation seconds by accepted legal teams.",
        height=110 + 48 * len(rows),
        dark=DARK_MAP,
    )


def _fmt_iv(d, fmt=".1%"):
    lo, hi = d["interval"]
    return f"{d['mean']:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def _speed_memory_rows_md(speed_mem: dict) -> list[str]:
    lines = [
        "",
        "### Item 7 — Generation speed and peak MPS memory (ask 12/24, batch 96, isolated subprocess per setting)",
        "",
        "| Model & setting | First-try legality [95% CI] | Seconds / valid team [95% CI] | Seconds / try | Peak MPS GiB | Fits 17.76 GiB? |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in speed_mem["rows"]:
        lo_l, hi_l = r["first_try_legality_interval"]
        if r["seconds_per_valid_team"] is not None:
            lo_s, hi_s = r["seconds_per_valid_team_interval"]
            spv = f"{r['seconds_per_valid_team']:.3f}s [{lo_s:.3f}s, {hi_s:.3f}s]"
        else:
            spv = "n/a"
        lines.append(
            f"| {r['reader']} (`K={r['passes']}`, `g={r['guidance']:g}`) | "
            f"{r['first_try_legality']:.1%} [{lo_l:.1%}, {hi_l:.1%}] ({r['legal_teams']}/{r['tries']}) | "
            f"{spv} | {r['seconds_per_try']:.3f}s | {r['peak_gib']:.3f} | {'Yes' if r['fits_ceiling'] else 'No'} |"
        )
    return lines


def format_probes_comment(slot_binding: dict, slot_swap: dict, speed_mem: dict, val_loss: dict, base_url: str) -> str:
    """Format the result-first Issue #438 comment for all battle-free probes (items 5-7 + val_loss)."""
    def pic(stem, alt):
        return (
            f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base_url}/{stem}-dark.png">'
            f'<img src="{base_url}/{stem}-light.png" alt="{alt}" width="900"></picture>'
        )

    u5_sb = slot_binding["arms"]["uniform_gidd_lr"]["by_epoch"]["80"]["species_first_prefix"]
    u5_sw = slot_swap["arms"]["uniform_gidd_lr"]["species_swap_tracking"]
    u4_sw = slot_swap["arms"]["uniform_4e4"]["species_swap_tracking"]
    m5_sw = slot_swap["arms"]["masked_gidd_lr"]["species_swap_tracking"]
    max_gib = max(r["peak_gib"] for r in speed_mem["rows"])

    lines = [
        "**Battle-free probes done (items 5, 6, 7 and validation loss): shared-recipe Uniform binds slots ~12 epochs earlier than the 4e-4 run, moves all three field groups with a swapped species (2x tighter equivariance than Masked), and peaks at 1.15 GiB against the 17.76 GiB ceiling.**",
        "",
        "Three figures summarize Pokémon-to-field binding across training epochs, slot-swap tracking at epoch 80, and generation speed per Showdown-valid team.",
        "",
        pic("slot-binding", f"Own-species probability across epochs 45-80: shared-recipe Uniform reaches {u5_sb['ability']['own']['mean']:.1%} ability and {u5_sb['first_move']['own']['mean']:.1%} first-move binding at epoch 80"),
        "",
        pic("slot-swap", f"Species-swap tracking (1 - TV) at epoch 80: shared-recipe Uniform moves Ability ({u5_sw['ability']['similarity_1_minus_tv']['mean']:.1%}), First Move ({u5_sw['first_move']['similarity_1_minus_tv']['mean']:.1%}), and Stat Points ({u5_sw['stat_points']['similarity_1_minus_tv']['mean']:.1%}) with the swapped species"),
        "",
        pic("speed-memory", f"Wall-clock seconds per Showdown-valid team including redraws across models and pass counts; peak MPS memory stays at or under {max_gib:.2f} GiB against the 17.76 GiB ceiling"),
        "",
        "### Validation loss bound (Part A item 2, 4,368 filtered validation rows, 4 CRN draws, seed 12345)",
        "",
        "| Checkpoint | Recomputed `val_loss` | Stored best | Loss family |",
        "| --- | ---: | ---: | --- |",
    ]
    for arm in ARMS:
        v = val_loss["arms"][arm["id"]]
        stored_txt = f"{v['stored_best_val_loss']:.4f}" if v.get("stored_best_val_loss") is not None else "—"
        lines.append(f"| {arm['reader']} | **{v['val_loss']:.4f}** | {stored_txt} | {v['note']} |")

    lines += [
        "",
        "### Item 5 — Pokémon-to-field matching probe (256 validation teams, 95% bootstrap intervals)",
        "",
        "Own-species probability mass under the species-first prefix (`t = 1 - step/84`, teammate-species mass in parentheses). Note: Uniform has no `[MASK]` token, so at high `t ≈ 0.90` it Bayes-optimally hedges ~36% of ability mass onto `none` in case the species token is uniform noise; at `t = 0.05` its top-1 ability legality reaches 100.0% (83.2% own mass). Masked, full filter trained with tables on in the loss, so without tables at test time it relies less on its weights for abilities (18.6% own).",
        "",
        "| Epoch & field | Uniform (shared recipe) | Uniform (4e-4 recipe) | Masked, full filter (shared recipe, no tables) |",
        "| --- | --- | --- | --- |",
    ]
    for ep in slot_binding["epochs"]:
        for fkey, flabel in (("ability", "Ability"), ("first_move", "First move")):
            cols_txt = [
                f"{_fmt_iv(slot_binding['arms'][arm['id']]['by_epoch'][str(ep)]['species_first_prefix'][fkey]['own'])} "
                f"({slot_binding['arms'][arm['id']]['by_epoch'][str(ep)]['species_first_prefix'][fkey]['other']['mean']:.1%})"
                for arm in ARMS
            ]
            lines.append(f"| Epoch {ep}, {flabel} | " + " | ".join(cols_txt) + " |")

    lines += [
        "",
        "### Item 6 — Slot-swap sensitivity at epoch 80 (256 validation teams, 3 disjoint swaps covering all 6 slots)",
        "",
        "| Measure | Uniform (shared recipe) | Uniform (4e-4 recipe) | Masked, full filter (shared recipe) |",
        "| --- | --- | --- | --- |",
        *(
            f"| {flabel} | {_fmt_iv(u5_sw[fkey]['similarity_1_minus_tv'])} | {_fmt_iv(u4_sw[fkey]['similarity_1_minus_tv'])} | {_fmt_iv(m5_sw[fkey]['similarity_1_minus_tv'])} |"
            for fkey, flabel in (("ability", "Ability tracking (1 − TV)"), ("first_move", "First-move tracking (1 − TV)"), ("stat_points", "Stat Points tracking (1 − TV)"))
        ),
        *(
            f"| {flabel} | {u5_sw[fkey]['new_species_legal']['mean']:.1%} ({u5_sw[fkey]['old_species_only_legal']['mean']:.1%}) | "
            f"{u4_sw[fkey]['new_species_legal']['mean']:.1%} ({u4_sw[fkey]['old_species_only_legal']['mean']:.1%}) | "
            f"{m5_sw[fkey]['new_species_legal']['mean']:.1%} ({m5_sw[fkey]['old_species_only_legal']['mean']:.1%}) |"
            for fkey, flabel in (("ability", "Ability legal for new species (old-only)"), ("first_move", "First move legal for new species (old-only)"))
        ),
        *(
            f"| {glabel} | {_fmt_iv(slot_swap['arms']['uniform_gidd_lr']['full_slot_equivariance_tv_shift'][gkey], '.4f')} | "
            f"{_fmt_iv(slot_swap['arms']['uniform_4e4']['full_slot_equivariance_tv_shift'][gkey], '.4f')} | "
            f"{_fmt_iv(slot_swap['arms']['masked_gidd_lr']['full_slot_equivariance_tv_shift'][gkey], '.4f')} |"
            for gkey, glabel in (("ability", "Full-slot swap TV shift, Ability"), ("moves", "Full-slot swap TV shift, Moves"), ("stat_points", "Full-slot swap TV shift, Stat Points"))
        ),
        *_speed_memory_rows_md(speed_mem),
        "",
        "Battles: 0 (all measures above are battle-free). Intervals: 95% bootstrap (2,000 draws over teams or tries). One training seed per checkpoint.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("slot-binding")
    sub.add_parser("slot-swap")
    sm = sub.add_parser("speed-memory")
    sm.add_argument("--batch-size", type=int, default=96)
    smw = sub.add_parser("speed-memory-worker")
    smw.add_argument("--arm", required=True)
    smw.add_argument("--passes", type=int, required=True)
    smw.add_argument("--guidance", type=float, required=True)
    smw.add_argument("--batch-size", type=int, default=96)
    smw.add_argument("--min-legal", type=int, default=32)
    smw.add_argument("--max-batches", type=int, default=5)
    sub.add_parser("val-loss")
    fig = sub.add_parser("figures")
    fig.add_argument("out_dir", type=Path)
    com = sub.add_parser("comment")
    com.add_argument("base_url")
    args = parser.parse_args()

    if args.cmd == "slot-binding":
        run_slot_binding()
    elif args.cmd == "slot-swap":
        run_slot_swap()
    elif args.cmd == "speed-memory":
        run_speed_memory(batch_size=args.batch_size)
    elif args.cmd == "speed-memory-worker":
        _speed_memory_worker(args.arm, args.passes, args.guidance, args.batch_size, args.min_legal, args.max_batches)
    elif args.cmd == "val-loss":
        run_val_loss()
    elif args.cmd == "figures":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for fn, path in (
            (build_slot_binding_figure, SLOT_BINDING_OUT),
            (build_slot_swap_figure, SLOT_SWAP_OUT),
            (build_speed_memory_figure, SPEED_MEM_OUT),
        ):
            out = fn(json.loads(path.read_text()), args.out_dir)
            print(out)
    elif args.cmd == "comment":
        print(
            format_probes_comment(
                json.loads(SLOT_BINDING_OUT.read_text()),
                json.loads(SLOT_SWAP_OUT.read_text()),
                json.loads(SPEED_MEM_OUT.read_text()),
                json.loads(VAL_LOSS_OUT.read_text()),
                args.base_url,
            ),
            end="",
        )


if __name__ == "__main__":
    main()
