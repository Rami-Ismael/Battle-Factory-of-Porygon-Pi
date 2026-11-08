"""One place to load PyTorch checkpoints.

torch.load is banned elsewhere by the Ruff rule TID251 (see pyproject.toml at
the repository root). weights_only=True refuses arbitrary pickled objects; it is
already torch's default on 2.12 and 2.14, so this makes it explicit.
Pass trusted_objects=True only for a file you wrote that stores more than
tensors and plain Python containers (numpy arrays, custom classes).

Run info saved before 2026-11-08 called an epoch a pass (best_pass, stopped_pass,
pass_ in each learning-curve entry). load_checkpoint gives such info the epoch
names; call epoch_names on run info read from a results JSON file.
"""
import torch

OLD_TO_EPOCH = {"best_pass": "best_epoch", "stopped_pass": "stopped_epoch"}


def epoch_names(info):
    """A copy of run info with the pre-2026-11-08 pass names renamed to epoch names, in the learning curve and in
    nested base and member info. Info that already uses epoch names comes back equal."""
    if not isinstance(info, dict): return info
    out = {OLD_TO_EPOCH.get(k, k) if OLD_TO_EPOCH.get(k) not in info else k: v for k, v in info.items()}
    if isinstance(out.get("curves"), list):
        out["curves"] = [{("epoch" if k == "pass_" and "epoch" not in e else k): v for k, v in e.items()}
                         if isinstance(e, dict) else e for e in out["curves"]]
    if isinstance(out.get("base"), dict): out["base"] = epoch_names(out["base"])
    if isinstance(out.get("members"), list): out["members"] = [epoch_names(m) for m in out["members"]]
    return out


def load_checkpoint(path, *, trusted_objects=False, **kwargs):
    ck = torch.load(path, weights_only=not trusted_objects, **kwargs)  # noqa: TID251
    if isinstance(ck, dict) and isinstance(ck.get("info"), dict): ck["info"] = epoch_names(ck["info"])
    return ck
