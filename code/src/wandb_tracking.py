"""Weights & Biases experiment tracking for 84-field training runs (train and fine-tune).

Owner decisions (2026-11-08):
- Scope: training runs only (Model84.train_new and Model84.fine_tuned). Sampling grids, battles and search loops stay in
  local results files and issue comments.
- Record: the learning curve in the local results JSON (info["curves"]) remains the authoritative record. Weights & Biases
  mirrors it, so an offline run or network outage loses nothing.
- Visibility: public project, matching the Hugging Face Hub fine-tunes. Never log private local paths into config or summary.
- Vocabulary: standard ML terms, with `epoch` as the step axis for every logged metric.

    with WandbTracker(name=run_name, config=cfg) as tracker:
        model = Model84.train_new(..., tracker=tracker)
        tracker.log_summary(model.info, stop_reason=..., checkpoint_name=out.name)
"""
import math
import os
import sys
from pathlib import Path

DEFAULT_PROJECT = "battle-factory-of-porygon-pi"
VALID_MODES = ("online", "offline", "disabled")
_SKIP_CONFIG_KEYS = frozenset({"curves", "base", "members", "wandb"})
_SUMMARY_KEYS = ("best_epoch", "stopped_epoch", "best_val_loss", "minutes", "n_train", "dropped_by_tables", "val_rows")
_FINAL_CURVE_KEYS = ("train_loss", "val_loss", "illegal_share", "errors_per_team", "copied_share")


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _to_number(value):
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else float(value)


def resolve_mode(mode=None, *, disabled=False, smoke=False, env=None):
    """Resolve the Weights & Biases mode: CLI override > WANDB_MODE env var > smoke default > online."""
    if disabled:
        return "disabled"
    env = os.environ if env is None else env
    chosen = mode if mode is not None else (env.get("WANDB_MODE") or "").strip() or None
    if chosen is None:
        return "disabled" if smoke else "online"
    if chosen not in VALID_MODES:
        raise ValueError(f"wandb mode must be one of {VALID_MODES}, got {chosen!r}")
    return chosen


def resolve_project(project=None, env=None):
    """Resolve the Weights & Biases project name."""
    env = os.environ if env is None else env
    return project or (env.get("WANDB_PROJECT") or "").strip() or DEFAULT_PROJECT


def resolve_entity(entity=None, env=None):
    """Resolve the optional Weights & Biases entity/team name."""
    env = os.environ if env is None else env
    return entity or (env.get("WANDB_ENTITY") or "").strip() or None


def default_wandb_dir(env=None):
    """Store local W&B files under ~/vgc-data/wandb when that data folder exists, unless WANDB_DIR is set."""
    env = os.environ if env is None else env
    if (env.get("WANDB_DIR") or "").strip():
        return env["WANDB_DIR"]
    vgc_data = Path.home() / "vgc-data"
    if vgc_data.is_dir():
        out = vgc_data / "wandb"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)
    return None


def _clean_value(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        return _clean_value(value.item())
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return Path(value).name if value.startswith(("/Users/", "/home/")) else value
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items() if k not in _SKIP_CONFIG_KEYS}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    return str(value)


def clean_config(config):
    """Sanitize a run's hyperparameter/config mapping for a public W&B project."""
    if not config:
        return {}
    return {str(k): _clean_value(v) for k, v in config.items() if k not in _SKIP_CONFIG_KEYS}


def epoch_metrics(entry, *, lr=None, step=None, grad_norm=None, best_val_loss=None):
    """Convert one learning-curve entry from Model84._fit into flat finite scalar metrics for wandb.log."""
    metrics = {}
    for key, value in entry.items():
        if key == "rule_shares" and isinstance(value, dict):
            for rule, share in value.items():
                if _is_finite_number(share):
                    metrics[f"rule_shares/{rule}"] = float(share)
        elif _is_finite_number(value):
            metrics[key] = _to_number(value)
    if "illegal_share" in metrics:
        metrics["legal_share"] = 1.0 - float(metrics["illegal_share"])
    extras = dict(learning_rate=lr, optimizer_step=step, grad_norm=grad_norm, best_val_loss=best_val_loss)
    for key, value in extras.items():
        if _is_finite_number(value):
            metrics[key] = _to_number(value)
    return metrics


def summary_metrics(info, *, stop_reason=None, checkpoint_name=None):
    """Extract final run summary fields from Model84 run info."""
    out = {}
    for key in _SUMMARY_KEYS:
        value = info.get(key)
        if _is_finite_number(value):
            out[key] = _to_number(value)
    curves = info.get("curves") or []
    if curves and isinstance(curves[-1], dict):
        last = curves[-1]
        for key in _FINAL_CURVE_KEYS:
            if _is_finite_number(last.get(key)):
                out[f"final_{key}"] = float(last[key])
    if stop_reason:
        out["stop_reason"] = str(stop_reason)
    if checkpoint_name:
        out["checkpoint"] = Path(checkpoint_name).name
    return out


def add_cli_args(parser):
    """Add standard Weights & Biases flags to a training script's ArgumentParser."""
    group = parser.add_argument_group("Weights & Biases tracking")
    group.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases tracking for this run")
    group.add_argument("--wandb-mode", choices=VALID_MODES, default=None,
                       help="W&B mode (online, offline, disabled); defaults to WANDB_MODE, disabled in smoke runs, else online")
    group.add_argument("--wandb-project", default=None, help=f"W&B project name (default: WANDB_PROJECT or {DEFAULT_PROJECT})")
    group.add_argument("--wandb-entity", default=None, help="W&B entity/team name (default: WANDB_ENTITY)")
    group.add_argument("--wandb-group", default=None, help="W&B run group for comparing arms side by side")
    group.add_argument("--wandb-name", default=None, help="override the W&B run name")
    group.add_argument("--wandb-tags", default=None, help="comma-separated extra W&B tags")
    return parser


def _parse_tags(base_tags, extra_tags):
    tags = [str(t).strip() for t in (base_tags or ()) if str(t).strip()]
    if extra_tags:
        tags.extend(t.strip() for t in str(extra_tags).split(",") if t.strip())
    return list(dict.fromkeys(tags))


def from_cli_args(args, *, name, job_type="train", tags=None, config=None, smoke=False, notes=None, wandb_module=None):
    """Build a WandbTracker from parsed CLI args, or a disabled tracker when args is None."""
    if args is None:
        return WandbTracker(disabled=True)
    return WandbTracker(
        name=args.wandb_name or name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=job_type,
        tags=_parse_tags(tags, args.wandb_tags),
        config=config,
        mode=args.wandb_mode,
        disabled=args.no_wandb,
        smoke=smoke,
        notes=notes,
        wandb_module=wandb_module,
    )


class WandbTracker:
    """Mirrors one training run's config, per-epoch learning curve and summary to Weights & Biases.

    Training never fails if Weights & Biases is absent, unauthenticated or offline: online init falls back to offline
    mode, and if that also fails the tracker disables itself with a warning while local JSON logging continues.
    """

    def __init__(self, *, name=None, project=None, entity=None, group=None, job_type="train", tags=None, config=None,
                 mode=None, disabled=False, smoke=False, dir=None, notes=None, wandb_module=None, env=None):
        self.name = name
        self.project = resolve_project(project, env=env)
        self.entity = resolve_entity(entity, env=env)
        self.group = group
        self.job_type = job_type
        self.tags = list(tags) if tags else []
        self._config = clean_config(config)
        self.mode = resolve_mode(mode, disabled=disabled, smoke=smoke, env=env)
        self.dir = default_wandb_dir(env=env) if dir is None and self.mode != "disabled" else dir
        self.notes = notes
        self._wandb = wandb_module
        self.run = None
        self.active = False
        self._warned_log = False

    def __enter__(self):
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb):
        self.finish()
        return False

    def _load_wandb(self):
        if self._wandb is not None:
            return self._wandb
        try:
            import wandb
        except ImportError as exc:
            print(f"[wandb] package not installed ({exc}); continuing without W&B tracking", file=sys.stderr, flush=True)
            self.mode = "disabled"
            return None
        self._wandb = wandb
        return wandb

    def _init_kwargs(self, mode):
        kw = dict(project=self.project, name=self.name, job_type=self.job_type, config=self._config, mode=mode)
        if self.entity:
            kw["entity"] = self.entity
        if self.group:
            kw["group"] = self.group
        if self.tags:
            kw["tags"] = self.tags
        if self.dir:
            kw["dir"] = self.dir
        if self.notes:
            kw["notes"] = self.notes
        return kw

    def _open_run(self, wandb):
        try:
            return wandb.init(**self._init_kwargs(self.mode))
        except Exception as exc:
            if self.mode != "online":
                print(f"[wandb] {self.mode} init failed ({exc}); disabling W&B for this run", file=sys.stderr, flush=True)
                self.mode = "disabled"
                return None
            print(f"[wandb] online init failed ({exc}); falling back to offline mode (local results JSON stays the record)",
                  file=sys.stderr, flush=True)
            self.mode = "offline"
            try:
                return wandb.init(**self._init_kwargs("offline"))
            except Exception as offline_exc:
                print(f"[wandb] offline init failed ({offline_exc}); disabling W&B for this run", file=sys.stderr, flush=True)
                self.mode = "disabled"
                return None

    def start(self, config=None):
        """Start the W&B run, or update its config if already active."""
        merged = clean_config(config)
        if merged:
            self._config.update(merged)
        if self.mode == "disabled":
            return self
        if self.active and self.run is not None:
            if merged:
                self._safe(lambda: self.run.config.update(merged, allow_val_change=True), "config update")
            return self
        wandb = self._load_wandb()
        if wandb is None:
            return self
        run = self._open_run(wandb)
        if run is None:
            return self
        self.run = run
        self.active = True
        self._safe(self._define_epoch_axis, "define_metric")
        return self

    def _define_epoch_axis(self):
        self.run.define_metric("epoch")
        self.run.define_metric("*", step_metric="epoch")

    def _safe(self, fn, label):
        try:
            fn()
        except Exception as exc:
            if not self._warned_log:
                print(f"[wandb] {label} failed ({exc}); local results JSON stays the record", file=sys.stderr, flush=True)
                self._warned_log = True

    def log_epoch(self, entry, *, lr=None, step=None, grad_norm=None, best_val_loss=None):
        """Log one epoch's learning-curve metrics to Weights & Biases."""
        if not self.active or self.run is None:
            return
        metrics = epoch_metrics(entry, lr=lr, step=step, grad_norm=grad_norm, best_val_loss=best_val_loss)
        if metrics:
            self._safe(lambda: self.run.log(metrics), "log")

    def log_summary(self, info, *, stop_reason=None, checkpoint_name=None):
        """Write final training summary scalars to Weights & Biases."""
        if not self.active or self.run is None:
            return
        summary = summary_metrics(info, stop_reason=stop_reason, checkpoint_name=checkpoint_name)
        if summary:
            self._safe(lambda: self.run.summary.update(summary), "summary update")

    def metadata(self):
        """Public metadata for the local results JSON to point back to this W&B run, or None when disabled."""
        if self.run is None or self.mode == "disabled":
            return None
        raw = dict(
            project=getattr(self.run, "project", None) or self.project,
            entity=getattr(self.run, "entity", None) or self.entity,
            run_id=getattr(self.run, "id", None),
            run_name=getattr(self.run, "name", None) or self.name,
            mode=self.mode,
            url=getattr(self.run, "url", None),
        )
        return {k: v for k, v in raw.items() if v is not None}

    def finish(self):
        """Finish the active W&B run if one is open."""
        if not self.active or self.run is None:
            return
        run = self.run
        self.active = False
        self._safe(run.finish, "finish")
