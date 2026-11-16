"""First-try legality during training (issue #248): after an epoch, the model as it stands draws a small batch of teams
at one ask, Showdown's validator checks each, and the epoch's log entry gains the illegal share, the curve the check
exists for (owner, 2026-11-11: it should fall over training). The first-try legal share is 1 minus it.

The draw is the ask grid's (ask_grid84.draw): guidance 2, temperature 1, species-first order, Stat Alignment repaired
to match the moves before the validator. The ask is the elite cut, 0.531, and every epoch draws from the same seed,
so the curve moves only when the model does. The draw runs in a forked random state, so training's own batches,
masks and times are the same with the check on or off.

    check = EpochLegality(teams=64)        # teams=0: no check, no fields
    entry.update(check(model))             # illegal_share, legality_teams, rejection_reasons, legality_seconds
    check.close()
"""
import re
import time
from collections import Counter

import torch

from hps_generate import Validator

TEAMS = 64                                  # standard error about ±0.06 on the share (owner, 2026-11-08)
ASK = 0.531                                 # the elite cut
GUIDANCE = 2.0                              # the ask grid's default guidance
TEMPERATURE = 1.0
SEED = 12345                                # every epoch draws the same random numbers, like the validation loss
TOP_REASONS = 5


def reason(err):
    """A rejection's first line with its Pokémon, move and item names removed, so counts group by kind of error."""
    return re.sub(r"\s+", " ", re.sub(r"[^:]*?(can't|cannot|has|is|must|needs)\b", r"\1", str(err).split("\n")[0])).strip()[:80]


class EpochLegality:
    """Checks one model per call. The validator process starts at the first call and stays up until close()."""

    def __init__(self, teams=TEAMS, ask=ASK, guidance=GUIDANCE, temperature=TEMPERATURE, seed=SEED):
        self.teams, self.ask, self.guidance, self.temperature, self.seed = teams, ask, guidance, temperature, seed
        self.validator = None

    def __call__(self, model):
        """The legality fields for model (a Model84); {} when teams is 0."""
        if not self.teams: return {}
        t0 = time.perf_counter()
        if self.validator is None: self.validator = Validator()
        rows = self._draw(model)
        reasons = Counter()
        for row in rows:
            err = self.validator(model.to_paste(row, repair=True)[0])
            if err is not None: reasons[reason(err)] += 1
        return dict(illegal_share=sum(reasons.values()) / len(rows), legality_teams=len(rows),
                    rejection_reasons=[[r, n] for r, n in reasons.most_common(TOP_REASONS)],
                    legality_seconds=round(time.perf_counter() - t0, 3))

    def _draw(self, model):
        device = next(model.nets[0][0].parameters()).device.type
        with torch.random.fork_rng(devices=[] if device == "cpu" else [0], device_type=device):
            return model.sample(self.teams, ask=self.ask, guidance=self.guidance, temperature=self.temperature,
                                seed=self.seed).cpu().numpy()

    def close(self):
        if self.validator is not None: self.validator.close(); self.validator = None
