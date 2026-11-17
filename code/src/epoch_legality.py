"""First-try legality during training (issue #248): after an epoch, the model as it stands draws a small batch of teams
at one ask, Showdown's validator checks each, and the epoch's log entry gains the illegal share, the curve the check
exists for (owner, 2026-11-11: it should fall over training). The first-try legal share is 1 minus it.

The draw is the ask grid's (ask_grid84.draw): guidance 2, temperature 1, species-first order, Stat Alignment repaired
to match the moves before the validator. The ask is the elite cut, 0.531, and every epoch draws from the same seed,
so the curve moves only when the model does. The draw runs in a forked random state, so training's own batches,
masks and times are the same with the check on or off.

tables=False draws without the legal Pokémon construction tables and the Stat Point budget (issue #361): any value but
[MASK] can be drawn, so the illegal share shows what the network learned. With them, the share is 0 by construction.
Showdown's validator rejects a Pokémon over the 66 Stat Point budget itself, so no second check is needed.

Drawn without the tables, a team usually breaks several rules at once (2026-11-16: the default model's epoch-80 weights
drew 32 of 32 teams illegal), so the whole-team share can sit at 1 while the network improves. errors_per_team, the mean
count of the validator's errors over all teams drawn (0 for a legal team), keeps moving then. rejection_reasons counts
each kind of error once per team that has it.

known=team_keys(training rows) adds copied_share: the share of drawn teams identical to a training team, with slot order
and move order ignored. An exact copy is always legal, so a falling illegal share needs this beside it.

    check = EpochLegality(teams=64)        # teams=0: no check, no fields; tables=False: draw without the tables
    entry.update(check(model))             # illegal_share, errors_per_team, legality_teams, rejection_reasons, legality_seconds
    check.close()
"""
import re
import time
from collections import Counter

import numpy as np
import torch

from hps_generate import Validator

TEAMS = 64                                  # standard error about ±0.06 on the share (owner, 2026-11-08)
ASK = 0.531                                 # the elite cut
GUIDANCE = 2.0                              # the ask grid's default guidance
TEMPERATURE = 1.0
SEED = 12345                                # every epoch draws the same random numbers, like the validation loss
TOP_REASONS = 5


VERB = re.compile(r"\b(can't|cannot|has|is|must|needs|transforms)\b")
NAMED = re.compile(r"\b(can't have|can't learn|can't hold|multiple copies of|transforms in-battle with) [^.,;]+")


FIELDS, SLOTS, STAT_POINTS = 8, 6, 6                 # fields per Pokémon in the 48-field block, Pokémon, Stat Points per Pokémon
MOVES = slice(3, 7)


def team_key(row):
    """A team's identity with slot order and move order ignored: the sorted tuple of its six Pokémon."""
    row = [int(v) for v in row]; stat0 = FIELDS * SLOTS
    pokemon = []
    for s in range(SLOTS):
        f = row[s * FIELDS:(s + 1) * FIELDS]
        pokemon.append((*f[:3], *sorted(f[MOVES]), *f[7:], *row[stat0 + s * STAT_POINTS:stat0 + (s + 1) * STAT_POINTS]))
    return tuple(sorted(pokemon))


def team_keys(rows):
    return {team_key(r) for r in np.asarray(rows)}


def errors(err):
    """The validator's separate errors in one rejection (it joins them with "; ")."""
    return [e.strip() for e in str(err).split("\n")[0].split("; ") if e.strip()]


def kind(error):
    """One error with its Pokémon, move, ability and item names and its numbers removed, so counts group by kind:
    Milotic can't learn Heat Wave. -> can't learn ..."""
    verb = VERB.search(error)
    text = error[verb.start():] if verb else error
    return re.sub(r"\.$", "", re.sub(r"\s+", " ", re.sub(r"\d+", "N", NAMED.sub(r"\1 ...", text))).strip())[:80]


def reason(err):
    """The kind of a rejection's first error."""
    found = errors(err)
    return kind(found[0]) if found else ""


class EpochLegality:
    """Checks one model per call. The validator process starts at the first call and stays up until close()."""

    def __init__(self, teams=TEAMS, ask=ASK, guidance=GUIDANCE, temperature=TEMPERATURE, seed=SEED, tables=True, known=None):
        self.teams, self.ask, self.guidance, self.temperature, self.seed = teams, ask, guidance, temperature, seed
        self.tables, self.known = tables, known
        self.validator = None

    def __call__(self, model):
        """The legality fields for model (a Model84); {} when teams is 0."""
        if not self.teams: return {}
        t0 = time.perf_counter()
        if self.validator is None: self.validator = Validator()
        rows = self._draw(model)
        reasons, rejected, count = Counter(), 0, 0
        for row in rows:
            err = self.validator(model.to_paste(row, repair=True)[0])
            if err is None: continue
            found = errors(err); rejected += 1; count += len(found)
            reasons.update({kind(e) for e in found})
        copies = {} if self.known is None else dict(copied_share=sum(team_key(r) in self.known for r in rows) / len(rows))
        return dict(illegal_share=rejected / len(rows), errors_per_team=count / len(rows), **copies, legality_teams=len(rows),
                    rejection_reasons=[[r, n] for r, n in reasons.most_common(TOP_REASONS)],
                    legality_seconds=round(time.perf_counter() - t0, 3))

    def _draw(self, model):
        device = next(model.nets[0][0].parameters()).device.type
        with torch.random.fork_rng(devices=[] if device == "cpu" else [0], device_type=device):
            return model.sample(self.teams, ask=self.ask, guidance=self.guidance, temperature=self.temperature,
                                seed=self.seed, tables=self.tables).cpu().numpy()

    def close(self):
        if self.validator is not None: self.validator.close(); self.validator = None
