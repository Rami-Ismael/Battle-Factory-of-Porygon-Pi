"""The sampling seed base of an ask grid: cell seed s draws with torch.manual_seed(seed0 + s).

Every grid drawn before 2026-11-15 used 7300. Its teams were battled and entered the training table (lossdata84.py reads
every results file and the matchup database), so a later model given the same random numbers redrew many of them: 67-70%
of Linear, species-first's line-ups at ask 6/24 replayed an earlier grid at the same seed, and its novelty read 0.19
against 0.45 on other seeds. A stored grid keeps the base its file records, so it still reproduces; a new grid gets a
base from its own name, so no two grids share a sampling seed.
"""
import hashlib
import json
from pathlib import Path

LEGACY = 7300                                            # every grid drawn before 2026-11-15
SPAN = 10                                                # room for seeds 1 to 9 under one base


def seed0(name, path):
    """The seed base of the grid called `name` whose results file is `path`: the recorded one if the file exists
    (7300 if it never wrote one), else 100,000 plus a multiple of SPAN taken from the name's SHA-256."""
    path = Path(path)
    if path.exists(): return int(json.loads(path.read_text()).get("seed0", LEGACY))
    return 100_000 + int(hashlib.sha256(name.encode()).hexdigest()[:12], 16) % 10**8 * SPAN
