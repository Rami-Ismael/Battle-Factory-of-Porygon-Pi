"""Unit tests for src/slot_order.py: every battle gets the same six Pokémon in a new order."""
import os
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from showdown_validator import VGC_BENCH
if VGC_BENCH.exists():      # the pinned vgc-bench checkout: Builder runs on the laptop and skips on CI
    sys.path.insert(0, str(VGC_BENCH))
import slot_order as SO

PASTE = "\n\n".join(f"{name} @ Leftovers\nAbility: Pressure\nLevel: 50\n- Protect"
                    for name in ["Basculegion", "Chandelure", "Floette-Eternal", "Garchomp", "Kingambit", "Sylveon"]) + "\n"


def names(paste):
    return [b.split(" @ ")[0] for b in paste.strip().split("\n\n")]


class ShufflePaste(unittest.TestCase):
    def test_keeps_every_block_unchanged(self):
        out = SO.shuffle_paste(PASTE, random.Random(1))
        self.assertEqual(sorted(out.strip().split("\n\n")), sorted(PASTE.strip().split("\n\n")))

    def test_every_pokemon_reaches_slot_one(self):
        rng = random.Random(2)
        first = Counter(names(SO.shuffle_paste(PASTE, rng))[0] for _ in range(600))
        self.assertEqual(len(first), 6)
        self.assertTrue(all(60 <= c <= 140 for c in first.values()), first)

    def test_seeded_rng_repeats(self):
        a = [SO.shuffle_paste(PASTE, r) for r in [random.Random(7)] for _ in range(5)]
        b = [SO.shuffle_paste(PASTE, r) for r in [random.Random(7)] for _ in range(5)]
        self.assertEqual(a, b)


class OnlyShuffled(unittest.TestCase):
    """Every battle shuffles: listed order was removed 2026-11-07 (docs/slot-order.md)."""
    def test_policy_is_tagged_shuffled(self):
        self.assertEqual((SO.MODE, SO.POLICY_TAG), ("shuffled", "+slots-shuffled"))

    def test_listed_order_refuses_to_start(self):
        src = str(Path(__file__).resolve().parent.parent / "src")
        run = subprocess.run([sys.executable, "-c", "import slot_order"], cwd=src, capture_output=True, text=True,
                             env={**os.environ, "VGC_SLOT_ORDER": "listed"})
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("every battle shuffles slots", run.stderr)


@unittest.skipIf(SO.RandomTeamBuilder is object, "vgc_bench is not importable")
class Builder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "team.txt"
        self.path.write_text(PASTE)

    def tearDown(self):
        self.dir.cleanup()

    def species_order(self, packed):
        return [mon.split("|")[0] or mon.split("|")[1] for mon in packed.split("]")]

    def test_yield_team_reorders_and_keeps_the_six(self):
        tb = SO.ShuffledTeamBuilder(1, None, "mb", custom_team_paths=[self.path], order_seed=3)
        orders = [tuple(self.species_order(tb.yield_team())) for _ in range(30)]
        self.assertTrue(all(sorted(o) == sorted(orders[0]) for o in orders))
        self.assertGreater(len(set(orders)), 10)

    def test_shuffle_does_not_touch_global_random(self):
        random.seed(5); expected = random.random()
        random.seed(5)
        tb = SO.ShuffledTeamBuilder(1, None, "mb", custom_team_paths=[self.path], order_seed=None)
        random.seed(5); tb._load_team(self.path)
        self.assertEqual(random.random(), expected)


if __name__ == "__main__":
    unittest.main()
