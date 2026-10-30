"""Defensive type coverage: for each attacking type, how many of a team's slots are weak and how many resist.

The D-Wave team-building webinar (2019, "Building an Optimal Pokémon Team with a Quantum Computer")
put two rules on a team as penalty terms of a quadratic unconstrained binary optimization (QUBO):
at most two members weak to any attacking type, and at least one member resisting it. Here they are
a measurement, never a search constraint: the search learns from battles, so these counts say whether
it found type coverage on its own, e.g. by comparing the top-50 meta teams with search output.

Types only, of the form the paste names: "Charizard-Mega-X" gets the Mega's Fire/Dragon, while
"Charizard" holding Charizardite X keeps Fire/Flying. Abilities (Levitate) and items (Air Balloon)
are ignored. A species missing from data/pokedex.json raises instead of counting as neither weak
nor resistant, which would quietly inflate the share of teams that pass.

    from type_coverage import defensive_coverage, coverage_failures
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperheuristic import TYPES, resistances, weaknesses
from hyperheuristic import types as slot_types

MAX_WEAK, MIN_RESIST = 2, 1


def defensive_coverage(team):
    """{attacking type: (weak count, resist count)} over all 18 types. An immunity counts as resisting."""
    unknown = [s["species"] for s in team if not slot_types(s)]
    if unknown:
        raise ValueError(f"species not in data/pokedex.json: {unknown}")
    weak = [weaknesses(s) for s in team]
    resist = [resistances(s) for s in team]
    return {t: (sum(t in w for w in weak), sum(t in r for r in resist)) for t in TYPES}


def coverage_failures(team):
    """Attacking types that break either rule: more than MAX_WEAK slots weak, or fewer than MIN_RESIST resisting."""
    cov = defensive_coverage(team)
    return {"too_many_weak": [t for t, (w, _) in cov.items() if w > MAX_WEAK],
            "too_few_resist": [t for t, (_, r) in cov.items() if r < MIN_RESIST]}
