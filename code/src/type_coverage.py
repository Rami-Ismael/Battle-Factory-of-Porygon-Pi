"""Defensive type coverage: for each attacking type, how many of a team's slots are weak and how many resist.

The D-Wave team-building webinar (2019, "Building an Optimal Pokémon Team with a Quantum Computer")
put two rules on a team as QUBO penalties: at most two members weak to any attacking type, and at
least one member resisting it. Here they are a measurement, never a search constraint: the search
learns from battles, so these counts say whether it found type coverage on its own, e.g. by comparing
the top-50 meta teams with search output.

Types only, as the species is written in the paste: abilities (Levitate), items (Air Balloon) and
Mega Evolution are ignored. A species missing from data/pokedex.json raises instead of counting as
neither weak nor resistant, which would quietly inflate the pass rate.

    from type_coverage import defensive_coverage, coverage_failures
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperheuristic import TYPES, resistances, weaknesses
from hyperheuristic import types as slot_types


def defensive_coverage(team):
    """{attacking type: (weak count, resist count)} over all 18 types. An immunity counts as resisting."""
    unknown = [s["species"] for s in team if not slot_types(s)]
    if unknown:
        raise ValueError(f"species not in data/pokedex.json: {unknown}")
    weak = [weaknesses(s) for s in team]
    resist = [resistances(s) for s in team]
    return {t: (sum(t in w for w in weak), sum(t in r for r in resist)) for t in TYPES}


def coverage_failures(team, max_weak=2, min_resist=1):
    """Attacking types that break either rule: more than max_weak slots weak, or fewer than min_resist resisting."""
    cov = defensive_coverage(team)
    return {"too_many_weak": [t for t, (w, _) in cov.items() if w > max_weak],
            "too_few_resist": [t for t, (_, r) in cov.items() if r < min_resist]}
