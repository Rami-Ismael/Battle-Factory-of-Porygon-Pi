#!/usr/bin/env python3
"""Coverage arithmetic only. No team enumeration or simulated battle results."""
import argparse
import json
from decimal import Decimal, localcontext
from math import ceil, log


def coverage_cost(team_count, battles_per_pair=400, games_per_second=1000):
    if team_count < 1 or battles_per_pair < 1 or games_per_second <= 0:
        raise ValueError('Counts and throughput must be positive')
    ordered_cells = team_count ** 2
    symmetric_pairs = team_count * (team_count + 1) // 2
    with localcontext() as context:
        context.prec = 50
        return {
            'teams': team_count,
            'ordered_cells': ordered_cells,
            'symmetric_pairs_including_diagonal': symmetric_pairs,
            'dense_counter_payload_bytes': 32 * ordered_cells,
            'symmetric_counter_payload_bytes': 32 * symmetric_pairs,
            'ordered_completed_battles': battles_per_pair * ordered_cells,
            'symmetric_completed_battles': battles_per_pair * symmetric_pairs,
            'assumed_completed_games_per_second': str(games_per_second),
            'symmetric_wall_years': str(
                Decimal(battles_per_pair * symmetric_pairs)
                / Decimal(str(games_per_second)) / Decimal(31557600)),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teams', type=int, nargs='+',
                        default=[100, 1000, 10000, 1000000, 2**180])
    parser.add_argument('--battles-per-pair', type=int, default=400)
    parser.add_argument('--games-per-second', type=Decimal, default=Decimal(1000))
    args = parser.parse_args()
    for team_count in args.teams:
        print(json.dumps(coverage_cost(team_count, args.battles_per_pair,
                                       args.games_per_second)))
    # Hoeffding, fixed independent sample counts: P(|estimate-p|>=e)<=2exp(-2ne²).
    print(json.dumps({'hoeffding_per_pair_n_for_error_0.05_confidence_0.95':
                      ceil(log(2 / 0.05) / (2 * 0.05**2))}))


if __name__ == '__main__':
    main()
