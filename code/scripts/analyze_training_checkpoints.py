"""Recompute the recorded checkpoint comparison without training or battles.

Inputs are immutable snapshots prepared from the saved grids and read-only battle
database. The matrix axes are ask, team, opponent. Each checkpoint uses sampling
seed 1, with 64 teams at each of 23 asks. Opponent bootstraps preserve the recorded
top-50 weights and share opponent draws between checkpoints.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

EPOCHS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 80)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--results', type=Path)
    inputs.add_argument('--grids-snapshot', type=Path)
    p.add_argument('--matrices', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstraps', type=int, default=4000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matrices = np.load(args.matrices)
    read_reference = gzip.open if args.reference.suffix == '.gz' else open
    with read_reference(args.reference, 'rt') as f:
        reference = json.load(f)
    if args.grids_snapshot:
        with gzip.open(args.grids_snapshot, 'rt') as f:
            snapshots = json.load(f)
    reference_keys = {tuple(row) for row in reference['lineups']}
    rng = np.random.default_rng(105)
    weights = matrices['opponent_weights']
    assert matrices['epochs'].tolist() == list(EPOCHS)
    assert int(matrices['sampling_seed']) == 1
    draws = rng.multinomial(len(weights), np.full(len(weights), 1 / len(weights)), args.bootstraps) * weights
    draws = draws / draws.sum(axis=1, keepdims=True)
    asks = np.arange(1, 24) / 24
    result = dict(epochs=[], pooled_repeats=[], differences=[], provenance={
        'training_seed': 0, 'sampling_seed': 1, 'bootstrap_seed': 105,
        'bootstraps': args.bootstraps, 'interval': '95% percentile bootstrap; whole teams within each ask and shared opponent columns across epochs; original opponent weights retained',
        'limits': 'Conditional on one training run and one sampling seed. Opponent resampling does not validate a different policy or opponent population. Epochs share an 80-epoch learning-rate schedule.',
        'opponents': len(weights), 'opponent_total_weight': float(weights.sum()),
        'policy_id': int(matrices['policy_id']), 'policy': 'bc_100.zip on both sides, shuffled slots',
        'simulator_revision': None,
        'audit': 'All 14,720 seed-1 teams have 49/49 opponent coverage and exactly 49 battles. Weighted matrix scores reproduce every saved score. The original checkpoint files matched all ten recorded hashes. Epoch-80 preflight passed, including its 20-team round trip. The battle database records showdown:unknown for the simulator revision.',
        'win_rate_definition': 'Weighted win rate against the top-50 list: 49 distinct opponents, one duplicated list entry has weight 2; behaviour-cloning policy on both sides, shuffled slots.',
        'reference': {k: v for k, v in reference.items() if k != 'lineups'},
        'reference_description': '54,766 distinct six-species line-ups from 170,785 labelled non-validation rows, fixed before the legal Pokémon construction tables dropped any row. Five rows were excluded from actual training. Novelty here is relative to that larger reference.',
        'matrix_sha256': sha(args.matrices), 'reference_sha256': sha(args.reference),
        'grids': [], 'best_validation_epoch': 80, 'stopped_epoch': 80,
        'stopping_reason': '80-epoch cap, not patience',
    })
    folder = 'code/reports/training-checkpoints-2026-11-05'
    result['provenance']['reproduce'] = (
        f'python3 code/scripts/analyze_training_checkpoints.py --grids-snapshot {folder}/grids.json.gz '
        f'--matrices {folder}/outcomes.npz --reference {folder}/reference.json.gz --output {folder}\n'
        'python3 code/scripts/render_checkpoint_report.py\n'
        '# Add --figures to the renderer for the evidence and teaser ECharts pages.'
    )
    boots = {}
    for ep in EPOCHS:
        if args.grids_snapshot:
            snapshot = snapshots[str(ep)]
            raw = snapshot['grid']
            source_path, source_sha = snapshot['source_path'], snapshot['source_sha256']
        else:
            path = args.results / f'stat_points_84_gen84_noise_byepoch_ep{ep:02d}.json'
            raw = json.loads(path.read_text())
            source_path, source_sha = str(path), sha(path)
        cells = sorted((c for c in raw['cells'].values() if c['seed'] == 1), key=lambda c: c['numerator'])
        assert [c['numerator'] for c in cells] == list(range(1, 24))
        assert all(len(c['y']) == (len(c['pastes']) if 'pastes' in c else c['pastes_count']) == len(c['stat_points']) == 64 for c in cells)
        y = np.array([c['y'] for c in cells])
        index = EPOCHS.index(ep)
        matrix = matrices['outcomes'][index]
        assert matrix.shape == (23, 64, len(weights)) and np.isfinite(matrix).all()
        assert (matrices['battles'][index] == 1).all()
        np.testing.assert_allclose(matrix @ weights / weights.sum(), y, atol=1e-12)
        # Each opponent draw is common to all epochs. Teams are independent
        # draws within each checkpoint and ask, not paired by array position.
        team_scores = (matrix.reshape(-1, len(weights)) @ draws.T).T.reshape(args.bootstraps, 23, 64)
        indices = rng.integers(0, 64, size=team_scores.shape)
        boot_means = np.take_along_axis(team_scores, indices, axis=2).mean(axis=2)
        boot_error = ((boot_means[:, :17] - asks[:17]) ** 2).mean(axis=1)
        boots[ep] = boot_error
        mean = y.mean(axis=1)
        mse = float(((mean[:17] - asks[:17]) ** 2).mean())
        row = dict(epoch=ep, mse=mse, mse_ci=interval(boot_error),
                   upper_mse=float(((mean[17:] - asks[17:]) ** 2).mean()),
                   slope_all_asks=float(np.polyfit(asks, mean, 1)[0]),
                   rmse=float(np.sqrt(mse)), teams=int(y.size), battles=int(matrix.size),
                   mean_within_ask_sd=float(y.std(axis=1, ddof=1).mean()), asks=[])
        all_lineups, new_lineups, full_points = set(), set(), []
        for i, cell in enumerate(cells):
            keys = {tuple(sorted(''.join(c.lower() for c in sp if c.isalnum()) for sp, _ in team)) for team in cell['stat_points']}
            novel = keys - reference_keys
            full = [sum(points.values()) == 66 for team in cell['stat_points'] for _, points in team]
            all_lineups.update(keys)
            new_lineups.update(novel)
            full_points.extend(full)
            lo, hi = interval(boot_means[:, i])
            row['asks'].append(dict(numerator=i + 1, ask=float(asks[i]), win=float(mean[i]),
                                    sd=float(y[i].std(ddof=1)), lo=lo, hi=hi,
                                    distinct_lineups=len(keys), new_lineups=len(novel),
                                    full_points_share=float(np.mean(full))))
        row.update(distinct_lineups=len(all_lineups), new_lineups=len(new_lineups),
                   full_points_share=float(np.mean(full_points)))
        result['epochs'].append(row)
        result['provenance']['grids'].append(dict(epoch=ep, path=source_path, sha256=source_sha,
                    checkpoint=raw['checkpoint'], checkpoint_sha256=raw['checkpoint_sha256'],
                    recorded_seeds=sorted({c['seed'] for c in raw['cells'].values()})))
        if ep in (21, 80):
            pooled = np.array([np.mean([v for c in raw['cells'].values() if c['numerator'] == k for v in c['y']]) for k in range(1,24)])
            result['pooled_repeats'].append(dict(epoch=ep, seeds=[1,2,3], mse=float(((pooled[:17] - asks[:17])**2).mean()),
                                                upper_mse=float(((pooled[17:] - asks[17:])**2).mean())))
    for earlier, later in ((1,34), (21,34), (21,80), (34,55), (34,80), (55,80)):
        a = next(r for r in result['epochs'] if r['epoch'] == earlier)
        b = next(r for r in result['epochs'] if r['epoch'] == later)
        result['differences'].append(dict(earlier=earlier, later=later, error_reduction=a['mse']-b['mse'],
                                           ci=interval(boots[earlier] - boots[later])))
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = ['epoch','mse','upper_mse','rmse','teams','battles','distinct_lineups','new_lineups','full_points_share','mean_within_ask_sd','slope_all_asks']
    with (args.output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in result['epochs'])
    with (args.output / 'asks.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', *result['epochs'][0]['asks'][0].keys()], lineterminator='\n')
        writer.writeheader()
        writer.writerows(dict(epoch=r['epoch'], **a) for r in result['epochs'] for a in r['asks'])
    print(json.dumps({'epochs':[{k:r[k] for k in fields[:2]+fields[6:]} for r in result['epochs']], 'differences':result['differences']}, indent=2))


if __name__ == '__main__':
    main()
