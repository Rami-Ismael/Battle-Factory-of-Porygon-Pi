"""The GPU speed probe (gpu_speed_probe_modal.py): which masks a timing fills, and the report it prints from the
per-GPU result files. No Modal or GPU is used."""
import json
import tempfile
import unittest
from pathlib import Path

import gpu_speed_probe_modal as P


def fill(batch, seconds_per_fill, tokens_per_second=1000.0, status='ok'):
    return {'batch': batch, 'status': status, 'seconds_per_fill': seconds_per_fill, 'forward_share': 0.9,
            'tokens_per_second': tokens_per_second, 'peak_gib': 2.0}


def result(gpu, full, variants=None):
    return {'gpu_asked': gpu, 'device': f'NVIDIA {gpu}', 'full': full,
            'variants': variants or [{'name': P.AS_RUN, 'fills': [fill(32, 1.0, 1000.0)]},
                                     {'name': 'float32 weights, TF32 off', 'fills': [fill(32, 4.0, 250.0)]}]}


class MedianRowsTest(unittest.TestCase):
    def test_a_batch_is_centred_on_the_median_text(self):
        order = [f'm{n}' for n in range(10)]
        self.assertEqual(P.median_rows(order, 1), ['m4'])
        self.assertEqual(P.median_rows(order, 4), ['m3', 'm4', 'm5', 'm6'])
        self.assertEqual(P.median_rows(order, 10), order)

    def test_a_batch_larger_than_the_grid_is_refused(self):
        with self.assertRaises(ValueError):
            P.median_rows(['a', 'b'], 3)
        with self.assertRaises(ValueError):
            P.median_rows(['a', 'b'], 0)


class ReportTest(unittest.TestCase):
    def test_cost_per_thousand_fills_is_the_hourly_price_times_the_fill_time(self):
        r = result('L4', [fill(142, 1.8)])
        self.assertAlmostEqual(P.dollars_per_thousand(r, r['full'][0]), 0.80 / 3600 * 1.8 * 1000)

    def test_the_largest_whole_fill_that_ran_is_the_headline(self):
        r = result('L4', [fill(1, 1.2), fill(142, 0.4), fill(270, 0.0, status='over memory')])
        self.assertEqual(P.full_fill(r)['batch'], 142)
        self.assertEqual(P.full_fill(r, 1)['batch'], 1)
        self.assertIsNone(P.full_fill(result('L4', [fill(142, 0.0, status='over memory')])))

    def test_each_precision_is_read_against_the_fills_own_setting(self):
        text = P.report([result('L4', [fill(142, 1.8), fill(270, 0.0, status='over memory')])])
        self.assertIn('| NVIDIA L4 | $0.80 | 142 | 1.800 | $0.400 | 90% | 2.0 GiB |', text)
        self.assertIn('| NVIDIA L4 | $0.80 | 270 | over memory | | | |', text)
        self.assertIn('| NVIDIA L4 | float32 weights, TF32 off | 32 | 250 | 0.25x |', text)
        self.assertIn(f'| NVIDIA L4 | {P.AS_RUN} | 32 | 1,000 | 1.00x |', text)

    def test_results_read_back_cheapest_gpu_first(self):
        with tempfile.TemporaryDirectory() as d:
            for gpu in ('H100', 'L4', 'L40S'):
                (Path(d) / f'{gpu}-diffugpt_m.json').write_text(json.dumps(result(gpu, [fill(1, 1.0)])))
            (Path(d) / 'L4-qwen3.json').write_text(json.dumps(result('L4', [fill(1, 1.0)])))
            self.assertEqual([r['gpu_asked'] for r in P.read_results('diffugpt_m', Path(d))], ['L4', 'L40S', 'H100'])


if __name__ == '__main__':
    unittest.main()
