import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline as p


def example_team():
    return [{'slot': i + 1, 'species': f'Species{i}', 'item': f'Item{i}',
             'ability': 'Ability', 'nature': 'Adamant',
             'statPoints': {'HP': 1, 'Atk': 32, 'Spe': 32},
             'moves': ['Move1', 'Move2', 'Move3', 'Move4'],
             'ivs': None} for i in range(6)]


class MaskTests(unittest.TestCase):
    def test_every_task_and_k_is_reproducible_and_preserves_input(self):
        original = example_team()
        pristine = copy.deepcopy(original)
        for task in ('items', 'stats', 'moves', 'pokemon', 'abilities', 'natures'):
            maximum = 24 if task == 'moves' else 6
            for k in range(1, maximum + 1):
                masked, slots = p.mask_team(original, task, k, 44)
                self.assertEqual(len(slots), k)
                self.assertEqual((masked, slots), p.mask_team(original, task, k, 44))
                p.check_completion(masked, original, task, slots)
                for slot in slots:
                    if task == 'pokemon':
                        self.assertIsNone(masked[slot])
                    elif task == 'moves':
                        self.assertIsNone(masked[slot // 4]['moves'][slot % 4])
                    else:
                        self.assertIsNone(masked[slot][p.FIELDS[task]])
        self.assertEqual(original, pristine)

    def test_protects_unmasked_values_and_nullable_features(self):
        original = example_team()
        masked, slots = p.mask_team(original, 'items', 1, 3)
        changed = copy.deepcopy(original)
        changed[slots[0]]['item'] = 'Another item'
        p.check_completion(masked, changed, 'items', slots)
        for field, value in [('nature', 'Jolly'), ('ivs', {'HP': 31}), ('teraType', 'Water')]:
            bad = copy.deepcopy(changed)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                p.check_completion(masked, bad, 'items', slots)

    def test_missing_spread_and_moves_fail(self):
        original = example_team()
        for task in ('stats', 'moves', 'pokemon'):
            masked, slots = p.mask_team(original, task, 1, 1)
            with self.assertRaises(ValueError):
                p.check_completion(masked, masked, task, slots)

    def test_whole_pokemon_can_change_identity_and_set(self):
        original = example_team()
        masked, slots = p.mask_team(original, 'pokemon', 2, 4)
        result = copy.deepcopy(original)
        for slot in slots:
            result[slot]['species'] = 'New species'
            result[slot]['item'] = 'New item'
        p.check_completion(masked, result, 'pokemon', slots)

    def test_inapplicable_tera_and_invalid_k_rejected(self):
        for task, k in [('tera', 1), ('stats', 7), ('moves', 25), ('unknown', 1), ('items', 0)]:
            with self.assertRaises(ValueError):
                p.mask_team(example_team(), task, k, 0)


class SelectionTests(unittest.TestCase):
    def test_round_robin_selection_uses_newest_first_and_eligible_rank(self):
        rows = [{'id': name, 'tournament': tournament, 'date': day, 'placement': place}
                for name, tournament, day, place in [
                    ('old1', 'old', '2026-08-01', 1),
                    ('new2', 'new', '2026-09-01', 2),
                    ('new5', 'new', '2026-09-01', 5),
                    ('old3', 'old', '2026-08-01', 3)]]
        self.assertEqual([r['id'] for r in p.select_teams(rows, 'top50', 3, 1)],
                         ['new2', 'old1', 'new5'])

    def test_missing_metadata_fails(self):
        with self.assertRaises(ValueError):
            p.select_teams([{'id': 'a'}], 'top50', 50, 1)


class CacheTests(unittest.TestCase):
    def test_cache_identity_includes_seed_settings_full_sets_and_simulator(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = p.BattleCache(Path(folder) / 'db.sqlite')
            config = {'format': 'format1', 'battle': {'type': 'showdown-random', 'policy_version': 'v1'}}
            team = example_team()
            with patch.object(p, 'bridge', return_value={'outcome': 'win'}) as battle:
                self.assertFalse(cache.evaluate(config, team, team, 1, 'sim1')[1])
                self.assertTrue(cache.evaluate(config, team, team, 1, 'sim1')[1])
                self.assertFalse(cache.evaluate(config, team, team, 2, 'sim1')[1])
                self.assertFalse(cache.evaluate(config, team, team, 1, 'sim2')[1])
                altered = copy.deepcopy(team)
                altered[0]['item'] = 'Different'
                self.assertFalse(cache.evaluate(config, altered, team, 1, 'sim1')[1])
                config['battle']['policy_version'] = 'v2'
                self.assertFalse(cache.evaluate(config, team, team, 1, 'sim1')[1])
                self.assertEqual(battle.call_count, 5)
            cache.db.close()

    def test_simulation_errors_are_never_cached_as_draws(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = p.BattleCache(Path(folder) / 'db.sqlite')
            config = {'format': 'format', 'battle': {'type': 'showdown-random'}}
            with patch.object(p, 'bridge', return_value={'outcome': 'error'}):
                with self.assertRaises(ValueError):
                    cache.evaluate(config, example_team(), example_team(), 1, 'sim')
            self.assertEqual(cache.db.execute('SELECT COUNT(*) FROM battles').fetchone()[0], 0)
            cache.db.close()


class AdapterTests(unittest.TestCase):
    def test_legacy_null_tera_is_removed_and_nonnull_tera_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'teams.json'
            team = example_team()
            for mon in team:
                mon['teraType'] = None
            p.save_json(file, {'teams': [{'team': team}]})
            loaded = p.load_teams(file)[0]['team']
            self.assertTrue(all('teraType' not in mon for mon in loaded))
            team[0]['teraType'] = 'Water'
            p.save_json(file, {'teams': [{'team': team}]})
            with self.assertRaises(ValueError):
                p.load_teams(file)

    def test_command_method_receives_masked_input_not_hidden_original(self):
        masked, slots = p.mask_team(example_team(), 'items', 1, 3)
        request = {'masked_team': masked, 'task': {'slots': slots}, 'meta_teams': []}
        with patch.object(p, 'command_json', return_value={'completed_team': example_team()}) as call:
            p.generate({'type': 'command', 'command': ['adapter']}, request, [], 1, 30)
            self.assertNotIn('original', call.call_args.args[1])
            self.assertIsNone(call.call_args.args[1]['masked_team'][slots[0]]['item'])

    def test_openrouter_missing_key_does_not_make_a_request(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(p.urllib.request, 'urlopen') as http:
            with self.assertRaises(ValueError):
                p.generate({'type': 'openrouter'}, {}, [], 1, 30)
            http.assert_not_called()

    def test_openrouter_json_response_and_usage_are_preserved(self):
        masked, slots = p.mask_team(example_team(), 'stats', 1, 4)
        request = {'masked_team': masked, 'meta_teams': [],
                   'task': {'masked_paths': p.masked_paths('stats', slots)}}
        response = {'id': 'response-id', 'model': 'returned-model', 'usage': {'cost': 0.01},
                    'choices': [{'message': {'content': json.dumps({'completed_team': example_team()})}}]}
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}), \
             patch.object(p.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as http:
            result, metadata = p.generate({'type': 'openrouter', 'model': 'test-model'}, request, [], 1, 30)
            self.assertEqual(result['completed_team'], example_team())
            self.assertEqual(metadata['usage']['cost'], 0.01)
            body = json.loads(http.call_args.args[0].data)
            self.assertEqual(body['model'], 'test-model')
            self.assertNotIn('original', json.loads(body['messages'][1]['content']))


class ShowdownIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent
        self.config = p.read_json(self.root / 'smoke.json')
        if not (Path(self.config['showdown_runtime']) / 'dist/sim').exists():
            self.skipTest('Local built Showdown runtime is unavailable')
        self.team = p.load_teams(self.root / self.config['starting_teams'])[0]['team']

    def test_real_validator_accepts_fixture_and_rejects_illegal_move(self):
        p.validate_team(self.config, self.team)
        invalid = copy.deepcopy(self.team)
        invalid[0]['moves'][0] = 'Definitely Not A Real Move'
        with self.assertRaises(ValueError):
            p.validate_team(self.config, invalid)

    def test_real_validator_rejects_excess_stat_points(self):
        invalid = copy.deepcopy(self.team)
        invalid[0]['statPoints']['HP'] = 999
        with self.assertRaises(ValueError):
            p.validate_team(self.config, invalid)

    def test_import_preserves_champions_spreads(self):
        paste = (self.root.parent / 'Blog/visuals/search-loop/team-example.txt').read_text()
        imported = p.bridge(self.config, 'import', paste=paste)['team']
        p.validate_team(self.config, imported)
        for actual, expected in zip(imported, self.team):
            self.assertEqual(actual['species'], expected['species'])
            for key in ('HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'):
                self.assertEqual(actual['statPoints'].get(key, 0), expected['statPoints'].get(key, 0))


if __name__ == '__main__':
    unittest.main()
