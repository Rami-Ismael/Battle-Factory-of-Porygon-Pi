"""The held-out grid runner (#237) on the CPU, with a fake tokenizer, a fake model and a fake validator.

The tasks are built here from code/data/regmb-vocabulary.json, so no vault file, checkpoint or GPU is read and the
test is in ci_tests.txt.
"""
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import torch

import inplace_prompt_grid as G
import masked_diffusion_text as D
from mask_paste import STATS, blank_slot


class FakeTokenizer:
    """A run of letters is one token, anything else one per character; mask and pad markers are one token each.
    Ids are handed out as tokens are first seen."""
    mask_token, pad_token, eos_token = '<|mask|>', '<|pad|>', '<|pad|>'
    bos_token_id, unk_token_id = None, None
    pattern = r'<\|mask\|>|<\|pad\|>|[A-Za-z]+|.'

    def __init__(self):
        self.ids = {self.mask_token: 0, self.pad_token: 1}
        self.tokens = [self.mask_token, self.pad_token]
        self.all_special_ids = [0, 1]

    def id(self, token):
        if token not in self.ids:
            self.ids[token] = len(self.tokens)
            self.tokens.append(token)
        return self.ids[token]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        matches = list(re.finditer(self.pattern, text, re.S))
        return {'input_ids': [self.id(m.group()) for m in matches], 'offset_mapping': [m.span() for m in matches]}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.tokens[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids))

    def __len__(self):
        return len(self.tokens)


VOCAB = json.loads(D.VOCABULARY.read_text())


def name(field, n):
    ids = [v for v in VOCAB['values'][field] if v]
    return VOCAB['names'][ids[n]]


def team():
    return [{'slot': i + 1, 'species': name('species', 3 * i), 'item': name('item', i), 'ability': name('ability', i),
             'nature': name('nature', i), 'moves': [name('move', 4 * i + j) for j in range(4)],
             'statPoints': dict(zip(STATS, (2, 0, 0, 32, 0, 32), strict=True)), 'ivs': dict.fromkeys(STATS, 31),
             'level': 50} for i in range(6)]


def make_task(kind, task_id):
    original = team()
    masked = copy.deepcopy(original)
    if kind == 'items':
        masked[2]['item'] = None
        paths = ['team[2].item']
    elif kind == 'moves':
        masked[1]['moves'][3] = None
        paths = ['team[1].moves[3]']
    else:
        masked[4] = None
        paths = ['team[4]']
    return {'id': task_id, 'source': 'TEST', 'task': kind, 'k': 1, 'seed': 0, 'slots': [],
            'masked_team': masked, 'masked_paths': paths, 'original': original}


TASKS = [make_task('items', 'a1'), make_task('moves', 'b2'), make_task('pokemon', 'c3')]
PROMPT = G.PROMPTS['inplace_header']


class FakeValidator:
    def __init__(self):
        self.pastes = []

    def __call__(self, paste):
        self.pastes.append(paste)
        return 'Pikachu has more than 32 Stat Points in HP.' if 'EVs: 99' in paste else None


def answer_model(tokenizer, task, vocab_size=None):
    """A model whose logits at each slot put all weight on the true value's token (padded as training pads)."""
    target = D.training_target(task, G.ASK, tokenizer, G.LIMIT, tokenizer.pad_token)
    truth = tokenizer(target, add_special_tokens=False)['input_ids']

    def forward_for(slots):
        def forward(x):
            logits = torch.full((x.shape[0], len(slots), len(tokenizer) + 5), -1e9)
            for n, position in enumerate(slots.tolist()):
                logits[:, n, truth[position]] = 0.0
            return logits
        return forward
    return forward_for


def quote_model(tokenizer):
    """A model that writes a quote into every slot token."""
    quote = tokenizer.id('"')

    def forward_for(slots):
        def forward(x):
            logits = torch.full((x.shape[0], len(slots), len(tokenizer) + 5), -1e9)
            logits[..., quote] = 0.0
            return logits
        return forward
    return forward_for


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer, self.names, self.validate = FakeTokenizer(), G.vocabulary(), FakeValidator()

    def built(self, task):
        return G.canvas(task, PROMPT, self.tokenizer, 0)['built']

    def test_the_true_values_are_legal_for_every_kind_of_mask(self):
        for task in TASKS:
            built = self.built(task)
            verdict = G.judge(task, built, G.true_values(task), self.validate, self.names)
            self.assertTrue(verdict['legal'], (task['task'], verdict))
            self.assertIsNone(verdict['failed'])

    def test_a_value_that_breaks_its_quotes_does_not_parse(self):
        task = TASKS[0]
        verdict = G.judge(task, self.built(task), ['","'], self.validate, self.names)
        self.assertEqual(verdict['failed'], 'parses')
        self.assertFalse(verdict['json_parses'])

    def test_a_value_that_overwrites_a_visible_one_is_caught(self):
        """A filled item that writes a species key after the real one: the JSON parses, and json keeps the later key,
        so the visible species reads differently."""
        task = TASKS[0]
        mon = task['original'][2]
        fill = f'{mon["item"]}","species":"{name("species", 40)}'
        verdict = G.judge(task, self.built(task), [fill], self.validate, self.names)
        self.assertTrue(verdict['json_parses'])
        self.assertTrue(verdict['reads_as_team'])
        self.assertEqual(verdict['failed'], 'visible')
        self.assertFalse(verdict['legal'])

    def test_a_name_outside_reg_mb_fails_even_when_showdown_accepts(self):
        """Showdown's validator accepts a junk Stat Alignment (2026-11-18); the Reg M-B step does not."""
        task = make_task('pokemon', 'd4')
        built = self.built(task)
        values = G.true_values(task)
        values[[s.field for s in built.slots].index('nature')] = '00'
        verdict = G.judge(task, built, values, self.validate, self.names)
        self.assertEqual(verdict['failed'], 'regmb')
        self.assertTrue(verdict['showdown_accepts'])
        self.assertEqual(verdict['reasons'], ['not a Reg M-B Stat Alignment'])

    def test_names_match_by_showdown_id(self):
        task = TASKS[0]
        verdict = G.judge(task, self.built(task), [task['original'][2]['item'].lower()], self.validate, self.names)
        self.assertTrue(verdict['regmb_names'])

    def test_showdown_errors_are_grouped(self):
        task = make_task('pokemon', 'e5')
        built = self.built(task)
        values = G.true_values(task)
        values[[s.path for s in built.slots].index('team[4].statPoints.HP')] = '99'
        verdict = G.judge(task, built, values, self.validate, self.names)
        self.assertEqual(verdict['failed'], 'showdown')
        self.assertEqual(verdict['reasons'], ['over 32 Stat Points in one stat'])


class FillGridTest(unittest.TestCase):
    def test_a_perfect_model_fills_every_mask_legally_and_a_quote_model_none(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as d:
            for task in TASKS:
                G.fill_grid([task], PROMPT, tokenizer, answer_model(tokenizer, task), 0, Path(d) / 'good.jsonl',
                            FakeValidator())
            G.fill_grid(TASKS, PROMPT, tokenizer, quote_model(tokenizer), 0, Path(d) / 'bad.jsonl', FakeValidator())
            good = G.read_rows(Path(d) / 'good.jsonl')
            bad = G.read_rows(Path(d) / 'bad.jsonl')
        self.assertEqual([r['legal'] for r in good], [True, True, True])
        self.assertEqual([r['legal'] for r in bad], [False, False, False])
        self.assertEqual({r['failed'] for r in bad}, {'parses'})
        summary = G.summarize(good)
        self.assertEqual(summary['legal'], 3)
        self.assertTrue(summary['passes_169_bar'])
        self.assertEqual(summary['real_values']['species'], {'real': 1, 'of': 1})
        self.assertEqual(summary['legal_by_task']['items'], {'legal': 1, 'of': 1})
        self.assertEqual(summary['recovered_unmasked_team'], 3)
        self.assertEqual(G.summarize(bad)['recovered_unmasked_team'], 0)

    def test_every_row_records_the_prompt_text_size_seed_and_values(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as d:
            G.fill_grid(TASKS[:1], PROMPT, tokenizer, quote_model(tokenizer), 0, Path(d) / 'f.jsonl', FakeValidator())
            row = G.read_rows(Path(d) / 'f.jsonl')[0]
        self.assertEqual(row['prompt'], 'inplace_header')
        self.assertEqual(row['version'], D.VERSION)
        self.assertEqual(row['ask'], 'unknown')
        self.assertEqual(row['seed'], G.mask_seed(G.SETTING['seed'], 'a1'))
        self.assertEqual(row['paths'], ['team[2].item'])
        self.assertEqual(len(row['values']), 1)
        self.assertGreater(row['prompt_tokens'], row['mask_tokens'])

    def test_a_stopped_run_resumes_without_refilling(self):
        tokenizer, calls = FakeTokenizer(), []
        base = quote_model(tokenizer)

        def counting(slots):
            calls.append(len(slots))
            return base(slots)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'f.jsonl'
            self.assertEqual(G.fill_grid(TASKS[:2], PROMPT, tokenizer, counting, 0, out, FakeValidator()), 2)
            self.assertEqual(G.fill_grid(TASKS, PROMPT, tokenizer, counting, 0, out, FakeValidator()), 1)
            self.assertEqual([r['id'] for r in G.read_rows(out)], ['a1', 'b2', 'c3'])
        self.assertEqual(len(calls), 3)

    def test_a_masks_draw_does_not_depend_on_the_masks_before_it(self):
        tokenizer = FakeTokenizer()

        def noisy(slots):
            return lambda x: torch.zeros(x.shape[0], len(slots), len(tokenizer) + 5)
        with tempfile.TemporaryDirectory() as d:
            G.fill_grid(TASKS, PROMPT, tokenizer, noisy, 0, Path(d) / 'forward.jsonl', FakeValidator())
            G.fill_grid(TASKS[::-1], PROMPT, tokenizer, noisy, 0, Path(d) / 'reverse.jsonl', FakeValidator())
            forward = {r['id']: r['slot_ids'] for r in G.read_rows(Path(d) / 'forward.jsonl')}
            reverse = {r['id']: r['slot_ids'] for r in G.read_rows(Path(d) / 'reverse.jsonl')}
        self.assertEqual(forward, reverse)


class RedrawTest(unittest.TestCase):
    """#169's validator loop for #171: a failing mask is drawn again, up to 4 more times, until a draw is legal."""

    def first_tries(self, d, tokenizer):
        """a1 legal on the first try; b2 and c3 not."""
        first = Path(d) / 'fills.jsonl'
        G.fill_grid(TASKS[:1], PROMPT, tokenizer, answer_model(tokenizer, TASKS[0]), 0, first, FakeValidator())
        G.fill_grid(TASKS[1:], PROMPT, tokenizer, quote_model(tokenizer), 0, first, FakeValidator())
        return first

    def legal_on_call(self, tokenizer, task, n, calls):
        """Quotes until the nth draw of this task, then the true values."""
        bad, good = quote_model(tokenizer), answer_model(tokenizer, task)

        def forward_for(slots):
            calls.append(task['id'])
            return good(slots) if calls.count(task['id']) >= n else bad(slots)
        return forward_for

    def test_a_failing_mask_is_redrawn_until_legal_and_a_legal_one_is_left(self):
        tokenizer, calls = FakeTokenizer(), []
        with tempfile.TemporaryDirectory() as d:
            first, out = self.first_tries(d, tokenizer), Path(d) / 'redraws.jsonl'
            G.redraw_grid(TASKS[1:2], first, PROMPT, tokenizer, self.legal_on_call(tokenizer, TASKS[1], 2, calls), 0,
                          out, FakeValidator())
            G.redraw_grid(TASKS[:1], first, PROMPT, tokenizer, quote_model(tokenizer), 0, out, FakeValidator())
            rows = G.read_rows(out)
            final = G.final_draws(first, out)
        self.assertEqual([(r['id'], r['draw'], r['legal']) for r in rows], [('b2', 1, False), ('b2', 2, True)])
        self.assertIn('paste', rows[1])
        self.assertNotIn('paste', rows[0])
        self.assertEqual({k: (r['draw'], r['legal']) for k, r in final.items()},
                         {'a1': (0, True), 'b2': (2, True), 'c3': (0, False)})

    def test_a_mask_that_never_turns_legal_gets_four_redraws_with_distinct_seeds(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as d:
            first, out = self.first_tries(d, tokenizer), Path(d) / 'redraws.jsonl'
            self.assertEqual(G.redraw_grid(TASKS, first, PROMPT, tokenizer, quote_model(tokenizer), 0, out,
                                           FakeValidator()), 8)
            rows = G.read_rows(out)
            first_seeds = {r['seed'] for r in G.read_rows(first)}
        self.assertEqual([r['draw'] for r in rows if r['id'] == 'c3'], [1, 2, 3, 4])
        seeds = [r['seed'] for r in rows]
        self.assertEqual(len(set(seeds)), 8)
        self.assertFalse(set(seeds) & first_seeds)
        self.assertEqual(rows[0]['seed'], G.redraw_seed(G.SETTING['seed'], 'b2', 1))

    def test_a_stopped_redraw_resumes_at_the_next_draw(self):
        tokenizer, calls = FakeTokenizer(), []
        with tempfile.TemporaryDirectory() as d:
            first, out = self.first_tries(d, tokenizer), Path(d) / 'redraws.jsonl'
            G.redraw_grid(TASKS[1:2], first, PROMPT, tokenizer, quote_model(tokenizer), 0, out, FakeValidator(),
                          redraws=2)
            G.redraw_grid(TASKS[1:2], first, PROMPT, tokenizer, self.legal_on_call(tokenizer, TASKS[1], 1, calls), 0,
                          out, FakeValidator())
            self.assertEqual(G.redraw_grid(TASKS[1:2], first, PROMPT, tokenizer, quote_model(tokenizer), 0, out,
                                           FakeValidator()), 0)
            rows = G.read_rows(out)
        self.assertEqual([(r['draw'], r['legal']) for r in rows], [(1, False), (2, False), (3, True)])
        self.assertEqual(calls, ['b2'])

    def test_a_legal_first_try_carries_its_paste(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as d:
            row = G.read_rows(self.first_tries(d, tokenizer))[0]
        self.assertTrue(row['legal'])
        self.assertEqual(row['paste'], G.team_paste(TASKS[0]['original']))


class BattleTest(unittest.TestCase):
    def test_stat_points_full_counts_pokemon_spending_all_66(self):
        import inplace_prompt_battle as B
        paste = ('A @ B\nEVs: 32 HP / 32 Atk / 2 Spe\n- X\n\nC @ D\nEVs: 10 HP / 10 Atk / 10 Def / 10 SpA / 10 SpD / '
                 '10 Spe\n- Y\n')
        self.assertEqual(B.stat_points_full(paste), (1, 6))


class CanvasTest(unittest.TestCase):
    def test_a_start_token_moves_every_slot_up_by_one(self):
        tokenizer = FakeTokenizer()
        plain = G.canvas(TASKS[0], PROMPT, tokenizer, 0)
        started = G.canvas(TASKS[0], PROMPT, tokenizer, 0, start_id=99)
        self.assertEqual(started['ids'], [99, *plain['ids']])
        self.assertEqual(started['slots'], [n + 1 for n in plain['slots']])

    def test_the_ask_line_reads_unknown(self):
        self.assertIn('Target win rate vs top meta teams: unknown.', G.canvas(TASKS[0], PROMPT, FakeTokenizer(), 0)['built'].text)

    def test_special_tokens_in_a_slot_are_dropped_and_counted(self):
        tokenizer = FakeTokenizer()
        c = G.canvas(TASKS[0], PROMPT, tokenizer, 0)
        ids = list(c['ids'])
        span = c['spans'][0]
        ids[span[0]] = tokenizer.id('Leftovers')
        for n in span[1:]:
            ids[n] = 1
        values, dropped = G.slot_values(tokenizer, ids, [span])
        self.assertEqual(values, ['Leftovers'])
        self.assertEqual(dropped, len(span) - 1)


class IntervalTest(unittest.TestCase):
    def test_exact_interval_matches_known_values(self):
        """The #169 numbers: 3 of 270 starts at 0.23%; 0 of 270 ends at 1.36%."""
        self.assertAlmostEqual(G.exact_interval(3, 270)[0], 0.00229, places=4)
        self.assertAlmostEqual(G.exact_interval(0, 270)[1], 0.01357, places=4)
        low, high = G.exact_interval(135, 270)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_masks_ids_give_stable_distinct_seeds(self):
        self.assertEqual(G.mask_seed(2026, 'a1'), G.mask_seed(2026, 'a1'))
        self.assertNotEqual(G.mask_seed(2026, 'a1'), G.mask_seed(2026, 'b2'))
        self.assertNotEqual(G.mask_seed(2026, 'a1'), G.mask_seed(2027, 'a1'))
        self.assertLess(G.mask_seed(2026, 'a1'), 2**63)


class PromptTest(unittest.TestCase):
    def test_todays_prompt_is_the_168_text(self):
        self.assertIs(G.PROMPTS['inplace_header'].build, D.in_place_text)
        self.assertEqual(G.PROMPTS['inplace_header'].name, 'Header only, values filled in place')

    def test_the_setting_is_one_fixed_sampler_setting(self):
        self.assertEqual(G.SETTING, {'steps': 128, 'temperature': 1.0, 'seed': 2026, 'scope': 'slots',
                                     'precision': 'native', 'batch': 1})
        self.assertEqual(G.SMALL_FOUR, ('diffugpt_s', 'diffugpt_m', 'qwen25c', 'qwen3'))


class PasteRowTest(unittest.TestCase):
    def test_a_uniform_legal_paste_reads_into_six_slots(self):
        paste = '\n\n'.join(f'Mon{i} @ Item{i}\nAbility: Ab\nLevel: 50\nEVs: 13 HP / 12 Atk\nLonely Nature\n- A\n- B'
                            for i in range(6))
        team = G.team_from_paste_row({'team': paste})
        self.assertEqual(len(team), 6)
        self.assertEqual(team[0]['moves'], ['A', 'B', '', ''])
        self.assertEqual(team[0]['statPoints']['HP'], 13)
        self.assertEqual(team[0]['nature'], 'Lonely')
        self.assertEqual(blank_slot(1)['level'], team[0]['level'])


if __name__ == '__main__':
    unittest.main()
