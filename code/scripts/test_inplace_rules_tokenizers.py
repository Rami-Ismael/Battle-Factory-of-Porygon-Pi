"""Tokenizer parity check for #445: the filler line has the exact same token count as the rules line in every
small-four tokenizer, both alone and added to every mask of the held-out grid.

Loads the four pinned Hugging Face tokenizers and the held-out grid, so it is not in ci_tests.txt; run locally with
venv-dlm:
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python -m unittest discover -s code/scripts -p test_inplace_rules_tokenizers.py
"""
import unittest

import count_masked_diffusion_text as C
import inplace_prompt_grid as G
import masked_diffusion_text as D


class SmallFourTokenizerParityTest(unittest.TestCase):
    def test_filler_line_matches_rules_line_token_count_in_every_small_four_tokenizer(self):
        tasks = G.grid_tasks()
        rules_prompt, filler_prompt = G.PROMPTS['inplace_rules'], G.PROMPTS['inplace_filler']
        for key in G.SMALL_FOUR:
            with self.subTest(candidate=key):
                candidate = C.candidate_by_key(key)
                tokenizer = C.load_tokenizer(candidate)
                start_id = G.start_token_id(tokenizer, candidate.sampler)
                rules_cost = G.line_tokens(tasks, rules_prompt, D.RULES_LINE, tokenizer, candidate.mask_id, start_id)
                filler_cost = G.line_tokens(tasks, filler_prompt, D.FILLER_LINE, tokenizer, candidate.mask_id, start_id)
                self.assertEqual(filler_cost['line_alone'], rules_cost['line_alone'])
                self.assertEqual(filler_cost['added_min'], rules_cost['added_min'])
                self.assertEqual(filler_cost['added_max'], rules_cost['added_max'])
                self.assertEqual(rules_cost['added_min'], rules_cost['added_max'])
                self.assertEqual(filler_cost['added_min'], filler_cost['added_max'])
                self.assertEqual(rules_cost['added_min'], rules_cost['line_alone'])
                for task in tasks:
                    r_canvas = G.canvas(task, rules_prompt, tokenizer, candidate.mask_id, start_id)
                    f_canvas = G.canvas(task, filler_prompt, tokenizer, candidate.mask_id, start_id)
                    self.assertEqual(len(f_canvas['ids']), len(r_canvas['ids']), task['id'])
                    self.assertEqual(f_canvas['slots'], r_canvas['slots'], task['id'])


if __name__ == '__main__':
    unittest.main()
