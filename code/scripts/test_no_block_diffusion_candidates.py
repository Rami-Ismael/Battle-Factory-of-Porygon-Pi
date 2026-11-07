"""Guard for docs/decisions/no-block-diffusion-models.md: no block diffusion model is a masked-model candidate."""
import re
import unittest

import count_masked_diffusion_text as C

BLOCK_DIFFUSION = re.compile(r'bd3lm|fast[_-]?dllm|sdar|block', re.I)


def block_diffusion_candidates(candidates):
    return [c.repo for c in candidates if BLOCK_DIFFUSION.search(c.repo) or BLOCK_DIFFUSION.search(c.label)]


class NoBlockDiffusionTest(unittest.TestCase):
    def test_no_candidate_is_a_block_diffusion_model(self):
        self.assertEqual(block_diffusion_candidates(C.CANDIDATES), [],
                         'block diffusion is dropped: see code/docs/decisions/no-block-diffusion-models.md')

    def test_the_guard_catches_fast_dllm_v2(self):
        fast = C.Candidate('fast', 'Fast-dLLM v2 1.5B', 'Efficient-Large-Model/Fast_dLLM_v2_1.5B', '0' * 40, 151665)
        self.assertEqual(block_diffusion_candidates([fast]), ['Efficient-Large-Model/Fast_dLLM_v2_1.5B'])


if __name__ == '__main__':
    unittest.main()
