---
created_at: 2026-11-02
updated_at: 2026-11-02
tags:
  - research
---

# Pretrained encoder for opponent prompts: findings (2026-11-02)

## Recommendation

**Use UniXcoder-base, a frozen encoder trained on code, on each Pokémon's JSON separately. Its score is the highest of the five encoders but only detectably above Qwen2.5-0.5B-Instruct, and it ties a plain count of shared species, so this test shows no added value from a pretrained encoder.**

- **Pick:** UniXcoder-base, one Pokémon at a time. Spearman correlation 0.258, 95% interval [0.173, 0.356]. UniXcoder minus the shared-species count is [-0.062, +0.044].
- **Runner-up:** BGE-small-en-v1.5, a small sentence-embedding model, one Pokémon at a time. Spearman correlation 0.225, [0.140, 0.319], at half UniXcoder's output size and 57% of its peak memory. UniXcoder minus BGE-small is [-0.028, +0.094], so choosing BGE-small costs nothing detectable.
- **Encode per Pokémon.** Every model scored lower on a whole team as one text.

Matchup similarity asks whether two meta teams get beaten by the same teams. It uses row teams, the teams with results against both. Every encoder's interval sits above zero, while a random-vector control's includes zero. None beats counting shared species. The target is stable, because two random halves of the row teams agree at 0.91, so the low scores don't come from noise in the target.

## Results

Spearman correlation between encoder cosine similarity and matchup similarity, over 1,176 pairs of the 49 top-50 meta teams. Intervals are 95% percentile bootstrap intervals: 1,000 times, resample the 49 teams and score the pairs among the distinct teams drawn.

| Method | Spearman | 95% interval | Row minus shared species | Seconds per team | Peak memory (MB) |
|---|---|---|---|---|---|
| Baseline: shared species count | 0.266 | [0.182, 0.355] | n/a | n/a | n/a |
| Baseline: shared item, ability, Stat Alignment or move count | 0.259 | [0.171, 0.354] | n/a | n/a | n/a |
| UniXcoder-base, JSON per Pokémon | 0.258 | [0.173, 0.356] | [-0.062, +0.044] | 0.181 | 1,589 |
| all-MiniLM-L6-v2, text sentence per Pokémon | 0.227 | [0.143, 0.327] | [-0.087, +0.008] | 0.036 | 796 |
| BGE-small-en-v1.5, JSON per Pokémon | 0.225 | [0.140, 0.319] | [-0.096, +0.012] | 0.155 | 911 |
| BGE-small-en-v1.5, text sentence per Pokémon | 0.224 | [0.141, 0.316] | [-0.094, +0.010] | 0.090 | 911 |
| UniXcoder-base, text sentence per Pokémon | 0.219 | [0.141, 0.312] | [-0.107, +0.021] | 0.265 | 1,589 |
| all-MiniLM-L6-v2, JSON per Pokémon | 0.204 | [0.129, 0.290] | [-0.129, +0.001] | 0.063 | 796 |
| SmolLM2-135M-Instruct, JSON per Pokémon | 0.195 | [0.115, 0.276] | [-0.145, -0.001] | 0.358 | 1,621 |
| UniXcoder-base, JSON whole team | 0.178 | [0.100, 0.275] | [-0.156, -0.024] | 0.280 | 1,589 |
| Qwen2.5-0.5B-Instruct, JSON per Pokémon | 0.154 | [0.070, 0.260] | [-0.196, -0.025] | 0.944 | 2,086 |
| BGE-small-en-v1.5, JSON whole team (truncated) | 0.154 | [0.083, 0.241] | [-0.228, +0.009] | 0.144 | 911 |
| SmolLM2-135M-Instruct, JSON whole team | 0.146 | [0.067, 0.226] | [-0.208, -0.026] | 0.363 | 1,621 |
| Qwen2.5-0.5B-Instruct, text sentence per Pokémon | 0.126 | [0.044, 0.224] | [-0.234, -0.040] | 1.045 | 2,086 |
| SmolLM2-135M-Instruct, text sentence per Pokémon | 0.119 | [0.056, 0.192] | [-0.234, -0.064] | 0.349 | 1,621 |
| all-MiniLM-L6-v2, JSON whole team (truncated) | 0.115 | [0.041, 0.191] | [-0.243, -0.065] | 0.024 | 796 |
| Qwen2.5-0.5B-Instruct, JSON whole team | 0.113 | [0.042, 0.199] | [-0.237, -0.066] | 0.829 | 2,086 |
| Control: random 384-dimension vectors | 0.017 | [-0.052, 0.096] | n/a | n/a | n/a |

Whole-team JSON runs past the input limit of BGE-small and all-MiniLM on every team (longest 692 tokens). Per-Pokémon JSON never truncates on any model (longest 121 tokens). Differences are the row's score minus the shared-species score, on the same bootstrap samples. Peak memory is one process's maximum resident memory across all its input forms.

## Correction to the first version of this file

The first version said every method scored near zero. That came from a bug: the target kept only row teams that are themselves top-50 teams, about 47 values per column, instead of all 188,114 row teams. It also fed models reworded sentences, not the JSON the prompt will use. This version fixes both, and adds the paired differences and split-half check.

## How the test works

1. **Teams.** The 49 top-50 meta teams, read-only from `~/vgc-data/matchup_regmb.sqlite`. No results for ranks 51 to 100 were read: a held-out team has no stored battle in this database under any battle configuration.
2. **Target.** Battle configuration `policy_id` 1 (behaviour-cloning policy on both sides, `bc_100.zip`), the largest in the database, with 188,114 row teams, each with a result against all 49. Subtract each row team's mean win rate, then correlate two opponents' columns. A win rate is wins ÷ battles. The target is reliable: two random halves of the row teams agree at Spearman 0.91, 0.92 and 0.91 across three splits, computed with each column's mean subtracted.
3. **Encoders.** Frozen, on CPU with 4 threads. BGE-small uses CLS-token pooling, as its configuration sets. The other four use mean pooling over tokens. Per-Pokémon vectors were averaged into one team vector for the comparison.

## What this test cannot show

- **It scores cosine similarity, not cross attention.** A trained attention layer can learn from vectors in ways cosine similarity can't. A tie here doesn't show the encoder is useless in the model.
- **It's a proxy.** Similar opponents needing similar counters is an assumption this test relies on. It doesn't show that generated teams beat anyone.
- **Only 49 teams and one battle configuration.** Other configurations and larger team sets weren't tested.
- **One pooling per model.** Other pooling might change the order, especially for the language models.

## Sources

| Model | Repository | Commit | Licence | Output size | Maximum positions |
|---|---|---|---|---|---|
| UniXcoder-base | [microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base) | 5604afdc9 | Apache 2.0 | 768 | 1,026 |
| BGE-small-en-v1.5 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 5c38ec7c4 | MIT | 384 | 512 |
| all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 1110a243f | Apache 2.0 | 384 | 512 (library limit 256) |
| Qwen2.5-0.5B-Instruct | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 7ae557604 | Apache 2.0 | 896 | 32,768 |
| SmolLM2-135M-Instruct | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 12fd25f77 | Apache 2.0 | 576 | 8,192 |

All five load with `trust_remote_code=False`. Scripts and raw outputs are scratch files: `/tmp/encoder-probe/scripts/probe2.py`, `/tmp/verify_sim.py`, `/tmp/split_half.py` and `/tmp/encoder-probe/out2/`.
