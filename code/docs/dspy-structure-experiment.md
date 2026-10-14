# DSPy structure experiment

Does packaging the masked-completion request as typed DSPy fields, with a typed
answer, change how often answers are legal and how well the legal teams play?

![Masked input, completion and evaluation, animated](../../blog/dspy-structure/mask-fill-battle.gif)

[Interactive version](../../blog/dspy-structure/index.html). It shows a real starter and mask, a recorded unreadable baseline answer, and a recorded legal answer with its 50-battle panel.

## Status, 14 October 2026

| Part | State |
|---|---|
| Phase one harness (structure only, no optimization) | Built and tested offline |
| Phase one paid run and battles | **Not run.** Needs the laptop runtime |
| Phase two harness (DSPy optimization) | Built and tested offline |
| Phase two run | Not run; follows phase one |
| Headroom from published data | Measured, below |

There are no win-rate results yet. Nothing below stands in for them. The run needs `OPENROUTER_API_KEY`, network access to `openrouter.ai`, the pinned Showdown validator, the behaviour-cloned battle checkpoint and the finished baseline folders in `code/results/`. All of these live on the author's laptop.

## What stays fixed and what changes

The comparison mirrors two finished baseline studies run for run: Ling 3.0 Flash with reasoning off and on (`reasoning-expanded-20261012`), and Mercury 2.5 with reasoning off, low, medium and high (`mercury-reasoning-20261013`). Both use the same 20 random legal starters × 7 masks (largest size per mask family) × 3 repeats = 420 requests per setting.

| Kept identical | Changed |
|---|---|
| Model, provider route, reasoning setting, temperature 0.7, output cap | How the request is packaged |
| Starters, masks, repeats, frozen opponent pool | How the reply is parsed |
| Instruction text, apart from the reply-format sentence | |
| Every input value: format, 49 opponent pastes, masked team, requested paths | |
| Completion checks (`apply_completion`) and the Showdown validator | |
| 50-battle panel, opponent schedule, behaviour-cloned policy on both sides | |

**Request.** `dspy_structure.py` turns each key of the baseline's JSON user message into a named, typed DSPy input field. The answer becomes one output field, `fills: list[Fill]`, and DSPy shows its JSON schema to the model. A `Fill` is a path plus a string, a Stat Point spread or a whole Pokémon. The instructions are the baseline system text with only these reply-format sentences removed (DSPy's layout replaces them):

- `Return only JSON: {"fills":[{"path":[0,"item"],"value":"..."}]}.` and `No prose or Markdown.` (original and regulation-reference prompts)
- `Return only strict JSON …` and `Use double-quoted keys and strings. No comments, …` (grounded prompt)

`check_information` re-reads every input field from the built request and fails unless it equals the baseline value. Preparation runs it on every task.

**Reply.** Two arms, both from DSPy 3.4.1:

| Arm | DSPy adapter | Reply layout |
|---|---|---|
| DSPy field-marker format | `ChatAdapter`, JSON fallback off | `[[ ## fills ## ]]` … `[[ ## completed ## ]]` |
| DSPy JSON-object format | `JSONAdapter`, no `response_format` | `{"fills": […]}` |

The typed parse is strict: `"12"` is not turned into `12`, and unknown keys are refused. DSPy does repair broken JSON (a trailing comma, say). That is part of the treatment, so each record stores `strict_json`, whether the model's own text parsed as written. The report counts answers that were legal only after this repair. No DSPy LM call, retry, fallback call or optimizer runs in phase one. Requests still go through `openrouter_generation.py`, with request hashes, pending markers and saved raw replies.

An optional third arm, `--resample-baseline`, sends the unchanged baseline request again alongside the structured ones. Its request hashes equal the originals. It separates the structure effect from provider drift since 12–13 October.

## Headroom in the published data

A reply format can only fix answers that failed on format. The Mercury 2.5 baseline rows in `blog/mask-lab/data.json` classify as follows (`dspy_structure_experiment.py headroom`):

| Mercury 2.5 reasoning | Legal | Unreadable answer | Wrong answer shape | Stat Points out of range | Rule violation | Legality if every format failure were fixed |
|---|---:|---:|---:|---:|---:|---:|
| off | 119 / 420 (28.3%) | 72 | 48 | 18 | 163 | at most 56.9% |
| low | 127 / 420 (30.2%) | 37 | 49 | 22 | 185 | at most 50.7% |
| medium | 141 / 420 (33.6%) | 42 | 16 | 25 | 192 | at most 47.4% |
| high | 155 / 420 (36.9%) | 20 | 1 | 36 | 206 | at most 41.9% |

Unreadable means the JSON did not parse. Wrong answer shape means a wrong fill count, or a duplicate or unmasked path. Not shown: 5 replies with no answer and 1 cut off. These are upper bounds: a fixed format still has to pass the rules. Most rejections are rule violations, which structure alone does not address. The headroom shrinks as reasoning effort grows, so any legality gain should be largest with reasoning off. The Ling studies' published summaries do not keep failure messages.

## Analysis, fixed before the run

Pairs match on model, reasoning setting, mask and repeat. Each difference is structured minus baseline.

- **Legality change**: all returned answers.
- **Win-rate change when both legal**: pairs where both teams passed and were scored. This is the win-rate comparison asked for, but it is conditional: a format that rescues hard masks changes which pairs are compared.
- **Wins delivered per request**: an illegal answer delivers no team and counts 0 wins. It is not counted as a battle loss. It is the one unconditional measure.

Intervals are 95% starter-cluster bootstraps with equal starter weights (`reasoning_experiment.cluster_interval`, 4,000 draws). Each interval comes with a two-sided starter-level sign-flip p-value. P-values are Holm-adjusted across all 12 model × reasoning × format cells. Verdicts: **improved** or **regressed** when the interval excludes zero; **no change beyond ±1 pp** (win rate) or **±5 pp** (legality) when the interval lies inside that margin; otherwise **unclear**. Pooled rows give each format's effect over every setting.

Expected precision: in the baseline studies, the both-legal win-rate differences between reasoning settings had 95% intervals about ±0.6 to ±0.9 pp wide per setting, with 94–101 pairs. Legal-team win rates are only 2–9%, so a real effect smaller than about 1 pp per setting will likely read as unclear.

The report refuses to compare when the battle configuration differs from the baseline's: same policy, simulator and panel fingerprints.

## Run phase one

On the laptop runtime, after `pip install dspy==3.4.1`:

```sh
PY=~/.local/share/vgc-pilot-runtime/venv/bin/python
$PY code/scripts/dspy_structure_experiment.py prepare --output code/results/dspy-structure-20261015 \
    --baseline code/results/reasoning-expanded-20261012 code/results/mercury-reasoning-20261013 \
    --adapters chat json --resample-baseline        # no calls; prints planned calls
$PY code/scripts/dspy_structure_experiment.py run --output code/results/dspy-structure-20261015
$PY code/scripts/dspy_structure_experiment.py battle --output code/results/dspy-structure-20261015 --workers 6 --watch
$PY code/scripts/dspy_structure_experiment.py report --output code/results/dspy-structure-20261015
```

Calls: 420 × 6 settings × 2 formats = 5,040, plus 2,520 for the re-sampled baseline. At the baseline's recorded cost ($0.98 for one pass of all six settings), expect about $3. Each structured request is about 2,200–2,500 characters longer (field list and answer schema); the 49 opponent pastes still dominate its length. Battles: the baseline had 810 legal answers per pass, so about 40,500 battles per format before exact-panel cache reuse. Generation and battles resume after interruption, as in the reasoning experiment.

## Phase two: optimization

`dspy_optimize.py` optimizes the structured program, then tests it through the same phase-one path.

1. `prepare` samples fresh legal starters (default 12 for training, 8 for development). They exclude the 20 test starters and every pool team, and use the largest mask per family. The parent run fixes model, provider, reasoning, sampling settings, prompt version and pool.
2. `optimize` runs GEPA (default; budget `light`) or MIPROv2 through `dspy.LM` on OpenRouter, with the same settings. The metric runs the unchanged checks. It scores 0 for an illegal answer and 1 for a legal one, and returns the exact rejection message as feedback. With `--battles`, a legal answer scores 0.5 + 0.5 × its 50-battle win rate. The output is `program.json`: instructions and demonstrations, development score before and after, and task-model cost.
3. Test: `prepare --program program.json` on the 20 test starters, then `report --reference <phase-one folder>`. The report pairs the optimized program with the baseline and with the unoptimized structure.

```sh
$PY code/scripts/dspy_optimize.py prepare --parent code/results/mercury-reasoning-20261013/off/r0 --output code/results/dspy-optimize-20261016
$PY code/scripts/dspy_optimize.py optimize --output code/results/dspy-optimize-20261016 --optimizer gepa --budget light
$PY code/scripts/dspy_structure_experiment.py prepare --output code/results/dspy-optimized-20261016 \
    --baseline code/results/mercury-reasoning-20261013 --adapters chat --program code/results/dspy-optimize-20261016/program.json
```

Optimization may rewrite the whole instruction text, including the regulation reference. Demonstrations repeat the 49-paste opponent list, so MIPROv2 uses none by default. The grounded prompt varies its input fields by task and is not supported.

## Limits

- The opponent pool is frozen and visible in every prompt. This measures in-pool performance, not held-out play or today's Regulation M-C meta.
- Showdown's RNG is not seeded. Repeats are not independent starters, and 20 starters bound the precision.
- The baseline was sampled on 12–13 October. Without the re-sampled arm, a difference mixes structure and provider drift.
- A shared failure classifier groups validator messages by text prefix. Ambiguous Stat Point failures (out of range, or a string instead of a number) stay in their own class.

## Files and checks

`code/scripts/dspy_structure.py`, `dspy_structure_experiment.py`, `dspy_optimize.py`, and two hooks in `llm_baseline.py` (`completion_payload` and `validate_record` use DSPy only when a manifest declares `structure`). The baseline request bytes are unchanged.

```sh
python -m unittest discover -s code/scripts -p 'test_dspy_*.py'   # 17 tests; no API key or simulator
```
