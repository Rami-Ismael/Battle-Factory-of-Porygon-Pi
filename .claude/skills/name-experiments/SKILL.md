---
name: name-experiments
description: "Name experiments, arms and runs, and shape the lists they appear in, in Rami's Pokémon counter-search project. Use when pre-registering an experiment or naming its arms; when writing a results table, figure legend or checkbox list of arms; or when a blog section, figure or reply quotes the arms of an experiment that already ran."
---

# Name experiments

Every experiment and arm carries two names:

- a **reader name** in plain English, shown in prose, tables, legends, figures and replies;
- a **code ID**, a short `snake_case` tag kept in code, config, filenames, JSON keys and checkpoints.

`AGENTS.md` ("Reader-facing wording") decides what a name must mean. This skill decides its form. The form follows two sources: the ReMDM paper's naming (one base method plus the one property that changes: ReMDM-cap, ReMDM-conf, ReMDM-loop), and FS Data Lab's "Five Principles for Generating Readable AI Text in Bulk" (readers see only the first two words of each list item).

## Name one arm

1. Read the arm's definition in its pre-registration doc or script. List every setting that differs from the control.
2. Write the reader name with the **difference first**: the differing setting, then the shared base, then the remaining settings. "Least-likely 10% remasked, 48 passes." Name the control by what it lacks: "No remasking."
3. Write the code ID as the base plus the changed property, ReMDM style: `remask_conf_10`, `remask_rand_10`. An ID already in code or results stays as it is.
4. In the pre-registration arm list, write the reader name first and the code ID in code formatting beside it.

Done when each arm's first two words name the setting that sets it apart from its siblings.

## Shape a list of arms

Apply all five rules to every results table, figure legend, checkbox list and pre-registration arm list:

1. **Difference first.** The first two words of each item tell it apart from the others. Words every item shares move to the column header or caption.
2. **One structure.** Every item puts its parts in the same order and the same grammatical form, such as "<selection rule> <amount> remasked, <passes> passes".
3. **One word per concept.** Pick one word ("remasked" or "redrawn", "pass" or "step") and use it in every item, matching the word the pre-registration doc uses.
4. **Variation carries meaning.** Items differ in length, wording or order only where their settings differ. A setting every item shares appears once, in the caption.
5. **Verdict verbs.** When the list reports results, the verb states the measured direction: "loses", "matches", "collapses diversity".

Done when every item passes all five rules and a reader can tell the items apart from their first two words alone.

## Rename an experiment that already ran

When reader-facing text quotes an earlier experiment (a blog section, a figure, a reply), rename its arms there with the steps above. Results files, JSON keys, scripts, checkpoints and the original docs keep their names. Where traceability matters, put the code ID in code formatting beside the reader name on first mention.

## Worked example

The remasking runs of 2026-10-30 (`remask_ask_gap.json`, `remask_why.json`), as one list:

| Code ID | Reader name |
| --- | --- |
| `plain` | No remasking |
| `remask05` | Least-likely 5% remasked, 48 passes |
| `remask10` | Least-likely 10% remasked, 48 passes |
| `remask20` | Least-likely 20% remasked, 48 passes |
| `remask10x2` | Least-likely 10% remasked, 96 passes |
| `random10` | Random 10% remasked, 48 passes |

The selection rule decided the result (least-likely selection collapsed diversity, random selection matched no remasking), so it leads every name.
