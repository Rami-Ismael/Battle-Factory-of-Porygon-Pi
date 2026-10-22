# Example: Build and judge novel VGC ideas

[Open the Codex chat](codex://threads/01a130a4-9dc2-7fc1-9851-f948833fdf00)

This chat's opening request is a strong match for **principle-separate-before-serializing-shared-state**. It asks for parallel makers, independent verifiers, a judge, and one final `variants/DECISIONS.md` containing all stages' results.

## Why this opening prompt benefits

Your original prompt already gives makers separate variant folders. The remaining ambiguity is who writes the common decision document and who edits a winning variant when several animation makers or fix agents work on it. Different agents editing different parts of the same file can still overwrite each other's changes.

This is a risk in the prompt's ownership rules, not evidence that the chat actually lost work.

## Add this to that opening prompt

Apply the separate-before-serializing-shared-state principle throughout this task. Give each maker its own variant directory and each verifier or judge its own report file. For multiple animation attempts on one winner, give each maker a separate copy; do not let them edit the same winning files concurrently. Assign one integrator to update `variants/DECISIONS.md` and apply the chosen changes after the relevant workers finish. Run fixes that touch the same files sequentially, or isolate them and let the integrator merge them. Keep shared inputs read-only. If a shared write target truly cannot be separated, use an enforced single writer or suitable cross-process coordination, rather than telling agents to take turns. Preserve other chats' changes and existing outputs.

## What changes in practice

| Original ambiguity | With this instruction |
| --- | --- |
| Everyone may contribute directly to the decision document. | Workers publish separate reports; one integrator writes the document. |
| Four animation makers may edit the same winner. | Each explores its own copy; the integrator combines selected changes. |
| One agent per fix may imply overlapping edits. | Independent fixes can run in parallel; overlapping fixes are isolated or sequenced. |

The useful improvement is explicit ownership at the points where the parallel work comes together. You do not need more subagents just to apply this principle.
