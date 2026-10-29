# Decision-model limits: options per question and context (checked 2026-10-26)

Each fact comes from the maker's own page. Quotes are short excerpts. This research fills the missing cells of Table 2 on the Jev Picks a Team page.

| Model | Options per choice question | Context limit (tokens) | Source |
|---|---|---|---|
| Jev (TypeSafe) | 255 | 32k per question, 64k per request, 262k with the long-context header | [Choice docs](https://docs.typesafe.ai/primitives/choice.md), [models](https://docs.typesafe.ai/models) |
| Liquid d1, d1-ultra (hosted) | not stated | not stated by Liquid: the d1 page says a per-question limit exists but gives no number; Vercel's listing says 65,536 | [d1 docs](https://docs.liquid.ai/lfm/models/d1) |
| Liquid d1-3B (open) | not stated | 32,768: "Context length: 32,768 tokens" | [d1-3b docs](https://docs.liquid.ai/lfm/models/d1-3b), [model card](https://huggingface.co/LiquidAI/d1-3B) |
| Upstage Solar Decide | 2 to 26: "A `choice` question takes 2 to 26 options"; more returns HTTP 422 | 512K window: "With a 512K context window, an entire document can serve as the state"; whether that covers the whole request is not stated | [API page](https://console.upstage.ai/api/systemone) |
| Torchcast Decision 27B | no hard cap: "a choice with more than 26 options takes extra rounds" | 262,144: "Context up to 262,144 tokens (256K)"; config `max_position_embeddings` 262144 | [model card](https://huggingface.co/torchcast-ai/torchcast-decision-27b) |
| CLM-8B | not stated | "States longer than 2048 tokens are truncated" by default; raise `--max-model-len` and `--max-tokens` together | [README](https://github.com/Contrastive-LM/CLM) |
| Perplexity pplx-decider (v1.2; v1.1 names route to it) | "A question accepts 1 to 255 options." | 256K: "Under 256K for the state, its images, and each question." The v1.1 open checkpoint's own inference code caps a decision at 8,192 tokens | [Decisions quickstart](https://docs.perplexity.ai/docs/decisions/quickstart.md), [v1.1 card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| Inception Mercury Decide | 1 to 255: "A map of 1–255 option names to descriptions" | not stated: the models table shows a dash | [decisions docs](https://docs.inceptionlabs.ai/capabilities/decisions.md), [models](https://docs.inceptionlabs.ai/get-started/models.md) |
| OpenAI Decisions API (gpt-6-luna) | not stated; 1 to 400 questions per request | not stated; 272K on the pricing page is a price tier, not a limit | [guide](https://developers.openai.com/api/docs/guides/decisions), [pricing](https://developers.openai.com/api/docs/pricing) |
| Patronus GLIDER | one input per call (rubric judge) | 8,192, tested to 12,000 | [model card](https://huggingface.co/PatronusAI/glider) |

Corrections to the earlier table: Perplexity's decision model and Inception's Mercury Decide both exist on their makers' own docs. The earlier "list only" came from OpenRouter pages that returned 404. Both use their own Decisions endpoints, and nobody compared their request shape with Jev's.
