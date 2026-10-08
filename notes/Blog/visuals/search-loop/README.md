# Pokémon Team Search Loop

The opening figure now also runs the actual small diffusion checkpoint **directly in the browser**. Select **Generate a new team** to sample a fresh candidate, watch its actual intermediate states, and check it with the pinned browser-side format validator. **Show fixed example** preserves the earlier illustrative demo.

Proposed workflow for a technical blog, implemented as a responsive D3/SVG figure. It does not claim that the complete experimental pipeline exists or show measured results.

## Preview

Open `index.html` directly in a browser, or from this directory run:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Then visit http://127.0.0.1:8765. The local preview server is not publication.

## Embed

Copy this entire directory into your blog's static/public assets. Embed the HTML as an iframe; the accompanying D3 file must remain beside it.

```html
<iframe
  src="/visuals/search-loop/index.html"
  title="Pokémon Team Search Loop: proposed experimental workflow"
  style="width:100%;height:1500px;border:0"
  loading="lazy"
></iframe>
```

The figure reflows below 760px of available figure width. A mobile iframe needs around 3000px height to show the whole figure without internal scrolling; set a responsive height in the host CSS or allow the iframe to scroll. Height depends on host width and installed fonts. Keyboard navigation stays inside the figure until the reader Tabs past its final control. No parent-window messaging, analytics, build step, or network calls are used.

This is a standalone document, not an Astro/MDX HTML fragment. An iframe works in Astro/MDX too. Do not paste the full document into `HtmlEmbed`. For native integration, adapt to the template's `.ai/skills/create-html-embed/SKILL.md` and `directives.md` (root-scoped styles, instance-local mounting and marker IDs, palette/theme variables, and HTML controls). Those conventions were inspected; no template integration or publication was performed.

## Light and dark screens

The figure follows `prefers-color-scheme` automatically, including changes while open. Backgrounds, stage fills, labels, arrows, focus rings, and explanation panels all adapt. Print output uses the light palette.

If your blog uses its own theme switch, set the iframe URL to `index.html?theme=dark` or `index.html?theme=light` to match it; omit the parameter for automatic mode. A standalone iframe cannot read a parent page’s theme class. The same document also supports `data-theme="dark"` or `data-theme="light"` on its HTML element.

## Dependencies and controls

- Vendored D3 **7.9.0**, `d3.v7.9.0.min.js`, ISC license in `D3-LICENSE`. Keep all three files together for offline use. Source: https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js.
- Modern browser with SVG and ResizeObserver. System font stacks: Georgia headings; Avenir Next / Segoe UI / sans-serif labels.
- Click, tap, or focus any of 13 nodes to open its four-part explanation. Tab/Shift+Tab use document order; arrow keys move through workflow order; Home/End jump to the first/last node; Enter/Space open details; Escape or the close button dismisses details and keeps/restores node focus.
- No automatic motion; reduced-motion styles explicitly disable animations and transitions. A no-JavaScript text fallback and a dependency-load error message are included.

## Methodological assumptions still to resolve

1. Define the regulation/version, battle policies (including team preview), target-meta snapshot and weights, search opponent pool and distribution, and independently reserved held-out opponents before running. Broader performance means performance on that declared held-out pool; it is not a universal or ladder-strength guarantee.
2. Set the search budget, battles per team, selection rule for battle batches, retention/weighting rule for generator updates, finalist count and selection rule, and uncertainty-reporting method. No numeric settings in the notes are silently adopted as established defaults.
3. Decide whether to use a surrogate, its inputs/model, cold-start sampling rule, and refit schedule. Predictions help choose battles; they never become observed outcomes. Skip the whole dashed branch if no surrogate is used.
4. Fix a canonical team key that includes all battle-relevant fields for the chosen format. Log raw attempts and invalids, define duplicate denominators, retain repeats when measuring the generator, and track roster composition frequencies separately. The diagram assumes rejection of invalids; repair requires separate raw/repaired logs and a declared protocol.
5. Freeze finalists before final battles. Use fresh outcomes for target-meta evaluation and keep held-out opponents/results out of generator training, surrogate fitting, search selection, and tuning. Subsequent development using those outcomes requires a new untouched evaluation set for a new generalization claim.
6. Choose initial training sources, representation, diffusion settings, update/replay rules, and independent run seeds. The method does not guarantee diversity or superiority to an LLM or other proposer. Selected-candidate performance also does not estimate the unfiltered generator's average performance.

Terminology checked against `Problem Statement.md`, `Diffusion as candidate proposer in black-box optimization over structured inputs.md`, `Todo Section.md`, `Metrics for Diversity.md`, and `Matchup Matrix.md`. The notes include historical proposals and pilot measurements; none are presented here as measured evidence for this complete workflow. Unknown matchups remain unknown.

## Design reference and verification

Inspected the [Hugging Face report](https://huggingfacem4-vlm-data-autoresearch.hf.space/) and its [D3/SVG loop source](https://huggingface.co/spaces/HuggingFaceM4/vlm-data-autoresearch/blob/main/app/src/content/embeds/vlm-loop.html). This figure uses original implementation/content, with the reference's restrained editorial styling, directed paths, and explanatory interactions. Mobile uses vertical reflow instead of the reference's minimum-width scrolling diagram.

Verified in headless Chrome at 1200px, 390px, and 320px viewport widths: 13 node labels and four-part explanations, all 14 directed connections, no horizontal page overflow or overflowing node text, focus/click interactions for every node, Tab, arrow keys, Home/End, Space/Enter, Escape, close/focus restoration, touch, and reduced-motion preference. Final evaluation has only finalist → fresh battles → report edges and no edge back to search. Screenshots are included as previews; these are not battle results. Screen-reader announcements have not been manually audited with assistive technology.

## Caption

“Candidate teams are generated, checked, and evaluated in battles. Observed results guide subsequent search rounds. Finalists undergo separate evaluation to measure both counter-meta strength and performance against held-out opponents.”

## Interactive controls update — 2026-09-13

The figure now includes four descriptive controls: training-data mixture, guidance strength, field-value sampling temperature, and elite fraction. Hover, tap, click, or focus a control to highlight its affected steps and read an explanation. These do not run experiments or display simulated results. Hovering a workflow node highlights its adjacent connections, following the Hugging Face reference. Stage boundaries use dashed outlines.

The generator-update step is labeled “Select training teams”: pilot experiments use win-rate-weighted sampling, while retaining a hard top fraction is a proposed alternative. Field sampling temperature is distinct from the temperature used for weighted training-team selection. Held-out evaluation is identified as proposed.

Fresh browser verification passed in headless Chrome at 1440px, 1200px, 390px, and 320px: no horizontal overflow, clipped node labels, or JavaScript errors; all 13 node explanations and four controls work. Click, touch, focus, arrow navigation, Home/End, Space, Escape, and close-button focus restoration were checked. Highlights persist when resizing between desktop and mobile. Dark-mode active controls have legible contrast, and reduced-motion disables transitions. The 14 directed connections were audited; final evaluation has no feedback path into search. Desktop, mobile, interaction, and dark-mode preview screenshots were refreshed. Manual assistive-technology testing remains outside this check.

## Team reveal slider

The opening figure uses a native range input and D3-rendered HTML cards to reveal all six team slots together. Steps: masked → species → ability/item → moves → nature → copied Stat Points and level → recorded validation. Arrow keys scrub the focused slider; adjacent buttons move one step; Reset returns to empty. At the final step, readers can inspect, copy, or download the complete simulator-format team.

This is an illustrative reveal of a fixed corpus example, not a saved diffusion trajectory. It does not run generation or validation in the browser. The final badge reflects the successful validation captured in `team-example-validation.json`; the team content and source metadata are in `team-example.json` and `team-example.txt`. Missing IV and Tera fields are not invented. Champions Stat Points are displayed explicitly; the source text uses an EVs label for them.

The slider follows the scrubber and incremental-reveal interaction of the reference’s `banner.html`, inspected alongside `vlm-mixture-banner.html`. It uses local assets and theme variables. The larger combined artifact needs additional iframe height compared with the loop-only sizing examples above; use a scrollable iframe or measure the embedded document height in your host integration.

Slider verification passed in headless Chrome at 1440px, 390px, and 320px. All seven stages were checked forward and backward against the expected visible fields in six cards. Previous/next/reset, Home/End/arrow keys, endpoint button states, export visibility, clipboard copy, and file download passed. Page and card content do not overflow horizontally. Desktop light/dark screenshots were visually inspected. Existing loop keyboard/interactions and edge checks still pass with the new section present; no browser errors occurred. Preview artifacts include `preview-team-empty.png`, `preview-team-midpoint.png`, `preview-team-complete.png`, `preview-team-dark.png`, and full-page responsive previews.

## Pokémon sprites

The six reveal cards now use locally vendored 96×96 static front sprites from Pokémon Showdown’s `https://play.pokemonshowdown.com/sprites/gen5/` collection, downloaded on 2026-09-13. Individual URLs and SHA-256 hashes are recorded in `sprites/sources.json`. The directory name identifies the static sprite style; it includes later-generation Pokémon. Preserve the `sprites/` directory beside the HTML for offline use.

Mappings: Excadrill → `excadrill.png`; Tyranitar → `tyranitar.png`; Staraptor → `staraptor.png`; Basculegion → `basculegion.png` (base/male); Sinistcha → `sinistcha.png` (base form); Raichu → `raichu.png` (ordinary, not Alolan). Tyranitar and Staraptor retain their base team identities even though they hold Mega Stones.

Images are decorative alongside readable species names and have empty alternative text. Fixed image frames prevent layout changes when loading or scrubbing. Step 0 shows neutral question-mark masks; steps 1–6 show sprites; reverse scrubbing to 0 hides them again. These are reference assets, not AI-generated art. Sprites are sourced from Pokémon Showdown; Pokémon artwork and characters belong to their respective rights holders (Nintendo / Creatures / GAME FREAK), and the assets are not relicensed as project code.

Sprite verification passed in Chrome at 1440px, 390px, and 320px. All six local 96×96 images loaded and visually match the named base species, including male Basculegion. Sprite frames retain their dimensions when scrubbing 0→1→6→0; images hide at step 0 and reappear from step 1. No card/page overflow was found. Light, dark, and mobile screenshots were inspected; existing reveal-state and loop interaction checks still pass. Updated previews include `preview-sprites-1440.png`, `preview-sprites-390.png`, `preview-sprites-320.png`, and `preview-sprites-dark.png`.

## Browser inference — 2026-09-13

Serve the entire directory over HTTP using the preview command above. Opening the file directly retains the illustrative example, but browser model loading requires HTTP. No Python inference server, account, API key, or generation API is used: the static server only delivers files. Copy the full directory, including `model/`, `runtime/`, sprites, `sampler-worker.js`, and `team-reveal.js`, when embedding or sharing.

The model is the existing 2,122,751-parameter TeamDiffusionHPS initial checkpoint trained on 692 real teams. It is exported to ONNX (~8.7 MB decimal). Vendored ONNX Runtime Web 1.22.0 uses the WASM-only entry and one thread, so ordinary static hosting does not require cross-origin isolation headers. A dedicated Worker keeps the UI responsive. Model, manifest, runtime and validator total about 29 MB on first use; subsequent runs reuse the Worker session and browser cache. ORT files are from jsDelivr’s pinned 1.22.0 distribution; the MIT license is in `runtime/ORT-LICENSE`.

Sampling matches the project’s constrained dependency order: six species, abilities/items, four moves per slot, then natures. Each of 48 forward passes runs the actual checkpoint, constrains the current field’s probabilities, and samples a value. Condition 4 is the null token and temperature is fixed to 1; this checkpoint is not being instructed to beat a particular team. Stat Points are copied from same-species corpus examples afterward. The browser then applies the original pinned Showdown format validator. A valid team is not evidence of battle strength.

A click draws a fresh browser seed and attempts at most five candidates. Rejected candidates are reported; only a successful validator result receives the passed badge. The slider replays the last attempt’s actual saved states. **Download sampling trace** includes all attempted traces, rejection reasons, seeds, model hashes, and settings; **Download team** exports the final candidate, which may be invalid if all attempts failed. The JS seeded PRNG is Mulberry32, so its samples are not expected to match PyTorch’s multinomial RNG despite matching probabilities.

Stop terminates the inference Worker. The partial state remains clearly unvalidated; another click restarts inference. Errors never fall back to a fabricated generated team. The original fixed corpus example is available explicitly via its own button. Generated species use the locally vendored sprite mapping in `model/sprites.json`; missing artwork uses readable initials and the exact species name.

Numerical evidence: the export is compared with the original PyTorch checkpoint in `model/verification.json`; actual browser WASM logits and constraint masks are checked against fixtures in `model/browser-verification.json`. The pinned browser validator has 66 exact parity cases in its runtime verification artifacts. Full generated-team UI verification is recorded separately by the QA agent.
