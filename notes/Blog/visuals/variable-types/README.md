# The anatomy of a Pokémon build

Open `index.html` to explore or play the approximately 20-second guided story. It runs locally without packages, fonts, or network requests. Publish `index.html`, `pelipper.png`, and `torkoal.png` together.

The three examples distinguish unordered categories, bounded integer allocations, and conditional availability. Conditional is a property of the search space, not a separate numeric data type. The displayed vector is deliberately a partial build. This illustration does not claim to measure ruggedness, interaction strength, or optimization difficulty.

## Add to a blog

Copy the three files to your site's public assets directory, for example `/visuals/variable-types/`, then embed:

```html
<iframe
  src="/visuals/variable-types/index.html"
  title="Interactive Pokémon team-building variable types"
  loading="lazy"
  style="display:block;width:100%;height:1040px;border:0;border-radius:12px;"
></iframe>
```

The generous iframe height accommodates narrow screens. Adjust it for your blog column, or let readers open the visual directly. `preview.png` is a static fallback; it does not contain the animation. The live page has no autoplay and pauses its story when hidden. Any manual interaction stops the story; Replay begins again.

## Design and motion

An editorial specimen layout: warm paper, serif headings, dark green ink, orange focus accents, and the same Pokémon build throughout the examples. Intended for occasional educational interaction; motion shows state changes and a point moving between allocations.

Native HTML, CSS, and Web Animations API. Transform/opacity only: 240 ms content changes with `cubic-bezier(.23,1,.32,1)`, 440 ms point transfers with `cubic-bezier(.77,0,.175,1)`, 140 ms press feedback. Keyboard actions and reduced-motion preferences update instantly. Content is usable without playing the story.

## Data and artwork

- Ability options: [Pokémon Showdown Pokédex](https://github.com/smogon/pokemon-showdown/blob/master/data/pokedex.ts).
- Champions stat allocations: [validator](https://github.com/smogon/pokemon-showdown/blob/master/sim/team-validator.ts), [format limits](https://github.com/smogon/pokemon-showdown/blob/master/sim/dex-formats.ts), and [Champions data](https://github.com/smogon/pokemon-showdown/tree/master/data/mods/champions). The example assigns 32 HP, 23 Special Attack, and 11 Speed, then transfers whole points while preserving the total of 66. Other allocations remain zero.
- Pixel sprites reused from this project's existing `search-loop/sprites` assets. Pokémon characters and artwork belong to their respective rights holders; they are not original illustrations created for this figure.

## Local verification

`check.cjs` checks selection updates, stat limits, dependent abilities, reduced motion, interruption, the full story, and horizontal overflow across 320–1120 px. Its Playwright require path points to the local Codex runtime; change that path if running elsewhere. Preview PNG files are captured by this check.
