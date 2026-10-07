# Rain, Electro Shot, and Weather Ball

Blog figure for a third-order interaction, written for readers new to Pokémon. Open `index.html` and step through it with Back / Next, the dots, or the ← → keys. Seven steps: the target and the doubles rule; a battle without Pelipper; a battle that leads Pelipper + Charizard; turn 1, where Pelipper switches out for Archaludon and the rain stays; turn 2, where the target faints; a check of every pair; and the interaction matrix.

This illustrates a mechanism. It is not a simulation and not a measured effect on win rate.

## Embed

`index.html` is self-contained: sprites are inlined, and there are no libraries, external fonts, or network calls. An iframe isolates the styles from the blog:

```html
<iframe src="/visuals/third-order-weather/index.html"
  title="A third-order interaction: rain, Electro Shot and Weather Ball"
  loading="lazy"
  style="display:block;width:100%;height:1050px;border:0;">
</iframe>
```

The last step (the matrix) is the tallest. Adjust the height to your column width; on phones the layout stacks.

## Interpretation

- "Power" is move power only. Electro Shot is 130. Weather Ball is 50 as a Normal move, or 100 as a Water move in rain, boosted ×1.5 by the rain, giving 150. Stats, STAB, type matchups and Pelipper's own attacks are left out.
- The 250-HP target is made up, so the maths stays simple.
- The score is 1 if a team's attacks reach 250 in one turn, and 0 otherwise. No single Pokémon and no pair reaches it (best pair 150); the trio does (280). Every pairwise cell is 0, and the third-order term is +1.
- The third-order effect comes from the threshold. Power itself adds up pair by pair: the trio's 280 is exactly what singles plus pairs predict. Only the yes/no "does it faint" outcome needs all three.
- Only two Pokémon are ever on the field together. Pelipper's rain outlasts its switch-out, which is how all three contribute to one turn.

## Sources

- Pokémon Showdown's [moves](https://github.com/smogon/pokemon-showdown/blob/master/data/moves.ts) (Electro Shot 130, Weather Ball 50 doubling in weather), [abilities](https://github.com/smogon/pokemon-showdown/blob/master/data/abilities.ts) (Drizzle), and [weather conditions](https://github.com/smogon/pokemon-showdown/blob/master/data/conditions.ts) (rain ×1.5 to Water).
- Charizard's Weather Ball is checked against the Champions learnset in Showdown's `data/mods/champions/learnsets.ts`.
- Sprites are reused from `search-loop/sprites`. Pokémon characters and artwork belong to their respective rights holders.

## Motion

Every animation explains something:
- Pokémon glide between "On the field" and "Waiting in the back" (layout animation), so a switch reads as a switch.
- Rain is a looping overlay while it is up.
- Electro Shot's chip pulses while it is charging.
- The HP bar drains, with a number ticker and a shake on a hit, and "Fainted!" pops in.
- The pair cards come in staggered and drain their own bars.
- The matrix cells come in staggered.

Easing is `cubic-bezier(.23,1,.32,1)`. Reduced-motion mode removes all of it and updates instantly.

Prototype history, kept as a record of what was tried: `../mega-slot-prototype/`.
