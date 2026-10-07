# Random noise to Pokémon team archetypes

Built 2026-09-28. A dark, Helbling-style 2D animation: a cloud of dots, one per team sampled from the masked diffusion checkpoint `temperature_p0.pt`, drifts into archetype clusters as the model reveals fields. Plain HTML, JavaScript (`<canvas>`) and Tailwind CSS (the `@tailwindcss/browser@4` build from jsDelivr, the only network request). Sprites are embedded.

## How a partly masked team gets a 2D position

For each team and each decode step *s* (0–48), the state is the team with the first *s* fields in dependency order revealed and the rest masked. `record-archetypes.py` resumes the same constrained sampler loop from that state and draws 24 completions (512 for the fully masked state, which every team shares). Each completion is labelled and projected:

- **Position:** the mean projected position of the completions.
- **Spread:** a fixed Gaussian direction per team, scaled by the completions' standard deviation. So the starting cloud's width is the model's real uncertainty, and it shrinks to zero at the finished team. The direction is decorative. The scale is not.
- **Colour:** the share of completions in each archetype. Grey means undecided.

Projection: shrinkage linear discriminant analysis (λ = 0.05, all three axes) on the species/ability/item/move multi-hot vectors of 1,500 separate model samples, labelled by the archetype rule. The recorder stores each state's 3D mean and covariance. `build.py` views them through one fixed linear 2D map, fitted so the archetype centres sit left (Trick Room), top (Weather), right (Tailwind) and bottom (Other). Because the map is linear, means and spreads carry over exactly. The first two discriminant axes alone put Tailwind and Other on top of each other.

Archetype rule, first match wins: Trick Room user; weather setter (Drizzle, Drought, Sand Stream, Snow Warning, or Mega Charizard Y / Tyranitar / Froslass / Abomasnow, checked against the bundled Showdown dex); Tailwind user; Other. Natures don't affect the label, so completions stop after the moves (step 42).

Hyper offense was dropped as a class. Defined as "≥ 4 members with no Protect or support move", it matched 1 of 692 corpus teams and 0 of 256 model samples.

Followed teams: the team nearest each archetype centroid, plus two random teams (seeded).

## Files

- `record-archetypes.py`: records `trace.json`, about 12 min on 11 spawned CPU processes (a forked pool deadlocks in PyTorch). Sizes can be overridden with `ARCH_SIZES=N,K,K0,REF`.
- `build.py`: embeds `trace.json` and the needed sprites (from `../search-loop/sprites`, Megas without a sprite fall back to the base forme) into `archetype-diffusion.html`.
- `template.html`: the page source. Edit this, not the built file.

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python record-archetypes.py
python3 build.py
```

No battles were run. The figure shows what the model generates, not how well those teams play. Stat Points are not sampled.
