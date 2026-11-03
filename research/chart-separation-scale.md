# What separation should our chart colors reach, and does Baldwin's Delta E of 11 compare with our 8?

Read and measured 2026-11-03. No recommendation.

**Baldwin's 11 is on a different scale from our 8, so the two don't compare directly.**

[Leonardo's compare tool](https://leonardocolor.io/tools.html) says the simulated colors "are then compared using the DeltaE 2000 color difference formula", and that "A color difference value of 11 or more is considered passing". Our 8 is OKLab distance ×100, and comes from the `dataviz` skill's validator.

I found no published threshold on an OKLab scale, and no conversion between the two.

## Measured on 400,000 random sRGB color pairs

| Starting from | Other scale's value (5th, 50th, 95th percentile) |
|---|---|
| CIEDE2000 close to 11 | OKLab ×100: 5.7, 9.9, 13.5 |
| OKLab ×100 close to 8 | CIEDE2000: 6.8, 10.9, 19.7 |
| OKLab ×100 close to 15 | CIEDE2000: 13.5, 21.8, 35.8 |

About half (49.6%) of pairs near our 8 reach Leonardo's 11. The middle values are close, but a single pair can land well away from them. These are random pairs without color-blindness simulation, not chart palettes.

## Our own pairs under red-green color blindness

| Pair | OKLab ×100 | CIEDE2000 |
|---|---|---|
| Red and amber, `dspy-structure` and Mask Lab | 4.3 | 5.7 |
| Green and orange, the meta posts | 8.5 | 16.1 |

Red and amber fail on both scales. Green and orange pass on both.

## Other sources

[Datawrapper](https://www.datawrapper.de/blog/colorblindness-part2) gives no number, and says "don't rely on these simulators too much".

Not read: Baldwin's Medium article returned 403, and the cols4all paper returned 301 then 404, so its thresholds aren't confirmed here. Whether Leonardo's palette control uses the same formula as its compare tool is also unconfirmed.

The measuring script is `scales.py`, run with scikit-image's `deltaE_ciede2000`.
