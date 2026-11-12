# Storyboard: Departures: where does our 84-field discrete diffusion code depart from MD4 and LLaDA?

Figure sentence: Seven of 15 parts follow MD4 or LLaDA; the other eight are ours alone.

## 1. Parts each paper matches

- Sentence: Five of our 15 parts match MD4 and three match LLaDA. One, the loss weight, matches both, so seven match at least one.
- Measured: Counts from the matrix in the issue's last comment, after the second check.

## 2. Follows a paper, or ours alone

- Sentence: Seven parts do what a paper does; the other eight are ours alone.
- Measured: The division by 84 matches LLaDA's Algorithm 1, not MD4's.

## 3. What the second check changed

- Sentence: A second check changed five of 45 cells, and the matching count rose from 6 to 7 of 15.
- Measured: Each correction was confirmed at its quoted line in the paper's LaTeX source.

## 4. The three departures this map tests

- Sentence: Masking rate and decoding order are not run yet; the four rows written before any result test them.
- Expected: Linear, random order loses to Linear, species-first, with an error under 2.8 times as large.
- Surprise: A linear row that wins by 0.002 or more.
## Cut
- The 15 part names: the issue's matrix lists them.
- The per-field exponents: issue 293 covers them.
- Temperature and the sampler's t formula: no paper result depends on them.
