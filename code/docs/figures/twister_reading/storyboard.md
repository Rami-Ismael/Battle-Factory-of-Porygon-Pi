# Storyboard: reading *Twist, Don't Tilt* against the ask grids

The frames for [the report page](index.html), in reading order. The figures were storyboarded on 2026-11-23, after the
results existed, so the "expected" lines record what the predictions file (SHA-256 `413666d4…`, 2026-11-15 18:07)
or the earlier sections led me to expect. The "surprise" lines name the result that would have changed the frame. One
of them happened: the rejection run (#358) won back about half of what reweighting predicted, and that result now has
its own frame (3g).

Every frame below feeds the section it sits in. No experiment was run only to fill a frame.

## Teaser

- **Sentence:** species-first order draws the model's own teams. Random order needs the lookahead, which bends teams, and
  the bent teams lose. The legal Pokémon construction tables cut more, but that cut changes nothing.
- **Expected:** one picture with three lanes (tables, lookahead, validator) and the measured numbers under each.
- **Surprise:** if the tables' cut tracked win rate as well as the lookahead's, the picture would need a single "pruning" lane.

## 1. Scope (HTML figure)

- **Sentence:** all five of the paper's claims are about decoding, and only our random-order rows have the tilt they describe.
- **Expected:** every claim lands in the decoding lane, and no claim varies a masking schedule.
- **Surprise:** a claim about training would have needed a training frame.

## 2. Predictions (HTML figure)

- **Sentence:** four predictions held, three failed or partly failed, and one cannot be decided.
- **Expected:** most hold, since the paper's claims are mathematical.
- **Surprise:** a failed prediction about the sign of the tilt.

## 3. The tilt in our teams

### 3a. Every draw is legal, and the team still comes out tilted (`00_exact_toy`)

- **Sentence:** removing illegal values one draw at a time, even exactly, changes which team the model gives.
- **Expected:** a nonzero total variation distance between the lookahead sampler and draw-then-reject.
- **Surprise:** a distance near zero, which would mean the per-step cut is the same on every path.
- **Measured:** 17.4% of the probability moves. Incineroar falls from 51.8% to 34.6%.
- **Form:** grouped bars, the two laws side by side on the axis the reader scans. The ability panel shows the mechanism:
  the share of move 2 each ability keeps.

### 3b. How much each row is bent (`02_tilt_per_team`)

- **Sentence:** species-first rows carry no tilt, and random-order rows carry about 14 nats per team.
- **Expected:** zero for species-first, a few nats for random order.
- **Surprise:** a nonzero species-first tilt, which would break the schedule comparison.

### 3c. Where the tilt sits (`03_tilt_by_field`)

- **Sentence:** random order's tilt sits at moves and abilities, the fields drawn before their species.
- **Cut:** the species-first rows are dropped. They were empty bars that took space and showed nothing.
- **Surprise:** tilt at species draws, which would point at Species Clause rather than the lookahead.

### 3d. The tables against the lookahead (`08_tables_vs_lookahead`)

- **Sentence:** the tables remove about eight times more probability than the lookahead, but their cut does not predict win rate.
- **Expected:** a large cut by the tables (untrained mass), with little tie to win rate.
- **Surprise:** a tables' Spearman near the lookahead's +0.39, which would mean the tables bend teams too.
- **Measured:** 108 and 151 nats, Spearman +0.10 and +0.04, against 13.6 and 15.2 nats with +0.39 and +0.32.

### 3e. Bent teams lose (`04_win_by_tilt_decile`)

- **Sentence:** the more the sampler bent a random-order team, the more that team loses.
- **Expected:** a rising line from the most-bent decile to the least.
- **Surprise:** a flat line, which would make the tilt harmless.

### 3f. Reweighting estimate (`05_tempered_correction`)

- **Sentence:** reweighting predicts that correcting the tilt lifts the average random-order team, but not the best one.
- **Expected:** the mean rises with the correction's strength, and best of 64 stays flat.
- **Note:** effective sample size is 1–2% at full strength, so read the curve's direction, not its end point.

### 3g. The fix, run (`09_fix_measured`)

- **Sentence:** drawing exactly from the target won back half of random order's gap, not the 84% reweighting estimated.
- **Expected (2026-11-15):** close to the reweighting estimate, 0.465.
- **Measured (#358, 2026-11-17):** 0.410, which wins back 49% [41%, 57%].
- **Why it redrew the frame:** the reweighting rested on 1–2% effective teams. This frame replaces the estimate as the
  number the blog quotes.

## 4. The model's consistency (`06_order_gap_by_field`)

- **Sentence:** the model is not one joint distribution, because the same team scores 18.2 nats higher species-first.

## 5. Experiment-design lessons (`01_paper_figure1`)

- **Sentence:** the paper's only bias evidence is one toy constraint, with no seeds and no matched compute.

## 6. The decoding rule (`07_non_top_share`)

- **Sentence:** about two fields in five are filled with a value that was not the model's top choice, in every row.

## Check of the two narratives

- **Figures alone:** teaser, then 3a → 3g read as mechanism, then size, then location, then the tables' contrast, then
  harm, then the estimated fix and the measured fix.
- **Words alone:** the lede and section 3's prose reach the same conclusion. The lookahead's tilt costs random order about
  half its gap, and the tables cost nothing.
