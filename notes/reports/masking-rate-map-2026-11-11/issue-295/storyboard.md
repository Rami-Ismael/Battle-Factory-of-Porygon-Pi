# Storyboard: Add the linear masking rate and the corrected t to the 84-field code, with tests

Figure sentence: Five build items let the four rows run, and the default table is done.

## 1. Train, sample, battle: where each build item sits

- Sentence: Each row goes through training, sampling and battles, and each step needs build items.
- Measured: One part of item 1 is done: the shuffled-only table is already the default training table.

## 2. Five build items and their state

- Sentence: The default table is in place, and the two training arms of item 1 come next.
- Measured: Done in commit b00a71ff: training reads the shuffled-only table unless --older-table is passed.

## 3. Code today against code after this ticket

- Sentence: Four fixed behaviours become settings, and today's behaviour stays the default.
- Expected: Stored runs reproduce unchanged, because every new setting defaults to today's behaviour.
- Surprise: A stored result changes under the new code.

## 4. What each test proves

- Sentence: Each build item has one test that checks the one thing it must do.
- Expected: All five pass before any training starts.
- Surprise: Guidance 2 fails to reproduce a stored cell, so the new setting changed old behaviour.
## Cut

- Training and battle results: they belong to issues 296 and 297.
- The novelty references: an open question, not a build item.
- Commit and checksum identifiers: the owner does not need them.
