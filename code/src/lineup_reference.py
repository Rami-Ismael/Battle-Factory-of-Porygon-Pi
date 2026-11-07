"""The two fixed line-up references for novelty on the 84-field model: training line-ups, and loop and tournament record line-ups."""
import re

import numpy as np

import ask_grid84 as A                                   # also sets up sys.path for the modules below
import team84 as T
from team_diversity import slots_from_paste


def species_of(paste):
    return tuple(sorted(s["species"] for s in slots_from_paste(paste)))


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def references():
    """Line-ups in the training teams, and in the loop and tournament records: the two fixed references for novelty."""
    data = np.load(A.DATA)
    data = {k: data[k] for k in ("all_X", "all_val", "loop_X", "records_X")}
    vocab, names, cols = T.regulation_vocab(), {}, [s * 8 for s in range(6)]
    for k in ("all_X", "loop_X", "records_X"):
        rows = data[k]
        for i in np.unique(rows[:, cols]):
            if int(i) not in names:
                at = np.argwhere(rows[:, cols] == i)[0]
                names[int(i)] = norm(T.decode(vocab, rows[at[0]])[at[1]]["species"])

    def named(rows):
        return {tuple(sorted(names[int(i)] for i in r)) for r in np.unique(rows[:, cols], axis=0)}

    return named(data["all_X"][~data["all_val"]]), named(data["loop_X"]) | named(data["records_X"])
