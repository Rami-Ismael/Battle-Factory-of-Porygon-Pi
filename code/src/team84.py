"""The 84-field team (issue #57): the 48 existing fields of `encode.Vocab` followed by six Stat Points per
Pokémon (`stat_points`), in the same Pokémon order (`encode.canon`, sorted by species)."""
import numpy as np

import stat_points as SP
from checkpoints import load_checkpoint
from corpus import norm
from encode import NF, NSLOT, Vocab, canon

COLS48 = NSLOT * NF
COLS = COLS48 + SP.COLS                                 # 84
LOOP_TEAMS_MODEL = "F1_final_medium_noheldout_regmb_v3"


def regulation_vocab(member=LOOP_TEAMS_MODEL):
    """The Regulation M-B vocabulary stored in a deployed checkpoint, so token ids match the 48-field model."""
    import lossdown as L
    return Vocab.from_regulation(load_checkpoint(L.CK / f"{member}.pt", map_location="cpu")["regulation"])


def encode(V, team):
    """int64 [84]; raises ValueError for a field outside the vocabulary or illegal Stat Points."""
    stat_tokens = SP.tokens_from_stat_points([s.get("evs", {}) for s in canon(team)])
    return np.concatenate([V.encode(team), np.array(stat_tokens, dtype=np.int64)])


def decode(V, row):
    """Six slot dicts (normalised species, ability, item, moves, Stat Alignment, Stat Points) from 84 tokens."""
    row = [int(v) for v in row]
    stats = SP.stat_points_from_tokens(row[COLS48:])
    slots = []
    for s in range(NSLOT):
        g = lambda j: V.decode_field(s * NF + j, row[s * NF + j])
        slots.append(dict(species=g(0), ability=g(1), item=g(2), nature=g(7),
                          moves=[g(3 + j) for j in range(4) if g(3 + j) not in ("", "[MASK]")], evs=stats[s]))
    return slots


__all__ = ["COLS", "COLS48", "SP", "decode", "encode", "norm", "regulation_vocab"]
