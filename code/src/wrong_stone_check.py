"""Refuse a training table that holds a wrong-stone team (owner, 2026-11-17, issue #368; retrain #385).

A wrong-stone team gives a Pokémon another species' Mega Stone. Showdown's validator accepts it, but the legal Pokémon
construction tables forbid it, and the owner calls these teams bad data. The trainer's own check runs on randomly masked
rows, so a team slips through whenever the species or the stone is masked: 11 reached training before #385. This check
reads every field of every row in every *_X group, so every 84-field training run starts from the filtered table.

To filter a stored table in place, keeping its teams and split:

    /tmp/vgc-pilot/.venv/bin/python src/lossdata84.py --drop-wrong-stones NAME
"""
import lossdown as L
import team84 as T

METHOD = "Tables.rejects (every field visible)"
REFUSAL = ("{table} holds wrong-stone teams ({found}): a Pokémon holding another species' Mega Stone, which the legal "
           "Pokémon construction tables forbid. Training reads only filtered tables. Filter it in place with "
           "`python src/lossdata84.py --drop-wrong-stones {stem}`.")


def wrong_stone_teams(table, vocab=None):
    """{group: rows rejected} for every *_X group of a loaded table, read with every field visible."""
    tables = L.Tables(vocab or T.regulation_vocab(), [])
    return {key[:-2]: int(tables.rejects(table[key][:, :T.COLS48]).sum()) for key in table if key.endswith("_X")}


def refuse_wrong_stones(table, vocab=None):
    """The check's record for the run info; stops the run if any group holds a wrong-stone team."""
    found = wrong_stone_teams(table, vocab)
    if any(found.values()):
        listed = ", ".join(f"{n} in {group}_X" for group, n in found.items() if n)
        raise SystemExit(REFUSAL.format(table=table["table"], found=listed, stem=table["table"].removesuffix(".npz")))
    return dict(method=METHOD, rows={group: len(table[f"{group}_X"]) for group in found}, rejected=0)
