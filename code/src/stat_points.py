"""Stat Points as six generated fields per Pokémon (issue #57).

A team's 84 fields are the 48 existing ones (for each of six Pokémon: species, ability, item, four moves, Stat Alignment)
plus six Stat Points per Pokémon in the order of `STATS`. Each Stat Point is 0 to 32 and a Pokémon's six total at
most 66. A Stat Point token is the value plus one; token 0 is [MASK].

This module loads no game data, so CI can test it.
"""
import torch

STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]      # the order of corpus.STATS
NSTAT = len(STATS)
NSLOT = 6
COLS = NSLOT * NSTAT                                    # 36 Stat Point columns per team
MAX_STAT = 32
MAX_TOTAL = 66
NVALUES = MAX_STAT + 1                                  # values 0..32


def tokens_from_stat_points(pokemon):
    """36 tokens from six Stat Point dicts (a missing stat is 0), in the same Pokémon order as the 48 fields."""
    if len(pokemon) != NSLOT:
        raise ValueError(f"a team has {NSLOT} Pokémon, got {len(pokemon)}")
    tokens = []
    for sp in pokemon:
        values = [int(sp.get(s, 0)) for s in STATS]
        if any(v < 0 or v > MAX_STAT for v in values) or sum(values) > MAX_TOTAL:
            raise ValueError(f"illegal Stat Points {sp}: each 0..{MAX_STAT}, total at most {MAX_TOTAL}")
        tokens += [v + 1 for v in values]
    return tokens


def stat_points_from_tokens(tokens):
    """Six Stat Point dicts from 36 tokens; a [MASK] token is an error."""
    tokens = [int(t) for t in tokens]
    if len(tokens) != COLS or min(tokens) < 1:
        raise ValueError("need 36 drawn Stat Point tokens, none masked")
    return [dict(zip(STATS, (t - 1 for t in tokens[i:i + NSTAT]))) for i in range(0, COLS, NSTAT)]


def allowed_stat_points(tokens):
    """Boolean [B, 36, 33]: for each Stat Point column, the values (not tokens) that keep the column at most 32
    and its Pokémon's total at most 66, given the Stat Points already drawn (masked ones count as 0)."""
    values = (tokens.view(-1, NSLOT, NSTAT) - 1).clamp(min=0)
    others = values.sum(-1, keepdim=True) - values                      # the Pokémon's other five, drawn so far
    budget = (MAX_TOTAL - others).clamp(max=MAX_STAT)                   # [B, 6, 6]
    v = torch.arange(NVALUES, device=tokens.device)
    return (v <= budget.unsqueeze(-1)).view(-1, COLS, NVALUES)
