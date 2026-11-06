"""The lookahead: forward checking for a sampler that may draw a Pokémon's set before its species (issue #205).

One rule for both diffusion models. The 84-field model calls it on its first 48 fields; the 48-field model moves onto
it when the rename branch is reconciled (#65). It reads only the legality tables in lossdown.Tables (BASE: each
species' base species; AB, IT, MV: which species can have which ability, item and move; ABIT: the item and ability
pairs Showdown ties) and the candidates tables.allowed() gives.

    ok = lookahead(tables, x48, c)          # Boolean [B, values] for column c, to AND into the legality mask

For a Pokémon whose species is masked, a value is allowed only if at least one candidate species can have it, with
the visible item or ability where Showdown ties the two. A candidate is a species legal with the Pokémon's visible
fields and Species Clause. A species draw may not take the base species that another masked Pokémon has as its only
candidate left. Stat Alignment is never restricted: no species depends on it. Once a Pokémon's species is filled the
rule allows every value, and the legality tables alone decide.
"""
import torch
import torch.nn.functional as F

from encode import NF, NSLOT

ABILITY, ITEM, NATURE = 1, 2, NF - 1                    # field offsets within a Pokémon; 3..6 are the moves


def lookahead(tables, x, c, al=None):
    """Boolean [B, values] for column c of the 48-field block x [B, 48] (0 = masked): the values that leave every
    Pokémon whose species is masked at least one candidate species. al: tables.allowed(x), if already computed."""
    al = tables.allowed(x) if al is None else al
    B, p, j = len(x), c // NF, c % NF
    species_masked = x.view(B, NSLOT, NF)[..., 0] == 0
    if j == 0:
        bases = F.one_hot(tables.BASE, tables.nbase).float()                          # [species, base]
        sb = (al["species"].float() @ bases) > 0                                      # [B, 6, base] bases still possible
        only = sb & ((sb.sum(-1) == 1) & species_masked).unsqueeze(-1)
        only[:, p] = False                                                            # p itself may take its last base
        return ~only.any(1)[:, tables.BASE]
    if j == NATURE: return torch.ones_like(al["nature"][:, p])
    S = al["species"][:, p].float()                                                   # [B, species] candidates of p
    table = {ABILITY: tables.AB, ITEM: tables.IT}.get(j, tables.MV)
    ok = (S @ table.float()) > 0
    if tables.ABIT is not None and j in (ABILITY, ITEM):
        other = x[:, p * NF + (ITEM if j == ABILITY else ABILITY)]                    # the visible item or ability
        if (other != 0).any():
            joint = tables.ABIT if j == ABILITY else tables.ABIT.permute(0, 2, 1)     # [species, other, values]
            both = torch.einsum("bn,bnv->bv", S, joint[:, other].permute(1, 0, 2).float()) > 0
            ok = torch.where((other != 0).unsqueeze(1), ok & both, ok)
    return torch.where(species_masked[:, p].unsqueeze(1), ok, torch.ones_like(ok))
