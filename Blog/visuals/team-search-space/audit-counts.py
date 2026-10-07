"""Independent, read-only arithmetic audit. Python standard library only.

Uses Newton identities instead of the original coefficient dynamic program.
Run: python3 Blog/visuals/team-search-space/audit-counts.py
"""
import json
from collections import defaultdict
from math import comb, factorial
from pathlib import Path

root = Path(__file__).resolve().parent
domain = json.loads((root / 'domain.json').read_text())
recorded = json.loads((root / 'counts.json').read_text())

def coefficient(weights, degree=6):
    weights = list(weights)
    powers = [0] + [sum(w ** k for w in weights) for k in range(1, degree + 1)]
    elementary = [1]
    for k in range(1, degree + 1):
        numerator = sum((-1) ** (i - 1) * elementary[k-i] * powers[i]
                        for i in range(1, k + 1))
        assert numerator % k == 0
        elementary.append(numerator // k)
    return elementary[-1]

def roster(weight):
    by_species = defaultdict(int)
    for form in domain['roster']:
        by_species[form['num']] += weight(form)
    return coefficient(by_species.values())

def spreads(total, exact=True):
    degree = 5 if exact else 6
    return sum((-1) ** j * comb(6, j) * comb(total - 33*j + degree, degree)
               for j in range(min(6, total // 33) + 1))

n = len({r['num'] for r in domain['roster']})
m = len(domain['items'])
items = sum(comb(6, k) * factorial(m) // factorial(m-k) for k in range(7))
forms = roster(lambda r: 1)
abilities = roster(lambda r: len(r['pairs']))
moves = roster(lambda r: sum(comb(len(p['moves']), min(4, len(p['moves'])))
                            for p in r['pairs']))
alignment_count = len(domain['alignments'])
full = spreads(66)
partial = spreads(66, exact=False)
expected = [comb(n, 6), forms, abilities, abilities*items, moves*items,
            moves*items*alignment_count**6, moves*items*alignment_count**6*full**6]
for stage, value in zip(recorded['stages'], expected):
    assert int(stage['count']) == value, stage['name']
    print(f"PASS {stage['name']}: {value}")
assert int(recorded['itemAssignments']) == items
assert recorded['fullySpentSpreads'] == full
assert recorded['partialSpreads'] == partial
assert int(recorded['withPartialSpreads']) == expected[-2]*partial**6
print(f'PASS item assignments: {items}; full SP: {full}; partial SP: {partial}')
print('Arithmetic verified against stored domain; this does not certify domain completeness or legality.')
