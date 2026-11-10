"""Researcher's inference from Austin et al. v3 Fig. 13, PDF page 31.
Usage: python3 count_trajectories.py paper.pdf output.json
Requires pypdf. No trained sampler is run.
"""
from pypdf import PdfReader
from collections import Counter
import json, sys, unicodedata
p = PdfReader(sys.argv[1]).pages[30]
rows = []
def operand(op, args, cm, tm):
    # Each printed space is a three-stroke graphic. Its horizontal stroke
    # is 2.79 PDF points wide. Count that stroke once, including at wraps.
    if rows and op == b'l' and len(args) == 2 and abs(float(args[0])-2.79) < .01 and float(args[1]) == 0:
        rows[-1]['state'] += ' '
def text(t, cm, tm, font, size):
    value = t.strip()
    if value in ['999', '800', '600', '400', '200', '0']:
        rows.append({'t': int(value), 'state': ''})
    elif rows and 'Figure' not in t and float(size) < 8:
        # PDF text extraction inserts layout whitespace; actual space tokens
        # are graphic marks handled above. NFKC expands fi/ff ligatures.
        rows[-1]['state'] += unicodedata.normalize('NFKC', ''.join(t.split()))
p.extract_text(visitor_operand_before=operand, visitor_text=text)
assert len(rows) == 18
assert all(len(r['state']) == 256 for r in rows)
assert all(set(r['state']) <= set('abcdefghijklmnopqrstuvwxyz ') for r in rows)
results = []
for k in range(3):
    sample = rows[k*6:(k+1)*6]
    steps = [r['t'] for r in sample]
    assert steps == [999, 800, 600, 400, 200, 0]
    states = [r['state'] for r in sample]
    changes = [[i+1 for i in range(256) if a[i] != b[i]] for a,b in zip(states,states[1:])]
    counts = [sum(i+1 in interval for interval in changes) for i in range(256)]
    last = [max([j+1 for j,c in enumerate(changes) if i+1 in c],default=0) for i in range(256)]
    results.append(dict(sample_top_to_bottom=k+1, steps=steps, states=states,
        position_indexing='one-based, including every space',
        observed_change_histogram=dict(sorted(Counter(counts).items())),
        changed_positions_by_interval=changes,
        last_observed_change_histogram=dict(sorted(Counter(last).items())),
        positions=[dict(position=i+1,values=[s[i] for s in states],
          observed_changes=counts[i],
          changed_intervals=[f'{steps[j]}->{steps[j+1]}' for j,c in enumerate(changes) if i+1 in c]) for i in range(256)]))
json.dump(results,open(sys.argv[2],'w'),indent=2)
print('Validated 18 states of 256 characters. Histograms:', [r['observed_change_histogram'] for r in results])
