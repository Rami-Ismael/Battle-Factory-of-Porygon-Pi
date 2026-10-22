"""Audit live allocation accounting and compare paired block outcomes."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def analyze(folder):
    rows=[json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
    metadata=json.loads((folder/'metadata.json').read_text())
    candidates={c:i for i,c in enumerate(metadata['candidate_ids'])}
    opponents={o:i for i,o in enumerate(metadata['opponent_ids'])}
    entries=collections.defaultdict(list)
    for line in (folder/'exploration.jsonl').read_text().splitlines():
        entry=json.loads(line);entries[entry['trial']].append(entry)
    failures=[]
    seen=set()
    for row in rows:
        identity=row['trial']
        if identity in seen:failures.append(f'Duplicate trial {identity}')
        seen.add(identity)
        results=entries[identity]
        eligible=sum(e['eligible_before_deadline'] for e in results)
        if len(results)!=row['physical_battles'] or eligible!=row['eligible_battles']:
            failures.append(f'Trial {identity}: exploration log count mismatch')
        if eligible+row['late_battles']!=len(results) or np.sum(row['counts'])!=eligible:
            failures.append(f'Trial {identity}: eligible/late count mismatch')
        matrix=np.zeros((len(candidates),len(opponents)),dtype=int)
        for entry in results:
            if entry['candidate'] not in candidates or entry['opponent'] not in opponents:
                failures.append(f'Trial {identity}: unknown team identity');continue
            if entry['eligible_before_deadline']:
                matrix[candidates[entry['candidate']],opponents[entry['opponent']]]+=1
        if not np.array_equal(matrix,np.array(row['counts'])):
            failures.append(f'Trial {identity}: candidate/opponent allocation mismatch')
        if bool(np.all(np.array(row['counts'])>=2))!=row['pilot_complete']:
            failures.append(f'Trial {identity}: pilot status mismatch')
        if row['pilot_complete']!=(row['selected_candidate'] is not None):
            failures.append(f'Trial {identity}: missing or premature selection')
    if set(entries)-seen:failures.append('Exploration records exist for missing trials')
    comparisons={}
    for treatment in sorted({r['treatment'] for r in rows}):
        grouped=collections.defaultdict(list)
        for r in rows:
            if r['treatment']==treatment:grouped[r['block']].append(r)
        accepted=[];excluded=[];quality=[];counts=[]
        for block,group in sorted(grouped.items()):
            reasons=[]
            if len(group)!=4 or sorted(r['position'] for r in group)!=[0,1,2,3]:reasons.append('incomplete block')
            if collections.Counter(r['method'] for r in group)!={'uniform':2,treatment:2}:reasons.append('unbalanced block')
            if any(r['clock']['pause_or_clock_jump'] for r in group):reasons.append('clock discontinuity')
            if reasons:excluded.append({'block':block,'reasons':reasons});continue
            accepted.append(block)
            a=[r for r in group if r['method']=='uniform'];b=[r for r in group if r['method']==treatment]
            counts.append(np.mean([r['eligible_battles'] for r in b])/np.mean([r['eligible_battles'] for r in a]))
            # Failures are retained separately, never imputed as a successful selection.
            if all(r['pilot_complete'] for r in group):
                quality.append(np.mean([r['heldout_score'] for r in b])-np.mean([r['heldout_score'] for r in a]))
        def interval(values,seed):
            if not values:return None
            values=np.array(values);rng=np.random.default_rng(seed)
            samples=rng.choice(values,(10000,len(values)),replace=True).mean(axis=1)
            return {'mean':float(values.mean()),'bootstrap_95_percent_interval':np.quantile(samples,[.025,.975]).tolist()}
        accepted_rows=[r for r in rows if r['treatment']==treatment and r['block'] in accepted]
        comparisons[treatment]={'accepted_blocks':accepted,'excluded_blocks':excluded,
            'quality_comparable_blocks':len(quality),'selected_holdout_score_difference':interval(quality,804),
            'eligible_battle_ratio':interval(counts,805),
            'pilot_failures':sum(not r['pilot_complete'] for r in accepted_rows),
            'methods':{method:{'trials':sum(r['method']==method for r in accepted_rows),
                'mean_selected_holdout_score':float(np.mean([r['heldout_score'] for r in accepted_rows if r['method']==method and r['pilot_complete']]))
                    if any(r['method']==method and r['pilot_complete'] for r in accepted_rows) else None,
                'mean_eligible_battles':float(np.mean([r['eligible_battles'] for r in accepted_rows if r['method']==method]))
                    if accepted_rows else None} for method in ['uniform',treatment]}}
    return {'trials':len(rows),'physical_battles':sum(r['physical_battles'] for r in rows),
        'eligible_battles':sum(r['eligible_battles'] for r in rows),'accounting_failures':failures,
        'comparisons':comparisons,'limitations':[
            'Intervals resample complete timing blocks, conditional on the same finite held-out outcome grid.',
            'They do not include held-out-label uncertainty or establish the truly strongest team.',
            'Five blocks are an exploratory screen; no multiplicity correction is applied.',
            'Post-deadline results are logged and drained but cannot influence selection.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    result=analyze(args.folder)
    (args.folder/'analysis-v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['accounting_failures']:raise SystemExit(1)
