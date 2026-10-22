"""Summarize held-out uncertainty and outcome changes without promoting proxies."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from allocation import load_grid
from run import analyze


def main():
    ap=argparse.ArgumentParser();ap.add_argument('corpus',type=Path)
    ap.add_argument('--student-run',type=Path,required=True);args=ap.parse_args()
    grid,candidates,opponents=load_grid(args.corpus)
    exploration,heldout=np.split(grid,2,axis=2)
    means=heldout.mean(axis=(1,2));n=heldout.shape[1]*heldout.shape[2]
    # Fixed held-out bounded observations, union bound over all candidate scores.
    radius=math.sqrt(math.log(2*len(candidates)/.05)/(2*n))
    score_intervals=[{'candidate':candidate,'mean':float(mean),
        'simultaneous_hoeffding95':[max(0,float(mean)-radius),min(1,float(mean)+radius)]}
        for candidate,mean in zip(candidates,means)]
    best=int(np.argmax(means));pair_count=len(candidates)*(len(candidates)-1)//2
    pair_radius=math.sqrt(2*math.log(2*pair_count/.05)/n)
    pairs=[]
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            difference=float(means[i]-means[j])
            pairs.append({'candidates':[candidates[i],candidates[j]],'difference':difference,
                'simultaneous_hoeffding95':[max(-1,difference-pair_radius),min(1,difference+pair_radius)]})
    rows=[json.loads(line) for line in (args.student_run/'trials.jsonl').read_text().splitlines()]
    accepted=analyze(rows).get('student',{}).get('accepted_block_ids',[])
    if not accepted:raise ValueError('No complete valid student timing blocks')
    measured=[r for r in rows if r.get('phase')=='measure' and r.get('block') in accepted]
    student_summary={}
    for variant in ['baseline','student']:
        selected=[r for r in measured if r['variant']==variant]
        battles=sum(r['completed'] for r in selected)
        student_summary[variant]={'battles':battles,'turns_per_battle':sum(r['turns'] for r in selected)/battles,
            'candidate_win_fraction':sum(r['wins'] for r in selected)/battles,
            'candidate_score':sum(r['wins']+.5*r['ties'] for r in selected)/battles,
            'note':'Both players use this variant; this is a changed evaluator, not student-versus-teacher head-to-head strength.'}
    report={'heldout_battles_per_candidate':n,'scores':score_intervals,'pairwise_differences':pairs,
        'empirical_best_candidate':candidates[best],
        'certified_unique_best':all(means[best]-means[i]>pair_radius for i in range(len(candidates)) if i!=best),
        'student_outcomes':student_summary,
        'assumptions':['Fixed balanced opponent objective; scores lie in [0,1].',
            'Bounds assume independent simulation replicates; common seeds across candidates are allowed.',
            'Seeded pseudorandom simulation approximates independent draws; deterministic replay is not fresh evidence.',
            'Conservative simultaneous bounds are for the fixed held-out dataset, not adaptively stopped exploration.',
            'Student outcome aggregates mix the three recorded candidate rows; they do not establish overall policy strength.']}
    (args.corpus/'quality-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'certified_unique_best':report['certified_unique_best'],
                     'student_outcomes':student_summary,'heldout_score_radius':radius}))


if __name__=='__main__':main()
