"""Replay allocation strategies against completed battles and untouched holdout.

Repeated shuffles measure stability on one finite corpus, not independent new
experiments. No adaptively stopped confidence interval is presented as valid.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def load_grid(folder):
    rows=[json.loads(line) for line in (folder/'outcomes.jsonl').read_text().splitlines()]
    candidates=sorted({r['candidate'] for r in rows})
    opponents=sorted({r['opponent'] for r in rows})
    replicates=sorted({r['replicate'] for r in rows})
    if replicates!=list(range(len(replicates))) or len(replicates)%2:
        raise ValueError('Expected contiguous even replicate schedule')
    grid=np.full((len(candidates),len(opponents),len(replicates)),np.nan)
    ci={c:i for i,c in enumerate(candidates)};oi={o:i for i,o in enumerate(opponents)}
    for row in rows:
        index=(ci[row['candidate']],oi[row['opponent']],row['replicate'])
        if not np.isnan(grid[index]):raise ValueError('Duplicate battle identity')
        if not row['finished']:raise ValueError('Nonterminal outcome')
        expected='exploration' if row['replicate']<len(replicates)//2 else 'heldout'
        if row['split']!=expected:raise ValueError('Incorrect corpus split')
        grid[index]=row['score']
    if np.isnan(grid).any():raise ValueError('Incomplete balanced corpus')
    if not np.isin(grid,[0,.5,1]).all():raise ValueError('Invalid battle score')
    return grid,candidates,opponents


def replay(grid,budget,method,seed,features=None):
    """Return selected candidate and actual sample counts; never access holdout."""
    c,o,r=grid.shape
    if method not in ['uniform','race','stratified','surrogate']:raise ValueError('Unknown allocation method')
    if budget<c*o*2 or budget>grid.size:raise ValueError('Budget outside pilot/corpus limits')
    rng=np.random.default_rng(seed)
    # Preserve common random numbers across candidates within each opponent.
    order=np.stack([rng.permutation(r) for _ in range(o)])
    counts=np.zeros((c,o),dtype=int);sums=np.zeros((c,o));squares=np.zeros((c,o))
    used=0
    def sample(i,j):
        nonlocal used
        if counts[i,j]>=r:return False
        value=grid[i,j,order[j,counts[i,j]]]
        counts[i,j]+=1;sums[i,j]+=value;squares[i,j]+=value*value;used+=1
        return True
    def scores():return (sums/np.maximum(counts,1)).mean(axis=1)
    for i in range(c):
        for j in range(o):
            sample(i,j);sample(i,j)
    active=list(range(c));next_elimination=used+max(c*o,(budget-used)//3)
    pilot_means=sums/counts
    # Fixed pilot variance prevents adaptive opponent allocation from using
    # held-out results. Equal opponent weights remain in the final score.
    pilot_variance=np.maximum(squares/counts-pilot_means**2,.05)
    if method=='surrogate':
        if features is None or features.shape[0]!=c:raise ValueError('Surrogate requires candidate features')
        x=np.column_stack([np.ones(c),features])
        penalty=np.eye(x.shape[1])*.5;penalty[0,0]=1e-8
        weights=np.linalg.solve(x.T@x+penalty,x.T@pilot_means.mean(axis=1))
        predicted=x@weights
        preferred=sorted(range(c),key=lambda i:(-predicted[i],i))[:math.ceil(c/2)]
    while used<budget:
        available=[i for i in active if np.any(counts[i]<r)]
        if not available:break
        if method=='race' and used>=next_elimination and len(active)>1:
            ranking=sorted(active,key=lambda i:(-scores()[i],i))
            active=ranking[:math.ceil(len(ranking)/2)]
            next_elimination=used+max(len(active)*o,(budget-used)//2)
            continue
        if method=='surrogate':
            # All pilot battles count toward the budget. Audit rejected teams
            # randomly instead of allowing permanent, unobserved rejection.
            shortlist=[i for i in preferred if i in available]
            if rng.random()<.2:i=int(rng.choice(available))
            else:i=min(shortlist or available,key=lambda i:(counts[i].sum(),i))
        else:i=min(available,key=lambda i:(counts[i].sum(),i))
        available_opponents=np.flatnonzero(counts[i]<r)
        if method=='stratified':
            # Greedy reduction in sum_j w_j^2 sigma_j^2 / n_j, w_j=1/o.
            gain=pilot_variance[i]/(counts[i]*(counts[i]+1))
            j=max(available_opponents,key=lambda j:(gain[j],-j))
        else:j=min(available_opponents,key=lambda j:(counts[i,j],j))
        sample(i,j)
    eligible=active if method=='race' else list(range(c))
    chosen=max(eligible,key=lambda i:(scores()[i],-i))
    return chosen,used,counts


def paired_variance(exploration):
    c,o,r=exploration.shape
    output=[]
    for a in range(c):
        for b in range(a+1,c):
            differences=exploration[a]-exploration[b]
            paired=float(differences.var(axis=1,ddof=1).sum()/(o*o*r))
            independent=float((exploration[a].var(axis=1,ddof=1)+exploration[b].var(axis=1,ddof=1)).sum()/(o*o*r))
            output.append({'candidate_indices':[a,b],'paired_variance_of_mean_difference':paired,
                'independent_variance_of_mean_difference':independent,
                'estimated_independent_over_paired':independent/paired if paired else None})
    return output


def analyze(folder,repeats=200):
    grid,candidates,opponents=load_grid(folder)
    exploration,heldout=np.split(grid,2,axis=2)
    heldout_scores=heldout.mean(axis=(1,2));best=float(heldout_scores.max())
    pilot=grid.shape[0]*grid.shape[1]*2
    metadata=json.loads((folder/'metadata.json').read_text())
    species=metadata.get('candidate_species')
    features=None
    if species:
        sets=[set(species[str(c)].split(',')) for c in candidates]
        vocabulary=sorted(set.union(*sets))
        features=np.array([[float(s in team) for s in vocabulary] for team in sets])
    budgets=sorted(set([pilot,min(exploration.size,pilot*2),min(exploration.size,pilot*4),exploration.size]))
    experiments=[]
    for budget in budgets:
        for method in ['uniform','race','stratified']+(['surrogate'] if features is not None else []):
            records=[]
            for repeat in range(repeats):
                selected,used,counts=replay(exploration,budget,method,4000+repeat,features)
                records.append({'candidate':candidates[selected],'battles':used,
                    'heldout_score':float(heldout_scores[selected]),'heldout_regret':best-float(heldout_scores[selected]),
                    'min_cell_count':int(counts.min()),'max_cell_count':int(counts.max())})
            experiments.append({'method':method,'budget':budget,
                'mean_actual_battles':float(np.mean([x['battles'] for x in records])),
                'mean_heldout_score':float(np.mean([x['heldout_score'] for x in records])),
                'mean_heldout_regret':float(np.mean([x['heldout_regret'] for x in records])),
                'holdout_best_selection_fraction':float(np.mean([x['heldout_regret']==0 for x in records])),
                'selected_counts':{str(c):sum(x['candidate']==c for x in records) for c in candidates},
                'replicates':records})
    return {'candidates':candidates,'opponents':opponents,'heldout_scores':heldout_scores.tolist(),
        'exploration_battles':exploration.size,'heldout_battles':heldout.size,
        'shuffle_repeats':repeats,'experiments':experiments,'paired_randomness':paired_variance(exploration),
        'limitations':['Shuffle repetitions reuse one finite corpus and are not independent experiments.',
            'Held-out scores have sampling error; regret uses the empirical held-out best, not known true strength.',
            'Common seeded RNG need not reduce variance; report every candidate pair.',
            'Racing may exhaust surviving candidates before its maximum budget; actual battle counts are reported.',
            'The surrogate is a pilot-trained ridge predictor over species presence, with 20% random audits; this tests allocation within this pool, not generalization to unseen teams.',
            'Results compare battle budgets, not wall-clock quality; variable battle lengths and scheduler cost require a live follow-up.',
            'No fixed-sample interval is used to justify adaptive stopping.']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('corpus',type=Path)
    ap.add_argument('--repeats',type=int,default=200);args=ap.parse_args()
    if args.repeats<1:ap.error('Positive repeat count required')
    report=analyze(args.corpus,args.repeats)
    (args.corpus/'allocation-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'experiments':len(report['experiments']),'paired_comparisons':len(report['paired_randomness'])}))


if __name__=='__main__':main()
