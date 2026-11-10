"""Post-hoc structural diversity of the frozen DiBO/CEM comparison. No battles."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from corpus import parse_team_text
from diversity_metrics import team_keys
from echarts_chart import write_chart


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def keys(paste):
    # Parsing blocks separately preserves itemless slots in the legacy parser.
    slots = [s for block in paste.strip().split('\n\n') for s in parse_team_text(block)]
    return team_keys(slots)


def composition_stats(compositions):
    n = len(compositions)
    if not n:
        raise ValueError('empty population')
    if any(len(c) != 6 or len(set(c)) != 6 for c in compositions):
        raise ValueError('species replacement distance assumes six distinct species/forms')
    counts = Counter(tuple(sorted(c)) for c in compositions)
    probs = np.array(list(counts.values()), float) / n
    # Sum shared species over unordered distinct team pairs without a large matrix.
    species = Counter(s for c in compositions for s in c)
    shared_pairs = sum(k * (k-1) / 2 for k in species.values())
    pairs = n * (n-1) / 2
    return dict(n=n, distinct_compositions=len(counts), effective_compositions=float(np.exp(-(probs*np.log(probs)).sum())),
                largest_composition_share=float(max(probs)), species_richness=len(species),
                mean_species_replacements=6-shared_pairs/pairs if pairs else None,
                composition_frequencies=[dict(species=list(c), count=k) for c,k in sorted(counts.items(), key=lambda p: (-p[1],p[0]))],
                species_frequencies=[dict(species=s,count=k) for s,k in species.most_common()])


def metrics(records, initial):
    representations = [keys(r['paste']) for r in records]
    out = composition_stats([k[2] for k in representations])
    out['unique_categorical_teams'] = len({k[0] for k in representations})
    out['unique_teams_with_spreads'] = len({k[1] for k in representations})
    out['composition_novel_draw_fraction'] = sum(k[2] not in initial for k in representations)/len(records)
    # Within a fixed roster, count how many complete Pokemon builds differ.
    # A build includes ability, item, nature and unordered moves, excluding spreads.
    groups = {}
    for categorical, _, composition in representations:
        groups.setdefault(composition, []).append(categorical)
    pairs = 0; identical_build_pairs = 0
    for teams in groups.values():
        pairs += len(teams)*(len(teams)-1)//2
        counts = Counter(slot for team in teams for slot in team)
        identical_build_pairs += sum(k*(k-1)//2 for k in counts.values())
    out['same_roster_pairs'] = pairs
    out['mean_build_changes_within_same_roster'] = 6-identical_build_pairs/pairs if pairs else None
    return out


def self_check():
    a = tuple('abcdef'); b = tuple('abcdeg'); c = tuple('hijklm')
    same = composition_stats([a, tuple(reversed(a))])
    assert same['distinct_compositions'] == 1 and same['mean_species_replacements'] == 0
    assert same['effective_compositions'] == 1
    assert composition_stats([a,b])['mean_species_replacements'] == 1
    assert composition_stats([a,c])['mean_species_replacements'] == 6
    three = composition_stats([a,a,b])
    assert math.isclose(three['mean_species_replacements'], 2/3)
    assert math.isclose(three['effective_compositions'], math.exp(-2/3*math.log(2/3)-1/3*math.log(1/3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args(); root = args.directory.resolve()
    self_check()
    proto = json.loads((root/'protocol.json').read_text())
    data = {a:json.loads((root/f'{a}.json').read_text()) for a in ['cem','dibo']}
    initial = {keys(r['paste'])[2] for r in proto['initial']}
    result = dict(inputs={n:sha(root/n) for n in ['protocol.json','cem.json','dibo.json','summary.json']},
                  sources={str(p):sha(p) for p in [Path(__file__),ROOT/'src/diversity_metrics.py',ROOT/'src/corpus.py']},
                  parser_data={n:sha(Path('/tmp/vgc-pilot/data')/(n+'.json')) for n in ['pokedex','learnsets','items','abilities','moves']},
                  initial_reference='The same frozen 5000 starting observations; not the full foundation-training corpus.',
                  populations={'proposals':'Saved pools: CEM after deduplication and repair; DiBO after deduplication and local refinement, before final deduplication/ranking. Raw accepted streams were not saved.',
                               'selected':'128 distinct teams actually battled per round',
                               'finalists':'16 teams locked by search score before independent evaluations'},
                  arms={})
    for arm,out in data.items():
        rounds=[]; accumulated=[]
        for g,r in sorted(out['rounds'].items(),key=lambda p:int(p[0])):
            assert len(r['selected'])==len(r['evaluated'])==128 and len(r['proposals'])==512
            assert [a['key'] for a in r['selected']]==[a['key'] for a in r['evaluated']]
            pm=metrics(r['proposals'],initial); sm=metrics(r['selected'],initial)
            # Independently recompute the experiment's stored proposal composition count.
            assert pm['distinct_compositions']==r['species_sets']
            accumulated.extend(r['evaluated'])
            qualified=[x for x in accumulated if x['y']>=.531]
            rounds.append(dict(round=int(g)+1,proposals=pm,selected=sm,search_mean=float(np.mean([x['y'] for x in r['evaluated']])),
                               cumulative_compositions=len({keys(x['paste'])[2] for x in accumulated}),
                               cumulative_qualifying_compositions=len({keys(x['paste'])[2] for x in qualified})))
        final=metrics(out['final_selected'],initial)
        assert len(out['final_selected'])==len(out['heldout'])==len(out['confirmation'])==16
        final['confirmation_mean']=sum(r['wins'] for r in out['confirmation'])/sum(r['battles'] for r in out['confirmation'])
        final['heldout_mean']=sum(r['wins'] for r in out['heldout'])/sum(r['battles'] for r in out['heldout'])
        final['teams']=[dict(index=i,species=list(keys(r['paste'])[2]),search_score=r['y'],
                             heldout_wins=out['heldout'][i]['wins'],heldout_battles=out['heldout'][i]['battles']) for i,r in enumerate(out['final_selected'])]
        cumulative=metrics(accumulated,initial)
        cumulative['search_mean']=float(np.mean([r['y'] for r in accumulated]))
        cumulative['qualifying_compositions']=rounds[-1]['cumulative_qualifying_compositions']
        result['arms'][arm]=dict(rounds=rounds,all_battled=cumulative,finalists=final)
    dest=root/'diversity';dest.mkdir(exist_ok=True)
    (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result,dest)
    report(result,dest)
    print(json.dumps({a:{k:{x:v for x,v in d[k].items() if x not in ['composition_frequencies','species_frequencies','teams']} for k in ['all_battled','finalists']} for a,d in result['arms'].items()},indent=2))


def plot(result, dest):
    """Four panels as one ECharts page, quality-and-diversity.html in dest. analyze_dibo_diversity_v2.py draws with this too."""
    colors = {'cem': '#7c4d94', 'dibo': '#087e8b'}
    pct = "js:v => Math.round(v * 100) + '%'"
    heads = ['Distinct rosters among 128 battled teams', 'Most common roster’s share — lower is broader',
             'Average species replacements between teams', 'Win rate and diversity of the same battle batch']
    grids = [{'left': f'{left}%', 'width': '40%', 'top': top, 'height': '30%'} for top in ('14%', '60%') for left in (6, 56)]
    rounds = {'type': 'value', 'min': 1, 'max': 10, 'interval': 1, 'name': 'Round', 'nameLocation': 'middle', 'nameGap': 26}
    xs = [{**rounds, 'gridIndex': g} for g in range(3)] + [
        {'type': 'value', 'gridIndex': 3, 'min': 0, 'max': 6, 'name': 'Average species replacements', 'nameLocation': 'middle', 'nameGap': 26}]
    ys = [{'type': 'value', 'gridIndex': 0, 'min': 0, 'max': 128},
          {'type': 'value', 'gridIndex': 1, 'min': 0, 'max': 1, 'axisLabel': {'formatter': pct}},
          {'type': 'value', 'gridIndex': 2, 'min': 0, 'max': 6},
          {'type': 'value', 'gridIndex': 3, 'min': 0, 'max': 1, 'axisLabel': {'formatter': pct}, 'name': 'Search win rate',
           'nameLocation': 'middle', 'nameGap': 40}]
    series = []
    for arm, d in result['arms'].items():
        label, c, rs = 'CEM' if arm == 'cem' else 'DiBO', colors[arm], d['rounds']
        x = [r['round'] for r in rs]
        for g, key in enumerate(('distinct_compositions', 'largest_composition_share', 'mean_species_replacements')):
            series.append({'type': 'line', 'name': label, 'color': c, 'symbolSize': 6, 'emphasis': {'focus': 'series'},
                           'xAxisIndex': g, 'yAxisIndex': g, 'data': [[a, r['selected'][key]] for a, r in zip(x, rs)]})
        series.append({'type': 'scatter', 'name': label, 'color': c, 'symbolSize': 11, 'xAxisIndex': 3, 'yAxisIndex': 3,
                       'data': [[r['selected']['mean_species_replacements'], r['search_mean']] for r in rs]})
    titles = [{'text': h, 'left': g['left'], 'top': f"{float(g['top'][:-1]) - 5}%", 'textStyle': {'fontSize': 13}} for h, g in zip(heads, grids)]
    write_chart({'title': titles, 'grid': grids, 'xAxis': xs, 'yAxis': ys, 'series': series, 'tooltip': {'trigger': 'item'},
                 'legend': {'data': ['CEM', 'DiBO'], 'top': 0, 'right': 0}},
                dest / 'quality-and-diversity.html', 'Structural diversity alongside performance · one matched-budget run', height=700)


def report(result,dest):
    arms=result['arms'];lines=['# Diversity and win rate: DiBO versus CEM','',
        'Measured retrospectively from the completed equal-battle-budget experiment. No additional battles or changes to finalist selection. Each method used 68,864 fresh battles. These are structural measurements of this run, not proof of different strategies or superiority across seeds.','',
        f"Across 1,280 battled teams, CEM covered **{arms['cem']['all_battled']['distinct_compositions']} rosters versus DiBO’s {arms['dibo']['all_battled']['distinct_compositions']}**. The 16 finalists contained {arms['cem']['finalists']['distinct_compositions']} CEM rosters versus {arms['dibo']['finalists']['distinct_compositions']} DiBO roster. Moves, items, abilities and natures can still differ within a roster. The two kinds of variety should be read separately: broad roster coverage and variation of builds within a roster.",'',
        '**Final outputs: the same 16 locked finalists used for independent evaluation.**','',
        '| Method | Heldout mean | Confirmation mean | Distinct rosters /16 | Effective rosters | Largest roster share | Mean species replacements | Species represented |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,d in arms.items():
        f=d['finalists'];lines.append(f"| {a.upper() if a=='cem' else 'DiBO'} | {f['heldout_mean']:.1%} | {f['confirmation_mean']:.1%} | {f['distinct_compositions']} | {f['effective_compositions']:.2f} | {f['largest_composition_share']:.1%} | {f['mean_species_replacements']:.2f} /6 | {f['species_richness']} |")
    lines += ['', '**Variation within the same roster.** All 16 finalists in each method are distinct in their categorical builds. Among pairs sharing the same six species, the following measures how many Pokémon differ in at least one ability, item, nature or move. Stat Points are excluded from this build measure.','',
              '| Method | Same-roster finalist pairs | Mean Pokémon builds changed /6 |','|---|---:|---:|']
    for a,d in arms.items():
        f=d['finalists'];lines.append(f"| {a.upper() if a=='cem' else 'DiBO'} | {f['same_roster_pairs']} | {f['mean_build_changes_within_same_roster']:.2f} |")
    lines += ['', 'DiBO shows more build variation among same-roster finalist pairs in this run, despite narrower roster coverage. A changed build can be a small nature/item change or a larger move change; this metric does not establish a new strategy or better quality. The averages condition on same-roster pairs, whose counts differ between methods. Neither metric alone is an overall diversity score.']
    lines += ['', '**Search outputs: all 1,280 battled teams per method.**','',
              '| Method | Search mean | Distinct rosters | Effective rosters | Largest roster share | Mean species replacements | Rosters with at least one ≥53.1% search label |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for a,d in arms.items():
        f=d['all_battled'];lines.append(f"| {a.upper() if a=='cem' else 'DiBO'} | {f['search_mean']:.1%} | {f['distinct_compositions']} | {f['effective_compositions']:.2f} | {f['largest_composition_share']:.1%} | {f['mean_species_replacements']:.2f} /6 | {f['qualifying_compositions']} |")
    lines += ['', 'The 53.1% cutoff is the existing CEM elite threshold, reused as an explicitly post-hoc quality-and-coverage summary for both methods. These are noisy search labels; one qualifying variant does not establish that its roster reliably exceeds the cutoff. Search scores are weighted across the top50 list and are not independent heldout performance.','',
              '[Win rate and structural diversity](quality-and-diversity.html)','',
              '**How diversity changed during search.** Win rate and selected-roster count below refer to the same 128 teams each round. Proposal counts refer to the separate saved 512-entry pools.','',
              '| Round | CEM proposal rosters /512 | DiBO proposal rosters /512 | CEM battled rosters /128 | DiBO battled rosters /128 | CEM search mean | DiBO search mean |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for c,d in zip(arms['cem']['rounds'],arms['dibo']['rounds']):
        lines.append(f"| {c['round']} | {c['proposals']['distinct_compositions']} | {d['proposals']['distinct_compositions']} | {c['selected']['distinct_compositions']} | {d['selected']['distinct_compositions']} | {c['search_mean']:.1%} | {d['search_mean']:.1%} |")
    lines += ['', '**Definitions and limits.** A roster is the unordered six species/form identifiers; changing moves, items or Stat Points does not create a new roster. Effective rosters is exp(Shannon entropy): 16 equally represented rosters score 16, while one repeated roster scores 1. Largest share exposes concentration hidden by a count of rare alternatives. Mean species replacements is 6 minus the average shared-species count across every unordered pair of different output entries; identical rosters score 0, disjoint rosters 6. Repeated rosters retain their frequencies.','',
              'Exact team uniqueness is also recorded with and without Stat Points, ignoring slot and move ordering. Because the experiment deliberately selected previously unseen distinct teams, exact uniqueness is partly enforced and is not sufficient evidence of diversity. Counts should only be compared at the same sample size and stage.','',
              'Saved CEM proposal pools are after deduplication and nature repair. Saved DiBO pools are after deduplication and local refinement; refinement can introduce repeats before final selection deduplicates again. Thus proposal-pool diversity describes delivered candidates at these stages, not raw generator entropy or raw duplicate rates. Raw accepted streams were not retained, so those rates cannot be reconstructed.','',
              'The metrics file includes composition frequencies, species frequencies, novelty relative to the frozen 5,000 initial observations, per-finalist results, and input/source hashes. Novelty to that initial pool is not novelty to the full pretrained corpus. Heldout scores have not been used to choose a different final team set.','',
              '[Machine-readable metrics](metrics.json) · [Original win-rate report](../analysis.md) · [Implementation diagnosis](../diagnostics/diagnosis.md)']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
