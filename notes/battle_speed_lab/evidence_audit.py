"""Verify preserved experiment evidence without treating a report as proof."""
import argparse
import hashlib
import json
from pathlib import Path
from run import HERE,analyze
from analyze_live_allocation import analyze as analyze_allocation


RUNS=['bc-screen-002','bc-optimization-001','bc-confirm-001',
      'campaign-001-direct-bc','campaign-001-cold-worker','campaign-001-more-settings',
      'campaign-001-concurrency','campaign-002-direct-engine','campaign-002-cold-engine',
      'campaign-002-scaling','campaign-002-snapshot-bench','campaign-002-remote-bench',
      'campaign-002-cache-isolation','campaign-003-queue','campaign-003-array',
      'campaign-003-node-actor','campaign-003-student','optimized-pool-001','deadline05-001','campaign-004-transport',
      'optimized-concurrency-001']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'EVIDENCE-AUDIT.json')
    args=ap.parse_args();records=[];failures=[];warnings=[]
    for name in RUNS:
        folder=HERE/'runs'/name
        try:
            metadata=json.loads((folder/'metadata.json').read_text())
            hashes=metadata.get('source_hashes',{})
            if not hashes:warnings.append({'run':name,'warning':'No archived source set; legacy worker/controller hashes exist but full code reproduction is weaker.'})
            if metadata.get('workload_sha256'):
                actual=hashlib.sha256((folder/'workload.json').read_bytes()).hexdigest()
                if actual!=metadata['workload_sha256']:raise ValueError('Workload hash differs')
            changed=[]
            for file,expected in hashes.items():
                data=(folder/'source'/file).read_bytes()
                if hashlib.sha256(data).hexdigest()!=expected:changed.append(file)
            if changed:raise ValueError('Archived source hashes differ: '+','.join(changed))
            rows=[json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
            comparisons=analyze(rows)
            if not comparisons:raise ValueError('No complete comparisons')
            config=metadata.get('args',{})
            if config.get('compare_concurrency'):
                expected={f'workers{n}_concurrency{c}' for n in config['workers'] for c in config['compare_concurrency'] if c!=config['concurrency']}
                if set(comparisons)!=expected:raise ValueError('Missing planned concurrency comparison')
            for treatment,result in comparisons.items():
                if result['blocks']<5:raise ValueError(f'{treatment}: fewer than five accepted blocks')
                selected=[r for r in rows if r.get('phase')=='measure' and r['treatment']==treatment
                          and r['block'] in result['accepted_block_ids']]
                if any(r['completed']<=0 or r['seconds']<=0 for r in selected):
                    raise ValueError('Nonpositive timing/count')
                for row in selected:
                    if 'results' in row and len(row['results'])!=row['completed']:
                        raise ValueError('Result count differs from reported completions')
                    if 'battles' in row and len(row['battles'])!=row['completed']:
                        raise ValueError('Battle record count differs from reported completions')
                    jobs=row.get('jobs',[])
                    if jobs:
                        if sum(j['completed'] for j in jobs)!=row['completed']:
                            raise ValueError('Nested job counts differ from reported completions')
                        if len({j['job_id'] for j in jobs})!=len(jobs):
                            raise ValueError('Duplicated nested job IDs')
                        for job in jobs:
                            if 'battles' in job and len(job['battles'])!=job['completed']:
                                raise ValueError('Nested battle records differ from job count')
                            if job.get('attribution_verified') is False:
                                raise ValueError('A nested job failed team attribution')
            records.append({'run':name,'archived_files_verified':len(hashes),
                'comparisons':{t:{k:r[k] for k in ['blocks','accepted_block_ids','invalid_clock_blocks','geometric_mean_speedup','trace_equivalent']}
                               for t,r in comparisons.items()}})
        except Exception as exc:
            failures.append({'run':name,'error':str(exc)})
    live=None
    try:
        folder=HERE/'runs/campaign-004-live-allocation'
        metadata=json.loads((folder/'metadata.json').read_text())
        for file,expected in metadata['source_hashes'].items():
            if hashlib.sha256((folder/'source'/file).read_bytes()).hexdigest()!=expected:
                raise ValueError('Live allocation source hash differs: '+file)
        corpus=Path(metadata['args']['corpus'])/'outcomes.jsonl'
        if hashlib.sha256(corpus.read_bytes()).hexdigest()!=metadata['corpus_sha256']:
            raise ValueError('Live allocation held-out corpus hash differs')
        live=analyze_allocation(folder)
        if live['accounting_failures']:raise ValueError(str(live['accounting_failures']))
        expected=metadata['args']['blocks']*len(metadata['args']['methods'])*4
        if live['trials']!=expected:raise ValueError('Incomplete live allocation campaign')
        if any(len(c['accepted_blocks'])<5 for c in live['comparisons'].values()):
            raise ValueError('Insufficient complete live allocation blocks')
    except Exception as exc:
        failures.append({'run':'campaign-004-live-allocation','error':str(exc)})
    scope=json.loads((HERE/'scope.json').read_text())
    report={'accepted_runs_audited':records,'live_allocation':live,'failures':failures,'warnings':warnings,'scope':scope,
        'limitations':['This verifies archived files and raw timing accounting, not correctness of all simulator mechanics.',
            'Run completion and validation coverage must also be checked against stage manifests and validation artifacts.',
            'Repeated seeded executions and calibrated array repetitions are not independent battle observations.',
            'Restricted-engine trace_equivalent=false is intentional; its damage-event validation is separate.',
            'The original clock-invalid engine runs and failed smoke attempts remain preserved but are not accepted evidence.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'runs_checked':len(records),'failures':failures,'warnings':warnings}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
