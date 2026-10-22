"""Derive an auditable summary without overwriting original run summaries."""
import argparse
import json
import statistics
from pathlib import Path
from run import analyze


def main():
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+',type=Path);args=ap.parse_args()
    for folder in args.runs:
        rows=[json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
        measured=[r for r in rows if r.get('phase')=='measure']
        data=analyze(rows)
        for name,result in data.items():
            relevant=[r for r in measured if r['treatment']==name and r['block'] in result['accepted_block_ids']]
            baseline=[r for r in relevant if r['variant']=='baseline']
            treatment=[r for r in relevant if r['variant']==name]
            result['completed_battles']=sum(r['completed'] for r in relevant)
            result['baseline_bps']=sum(r['completed'] for r in baseline)/sum(r['seconds'] for r in baseline)
            result['treatment_bps']=sum(r['completed'] for r in treatment)/sum(r['seconds'] for r in treatment)
            external=[s['external_cpu_percent'] for r in relevant for s in r.get('telemetry',[]) if 'external_cpu_percent'in s]
            if external:result['external_cpu_percent']={'min':min(external),'median':statistics.median(external),'max':max(external)}
        output={'run':str(folder),'measured_battles':sum(r['completed'] for r in measured),
                'trials':len(measured),'comparisons':data,'note':'Analysis of available complete blocks; process completion must be checked separately.'}
        (folder/'analysis-v2.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(output),flush=True)


if __name__=='__main__':main()
