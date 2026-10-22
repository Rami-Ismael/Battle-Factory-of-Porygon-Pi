#!/usr/bin/env python3
"""The fine-tuned pretrained text diffusion model (Fast-dLLM v2 1.5B + LoRA, code repo src/textdiff.py) on the
same 270 masks the Mask Lab gave Ling, the random fills and the scratch diffusion decoder.

    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python mask_lab_textdiff.py generate [--limit N] [--family F] [--guidance 2]
    /tmp/vgc-pilot/.venv/bin/python mask_lab_textdiff.py collect      # spreads, Showdown validation, candidates.json
    /tmp/vgc-pilot/.venv/bin/python mask_lab_battle.py                # scores the new candidates (old panels are cached)
    /tmp/vgc-pilot/.venv/bin/python mask_lab_textdiff.py report       # per family: legality, vs original / random / diffusion

How the text model fills a mask. The model writes teams left to right in 32-token blocks, so a masked field is
filled by *constrained infilling*: the team is written in the training serialisation (species @ item / Ability /
Nature / four moves; known moves first in alphabetical order, masked moves last), the text up to the masked field is
teacher-forced as the prefix, the model generates the field's text (one attempt, cut at the line end; a whole
masked Pokémon is cut at the blank line), and the filled field joins the prefix for the next mask. The model sees
every known field before the mask and nothing after it (the scratch decoder sees both sides). Prompt: the training
prompt with ask 0.5, classifier-free guidance 2 (the owner's selected diffusion profile), temperature 1, top-p 1,
threshold 0.9. Stat Points are not text: masked spreads are a same-species corpus spread or a random legal spread,
exactly as the scratch arm. No repair, no retries; unmasked fields are unchanged by construction.
"""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/prompt-ab'
WORK = Path.home() / 'vgc-data/mask_lab_diffusion'          # baseline/manifest.json: the 5 starts × 270 masks, decoder format
OUT = Path.home() / 'vgc-data/mask_lab_textdiff'
REPO = Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
ADAPTER = Path.home() / '.local/share/vgc-pilot-runtime/textdiff/lora_r32/final'
STATS = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']
SEED = 20261008
KIND = 'textdiff'


def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def mask_map(paths):
    """{slot: {'species','item','ability','nature','evs', 'moves': {m,...}}} from the manifest paths."""
    mm = {}
    for p in paths:
        s = p[0]; d = mm.setdefault(s, {})
        if len(p) == 1: d.update(species=True, item=True, ability=True, nature=True, evs=True); d['moves'] = {0, 1, 2, 3}
        elif p[1] == 'moves': d.setdefault('moves', set()).add(p[2])
        else: d[p[1]] = True
    return mm


def parse_block(block):
    lines = block.split('\n')
    head = lines[0] if lines else ''
    sp, _, it = head.partition(' @ ')
    ab = next((l[len('Ability: '):] for l in lines if l.startswith('Ability: ')), None)
    na = next((l[:-len(' Nature')] for l in lines if l.endswith(' Nature')), None)
    mv = [l[2:].strip() for l in lines if l.startswith('- ')]
    if not sp.strip() or ab is None or na is None or len(mv) != 4 or not all(mv): return None
    return dict(species=sp.strip(), item=it.strip(), ability=ab.strip(), nature=na.strip(), moves=mv)


# ---------------------------------------------------------------- generate (venv-dlm)
def generate(args):
    sys.path.insert(0, str(REPO / 'src'))
    import torch, textdiff_port as P, textdiff as T
    from peft import PeftModel
    tok, m = P.load()
    m = PeftModel.from_pretrained(m, str(ADAPTER)).merge_and_unload().eval()
    chat = lambda t: tok.apply_chat_template([{"role": "system", "content": T.SYSTEM}, {"role": "user", "content": t}],
                                             add_generation_prompt=True, return_tensors="pt").to(P.DEV)
    cond, unc = chat(T.prompt_text(args.ask)), chat(T.prompt_text(None))
    manifest = json.loads((WORK / 'baseline/manifest.json').read_text())
    trials = [t for t in manifest['trials'] if not args.family or t['task'] == args.family]
    if args.limit: trials = trials[:args.limit]
    dest = OUT / 'raw'; dest.mkdir(parents=True, exist_ok=True)
    (OUT / 'generation-config.json').write_text(json.dumps(dict(
        model=P.MODEL_ID, adapter=str(ADAPTER), ask=args.ask, guidance=args.guidance, temperature=1.0, top_p=1.0,
        threshold=0.9, block_size=32, attempts=1, seed=SEED, method='constrained left-to-right infilling, one field at a time',
        stat_points='same-species corpus spread or random legal spread (collect step), not model output'), indent=1))

    def gen_text(prefix_text, max_new):
        """Generated text after the prefix, and whether the writer stopped (emitted <|im_end|>) within the budget."""
        ptoks = tok(prefix_text, add_special_tokens=False, return_tensors='pt').input_ids.to(P.DEV)
        c = torch.cat([cond, ptoks], 1); u = torch.cat([unc, ptoks], 1)
        if args.guidance == 1.0:
            out = m.generate(c, max_new_tokens=max_new, threshold=0.9, block_size=32, small_block_size=8, temperature=1.0, top_p=1.0)
        else:
            out = P.generate_cfg(m, c, u, args.guidance, max_new_tokens=max_new, threshold=0.9, temperature=1.0, top_p=1.0)
        new = out[0, c.shape[1]:]
        stopped = bool((new == 151645).any())
        return tok.decode(new, skip_special_tokens=True), stopped

    for i, trial in enumerate(trials):
        f = dest / (trial['id'] + '.json')
        if f.exists(): continue
        torch.manual_seed(SEED + i)
        start = manifest['starts'][trial['start']]
        mm = mask_map(trial['paths'])
        t0 = time.perf_counter(); fills = {}; raw = []; error = None
        done = []                                                   # finished slot texts
        for s, slot in enumerate(start):
            masked = mm.get(s, {})
            sep = ('\n\n'.join(done) + '\n\n') if done else ''
            if masked.get('species'):                               # whole Pokémon: generate the block up to the blank line
                text, stopped = gen_text(sep, 160); raw.append(dict(slot=s, field='slot', text=text[:400]))
                block = text.split('\n\n')[0].strip(); parsed = parse_block(block) if ('\n\n' in text or stopped) else None
                if parsed is None: error = f'slot {s}: unparseable block'; break
                fills[s] = parsed; done.append(block); continue
            cur = []; filled = dict(moves=[])
            # head line
            if masked.get('item'):
                head = slot['species'] + ' @'                      # no trailing space: the item token carries its own leading space
                text, stopped = gen_text(sep + head, 64); raw.append(dict(slot=s, field='item', text=text[:120]))
                if '\n' not in text and not stopped: error = f'slot {s}: item line not terminated'; break
                filled['item'] = text.split('\n')[0].strip(); cur.append(head + ' ' + filled['item'])
            else:
                cur.append(slot['species'] + (f" @ {slot['item']}" if slot['item'] else ''))
            # the remaining lines, in the training order
            known = sorted([mv for j, mv in enumerate(slot['moves']) if j not in masked.get('moves', ()) and mv], key=norm)
            plan = [('ability', None if masked.get('ability') else f"Ability: {slot['ability']}"),
                    ('nature', None if masked.get('nature') else f"{slot['nature']} Nature")]
            plan += [('move', f'- {mv}') for mv in known] + [('move', None)] * len(masked.get('moves', ()))
            for kind, line in plan:
                if line is not None: cur.append(line); continue
                text, stopped = gen_text(sep + '\n'.join(cur) + '\n', 64); value = text.split('\n')[0].strip()
                raw.append(dict(slot=s, field=kind, text=text[:120]))
                if '\n' not in text and not stopped: error = f'slot {s}: {kind} line not terminated'; break
                if kind == 'ability':
                    if not value.startswith('Ability: ') or not value[9:].strip(): error = f'slot {s}: ability line malformed'; break
                    filled['ability'] = value[9:].strip()
                elif kind == 'nature':
                    if not value.endswith(' Nature') or not value[:-7].strip(): error = f'slot {s}: nature line malformed'; break
                    filled['nature'] = value[:-7].strip()
                else:
                    if not value.startswith('- ') or not value[2:].strip(): error = f'slot {s}: move line malformed'; break
                    filled['moves'].append(value[2:].strip())
                cur.append(value)
            if error: break
            fills[s] = filled; done.append('\n'.join(cur))
        row = dict(trial=trial['id'], start=trial['start'], task=trial['task'], k=trial['k'], paths=trial['paths'],
                   fills={str(k): v for k, v in fills.items()}, error=error, raw=raw, seconds=time.perf_counter() - t0)
        f.write_text(json.dumps(row, indent=1))
        if i % 10 == 0 or args.limit: print(f"{i}/{len(trials)} {trial['task']} k={trial['k']} {'ok' if not error else error} {row['seconds']:.1f}s", flush=True)


# ---------------------------------------------------------------- collect (project venv)
def collect(args):
    sys.path.insert(0, '/tmp/vgc-pilot/src'); sys.path.insert(0, str(REPO / 'scripts')); sys.path.insert(0, str(HERE))
    import numpy as np
    import llm_baseline as B
    import asked_vs_got as X
    import hps_generate
    import mask_lab_prepare as M
    from corpus import norm as cnorm
    corpus, V, C, look, spreads = X.setup()
    h, _ = B.runtime(); validator = h.Validator()
    manifest = json.loads((WORK / 'baseline/manifest.json').read_text())
    tasks = {t['id']: t for t in json.loads((RUN / 'eval2.tasks.json').read_text())['tasks']}
    rng = np.random.default_rng(SEED)
    (OUT / 'completions').mkdir(parents=True, exist_ok=True)
    stats = dict(valid=0, invalid=0, missing=0); rows = {}; new = []
    try:
        for trial in manifest['trials']:
            f = OUT / 'raw' / (trial['id'] + '.json')
            if not f.exists(): stats['missing'] += 1; continue
            g = json.loads(f.read_text()); start = manifest['starts'][trial['start']]
            team = copy.deepcopy(start); error = g['error']
            if not error:
                mm = mask_map(trial['paths'])
                try:
                    for s, masked in mm.items():
                        fill = g['fills'][str(s)]
                        if masked.get('species'):
                            team[s].update(species=fill['species'], item=fill['item'], ability=fill['ability'], nature=fill['nature'], moves=list(fill['moves']))
                        else:
                            for k in ('item', 'ability', 'nature'):
                                if masked.get(k): team[s][k] = fill[k]
                            for j, mi in enumerate(sorted(masked.get('moves', ()))):
                                team[s]['moves'][mi] = fill['moves'][j]
                        if masked.get('evs') or masked.get('species'):
                            pl = spreads.get(cnorm(team[s]['species']))
                            team[s]['evs'] = dict(pl[int(rng.integers(len(pl)))]) if pl else hps_generate.sample_spread(rng)
                    B.validate_shape(team)
                    if B.masked(team, trial['paths']) != B.masked(start, trial['paths']): raise ValueError('unmasked field changed')
                    error = validator(B.paste(team, h))
                except (ValueError, KeyError, IndexError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
            valid = error is None
            row = dict(trial=trial['id'], valid=valid, supported=True, error=error, team=team if valid else None, seconds=g['seconds'])
            (OUT / 'completions' / (trial['id'] + '.json')).write_text(json.dumps(row, indent=1))
            rows[trial['id']] = dict(valid=valid, error=(error or '')[:160], seconds=g['seconds'])
            stats['valid' if valid else 'invalid'] += 1
            if valid:
                t2 = [dict(slot=i + 1, species=s['species'], item=s['item'], ability=s['ability'], nature=s['nature'],
                           moves=list(s['moves']), statPoints=dict(s['evs'])) for i, s in enumerate(team)]
                new.append(dict(label=f"{KIND}:{trial['id']}", kind=KIND, task_id=trial['id'], source=tasks[trial['id']]['source'],
                                team=t2, paste=M.to_paste(t2)))
    finally:
        validator.close()
    cands_path = RUN / 'candidates.json'
    cands = [c for c in json.loads(cands_path.read_text())['candidates'] if c['kind'] != KIND] + new
    cands_path.write_text(json.dumps({'candidates': cands}, indent=1) + '\n')
    (RUN / f'{KIND}_rows.json').write_text(json.dumps(rows, indent=1) + '\n')
    print(KIND, 'completions', stats, '| candidates now', len(cands))


# ---------------------------------------------------------------- report (any venv)
def report(args):
    import random
    panels = json.loads((RUN / 'panels.json').read_text())['panels']
    tasks = json.loads((RUN / 'eval2.tasks.json').read_text())
    rows = {k: json.loads((RUN / f'{k}_rows.json').read_text()) for k in (KIND, 'diffusion')}
    score = lambda label: (panels.get(label) or {}).get('score')
    fam = {}
    for t in tasks['tasks']:
        tid, src = t['id'], t['source']
        o = score(f'orig:{src}')
        fam.setdefault(t['task'], []).append(dict(start=src, k=t['k'], original=o, random=score(f'random:{tid}'),
                                                  diffusion=score(f'diffusion:{tid}'), text=score(f'{KIND}:{tid}'),
                                                  text_legal=rows[KIND].get(tid, {}).get('valid', False),
                                                  diff_legal=rows['diffusion'].get(tid, {}).get('valid', False)))

    def cluster(pairs):
        """Equal-start-weighted mean of a paired difference with a bootstrap over the starts."""
        by = {}
        for s, v in pairs: by.setdefault(s, []).append(v)
        if not by: return None
        means = [sum(v) / len(v) for v in by.values()]
        rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
        return dict(mean=sum(means) / len(means), lo=draws[100], hi=draws[-100], n=sum(len(v) for v in by.values()))
    fmt = lambda c: '—' if not c else f"{100*c['mean']:+.2f} [{100*c['lo']:+.1f},{100*c['hi']:+.1f}] n={c['n']}"
    groups = list(fam.items()) + [('all', [r for v in fam.values() for r in v])]
    out = {}
    print(f"{'family':10s} {'legal text':>11s} {'legal diff':>11s} {'text−orig (pp)':>26s} {'text−random':>26s} {'text−diffusion':>26s} {'net text':>24s} {'net diffusion':>24s}")
    for family, rs in groups:
        lt = sum(r['text_legal'] for r in rs); ld = sum(r['diff_legal'] for r in rs)
        vo = cluster([(r['start'], r['text'] - r['original']) for r in rs if r['text'] is not None and r['original'] is not None])
        vr = cluster([(r['start'], r['text'] - r['random']) for r in rs if r['text'] is not None and r['random'] is not None])
        vd = cluster([(r['start'], r['text'] - r['diffusion']) for r in rs if r['text'] is not None and r['diffusion'] is not None])
        net_t = cluster([(r['start'], (r['text'] if r['text'] is not None else r['original']) - r['original']) for r in rs if r['original'] is not None])
        net_d = cluster([(r['start'], (r['diffusion'] if r['diffusion'] is not None else r['original']) - r['original']) for r in rs if r['original'] is not None])
        print(f"{family:10s} {lt:4d}/{len(rs):<6d} {ld:4d}/{len(rs):<6d} {fmt(vo):>26s} {fmt(vr):>26s} {fmt(vd):>26s} {fmt(net_t):>24s} {fmt(net_d):>24s}")
        out[family] = dict(n=len(rs), legal_text=lt, legal_diffusion=ld, vs_original=vo, vs_random=vr, vs_diffusion=vd, net_gain_text=net_t, net_gain_diffusion=net_d)
    (RUN / f'{KIND}_report.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate'); g.add_argument('--limit', type=int); g.add_argument('--family'); g.add_argument('--guidance', type=float, default=2.0); g.add_argument('--ask', type=float, default=0.5)
    sub.add_parser('collect')
    sub.add_parser('report')
    a = ap.parse_args()
    {'generate': generate, 'collect': collect, 'report': report}[a.cmd](a)
