"""Audit the rule table of every in-place fill: is each verdict right, and which rules break, per model and prompt.

Owner, 2026-11-25 (#239): a deep check of what broke, against the rule table, to make sure the judging works. For each
fill in D/<prompt>/<model>/fills.jsonl:

    rejudge      judge the saved values again from scratch (tokenizer, canvas, Showdown); the verdict, first failed
                 step, reasons and every rule must come out as saved
    legal_clean  a legal fill breaks no rule; an illegal one breaks at least one
    step_rule    the first failed step has its rule: parses -> does not parse, visible -> changed a visible value,
                 regmb -> a Reg M-B name rule, showdown -> a Showdown-confirmed or team rule
    team_vs_sd   on every team Showdown saw, our own team checks (item twice, species twice, same move twice, Stat
                 Points over 66, a Stat Point over 32) against Showdown's messages for the same rule
    unmapped     Showdown messages that map to no rule ('other Showdown error'), listed whole
    examples     two fills per broken rule, with the values or messages that broke it, to read by eye

    audit --dir D [--prompts P,...] [--keys K,...] --out F.json     writes F.json and prints a Markdown summary

Run with venv-dlm from code/scripts.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inplace_prompt_grid as G

PROMPTS = ('inplace_header', 'inplace_instructions', 'inplace_header_example', 'inplace_example')
TEAM_RULES = {'item used more than once': r'Item Clause', 'species used more than once': r'Species Clause',
              'same move twice on one Pokémon': r'multiple copies of', 'Stat Points over 66': r'total Stat Points',
              'a Stat Point over 32': r'more than \d+ Stat Points in'}
SHOWDOWN_STEP_RULES = set(G.SHOWDOWN_ONLY) | set(TEAM_RULES) | {'wrong Mega Stone'}


def audit_run(directory: Path, prompt: str, key: str, validate: Any, names: dict[str, frozenset[str]],
              tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import count_masked_diffusion_text as C
    import masked_diffusion_memory_bench as B
    candidate = C.candidate_by_key(key)
    tokenizer = B.load_tokenizer(B.candidate_info(candidate))
    start_id = G.start_token_id(tokenizer, candidate.sampler)
    rows = G.read_fills(directory / prompt / key / 'fills.jsonl')
    out: dict[str, Any] = {'fills': len(rows), 'rejudge_mismatch': [], 'legal_but_broken': [], 'illegal_but_clean': [],
                           'step_without_rule': [], 'team_vs_showdown': {}, 'unmapped': Counter(),
                           'examples': defaultdict(list), 'broken': Counter(), 'judged': Counter()}
    agree = {rule: Counter() for rule in TEAM_RULES}
    prompt_obj = G.PROMPTS[prompt]
    for row in rows:
        c = G.canvas(tasks[row['id']], prompt_obj, tokenizer, candidate.mask_id, start_id)
        if prompt_obj.mode == 'freetext':
            import masked_diffusion_text as D
            raw_tail, _, _, _ = D.trim_answer_ids(tokenizer, row['slot_ids'][0])
            again = G.judge_freetext(tasks[row['id']], c['built'].answer_prefix + raw_tail, validate, names)
        else:
            again = G.judge(tasks[row['id']], c['built'], row['values'], validate, names)
        team = again.pop('team', None)
        if not all(again[k] == row[k] for k in ('legal', 'failed', 'reasons', 'rules', 'showdown_errors')):
            out['rejudge_mismatch'].append(row['id'])
        check_rules(row, again, out)
        if team is not None and again['showdown_errors'] is not None:
            check_against_showdown(row['id'], team, again, out, agree)
    out['team_vs_showdown'] = {rule: {f"ours {'yes' if a[1] else 'no'}, Showdown {'yes' if a[3] else 'no'}": n
                                      for a, n in c.items()} for rule, c in agree.items()}
    out['unmapped'] = dict(out['unmapped'].most_common(15))
    out['examples'] = dict(out['examples'])
    out['broken'], out['judged'] = dict(out['broken']), dict(out['judged'])
    return out


def check_rules(row: dict[str, Any], again: dict[str, Any], out: dict[str, Any]) -> None:
    """Legal iff clean, the first failed step has its rule, and the per-rule counts and examples."""
    rules = again['rules']
    broken = [r for r, v in rules.items() if v]
    for r, v in rules.items():
        out['judged'][r] += v is not None
        if v:
            out['broken'][r] += 1
            if len(out['examples'][r]) < 2:
                out['examples'][r].append(example(row, again))
    if again['legal'] and broken:
        out['legal_but_broken'].append((row['id'], broken))
    if not again['legal'] and not broken:
        out['illegal_but_clean'].append(row['id'])
    need = {'parses': {'does not parse as a team'}, 'visible': {'changed a visible value'},
            'regmb': set(G.NAME_RULES.values()), 'showdown': SHOWDOWN_STEP_RULES}.get(again['failed'], set())
    if need and not need & set(broken):
        out['step_without_rule'].append((row['id'], again['failed']))


def check_against_showdown(task_id: str, team: list[dict[str, Any]], again: dict[str, Any], out: dict[str, Any],
                           agree: dict[str, Counter]) -> None:
    """Our team rules, a second reading of them, and Showdown's message for the same rule; unmapped messages."""
    import re
    errors = again['showdown_errors']
    for rule, pattern in TEAM_RULES.items():
        sd = any(re.search(pattern, e) for e in errors)
        mine = _own_check(team, rule)
        if mine != bool(again['rules'][rule]):
            out['step_without_rule'].append((task_id, f'second reading disagrees on {rule}'))
        agree[rule][('ours', mine, 'showdown', sd)] += 1
    for e in errors:
        if not any(re.search(p, e) for p, _ in G.SHOWDOWN_RULES):
            out['unmapped'][e] += 1


def _own_check(team: list[dict[str, Any]], rule: str) -> bool:
    """The same team rules, written again here without rule_breaks, as a second reading."""
    ids = lambda xs: [G.to_id(x) for x in xs if G.to_id(x)]
    if rule == 'item used more than once':
        items = ids(m['item'] for m in team)
        return len(items) != len(set(items))
    if rule == 'species used more than once':
        _, base = G.dex()
        species = [base.get(s, s) for s in ids(m['species'] for m in team)]
        return len(species) != len(set(species))
    if rule == 'same move twice on one Pokémon':
        return any(len(ids(m['moves'])) != len(set(ids(m['moves']))) for m in team)
    if rule == 'Stat Points over 66':
        return any(sum(m['statPoints'].values()) > 66 for m in team)
    return any(v > 32 for m in team for v in m['statPoints'].values())


def example(row: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    named = [f'{p}={v!r}' for p, v in zip(row['paths'], row['values'], strict=True)][:8]
    return {'id': row['id'], 'task': f"{row['task']} {row['k']}", 'filled': named,
            'showdown': (verdict['showdown_errors'] or [])[:3], 'reasons': verdict['reasons'][:3]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--dir', type=Path, required=True)
    a.add_argument('--prompts', default=','.join(PROMPTS))
    a.add_argument('--keys', default=','.join(G.SMALL_FOUR))
    a.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from showdown_validator import Validator
    names, validate = G.vocabulary(), Validator()
    tasks = {t['id']: t for t in G.grid_tasks()}
    result: dict[str, Any] = {}
    try:
        for prompt in args.prompts.split(','):
            for key in args.keys.split(','):
                if (args.dir / prompt / key / 'fills.jsonl').exists():
                    result[f'{prompt}/{key}'] = r = audit_run(args.dir, prompt, key, validate, names, tasks)
                    print(f"{prompt}/{key}: {r['fills']} fills, rejudge mismatches {len(r['rejudge_mismatch'])}, "
                          f"legal but broken {len(r['legal_but_broken'])}, illegal but clean "
                          f"{len(r['illegal_but_clean'])}, step without rule {len(r['step_without_rule'])}", flush=True)
    finally:
        validate.close()
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=list) + '\n')
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
