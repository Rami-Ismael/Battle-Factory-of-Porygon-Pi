"""Jev, TypeSafe AI's decision model, on the Mask Lab grids' own masks.

    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py prompt --grid meta --task items      # print one request, send nothing
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py fake --grid meta --task items        # fill with the fake chooser, check legality
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py prompts --grid meta --out FILE.json  # one request per task type, for review
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py live --grid meta --task items        # ONE live Jev call on one mask
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py run --grid meta --dry-run            # price a full run, send nothing
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py run --grid meta --limit 5            # the first 5 masks, 4 runs each
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py collect --grid meta                  # append legal fills as candidates
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_grid.py report --grid meta                   # after the grid's battle step

prompt prints the first request of the first mask of a task type on a grid, with its token
estimate. fake fills every mask of that task type (all task types when --task is omitted) with
FakeChooser, which answers with uniform probabilities, and checks each filled team with the Showdown
validator. prompts writes the first request of each task type's first one-field mask, with its
token estimate, for the owner to read before any call. live fills the first one-field mask of a
task type with Jev itself (one paid call; TYPESAFE_API_KEY or the Keychain item "typesafe"), logs
it to ~/vgc-data/jev/<grid>-live.jsonl and checks the filled team. run fills a grid's masks four
times each with Jev (see jev_run.py): --dry-run prices it, --max-calls refuses a run whose estimated
question count is larger, and a rerun resumes. The others send nothing.
See docs/jev-baseline-spec.md.
"""
import argparse
import json

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_fields  # noqa: F401 -- registers the one-question field builders
import jev_fill

import jev_species  # noqa: F401 -- registers the species tournament


def masks(grid: str, task_type: str | None) -> list[jev_fill.Task]:
    return [t for t in jev_fill.tasks(grid) if task_type in (None, t['task'])]


def prompt(a: argparse.Namespace) -> None:
    task = masks(a.grid, a.task)[0]
    chooser = jev_fill.FakeChooser()
    try:
        jev_fill.fill_mask(task, chooser, jev_fill.meta_pastes())
    except NotImplementedError as error:
        if not chooser.requests:
            raise SystemExit(str(error)) from error
    request = chooser.requests[0]
    print(json.dumps(request, indent=1, ensure_ascii=False))
    print(f"\n{a.grid} {task['id']} ({task['task']}, k={task['k']}): {len(chooser.requests)} question(s) asked; "
          f"first question about {jev_fill.estimate_tokens(request)} tokens (limit {jev_fill.TOKEN_LIMIT})")


def fake(a: argparse.Namespace) -> None:
    pipeline, meta = jev_fill.load_pipeline(), jev_fill.meta_pastes()
    counts: dict[str, dict[str, int]] = {}
    for n, task in enumerate(masks(a.grid, a.task)):
        try:
            team, _ = jev_fill.fill_mask(task, jev_fill.FakeChooser(n), meta, run=a.run, seed=n)
            jev_fill.check_legal(pipeline, task, team)
            status = 'legal'
        except NotImplementedError:
            status = 'not built yet'
        except jev_fill.QuestionTooLarge as error:
            status = 'too large'
            print(f"{task['id']} {task['task']} k={task['k']}: {error}")
        except ValueError as error:
            status = 'illegal'
            print(f"{task['id']} {task['task']} k={task['k']}: {str(error)[:200]}")
        counts.setdefault(task['task'], {}).setdefault(status, 0)
        counts[task['task']][status] += 1
    print(json.dumps(counts, indent=1))


TASK_ORDER = ('pokemon', 'items', 'moves', 'abilities', 'natures', 'stats')


def prompts(a: argparse.Namespace) -> None:
    meta, out = jev_fill.meta_pastes(), []
    for task_type in TASK_ORDER:
        task = next(t for t in masks(a.grid, task_type) if t['k'] == 1)
        chooser = jev_fill.FakeChooser()
        jev_fill.fill_mask(task, chooser, meta)
        request = chooser.requests[0]
        text = json.dumps(request, ensure_ascii=False)
        out.append({'task': task_type, 'task_id': task['id'], 'source': task['source'], 'request': request,
                    'questions_in_mask': len(chooser.requests), 'characters': len(text),
                    'tokens': jev_fill.estimate_tokens(request)})
        print(f"{task_type:9} {task['id']}  {len(chooser.requests):2} questions; first: "
              f"{len(next(iter(request['questions'].values()))['criteria']):3} options, ~{out[-1]['tokens']} tokens")
    with open(a.out, 'w') as f:
        json.dump({'grid': a.grid, 'chars_per_token': jev_fill.CHARS_PER_TOKEN, 'prompts': out}, f, ensure_ascii=False)


def live(a: argparse.Namespace) -> None:
    import jev_client
    from pathlib import Path
    task = next(t for t in masks(a.grid, a.task) if t['k'] == 1)
    chooser = jev_client.JevChooser(Path.home() / f'vgc-data/jev/{a.grid}-live.jsonl')
    try:
        team, records = jev_fill.fill_mask(task, chooser, jev_fill.meta_pastes())
    finally:
        chooser.close()
    jev_fill.check_legal(jev_fill.load_pipeline(), task, team)
    for r in records:
        top = sorted(r['probabilities'].items(), key=lambda kv: -kv[1])[:5]
        print(f"{r['path']}: {r['choice']} (confidence {r['confidence']:.3f}); top 5 {[(n, round(p, 3)) for n, p in top]}")
        print(f"  estimate {r['tokens']} tokens")
    u = chooser.usage
    print(f"{a.grid} {task['id']} ({task['task']}): legal; {u['calls']} call(s), {u['input_tokens']} input tokens "
          f"reported, ${chooser.dollars():.5f}")


def run(a: argparse.Namespace) -> None:
    import jev_client
    import jev_run
    pairs, meta = jev_run.pending(a.grid, a.limit), jev_fill.meta_pastes()
    est = jev_run.estimate(pairs, meta)
    dollars = est['tokens'] * jev_client.PRICE_PER_MILLION_INPUT / 1_000_000
    print(f"{a.grid}: {est['pairs']} (mask, run) pairs to send, {est['questions']} questions, "
          f"~{est['tokens']:,} input tokens, ~${dollars:.2f} at ${jev_client.PRICE_PER_MILLION_INPUT}/M (estimate)")
    if a.dry_run or not pairs:
        return
    if est['questions'] > a.max_calls:
        raise SystemExit(f"{est['questions']} questions exceeds --max-calls {a.max_calls}")
    choosers: list[jev_client.JevChooser] = []

    def make() -> jev_fill.Chooser:
        chooser = jev_client.JevChooser(jev_run.DATA / f'{a.grid}-calls.jsonl')
        choosers.append(chooser)
        return chooser

    try:
        counts = jev_run.execute(a.grid, pairs, make, meta, jev_fill.load_pipeline(), a.workers)
    finally:
        for c in choosers:
            c.close()
    used = sum(c.usage['input_tokens'] for c in choosers)
    print(f"{a.grid}: finished {counts}; {sum(c.usage['calls'] for c in choosers)} calls, {used:,} input tokens "
          f"reported, ${used * jev_client.PRICE_PER_MILLION_INPUT / 1_000_000:.4f}")


def collect(a: argparse.Namespace) -> None:
    import jev_report
    print(f'{a.grid}: {jev_report.collect(a.grid)}')


def report(a: argparse.Namespace) -> None:
    import jev_report
    jev_report.print_report(jev_report.report(a.grid))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prompt', 'fake', 'prompts', 'live', 'run', 'collect', 'report'):
        q = sub.add_parser(name)
        q.add_argument('--grid', choices=list(jev_fill.GRIDS), required=True)
        q.add_argument('--task', choices=['pokemon', 'moves', 'items', 'abilities', 'natures', 'stats'])
    sub.choices['fake'].add_argument('--run', choices=['top', 'sample'], default='top')
    sub.choices['prompts'].add_argument('--out', required=True)
    r = sub.choices['run']
    r.add_argument('--dry-run', action='store_true')
    r.add_argument('--limit', type=int, default=0, help='only the first N masks of the grid')
    r.add_argument('--max-calls', type=int, default=200, help='refuse a run estimated to ask more questions')
    r.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    {'prompt': prompt, 'fake': fake, 'prompts': prompts, 'live': live, 'run': run, 'collect': collect,
     'report': report}[a.command](a)
