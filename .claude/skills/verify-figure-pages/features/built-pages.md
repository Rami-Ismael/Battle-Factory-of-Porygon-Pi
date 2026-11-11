# Built pages

The page a reader opens is a built file. It must be the one its template and data produce today, so an edit to the template that was never rebuilt does not ship a stale page.

## Sub-features

- `build-fresh`: running the build changes nothing that git can see.

## How to get to it (user POV)

- Open `artifacts/wrong-mega-stone-validation-loss/index.html`, built from `template.html`.
- Open `artifacts/wrong-mega-stone-validation-loss/storyboard.html`, built from `storyboard.template.html`.

## Driving it with build.py and git

Preconditions:

- The folder's `build.py` reads only files inside the repo. This one reads `probe_worst_check.json` and `per_team_validation_loss.json` beside it.

- **Record, build, compare.** Run:

  ```
  F=artifacts/wrong-mega-stone-validation-loss
  before="$(git status --porcelain -- "$F")"
  python3 "$F/build.py"
  after="$(git status --porcelain -- "$F")"
  [ "$before" = "$after" ] && echo "PASS build-fresh" || { echo "FAIL build-fresh"; git status --porcelain -- "$F"; }
  ```

  `PASS build-fresh` means the built pages were already current. On `FAIL`, the listed built files changed: they were stale, and they now belong in the same commit as the template or data edit.

## Gotchas

- The build never reruns the replay. `probe_worst_check.py` needs the frozen run code on the owner's laptop and loads a model, so check for other sessions' jobs before running it.
- A build script that reads files outside the repo, such as `~/vgc-data`, can only be checked on the laptop. Report `build-fresh` as not run for that page elsewhere.
- A page with no `build.py` is hand-written. `build-fresh` does not apply to it.
