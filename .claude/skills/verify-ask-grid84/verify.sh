#!/bin/bash
# Drive the 84-field ask grid end to end in a private battle database and keep the evidence.
#   verify.sh doctor                 read-only: is this laptop ready to drive the grid?
#   verify.sh launch                 make a run folder and seed its private database; prints RUN=<folder>
#   verify.sh drive smoke  <RUN>     sample 4 teams at ask 12/24, battle them, report the ask-versus-got error
#   verify.sh drive preflight <RUN>  round trip, copies, separation, conditioning, and 12 control teams battled
#   verify.sh cleanup <RUN>          stop what this run started, drop scratch state, keep the evidence
# AG84_MODEL picks the checkpoint (default gen84_all_ep21).
set -uo pipefail

SKILL="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SKILL/../../.." && pwd)"
SRC="$REPO/code/src"
PY=/tmp/vgc-pilot/.venv/bin/python
MODEL="${AG84_MODEL:-gen84_all_ep21}"
ROOT="$HOME/vgc-data/verify/ask-grid84"
SMOKE_FILE="$HOME/vgc-data/results/stat_points_84_${MODEL}_smoke.json"
fail=0

check() {   # check <label> <command...>: print PASS or FAIL for one condition
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "PASS  $label"; else echo "FAIL  $label"; fail=1; fi
}

doctor() {
  check "staged Python at $PY" test -x "$PY"
  check "staged source /tmp/vgc-pilot/src" test -d /tmp/vgc-pilot/src
  check "behaviour-cloning policy /tmp/bc_100.zip" test -e /tmp/bc_100.zip
  check "VGCPastes pool /tmp/vgc-pilot/vgc-bench/teams/reg_mb" test -d /tmp/vgc-pilot/vgc-bench/teams/reg_mb
  check "fast battle runtime ~/vgc-data/battle-evaluations/runtime" test -d "$HOME/vgc-data/battle-evaluations/runtime"
  check "slot order is shuffled" test "${VGC_SLOT_ORDER:-shuffled}" = shuffled
  local ck
  ck="$(cd "$SRC" && AG84_MODEL="$MODEL" "$PY" -c 'import ask_grid84 as A; print(A.checkpoint_path())' 2>/dev/null | tail -1)"
  check "checkpoint for $MODEL: ${ck:-unresolved}" test -f "$ck"
  check "top-50 list readable" bash -c "cd '$SRC' && '$PY' -c 'import top50, json; assert len(json.load(open(top50.TOP50))) == 50'"
  if [ -e "$SMOKE_FILE" ]; then echo "FAIL  $SMOKE_FILE exists: another smoke run owns it, or a run was not cleaned up"; fail=1
  else echo "PASS  no smoke results file in the way"; fi
  echo "INFO  other jobs on this laptop (battles share the CPU):"
  python3 "$HOME/vgc-data/whats_running.py" 2>/dev/null | sed -n '1,8p' | sed 's/^/      /'
  [ "$fail" = 0 ] && echo "DOCTOR PASS" || echo "DOCTOR FAIL"
  return "$fail"
}

launch() {
  local run="$ROOT/$(date +%Y%m%d-%H%M%S)-$MODEL"
  mkdir -p "$run"
  echo "$MODEL" > "$run/model"
  # matchup_db.py seed looks for the top-50 list inside this checkout and silently builds no top50 set when it is
  # missing; pass top50.TOP50, the list ask_grid84 and the battles use, and require 49 distinct opponents.
  (cd "$SRC" && "$PY" - "$run/matchup.sqlite" <<'EOF'
import json, sys
import matchup_db as M, top50 as T
con = M.connect(sys.argv[1]); print(json.dumps(M.seed(con, top50=T.TOP50)))
n = con.execute("SELECT COUNT(*) FROM team_set WHERE set_name='top50'").fetchone()[0]
print("top50 opponents:", n); sys.exit(0 if n == 49 else 1)
EOF
  ) > "$run/seed.log" 2>&1 || { echo "seed failed or top50 set incomplete: $run/seed.log"; return 1; }
  (cd "$SRC" && "$PY" matchup_db.py --db "$run/matchup.sqlite" stats) > "$run/db_before.json" 2>&1
  echo "RUN=$run"
}

step() {    # step <RUN> <name> <ask_grid84 command> [extra env]: one logged ask_grid84 call
  local run="$1" name="$2" cmd="$3"; shift 3
  (cd "$SRC" && env AG84_SMOKE=1 AG84_MODEL="$(cat "$run/model")" MATCHUP_DB="$run/matchup.sqlite" "$@" \
     "$PY" ask_grid84.py "$cmd" > "$run/$name.log" 2>&1 & echo $! > "$run/$name.pid"; wait $!)
  local code=$?
  echo "$code" > "$run/$name.exit"
  echo "$name exit $code  log $run/$name.log"
  return "$code"
}

drive_smoke() {
  local run="$1"
  [ -e "$SMOKE_FILE" ] && { echo "refusing: $SMOKE_FILE exists and is not this run's"; return 1; }
  touch "$run/owns_smoke_file"
  step "$run" gen gen && step "$run" battle battle && step "$run" report report || return 1
  cp "$SMOKE_FILE" "$run/smoke_results.json"
  (cd "$SRC" && "$PY" matchup_db.py --db "$run/matchup.sqlite" stats) > "$run/db_after.json" 2>&1
  "$PY" - "$run" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); d = json.load(open(run / "smoke_results.json"))
cells = list(d["cells"].values()); c = cells[0]; line = d["summary"]["line"]
checks = {
    "one cell at ask 12/24": len(cells) == 1 and c["numerator"] == 12,
    "4 teams passed Showdown": len(c["pastes"]) == 4,
    "every team scored": len(c.get("y", [])) == 4 and all(0 <= y <= 1 for y in c["y"]),
    "4 x 49 battles recorded in the cell": c.get("battles") == 196,
    "private database grew by 196 battles": json.load(open(run / "db_after.json"))["battles"] - json.load(open(run / "db_before.json"))["battles"] == 196,
    "ask-versus-got error reported": line.get("mse_scored_asks") is not None,
    "report finished": "AG84_REPORT_DONE" in (run / "report.log").read_text(),
}
for k, v in checks.items(): print("PASS " if v else "FAIL ", k)
print(f"win {sum(c['y']) / 4:.3f} at ask 0.500; ask-versus-got error {line['mse_scored_asks']:.4f}")
print("SMOKE PASS" if all(checks.values()) else "SMOKE FAIL")
sys.exit(0 if all(checks.values()) else 1)
EOF
}

drive_preflight() {
  local run="$1"
  [ -e "$SMOKE_FILE" ] && [ ! -e "$run/owns_smoke_file" ] && { echo "refusing: $SMOKE_FILE exists and is not this run's"; return 1; }
  touch "$run/owns_smoke_file"
  step "$run" preflight preflight || return 1
  cp "$SMOKE_FILE" "$run/smoke_results.json"
  grep -E 'AG84_PREFLIGHT_DONE (PASS|FAIL)' "$run/preflight.log"
  grep -q 'AG84_PREFLIGHT_DONE PASS' "$run/preflight.log"
}

cleanup() {
  local run="$1" pid
  for name in gen battle report preflight; do
    [ -f "$run/$name.pid" ] || continue
    pid="$(cat "$run/$name.pid")"
    kill "$pid" 2>/dev/null && echo "stopped $name ($pid)"
    pgrep -f "battle-evaluations/[0-9-]*-$pid-" | while read -r w; do kill "$w" && echo "stopped battle worker $w"; done
  done
  if [ -e "$run/owns_smoke_file" ] && [ -e "$SMOKE_FILE" ]; then
    [ -e "$run/smoke_results.json" ] || cp "$SMOKE_FILE" "$run/smoke_results.json"
    rm -f "$SMOKE_FILE" "${SMOKE_FILE%.json}.tmp" && echo "removed $SMOKE_FILE (copy kept in $run)"
  fi
  rm -f "$run/matchup.sqlite" "$run/matchup.sqlite-wal" "$run/matchup.sqlite-shm" "$run/owns_smoke_file"
  echo "evidence kept in $run:"; ls "$run"
}

case "${1:-}" in
  doctor) doctor ;;
  launch) launch ;;
  drive) case "${2:-}" in smoke) drive_smoke "$3" ;; preflight) drive_preflight "$3" ;; *) echo "drive smoke|preflight <RUN>"; exit 2 ;; esac ;;
  cleanup) cleanup "$2" ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
