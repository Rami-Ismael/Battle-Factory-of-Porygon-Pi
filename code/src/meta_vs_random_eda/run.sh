#!/bin/sh
# Meta vs random teams EDA. Writes intermediate JSON to ./out and the page to code/docs/meta-vs-random-eda.html.
set -e
H="$(cd "$(dirname "$0")" && pwd)"; O="$H/out"; mkdir -p "$O"; cp "$H/template.html" "$O"/; export PYTHONPATH="$H"
sqlite3 -csv ~/vgc-data/matchup_regmb.sqlite "
with t50 as (select team_id from team_set where set_name='top50')
select m.row_team, t.origin, sum(m.wins), sum(m.battles), count(*)
from matchup m join team t on t.team_id=m.row_team
where m.policy_id=1 and m.col_team in (select team_id from t50) and t.legal=1
group by m.row_team" > "$O/scores.csv"
for s in a_compare b_fills c_moves d_species e_all; do python3 "$H/$s.py" "$O"; done
python3 "$H/f_page.py" "$O" "$H/../../docs/meta-vs-random-eda.html"
