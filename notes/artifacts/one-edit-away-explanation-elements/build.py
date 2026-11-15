"""Build the explanation-elements site: a map view and one hash-routed view per element.

Writes two files from the same body: a full local index.html, and a body-only page for publishing.
"""
import glob
import json
import sys
from html import escape
from pathlib import Path

here = Path(__file__).parent
site_dir = Path(sys.argv[1])
pairs = []
for f in sorted(glob.glob(str(site_dir / "[0-9][0-9]-*.json"))):
    d = json.loads(Path(f).read_text())
    d["slug"] = Path(f).stem[3:]
    pairs.append(d)
data = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
social = json.dumps(json.loads((site_dir / "social.json").read_text()), ensure_ascii=False).replace("</", "<\\/")

# The owner's checklist of 2026-11-13, one row per checkbox that a subagent handled: where it sits, what came back, where to read it.
CHECKLIST = [
    ("Make connections", "Define what is the reader default", "Defined: a small step in the input gives a small change in the score, the assumption a Gaussian process is built on.", "contrast-reader-default"),
    ("Make connections", "Look like x but differ", "A team looks like a point on the Pest Control benchmark, but one whole-Pokemon swap moves it 0.65 of the spread between teams. One candidate misdescribed the benchmark and is fixed.", "looks-like-but-differs"),
    ("Make connections", "Borrow from a sibling method", "The siblings are now the random-walk check in NAS-Bench-101, Weinberger's landscape measure and the Pest Control step.", "borrow-sibling-method"),
    ("Make connections", "An everyday picture", "Three new pictures that each carry the edit, the score change and the luck. Recommended: the football club.", "everyday-picture"),
    ("Make connections", "Epistasis is a good one", "Marked as your pick and defined from a source that was opened. For Pokemon it stays a hypothesis.", "neighbouring-field-term"),
    ("Provide context", "Design decisions: the starting point", "The decision is written three ways. The finished run started from tournament-ready teams, not from the top 50. No current number exists for random teams.", "design-decisions"),
    ("Provide context", "Design decisions: what can we say about noise", "Two scores of one unchanged team differ by 3.4 to 3.7 points. The correction is explained with the three limits it leaves.", "design-decisions"),
    ("Provide context", "Technical constraints: 1,400 one-move neighbours", "Your numbers hold: 61.2 moves on average gives 1,372 neighbours. Added the cost, about 9 seconds a score.", "technical-constraints"),
    ("Provide context", "etc", "Historical reasons, implications and specific examples are cleared of retired numbers. The examples are three real edits on one team.", "specific-examples"),
    ("Talk about the subject", "The bigger picture, left empty", "Three pictures to choose from. Recommended: the benchmark comparison.", "bigger-picture"),
    ("Talk about the subject", "History: the Weinberger sentence", "Your sentence is kept word for word and marked as your pick, now with its full citation.", "field-history"),
    ("Talk about the subject", "Why: González-Duque, Section 4.2.1", "That section is about problem complexity, not smoothness. The step to smoothness is marked as your inference.", "reasons-justifications"),
    ("Admit opinion and perspective", "Alternatives and counter-examples, left empty", "Three admissions to choose from. Recommended: half of your own benchmark table is rough.", "counter-examples"),
    ("Other tasks", "A version for each site in Where to post", "12 versions, each with three opening lines and a character count.", "social"),
    ("Other tasks", "A source that smoothness decides which methods look good", "Say that ruggedness can change which method looks best. Three checked sources.", "answers"),
    ("Other tasks", "The full Weinberger citation and its landscape", "Biological Cybernetics 63, 1990, tested on Kauffman's N-k model. Your sentence is accurate.", "answers"),
    ("Other tasks", "Why benchmarks are mostly smooth", "Name the measured ones. Papers explain why benchmarks are simple, not why they are smooth.", "answers"),
    ("Other tasks", "Does one move depend on the other five Pokemon", "Keep it as a hypothesis. A 29-minute measurement would settle it.", "answers"),
    ("Other tasks", "Your paragraph on mathematics", "Kept as you wrote it and checked line by line. Reinforcement learning was studied, never trained.", "what-the-project-taught-me"),
]


def top50_panel():
    """The study that edits the 49 top-50 teams themselves (issue #317): its registered sizes beside the check made after the result."""
    res = json.loads((site_dir / "top50-result.json").read_text())
    span = lambda value, ends: f"{value:.2f} [{ends[0]:.2f}, {ends[1]:.2f}]"
    rows = "".join(f'<tr><th scope="row">{escape(row["label"])}</th><td>{span(row["size"], row["size_ci"])}</td><td>{"yes" if row["above_zero"] else "not shown"}</td>'
                   f'<td>{span(row["check"], row["check_ci"])}</td><td>{row["share"]:.2f}</td><td>{row["against_pest_control_step"]}</td><td>{span(row["signed"], row["signed_ci"])}</td>'
                   f'<td>{row["other_groups"]["outside_top50"]["size"]:.2f}</td><td>{row["other_groups"]["search_found"]["size"]:.2f}</td></tr>' for row in res["rows"])
    rank = res["rank_correlation"]
    return ('<div class="flags rerun" id="top50"><h2>Top-50 result: one edit of each kind on the 49 top-50 teams themselves</h2>'
            '<p><b>Checked 2026-11-14.</b> The run matched its plan, 1,078 scores of 384 battles. On top-50 teams a whole-Pokemon swap, a move edit and an item edit are above zero. '
            'Stat alignment and Stat Points edits are not shown. Only the whole-Pokemon swap is larger than a Pest Control step. Random edits lower win rate on average for four kinds.</p>'
            '<div class="scroll"><table><thead><tr><th>Edit</th><th>Change as planned, points, 95% interval</th><th>Above zero</th><th>Check: against all four unchanged scores</th>'
            '<th>Share of the spread between teams</th><th>Against one Pest Control step</th><th>Average change, points</th><th>Tournament-ready teams, change</th><th>Search-found teams, change</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p class="note">The plan measured every edit against the team\'s first score, which for top-50 teams came out 0.78 points below its replays. Edits mostly lower win rate, so the planned sizes read low. '
            'The check column uses all four unchanged scores of a team and was computed after the result, so it is exploratory. It moves the ability edit to above zero and removes the apparent gap to the tournament-ready teams. '
            f'Rank correlation between how much a team wins and how far its edits move it: {rank["top50_only"]:.2f}, interval {rank["top50_only_ci95"][0]:.2f} to {rank["top50_only_ci95"][1]:.2f}, as planned, '
            'and −0.01, interval −0.28 to 0.26, with a rank that shares no seed with the edits. No trend is shown. '
            f'The spread here is pooled over three groups, {res["spread_between_teams_points"]:.2f} points. Details in <a href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/317">issue #317</a>.</p></div>')


def checklist_panel():
    groups = {}
    for group, box, result, target in CHECKLIST:
        groups.setdefault(group, []).append(f'<div class="chk" data-go="{target}"><a href="#{target}">{escape(box)}</a><span>{escape(result)}</span></div>')
    blocks = "".join(f'<div class="chkgroup"><h3>{escape(group)}</h3>{"".join(rows)}</div>' for group, rows in groups.items())
    return (f'<div class="panel" id="checklist"><h2>Your checklist of 2026-11-13: {len(CHECKLIST)} checkboxes, one subagent each</h2>'
            '<p class="hint">Each row is one checkbox. It opens what that subagent brought back. No box in your note was ticked.</p>'
            f'<div class="chkgrid">{blocks}</div></div>')


def answers_panel():
    cards = []
    for answer in json.loads((site_dir / "answers.json").read_text()):
        sources = "".join(f"<li>{escape(source)}</li>" for source in answer["sources"])
        cards.append(f'<article><h3>{escape(answer["question"])}</h3><p>{escape(answer["verdict"])}</p><ul>{sources}</ul>'
                     f'<p class="hint">Full report: <code>{escape(answer["report"])}</code></p></article>')
    return f'<div class="panel" id="answers"><h2>The four questions you handed back</h2><div class="qa">{"".join(cards)}</div></div>'

ISSUE = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/313"
EDITS = {"A_move_swap": "One move replaced", "B_ability_swap": "Ability replaced", "C_item_swap": "Held item replaced",
         "D_alignment_swap": "Stat alignment replaced", "E_spread_copy": "Stat Points copied from another set",
         "F_candidate_copy": "Whole Pokemon copied from a real set"}


def rerun_panel():
    """The rerun of the six-edit-kind study: a pending note until rerun-result.json exists, then its table.

    rerun-result.json is the queue's ruggedness2_shuffled50.json, optionally with a top-level
    "checked" string written after the result has been judged.
    """
    path = site_dir / "rerun-result.json"
    link = f'<a href="{ISSUE}">issue #313</a>'
    if not path.exists():
        return ('<div class="flags rerun"><h2>Rerun queued for points 1 and 5</h2><p>The six-edit-kind study is queued again with shuffled '
                f'slot order against all 50 meta files, 49 distinct teams. Its table lands here and in {link} when the battles finish.</p></div>')
    res = json.loads(path.read_text())
    checked = res.get("checked")
    readings = json.loads((site_dir / "rerun-readings.json").read_text())      # sizes with intervals and the two readings, no size cutoff
    rows = []
    for op, name in EDITS.items():
        cells = [f"<th scope=\"row\">{name}</th>"]
        for group in ("outside_top50", "search_found"):
            c = readings["groups"][group]["kinds"][op]
            low, high = c["change_points_ci95"]
            cells.append(f"<td>{c['change_points']:.2f} [{low:.2f}, {high:.2f}]</td><td>{c['share_of_spread']:.2f}</td>"
                         f"<td>{'yes' if c['above_zero'] else 'not shown'}</td><td>{c['against_pest_control_step']}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    steps = " and ".join(f'{p["step_share_of_spread"]:.2f}' for p in readings["pest_control"].values())
    status = (f"<p><b>Checked.</b> {checked}</p>" if checked else
              "<p><b>Not checked yet.</b> The queue wrote this table the moment the analysis ended. Do not quote it before the checks are posted.</p>")
    head = "<th>Change, points, 95% interval</th><th>Share of the spread between teams</th><th>Above zero</th><th>Against one Pest Control step</th>"
    return ('<div class="flags rerun"><h2>Rerun result: one edit of each kind, shuffled slots, all 50 meta files</h2>' + status +
            '<div class="scroll"><table><thead><tr><th rowspan="2">Edit</th><th colspan="4">From 24 tournament-ready teams</th><th colspan="4">From 16 search-found teams</th></tr>'
            f'<tr>{head}{head}</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
            '<p class="note">Each score is 384 battles with the behaviour-cloning policy on both sides. Change is the noise-corrected size of one edit in win-rate percentage points. '
            f'No size cutoff is used. Above zero: the interval excludes zero. The spread of true scores between starting teams is {readings["spread_between_teams_points"]:.2f} points, '
            f'and one Pest Control step is {steps} of the spread between its random and its hill-climbed starting points; a kind is larger when the interval of the difference lies above zero against both. '
            f'Details in {link}.</p>'
            '<p class="note"><b>Corrected 2026-11-13.</b> An earlier version of this table pooled the benchmark\'s two kinds of starting point, which made its step 0.14 and read four edit kinds as larger. '
            'You chose each kind on its own spread. The whole-Pokemon swap from search-found teams misses "larger" narrowly.</p></div>')

body = r"""<title>One Edit Away Elements</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500;600&family=Schibsted+Grotesk:wght@600;700&display=swap">
<style>
/* Layout: one map of five action boxes, each row a link to that element's own view; hash routes between them. */
:root{
  --bg:#f3f5f8; --panel:#ffffff; --ink:#18202b; --muted:#5b6675; --line:#d5dbe3;
  --c0:#2a5bc4; --c0bg:#e4ecfa; --c1:#17794a; --c1bg:#e0f2e8; --c2:#b0560f; --c2bg:#fae9d8;
  --c3:#7a5fc0; --c3bg:#ece6fa; --c4:#a82b7c; --c4bg:#f8e2f0;
  --display:"Schibsted Grotesk","Helvetica Neue",Arial,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#10141a; --panel:#19202a; --ink:#e8ecf1; --muted:#9aa5b4; --line:#2e3846;
    --c0:#8db0ff; --c0bg:#17233f; --c1:#6ccf96; --c1bg:#12291d; --c2:#f0a25e; --c2bg:#2d1f12;
    --c3:#b9a4f5; --c3bg:#231c3a; --c4:#ee8fcb; --c4bg:#2e1526; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#10141a; --panel:#19202a; --ink:#e8ecf1; --muted:#9aa5b4; --line:#2e3846;
  --c0:#8db0ff; --c0bg:#17233f; --c1:#6ccf96; --c1bg:#12291d; --c2:#f0a25e; --c2bg:#2d1f12;
  --c3:#b9a4f5; --c3bg:#231c3a; --c4:#ee8fcb; --c4bg:#2e1526; color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:400 15px/1.5 var(--body);
  padding-block:24px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1240px;margin:0 auto}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0}
h1{font-size:26px;line-height:1.2}
a{color:inherit}
.sub{color:var(--muted);margin:6px 0 18px;max-width:68ch}
.flags{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.flags h2{font-size:15px;margin-bottom:8px}
.flags ol{margin:0;padding-left:20px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 28px}
.flags li{font-size:13.5px}
.flags b{font-weight:600}
.rerun{border-color:var(--c1)}
.rerun p{margin:0 0 8px;font-size:13.5px;max-width:78ch}
.rerun .note{color:var(--muted);margin:8px 0 0}
.rerun a{color:var(--c0)}
.scroll{overflow-x:auto}
.rerun table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;min-width:720px;width:100%}
.rerun th,.rerun td{border:1px solid var(--line);padding:5px 8px;text-align:right}
.rerun th[scope="row"],.rerun thead th:first-child{text-align:left}
.rerun thead th{font-weight:600;text-align:center}
.grid{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr))}
.box{border:1.5px solid var(--c);background:var(--cbg);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:6px;min-width:0}
.box h2{font-size:16px;color:var(--c)}
.box .does{color:var(--muted);font-size:13px;margin:0 0 2px}
.k0{--c:var(--c0);--cbg:var(--c0bg)} .k1{--c:var(--c1);--cbg:var(--c1bg)} .k2{--c:var(--c2);--cbg:var(--c2bg)}
.k3{--c:var(--c3);--cbg:var(--c3bg)} .k4{--c:var(--c4);--cbg:var(--c4bg)}
.row{display:grid;grid-template-columns:150px minmax(0,1fr);background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.row:hover,.row:focus-within{border-color:var(--c);outline:2px solid var(--c);outline-offset:1px}
.row .name a{text-decoration:none;outline:none}
.row .name{padding:8px 10px;font-weight:600;font-size:14px;border-right:1px solid var(--line)}
.row .name small{display:block;font-weight:400;color:var(--muted);font-size:12px;margin-top:3px}
.row .lead{padding:8px 10px;font-size:13.5px;min-width:0}
.row .pat{display:block;color:var(--c);font-size:12px;margin-bottom:3px}
.crumbs{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:13.5px;margin-bottom:14px}
.crumbs a{text-decoration:none;border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:4px 10px}
.crumbs a:hover,.crumbs a:focus-visible{border-color:var(--c);outline:none}
.crumbs .sp{flex:1}
.head{border:1.5px solid var(--c);background:var(--cbg);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.head .act{color:var(--c);font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.head h1{margin:2px 0 4px}
.head p{margin:0;color:var(--muted);max-width:68ch}
.cols{display:grid;grid-template-columns:minmax(0,320px) minmax(0,1fr);gap:14px;align-items:start}
.side,.cand,.gapsbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.side dl{margin:0;display:grid;gap:10px}
.side dt{color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.side dd{margin:2px 0 0;font-size:14px}
.side .extra{white-space:pre-wrap}
.list{display:flex;flex-direction:column;gap:10px;min-width:0}
.cand .n{color:var(--c);font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.cand p{margin:4px 0 8px;font-size:16px;max-width:68ch}
details{font-size:13px;color:var(--muted)}
summary{cursor:pointer}
details ul{margin:6px 0 0;padding-left:18px}
details li{margin:5px 0}
code{font:12px var(--mono);overflow-wrap:anywhere}
.gapsbox h2{font-size:15px;margin-bottom:6px}
.gapsbox ul{margin:0;padding-left:18px}
.gapsbox li{font-size:14px;margin:4px 0;max-width:72ch}
.panel{scroll-margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.panel h2{font-size:15px;margin-bottom:4px}
.hint{color:var(--muted);font-size:13px;margin:0 0 10px;max-width:72ch}
.chkgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 20px}
.chkgroup{display:flex;flex-direction:column;gap:5px;min-width:0}
.chkgroup h3{font:600 12px/1.3 var(--body);color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.chk{border:1px solid var(--line);border-radius:8px;padding:7px 10px;background:var(--bg);cursor:pointer}
.chk:hover,.chk:focus-within{border-color:var(--c0);outline:2px solid var(--c0);outline-offset:1px}
.chk a{display:block;font-weight:600;font-size:13.5px;text-decoration:none;outline:none}
.chk a::before{content:"";display:inline-block;width:11px;height:11px;border:1.5px solid var(--muted);border-radius:3px;margin-right:8px;vertical-align:-1px}
.chk span{display:block;color:var(--muted);font-size:13px;margin-top:2px}
.qa{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:8px}
.qa article{border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0;background:var(--bg)}
.qa h3{font:600 13px/1.35 var(--body);color:var(--muted);margin-bottom:6px}
.qa p{margin:0 0 6px;font-size:14.5px}
.qa ul{margin:0 0 6px;padding-left:18px}
.qa li{font-size:13px;color:var(--muted);margin:4px 0;overflow-wrap:anywhere}
.qa .hint{margin:0}
.yours,.ans{border-radius:10px;padding:10px 14px;min-width:0}
.yours{border:1.5px dashed var(--c);background:var(--panel)}
.ans{border:1px solid var(--c);background:var(--cbg)}
.yours h2,.ans h2{font:600 12px/1.3 var(--body);color:var(--c);letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.yours blockquote{margin:4px 0;font-size:14.5px;max-width:72ch;overflow-wrap:anywhere}
.yours .none{color:var(--muted);font-size:14px;margin:0}
.ans h3{font:600 14px/1.35 var(--body);margin:6px 0 2px}
.ans p{margin:0 0 6px;font-size:14.5px;max-width:72ch}
.ans .did{color:var(--muted);font-size:13px}
.cand.pick{border:2px solid var(--c)}
.badge{display:inline-block;margin-left:8px;padding:1px 8px;border-radius:999px;background:var(--c);color:var(--panel);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.ver{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;min-width:0}
.ver h2{font-size:17px}
.ver .where{color:var(--muted);font-size:13px;margin:2px 0 8px}
.ver h3{font:600 12px/1.3 var(--body);color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin:10px 0 4px}
.ver ol,.ver ul{margin:0;padding-left:20px;font-size:14px;max-width:72ch}
.ver li{margin:3px 0}
.post{position:relative;white-space:pre-wrap;font-size:14.5px;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:6px 0;background:var(--bg);max-width:76ch;overflow-wrap:anywhere}
.post .meta{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;white-space:normal}
.copy{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:3px 10px;cursor:pointer}
.copy:hover,.copy:focus-visible{border-color:var(--c0)}
.jump{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.jump a{text-decoration:none;border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:3px 12px;font-size:13px}
@media (max-width:860px){.chkgrid,.qa{grid-template-columns:minmax(0,1fr)}}
@media (max-width:860px){.grid,.flags ol,.cols{grid-template-columns:minmax(0,1fr)}.row{grid-template-columns:minmax(0,1fr)}.row .name{border-right:0;border-bottom:1px solid var(--line)}}
</style>

<main>
<section id="map">
  <h1 id="title">One edit away: explanation elements</h1>
  <p class="sub">Material for the smoothness post. Each box below is one Diataxis action. Each row is one element, with its sentence frame and the candidate sentence you picked, or the first one when you have not picked. A row opens that element's page: your note, the candidates, their sources and the open questions.</p>
  __CHECKLIST__
  __ANSWERS__
  __RERUN__
  __TOP50__
  <div class="flags">
    <h2>Where the sources disagree with the draft</h2>
    <ol>
      <li><b>The "One edit away" figure shows battles that no longer count.</b> Its pilot and the October study ran in listed slot order. The rerun above measured all six edit kinds with shuffled slots, so the post cites the rerun.</li>
      <li><b>Two studies now cover two starting points.</b> Your note names the top 50 as where the edits start. The rerun started from 24 tournament-ready teams outside the top 50 and 16 search-found teams. The study that edits the top 50 themselves finished on 2026-11-14 and is the second table above. Say which one a number comes from.</li>
      <li><b>"Benchmarks are smooth, Pokemon is not" holds for one edit kind.</b> A whole-Pokemon swap on a tournament-ready team is 0.65 of the spread between teams, larger than a Pest Control step at 0.37 and 0.32. Move, item and ability edits cannot be told apart from that step.</li>
      <li><b>"A random team performs terribly" has no current number.</b> The only one is 1.845%, from battles in listed slot order with a sampler that is not uniform over legal teams.</li>
      <li><b>The size of the change depends on the edit kind.</b> A whole-Pokemon swap and a move edit are above zero in both groups. Item and ability edits are above zero from tournament-ready teams only. Stat alignment and Stat Points edits are not shown above zero, which is not proof they do nothing.</li>
      <li><b>"Other benchmarks are smooth" is half of your own table.</b> Of 10 measured benchmark tasks, 5 stay smooth and 5 are rough, at least near good points. Name the smooth ones the post compares against.</li>
      <li><b>Reinforcement learning was studied, never trained.</b> Your closing paragraph lists it beside MAP-Elites and the cross-entropy method, which were built and run. Every battle uses a behaviour-cloning policy.</li>
    </ol>
  </div>
  <div class="grid" id="grid"></div>
</section>
<section id="detail" hidden></section>
<section id="socialview" hidden></section>
</main>

<script id="data" type="application/json">__DATA__</script>
<script id="social" type="application/json">__SOCIAL__</script>
<script>
const pairs = JSON.parse(document.getElementById('data').textContent);
const actions = ["Make connections","Provide context","Talk about the subject","Admit opinion and perspective","Keep it bounded"];
const does = ["Tie the idea to something the reader already holds","Say why the work has this shape","Step back and discuss the idea itself","Concede what cuts against the claim","Answer one why question and stop"];
const short = {'Scope set by one real or imagined "why" question':'One "why" question','Alternatives, counter-examples, other approaches':'Counter-examples'};
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const clip = (s, n) => s.length > n ? s.slice(0, s.lastIndexOf(' ', n - 2)) + ' …' : s;
const kind = p => 'k' + actions.indexOf(p.action);
const mapView = document.getElementById('map'), detail = document.getElementById('detail'), grid = document.getElementById('grid');
const social = JSON.parse(document.getElementById('social').textContent), socialView = document.getElementById('socialview');
const picked = p => Number.isInteger(p.picked) ? p.picked : -1;
document.getElementById('title').textContent = 'One edit away: ' + pairs.length + ' explanation elements';

actions.forEach((a, i) => {
  const box = el('section', 'box k' + i);
  box.append(el('h2', '', a), el('p', 'does', does[i]));
  pairs.filter(p => p.action === a).forEach(p => {
    const row = el('div', 'row'); row.dataset.go = p.slug;
    const name = el('div', 'name'), title = el('a', '', short[p.element] || p.element); title.href = '#' + p.slug; name.append(title);
    const open = p.gaps.filter(g => !g.startsWith('Answered')).length;
    name.append(el('small', '', p.candidates.length + ' candidates · ' + open + ' open' + (picked(p) >= 0 ? ' · your pick' : '')));
    const lead = el('div', 'lead');
    lead.append(el('span', 'pat', clip(p.sentence_pattern, 110)), document.createTextNode(clip(p.candidates[Math.max(picked(p), 0)].sentence, 230)));
    row.append(name, lead); box.append(row);
  });
  grid.append(box);
});

function showDetail(p) {
  const i = pairs.indexOf(p), prev = pairs[i - 1], next = pairs[i + 1];
  detail.className = kind(p); detail.replaceChildren();
  const crumbs = el('nav', 'crumbs');
  const back = el('a', '', '← All ' + pairs.length + ' elements'); back.href = '#map'; crumbs.append(back, el('span', 'sp'));
  if (prev) { const a = el('a', '', '← ' + (short[prev.element] || prev.element)); a.href = '#' + prev.slug; crumbs.append(a); }
  if (next) { const a = el('a', '', (short[next.element] || next.element) + ' →'); a.href = '#' + next.slug; crumbs.append(a); }
  const head = el('header', 'head');
  head.append(el('div', 'act', p.action), el('h1', '', p.element), el('p', '', p.what_it_does));
  const cols = el('div', 'cols'), side = el('aside', 'side'), dl = el('dl');
  [['Use it when', p.when_to_use], ['Sentence frame', p.sentence_pattern], ['dLLM post example', p.dllm_example && '"' + p.dllm_example + '"'], ['Sits beside, in the draft', '"' + p.anchor_in_draft + '"']]
    .filter(([, v]) => v).forEach(([k, v]) => { const w = el('div'); w.append(el('dt', '', k), el('dd', '', v)); dl.append(w); });
  if (p.extra) { const w = el('div'); w.append(el('dt', '', p.extra.label), el('dd', 'extra', p.extra.text)); dl.append(w); }
  side.append(dl);
  const list = el('div', 'list');
  if (p.owner_note !== undefined) {
    const yours = el('section', 'yours'); yours.append(el('h2', '', 'Your note on this checkbox'));
    if (p.owner_note && p.owner_note.length) p.owner_note.forEach(t => yours.append(el('blockquote', '', t)));
    else yours.append(el('p', 'none', p.owner_note ? 'You left the box under this element empty.' : 'You listed this element with no note.'));
    list.append(yours);
  }
  if ((p.owner_answers && p.owner_answers.length) || p.applied) {
    const ans = el('section', 'ans'); ans.append(el('h2', '', 'What the subagent brought back'));
    (p.owner_answers || []).forEach(a => ans.append(el('h3', '', a.label), el('p', '', a.text)));
    if (p.applied) ans.append(el('p', 'did', p.applied));
    list.append(ans);
  }
  p.candidates.forEach((c, n) => {
    const card = el('article', 'cand' + (n === picked(p) ? ' pick' : '')), tag = el('div', 'n', 'Candidate ' + (n + 1));
    if (n === picked(p)) tag.append(el('span', 'badge', 'Your pick'));
    card.append(tag, el('p', '', c.sentence));
    const d = el('details');
    if (c.facts.length) {
      d.open = true; d.append(el('summary', '', c.facts.length + ' sourced fact' + (c.facts.length > 1 ? 's' : '')));
      const ul = el('ul'); c.facts.forEach(f => { const li = el('li', '', f.claim + ' '); li.append(el('code', '', f.source)); ul.append(li); }); d.append(ul);
    } else d.append(el('summary', '', 'No project fact in this sentence'));
    card.append(d); list.append(card);
  });
  if (p.gaps.length) { const g = el('section', 'gapsbox'); g.append(el('h2', '', 'Open questions for Rami')); const ul = el('ul'); p.gaps.forEach(q => ul.append(el('li', '', q))); g.append(ul); list.append(g); }
  cols.append(side, list); detail.append(crumbs, head, cols);
}
function showSocial() {
  socialView.replaceChildren();
  const crumbs = el('nav', 'crumbs'), back = el('a', '', '← All ' + pairs.length + ' elements'); back.href = '#map'; crumbs.append(back);
  const head = el('header', 'head k0');
  head.append(el('div', 'act', 'Where to post'), el('h1', '', social.versions.length + ' versions, one for each site'),
    el('p', '', 'Drafts in your voice for you to edit. LinkedIn goes out first. Each version has three opening lines to choose from. Skipped: ' + social.skipped.join(' ')));
  const claim = el('section', 'ans k0'); claim.append(el('h2', '', 'The claim every version carries'), el('p', '', social.claim));
  const jump = el('nav', 'jump');
  social.versions.forEach(v => { const a = el('a', '', v.site + (v.site === 'Discord' ? ', ' + v.place.split(',')[0] : '')); a.href = '#v-' + v.id; a.addEventListener('click', e => { e.preventDefault(); document.getElementById('v-' + v.id).scrollIntoView({behavior: 'smooth'}); }); jump.append(a); });
  socialView.append(crumbs, head, claim, jump);
  social.versions.forEach(v => {
    const box = el('article', 'ver'); box.id = 'v-' + v.id;
    box.append(el('h2', '', v.site), el('p', 'where', v.place + ' · for ' + v.reader + ' · limit ' + v.limit));
    box.append(el('h3', '', 'Three opening lines'));
    const hooks = el('ol'); v.hooks.forEach(h => hooks.append(el('li', '', h))); box.append(hooks);
    box.append(el('h3', '', v.posts.length > 1 ? v.posts.length + ' parts' : 'The post'));
    v.posts.forEach((text, n) => {
      const post = el('div', 'post'), meta = el('div', 'meta'), copy = el('button', 'copy', 'Copy');
      copy.type = 'button';
      copy.addEventListener('click', () => navigator.clipboard.writeText(text).then(() => { copy.textContent = 'Copied'; setTimeout(() => copy.textContent = 'Copy', 1200); }, () => { copy.textContent = 'Select and copy by hand'; }));
      meta.append(el('span', '', (v.posts.length > 1 ? 'Part ' + (n + 1) + ' · ' : '') + text.length + ' characters'), copy);
      post.append(meta, document.createTextNode(text)); box.append(post);
    });
    box.append(el('h3', '', 'Attach'), el('p', 'where', v.attach));
    if (v.cautions.length) { box.append(el('h3', '', 'Before you post')); const ul = el('ul'); v.cautions.forEach(c => ul.append(el('li', '', c))); box.append(ul); }
    socialView.append(box);
  });
}
// A row or checklist box opens its page on a click anywhere in it, unless the click ended a text selection.
document.addEventListener('click', e => {
  const box = e.target.closest('[data-go]');
  if (box && !e.target.closest('a') && !String(getSelection())) location.hash = box.dataset.go;
});
function route() {
  const hash = location.hash.slice(1), p = pairs.find(x => x.slug === hash), isSocial = hash === 'social';
  mapView.hidden = !!p || isSocial; detail.hidden = !p; socialView.hidden = !isSocial;
  if (p) { showDetail(p); window.scrollTo(0, 0); }
  else if (isSocial) { showSocial(); window.scrollTo(0, 0); }
  else { const target = hash && document.getElementById(hash); if (target && target !== mapView) document.fonts.ready.then(() => target.scrollIntoView()); }
}
window.addEventListener('hashchange', route); route();
</script>
"""
body = (body.replace("__DATA__", data).replace("__SOCIAL__", social).replace("__RERUN__", rerun_panel()).replace("__TOP50__", top50_panel())
        .replace("__CHECKLIST__", checklist_panel()).replace("__ANSWERS__", answers_panel()))
(here / "site-body.html").write_text(body)
(site_dir / "site-body.html").write_text(body)
full = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n' + body + "</body>\n</html>\n"
(site_dir / "index.html").write_text(full)
print(len(pairs), "pairs;", (site_dir / "index.html").stat().st_size, "bytes;", [p["slug"] for p in pairs])
