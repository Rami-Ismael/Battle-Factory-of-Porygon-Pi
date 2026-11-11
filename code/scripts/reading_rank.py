"""Rank the Reading List by relative importance (issue #301; method in research/ranking-a-long-list.md).

Every item is placed against the whole list by pairwise comparisons, each asked in both orders, judged by Jev
through jev_client.JevChooser. Nothing here writes to the vault: the ranking lives in ~/vgc-data/reading-rank/.

    python scripts/reading_rank.py items      parse Papers/Reading List.md into items.json
    python scripts/reading_rank.py cards      fetch an abstract or post text per item into cards.json (cached)
    python scripts/reading_rank.py pilot      30 random pairs in both orders: order agreement and cost
    python scripts/reading_rank.py rank       stable merge sort of every item, then one Bradley-Terry fit -> ranking.json
    python scripts/reading_rank.py axes       each item scored 0 to 4 on the decision matrix's two axes (#260) -> axes.json
    python scripts/reading_rank.py insert     daily: binary-search each new Reading List item into the ranking, refit, rewrite
    python scripts/reading_rank.py save       copy the ranking and its comparison log into code/data/reading-rank/ to commit
"""

import hashlib
import html
import json
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VAULT = Path.home() / ('Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc '
                       'pokemon format to find the right counter to a pokemon team')
READING_LIST = VAULT / 'Papers/Reading List.md'
OUT = Path.home() / 'vgc-data/reading-rank'
AGENT = {'User-Agent': 'reading-rank/0.1 (personal research reading list)'}
CARD_CHARS = 900

QUESTION = 'Which would help the counter-team search more if read next?'
BRIEF = (
    'The reader builds a search for Pokemon VGC (Regulation M-B) teams that beat the current top-50 meta teams. '
    'A discrete (masked) diffusion model writes whole six-Pokemon teams, conditioned on an asked win rate, and a search '
    'loop battles candidates with a behaviour-cloning policy to label them. Open problems: getting win rate to follow the '
    'ask, keeping teams diverse and novel, guidance strength, training labels from noisy battles, and planning which '
    'papers to read. The reader learns by working problems and prefers blogs before papers. '
    'Rank closest to the search itself first: building teams or decks in games, finding candidates beyond the training '
    'data, novelty and diversity of what a generator proposes, and judging or battling candidates. Diffusion technique '
    'ranks next when it changes how the ask steers generation; general machine-learning theory and workflow advice rank '
    'below both (owner picks, 2026-11-11: Q-DeckRec and Verifier-Constrained Flow Expansion over diffusion-guidance papers). '
    'Judge each option on two things: after its abstract, does it have a future for the counter-team search; and while '
    'reading, how much of it would be new to someone who already knows masked discrete diffusion, classifier-free '
    'guidance, quality diversity and black-box optimisation.'
)


def get(url: str, timeout: float = 20) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=timeout).read()


def url_key(url: str) -> str:
    return re.sub(r'^https?://(www\.)?', '', url).rstrip('/')


def item_id(url: str) -> str:
    """Stable across edits of the Reading List: new rows go in at the top, so row order cannot name an item."""
    return 'u' + hashlib.sha1(url_key(url).encode()).hexdigest()[:10]


def items() -> None:
    rows, section = [], ''
    for line in READING_LIST.read_text().splitlines():
        if line.startswith('#'):
            section = line.lstrip('#').strip()
        m = re.match(r'^\|\s*(?:[^\s\[|]+\s+)?\[([^\]]+)\]\(([^)\s]+)\)\s*\|(.*)$', line)   # an icon such as 🔒 may lead
        if m:
            cells = [c.strip() for c in m.group(3).split('|')]
            rows.append(dict(title=m.group(1).strip(), url=m.group(2), section=section, details=' · '.join(c for c in cells if c)))
    seen, out = set(), []
    for r in rows:
        key = url_key(r['url'])
        if key in seen:
            continue
        seen.add(key)
        r['id'] = item_id(r['url'])
        out.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'items.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f'{len(rows)} rows, {len(out)} distinct items, {len({r["section"] for r in out})} sections')


def arxiv_id(url: str) -> str | None:
    m = re.search(r'(?:(?:arxiv\.org|alphaxiv\.org)/(?:abs|pdf|html|overview)/|semanticscholar\.org/arXiv:)(\d{4}\.\d{4,5})', url)
    return m.group(1) if m else None


def arxiv_abstracts(ids: list[str]) -> dict[str, str]:
    out = {}
    for i in range(0, len(ids), 50):
        q = urllib.parse.urlencode({'id_list': ','.join(ids[i:i + 50]), 'max_results': 50})
        root = ET.fromstring(get(f'https://export.arxiv.org/api/query?{q}', 60))
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        for e in root.findall('a:entry', ns):
            aid = re.search(r'(\d{4}\.\d{4,5})', e.findtext('a:id', '', ns) or '')
            if aid:
                out[aid.group(1)] = ' '.join((e.findtext('a:summary', '', ns) or '').split())
    return out


def s2_id(url: str) -> str | None:
    m = re.search(r'semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})', url) or re.search(r'doi\.org/(10\.[^\s?#]+)', url)
    if not m:
        return None
    return m.group(1) if 'semanticscholar' in m.group(0) else f'DOI:{m.group(1)}'


def s2_abstracts(ids: list[str]) -> dict[str, str]:
    """Abstract, else the TLDR, from Semantic Scholar's batch endpoint."""
    out = {}
    for i in range(0, len(ids), 400):
        req = urllib.request.Request('https://api.semanticscholar.org/graph/v1/paper/batch?fields=abstract,tldr',
                                     data=json.dumps({'ids': ids[i:i + 400]}).encode(), headers={**AGENT, 'Content-Type': 'application/json'})
        for attempt in range(3):                               # the batch endpoint answers 429 under load
            try:
                got = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except urllib.error.HTTPError as err:
                if err.code != 429 or attempt == 2:
                    print(f'Semantic Scholar unavailable ({err.code}); those items stay title only')
                    return out
                __import__('time').sleep(20 * (attempt + 1))
        for k, r in zip(ids[i:i + 400], got):
            if r:
                out[k] = r.get('abstract') or ((r.get('tldr') or {}).get('text')) or ''
    return out


def page_text(url: str) -> str:
    """Post text for an X link, otherwise the page's description or first paragraphs."""
    m = re.match(r'https?://(?:x|twitter)\.com/([^/]+)/status/(\d+)', url)
    if m:
        t = json.loads(get(f'https://api.fxtwitter.com/{m.group(1)}/status/{m.group(2)}'))['tweet']
        parts = [t.get('text', ''), (t.get('quote') or {}).get('text', ''), (t.get('article') or {}).get('title', '')]
        return ' '.join(' '.join(p.split()) for p in parts if p)
    m = re.match(r'https?://github\.com/([^/]+)/([^/#?]+)', url)
    if m:
        r = json.loads(get(f'https://api.github.com/repos/{m.group(1)}/{m.group(2)}'))
        return r.get('description') or ''
    m = re.search(r'openreview\.net/(?:forum|pdf)\?id=([^&]+)', url)
    if m:
        notes = json.loads(get(f'https://api2.openreview.net/notes?id={m.group(1)}')).get('notes') or [{}]
        ab = (notes[0].get('content') or {}).get('abstract')
        return (ab.get('value') if isinstance(ab, dict) else ab) or ''
    doc = get(url).decode('utf-8', 'replace')
    for pat in (r'<meta[^>]+(?:name|property)=["\'](?:og:description|description|citation_abstract)["\'][^>]+content=["\']([^"\']+)',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\'](?:og:description|description)["\']'):
        m = re.search(pat, doc, re.I)
        if m and len(m.group(1)) > 60:
            return html.unescape(m.group(1))
    doc = re.sub(r'<(script|style|nav|header|footer|aside|form|noscript)\b.*?</\1>', ' ', doc, flags=re.S | re.I)
    body = (re.search(r'<article\b.*?</article>', doc, re.S | re.I) or re.search(r'<main\b.*?</main>', doc, re.S | re.I))
    paras = [re.sub(r'<[^>]+>', '', p) for p in re.findall(r'<p[^>]*>(.*?)</p>', body.group(0) if body else doc, re.S | re.I)]
    return html.unescape(' '.join(' '.join(p.split()) for p in paras if len(p.split()) >= 15))[:CARD_CHARS]


def cards() -> None:
    its = json.loads((OUT / 'items.json').read_text())
    path = OUT / 'cards.json'
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [it for it in its if it['id'] not in have or not have[it['id']].get('text')]
    ids = {it['id']: arxiv_id(it['url']) for it in todo}
    abstracts = arxiv_abstracts(sorted({a for a in ids.values() if a}))
    s2 = {it['id']: s2_id(it['url']) for it in todo if not ids[it['id']]}
    s2_text = s2_abstracts(sorted({v for v in s2.values() if v}))

    def one(it):
        a = ids[it['id']]
        try:
            if a and a in abstracts:
                text, source = abstracts[a], 'arxiv abstract'
            elif s2.get(it['id']) and s2_text.get(s2[it['id']]):
                text, source = s2_text[s2[it['id']]], 'semantic scholar'
            else:
                text, source = page_text(it['url']), 'page'
        except Exception as e:
            text, source = '', f'unread: {type(e).__name__}'
        return it['id'], dict(text=' '.join(text.split())[:CARD_CHARS], source=source)

    with ThreadPoolExecutor(8) as pool:
        for k, v in pool.map(one, todo):
            have[k] = v
    path.write_text(json.dumps(have, indent=1, ensure_ascii=False))
    n = sum(1 for it in its if have.get(it['id'], {}).get('text'))
    print(f'{n} of {len(its)} items have text; {len(its) - n} title only')


def card(it: dict, c: dict) -> str:
    body = c.get('text') or '(no abstract: judge from the title and details)'
    return f"{it['title']}. Shelf: {it['section']}. {('Details: ' + it['details'] + '. ') if it['details'] else ''}Summary: {body}"


def ask(chooser, pairs: list[tuple[dict, dict]], cs: dict) -> list[float]:
    """P(first item preferred) per pair, from one Jev request holding every pair as its own question."""
    questions = {f'p{i}': {'type': 'choice', 'instructions': QUESTION,
                           'criteria': {'A': card(a, cs[a['id']]), 'B': card(b, cs[b['id']])}} for i, (a, b) in enumerate(pairs)}
    answers = chooser.choose({'asking_about': 'reading-list ranking', 'project': BRIEF}, questions)
    return [answers[f'p{i}']['probabilities'].get('A', 0.0) for i in range(len(pairs))]


def pilot(n: int = 30) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jev_client
    its = json.loads((OUT / 'items.json').read_text())
    cs = json.loads((OUT / 'cards.json').read_text())
    rng = random.Random(30101)
    pairs = [tuple(rng.sample(its, 2)) for _ in range(n)]
    chooser = jev_client.JevChooser(OUT / 'jev-calls.jsonl')
    fwd, rev = [], []
    for i in range(0, n, 10):
        fwd += ask(chooser, pairs[i:i + 10], cs)
        rev += ask(chooser, [(b, a) for a, b in pairs[i:i + 10]], cs)
    rows = []
    for (a, b), pf, pr in zip(pairs, fwd, rev):
        p = (pf + (1 - pr)) / 2                                  # P(a preferred), averaged over both orders
        rows.append(dict(a=a['id'], b=b['id'], a_title=a['title'], b_title=b['title'], p_forward=pf, p_reversed_b=pr,
                         p_a=p, agree=(pf > 0.5) == (pr < 0.5)))
    agree = sum(r['agree'] for r in rows) / len(rows)
    first = sum(pf > 0.5 for pf in fwd + rev) / (2 * n)
    out = dict(pairs=n, calls=chooser.usage['calls'], input_tokens=chooser.usage['input_tokens'], dollars=chooser.dollars(),
               order_agreement=agree, share_first_option_chosen=first,
               mean_confidence=sum(abs(r['p_a'] - 0.5) * 2 for r in rows) / n, rows=rows)
    (OUT / 'pilot.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


BATCH, PRIOR, SEED = 40, 0.1, 40301


def judge(chooser, pairs: list[tuple[dict, dict]], cs: dict, log) -> list[float]:
    """P(first preferred) per pair, averaged over both orders; every judgment appended to log."""
    out = []
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        f, r = ask(chooser, chunk, cs), ask(chooser, [(b, a) for a, b in chunk], cs)
        for (a, b), pf, pr in zip(chunk, f, r):
            p = (pf + 1 - pr) / 2
            log.write(json.dumps(dict(a=a['id'], b=b['id'], p_forward=pf, p_reversed_b=pr, p_a=p)) + '\n')
            out.append(p)
    return out


def bradley_terry(ids: list[str], comps: list[dict]):
    import numpy as np
    ix = {k: n for n, k in enumerate(ids)}
    a = np.array([ix[c['a']] for c in comps]); b = np.array([ix[c['b']] for c in comps]); y = np.array([c['p_a'] for c in comps])
    th = np.zeros(len(ids))
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(th[a] - th[b])))
        g = np.zeros(len(ids)); np.add.at(g, a, y - p); np.add.at(g, b, p - y)
        th += 0.05 * (g - PRIOR * th)
        th -= th.mean()
    return th


def rank() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jev_client
    its = json.loads((OUT / 'items.json').read_text())
    cs = json.loads((OUT / 'cards.json').read_text())
    order = its[:]
    random.Random(SEED).shuffle(order)                         # mix shelves before sorting
    chooser = jev_client.JevChooser(OUT / 'jev-calls.jsonl')
    runs = [[x] for x in order]
    with (OUT / 'comparisons.jsonl').open('a') as log:
        while len(runs) > 1:
            merges = [[runs[k], runs[k + 1], 0, 0, []] for k in range(0, len(runs) - 1, 2)]
            odd = [runs[-1]] if len(runs) % 2 else []
            while True:
                live = [m for m in merges if m[2] < len(m[0]) and m[3] < len(m[1])]
                if not live:
                    break
                ps = judge(chooser, [(m[0][m[2]], m[1][m[3]]) for m in live], cs, log)
                for m, p in zip(live, ps):
                    if p >= 0.5:
                        m[4].append(m[0][m[2]]); m[2] += 1
                    else:
                        m[4].append(m[1][m[3]]); m[3] += 1
            runs = [m[4] + m[0][m[2]:] + m[1][m[3]:] for m in merges] + odd
            print(f'  {len(runs)} runs left, {chooser.usage["calls"]} calls, ${chooser.dollars():.3f}', flush=True)
    merged = runs[0]
    comps = [json.loads(l) for l in (OUT / 'comparisons.jsonl').read_text().splitlines()]
    ids = [it['id'] for it in its]
    th = bradley_terry(ids, comps)
    by = {it['id']: it for it in its}
    ranked = sorted(ids, key=lambda k: -th[ids.index(k)])
    n = len(ranked)
    rows = [dict(rank=r + 1, of=n, top_percent=round(100 * (r + 1) / n, 1), theta=float(th[ids.index(k)]),
                 comparisons=sum(1 for c in comps if k in (c['a'], c['b'])), merge_position=merged.index(by[k]) + 1,
                 **{f: by[k][f] for f in ('id', 'title', 'url', 'section')}) for r, k in enumerate(ranked)]
    agree = sum(1 for x in rows if abs(x['rank'] - x['merge_position']) <= 10) / n
    out = dict(method='order-swapped Jev pairwise comparisons, stable merge sort, Bradley-Terry fit (issue #301)',
               brief=BRIEF, question=QUESTION, items=n, judgments=len(comps), calls=chooser.usage['calls'],
               dollars=chooser.dollars(), within_10_of_merge_order=agree, ranking=rows)
    (OUT / 'ranking.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ('ranking', 'brief')}, indent=1))


AXES = {
    'future': ('After reading only its abstract: how much of a future does it have for the counter-team search?',
               ['none', 'a little: background only', 'some: one idea worth trying later', 'clear: a method to try soon',
                'strong: it changes what the search does next']),
    'new': ('While reading it: how much of it would be new to the reader?',
            ['almost nothing new', 'a little new', 'about half new', 'mostly new', 'nearly all new']),
}


def axes() -> None:
    """Pointwise 0-4 scores per axis. Placement only: the order comes from rank(), never from these scores."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jev_client
    its = json.loads((OUT / 'items.json').read_text())
    cs = json.loads((OUT / 'cards.json').read_text())
    chooser = jev_client.JevChooser(OUT / 'jev-calls.jsonl')
    out = score_axes(chooser, its, cs)
    (OUT / 'axes.json').write_text(json.dumps(dict(axes={k: dict(question=q, levels=l) for k, (q, l) in AXES.items()},
                                                   calls=chooser.usage['calls'], dollars=chooser.dollars(), scores=out), indent=1))
    print(f'{len(out)} items scored, {chooser.usage["calls"]} calls, ${chooser.dollars():.3f}')


def score_axes(chooser, its: list[dict], cs: dict) -> dict:
    out = {}
    for i in range(0, len(its), 20):
        chunk = its[i:i + 20]
        qs = {f'{ax}_{it["id"]}': {'type': 'score', 'instructions': f'{q} Item: {card(it, cs[it["id"]])}',
                                   'criteria': {str(n): lv for n, lv in enumerate(levels)}}
              for it in chunk for ax, (q, levels) in AXES.items()}
        ans = chooser.choose({'asking_about': 'reading-list decision matrix', 'project': BRIEF}, qs)
        for it in chunk:
            out[it['id']] = {ax: float(ans[f'{ax}_{it["id"]}']['score']) for ax in AXES}
    return out


def write_ranking(order: list[str], comps: list[dict], extra: dict) -> list[dict]:
    """Refit Bradley-Terry on every stored comparison among the items in order, and write ranking.json by score."""
    its = {it['id']: it for it in json.loads((OUT / 'items.json').read_text())}
    keep = set(order)
    comps = [c for c in comps if c['a'] in keep and c['b'] in keep]
    th = bradley_terry(order, comps)
    score = dict(zip(order, th))
    ranked = sorted(order, key=lambda k: -score[k])
    n = len(ranked)
    count = {}
    for c in comps:
        count[c['a']] = count.get(c['a'], 0) + 1
        count[c['b']] = count.get(c['b'], 0) + 1
    rows = [dict(rank=r + 1, of=n, top_percent=round(100 * (r + 1) / n, 1), theta=float(score[k]), comparisons=count.get(k, 0),
                 **{f: its[k][f] for f in ('id', 'title', 'url', 'section')}) for r, k in enumerate(ranked)]
    out = dict(method='order-swapped Jev pairwise comparisons, stable merge sort, Bradley-Terry fit (issue #301)',
               brief=BRIEF, question=QUESTION, items=n, judgments=len(comps), **extra, ranking=rows)
    (OUT / 'ranking.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return rows


def place(chooser, its: dict, cs: dict, order: list[str], new: list[str], log) -> list[str]:
    """Binary-search every new item into order in lockstep, then judge each against its two new neighbours."""
    lo, hi = {k: 0 for k in new}, {k: len(order) for k in new}
    while any(lo[k] < hi[k] for k in new):
        live = [k for k in new if lo[k] < hi[k]]
        mids = {k: (lo[k] + hi[k]) // 2 for k in live}
        for k, p in zip(live, judge(chooser, [(its[k], its[order[mids[k]]]) for k in live], cs, log)):
            if p >= 0.5:
                hi[k] = mids[k]
            else:
                lo[k] = mids[k] + 1
    placed = order[:]
    for k in sorted(new, key=lambda k: -lo[k]):                # insert from the bottom so earlier positions hold
        placed.insert(lo[k], k)
    log.flush()
    done = {(c['a'], c['b']) for c in map(json.loads, (OUT / 'comparisons.jsonl').read_text().splitlines())}
    nb = [(its[k], its[placed[j]]) for k in new for j in (placed.index(k) - 1, placed.index(k) + 1)
          if 0 <= j < len(placed) and (k, placed[j]) not in done and (placed[j], k) not in done]
    if nb:
        judge(chooser, nb, cs, log)
    return placed


def insert() -> None:
    """Place every Reading List item the ranking lacks: binary search over the current order (about 9 comparisons,
    each asked in both orders, run in lockstep across new items), then a check against both new neighbours, then one
    refit over all comparisons. Items deleted from the Reading List leave the ranking."""
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jev_client
    items()
    cards()
    its = {it['id']: it for it in json.loads((OUT / 'items.json').read_text())}
    cs = json.loads((OUT / 'cards.json').read_text())
    old = json.loads((OUT / 'ranking.json').read_text())
    order = [x['id'] for x in old['ranking'] if x['id'] in its]
    removed = [x['id'] for x in old['ranking'] if x['id'] not in its]
    new = [k for k in its if k not in set(order)]
    if not new and not removed:
        print(f'nothing to insert: {len(order)} items already ranked')
        return
    chooser = jev_client.JevChooser(OUT / 'jev-calls.jsonl')
    with (OUT / 'comparisons.jsonl').open('a') as log:
        placed = place(chooser, its, cs, order, new, log)
    ax = json.loads((OUT / 'axes.json').read_text())
    ax['scores'].update(score_axes(chooser, [its[k] for k in new], cs) if new else {})
    (OUT / 'axes.json').write_text(json.dumps(ax, indent=1))
    comps = [json.loads(line) for line in (OUT / 'comparisons.jsonl').read_text().splitlines()]
    rows = write_ranking(placed, comps, dict(updated=time.strftime('%Y-%m-%d %H:%M'), calls=chooser.usage['calls'],
                                              dollars=chooser.dollars()))
    rank = {x['id']: x['rank'] for x in rows}
    with (OUT / 'ranking-history.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M'), items=len(rows), inserted={k: rank[k] for k in new},
                                removed=removed, calls=chooser.usage['calls'], dollars=chooser.dollars())) + '\n')
    for k in sorted(new, key=rank.get):
        print(f'  #{rank[k]} / {len(rows)}  {its[k]["title"]}')
    print(f'{len(new)} inserted, {len(removed)} removed, {chooser.usage["calls"]} calls, ${chooser.dollars():.3f}')


def save() -> None:
    """Copy the ranking, its comparison log, the axis scores and the item list into the repository to commit."""
    import shutil
    dest = Path(__file__).resolve().parents[1] / 'data/reading-rank'
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('ranking.json', 'comparisons.jsonl', 'axes.json', 'items.json', 'ranking-history.jsonl'):
        if (OUT / name).exists():
            shutil.copy2(OUT / name, dest / name)
    print(f'saved to {dest}')


if __name__ == '__main__':
    {'items': items, 'cards': cards, 'pilot': pilot, 'rank': rank, 'axes': axes, 'insert': insert, 'save': save}[sys.argv[1]]()
