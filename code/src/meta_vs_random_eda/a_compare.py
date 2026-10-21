import sqlite3, csv, json, sys, numpy as np
sys.path.insert(0, sys.argv[1]); from feats import parse, mon_features, team_features
con = sqlite3.connect("/Users/ramiismael/vgc-data/matchup_regmb.sqlite")
top = {r[0] for r in con.execute("select team_id from team_set where set_name='top50'")}
sc = {int(r[0]): (r[1], int(r[2]) / int(r[3]), int(r[3])) for r in csv.reader(open(sys.argv[1] + "/scores.csv"))}
paste = dict(con.execute("select team_id, paste from team where origin='vgcpastes' or origin like 'randomfill:%'"))
groups = {"real": [t for t, (o, w, n) in sc.items() if o == "vgcpastes" and t not in top],
          "random6": [t for t, (o, w, n) in sc.items() if o == "randomfill:pokemon6"]}
for g, ids in groups.items():
    w = np.array([sc[t][1] for t in ids]); print(g, len(ids), "win rate mean %.3f median %.3f p90 %.3f max %.3f" % (w.mean(), np.median(w), np.quantile(w, .9), w.max()), "battles/team", np.mean([sc[t][2] for t in ids]))
rows = {g: [mon_features(m) for t in ids for m in parse(paste[t])] for g, ids in groups.items()}
print("unknown species:", {g: sum(1 - r["known"] for r in rs) for g, rs in rows.items()})
keys = ["bst", "spe", "off", "bulk", "mega", "n_dmg", "n_stab", "matched_stat", "split_attacker", "spread", "protect", "speed_ctrl", "support", "n_status", "neutral_nature", "ev_max"]
res = {}
print(f"{'feature':15s} {'real':>8s} {'random':>8s} {'d':>6s}")
for k in keys:
    a = np.array([r[k] for r in rows["real"] if r[k] is not None], float); b = np.array([r[k] for r in rows["random6"] if r[k] is not None], float)
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    res[k] = dict(real=a.mean(), random=b.mean(), d=d, real_hist=np.histogram(a, 20)[0].tolist())
    print(f"{k:15s} {a.mean():8.2f} {b.mean():8.2f} {d:6.2f}")
from collections import Counter
for g in rows:
    c = Counter(r["species"] for r in rows[g]); print(g, "distinct species", len(c), "top", c.most_common(8))
json.dump(dict(res=res, n={g: len(i) for g, i in groups.items()}), open(sys.argv[1] + "/a_compare.json", "w"), default=float)
