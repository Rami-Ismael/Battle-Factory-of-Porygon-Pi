"""Write the HTML figure page from the analysis JSON files."""
import json, sys, html
S, OUT = sys.argv[1], sys.argv[2]
fill = json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_baseline.json'))["summary"]
mv = json.load(open(S + "/c_moves.json")); al = json.load(open(S + "/e_all.json")); sp = json.load(open(S + "/d_species.json"))
W = 560

def hbars(rows, xmax, ref=None, fmt="{:.3f}", label_w=190, xmin=0.0):
    """rows: [(label, value, (lo,hi) or None)]. One hue; value labels in ink."""
    h = 30 * len(rows) + 34; x0 = label_w; sx = lambda v: x0 + (W - x0 - 60) * (v - xmin) / (xmax - xmin)
    o = [f'<svg viewBox="0 0 {W} {h}" role="img">']
    for t in ([xmin] if xmin else []) + [0, xmax / 2, xmax]:
        o.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="4" y2="{h-24}" class="grid"/><text x="{sx(t):.1f}" y="{h-8}" class="tick" text-anchor="middle">{t:.2f}</text>')
    if ref is not None:
        o.append(f'<line x1="{sx(ref[0]):.1f}" x2="{sx(ref[0]):.1f}" y1="2" y2="{h-24}" class="ref"/><text x="{sx(ref[0])-4:.1f}" y="{h-26}" class="tick" text-anchor="end">{ref[1]}</text>')
    for i, (lab, v, c) in enumerate(rows):
        y = 8 + 30 * i
        tip = f"{lab}: {fmt.format(v)}" + (f" (95% CI {c[0]:+.3f} to {c[1]:+.3f})" if c else "")
        o.append(f'<g class="mark"><title>{html.escape(tip)}</title><rect x="0" y="{y-4}" width="{W}" height="28" fill="transparent"/>'
                 f'<text x="{x0-8}" y="{y+15}" text-anchor="end" class="lab">{html.escape(lab)}</text>'
                 f'<rect x="{sx(0):.1f}" y="{y+2}" width="{max(sx(v)-sx(0),1):.1f}" height="18" rx="4" class="bar"/>')
        if c: o.append(f'<line x1="{sx(c[0]):.1f}" x2="{sx(c[1]):.1f}" y1="{y+11}" y2="{y+11}" class="ci"/>')
        o.append(f'<text x="{sx(c[1] if c else v)+6:.1f}" y="{y+15}" class="val">{fmt.format(v)}</text></g>')
    return "".join(o) + "</svg>"

def line(curve, xlab, min_n=50):
    pts = [(int(k), v[0], v[1]) for k, v in curve.items() if v[1] >= min_n]; pts.sort()
    h = 220; x0, x1, y0, y1 = 44, W - 20, 190, 14; kmax = pts[-1][0]
    sx = lambda k: x0 + (x1 - x0) * k / kmax; sy = lambda v: y0 - (y0 - y1) * v / 0.6
    o = [f'<svg viewBox="0 0 {W} {h+20}" role="img">']
    for t in [0, .2, .4, .6]: o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/><text x="{x0-6}" y="{sy(t)+4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    o.append('<polyline fill="none" class="ln" points="' + " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v, n in pts) + '"/>')
    for k, v, n in pts:
        o.append(f'<g class="mark"><title>{k} {xlab}: win rate {v:.3f} over {n:,} teams</title><circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="12" fill="transparent"/><circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="4.5" class="dot"/></g>'
                 f'<text x="{sx(k):.1f}" y="{y0+18}" class="tick" text-anchor="middle">{k}</text>')
    o.append(f'<text x="{(x0+x1)/2}" y="{h+14}" class="tick" text-anchor="middle">{xlab} on the team</text>')
    return "".join(o) + "</svg>"

parts = [("Abilities, all 6", "ability6"), ("Stat Points, all 6", "stats6"), ("Stat Alignment, all 6", "alignment6"),
         ("Items, all 6", "item6"), ("Moves, all 24", "move24"), ("Whole Pokémon, all 6", "pokemon6")]
fig1 = hbars([(l, fill[k]["filled"], None) for l, k in parts], 0.5, ref=(0.446, "real team 0.446"))
order = ["Protect", "Spread attack", "STAB attack", "Support move", "Speed control", "Coverage attack"]
fig2 = hbars([(c, mv["cost"][c]["value"][0], mv["cost"][c]["value"][1:]) for c in order], 0.08, xmin=-0.02, fmt="+{:.3f}")
mix = "".join(f"<tr><td>{c}</td><td>{mv['mix'][c][0]:.0%}</td><td>{mv['mix'][c][1]:.0%}</td></tr>" for c in [*order, "Other status"])
fig3 = line(al["spread_curve"], "spread attacks"); fig4 = line(al["protect_curve"], "Protect moves")
bk = sp["by_k"]; ktab = "".join(f"<tr><td>{k}</td><td>{v['actual']:+.3f}</td><td>{v['move_pred']:+.3f}</td></tr>" for k, v in bk.items())
page = open(S + "/template.html").read()
for k, v in dict(FIG1=fig1, FIG2=fig2, FIG3=fig3, FIG4=fig4, MIX=mix, KTAB=ktab, NALL=f"{al['n']:,}").items(): page = page.replace("{{" + k + "}}", v)
open(OUT, "w").write(page)
