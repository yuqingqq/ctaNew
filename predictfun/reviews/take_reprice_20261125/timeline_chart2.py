"""Draw timeline2.json: what Binance does around the top taker's send (USER 2026-11-26: "You mean they send at around -50ms right? Then we
check what happens in Binance around that time"). Every number is read from timeline2.json / fire_arrival.json.
x = ms from the Binance move (first moves: no move in the same direction in the previous 300 ms). Top strip: when the top taker's order
reaches Predict. Small multiples: five Binance flows as a multiple of their rate at random times (log scale; 10 ms bins, the last 5 ms bin
before the move kept apart), orange = the moves the top taker caught, gray = all first moves; the send window (middle half) shaded, its
median as a line, the value in the 20 ms around the median send printed in each panel; the sixth panel is the best-level imbalance.
Writes timeline_chart2.svg / .html next to this file.
Usage: python3 predictfun/reviews/take_reprice_20261125/timeline_chart2.py
"""
import html
import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
D = json.loads((H / 'timeline2.json').read_text())
FA = json.loads((H / 'fire_arrival.json').read_text())
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
TOP, ALL = '#eb6834', '#52514e'
FONT = "system-ui, -apple-system, 'Segoe UI', 'DejaVu Sans', sans-serif"
W, HT = 1240, 900
E = np.asarray(D['edges_ms'])
lo = E[:-1]
arr = D['top1_arrival_rel_move']
S_TOP, S_ALL = D['sets']['top_caught_first'], D['sets']['all_first']
rand_imb = FA['groups']['TOP1']['trajectory_at_arrival']['imb_0']['random_mean']
TMIN, TMAX = -300., 0.
YMIN, YMAX = .1, 100.
svg = []


def esc(s):
    return html.escape(s, quote=True)


def mfmt(v):
    s = f'{v:+.0f}'
    return s.replace('-', '−')


def text(x, y, s, size=12, weight=400, fill=INK, anchor='start'):
    svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke, w=1., op=1.):
    svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}" stroke-opacity="{op}" stroke-linecap="round"/>')


def bins_ratio(S, key, q):
    """10 ms bins over [-300, -10) and the 5 ms bin [-10, -5): event / random rate; centers."""
    r = S['rates'][key]
    ev, rd = np.asarray(r[f'event_{q}_per_ms']), np.asarray(r[f'random_{q}_per_ms'])
    xs, ys = [], []
    for a in np.arange(-300., -10., 10.):
        m = (lo >= a) & (lo < a + 10.)
        xs.append(a + 5.)
        ys.append(ev[m].sum() / rd[m].sum() if rd[m].sum() > 0 else np.nan)
    m = (lo >= -10.) & (lo < -5.)
    xs.append(-7.5)
    ys.append(ev[m].sum() / rd[m].sum() if rd[m].sum() > 0 else np.nan)
    return np.asarray(xs), np.asarray(ys)


def at_send(S, key, q):
    """The ratio over the 20 ms around the median send ([-60, -40))."""
    r = S['rates'][key]
    ev, rd = np.asarray(r[f'event_{q}_per_ms']), np.asarray(r[f'random_{q}_per_ms'])
    m = (lo >= -60.) & (lo < -40.)
    return ev[m].sum() / rd[m].sum()


def panel(x0, y0, w, h, title, key, q, log=True):
    X = lambda t: x0 + (t - TMIN) * w / (TMAX - TMIN)
    if log:
        Y = lambda v: y0 + h - (math.log10(min(max(v, YMIN), YMAX)) - math.log10(YMIN)) * h / (math.log10(YMAX) - math.log10(YMIN))
    else:
        Y = lambda v: y0 + h - v * h
    text(x0, y0 - 30, title, 13, 600)
    svg.append(f'<rect x="{X(arr["p25"]):.1f}" y="{y0:.1f}" width="{X(arr["p75"]) - X(arr["p25"]):.1f}" height="{h:.1f}" fill="{TOP}" fill-opacity="0.08"/>')
    ticks = (.1, .3, 1., 3., 10., 30., 100.) if log else (0., .25, .5, .75, 1.)
    for v in ticks:
        line(x0, Y(v), x0 + w, Y(v), AXIS if v in (1.,) or (not log and v == 0.) else GRID, 1)
        lab = (f'{v:g}×' if log else (f'{v:+.2f}' if v else '0'))
        text(x0 - 6, Y(v) + 4, lab, 11, 400, MUTED, 'end')
    for t in (-300, -200, -100, 0):
        text(X(t), y0 + h + 16, mfmt(t) if t else '0', 11, 400, MUTED, 'middle')
    line(X(arr['p50']), y0, X(arr['p50']), y0 + h, TOP, 1, .9)
    line(X(0.), y0, X(0.), y0 + h, INK2, 1.5)
    if log:
        for S, col in ((S_ALL, ALL), (S_TOP, TOP)):
            xs, ys = bins_ratio(S, key, q)
            ok = np.isfinite(ys)
            pts = ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(xs[ok], ys[ok]))
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        vt, va = at_send(S_TOP, key, q), at_send(S_ALL, key, q)
        text(x0, y0 - 12, f'at its send: {vt:.1f}× normal (all first moves {va:.1f}×)', 12, 400, INK2)
        return vt, va
    lags = sorted(float(k.split('_')[1]) for k in S_TOP['state'] if k.startswith('imb_'))
    for S, col in ((S_ALL, ALL), (S_TOP, TOP)):
        xs = [-x for x in lags if x > 0]
        ys = [S['state'][f'imb_{int(x)}']['mean'] for x in lags if x > 0]
        o = np.argsort(xs)
        pts = ' '.join(f'{X(xs[i]):.1f},{Y(ys[i]):.1f}' for i in o)
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i in o:
            svg.append(f'<circle cx="{X(xs[i]):.1f}" cy="{Y(ys[i]):.1f}" r="5.5" fill="{SURF}"/><circle cx="{X(xs[i]):.1f}" cy="{Y(ys[i]):.1f}" r="3.5" fill="{col}"/>')
    line(x0, Y(rand_imb), x0 + w, Y(rand_imb), MUTED, 1)
    text(x0 + 4, Y(rand_imb) - 5, f'random times {rand_imb:+.2f}', 11, 400, INK2)
    it = float(np.interp(-50., [-x for x in lags[::-1]], [S_TOP['state'][f'imb_{int(x)}']['mean'] for x in lags[::-1]]))
    ia = float(np.interp(-50., [-x for x in lags[::-1]], [S_ALL['state'][f'imb_{int(x)}']['mean'] for x in lags[::-1]]))
    text(x0, y0 - 12, f'at its send: {it:+.2f} (all first moves {ia:+.2f})', 12, 400, INK2)
    return it, ia


# header
text(24, 34, "What Binance does around the top taker's send (t = 0: the move; first moves only)", 19, 600)
text(24, 58, f"Orange: the {S_TOP['n_events']:,} first moves the top taker's orders landed ahead of. Gray: {S_ALL['n_events']:,} sampled first moves "
             "(no move in the same direction in the previous 300 ms).", 13, 400, INK2)
text(24, 78, "Flows are a multiple of the same flow at random times nearby (log scale). Shaded: the middle half of its send times; orange line: the median.", 13, 400, INK2)

# top strip: when its orders reach Predict
x0s, ws, y0s, hs = 120, 1080, 118, 70
Xs = lambda t: x0s + (t - TMIN) * ws / (TMAX - TMIN)
hist = np.asarray(arr['hist_5ms'], float)
hmax = hist[lo < 0].max()
for a, c in zip(lo, hist):
    if a < 0 and c > 0:
        bh = c / hmax * hs
        svg.append(f'<rect x="{Xs(a) + 1:.1f}" y="{y0s + hs - bh:.1f}" width="{ws / 60 - 2:.1f}" height="{bh:.1f}" rx="2" fill="{TOP}"/>')
line(x0s, y0s + hs, x0s + ws, y0s + hs, AXIS, 1)
line(Xs(arr['p50']), y0s - 4, Xs(arr['p50']), y0s + hs, INK, 1.5)
line(Xs(0.), y0s - 4, Xs(0.), y0s + hs, INK2, 1.5)
text(Xs(arr['p50']) - 8, y0s + 8, f"its order reaches Predict: median {mfmt(arr['p50'])} ms (middle half {mfmt(arr['p25'])} to {mfmt(arr['p75'])})", 12, 600, INK, 'end')
text(Xs(0.) - 6, y0s + hs + 16, 'move', 11, 600, INK2, 'end')
for t in (-300, -250, -200, -150, -100, -50):
    text(Xs(t), y0s + hs + 16, mfmt(t), 11, 400, MUTED, 'middle')
text(24, y0s + 30, 'When it', 12, 600)
text(24, y0s + 46, 'sends', 12, 600)

# small multiples
PW, PH = 316, 170
cols = (112, 500, 888)
rows = (300, 590)
spec = [('Trades their way', 'opp_trade', 'btc'), ('Trades against them', 'near_trade', 'btc'), ('New orders on the far side', 'opp_add', 'btc'),
        ('Cancels on the far side', 'opp_cancel', 'btc'), ('New orders on their side', 'near_add', 'btc'), ('Imbalance their way (mean)', None, None)]
vals = {}
for i, (title, key, q) in enumerate(spec):
    vals[title] = panel(cols[i % 3], rows[i // 3], PW, PH, title, key, q, log=key is not None)
# legend
ly = HT - 58
for j, (col, nm) in enumerate(((TOP, 'moves the top taker caught'), (ALL, 'all first moves'), (MUTED, 'random times (imbalance)'))):
    lx = 24 + j * 300
    line(lx, ly, lx + 22, ly, col, 2)
    text(lx + 30, ly + 4, nm, 12, 400, INK2)
text(24, HT - 30, "x axis of every panel: ms from the Binance move. Flows in BTC per ms (counts for ticks); 10 ms bins, the last point is the 5 ms before the move.", 12, 400, MUTED)
text(24, HT - 12, "Send = its order reaching Predict (settlement − 150 − 34 ms, about ±15 ms); four already-used days; only its filled orders are visible.", 12, 400, MUTED)

body = '\n'.join(svg)
SVG = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}" font-family="{FONT}" role="img" '
       f'aria-label="Binance flows around the top taker send">'
       f'<rect width="{W}" height="{HT}" fill="{SURF}"/>\n{body}\n</svg>\n')
(H / 'timeline_chart2.svg').write_text(SVG)
rows_t = ''.join(f'<tr><td>{esc(k)}</td><td class="n">{v[0]:.2f}</td><td class="n">{v[1]:.2f}</td></tr>' for k, v in vals.items())
HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Binance around the top taker's send</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 24px; background: #f9f9f7; color: {INK}; font-family: {FONT}; }}
figure {{ margin: 0; background: {SURF}; border-radius: 8px; box-shadow: 0 0 0 1px rgba(11,11,11,0.10); padding: 8px; max-width: {W + 16}px; }}
svg {{ width: 100%; height: auto; display: block; }}
table {{ border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid {GRID}; }}
th {{ color: {INK2}; font-weight: 600; }}
td.n {{ font-variant-numeric: tabular-nums; text-align: right; }}
caption {{ text-align: left; font-weight: 600; padding-bottom: 8px; }}
</style></head><body>
<figure>{SVG}</figure>
<table><caption>Table view: at its send (60 to 40 ms before the move); flows as a multiple of random times, imbalance as a mean</caption>
<tr><th>Binance</th><th>Moves the top taker caught</th><th>All first moves</th></tr>{rows_t}</table>
</body></html>
"""
(H / 'timeline_chart2.html').write_text(HTML)
print(json.dumps({k: [round(v[0], 2), round(v[1], 2)] for k, v in vals.items()}))
