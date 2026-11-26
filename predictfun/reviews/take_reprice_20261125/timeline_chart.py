"""Draw timeline.json as one picture (USER 2026-11-26: "show me a clear timeline of the events ... with clear time, and what we do, I want
to check the difference"). Every number on it is read from timeline.json / fire_arrival.json; nothing is typed in.
Top panel: four lanes on one clock, t = 0 when Binance's top level empties (the price moves): Binance futures, the top taker's order, Predict's
makers, our REPRICE-8 order; medians as dots, the middle half (p25-p75) as whiskers / bars, the 150 ms speed bump as a tinted bar.
Bottom panel, same x: the mean best-level imbalance toward the move before the top taker's moves and before all Binance moves, random times
as a hairline. Writes timeline_chart.svg and timeline_chart.html (the SVG plus a table view) next to this file.
Usage: python3 predictfun/reviews/take_reprice_20261125/timeline_chart.py
"""
import html
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
T = json.loads((H / 'timeline.json').read_text())
FA = json.loads((H / 'fire_arrival.json').read_text())
t1, rp, bu = T['top1'], T['reprice8'], T['build_up']
HOLD = 150.

INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
TOP, US = '#eb6834', '#2a78d6'                                # categorical slots 2 / 1 (validated pair); context in gray
FONT = "system-ui, -apple-system, 'Segoe UI', 'DejaVu Sans', sans-serif"
W, X0, X1, TMIN, TMAX = 1240, 210, 1190, -320., 240.
Y_L = dict(bn=160, top=250, mk=340, us=430)
Y_GAP, Y_AX = 512, 548
B_TOP, B_BOT = 640, 790
HT = 864


def X(t):
    return X0 + (t - TMIN) * (X1 - X0) / (TMAX - TMIN)


def YB(v):
    return B_BOT - v * (B_BOT - B_TOP)


def fmt(v, sign=True):
    s = f'{v:+.1f}' if abs(v - round(v)) > 1e-9 else f'{v:+.0f}'
    return (s if sign else s.lstrip('+')).replace('-', '\u2212')


def esc(s):
    return html.escape(s, quote=True)


svg = []


def text(x, y, s, size=13, weight=400, fill=INK, anchor='start', title=None):
    t = f'<title>{esc(title)}</title>' if title else ''
    svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{t}{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke, w=1., op=1.):
    svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}" stroke-opacity="{op}" stroke-linecap="round"/>')


def dot(x, y, fill, title, r=5.5, hollow=False):
    f = SURF if hollow else fill
    svg.append(f'<g><title>{esc(title)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2:.1f}" fill="{SURF}"/>'
               f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{f}" stroke="{fill}" stroke-width="{2.5 if hollow else 0}"/>'
               f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="transparent"/></g>')


def whisker(t_lo, t_hi, y, col, title):
    svg.append(f'<g><title>{esc(title)}</title>')
    line(X(t_lo), y, X(t_hi), y, col, 2)
    line(X(t_lo), y - 6, X(t_lo), y + 6, col, 2)
    line(X(t_hi), y - 6, X(t_hi), y + 6, col, 2)
    svg.append(f'<rect x="{X(t_lo):.1f}" y="{y - 12:.1f}" width="{X(t_hi) - X(t_lo):.1f}" height="24" fill="transparent"/></g>')


def bar(t_lo, t_hi, y, col, op, title, h=8):
    svg.append(f'<g><title>{esc(title)}</title><rect x="{X(t_lo):.1f}" y="{y - h / 2:.1f}" width="{X(t_hi) - X(t_lo):.1f}" height="{h}" rx="4" fill="{col}" fill-opacity="{op}"/>'
               f'<rect x="{X(t_lo):.1f}" y="{y - 12:.1f}" width="{X(t_hi) - X(t_lo):.1f}" height="24" fill="transparent"/></g>')


# ---- the numbers (all read from the JSON) ----
arr = t1['arrival_rel_move_ms']                                     # top taker's arrival relative to the move
a25, a50, a75 = arr['p25'], arr['p50'], arr['p75']
m50 = a50 + HOLD
trig, dec, uarr, umatch = rp['trigger_after_move_ms'], rp['decision_after_move_ms'], rp['arrival_after_move_ms'], rp['match_after_move_ms']
mk = rp['predict_makers_pull_after_move_ms']
lagx = [-float(k) for k in bu['top1'] if k != 'n']
ib_top = [bu['top1'][k]['imb_mean'] for k in bu['top1'] if k != 'n']
ib_all = [bu['all'][k]['imb_mean'] for k in bu['all'] if k != 'n']
o = np.argsort(lagx)
lagx, ib_top, ib_all = np.asarray(lagx)[o], np.asarray(ib_top)[o], np.asarray(ib_all)[o]
pre = lagx < 0                                                      # the state just before the move (0 = after its update)
rand_imb = FA['groups']['TOP1']['trajectory_at_arrival']['imb_0']['random_mean']
gap_arr = uarr[1] - a50
gap_match = umatch[1] - m50
imb_at_top = float(np.interp(a50, lagx[pre], ib_top[pre]))
n_top, n_all = bu['top1']['n'], bu['all']['n']
ahead, after_since = t1['ahead'], t1['after_a_move_since_ms']['p50']

# ---- header ----
text(24, 34, "One race on one clock: t = 0 is when Binance's top level empties (the price moves)", 19, 600)
text(24, 58, f"Dots are medians; whiskers and pale bars span the middle half (25th\u201375th percentile). Top taker: {n_top:,} of its orders that reached Predict "
             f"before a move that came within 300 ms.", 13, 400, INK2)
text(24, 78, "Us: the REPRICE-8 replay (decision = trigger + 3.5 ms, 22 ms assumed path). Predict makers: when the stale quotes REPRICE aimed at were pulled. "
             "Four already-used days.", 13, 400, INK2)

# ---- grid, axis ----
for t in range(-300, 201, 50):
    line(X(t), 122, X(t), Y_GAP + 8, GRID, 1)
    line(X(t), B_TOP - 6, X(t), B_BOT, GRID, 1)
    text(X(t), Y_AX + 4, fmt(float(t)) if t else '0', 12, 400, MUTED, 'middle')
text(X(TMAX), Y_AX + 22, 'ms from the Binance move', 12, 400, MUTED, 'end')
line(X(0), 104, X(0), B_BOT, INK2, 1.5)
text(X(0) + 8, 112, 'Binance top level empties', 12, 600, INK)

# ---- lanes ----
for key, (name, sub) in dict(bn=('Binance futures', 'top of book'), top=('Top taker', 'its order on Predict'), mk=('Predict makers', 'the stale quote'),
                             us=('Us (REPRICE-8)', 'our order on Predict')).items():
    text(X0 - 26, Y_L[key] - 2, name, 14, 600, INK, 'end')
    text(X0 - 26, Y_L[key] + 15, sub, 12, 400, INK2, 'end')
    if key != 'us':
        line(X(TMIN), Y_L[key] + 55, X(TMAX), Y_L[key] + 55, GRID, 1)

# Binance: the build-up and the move
y = Y_L['bn']
bar(-300., 0., y, MUTED, .22, f'book leans toward the move: imbalance {ib_top[0]:+.2f} at -300 ms -> {ib_top[pre][-1]:+.2f} at -5 ms', 10)
text(X(-300), y - 14, f'book leans toward the move: imbalance {ib_top[0]:+.2f} at {fmt(lagx[0])} ms \u2192 {ib_top[pre][-1]:+.2f} at {fmt(lagx[pre][-1])} ms', 12, 400, INK2)
dot(X(0), y, INK2, 'Binance: the top level on the far side empties, the mid moves', 6)

# top taker
y = Y_L['top']
whisker(a25, a75, y, TOP, f'top taker order reaches Predict: middle half {fmt(a25)} to {fmt(a75)} ms')
bar(a50, m50, y + 24, TOP, .25, f'top taker: 150 ms speed bump, {fmt(a50)} to {fmt(m50)} ms')
dot(X(a50), y, TOP, f'top taker order reaches Predict: median {fmt(a50)} ms')
dot(X(m50), y + 24, TOP, f'top taker order matches the stale quote: median {fmt(m50)} ms', 5.5, True)
text(X(a50), y - 16, f'order reaches Predict {fmt(a50)} ms', 13, 600, INK, 'middle')
text((X(a50) + X(m50)) / 2, y + 46, '150 ms speed bump', 12, 400, INK2, 'middle')
text(X(m50) + 12, y + 28, f'matches {fmt(m50)} ms', 13, 600, INK)
text(X(-300), y + 46, f'{100 * (1 - ahead):.0f}% of its orders land just after a move (median {after_since:.0f} ms after)', 12, 400, INK2)

# Predict makers
y = Y_L['mk']
bar(mk['p25'], mk['p75'], y, MUTED, .35, f"Predict makers pull the stale quote: middle half {fmt(mk['p25'])} to {fmt(mk['p75'])} ms")
dot(X(mk['p50']), y, INK2, f"Predict makers pull the stale quote: median {fmt(mk['p50'])} ms")
text(X(mk['p50']), y - 16, f"pull the stale quote {fmt(mk['p50'])} ms", 13, 600, INK, 'middle')

# us
y = Y_L['us']
whisker(uarr[0], uarr[2], y, US, f'our order reaches Predict: middle half {fmt(uarr[0])} to {fmt(uarr[2])} ms')
bar(uarr[1], umatch[1], y + 24, US, .25, f'our 150 ms speed bump, {fmt(uarr[1])} to {fmt(umatch[1])} ms')
line(X(trig[1]), y - 7, X(trig[1]), y + 7, US, 2)
dot(X(uarr[1]), y, US, f'our order reaches Predict: median {fmt(uarr[1])} ms')
dot(X(umatch[1]), y + 24, US, f'our order matches if the quote is still there: median {fmt(umatch[1])} ms', 5.5, True)
text(X(0) - 8, y + 4, f'sees the move {fmt(trig[1])} ms, decides {fmt(dec[1])} ms', 12, 400, INK2, 'end')
text(X(uarr[1]) + 10, y - 16, f'order reaches Predict {fmt(uarr[1])} ms', 13, 600, INK)
text((X(uarr[1]) + X(umatch[1])) / 2, y + 46, '150 ms speed bump', 12, 400, INK2, 'middle')
text(X(umatch[1]) - 12, y + 6, f'matches {fmt(umatch[1])} ms', 13, 600, INK, 'end')

# the gap
yg = Y_GAP
svg.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>')
svg.append(f'<line x1="{X(a50):.1f}" y1="{yg:.1f}" x2="{X(uarr[1]) - 4:.1f}" y2="{yg:.1f}" stroke="{INK}" stroke-width="1.5" marker-end="url(#ah)"/>')
text(X(a50) - 10, yg + 4, f'we land {gap_arr:.1f} ms after the top taker', 13, 600, INK, 'end')
text(X(uarr[1]) + 10, yg + 4, f'and match {gap_match:.1f} ms after it ({fmt(m50)} vs {fmt(umatch[1])} ms)', 12, 400, INK2)
for xg, y0, col in ((X(a50), Y_L['top'] + 8, TOP), (X(uarr[1]), Y_L['us'] + 8, US)):
    line(xg, y0, xg, Y_GAP - 8, col, 1, .45)
    line(xg, B_TOP - 6, xg, B_BOT, col, 1, .45)

# ---- bottom panel: imbalance before the move ----
text(24, B_TOP - 26, "Best-level imbalance toward the move before it happens (mean; +1 = the far side is empty)", 14, 600)
for v in (0., .25, .5, .75, 1.):
    line(X(TMIN), YB(v), X(TMAX), YB(v), GRID if v else AXIS, 1)
    text(X(TMIN) - 8, YB(v) + 4, f'{v:+.2f}' if v else '0', 12, 400, MUTED, 'end')
line(X(TMIN), YB(rand_imb), X(0), YB(rand_imb), MUTED, 1)
text(X(-300), YB(rand_imb) - 6, f'random times {rand_imb:+.2f}', 12, 400, INK2)


def series(xs, ys, col, name):
    pts = ' '.join(f'{X(a):.1f},{YB(b):.1f}' for a, b in zip(xs, ys))
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for a, b in zip(xs, ys):
        dot(X(a), YB(b), col, f'{name}: {b:+.2f} at {fmt(a)} ms', 4)


series(lagx[pre], ib_all[pre], INK2, f'before every Binance move ({n_all:,} sampled)')
series(lagx[pre], ib_top[pre], TOP, 'before the moves the top taker caught')
text(X(lagx[pre][-1]) + 12, YB(ib_all[pre][-1]) + 4, f'{ib_all[pre][-1]:+.2f} every move', 12, 600, INK)
text(X(lagx[pre][-1]) + 12, YB(ib_top[pre][-1]) + 4, f"{ib_top[pre][-1]:+.2f} the top taker's moves", 12, 600, INK)
dot(X(a50), YB(imb_at_top), TOP, f'imbalance when the top taker order lands: {imb_at_top:+.2f}', 5)
yl = YB(.14)
line(X(a50), YB(imb_at_top) + 7, X(a50) - 30, yl - 12, INK2, 1)
text(X(a50) - 34, yl - 2, f'imbalance {imb_at_top:+.2f} when the top taker order lands', 12, 400, INK2, 'end')
# legend (two series + the baseline), top left where the lines are low
lx, ly = X(-300), B_TOP + 12
for i, (col, nm) in enumerate(((TOP, 'before the moves the top taker caught'), (INK2, f'before every Binance move ({n_all:,} sampled)'), (MUTED, 'random times'))):
    line(lx, ly + 20 * i, lx + 22, ly + 20 * i, col, 2)
    text(lx + 30, ly + 20 * i + 4, nm, 12, 400, INK2)
text(24, HT - 30, "Clocks: Binance event time; Predict from settlement ids (arrival = settlement \u2212 150 \u2212 34 ms, about \u00b115 ms per order); "
                  "the Binance / Predict clock offset is not measured.", 12, 400, MUTED)
text(24, HT - 12, "Only the top taker's filled orders are visible. Point estimates on Oct 23, Oct 31, Nov 1 and Nov 2.", 12, 400, MUTED)

body = '\n'.join(svg)
SVG = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}" font-family="{FONT}" role="img" '
       f'aria-label="Timeline of one race: Binance move at 0 ms, top taker order at {fmt(a50)} ms, our order at {fmt(uarr[1])} ms">'
       f'<rect width="{W}" height="{HT}" fill="{SURF}"/>\n{body}\n</svg>\n')
(H / 'timeline_chart.svg').write_text(SVG)

rows = [('Binance top level empties (the move)', '0', '-', 'definition of t = 0'),
        ('Top taker order reaches Predict', fmt(a50), f'{fmt(a25)} to {fmt(a75)}', f'{n_top:,} orders ahead of a move within 300 ms (settlement clock)'),
        ('Top taker order matches (after the 150 ms bump)', fmt(m50), f'{fmt(a25 + HOLD)} to {fmt(a75 + HOLD)}', 'arrival + 150 ms'),
        ('Predict makers pull the stale quote', fmt(mk['p50']), f"{fmt(mk['p25'])} to {fmt(mk['p75'])}", 'REPRICE events, quotes pulled'),
        ('We see the move (REPRICE-8 trigger)', fmt(trig[1]), f'{fmt(trig[0])} to {fmt(trig[2])}', 'replay'),
        ('We decide', fmt(dec[1]), f'{fmt(dec[0])} to {fmt(dec[2])}', 'trigger + 1.5 ms feed + 2 ms decision'),
        ('Our order reaches Predict', fmt(uarr[1]), f'{fmt(uarr[0])} to {fmt(uarr[2])}', '+ 22 ms assumed path'),
        ('Our order matches if the quote is still there', fmt(umatch[1]), f'{fmt(umatch[0])} to {fmt(umatch[2])}', '+ 150 ms bump')]
tbl = ''.join(f'<tr><td>{esc(a)}</td><td class="n">{esc(b)}</td><td class="n">{esc(c)}</td><td>{esc(d)}</td></tr>' for a, b, c, d in rows)
HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>One race on one clock</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 24px; background: #f9f9f7; color: {INK}; font-family: {FONT}; }}
figure {{ margin: 0; background: {SURF}; border-radius: 8px; box-shadow: 0 0 0 1px rgba(11,11,11,0.10); padding: 8px; max-width: {W + 16}px; }}
svg {{ width: 100%; height: auto; display: block; }}
table {{ border-collapse: collapse; margin-top: 20px; font-size: 14px; max-width: {W}px; }}
th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid {GRID}; }}
th {{ color: {INK2}; font-weight: 600; }}
td.n {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
caption {{ text-align: left; font-weight: 600; padding-bottom: 8px; }}
</style></head><body>
<figure>{SVG}</figure>
<table><caption>Table view: ms from the Binance move (median; middle half = 25th-75th percentile)</caption>
<tr><th>Event</th><th>Median</th><th>Middle half</th><th>Source</th></tr>{tbl}</table>
</body></html>
"""
(H / 'timeline_chart.html').write_text(HTML)
print('wrote', H / 'timeline_chart.svg', H / 'timeline_chart.html', f'gap {gap_arr:.1f} ms, imbalance at the top taker arrival {imb_at_top:+.2f}')
