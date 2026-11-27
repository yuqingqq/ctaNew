"""Draw group3_profile.json: the top taker's group-3 fires against every move (the opportunity set) and our chase model's top 10 %, and the
share of moves it chases (USER 2026-11-27: "can we check group 3 fires of top taker, what's the distribution of remaining time, firing
probability, include more information for analysis, check what we still miss"). Every number is read from group3_profile.json.
Writes group3_profile_chart.html (write-once).
Usage: python3 predictfun/reviews/take_reprice_20261125/group3_profile_chart.py [<out dir>]
"""
import html
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import signal_vs_noise_chart as SC  # noqa: E402

OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 else H
J = json.loads((H / 'group3_profile.json').read_text())
LIGHT = dict(SC.LIGHT, s3='#1baf7a')
DARK = dict(SC.DARK, s3='#199e70')
SERIES = (('its_group3', "its group-3 fires", 's1', 'circle'), ('every_move_onset', 'every move (first tick)', 's2', 'square'),
          ('our_model_top10', "our model's top 10%", 's3', 'triangle'))
PW, PH, ML, MR, MT, MB = 400, 262, 48, 14, 58, 50
RULES = SC.RULES + ' .viz .s3l{stroke:var(--s3)} .viz .s3f{fill:var(--s3)}'


NAMES = dict(b_lfar1='far side thicker', x_far_r300='far side grown over 300 ms', b_imb1='imbalance its way', b_spread='Binance spread',
             b_lnear1='near side thicker', u_okx_swap_gap_d300='OKX vs Binance futures over 300 ms', x_near_r300='near side grown over 300 ms',
             chase_gap_c='Predict still to catch up', u_okx_swap_gap_d5s='OKX vs Binance futures over 5 s', chase_implied_c='implied Predict change',
             x_imb_d100='imbalance change over 100 ms', pf_extreme='Predict price far from 50', fv_delta='Predict sensitivity to BTC')


def fmt_edge(x, unit=''):
    if x is None or abs(x) >= 1e17:
        return None
    if abs(x - round(x)) < .02:
        x = round(x)
    return f'{x:g}{unit}'.replace('-', '\u2212')


def labels_of(buckets, unit=''):
    out = []
    for b in buckets:
        lo, hi = fmt_edge(b['lo'], unit), fmt_edge(b['hi'], unit)
        out.append(f'< {hi}' if lo is None or (b['lo'] <= -1e17) else (f'>= {lo}' if hi is None or b['hi'] >= 1e17 else (f'{lo}…{hi}' if b['lo'] < 0 else f'{lo}-{hi}')))
    return out


def axis(v):
    """The smallest top of 4 or 5 round steps (1 / 2 / 2.5 / 5 x 10^p) that holds v -> (top, steps)."""
    best = None
    for n in (4, 5):
        for p_ in range(-4, 4):
            for m in (1, 2, 2.5, 5):
                st = m * 10. ** p_
                if st * n >= v:
                    if best is None or st * n < best[0] - 1e-12:
                        best = (st * n, n)
                    break
            else:
                continue
            break
    return best


def marker(shape, x, y, cls):
    if shape == 'circle':
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="4" stroke="var(--surf)" stroke-width="2"/>'
    if shape == 'square':
        return f'<rect class="{cls}" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1" stroke="var(--surf)" stroke-width="2"/>'
    return f'<path class="{cls}" d="M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y + 4:.1f} L{x - 5:.1f},{y + 4:.1f} Z" stroke="var(--surf)" stroke-width="2"/>'


def frame(title, sub, cats, ymax, yfmt, pid, n=4):
    a = [f'<svg class="viz" role="img" aria-labelledby="{pid}t" viewBox="0 0 {PW} {PH}" width="{PW}" height="{PH}">',
         f'<rect class="bg" width="{PW}" height="{PH}"/>', f'<text id="{pid}t" class="t1" x="{ML - 36}" y="18" font-size="13.5" font-weight="600">{html.escape(title)}</text>',
         f'<text class="t3" x="{ML - 36}" y="35" font-size="11.5">{html.escape(sub)}</text>']
    x0, x1, y0, y1 = ML, PW - MR, PH - MB, MT
    for k in range(n + 1):
        v = ymax * k / n
        y = y0 - (y0 - y1) * k / n
        a.append(f'<line class="gr" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke-width="1"/>')
        a.append(f'<text class="t3" x="{x0 - 6}" y="{y + 4:.1f}" font-size="10.5" text-anchor="end">{yfmt(v)}</text>')
    step = (x1 - x0) / len(cats)
    xs = [x0 + step * (i + .5) for i in range(len(cats))]
    wid = lambda c: sum({'.': 3., '-': 3.6, ' ': 3., '<': 6., '>': 6., '=': 6., '\u2026': 10., '\u2212': 7.5}.get(ch, 5.8) for ch in c)
    tilt = max(wid(c) for c in cats) > step - 6                           # labels wider than their slot are angled
    for x, c in zip(xs, cats):
        if tilt:
            a.append(f'<text class="t3" x="{x + 4:.1f}" y="{y0 + 13}" font-size="10" text-anchor="end" transform="rotate(-35 {x + 4:.1f} {y0 + 13})">{html.escape(c)}</text>')
        else:
            a.append(f'<text class="t3" x="{x:.1f}" y="{y0 + 15}" font-size="10" text-anchor="middle">{html.escape(c)}</text>')
    return a, xs, (lambda v: y0 - (y0 - y1) * min(v, ymax) / ymax), step


def lines_panel(fld, title, sub, unit='', pid='p'):
    D = J['distributions'][fld]
    cats = labels_of(D['its_group3']['buckets'], unit)
    vals = {s: [b['share'] for b in D[s]['buckets']] for s, *_ in SERIES if s in D}
    ymax, n = axis(max(max(v) for v in vals.values()) * 1.05)
    a, xs, sy, _ = frame(title, sub, cats, ymax, lambda v: f'{100 * v:.0f}%', pid, n)
    for s, nm, tok, shape in SERIES:
        if s not in vals:
            continue
        pts = ' '.join(f'{x:.1f},{sy(v):.1f}' for x, v in zip(xs, vals[s]))
        a.append(f'<polyline class="{tok}l" points="{pts}" fill="none" stroke-width="2"/>')
        for x, v, c in zip(xs, vals[s], cats):
            tip = f'{nm}: {100 * v:.1f}% in {c} (n = {D[s]["n"]:,})'
            a.append(f'<g data-tip="{html.escape(tip)}">{marker(shape, x, sy(v), tok + "f")}<circle cx="{x:.1f}" cy="{sy(v):.1f}" r="10" fill="transparent"/></g>')
    a.append('</svg>')
    rows = ''.join(f'<tr><td>{html.escape(c)}</td>' + ''.join(f'<td>{100 * vals[s][i]:.1f}%</td>' for s, *_ in SERIES if s in vals) + '</tr>' for i, c in enumerate(cats))
    head = ''.join(f'<th>{html.escape(nm)} (n = {D[s]["n"]:,})</th>' for s, nm, *_ in SERIES if s in vals)
    return ''.join(a), f'<table><caption>{html.escape(title)}</caption><tr><th>bucket</th>{head}</tr>{rows}</table>'


def bars_panel(cats, vals, title, sub, yfmt, tipf, pid, ref=None):
    ymax, n = axis(max(v for v in vals if v is not None) * 1.05)
    a, xs, sy, step = frame(title, sub, cats, ymax, yfmt, pid, n)
    w = min(28., step * .62)
    y0 = PH - MB
    for x, v, c in zip(xs, vals, cats):
        if v is None:
            continue
        y = sy(v)
        h_ = max(y0 - y, 0.)
        r = min(4., h_ / 2.)
        path = (f'M{x - w / 2:.1f},{y0:.1f} L{x - w / 2:.1f},{y + r:.1f} Q{x - w / 2:.1f},{y:.1f} {x - w / 2 + r:.1f},{y:.1f} L{x + w / 2 - r:.1f},{y:.1f} '
                f'Q{x + w / 2:.1f},{y:.1f} {x + w / 2:.1f},{y + r:.1f} L{x + w / 2:.1f},{y0:.1f} Z')
        a.append(f'<g data-tip="{html.escape(tipf(c, v))}"><path class="s1f" d="{path}"/><rect x="{x - step / 2:.1f}" y="{MT}" width="{step:.1f}" height="{y0 - MT}" fill="transparent"/></g>')
    if ref is not None:
        yr = sy(ref[0])
        a.append(f'<line class="mk" x1="{ML}" x2="{PW - MR}" y1="{yr:.1f}" y2="{yr:.1f}" stroke-width="1" stroke-dasharray="4 3"/>')
        a.append(f'<line class="mk" x1="{ML - 36}" x2="{ML - 14}" y1="47" y2="47" stroke-width="1" stroke-dasharray="4 3"/>')
        a.append(f'<text class="t2" x="{ML - 10}" y="51" font-size="10.5">{html.escape(ref[1])}</text>')
    a.append('</svg>')
    rows = ''.join(f'<tr><td>{html.escape(c)}</td><td>{"" if v is None else html.escape(yfmt(v, True))}</td></tr>' for c, v in zip(cats, vals))
    return ''.join(a), f'<table><caption>{html.escape(title)}</caption><tr><th>bucket</th><th>value</th></tr>{rows}</table>'


def main():
    fn = OUTD / 'group3_profile_chart.html'
    if fn.exists():
        raise SystemExit('chart exists (write-once)')
    Dd, FP, CO, SPD, W1, W2 = J['distributions'], J['firing_probability'], J['coverage'], J['speed'], J['what_we_miss'], J['what_we_miss_at_its_decision']
    g3n = Dd['time_left_s']['its_group3']['n']
    q = lambda f, s='its_group3', p='p50': Dd[f][s]['q'][p]
    panels, tables = [], []
    for i, (fld, t, s, u) in enumerate((('time_left_s', 'Time left in the 5-minute window', 's at its arrival / our decision', ''),
                                        ('side_best_c', 'Price of the side it buys', 'best price of its side, c', ''),
                                        ('move_bps', 'Binance move so far', 'mid its way since the onset, bps', ''),
                                        ('far_size_btc', 'Binance far side left', 'best-level size that must empty, BTC', ''),
                                        ('chase_gap_c', 'Predict still to catch up', 'implied minus actual change since the onset, c', ''),
                                        ('drift_since_open_bps', 'BTC since the window opened', 'spot drift its way, bps', ''))):
        svg, tb = lines_panel(fld, t, s, u, pid=f'p{i}')
        panels.append(svg); tables.append(tb)
    pct = lambda v, full=False: f'{100 * v:.{1 if full else 0}f}%'
    pct1 = lambda v, full=False: f'{100 * v:.1f}%'
    for i, (k, t, s) in enumerate((('side_best_c', 'How often it chases a move: by price', 'share of moves it chases, by its side\'s best price, c'),
                                   ('move_size_first_300ms_bps', 'How often it chases a move: by size', 'by the move\'s size in its first 300 ms, bps (known later)'))):
        b = FP[k]
        cats = labels_of(b)
        svg, tb = bars_panel(cats, [x['chased'] for x in b], t, s, pct1, lambda c, v, b=b: f'{100 * v:.2f}% of moves chased ({c})', f'q{i}', ref=(FP['overall'], f'all moves {100 * FP["overall"]:.1f}%'))
        panels.append(svg); tables.append(tb)
    b = FP['our_best_edge_in_move']
    svg, tb = bars_panel([str(x['decile']) for x in b], [x['chased'] for x in b], 'How often it chases: by our model', 'by the move\'s best edge in our model, decile (10 = best)', pct1,
                         lambda c, v: f'decile {c}: {100 * v:.2f}% of moves chased', 'q2', ref=(FP['overall'], f'all moves {100 * FP["overall"]:.1f}%'))
    panels.append(svg); tables.append(tb)
    cats = ['<= 10 ms', '10-25.5 ms', '> 25.5 ms']
    keys = ('within_10ms', '10ms_to_our_25.5ms', 'after_our_25.5ms')
    svg, tb = bars_panel(cats, [SPD[k]['share'] for k in keys], 'How soon after the last tick it lands', 'share of its group-3 fires; our chase lands 25.5 ms after a tick', pct,
                         lambda c, v: f'{100 * v:.0f}% of its group-3 fires land {c} after the last tick', 'q3')
    panels.append(svg); tables.append(tb)
    c1 = lambda v, full=False: f'{v:+.{2 if full else 1}f}c'
    svg, tb = bars_panel(cats, [SPD[k]['v1_mean_c'] for k in keys], 'Its value by speed', 'its 1 s value, c per share after fee', c1,
                         lambda c, v: f'landing {c} after the last tick: {v:+.2f}c per share at 1 s', 'q4')
    panels.append(svg); tables.append(tb)
    bb = CO['its_v1_by_our_percentile']
    svg, tb = bars_panel([x['bucket'] for x in bb], [x['v1_mean_c'] for x in bb], 'Its value by our model\'s rating', 'its 1 s value by our percentile at our last tick before it', c1,
                         lambda c, v: f'our rating {c}: its 1 s value {v:+.2f}c per share', 'q5')
    panels.append(svg); tables.append(tb)
    legend = ''.join(f'<span class="lg"><svg width="16" height="12">{marker(sh, 8, 6, tok + "f")}</svg>{html.escape(nm)}</span>' for _, nm, tok, sh in SERIES)
    beyond = ', '.join(f"{NAMES.get(x['input'], x['input'])} ({'more' if x['auc'] > .5 else 'less'} in its chases, AUC {x['auc']:.2f})" for x in W2['inputs_beyond_null'][:6])
    intro = (f"<h1>The top taker's group-3 fires: its order lands after a Binance move its way had started</h1>"
             f"<p>{g3n:,} fires on the consumed days (four days; it traded on three), against every Binance move at its first tick (our chase population) and our chase model's top 10%. "
             f"Median time left {q('time_left_s'):.0f} s, its side's price {q('side_best_c'):.0f}c (moves overall {q('side_best_c', 'every_move_onset'):.0f}c, our top 10% {q('side_best_c', 'our_model_top10'):.0f}c), "
             f"the move so far {q('move_bps'):.2f} bps at tick {q('tick_index'):.0f}, landing {q('arrival_after_onset_ms'):.0f} ms after the onset and {q('arrival_after_last_tick_ms'):.0f} ms after the last tick; "
             f"it pays {q('pay_through_c'):.0f}c above the best it could see (median) and marks {Dd['its_group3_outcomes']['v1_mean_c']:+.2f}c per share at 1 s. "
             f"It chases {100 * FP['overall']:.1f}% of moves. At the same model edge and tick, the moves it chases move Predict {W2['realised_predict_move_1s_c']['chased']:+.1f}c vs "
             f"{W2['realised_predict_move_1s_c']['matched']:+.1f}c; the inputs that separate them most: {html.escape(beyond)}; a shuffled label reaches {W2['null_p95']:.3f} from 0.5 at its 95th percentile.</p>"
             f"<p class='lgd'>{legend}</p>")
    style = (f'.viz-root{{color-scheme:light;{SC.tokens(LIGHT)}}} @media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;{SC.tokens(DARK)}}}}} '
             f':root[data-theme="dark"] .viz-root{{color-scheme:dark;{SC.tokens(DARK)}}} {RULES} '
             f'body{{margin:0}} .viz-root{{background:var(--surf);color:var(--ink);font-family:{SC.FONT};padding:12px 24px 40px}} h1{{font-size:21px;margin:8px 0 4px}} '
             f'p{{max-width:1240px;margin:4px 0 10px;color:var(--ink2);font-size:14px;line-height:1.5}} .grid{{display:grid;grid-template-columns:repeat(3,{PW}px);gap:14px 18px}} '
             f'.lg{{display:inline-flex;align-items:center;gap:6px;margin-right:18px;color:var(--ink);font-size:13px}} '
             f'#tip{{position:absolute;display:none;pointer-events:none;background:var(--surf);color:var(--ink);border:1px solid var(--grid);border-radius:6px;padding:7px 9px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15)}} '
             f'details{{margin-top:18px}} table{{border-collapse:collapse;margin:8px 0 16px;font-size:12.5px}} th,td{{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right}} '
             f'th:first-child,td:first-child{{text-align:left}} caption{{text-align:left;color:var(--ink2);padding:4px 0}}')
    js = """<script>const tip=document.getElementById('tip');document.querySelectorAll('[data-tip]').forEach(g=>{g.addEventListener('mousemove',ev=>{tip.textContent=g.dataset.tip;tip.style.display='block';tip.style.left=(ev.pageX+14)+'px';tip.style.top=(ev.pageY+14)+'px';});g.addEventListener('mouseleave',()=>{tip.style.display='none';});});</script>"""
    fn.write_text(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>The top taker\'s group-3 fires</title><style>{style}</style></head><body><div class="viz-root">{intro}'
                  f'<div class="grid">{"".join(panels)}</div><details><summary>Table view</summary>{"".join(tables)}</details><div id="tip"></div></div>{js}</body></html>')
    print('wrote', fn.name, len(panels), 'panels')


if __name__ == '__main__':
    main()
