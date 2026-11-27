"""Draw signal_vs_noise_examples.json: the Binance futures book around a typical top-taker fire, a look-alike our public score rates the
same (no fire, no move) and an ordinary moment (USER 2026-11-27: "can you show me a typical signal they extract and a normal one without
the signal"). Every number is read from signal_vs_noise_examples.json / signal_vs_noise.json. Columns = examples; rows = best-level
sizes (far side = the side that must empty for a move their way; near side), aggressive trades per 10 ms (their way up, against down), the
Binance mid since t (bps). x = ms from t (the fire: its order's arrival at Predict). Writes signal_vs_noise_chart.svg / .html (write-once).
Usage: python3 predictfun/reviews/take_reprice_20261125/signal_vs_noise_chart.py [<out dir>]
"""
import html
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 else H
EX = json.loads((H / 'signal_vs_noise_examples.json').read_text())
RES = json.loads((H / 'signal_vs_noise.json').read_text())
ORDER = ('typical_fire', 'lookalike_no_move', 'ordinary_moment')
TITLES = {'typical_fire': "Typical fire: the top taker's order", 'lookalike_no_move': 'Look-alike: same score, no fire, no move',
          'ordinary_moment': 'Ordinary moment'}
LIGHT = dict(surf='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', s1='#2a78d6', s2='#eb6834')
DARK = dict(surf='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#a3a29b', grid='#383835', s1='#3987e5', s2='#d95926')
FONT = "system-ui, -apple-system, 'Segoe UI', 'DejaVu Sans', sans-serif"
XMIN, XMAX, BIN = -1000., 300., 10.
W, HT = 1260, 930
ML, MR, MT, GAPX, GAPY = 70, 40, 184, 40, 26
SMIN = .1
ROWS = (('size', 'Best-level size, BTC (log scale; under 0.1 drawn at 0.1)', 230), ('trades', 'Aggressive trades, BTC per 10 ms', 150), ('mid', 'Binance mid since t, bps', 130))
RULES = """.viz .bg{fill:var(--surf)} .viz .t1{fill:var(--ink)} .viz .t2{fill:var(--ink2)} .viz .t3{fill:var(--muted)} .viz .gr{stroke:var(--grid)}
.viz .mk{stroke:var(--muted)} .viz .s1l{stroke:var(--s1)} .viz .s2l{stroke:var(--s2)} .viz .s1f{fill:var(--s1)} .viz .s2f{fill:var(--s2)}
.viz .inkl{stroke:var(--ink)} .viz .dot{fill:var(--surf);stroke:var(--ink)} .viz .xh{stroke:var(--ink2)}"""


def tokens(p):
    return ''.join(f'--{k}:{v};' for k, v in p.items())


def idx_at(e, x):
    return int(np.searchsorted(np.asarray(e['t_rel_ms'], float), x, 'right') - 1)


def summary(e):
    j0, j3 = idx_at(e, 0.), idx_at(e, -300.)
    far, near = np.asarray(e['far_size'], float), np.asarray(e['near_size'], float)
    tt, q, w = np.asarray(e['trades_t_rel_ms'], float), np.asarray(e['trades_btc'], float), np.asarray(e['trades_their_way'], float)
    last = (tt > -300.) & (tt <= 0.)
    mv = e['first_move_their_way_ms']
    return dict(far=float(far[j0]), near=float(near[j0]), imb=float((near[j0] - far[j0]) / (near[j0] + far[j0])),
                far_300=float(far[j3]) if j3 >= 0 else None, near_300=float(near[j3]) if j3 >= 0 else None,
                its_300=float(q[last & (w > 0)].sum()), ag_300=float(q[last & (w < 0)].sum()),
                move=mv if (mv is not None and mv <= XMAX) else None, score=float(e['score']), d=float(e['d']))


def nice(v):
    for p in range(-2, 4):
        for m in (1, 2, 2.5, 5):
            if m * 10 ** p >= v:
                return m * 10 ** p
    return v


def steps(t, v, sx, sy, j):
    d_ = f'M{sx(XMIN):.1f},{sy(v[j]):.1f}'
    last = v[j]
    for tx, vv in zip(t[j + 1:], v[j + 1:]):
        if tx > XMAX:
            break
        d_ += f' H{sx(tx):.1f} V{sy(vv):.1f}'; last = vv
    return d_ + f' H{sx(XMAX):.1f}', last


def build():
    S = {k: summary(EX[k]) for k in ORDER}
    lo = {k: max(idx_at(EX[k], XMIN), 0) for k in ORDER}
    ymax_size = nice(max(max(EX[k]['far_size'][lo[k]:] + EX[k]['near_size'][lo[k]:]) for k in ORDER) * 1.05)
    lmin, lmax = np.log10(SMIN), np.log10(ymax_size)
    bins = np.arange(XMIN, XMAX + BIN, BIN)
    tr = {}
    for k in ORDER:
        e = EX[k]
        tt, q, w = np.asarray(e['trades_t_rel_ms'], float), np.asarray(e['trades_btc'], float), np.asarray(e['trades_their_way'], float)
        tr[k] = (np.histogram(tt[w > 0], bins, weights=q[w > 0])[0], np.histogram(tt[w < 0], bins, weights=q[w < 0])[0])
    ymax_tr = nice(max(max(u.max(), d_.max()) for u, d_ in tr.values()) * 1.05 or 1.)
    ymax_mid = nice(max(max(np.abs(np.asarray(EX[k]['mid_bps'], float)[lo[k]:]).max() for k in ORDER), .5) * 1.1)
    colw = (W - ML - MR - 2 * GAPX) / 3.
    tops, y = [], MT
    for _, _, h in ROWS:
        tops.append(y); y += h + GAPY
    out = []
    a = out.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}" font-family="{FONT}" role="img" aria-labelledby="svn-t svn-d" class="viz">')
    a('<title id="svn-t">The Binance book before a top-taker fire, a look-alike our model scores the same, and an ordinary moment</title>')
    a('<desc id="svn-d">Three columns of three panels: best-level sizes, aggressive trades and the Binance mid change, from -1000 to +300 ms around each moment.</desc>')
    a(f'<rect class="bg" x="0" y="0" width="{W}" height="{HT}"/>')
    a(f'<text class="t1" x="{ML}" y="34" font-size="21" font-weight="600">What the top taker reacts to, next to a moment our model cannot tell apart from it</text>')
    a(f'<text class="t2" x="{ML}" y="58" font-size="14">Binance futures best level from 1 s before to 0.3 s after each moment (t = 0: for the fire, its order reaching Predict). '
      f'All three are bets on up: the far side is the ask.</text>')
    lx = ML
    for cls, lab in (('s1', 'Far side (must empty for the price to move their way); trades hitting it'), ('s2', 'Near side; trades hitting it')):
        a(f'<rect class="{cls}f" x="{lx}" y="78" width="16" height="4" rx="2"/><text class="t2" x="{lx + 22}" y="84" font-size="13">{html.escape(lab)}</text>')
        lx += 520 if cls == 's1' else 230
    a(f'<line class="mk" x1="{lx}" y1="80" x2="{lx + 18}" y2="80" stroke-width="1.5" stroke-dasharray="3 3"/><text class="t2" x="{lx + 24}" y="84" font-size="13">t = 0</text>')
    geo = []
    for c, k in enumerate(ORDER):
        e, s_ = EX[k], S[k]
        x0 = ML + c * (colw + GAPX)
        sx = lambda x, x0=x0: x0 + (x - XMIN) / (XMAX - XMIN) * colw
        geo.append(dict(x0=x0, w=colw))
        a(f'<text class="t1" x="{x0}" y="{MT - 62}" font-size="15" font-weight="600">{html.escape(TITLES[k])}</text>')
        mv = f"moved their way +{s_['move']:.0f} ms" if s_['move'] is not None else 'no move in 300 ms'
        a(f'<text class="t2" x="{x0}" y="{MT - 44}" font-size="12.5">public score {100 * s_["score"]:.1f}% · imbalance their way {s_["imb"]:+.2f}</text>')
        a(f'<text class="t2" x="{x0}" y="{MT - 28}" font-size="12.5">far {s_["far"]:.2f} · near {s_["near"]:.2f} BTC · {mv}</text>')
        for r, (key, lab, h) in enumerate(ROWS):
            yt = tops[r]
            if key == 'size':
                sy = lambda v, yt=yt, h=h: yt + h - (np.log10(max(v, SMIN)) - lmin) / (lmax - lmin) * h
                ticks = np.array([x for x in (.1, .3, 1., 3., 10., 30., 100.) if x <= ymax_size])
            else:
                m_ = ymax_tr if key == 'trades' else ymax_mid
                sy = lambda v, yt=yt, h=h, m_=m_: yt + h / 2. - v / m_ * (h / 2.)
                ticks = np.array([-m_, -m_ / 2, 0, m_ / 2, m_])
            for tv in ticks:
                yy = sy(tv)
                a(f'<line class="gr" x1="{x0}" y1="{yy:.1f}" x2="{x0 + colw:.1f}" y2="{yy:.1f}" stroke-width="1"/>')
                if c == 0:
                    a(f'<text class="t3" x="{x0 - 8}" y="{yy + 4:.1f}" font-size="11" text-anchor="end">{(abs(tv) if key == "trades" else tv):g}</text>')
            if c == 0:
                a(f'<text class="t2" x="{x0}" y="{yt - 7}" font-size="12">{html.escape(lab)}</text>')
            if r == len(ROWS) - 1:
                for xv in (-1000, -750, -500, -250, 0, 250):
                    a(f'<text class="t3" x="{sx(xv):.1f}" y="{yt + h + 16}" font-size="11" text-anchor="middle">{xv:+d}</text>' if xv else
                      f'<text class="t3" x="{sx(xv):.1f}" y="{yt + h + 16}" font-size="11" text-anchor="middle">0</text>')
            a(f'<line class="mk" x1="{sx(0):.1f}" y1="{yt}" x2="{sx(0):.1f}" y2="{yt + h}" stroke-width="1.5" stroke-dasharray="3 3"/>')
            t = np.asarray(e['t_rel_ms'], float)
            if key == 'size':
                ends = {}
                for ser, cls, nm in (('far_size', 's1', 'far'), ('near_size', 's2', 'near')):
                    d_, last = steps(t, np.asarray(e[ser], float), sx, sy, lo[k])
                    a(f'<path class="{cls}l" d="{d_}" fill="none" stroke-width="2" stroke-linejoin="round"><title>{nm} side size</title></path>')
                    ends[nm] = sy(last)
                if abs(ends['far'] - ends['near']) < 12:                       # keep the two end labels apart
                    m_ = (ends['far'] + ends['near']) / 2.
                    hi_, lo_ = ('far', 'near') if ends['far'] <= ends['near'] else ('near', 'far')
                    ends[hi_], ends[lo_] = m_ - 6, m_ + 6
                for nm in ('far', 'near'):
                    a(f'<text class="t2" x="{sx(XMAX) + 4:.1f}" y="{ends[nm] + 4:.1f}" font-size="11">{nm}</text>')
            elif key == 'trades':
                up, dn = tr[k]
                bw = colw / len(up)
                for i_, (u, dd) in enumerate(zip(up, dn)):
                    xx = sx(bins[i_]) + .5
                    if u > 0:
                        a(f'<rect class="s1f" x="{xx:.1f}" y="{sy(u):.1f}" width="{max(bw - 1., 1.):.1f}" height="{sy(0) - sy(u):.1f}" rx="1"><title>{bins[i_]:+.0f} ms: {u:.3f} BTC their way</title></rect>')
                    if dd > 0:
                        a(f'<rect class="s2f" x="{xx:.1f}" y="{sy(0):.1f}" width="{max(bw - 1., 1.):.1f}" height="{sy(-dd) - sy(0):.1f}" rx="1"><title>{bins[i_]:+.0f} ms: {dd:.3f} BTC against</title></rect>')
                if c == 0:
                    a(f'<text class="t3" x="{x0 + 4}" y="{yt + 12}" font-size="10.5">their way</text><text class="t3" x="{x0 + 4}" y="{yt + h - 4}" font-size="10.5">against</text>')
            else:
                v = np.asarray(e['mid_bps'], float)
                d_, _ = steps(t, v, sx, sy, lo[k])
                a(f'<path class="inkl" d="{d_}" fill="none" stroke-width="2" stroke-linejoin="round"><title>Binance mid since t, bps</title></path>')
                if s_['move'] is not None:
                    vm = v[idx_at(e, s_['move'])]
                    a(f'<circle class="dot" cx="{sx(s_["move"]):.1f}" cy="{sy(vm):.1f}" r="5" stroke-width="2"><title>first move their way: +{s_["move"]:.0f} ms</title></circle>')
                    a(f'<text class="t2" x="{sx(s_["move"]) - 8:.1f}" y="{sy(vm) - 10:.1f}" font-size="11.5" text-anchor="end">moves their way, +{s_["move"]:.0f} ms</text>')
        a(f'<text class="t3" x="{x0 + colw / 2:.1f}" y="{tops[-1] + ROWS[-1][2] + 34}" font-size="11.5" text-anchor="middle">ms from t</text>')
        a(f'<line class="xh" id="xh{c}" x1="0" y1="{MT}" x2="0" y2="{tops[-1] + ROWS[-1][2]}" stroke-width="1" visibility="hidden"/>')
    la, tog, imi = RES['lookalikes'], RES['all_together'], RES['imitation']
    yb = tops[-1] + ROWS[-1][2] + 70
    lines = [(f"Across all {la['fires']:,} of its fires on these days, Binance moved their way within 300 ms after {100 * la['fire_move_rate']:.0f}%; after the {la['controls']:,} look-alikes "
              f"our public score rates the same, {100 * la['lookalike_move_rate']:.1f}%.", 't1'),
             (f"A model can tell its fires from the look-alikes (held-out AUC {tog['auc']:.2f}): more imbalance building, more of the far side withdrawn, as in the left column.", 't1'),
             (f"But that look does not predict moves at ordinary moments (AUC {imi['auc_fire_likeness_for_moves']:.2f}): what tells it when to fire is not in the data we record.", 't1'),
             ('Four consumed days; development evidence. Examples: the fire nearest the median fire on nine key book inputs, and its nearest look-alike.', 't3')]
    for i_, (ln, cls) in enumerate(lines):
        a(f'<text class="{cls}" x="{ML}" y="{yb + 22 * i_}" font-size="{13.5 if cls == "t1" else 11.5}">{html.escape(ln)}</text>')
    a('</svg>')
    return '\n'.join(out), S, geo, tops


JS = """<script>
const D = %s, G = %s, X0 = %s, X1 = %s, Y0 = %d, Y1 = %d;
const svg = document.querySelector('svg.viz'), tip = document.getElementById('tip');
function at(t, x){let lo = 0, hi = t.length - 1, j = -1; while (lo <= hi){const m = (lo + hi) >> 1; if (t[m] <= x){j = m; lo = m + 1;} else hi = m - 1;} return j;}
svg.addEventListener('mousemove', ev => {
  const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse());
  G.forEach((g, c) => document.getElementById('xh' + c).setAttribute('visibility', 'hidden'));
  const c = G.findIndex(g => q.x >= g.x0 && q.x <= g.x0 + g.w);
  if (c < 0 || q.y < Y0 || q.y > Y1){tip.style.display = 'none'; return;}
  const g = G[c], x = X0 + (q.x - g.x0) / g.w * (X1 - X0), e = D[c], j = at(e.t, x);
  const l = document.getElementById('xh' + c); l.setAttribute('x1', q.x); l.setAttribute('x2', q.x); l.setAttribute('visibility', 'visible');
  if (j < 0){tip.style.display = 'none'; return;}
  let up = 0, dn = 0; for (let i = 0; i < e.tt.length; i++){ if (e.tt[i] > x - 10 && e.tt[i] <= x){ if (e.w[i] > 0) up += e.q[i]; else dn += e.q[i]; } }
  tip.innerHTML = `<b>${x >= 0 ? '+' : ''}${x.toFixed(0)} ms</b><br>far side ${e.far[j].toFixed(3)} BTC<br>near side ${e.near[j].toFixed(3)} BTC<br>mid ${e.mid[j] >= 0 ? '+' : ''}${e.mid[j].toFixed(2)} bps<br>trades, last 10 ms: ${up.toFixed(3)} their way, ${dn.toFixed(3)} against`;
  tip.style.display = 'block'; tip.style.left = (ev.pageX + 14) + 'px'; tip.style.top = (ev.pageY + 14) + 'px';
});
svg.addEventListener('mouseleave', () => {tip.style.display = 'none'; G.forEach((g, c) => document.getElementById('xh' + c).setAttribute('visibility', 'hidden'));});
</script>"""


def main():
    svg, S, geo, tops = build()
    fs, fh = OUTD / 'signal_vs_noise_chart.svg', OUTD / 'signal_vs_noise_chart.html'
    if fs.exists() or fh.exists():
        raise SystemExit('chart exists (write-once)')
    lit = RULES
    for k_, v_ in LIGHT.items():
        lit = lit.replace(f'var(--{k_})', v_)
    fs.write_text(svg.replace('class="viz">', f'class="viz"><style>{lit}</style>', 1))          # literal colours: static renderers lack CSS variables
    style = (f'.viz-root{{color-scheme:light;{tokens(LIGHT)}}} @media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;{tokens(DARK)}}}}} '
             f':root[data-theme="dark"] .viz-root{{color-scheme:dark;{tokens(DARK)}}} {RULES} '
             f'body{{margin:0}} .viz-root{{background:var(--surf);color:var(--ink);font-family:{FONT};padding:12px;min-height:100vh}} '
             f'#tip{{position:absolute;display:none;pointer-events:none;background:var(--surf);color:var(--ink);border:1px solid var(--grid);border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.45;box-shadow:0 2px 8px rgba(0,0,0,.15)}} '
             f'table{{border-collapse:collapse;margin:8px 70px 24px;font-size:13px}} th,td{{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right}} '
             f'th:first-child,td:first-child{{text-align:left}} caption{{text-align:left;color:var(--ink2);padding:6px 0}}')

    def f2(x):
        return '' if x is None else f'{x:.2f}'
    rows = ''.join(f"<tr><td>{html.escape(TITLES[k])}</td><td>{100 * S[k]['score']:.1f}%</td><td>{S[k]['imb']:+.2f}</td><td>{S[k]['far']:.2f}</td><td>{f2(S[k]['far_300'])}</td>"
                   f"<td>{S[k]['near']:.2f}</td><td>{f2(S[k]['near_300'])}</td><td>{S[k]['its_300']:.3f}</td><td>{S[k]['ag_300']:.3f}</td>"
                   f"<td>{('+%.0f ms' % S[k]['move']) if S[k]['move'] is not None else 'none'}</td></tr>" for k in ORDER)
    table = ('<table><caption>The three moments at t = 0 (table view of the chart)</caption><tr><th>moment</th><th>public score</th><th>imbalance their way</th>'
             '<th>far side, BTC</th><th>far side 300 ms before</th><th>near side, BTC</th><th>near side 300 ms before</th><th>trades their way, last 300 ms, BTC</th>'
             f'<th>trades against, last 300 ms, BTC</th><th>first move their way</th></tr>{rows}</table>')
    data = [dict(t=EX[k]['t_rel_ms'], far=EX[k]['far_size'], near=EX[k]['near_size'], mid=EX[k]['mid_bps'], tt=EX[k]['trades_t_rel_ms'], q=EX[k]['trades_btc'],
                 w=EX[k]['trades_their_way']) for k in ORDER]
    js = JS % (json.dumps(data), json.dumps(geo), XMIN, XMAX, tops[0], tops[-1] + ROWS[-1][2])
    fh.write_text(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Signal vs noise: a top-taker fire, a look-alike, an ordinary moment</title><style>{style}</style></head>'
                  f'<body><div class="viz-root">{svg}{table}<div id="tip"></div></div>{js}</body></html>')
    print('wrote', fs.name, fh.name)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
