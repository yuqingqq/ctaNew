"""Draw fire_gallery_examples.json: 8 randomly drawn top-taker fires, each next to its nearest look-alike and an ordinary moment of the same
half hour, with SIGNAL_VS_NOISE's chart (USER 2026-11-27: "can you show me more charts of real fires vs normal conditions"). Every number is
read from fire_gallery_examples.json and visual_twins.json. Writes fire_gallery_chart.html (write-once) and one SVG per fire.
Usage: python3 predictfun/reviews/take_reprice_20261125/fire_gallery_chart.py [<out dir> [<examples dir>]]
"""
import datetime as dt
import html
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import signal_vs_noise_chart as SC  # noqa: E402

OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 else H
IN = Path(sys.argv[2]) if len(sys.argv) > 2 else H
G = json.loads((IN / 'fire_gallery_examples.json').read_text())
VTJ = json.loads((H / 'visual_twins.json').read_text())
ORDER = ('fire', 'lookalike', 'ordinary')
SC.HT = 830


def when(e):
    return dt.datetime.fromtimestamp(e['t'] / 1000., dt.timezone.utc).strftime('%m-%d %H:%M:%S UTC')


def outcome(e):
    m = e['first_move_their_way_ms']
    return f'moved +{m:.0f} ms' if (m is not None and m <= SC.XMAX) else 'no move in 300 ms'


def figure(k, s):
    bet = 'up' if s['fire']['d'] > 0 else 'down'
    titles = {'fire': f"Fire {k + 1}: the top taker's order", 'lookalike': 'Closest look-alike (no fire)', 'ordinary': 'Ordinary moment (within 30 min)'}
    head = (f"Fire {k + 1} of {len(G['sets'])} · {when(s['fire'])} · bet {bet}",
            f"Binance futures best level, 1 s before to 0.3 s after each moment; far side = the {'ask' if bet == 'up' else 'bid'} (must empty for a move their way).")
    lines = [(f"Look-alike: {when(s['lookalike'])}, distance {s['lookalike']['distance_to_fire']:.2f} on the nine key inputs. "
              f"Ordinary moment: {when(s['ordinary'])}.", 't3')]
    svg, S, geo, tops = SC.build(s, ORDER, titles, head, lines)
    svg = svg.replace('id="svn-t"', f'id="g{k}t"').replace('id="svn-d"', f'id="g{k}d"').replace('aria-labelledby="svn-t svn-d"', f'aria-labelledby="g{k}t g{k}d"')
    for c in range(3):
        svg = svg.replace(f'id="xh{c}"', f'id="g{k}xh{c}"')
    return svg, S, geo, tops


JS = """<script>
const ALL = %s, G = %s, X0 = %s, X1 = %s, Y0 = %d, Y1 = %d;
const tip = document.getElementById('tip');
function at(t, x){let lo = 0, hi = t.length - 1, j = -1; while (lo <= hi){const m = (lo + hi) >> 1; if (t[m] <= x){j = m; lo = m + 1;} else hi = m - 1;} return j;}
document.querySelectorAll('svg.viz').forEach((svg, k) => {
  const hide = () => [0, 1, 2].forEach(c => document.getElementById(`g${k}xh${c}`).setAttribute('visibility', 'hidden'));
  svg.addEventListener('mousemove', ev => {
    const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse());
    hide();
    const c = G.findIndex(g => q.x >= g.x0 && q.x <= g.x0 + g.w);
    if (c < 0 || q.y < Y0 || q.y > Y1){tip.style.display = 'none'; return;}
    const g = G[c], x = X0 + (q.x - g.x0) / g.w * (X1 - X0), e = ALL[k][c], j = at(e.t, x);
    const l = document.getElementById(`g${k}xh${c}`); l.setAttribute('x1', q.x); l.setAttribute('x2', q.x); l.setAttribute('visibility', 'visible');
    if (j < 0){tip.style.display = 'none'; return;}
    let up = 0, dn = 0; for (let i = 0; i < e.tt.length; i++){ if (e.tt[i] > x - 10 && e.tt[i] <= x){ if (e.w[i] > 0) up += e.q[i]; else dn += e.q[i]; } }
    tip.innerHTML = `<b>${x >= 0 ? '+' : ''}${x.toFixed(0)} ms</b><br>far side ${e.far[j].toFixed(3)} BTC<br>near side ${e.near[j].toFixed(3)} BTC<br>mid ${e.mid[j] >= 0 ? '+' : ''}${e.mid[j].toFixed(2)} bps<br>trades, last 10 ms: ${up.toFixed(3)} their way, ${dn.toFixed(3)} against`;
    tip.style.display = 'block'; tip.style.left = (ev.pageX + 14) + 'px'; tip.style.top = (ev.pageY + 14) + 'px';
  });
  svg.addEventListener('mouseleave', () => {tip.style.display = 'none'; hide();});
});
</script>"""


def main():
    fh = OUTD / 'fire_gallery_chart.html'
    if fh.exists():
        raise SystemExit('chart exists (write-once)')
    figs, rows, data = [], [], []
    geo = tops = None
    lit = SC.RULES
    for k_, v_ in SC.LIGHT.items():
        lit = lit.replace(f'var(--{k_})', v_)
    for k, s in enumerate(G['sets']):
        svg, S, geo, tops = figure(k, s)
        figs.append(svg)
        (OUTD / f'fire_gallery_{k + 1}.svg').write_text(svg.replace('class="viz">', f'class="viz"><style>{lit}</style>', 1))
        data.append([dict(t=s[r]['t_rel_ms'], far=s[r]['far_size'], near=s[r]['near_size'], mid=s[r]['mid_bps'], tt=s[r]['trades_t_rel_ms'], q=s[r]['trades_btc'],
                          w=s[r]['trades_their_way']) for r in ORDER])
        for r, nm in zip(ORDER, ('fire', 'closest look-alike', 'ordinary moment')):
            x = S[r]
            rows.append(f"<tr><td>{k + 1}</td><td>{nm}</td><td>{when(s[r])}</td><td>{'up' if s[r]['d'] > 0 else 'down'}</td><td>{100 * x['score']:.1f}%</td><td>{x['imb']:+.2f}</td>"
                        f"<td>{x['far']:.2f}</td><td>{x['near']:.2f}</td><td>{x['its_300']:.3f}</td><td>{x['ag_300']:.3f}</td><td>{html.escape(outcome(s[r]))}</td></tr>")
    mr = VTJ['move_rate']
    moved = sum(1 for s in G['sets'] if s['fire']['y'] == 1)
    intro = (f"<h1>Real fires against normal conditions</h1><p>{len(G['sets'])} of the top taker's {G['fires_in_pool']:,} fires on the four consumed days, drawn at random "
             f"(seed {G['seed']}, not chosen by eye); {moved} of them were followed by a Binance move their way within 300 ms (all its fires: {100 * mr['fires']:.0f}%). "
             f"Next to each: the ordinary moment whose book looks most like it on nine key inputs (no fire) and a random ordinary moment within 30 min (same bet). "
             f"Across all fires, the closest look-alikes moved {100 * mr['twins_any_time']:.1f}% of the time and random moments {100 * mr['random']:.1f}% (VISUAL_TWINS). "
             f"t = 0: for the fire, its order reaching Predict. Hover a panel for values; the table at the end lists every moment.</p>")
    table = ('<table><caption>Every moment at t = 0 (table view)</caption><tr><th>fire</th><th>moment</th><th>time</th><th>bet</th><th>public score</th><th>imbalance their way</th>'
             f'<th>far side, BTC</th><th>near side, BTC</th><th>trades their way, last 300 ms</th><th>trades against, last 300 ms</th><th>outcome</th></tr>{"".join(rows)}</table>')
    style = (f'.viz-root{{color-scheme:light;{SC.tokens(SC.LIGHT)}}} @media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;{SC.tokens(SC.DARK)}}}}} '
             f':root[data-theme="dark"] .viz-root{{color-scheme:dark;{SC.tokens(SC.DARK)}}} {SC.RULES} '
             f'body{{margin:0}} .viz-root{{background:var(--surf);color:var(--ink);font-family:{SC.FONT};padding:12px 12px 40px}} h1{{font-size:22px;margin:8px 70px 4px}} '
             f'p{{max-width:1120px;margin:4px 70px 14px;color:var(--ink2);font-size:14px;line-height:1.5}} svg.viz{{display:block;margin-bottom:10px}} '
             f'#tip{{position:absolute;display:none;pointer-events:none;background:var(--surf);color:var(--ink);border:1px solid var(--grid);border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.45;box-shadow:0 2px 8px rgba(0,0,0,.15)}} '
             f'table{{border-collapse:collapse;margin:8px 70px 24px;font-size:13px}} th,td{{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:right}} '
             f'th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}} caption{{text-align:left;color:var(--ink2);padding:6px 0}}')
    js = JS % (json.dumps(data), json.dumps(geo), SC.XMIN, SC.XMAX, tops[0], tops[-1] + SC.ROWS[-1][2])
    fh.write_text(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Real fires against normal conditions</title><style>{style}</style></head>'
                  f'<body><div class="viz-root">{intro}{"".join(figs)}{table}<div id="tip"></div></div>{js}</body></html>')
    print('wrote', fh.name, 'and', len(figs), 'SVGs')


if __name__ == '__main__':
    main()
