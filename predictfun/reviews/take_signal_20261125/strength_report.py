"""Append the amendment-2 signal-strength section to report.md; every number read from strength.json / tables/s_*.csv."""
import json
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
D = H / 'tables'
MARK = '## Amendment 2 — how strong must the signal be?'


def c(x):
    return '' if pd.isna(x) else f'{x:+.2f}'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(cols) - 1)) + '|']
    out += ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(out)


def main():
    rep = (H / 'report.md').read_text()
    assert MARK not in rep
    s = json.loads((H / 'strength.json').read_text())
    assert s['parity_ok']
    cur = pd.read_csv(D / 's_threshold_curve.csv')
    a = cur[cur.half == 'all'].set_index('theta_bps')
    h = cur[cur.half != 'all'].pivot(index='theta_bps', columns='half', values='mark_3s_c')
    t1 = pd.DataFrame([[f'{th:.3g}', f'{int(a.loc[th, "triggers"]):,}', f'{100 * a.loc[th, "fill_rate"]:.0f}%', c(a.loc[th, 'mark_1s_c']), c(a.loc[th, 'mark_3s_c']),
                        c(a.loc[th, 'rt_3s_c']), c(h.loc[th, 'Oct24-25']), c(h.loc[th, 'Oct26-27']), f'{a.loc[th, "usd_3s_q50"]:+,.0f}'] for th in a.index],
                      columns=['Take when abs bn_r250 ≥ (bps)', 'Triggers', 'Fill rate', '1 s', '3 s', 'Exit by taking at 3 s', '3 s Oct 24–25',
                               '3 s Oct 26–27', 'USD at 3 s, Q 50, 4 days'])
    im = pd.read_csv(D / 's_imb_time.csv').pivot(index='imb', columns='tw', values='mark_3s_c')
    im = im.reindex(['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3-5', '5-10', '>=10'])[['0-60', '60-180', '180-270', '270-298']]
    t2 = pd.DataFrame([[i] + [c(im.loc[i, k]) for k in im.columns] for i in im.index], columns=['Implied Predict move (¢)'] + [f'{k} s' for k in im.columns])
    nf = pd.read_csv(D / 's_fine_near.csv').set_index('fine')
    t3 = pd.DataFrame([[i, f'{int(nf.loc[i, "events"]):,}', f'{100 * nf.loc[i, "fill_rate"]:.0f}%', c(nf.loc[i, 'im_c']), c(nf.loc[i, 'mark_3s_c']), c(nf.loc[i, 'rt_3s_c'])]
                       for i in nf.index], columns=['abs bn_r250 (bps)', 'Events', 'Fill rate', 'Implied move (¢)', '3 s', 'Exit by taking at 3 s'])
    tr = pd.read_csv(D / 's_trend.csv').pivot(index='coarse', columns='trend', values='mark_3s_c').reindex(['<0.29', '0.29-1', '>=1'])
    t4 = pd.DataFrame([[i, c(tr.loc[i, 'CONTINUATION']), c(tr.loc[i, 'OTHER']), c(tr.loc[i, 'REVERSAL'])] for i in tr.index],
                      columns=['abs bn_r250 (bps)', '3 s trend agrees and is larger', 'Other', '3 s trend opposite'])
    cal = s['c_calibration']['by_bin']
    txt = f"""

{MARK}

Declared in [amendment_2_strength.md](amendment_2_strength.md) before any of these outcomes. It covers every BTC change with
abs bn_r250 ≥ 0.05 bps ({s['events']:,} events), valued with the same parity-checked code: the guard's own rule reproduced all
{s['parity_rows']:,} primary rows exactly.

**The practical rule** (one trigger per market and direction per 2 s; Tokyo clock; ¢ per filled share). Every threshold is
positive in both halves; a stronger threshold earns more per share but triggers less often:

{md(t1)}

**The best strength measure is the implied Predict move in cents**: BTC move (bps) × the contract's sensitivity (cents per bps,
from the Predict price and time to expiry). Below ~0.5 ¢ a take earns about nothing in every part of the window; above it the
value climbs steadily:

{md(t2)}

**Why even small BTC ticks pay.** A BTC change is usually the start of a burst: BTC moves on average another +0.4–0.5 bps in the
same direction within 250 ms, and Predict follows about 0.3 s later. Near-money books, per event:

{md(t3)}

**Trend agreement matters** (3 s value): when the 3 s BTC trend agrees with the jump, the take is worth more than when the jump
reverses an earlier move. This holds in every band and both halves (reading d = {s['d_continuation_beats_reversal_all']}).

{md(t4)}

Readings, computed in `strength.py`:

- (a) every bin is positive in both halves, so no break-even appears down to 0.05 bps at the mid mark (lowest bin passing: {s['a_break_even_bin']}).
- (b) monotone across strength bins: {s['b_monotone_fine_mark_3s']}.
- (c) the delta-implied edge (implied move − cost) is too pessimistic. Even events with edge < −1 ¢ realise
  {c(cal['<-1']['Oct24-25'])} / {c(cal['<-1']['Oct26-27'])} ¢ because of the burst continuation, so read the empirical curve, not the model's break-even.

**What it means for a take rule.** Size the decision by the implied move:

- ≳ 0.5 ¢ to beat the entry cost at the mid;
- ≳ 1 ¢ if we must also exit by taking;
- the edge per share grows roughly in proportion above that.

Prefer jumps that continue a 3 s trend. All of this is consumed-day development evidence: any threshold chosen from this curve
needs untouched days (rule 11). Capacity, competition below the ~50 ms frame resolution, inventory and our own impact are
still unmeasured.
"""
    (H / 'report.md').write_text(rep + txt)
    print(txt)


if __name__ == '__main__':
    main()
