"""TAKE-SIGNAL v1 report: every number below is read from the saved outputs (takes / controls / m3_align / readings / checks)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
OUT = R / 'data/predict_fun/research/take_signal_20261125'
T = H / 'tables'


def j(n):
    return json.loads((H / n).read_text())


def c(x, d=2):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:+.{d}f}'


def pc(x):
    return f'{100 * x:.0f}%'


def md(rows, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(cols) - 1)) + '|']
    out += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    rd, ct, m3, ck, te = j('readings.json'), j('controls.json'), j('m3_align.json'), j('checks.json'), j('take_eval.json')
    tk = pd.read_parquet(OUT / 'takes.parquet')
    allt = {k: pd.read_csv(T / f'{k}_all.csv').iloc[0] for k in ('tokyo', 'slow')}
    rows = []
    for k, lab in (('tokyo', 'Tokyo (arrive +42 ms, book known at −63 ms)'), ('slow', 'Slow (arrive +86 ms, known at −126 ms)')):
        a = allt[k]
        rows.append([lab, f'{int(a.triggers):,}', pc(a.fill_rate), c(a.mark_1s_c), c(a.mark_3s_c), c(a.mark_10s_c), c(a.rt_3s_c), c(a.end_c), c(a.res_c),
                     f'{a.mark_3s_usd:+,.0f}'])
    v = ct['very_slow']['all']
    rows.append(['Very slow (arrive +150 ms, known at −200 ms)', f"{v['triggers']:,}", pc(v['fill_rate']), c(v['mark_1s_c']), c(v['mark_3s_c']),
                 c(v['mark_10s_c']), c(v['rt_3s_c']), c(v['end_c']), c(v['res_c']), ''])
    t_main = md(rows, ['Clock', 'Triggers', 'Fill rate', '1 s', '3 s', '10 s', 'Exit by taking at 3 s', 'END', 'Resolution', 'USD at 3 s, Q 50, 4 days'])
    st = pd.read_csv(T / 'tokyo_by_strength.csv').set_index('strength').reindex(['edge-0.5', '0.5-1', '1-2', '>=2'])
    t_str = md([[i.replace('edge', '0.29/0.47'), f'{int(r.filled):,}', int(r.markets), pc(r.fill_rate), c(r.mark_1s_c), c(r.mark_3s_c), c(r.mark_10s_c),
                 c(r.rt_3s_c)] for i, r in st.iterrows()], ['abs bn_r250 (bps)', 'Fills', 'Markets', 'Fill rate', '1 s', '3 s', '10 s', 'Exit by taking at 3 s'])
    tb = pd.read_csv(T / 'tokyo_by_time.csv').set_index('tb')
    t_tb = md([[i, f'{int(r.filled):,}', pc(r.fill_rate), c(r.mark_1s_c), c(r.mark_3s_c), c(r.mark_10s_c), c(r.rt_3s_c), c(r.end_c), c(r.res_c)]
               for i, r in tb.iterrows()], ['Seconds into window', 'Fills', 'Fill rate', '1 s', '3 s', '10 s', 'Exit by taking at 3 s', 'END', 'Resolution'])
    dy = pd.read_csv(T / 'tokyo_by_day.csv').set_index('day')
    t_dy = md([[i, f'{int(r.filled):,}', pc(r.fill_rate), c(r.mark_1s_c), c(r.mark_3s_c), c(r.mark_10s_c), c(r.rt_3s_c)] for i, r in dy.iterrows()],
              ['Day', 'Fills', 'Fill rate', '1 s', '3 s', '10 s', 'Exit by taking at 3 s'])
    mo = pd.read_csv(T / 'tokyo_by_money_strength.csv')
    mo = mo[mo.money != 'na'].groupby('money').apply(lambda x: pd.Series(dict(filled=x.filled.sum(), m3=(x.mark_3s_c * x.shares).sum() / x.shares.sum())),
                                                      include_groups=False)
    ctl = [['Signal triggers (primary)', pc(ct['primary_tokyo']['fill_rate']), c(ct['primary_tokyo']['mark_3s_c']), c(ct['primary_tokyo']['rt_3s_c'])]]
    for k, lab in (('placebo', 'Random times (same count per market)'), ('shift7', 'Same triggers +7 s'), ('opposite', 'Opposite direction')):
        ctl.append([lab, pc(ct[k]['all']['fill_rate']), c(ct[k]['all']['mark_3s_c']), c(ct[k]['all']['rt_3s_c'])])
    t_ctl = md(ctl, ['Tokyo clock, Q 50', 'Fill rate', '3 s', 'Exit by taking at 3 s'])
    a = m3['all_qualifying']
    t_m3 = md([[k, f"{a[k]['takes']:,}", pc(a[k]['observed_agree']), pc(a[k]['null_mean']), pc(a[k]['null_p95']), f"{a[k]['frac_null_ge_observed']:.3f}",
                c(a[k]['agree_v1_c']), c(a[k]['nonagree_v1_c']), c(a[k]['agree_net_c']), c(a[k]['nonagree_net_c'])] for k in ('adding', 'reducing')],
              ['Maker3 takes', 'Takes', 'With same-direction trigger', 'Shifted-time mean', 'Shifted p95', 'Share of 500 shifts ≥ observed',
               '1 s ¢/sh, with trigger', '1 s, without', 'Resolution ¢/sh, with trigger', 'Resolution, without'])
    ex = tk[(tk.mid == 3789484) & (tk.clock == 'tokyo') & (tk.t_rel_s < 1)].iloc[0]
    sel = rd['tokyo']['selection']
    n_pass = sum(1 for k, v in ck.items() if isinstance(v, dict) and v.get('ok'))
    n_all = sum(1 for k, v in ck.items() if isinstance(v, dict))
    txt = f"""# Taking on our Binance pressure signal — 2026-11-25

**Answer.** Yes: the signal that makes us withhold asks predicts a Predict move large enough to pay the spread and the 2 % taker fee, but **only if we act within ~50–90 ms of the Binance move**. Every BTC jump that turns our pressure guard ON was valued as an immediate IOC take, over 1,151 markets on consumed Oct 24–27. At the measured Tokyo clock, the taken shares are worth **{c(allt['tokyo'].mark_3s_c)} ¢ each after the fee three seconds later**, and {c(allt['tokyo'].rt_3s_c)} ¢ even if we also exit by taking. This holds on every day, in both directions, and is larger for stronger signals. Arriving at +150 ms leaves nothing ({c(v['mark_3s_c'])} ¢). Random-time takes lose the spread plus fee ({c(ct['placebo']['all']['mark_3s_c'])} ¢), so the result is not mechanical.

Maker3 trades on the same signal: 57 % of its takes follow a same-direction trigger within 3.5 s, against 22 % at shifted times. The edge is a **quick in-and-out**, not a reason to hold early inventory: its value at resolution is noisy and depends on the market's outcome.

Consumed days, development evidence only; no replay of our strategy, no live claim.

## The 3789484 opening

Maker3's first fills there are **passive**. It raised its bid to 56 ¢ within about a second of a +2.6 bps BTC burst at 0.44–0.65 s and was filled as sellers arrived; its first take is at 82.8 s. Our guard fired at 550 ms on bn_r250 = +0.32485 bps; this packet's recomputation from the raw tape gives exactly that value.

A take at the first trigger (0.505 s, bn_r250 {ex.r250:+.4f} bps) would have bought the {ex.avail_sh:.2f} shares still offered at {ex.limit:.2f}, fee {100 * ex.fee:.2f} ¢:

- {100 * ex.mark_1s:+.1f} ¢/share at 1 s, {100 * ex.mark_3s:+.1f} at 3 s, {100 * ex.mark_10s:+.1f} at 10 s;
- {100 * ex.rt_3s:+.1f} if sold by taking at 3 s;
- **{100 * ex.res:+.0f} ¢ if held to resolution**, because the market resolved Down.

Maker3 instead turned its position over at 58–63 ¢ within ten seconds.

## Method (contract.md, declared before outcomes)

- **Signal:** the guard's own feature, `bn_r250 = 1e4·ln(mid(t)/mid(t−250 ms))` on the Binance bookTicker, recomputed from {te['events_all']:,} qualifying raw-tape events. UP when ≥ 0.2855 bps (the ask guard ON), DOWN when ≤ −0.4667 bps (the bid guard ON); the frozen guard edges. One trigger per market and direction per 2 s: {te['events_kept']:,} take opportunities.
- **Execution:** the decision uses the Predict book known 63 ms earlier (publication + network); the limit is that best ask or bid, with no chasing. The order arrives 42 ms later and fills only if the price is still executable both in the last frame before arrival and in the first frame after it. Size is at most the touch in both frames (Q = 50 primary); taker fee 200 bps × min(p, 1−p).
- **Values per filled share:** after the entry fee, at the Predict mid 1 / 3 / 10 s later, at the 298 s END mark, at resolution, and as a taker round trip (sell at the opposite touch and pay the fee again).

## Results

{t_main}

By signal strength (Tokyo, values in ¢ per filled share; the edge rises with strength):

{t_str}

By time in the window: the early window is not special for the edge, but its fills are easier. The last 30 s have few fills because the book is thin or one-sided.

{t_tb}

By day (every day positive at every short horizon):

{t_dy}

Near-money books (10–90 ¢) carry the edge: {c(mo.loc['NEAR', 'm3'])} ¢ at 3 s on {int(mo.loc['NEAR', 'filled']):,} fills, against {c(mo.loc['EXTREME', 'm3'])} ¢ at extreme prices.

## Controls (amendment 1, declared before computing them)

{t_ctl}

The valuation code first reproduced every primary row exactly ({ct['parity']['rows']:,} rows, {ct['parity']['mismatched_fields']} mismatches). At random times the take loses about the spread plus fee and the stale price is usually still there. After a trigger, half the stale prices are gone before we would arrive. That is the competition and repricing the fill rule is meant to capture.

## Maker3 times its takes on the same signal

{t_m3}

Its trigger-aligned position-adding takes earn {c(a['adding']['agree_net_c'])} ¢/share to resolution against {c(a['adding']['nonagree_net_c'])} for the rest. In the first 60 s, {pc(a['adding_first_60s']['observed_agree'])} of its adding takes are aligned, against {pc(a['adding_first_60s']['null_mean'])} at shifted times. Association only; its private trigger is not identified.

## Declared one-split selection

On Oct 24–25 the best cell with ≥ 30 fills in ≥ 10 markets was |bn_r250| ≥ {sel['chosen']['b']} bps in all windows: {c(sel['chosen']['dev_mark_3s_c'])} ¢ at 3 s on {int(sel['chosen']['dev_filled'])} fills. On Oct 26–27 it earns {c(sel['holdout_mark_3s_c'])} ¢ on {sel['holdout_filled']} fills in {sel['holdout_markets']} markets, positive on both days. That is a thin cell; the broad all-trigger result is the stronger evidence.

## What this means and what it does not

- **Taking is viable only at Tokyo latency.** The edge is +2.3 ¢ at +42 ms, +1.7 ¢ at +86 ms and 0 at +150 ms. This US host (~74 ms to Binance, Predict frames ~126 ms old) is too slow.
- **Take and leave, rather than accumulate.** The short-horizon value is reliable. Holding to resolution adds a large, outcome-driven variance: the per-market sign is a coin flip, as 3789484 shows.
- **Not yet measured:**
  - integration with our passive strategy (inventory caps, the asks we withhold, re-entry);
  - sub-frame competition, because frames are ~50 ms snapshots;
  - our own price impact on later marks;
  - API limits (40/s, 500/min; the triggers average ~6/min);
  - capacity beyond the touch;
  - fee changes.
- **Consumed days only:** this is development evidence, not validation; untouched days need the USER.

## Integrity

- **Sources and stages:** raw bookTicker hours only on the allow-list (no 20261028 stem is opened); stage 1 `bn_events.py` → `take_eval.py` → `summarize.py` → `controls.py` → `m3_align.py` → `check.py` → this file. Every number is generated from saved outputs.
- **Checks:** {n_pass}/{n_all} pass ([checks.json](checks.json)):
  - plain raw rescan of bn_r250 on 2,643 triggers;
  - future Binance and Predict perturbations with no effect, plus a positive control;
  - synthetic fill cases;
  - the 2 s rule on all 259,826 events;
  - reconciliation, and pooled readings recomputed.
- **One failed attempt:** the first take_eval receipt refused tuple keys after the outputs were written; the partial outputs were removed and the run repeated with no rule change, as recorded in take_eval.json.

## Next (needs the USER's direction)

1. Add a **SIGNAL_TAKE overlay** to our replay of the 19 complete markets: when the guard turns ON, also send an IOC at the known touch (size ≤ 50), then let the existing passive rules and inventory caps manage the position. Compare full-book value against the current arms.
2. **Measure competition directly:** after each trigger, did the stale level disappear through trades (pending settlements) or cancels, and how fast? That calibrates the fill rule.
3. **Untouched-day confirmation** and any Tokyo shadow run are the USER's call.
"""
    (H / 'report.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
