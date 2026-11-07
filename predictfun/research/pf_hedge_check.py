"""Taker hedge of the leftover at the stop (user 2026-11-07: "can we also buy down to hedge our up position, buy up to hedge our down
position, check how it performs"). On Predict.fun the Up and Down books are one book (pf_io.yes_side: a Down bid at q is an Up ask at
1 - q), so buying Down = selling Up and our passive asks are already Down buys; what is new here is hedging by TAKING: at the stop,
cross the book to cut the leftover, paying the spread and the taker fee.
Read-only on the 9 consumed round-i days (ivJ / ivX runs of run_inv.sh); no replay, no forward read; in-sample, descriptive.
Exact without a replay because nothing trades after the stop (fills after the stop are counted; 0 expected): the hedge cannot change any
earlier decision. Per window of arm A (v1, CAP10: stop 60 s; RO60-240: stop 240 s) with net n at the stop:
  decision at stop + D (D = 0.5 s; sensitivity 2.0 s), arrival + 40 ms (PF_RL_LPLACE): walk the venue book frame at or before
  ws + stop + D + 0.04 s (last valid frame, engine order), up to 5 levels: n > 0 sells into the bids (= buys Down), n < 0 buys the
  asks (= sells Down); depth shortfall stays unhedged (counted).
  rules: HF cut to 0; H10 cut to 10 shares; H25 cut to 25 shares; none (the arm as run).
  hedge leg P&L = s_h x (y - p) / 100 per share; taker fee 0.02 x min(p, 100 - p) / 100 $ per share (full rate, no discount, no
  rebate); cost vs mid = |p - m| / 100 at the hedge frame.
Window settle = engine win.pnl + hedge legs - fees. Controls: 'none' totals equal report_inv_v1 settle; for every fully hedged window
the settle equals the cash of its fills and hedge legs minus fees (no outcome left) within 1e-9 $; falsifier: the same check on
hedge legs with the side flipped must fail.
Per (timing, arm, rule, D): settle, days > 0, day sd, worst day, max drawdown, $150 cap refusal / halt within 1 day (pf_cap_check),
window sd; hedge shares, cost c/sh (spread + fee), shortfall; differences vs the same arm unhedged and vs v1 (day t, 8 df).
Output step4/report_hedge_v1.json. Usage: python3 -m predictfun.research.pf_hedge_check
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_inv_report import arms, stem, run_dir, DAYS, S4, T8
from .pf_cap_check import first_refusal, first_halt, mdd, PER_DAY

ARMS_H = {'v1': 60.0, 'CAP10': 60.0, 'RO60-240': 240.0}
RULES = {'none': None, 'HF': 0.0, 'H10': 10.0, 'H25': 25.0}
DELAYS = (0.5, 2.0); LPLACE = 0.04; FEE = 0.02; NL = 5


def book_maps(day, lo, hi):
    S1 = f'{io.OUT}/step1/{day}'
    cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{k}_{x}' for s in 'ba' for k in range(1, NL + 1) for x in 'ps']
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable'); bk = bk[(bk.t_ms.values >= lo) & (bk.t_ms.values < hi)]
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)]
    out = {}
    for m, g in bk.groupby('mid', sort=False):
        B = np.stack([np.stack([g[f'b{k}_p'].values, g[f'b{k}_s'].values], 1) for k in range(1, NL + 1)], 1).astype(float)
        A = np.stack([np.stack([g[f'a{k}_p'].values, g[f'a{k}_s'].values], 1) for k in range(1, NL + 1)], 1).astype(float)
        out[m] = (g.t_ms.values.astype(np.float64), B, A)
    return out


def take(levels, q, st):
    """walk (price, size) levels in priority order for q shares; -> [(p, x)], shortfall."""
    ok = [(p, s) for p, s in levels if 0 <= p <= 100 and s > 0]
    legs = []; rem = q
    for p, s in ok:
        if rem <= 1e-12:
            break
        x = min(rem, s); legs.append((p, x)); rem -= x
    return legs, rem


def hedge_window(BK, mid, ws, stop, D, n, target, y, st):
    """-> (pnl $, fee $, cost_vs_mid $, hedged shares, shortfall, cash $, legs signed) for one window."""
    if target is None or abs(n) <= target + 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
    q = abs(n) - target; v = BK.get(mid)
    if v is None:
        st['no_frame'] += 1; return 0.0, 0.0, 0.0, 0.0, q, 0.0, []
    t, B, A = v; i = int(np.searchsorted(t, ws * 1000.0 + (stop + D + LPLACE) * 1000.0, side='right')) - 1
    if i < 0:
        st['no_frame'] += 1; return 0.0, 0.0, 0.0, 0.0, q, 0.0, []
    m = (B[i, 0, 0] + A[i, 0, 0]) / 2.0
    if n > 0:                                          # long Up: sell into the bids (= buy Down at 100 - p)
        lv = [tuple(B[i, k]) for k in range(NL)]; sh = -1.0
        bp = [p for p, s in lv if 0 <= p <= 100 and s > 0]
        st['book_order_bad'] += int(bp != sorted(bp, reverse=True))
    else:                                              # short Up: buy the asks (= sell Down)
        lv = [tuple(A[i, k]) for k in range(NL)]; sh = 1.0
        ap = [p for p, s in lv if 0 <= p <= 100 and s > 0]
        st['book_order_bad'] += int(ap != sorted(ap))
    legs, rem = take(lv, q, st)
    pnl = sum(sh * x * (y - p) / 100.0 for p, x in legs); fee = sum(x * FEE * min(p, 100.0 - p) / 100.0 for p, x in legs)
    cost = sum(x * abs(p - m) / 100.0 for p, x in legs); cash = sum(-sh * x * p / 100.0 for p, x in legs)
    return pnl, fee, cost, q - rem, rem, cash, [(sh, p, x) for p, x in legs]


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T8 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(float(m), 2), lo=round(float(m - h), 2), hi=round(float(m + h), 2), days_pos=int((x > 0).sum()),
                excludes_0=bool(m - h > 0 or m + h < 0))


def main():
    MAIN, _, _, _ = arms()
    if not json.load(open(f'{S4}/inv_check.json')).get('PASS'):
        sys.exit('REFUSED: step4/inv_check.json is not PASS')
    prior = json.load(open(f'{S4}/report_inv_v1.json'))
    st = dict(windows=0, fills_after_stop=0, no_frame=0, book_order_bad=0, full_hedge_checked=0)
    rows = []; falsifier_fired = None
    for d in DAYS:
        reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws', 'outcome']).set_index('mid')
        ib = json.load(open(f'{run_dir("ivJ", d)}/summary.json'))['input_bounds']; BK = book_maps(d, ib['t_lo_ms'], ib['t_hi_ms'])
        for tm in ('J', 'X'):
            assert json.load(open(f'{run_dir("iv" + tm, d)}/summary.json'))['input_bounds'] == ib
            for lab, stop in ARMS_H.items():
                base = run_dir('iv' + tm, d); spec = MAIN[lab]
                f = pd.read_parquet(f'{base}/fills/{stem(spec)}.parquet'); W = pd.read_parquet(f'{base}/win/{stem(spec)}.parquet')
                st['fills_after_stop'] += int((f.t.values >= stop).sum())
                s = np.where(f.side.values == 0, 1.0, -1.0)
                g = pd.DataFrame(dict(mid=f.mid.values, net=s * f.q.values, cash=-s * f.q.values * f.pt.values / 100.0)).groupby('mid').sum()
                for w in W.itertuples(index=False):
                    st['windows'] += int(tm == 'J' and lab == 'v1')
                    n = float(g.net.get(w.mid, 0.0)); y = 100.0 if reg.loc[w.mid, 'outcome'] == 'Up' else 0.0
                    if abs(n - w.net_end) > 1e-6:
                        sys.exit(f'REFUSED: fill net {n} != win net_end {w.net_end} ({tm} {d} {lab} {w.mid})')
                    for rule, target in RULES.items():
                        for D in (DELAYS if target is not None else DELAYS[:1]):
                            pnl, fee, cost, hq, short, cash, legs = hedge_window(BK, w.mid, float(reg.loc[w.mid, 'ws']), stop, D, n, target, y, st)
                            tot = float(w.pnl) + pnl - fee
                            if rule == 'HF' and hq > 0 and short <= 1e-12:     # control: no outcome left -> settle = cash - fees
                                c0 = float(g.cash.get(w.mid, 0.0)) + cash - fee
                                if abs(tot - c0) > 1e-9:
                                    sys.exit(f'CONTROL FAILED: fully hedged window settle {tot} != cash {c0} ({tm} {d} {lab} {w.mid})')
                                st['full_hedge_checked'] += 1
                                if falsifier_fired is None:            # falsifier: hedge legs with the side flipped must break it
                                    pnl_b = sum(-sh * x * (y - p) / 100.0 for sh, p, x in legs)
                                    falsifier_fired = abs(float(w.pnl) + pnl_b - fee - c0) > 1e-9
                                    if not falsifier_fired:
                                        sys.exit('FALSIFIER DID NOT FIRE: flipped hedge legs passed the cash control')
                            rows.append((tm, lab, rule, D, d, w.ws, w.mid, tot, float(w.min_worst), fee, cost, hq, short, abs(n)))
        del BK
        io.mem_check(f'day {d}')
    R = pd.DataFrame(rows, columns=['tm', 'arm', 'rule', 'D', 'day', 'ws', 'mid', 'pnl', 'min_worst', 'fee', 'cost', 'hq', 'short', 'absn'])
    out = dict(as_of=pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ'), days=DAYS, statuses=st, falsifier_fired=falsifier_fired,
               fee='0.02 x min(p, 1 - p) per share, full rate, no rebate', delays_s=DELAYS, place_latency_s=LPLACE, results={})
    ctrl = {}
    for (tm, lab, rule, D), G in R.groupby(['tm', 'arm', 'rule', 'D'], sort=False):
        G = G.sort_values('ws'); p = G.pnl.values; mw = G.min_worst.values; n = len(G)
        day = G.groupby('day').pnl.sum().reindex(DAYS).values
        ref = [first_refusal(p[i:i + PER_DAY], mw[i:i + PER_DAY]) for i in range(n - PER_DAY + 1)]
        hlt = [first_halt(p[i:i + PER_DAY]) for i in range(n - PER_DAY + 1)]
        r = dict(settle=round(float(p.sum()), 1), days_pos=int((day > 0).sum()), day_sd=round(float(day.std(ddof=1)), 1),
                 worst_day=round(float(day.min()), 1), max_drawdown=round(mdd(p), 1), window_sd=round(float(p.std()), 2),
                 worst_window=round(float(p.min()), 1),
                 cap150_refusal_1d_pct=round(100.0 * np.mean([x is not None for x in ref]), 1), cap150_halt_1d_pct=round(100.0 * np.mean([x is not None for x in hlt]), 1),
                 hedged_shares=round(float(G.hq.sum()), 0), leftover_shares=round(float(G.absn.sum()), 0), shortfall_shares=round(float(G.short.sum()), 1),
                 hedge_spread_c_sh=round(100.0 * G.cost.sum() / G.hq.sum(), 3) if G.hq.sum() > 0 else None,
                 hedge_fee_c_sh=round(100.0 * G.fee.sum() / G.hq.sum(), 3) if G.hq.sum() > 0 else None,
                 hedge_cost_usd=round(float(G.cost.sum() + G.fee.sum()), 1), per_day=dict(zip(DAYS, np.round(day, 1).tolist())), _day=day)
        out['results'][f'{tm}|{lab}|{rule}|{D}'] = r
        if rule == 'none':
            ctrl[(tm, lab)] = abs(r['settle'] - prior['timings'][tm][lab]['settle']) <= 0.1
    if not all(ctrl.values()):
        sys.exit(f'CONTROL FAILED: unhedged totals differ from report_inv_v1: {ctrl}')
    out['control_none_equals_report_inv'] = True
    for k, r in out['results'].items():
        tm, lab, rule, D = k.split('|')
        base = out['results'][f'{tm}|{lab}|none|{DELAYS[0]}']; v1 = out['results'][f'{tm}|v1|none|{DELAYS[0]}']
        r['vs_same_arm_unhedged'] = dict(settle=ci(r['_day'] - base['_day']), day_sd_ratio=round(r['day_sd'] / base['day_sd'], 3))
        r['vs_v1'] = dict(settle=ci(r['_day'] - v1['_day']), day_sd_ratio=round(r['day_sd'] / v1['day_sd'], 3))
    for r in out['results'].values():
        del r['_day']
    io.write_json(f'{S4}/report_hedge_v1.json', out)
    print(f"statuses {st}; falsifier fired {falsifier_fired}; control none == report_inv_v1: {ctrl}")
    for tm in ('J', 'X'):
        print(f'\n== {tm}: $ over 9 days | hedge: shares hedged / leftover, spread + fee c/sh | vs same arm unhedged settle $/day (t 8 df) | vs v1')
        for k, r in out['results'].items():
            if not k.startswith(tm + '|'):
                continue
            _, lab, rule, D = k.split('|')
            hs = f"{r['hedged_shares']:6.0f}/{r['leftover_shares']:6.0f} {r['hedge_spread_c_sh'] or 0:5.2f}+{r['hedge_fee_c_sh'] or 0:4.2f}"
            a = r['vs_same_arm_unhedged']['settle']; b = r['vs_v1']['settle']
            print(f"{lab:9s} {rule:4s} D{D:<4} settle {r['settle']:7.1f} ({r['days_pos']}/9) daysd {r['day_sd']:6.1f} worst {r['worst_day']:7.1f} dd {r['max_drawdown']:7.1f} "
                  f"cap {r['cap150_refusal_1d_pct']:5.1f}/{r['cap150_halt_1d_pct']:5.1f} | {hs} | {a['mean']:+7.2f} ({a['lo']:+.1f}..{a['hi']:+.1f}) | {b['mean']:+7.2f} ({b['lo']:+.1f}..{b['hi']:+.1f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
