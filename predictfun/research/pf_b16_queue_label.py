"""B16 label (2026-11-11, USER: "the model may not predict the right action to take"): what OUR order would have earned, on the
guard grid's rows (step4/badfill_info_tx_tk/grid.parquet: every 500 ms of the live 0-240 s window, 8 read days).

The live guard's label (pf_badfill_grid_tx.py) is v = s (m(t_sw + 1 s) - p) if the known best p is traded THROUGH in (tau + 22,
tau + 522] ms, else 0: it never sees a fill from trades AT our price. On 11-08, 61% of our at-best shares came from such trades (median
queue reach 0.93 on heavy levels, -0.59 c/sh; B12 / B12e / B12f). This label is the value of a 5-share order of ours at the BACK of
the queue at the known best p (the grid's p, the same Tokyo clock and trade times, one time per taker order). Primary (ql, the keep /
pull decision the guard makes): the order RESTS there, exposed from tau + 22 ms (when a pull would land, as the old label) to tau + 522
ms, with ahead = the size shown at p in the last venue frame at or before tau + 22 (live orders churn, so they sit at the back). Beside
(qlf, a fresh post): live at tau + 40 ms (L_PLACE); refused (0) if its price crosses the book then (post-only); ahead = the size shown at
p in the last venue frame before it, raised to the next frame's if larger and <= 100 ms later (QACT max); the same end.
  fills     rule M, primary (the v3 engine's fill model v2, pf_step4_ladder_v3: before each trade the size ahead is capped by the
            size shown at p in every frame since the last one applied, strictly before the trade - a level that thins moves us up;
            beyond the 5 shown levels no cap): a trade at p first uses up the size ahead, then fills us; a trade through p (bid: below,
            ask: above) fills what is left of us at p. Rule L beside (qll_*): the same without the frame cap (no credit for cancels
            ahead, the lower bound). Legs in time order (tx anchor), price-first at one time.
  value     per fill, s (m(t_f + h) - p) per share for h = 1 s (primary) and 5 s, m = the venue mid of the last frame <= t_f + h;
            label = sum(shares x value) / 5 (c per share of the order; a full fill on a sweep equals the old label exactly)
Outputs step4/badfill_info_tx_tk/grid_ql.parquet (day, mid, tau aligned to grid.parquet's rows): ql_bid / ql_ask (1 s), ql5_bid /
ql5_ask (5 s), qf_bid / qf_ask (shares filled), qs_bid / qs_ask (filled by a sweep), qa_bid / qa_ask (size ahead), and the fresh post's
qlf_* / qff_* / qsf_* (rule M) and the resting order under rule L, qll_* / qfl_*; a report with statuses (no frame, no mid after a fill, post-only refusals, deeper than shown) and the controls.
Controls (any failure -> no output): K1 on rows whose old label is non-zero and with no fill of ours before the first through-trade,
the resting label equals the old one exactly (same window, same sweep time, same markout); K2 synthetic cases through the same fill function (queue use-up,
partial fill, sweep after a partial, no fill behind the queue, ask side mirror; rule M: a thinned level moves us up, a frame at the
trade's own time is not applied first, a deeper-than-shown frame caps nothing); K3 planted errors (no queue, cancels always credited,
the frame cap ignored) must each break a K2 case.
Usage: PF_RL_DDEC=2 predictfun/research/run_light.sh b16_label -- python3 -m predictfun.research.pf_b16_queue_label
"""
import os
import sys

import numpy as np
import pandas as pd

from . import pf_io as io
from . import pf_clock_rl as CK

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
GRID = f'{io.OUT}/step4/badfill_info_tx_tk/grid.parquet'
OUT = f'{io.OUT}/step4/badfill_info_tx_tk/grid_ql.parquet'
L_PLACE, L_CXL, W, SIZE, QACT_MS = 40.0, 22.0, 500.0, 5.0, 100.0
ms_ = lambda d: pd.Timestamp(d, tz='UTC').value // 1_000_000
SEAL_MS = (ms_('2026-10-28'), ms_('2026-10-31')); FWD_MS = ms_('2026-11-04')
TXK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
PLANT = {'no_queue': False, 'credit_cancels': False, 'no_cap': False}   # planted errors for K3 only
st = dict(rows=0, no_frame_live=0, no_mid_after=0, postonly_refused=0, deeper_than_shown=0, legs=0, legs_untimed_excluded=0)


def tx_times(lt, S1):
    """pf_badfill_grid_tx.tx_times with PF_TXKEY=tk: one time per taker order (earliest WS-timed leg, else earliest E2)."""
    ws_t = np.where(lt.how.values > 0, lt.t_ms.values.astype(float), np.nan); e2_t = np.where(lt.how.values == 0, lt.t_e2.values, np.nan)
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet').tk.values; assert len(tk) == len(lt), 'legs_tk not aligned'
    grp = [pd.Series(np.where(tk >= 0, tk, -1 - np.arange(len(lt))), index=lt.index)]
    a = pd.Series(ws_t, index=lt.index).groupby(grp, sort=False, dropna=False).transform('min').values
    b = pd.Series(e2_t, index=lt.index).groupby(grp, sort=False, dropna=False).transform('min').values
    anc = np.where(np.isfinite(a), a, b)
    st['legs'] += len(lt); st['legs_untimed_excluded'] += int((~np.isfinite(anc)).sum())
    return anc


def fill_order(s, p, ahead, legs, size=SIZE, caps=None):
    """Our order at p behind `ahead` shares; legs = [(t, price, shares)] in time order after it went live, inside the exposure.
    caps = None: rule L (cancels ahead never move us up). caps = [(t_frame, shown size at p)] in time order: rule M, the size ahead is
    capped by every frame strictly before each trade. Returns [(t, shares, sweep)]."""
    out = []; rem = size; a = 0.0 if PLANT['no_queue'] else float(ahead); k = 0
    for t, pt, q in legs:
        if rem <= 1e-12:
            break
        if caps is not None and not PLANT['no_cap']:
            while k < len(caps) and caps[k][0] < t:
                a = min(a, caps[k][1]); k += 1
        through = (pt < p) if s > 0 else (pt > p)
        if through:
            out.append((t, rem, True)); rem = 0.0; break
        if pt != p:
            continue
        if PLANT['credit_cancels']:
            a = 0.0
        if q <= a + 1e-12:
            a -= q; continue
        f = min(rem, q - a); a = 0.0; rem -= f; out.append((t, f, False))
    return out


def k2_cases():
    """Synthetic cases with known answers (bid s = +1 at p = 50; ask s = -1 at p = 51)."""
    c = []
    c.append(fill_order(1, 50, 10, [(1.0, 50, 4.0), (2.0, 50, 8.0)]) == [(2.0, 2.0, False)])                 # queue used up, partial
    c.append(fill_order(1, 50, 10, [(1.0, 50, 4.0), (2.0, 50, 8.0), (3.0, 49, 1.0)]) == [(2.0, 2.0, False), (3.0, 3.0, True)])
    c.append(fill_order(1, 50, 10, [(1.0, 50, 9.0)]) == [])                                                  # still behind the queue
    c.append(fill_order(1, 50, 0, [(1.0, 51, 9.0), (2.0, 50, 2.0)]) == [(2.0, 2.0, False)])                  # other price ignored
    c.append(fill_order(-1, 51, 3, [(1.0, 51, 10.0)]) == [(1.0, 5.0, False)])                                # ask side, full fill
    c.append(fill_order(-1, 51, 3, [(1.0, 52, 1.0)]) == [(1.0, 5.0, True)])                                  # ask swept through
    c.append(fill_order(1, 50, 100, [(2.0, 50, 4.0)], caps=[(1.0, 3.0)]) == [(2.0, 1.0, False)])             # M: thinned to 3 ahead
    c.append(fill_order(1, 50, 100, [(2.0, 50, 4.0)], caps=[(2.0, 3.0)]) == [])                              # M: same-time frame later
    c.append(fill_order(1, 50, 100, [(2.0, 50, 4.0)], caps=[(1.0, float('inf'))]) == [])                     # M: deeper than shown
    c.append(fill_order(1, 50, 100, [(2.0, 50, 4.0)]) == [])                                                 # L: no credit
    return c


def controls():
    ok = all(k2_cases())
    broke = {}
    for k in PLANT:
        PLANT[k] = True; broke[k] = not all(k2_cases()); PLANT[k] = False
    return ok and all(broke.values()), dict(k2=k2_cases(), k3=broke)


def build(d, G):
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    rows = G[G.day == d]
    cols = ['mid', 't_ms', 'recv_ns'] + [f'{x}{i}_{y}' for x in 'ba' for i in range(1, 6) for y in 'ps']
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
    t_ = bk.t_ms.values; bk = bk[~((t_ >= SEAL_MS[0]) & (t_ < SEAL_MS[1])) & (t_ < FWD_MS)]
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=TXK + ['side', 'ptick', 'shares', 't_ms', 'how', 't_e2'])
    lt['ta'] = tx_times(lt, S1); lt = lt[np.isfinite(lt.ta.values)]
    LB = {}
    for k, x in lt.groupby(['mid', 'side'], sort=False):
        o = np.lexsort((x.ptick.values * (1 if k[1] == 1 else -1), x.ta.values))   # one time: the maker side's best price first
        LB[k] = (x.ta.values[o].astype(float), x.ptick.values[o].astype(int), x.shares.values[o].astype(float))
    BKD = {m: g for m, g in bk.groupby('mid', sort=False)}; del bk, lt
    res = {c: np.full(len(rows), np.nan) for c in ('ql_bid', 'ql_ask', 'ql5_bid', 'ql5_ask', 'qf_bid', 'qf_ask', 'qa_bid', 'qa_ask',
                                                   'qll_bid', 'qll_ask', 'qfl_bid', 'qfl_ask', 'qlf_bid', 'qlf_ask', 'qff_bid', 'qff_ask')}
    res.update({c: np.zeros(len(rows), bool) for c in ('qs_bid', 'qs_ask', 'qsf_bid', 'qsf_ask')})
    k1 = dict(rows=0, equal=0, first_diff=None)
    for mid, idx in rows.groupby('mid', sort=False).indices.items():
        g = BKD.get(mid)
        if g is None:
            continue
        kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, shift)); tf = g.t_ms.values.astype(float)
        b1 = g.b1_p.values; a1 = g.a1_p.values; vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        tau = rows.tau.values[idx].astype(float); j = np.searchsorted(kf, tau, side='right') - 1
        good = j >= 0; jj = np.clip(j, 0, None); pb = np.where(good, b1[jj], -1); pa = np.where(good, a1[jj], -1)
        good &= (pb >= 0) & (pa > pb)
        P = {0: [g[f'b{i}_p'].values for i in range(1, 6)], 1: [g[f'a{i}_p'].values for i in range(1, 6)]}
        Z = {0: [g[f'b{i}_s'].values for i in range(1, 6)], 1: [g[f'a{i}_s'].values for i in range(1, 6)]}

        def shown(si, ci, p):
            """Size ahead at p in frame ci: the level's size; 0 if p is better than our side's best or falls in an empty gap between
            shown levels (nobody there); inf if p is deeper than the 5 shown levels (unknown, never filled but by a sweep)."""
            lv = [(pp[ci], zz[ci]) for pp, zz in zip(P[si], Z[si]) if pp[ci] >= 0]
            for pp, zz in lv:
                if pp == p:
                    return float(zz)
            if not lv or (p > lv[0][0] if si == 0 else p < lv[0][0]):
                return 0.0
            if (p > lv[-1][0]) if si == 0 else (p < lv[-1][0]):
                return 0.0
            st['deeper_than_shown'] += 1
            return float('inf')
        for si, (s, p, nm) in enumerate(((1.0, pb, 'bid'), (-1.0, pa, 'ask'))):
            L = LB.get((mid, si))
            for n in np.flatnonzero(good):
                r = idx[n]; t_end = tau[n] + L_CXL + W
                for kind, t_live in (('rest', tau[n] + L_CXL), ('fresh', tau[n] + L_PLACE)):
                    ci = np.searchsorted(tf, t_live, side='right' if kind == 'rest' else 'left') - 1
                    if ci < 0:
                        st['no_frame_live'] += 1; continue
                    if kind == 'fresh' and ((s > 0 and a1[ci] >= 0 and p[n] >= a1[ci]) or (s < 0 and b1[ci] >= 0 and p[n] <= b1[ci])):
                        st['postonly_refused'] += 1                    # post-only: our price crosses the book when it goes live
                        res[f'qlf_{nm}'][r] = res[f'qff_{nm}'][r] = 0.0; continue
                    ahead = shown(si, ci, p[n])
                    if kind == 'fresh' and ci + 1 < len(tf) and tf[ci + 1] - t_live <= QACT_MS:
                        ahead = max(ahead, shown(si, ci + 1, p[n]))
                    legs = []
                    if L is not None:
                        T, PT, Q = L; i0 = np.searchsorted(T, t_live, side='right'); i1 = np.searchsorted(T, t_end, side='right')
                        legs = list(zip(T[i0:i1].tolist(), PT[i0:i1].tolist(), Q[i0:i1].tolist()))
                    caps = []
                    if legs:
                        k1_ = np.searchsorted(tf, legs[-1][0], side='left')
                        caps = [(tf[kk], shown(si, kk, p[n])) for kk in range(ci + 1, k1_)]
                    variants = (('M', caps), ('L', None)) if kind == 'rest' else (('M', caps),)
                    for rule, cp in variants:
                        fills = fill_order(s, p[n], ahead, legs, caps=cp)
                        v1 = v5 = 0.0; miss = False
                        for tfl, q, sw in fills:
                            k = np.searchsorted(tv, tfl + 1000.0, side='right') - 1; k5 = np.searchsorted(tv, tfl + 5000.0, side='right') - 1
                            if k < 0 or k5 < 0:
                                miss = True; break
                            v1 += q * s * (mv[k] - p[n]); v5 += q * s * (mv[k5] - p[n])
                        if miss:
                            st['no_mid_after'] += 1; continue
                        qf = sum(f[1] for f in fills); qs = any(f[2] for f in fills)
                        if kind == 'rest' and rule == 'M':
                            res[f'qa_{nm}'][r] = ahead; res[f'ql_{nm}'][r] = v1 / SIZE; res[f'ql5_{nm}'][r] = v5 / SIZE
                            res[f'qf_{nm}'][r] = qf; res[f'qs_{nm}'][r] = qs
                        elif kind == 'rest':
                            res[f'qll_{nm}'][r] = v1 / SIZE; res[f'qfl_{nm}'][r] = qf
                        else:
                            res[f'qlf_{nm}'][r] = v1 / SIZE; res[f'qff_{nm}'][r] = qf; res[f'qsf_{nm}'][r] = qs
                        # K1 (both rules on the resting order): the old label when no fill of ours came before the first sweep through
                        old = rows[f'v_{nm}'].values[r]
                        if kind == 'rest' and old != 0 and fills and fills[0][2]:
                            k1['rows'] += 1; same = abs(v1 / SIZE - old) < 1e-9
                            k1['equal'] += int(same)
                            if not same and k1['first_diff'] is None:
                                k1['first_diff'] = (int(mid), float(tau[n]), nm, rule, float(v1 / SIZE), float(old))
    st['rows'] += len(rows)
    out = pd.DataFrame(dict(day=d, mid=rows.mid.values, tau=rows.tau.values, **res))
    io.mem_check(f'label {d}')
    return out, k1


def main():
    ok, ctl = controls()
    print('controls', ctl, flush=True)
    if not ok:
        sys.exit('REFUSED: K2 / K3 synthetic controls failed')
    if os.path.exists(OUT):
        sys.exit(f'REFUSED: {OUT} exists')
    G = pd.read_parquet(GRID, columns=['day', 'mid', 'tau', 'v_bid', 'v_ask'])
    parts, k1s = [], {}
    for d in DAYS:
        o, k1 = build(d, G); parts.append(o); k1s[d] = k1; print(d, len(o), k1, flush=True)
    R = pd.concat(parts, ignore_index=True)
    if len(R) != len(G) or not ((R.day.values == G.day.values).all() and (R.mid.values == G.mid.values).all() and (R.tau.values == G.tau.values).all()):
        sys.exit('REFUSED: label rows not aligned with grid.parquet')
    k1_rows = sum(v['rows'] for v in k1s.values()); k1_eq = sum(v['equal'] for v in k1s.values())
    rep = dict(statuses=st, k1=k1s, k1_rows=k1_rows, k1_equal=k1_eq, controls=ctl,
               fill_share={s: float(np.mean(R[f'qf_{s}'].values > 0)) for s in ('bid', 'ask')},
               sweep_share={s: float(np.mean(R[f'qs_{s}'].values)) for s in ('bid', 'ask')},
               old_nonzero_share={s: float(np.mean(G[f'v_{s}'].values != 0)) for s in ('bid', 'ask')},
               mean_ql={s: float(np.nanmean(R[f'ql_{s}'].values)) for s in ('bid', 'ask')},
               mean_old={s: float(np.mean(G[f'v_{s}'].values)) for s in ('bid', 'ask')})
    io.write_json(f'{OUT[:-8]}_report.json', rep)
    if k1_rows < 1000 or k1_eq != k1_rows:
        sys.exit(f'REFUSED: K1 {k1_eq} of {k1_rows} rows equal (need all, >= 1000)')
    R.to_parquet(OUT, index=False)
    print({k: rep[k] for k in ('k1_rows', 'k1_equal', 'fill_share', 'sweep_share', 'old_nonzero_share', 'mean_ql', 'mean_old')}, flush=True)


if __name__ == '__main__':
    main()
