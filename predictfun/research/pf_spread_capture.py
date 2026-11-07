"""Spread capture of the v1 inventory-mechanism arms (user 2026-11-07: "What about capture spread for different methods, I think this
will be critical to compare different mm strategies"). Read-only on the 9 consumed round-i days (ivJ / ivX runs of run_inv.sh, the 11
main arms x J / X); no replay, no forward read. Descriptive, in-sample: no null and no selection; intervals vs v1 are day-clustered
t (8 df, 2.306) and unadjusted.
Per fill (pf_mm_decomp.decompose; reconciled to the engine on every arm-day, falsifier: one fill's shares doubled must fail), c / share:
  spread    s (m0 - p)       against the venue mid of the last valid frame strictly before the fill
  adverse   s (m1 - m0)      the mid move within 1 s (pf_mm_decomp 'informed')
  plus1     spread + adverse realised spread at 1 s (= engine mo1)
  drift     s (mS - m1)      1 s to the arm's stop + 100 ms (RO arms: 120 / 240 s, so not like-for-like with v1)
  mark      plus1 + drift    (= engine mtm_stop)
and hs = (a1 - b1) / 2 of the same venue frame (our orders are not in it): capture = spread / hs (share-weighted).
Splits: reducing vs increasing shares (net from the arm's own fills in engine order; a fill that crosses zero is split by shares),
lane (0 touch / 1 rung), phase (t < 60 s / >= 60 s). Benchmark: real makers on the same days (pf_mm_decomp.maker_legs / maker_terms,
engine tx timing of the ivJ runs), bands top 5 / 6-20 / rest / all makers, first 60 s and 0-240 s.
Controls: this script's frame maps equal pf_mm_decomp.mid_maps on the first day; m0 recovered from the spread term equals the map
lookup on every arm-day; missing mids are counted (statuses), never dropped silently.
Output step4/report_spread_capture_v1.json. Usage: python3 -m predictfun.research.pf_spread_capture
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_mm_decomp import mid_maps, lookup, decompose, recon, maker_legs, maker_terms
from .pf_inv_report import arms, name, stem, run_dir, DAYS, S4, T8

TERMS = ('spread', 'informed', 'plus1', 'drift', 'mark', 'carry', 'settle', 'hsq')
GROUPS = ('all', 'red', 'inc', 'touch', 'rung', 't<60', 't>=60')


def stop_of(spec):
    for r in spec.split('|')[1].split('+'):
        if r.startswith('S') and r[1:].isdigit():
            return float(r[1:])
    raise SystemExit(f'REFUSED: no stop token in {spec}')


def frame_maps(day, lo, hi):
    """{mid: (t_ms, mid)} and {mid: (t_ms, half-spread)} of valid frames, built exactly like pf_mm_decomp.mid_maps (engine order)."""
    S1 = f'{io.OUT}/step1/{day}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))],
                   ignore_index=True)
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable'); bk = bk[(bk.t_ms.values >= lo) & (bk.t_ms.values < hi)]
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)]
    BY, HS = {}, {}
    for m, g in bk.groupby('mid', sort=False):
        t = g.t_ms.values.astype(np.float64); BY[m] = (t, (g.b1_p.values + g.a1_p.values) / 2.0); HS[m] = (t, (g.a1_p.values - g.b1_p.values) / 2.0)
    return BY, HS


def fill_frame(f, D, HS, reg):
    """per-fill weights of each group and hs; red / inc split by shares from the arm's own net in engine order."""
    o = f.reset_index(drop=True)
    s = np.where(o.side.values == 0, 1.0, -1.0); q = o.q.values.astype(float)
    order = np.lexsort((np.arange(len(o)), o.t.values, o.mid.values))          # stable (mid, t): file order on ties
    sq = pd.Series((s * q)[order]); cum = sq.groupby(o.mid.values[order]).cumsum().values
    nb = np.empty(len(o)); nb[order] = cum - sq.values
    red = np.where(nb * s < 0, np.minimum(q, np.abs(nb)), 0.0); inc = q - red
    ta = reg.loc[o.mid.values, 'ws'].values * 1000.0 + o.t.values * 1000.0
    hs = lookup(HS, o.mid.values, np.round(ta, 3), 'left')
    W = {'all': np.ones(len(o)), 'red': red / q, 'inc': inc / q, 'touch': (o.lane.values == 0) * 1.0, 'rung': (o.lane.values == 1) * 1.0,
         't<60': (o.t.values < 60) * 1.0, 't>=60': (o.t.values >= 60) * 1.0}
    return W, hs


def sums(D, w, hsq):
    r = dict(q=float((D.q.values * w).sum()))
    for c in TERMS:
        v = (hsq if c == 'hsq' else D[c].values) * w; ok = np.isfinite(v)
        r[c] = float(v[ok].sum()); r[f'q_{c}'] = float((D.q.values * w)[ok].sum())
    return r


def cps(r, c):
    return 100.0 * r[c] / r[f'q_{c}'] if r.get(f'q_{c}', 0.0) > 0 else float('nan')


def pool(rs):
    o = {}
    for r in rs:
        for k, v in r.items():
            o[k] = o.get(k, 0.0) + v
    return o


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; m = x.mean(); h = T8 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(float(m), 3), lo=round(float(m - h), 3), hi=round(float(m + h), 3), n_days=int(len(x)),
                days_pos=int((x > 0).sum()), excludes_0=bool(m - h > 0 or m + h < 0))


def main():
    MAIN, _, _, _ = arms()
    if not json.load(open(f'{S4}/inv_check.json')).get('PASS'):
        sys.exit('REFUSED: step4/inv_check.json is not PASS')
    st = dict(fills=0, no_m0=0, no_m1=0, no_mS=0, no_hs=0, maker_legs=0, maker_untimed=0, maker_no_m0=0, maker_no_m1=0, maker_no_hs=0)
    R = {}; MK = {}; falsifier = None; map_control = None; m0_max_err = 0.0
    for d in DAYS:
        reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws', 'outcome']).set_index('mid')
        sJ = json.load(open(f'{run_dir("ivJ", d)}/summary.json')); ib = sJ['input_bounds']
        BY, HS = frame_maps(d, ib['t_lo_ms'], ib['t_hi_ms'])
        if map_control is None:
            ref = mid_maps(d, ib['t_lo_ms'], ib['t_hi_ms'])
            map_control = bool(ref.keys() == BY.keys() and all(np.array_equal(ref[m][0], BY[m][0]) and np.array_equal(ref[m][1], BY[m][1]) for m in ref))
            del ref
            if not map_control:
                sys.exit('CONTROL FAILED: frame map differs from pf_mm_decomp.mid_maps')
        for tm in ('J', 'X'):
            sm = json.load(open(f'{run_dir("iv" + tm, d)}/summary.json'))
            assert sm['input_bounds'] == ib, (tm, d)
            for lab, spec in MAIN.items():
                base = run_dir('iv' + tm, d)
                f = pd.read_parquet(f'{base}/fills/{stem(spec)}.parquet'); Wn = pd.read_parquet(f'{base}/win/{stem(spec)}.parquet')
                D = decompose(f, BY, reg, stop_of(spec))
                bad = recon(D, f, Wn)
                if bad:
                    sys.exit(f'RECONCILIATION FAILED {tm}/{d}/{lab}: {bad}')
                if falsifier is None and len(f) > 1:
                    fb = f.copy(); fb.loc[fb.index[0], 'q'] *= 2.0
                    falsifier = recon(decompose(fb, BY, reg, stop_of(spec)), f, Wn)
                    if not falsifier:
                        sys.exit('FALSIFIER DID NOT FIRE: the reconciliation passed a corrupted copy')
                W, hs = fill_frame(f, D, HS, reg)
                s = np.where(f.side.values == 0, 1.0, -1.0)
                m0 = f.pt.values + 100.0 * D.spread.values / (s * D.q.values)
                ta = np.round(reg.loc[f.mid.values, 'ws'].values * 1000.0 + f.t.values * 1000.0, 3)
                m0b = lookup(BY, f.mid.values, ta, 'left'); okm = np.isfinite(m0) | np.isfinite(m0b)
                err = float(np.nanmax(np.abs(m0[okm] - m0b[okm]))) if okm.any() else 0.0
                if not np.array_equal(np.isfinite(m0), np.isfinite(m0b)) or err > 1e-6:
                    sys.exit(f'CONTROL FAILED: m0 from the spread term vs the frame map {tm}/{d}/{lab} (max err {err})')
                m0_max_err = max(m0_max_err, err)
                st['fills'] += len(D); st['no_m0'] += int(D.spread.isna().sum()); st['no_m1'] += int(D.plus1.isna().sum())
                st['no_mS'] += int(D.mark.isna().sum()); st['no_hs'] += int(np.isnan(hs).sum())
                hsq = hs * D.q.values / 100.0          # $ like the other terms (hs is in cents)
                R[(tm, lab, d)] = {g: sums(D, W[g], hsq) for g in GROUPS}
        L = maker_legs(d, sJ['tx_timing']['legs_retimed']); st['maker_legs'] += len(L)
        t = L.t_eng.values.astype(float); Dm, unt = maker_terms(L, t, BY, reg); st['maker_untimed'] += unt
        ok = np.isfinite(t); hm = lookup(HS, L.mid.values[ok], t[ok], 'left'); Dm = Dm.assign(hsq=hm * Dm.q.values / 100.0)
        st['maker_no_m0'] += int(Dm.spread.isna().sum()); st['maker_no_m1'] += int(Dm.plus1.isna().sum()); st['maker_no_hs'] += int(np.isnan(hm).sum())
        for part, msk in (('first 60 s', Dm.w60.values), ('0-240 s', np.ones(len(Dm), bool))):
            for bd in ('top 5', '6-20', 'rest', 'all makers'):
                g = Dm[msk] if bd == 'all makers' else Dm[msk & (Dm.band.values == bd)]
                r = dict(q=float(g.q.sum()))
                for c in ('spread', 'informed', 'plus1', 'hsq'):
                    okc = np.isfinite(g[c].values); r[c] = float(g[c].values[okc].sum()); r[f'q_{c}'] = float(g.q.values[okc].sum())
                MK[(part, bd, d)] = r
        del BY, HS, L, Dm
        io.mem_check(f'day {d}')

    out = dict(as_of=pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ'), days=DAYS, statuses=st, reconciliation='PASS (every arm-day)',
               falsifier_fired=falsifier, map_control=map_control, m0_control_max_err_c=m0_max_err,
               note='c / share unless $; in-sample, descriptive; RO arms mark / drift at 120 / 240 s', timings={}, makers={})
    for tm in ('J', 'X'):
        T = {}
        for lab in MAIN:
            P = {g: pool([R[(tm, lab, d)][g] for d in DAYS]) for g in GROUPS}
            a = P['all']
            row = dict(fills_shares=round(a['q'], 0), spread_c=round(cps(a, 'spread'), 3), half_spread_c=round(cps(a, 'hsq'), 3),
                       capture=round(cps(a, 'spread') / cps(a, 'hsq'), 3), adverse_1s_c=round(cps(a, 'informed'), 3),
                       plus1_c=round(cps(a, 'plus1'), 3), retained_1s=round(cps(a, 'plus1') / cps(a, 'spread'), 3),
                       drift_c=round(cps(a, 'drift'), 3), mark_c=round(cps(a, 'mark'), 3),
                       usd=dict(spread=round(a['spread'], 1), adverse_1s=round(a['informed'], 1), plus1=round(a['plus1'], 1),
                                drift=round(a['drift'], 1), mark=round(a['mark'], 1), carry=round(a['carry'], 1), settle=round(a['settle'], 1)),
                       groups={g: dict(share_of_q=round(P[g]['q'] / a['q'], 3) if a['q'] else None, spread_c=round(cps(P[g], 'spread'), 3),
                                       half_spread_c=round(cps(P[g], 'hsq'), 3), adverse_1s_c=round(cps(P[g], 'informed'), 3),
                                       plus1_c=round(cps(P[g], 'plus1'), 3), mark_c=round(cps(P[g], 'mark'), 3), plus1_usd=round(P[g]['plus1'], 1))
                               for g in GROUPS if g != 'all'})
            if lab != 'v1':
                dd = lambda c: [R[(tm, lab, d)]['all'][c] - R[(tm, 'v1', d)]['all'][c] for d in DAYS]
                dc = lambda c: [cps(R[(tm, lab, d)]['all'], c) - cps(R[(tm, 'v1', d)]['all'], c) for d in DAYS]
                row['vs_v1'] = dict(spread_usd_day=ci(dd('spread')), plus1_usd_day=ci(dd('plus1')), mark_usd_day=ci(dd('mark')),
                                    spread_c_day=ci(dc('spread')), plus1_c_day=ci(dc('plus1')), adverse_1s_c_day=ci(dc('informed')))
            row['per_day_plus1_c'] = {d: round(cps(R[(tm, lab, d)]['all'], 'plus1'), 3) for d in DAYS}
            row['per_day_plus1_usd'] = {d: round(R[(tm, lab, d)]['all']['plus1'], 2) for d in DAYS}
            T[lab] = row
        out['timings'][tm] = T
    # added after the first read (multiplicity companion to the +1 s intervals): exact 512 whole-day sign flips of the +1 s $ / day
    # difference vs v1; raw p per comparison and max-|t| over the 10 arms x 2 timings (pf_cap10_attribution.signflip, its controls run first)
    from .pf_cap10_attribution import signflip, controls
    controls()
    fam = [(tm, lab) for tm in ('J', 'X') for lab in MAIN if lab != 'v1']
    Mx = np.array([[R[(tm, lab, d)]['all']['plus1'] - R[(tm, 'v1', d)]['all']['plus1'] for (tm, lab) in fam] for d in DAYS])
    tt, praw, padj = signflip(Mx)
    out['plus1_signflip'] = {f'{tm} {lab}': dict(t=round(float(tt[i]), 2), p_raw=round(float(praw[i]), 4), p_maxt_20=round(float(padj[i]), 4))
                             for i, (tm, lab) in enumerate(fam)}
    for part in ('first 60 s', '0-240 s'):
        for bd in ('top 5', '6-20', 'rest', 'all makers'):
            P = pool([MK[(part, bd, d)] for d in DAYS])
            out['makers'][f'{part}|{bd}'] = dict(shares_day=round(P['q'] / len(DAYS), 0), spread_c=round(cps(P, 'spread'), 3),
                                                  half_spread_c=round(cps(P, 'hsq'), 3), capture=round(cps(P, 'spread') / cps(P, 'hsq'), 3),
                                                  adverse_1s_c=round(cps(P, 'informed'), 3), plus1_c=round(cps(P, 'plus1'), 3),
                                                  retained_1s=round(cps(P, 'plus1') / cps(P, 'spread'), 3))
    io.write_json(f'{S4}/report_spread_capture_v1.json', out)
    print(f"statuses {st}\nreconciliation PASS every arm-day; falsifier fired: {falsifier}; map control {map_control}; m0 max err {m0_max_err:.2e}")
    for tm in ('J', 'X'):
        print(f'\n== {tm}: c / share (spread vs mid at fill; half-spread of the venue book; adverse = mid move within 1 s; +1 s = spread + adverse;'
              f' drift to stop; mark = +1 s + drift) | reducing / increasing shares | vs v1 +1 s $/day')
        print(f"{'arm':14s} {'shares':>7s} {'spread':>7s} {'h-sprd':>7s} {'capt':>5s} {'advers':>7s} {'+1 s':>6s} {'ret':>5s} {'drift':>7s} {'mark':>6s} "
              f"| {'red%':>5s} {'red+1s':>6s} {'inc+1s':>6s} {'t>=60%':>6s} {'>=60+1s':>7s} | vs v1 +1 s $/day")
        for lab, r in out['timings'][tm].items():
            g = r['groups']; v = r.get('vs_v1')
            print(f"{lab:14s} {r['fills_shares']:7.0f} {r['spread_c']:7.3f} {r['half_spread_c']:7.3f} {r['capture']:5.2f} {r['adverse_1s_c']:+7.3f} {r['plus1_c']:6.3f} "
                  f"{r['retained_1s']:5.2f} {r['drift_c']:+7.3f} {r['mark_c']:6.3f} | {100 * g['red']['share_of_q']:5.1f} {g['red']['plus1_c']:6.3f} {g['inc']['plus1_c']:6.3f} "
                  f"{100 * g['t>=60']['share_of_q']:6.1f} {g['t>=60']['plus1_c']:7.3f}"
                  + (f" | {v['plus1_usd_day']['mean']:+6.2f} ({v['plus1_usd_day']['lo']:+.2f}..{v['plus1_usd_day']['hi']:+.2f})" if v else ''))
    print('\n== +1 s $ / day vs v1: exact sign-flip p (raw / max-|t| over 10 arms x 2 timings)')
    for k, r in out['plus1_signflip'].items():
        print(f"{k:16s} t {r['t']:+6.2f}  p_raw {r['p_raw']:.4f}  p_maxt {r['p_maxt_20']:.4f}")
    print('\n== real makers, same 9 days (engine tx timing): c / share')
    for k, r in out['makers'].items():
        print(f"{k:24s} sh/day {r['shares_day']:8.0f} spread {r['spread_c']:6.3f} h-sprd {r['half_spread_c']:6.3f} capt {r['capture']:5.2f} "
              f"adverse {r['adverse_1s_c']:+6.3f} +1 s {r['plus1_c']:6.3f} ret {r['retained_1s']:5.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
