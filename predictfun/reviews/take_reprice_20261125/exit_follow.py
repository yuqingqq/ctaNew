"""Exits for our chase takes that follow how the top taker exits, and the P&L (USER 2026-11-27: "Design an exit mechanism, follow what top
takers do, then check the pnl performance"). Development evidence on the four consumed days, every market; no orders; declared before the run.
What the top taker does (EXIT_TIMES / EXIT_PROFILE, every match record, settlement-timed): it exits only by taking; 48 % of what it opens is
taken back after a median 39 s (middle half 16-75 s), 51 % is held to settlement (mostly favourites: 60 % of that volume above 60c); its
take-backs are gains of about +15c and losses of about -14c, losses cut sooner; 3 % of its exit volume falls in the window's last 30 s.
Entries (CHASE_TAKES' rules, decided 3.5 ms after the tick, matched 175.5 ms after it): a 50-share IOC at the known best + 3c / + 10c on
FILL_AWARE's selections (scores and pooled thresholds reproduced: a check): MOVE top 10 % / top 1 %, DECOMP_s10 top 1 %, DIRECT_TB_s3 top 10 %;
control: 5,000 moves at their first tick (seeded). Entry books: optimistic = the last snapshot at or before the match; conservative = its
levels at their size in the next snapshot. Exits on the last snapshot at or before the exit match for both (CHASE_TAKES' take-back rule;
changed after the one-day rehearsal: the next-snapshot rule on exits sends unsold shares to settlement, which is not conservative).
Exit policies (declared; the side's mid on the known clock against our entry price, its way; every exit an IOC at every shown level,
decided at the known time of the frame that triggers it, matched 172 ms later; the rest held to settlement, ties 50; fees 2 % x min(p, 100 - p)):
  HOLD: to settlement. TB1S: taken back 1 s after the entry match (FILL_AWARE's take-back, reproduced: a check). TB39: 39 s after it.
  TPSL: taken back when the side's mid is >= entry + 15c or <= entry - 14c, until 30 s before the window's end; then held.
  FAV_ONLY: at 39 s, taken back if the side's mid is below 60c, else held to settlement.
  FOLLOW: TPSL, and at 39 s taken back if the side's mid is below 60c (a favourite stays under TPSL); the last 30 s held.
  FOLLOW_T75: FOLLOW with 75 s. FOLLOW_FAV50: FOLLOW with 50c.
  A time or trigger in the last 30 s, or an exit match beyond the market's recorded path, means held (counted).
P&L per action = its way x (exit proceeds + the rest x 100 x payout - entry cost) - fees, in c for the shares filled; per intended share
(/ 50 per action) and in $; by day; days > 0; exit mix and holding time. Paying (computed): per intended share > 0 pooled AND >= 3 of 4 days
AND >= 150 actions. Independent positions (no position cap, our own trades do not move the book). Seen days.
Outputs (write-once): OUT/exit_follow/<day>.parquet (per action), exit_follow.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_follow.py \
       selftest | labels <day> [<dir>] | read [<dir>]
"""
import collections
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_takes as CT  # noqa: E402
import fill_aware as FA  # noqa: E402

C, TR, TK, TE, TM, WT, F, VR, MM, DV = CT.C, CT.TR, CT.TK, CT.TE, CT.TM, CT.WT, CT.F, CT.VR, CT.MM, CT.DV
OUT = WT.OUT / 'exit_follow'
DAYS, Q, fee = CT.DAYS, CT.Q, CT.fee
TP, SL, T_MED, T_UQ, FAV, LAST_MS = 15., 14., 39000., 75000., 60., 30000.
DEC, NET, HOLD = 0., CT.NET, TM.HOLD
POLICIES = ('HOLD', 'TB1S', 'TB39', 'TPSL', 'FAV_ONLY', 'FOLLOW', 'FOLLOW_T75', 'FOLLOW_FAV50')
SELS = ('MOVE_top10', 'MOVE_top1', 'DECOMP_s10_top1', 'DIRECT_TB_s3_top10', 'CONTROL_onset')
N_CONTROL, MIN_ORDERS, SEED = 5000, 150, 20261127
COMBOS = [(b, s) for b in ('A', 'AB') for s in (3., 10.)]


def selections():
    """FILL_AWARE's selections (pooled thresholds) and the control, as (day, t, d) rows with one flag column per selection."""
    P = pd.read_parquet(FA.OUT / 'predictions.parquet')
    pts = pd.concat([pd.read_parquet(CT.OUT / f'{d}_points.parquet', columns=['day', 't', 'd', 'mv_k']) for d in DAYS], ignore_index=True)
    P = P.merge(pts, on=['day', 't', 'd'], how='left', validate='1:1')
    key = P.key.to_numpy()
    ok = np.isfinite(P.MOVE.to_numpy(float))
    flags = {}
    for nm, col, q in (('MOVE_top10', 'MOVE', .1), ('MOVE_top1', 'MOVE', .01), ('DECOMP_s10_top1', 'DECOMP_s10', .01), ('DIRECT_TB_s3_top10', 'DIRECT_TB_s3', .1)):
        sc = P[col].to_numpy(float)
        okk = ok & np.isfinite(sc)
        flags[nm] = CT.first_per_move(okk & (sc >= np.quantile(sc[okk], 1. - q)), key)
    on = CT.first_per_move(P.mv_k.to_numpy(float) == 1, key)
    flags['CONTROL_onset'] = np.sort(np.random.default_rng(SEED).choice(on, min(N_CONTROL, len(on)), replace=False))
    J = json.loads((H / 'fill_aware.json').read_text())
    for nm in SELS[:-1]:
        C.require(len(flags[nm]) == J['selections'][nm]['actions'], f'{nm}: FILL_AWARE\'s selection not reproduced')
    rows = np.unique(np.concatenate(list(flags.values())))
    S = P.iloc[rows][['day', 't', 'd', 'key']].reset_index(drop=True)
    for nm, ix in flags.items():
        S[nm] = np.isin(rows, ix)
    return S


def trigger(kt, sm, e, cut):
    """The known time of the first frame (up to cut) whose side mid is >= e + TP or <= e - SL, or None."""
    hit = (kt <= cut) & np.isfinite(sm) & ((sm >= e + TP) | (sm <= e - SL))
    return float(kt[np.argmax(hit)]) if hit.any() else None


def mid_at(kt, sm, x):
    j = int(np.searchsorted(kt, x, 'right') - 1)
    return sm[j] if j >= 0 else np.nan


def decide(policy, trig, mid39, mid75, act, cut):
    """Exit decision time (path clock, the known time of the triggering frame) or None = held to settlement; trig = trigger(...),
    mid39 / mid75 = the side's mid known 39 / 75 s after the entry match; cut = the last exit time (30 s before the window's end)."""
    if policy == 'HOLD':
        return None
    if policy == 'TB1S':
        return act + 1000.
    if policy == 'TB39':
        return act + T_MED if act + T_MED <= cut else None
    if policy == 'TPSL':
        return trig
    T = T_UQ if policy == 'FOLLOW_T75' else T_MED
    fav = 50. if policy == 'FOLLOW_FAV50' else FAV
    mid_T = mid75 if policy == 'FOLLOW_T75' else mid39
    if policy == 'FAV_ONLY':
        return act + T if (act + T <= cut and np.isfinite(mid_T) and mid_T < fav) else None
    if trig is not None and trig <= act + T:
        return trig
    if act + T <= cut and np.isfinite(mid_T) and mid_T < fav:
        return act + T
    return trig


def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = selections()
    S = S[S.day == day].reset_index(drop=True)
    L = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['t', 'd', 'mid', 'lim', 'payout'] + [f'{x}{b}_{k}' for b in 'AB' for x in ('p', 's') for k in range(5)])
    S = S.merge(L, on=['t', 'd'], how='left', validate='1:1')
    C.require(S.mid.notna().all(), 'an action without its CHASE_TAKES labels')
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    kind = {(b, s, p): np.full(len(S), '', dtype=object) for b, s in COMBOS for p in POLICIES}
    status = collections.Counter()
    parity_bad = 0
    src = MM.KP.Tape(); off = src.offset(day)
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}: CHASE_TAKES labelled it')
            p, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
            C.require(p is not None, f'{m}: no path')
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            f0, fk = f[:, 0], f[:, 1]
            okb = (f[:, 2] >= 0) & (f[:, 12] <= 100) & (f[:, 2] < f[:, 12])
            midy = np.where(okb, (f[:, 2] + f[:, 12]) / 2., np.nan)
            cut = float(rg.loc[m, 'we']) * 1000. - t0 - LAST_MS
            for i in by_mid[m]:
                t, d, lim, pay = float(S.t.iat[i]), float(S.d.iat[i]), float(S.lim.iat[i]), float(S.payout.iat[i])
                act = t + TM.FEED + TM.DECIDE - t0 + NET + HOLD
                jv = int(np.searchsorted(f0, act, 'right') - 1)
                side = 1 if d > 0 else 0
                pa, sa = TK.side_levels(f[jv], side)
                pb, sb = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                if not (np.array_equal(pa, S.loc[i, [f'pA_{k}' for k in range(5)]].to_numpy(float)) and np.array_equal(sa, S.loc[i, [f'sA_{k}' for k in range(5)]].to_numpy(float))):
                    parity_bad += 1
                bk = {'A': (pa[None], sa[None]), 'AB': TK.minbook(pa[None], sa[None], pb[None], sb[None])}
                k0 = int(np.searchsorted(fk, act, 'right'))
                kt = fk[k0:]
                sm = midy[k0:] if d > 0 else 100. - midy[k0:]
                for b, s in COMBOS:
                    fl, co, fe_ = (x[0] for x in TK.vwalk(*bk[b], np.array([d]), np.array([lim + d * s])))
                    o[f'{b}_s{int(s)}_fill'][i] = fl
                    if fl <= 0:
                        for pol in POLICIES:
                            o[f'{b}_s{int(s)}_{pol}_pnl'][i] = 0.
                            kind[(b, s, pol)][i] = 'no_fill'
                        continue
                    e = co / fl if d > 0 else 100. - co / fl
                    trig, m39, m75 = trigger(kt, sm, e, cut), mid_at(kt, sm, act + T_MED), mid_at(kt, sm, act + T_UQ)
                    for pol in POLICIES:
                        x = decide(pol, trig, m39, m75, act, cut)
                        xq = xc = xf = 0.
                        kd = 'held'
                        if x is not None:
                            mt = x + DEC + NET + HOLD
                            if mt > end:
                                kd = 'held_beyond_path'
                            else:
                                je = int(np.searchsorted(f0, mt, 'right') - 1)
                                eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
                                xq_, xc_, xf_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([fl]))
                                xq, xc, xf = float(xq_[0]), float(xc_[0]), float(xf_[0])
                                kd = 'tb' if pol in ('TB1S', 'TB39') else ('time' if (pol != 'TPSL' and abs(x - (act + (T_UQ if pol == 'FOLLOW_T75' else T_MED))) < 1e-6) else 'trigger')
                                o[f'{b}_s{int(s)}_{pol}_hold_ms'][i] = x - act
                        o[f'{b}_s{int(s)}_{pol}_pnl'][i] = d * (xc + (fl - xq) * 100. * pay - co) - fe_ - xf
                        o[f'{b}_s{int(s)}_{pol}_exit_share'][i] = xq / fl
                        kind[(b, s, pol)][i] = kd
                status['actions'] += 1
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    C.require(parity_bad == 0, f'{parity_bad} entry books differ from CHASE_TAKES\' labels')
    out = S[['day', 't', 'd', 'key', 'mid', 'payout'] + list(SELS)].copy()
    for k_, v in o.items():
        out[k_] = v
    for (b, s, pol), v in kind.items():
        out[f'{b}_s{int(s)}_{pol}_kind'] = v
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, actions=int(len(S)), status=dict(status)))
    print(day, len(S), dict(status), 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_follow.json'
    C.require(not fn.exists(), 'exit_follow.json exists (write-once)')
    src = od or OUT
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    day = X.day.to_numpy()
    out = dict(asof='2026-11-27', rehearsal=reh, actions=int(len(X)), selections={}, verdict={})
    for sel in SELS:
        m = X[sel].to_numpy(bool)
        so = dict(actions=int(m.sum()), cells={})
        for b, s in COMBOS:
            pre = f'{b}_s{int(s)}'
            fl = X[f'{pre}_fill'].to_numpy(float)
            for pol in POLICIES:
                pn = X[f'{pre}_{pol}_pnl'].to_numpy(float)
                kd = X[f'{pre}_{pol}_kind'].to_numpy()
                ii = np.flatnonzero(m)
                by = {dh: float(pn[ii][day[ii] == dh].sum() / (Q * (day[ii] == dh).sum())) for dh in DAYS if (day[ii] == dh).any()}
                dp = int(sum(v > 0 for v in by.values()))
                val = float(pn[ii].sum() / (Q * len(ii))) if len(ii) else None
                filled = ii[fl[ii] > 0]
                hm = X[f'{pre}_{pol}_hold_ms'].to_numpy(float)[filled] if f'{pre}_{pol}_hold_ms' in X else np.array([])
                so['cells'][f'{pre}_{pol}'] = dict(per_intended_c=val, per_filled_share_c=float(pn[filled].sum() / fl[filled].sum()) if len(filled) else None,
                                                  total_usd=float(pn[ii].sum() / 100.), usd_by_day={dh: float(pn[ii][day[ii] == dh].sum() / 100.) for dh in DAYS if (day[ii] == dh).any()},
                                                  by_day=by, days_positive=dp, fill=float(fl[ii].sum() / (Q * len(ii))) if len(ii) else None,
                                                  exit_mix={k: float(np.mean(kd[filled] == k)) for k in ('held', 'tb', 'time', 'trigger', 'held_beyond_path')} if len(filled) else None,
                                                  exit_hold_s_median=float(np.nanmedian(hm) / 1000.) if np.isfinite(hm).any() else None,
                                                  paying=bool(val is not None and val > 0 and dp >= 3 and len(ii) >= MIN_ORDERS))
        out['selections'][sel] = so
    # the 1 s take-back against FILL_AWARE's (same entries, same exit rule)
    FJ = json.loads((H / 'fill_aware.json').read_text())
    par = {}
    for sel in SELS[:-1]:
        for b, s in COMBOS:
            mine = out['selections'][sel]['cells'][f'{b}_s{int(s)}_TB1S']['per_intended_c']
            ref = FJ['selections'][sel]['books'][b][f's{int(s)}_take_back_1s']['value_c']
            par[f'{sel}_{b}_s{int(s)}'] = dict(here=mine, fill_aware=ref)
            if not reh:
                C.require(abs(mine - ref) < 1e-9, f'{sel} {b} s{int(s)}: the 1 s take-back differs from FILL_AWARE ({mine} vs {ref})')
    out['take_back_parity'] = par
    cells = [(sel, k, c) for sel, so in out['selections'].items() for k, c in so['cells'].items()]
    strat = [x for x in cells if x[0] != 'CONTROL_onset']
    out['verdict'] = dict(cells=len(cells), paying=[dict(selection=a, cell=k, per_intended_c=c['per_intended_c'], total_usd=c['total_usd'], days_positive=c['days_positive'])
                                                   for a, k, c in strat if c['paying']],
                          paying_conservative=[dict(selection=a, cell=k, per_intended_c=c['per_intended_c'], total_usd=c['total_usd'], days_positive=c['days_positive'])
                                               for a, k, c in strat if c['paying'] and k.startswith('AB_')],
                          best_conservative=max(((a, k, c['per_intended_c'], c['total_usd'], c['days_positive']) for a, k, c in strat if k.startswith('AB_') and c['per_intended_c'] is not None),
                                                key=lambda x: x[2]),
                          control_paying=[k for a, k, c in cells if a == 'CONTROL_onset' and c['paying']])
    TR.dump(fn, out)
    print(json.dumps(out['verdict'], indent=1, default=str)[:3000])
    for sel, so in out['selections'].items():
        for pre in ('AB_s3', 'AB_s10', 'A_s10'):
            print(sel, pre, {pol: (round(so['cells'][f'{pre}_{pol}']['per_intended_c'], 2), so['cells'][f'{pre}_{pol}']['days_positive']) for pol in POLICIES})


def decide_arrays(policy, kt, sm, e, act, cut):
    return decide(policy, trigger(kt, sm, e, cut), mid_at(kt, sm, act + T_MED), mid_at(kt, sm, act + T_UQ), act, cut)


def selftest():
    decide_ = decide_arrays
    kt = np.array([1000., 2000., 3000., 40000., 50000., 90000.])
    sm = np.array([50., 52., 66., 55., 30., 80.])
    act, cut, e = 0., 100000., 50.
    assert decide_('HOLD', kt, sm, e, act, cut) is None and decide_('TB1S', kt, sm, e, act, cut) == 1000.
    assert decide_('TB39', kt, sm, e, act, cut) == 39000. and decide_('TB39', kt, sm, e, act, 30000.) is None
    assert decide_('TPSL', kt, sm, e, act, cut) == 3000.                                 # 66 >= 50 + 15
    assert decide_('TPSL', kt, sm, e, act, 2500.) is None                                # the trigger falls after the cut: held
    sm2 = np.array([50., 52., 54., 55., 30., 80.])
    assert decide_('TPSL', kt, sm2, e, act, cut) == 50000.                               # 30 <= 50 - 14
    assert decide_('FAV_ONLY', kt, sm2, e, act, cut) == 39000.                           # mid at 39 s = 54 < 60
    assert decide_('FOLLOW', kt, sm2, e, act, cut) == 39000.                             # no trigger by 39 s, not a favourite
    assert decide_('FOLLOW_FAV50', kt, sm2, e, act, cut) == 50000.                       # 54 >= 50: a favourite, the stop at 50 s
    assert decide_('FOLLOW_T75', kt, sm2, e, act, cut) == 50000.                         # the stop comes before 75 s
    kt3, sm3 = np.array([1000., 2000., 3000., 38000., 50000., 90000.]), np.array([50., 52., 54., 62., 61., 63.])
    assert decide_('FOLLOW', kt3, sm3, e, act, cut) is None                             # a favourite at 39 s (62 known since 38 s), no trigger: held
    assert decide_('FOLLOW', kt, sm3, e, act, cut) == 39000.                            # the same prices a frame later: 54 known at 39 s, taken back
    sm4 = np.array([np.nan, 52., 54., np.nan, 61., 63.])
    assert decide_('FAV_ONLY', kt, sm4, e, act, cut) == 39000.                       # the last valid mid before 39 s is 54
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_follow selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
