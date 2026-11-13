"""MOVE-DOWN + WHY-GONE (2026-11-13, session ce68ebe0). USER: "The requote side, you mean their fill lands around 300ms but we repost 350 ms
later, can we do faster. Bid side, why we left?"
After SIDE-MOVES (94cc2d61): on the protection side (the side the price moves into: asks in a pump, bids in a dump) the top makers quoting
the old best are filled at the new best a median 302 ms after it forms, usually first in its queue; our rung 0 has no order at the birth 89%
of the time (the guard pulled it) and an order there in 35% of events, live at 347 ms. On the other side our held old-best order is gone
within 1 s two-thirds of the time.

PART 1, MOVE-DOWN (protection side, rung 0 of the guard ladder: 30 sh, kept through partial fills; deeper rungs unchanged and independent,
so a ladder difference equals the rung-0 difference). Today (LDG) rung 0 is pulled while the guard is on (v <= -0.10) and re-enters at the
known best once the guard has been clear for 250 ms. Arms:
  LDG    today (reference)
  FAST   today's rule with BEST-GAP's fast path (5 ms order path, book known at stamp + 15.2 ms)
  MD     move down: once the known best is worse than it was when the guard came on (the move has happened), enter at the new best at once,
         while the guard is still on and without the 250 ms wait; keep that order through the rest of this guard episode (pulled at the
         next onset, held / drift / cross rules as today); after it is used up, enter again at the current best if still moved
  MDP    MD, but while the guard is on the order is pulled if PULL-DEPTH's out-of-day P(move >= 1 more tick) >= 0.25 (tau as before)
  MDF    MD with the FAST path
  Implemented in the simulator's rest lane with idle-only triggers, a decision callback and the 250 ms wait carried by the entry line
  (the simulator's own wait switched off); with entries during the guard disabled this must equal LDG (control i).
Measured at the protection side's new best after a signal (PULL-DEPTH's adverse levels; calm fallback births beside), M 50:
  presence  our rung-0 order live at the new price while it is the best (<= 2 s): share of events, go-live after the birth (median,
            share <= 100 / 250 ms), queue ahead at go-live (shown size, rule M)
  LDG's delay parts: guard still on at the birth (share), birth to the guard clearing (median)
  fills     WS-timed, aged: shares, c/sh at 1 s, STOP share (top 5 there: +0.30 c/sh, 82% STOP, PULL-DEPTH)
  $/day     rung 0, w/o young, at 1 s: WS / E2 / all trades, M 0 / 50 / 250; age split (C1, C12); cancel-pending share (C3)
PART 2, WHY-GONE (the other side: bids in a pump, asks in a dump; SIDE-MOVES' 'other' events and calm improvements), LDG rung 0, M 50,
the simulator copy with order bookkeeping (sim4): at the birth HELD-OLD / AT-NEW / HELD-OTHER / PEND / OUT; for a held old order, how it
ended within 1 s (filled; trig = our guard on that side; drift = the best moved > 3 ticks away; cross; nobook; or still live) and when;
for OUT, how the last order ended and how long before the birth; the guard on that side at the birth.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule >= 7 of 9; a day with < 30 events / fills in a compared cell does not
count):
 R520 FASTER (MD, MDF, FAST against LDG): median go-live after the birth (events with our order present), lower on >= 7 days -> FASTER;
      higher on >= 7 -> SLOWER; else NOT-SHOWN. Presence, <= 100 ms share and queue ahead printed.
 R521 VALUE (MD, MDP, MDF, FAST against LDG): rung-0 $/day w/o young at 1 s on WS-timed trades, higher at M 0 AND M 50 on >= 7 days ->
      HELPS; lower at both on >= 7 -> HURTS; else NOT-SHOWN. M 250 (sign holds / flips) and all trades printed.
 R522 AT-THE-NEW-BEST (MD, MDP, MDF against LDG): aged WS c/sh at 1 s of fills at adverse levels, higher on >= 7 days -> HIGHER; lower on
      >= 7 -> LOWER; else NOT-SHOWN. STOP share printed.
 R523 WHY-GONE (descriptive): the state and end-reason splits of PART 2.
What the outcomes mean, fixed in advance: R520 FASTER = dropping the guard-clear and 250 ms waits gets our order to the new best sooner;
R521 / R522 HELPS / HIGHER = being there sooner pays at rule M; HURTS / LOWER = the new best right after a move is still dangerous for an
order that arrives after it forms (the top makers' front place there is not reached by arriving sooner after the birth). A pass is a
candidate on consumed days (C8).
Controls (any failure -> no output): (i) MD with entries during the guard disabled equals LDG fill for fill and order for order on every
window-side and clock, and the same under the FAST path equals BEST-GAP's FAST lane; MD differs from LDG somewhere; (ii) sim4 equals the
simulator (LDG rung 0, M 50) on every window-side; (iii) every rung-0 order maps to its placement step (price = the known best then);
(iv) synthetic cases (the entry line, the move test for both sides, the episode keep rule) with three planted errors (the 250 ms tail
dropped from the entry line, an MD order kept into a later episode, the move test's sign flipped) that must each fail a case; clocks and
the cooldown switch restored after every run; (v) memory self-guard.
Limits fixed in advance: replay rule M (B1a, C5), replay volumes inflated (C4); consumed days (C8); rung 0 only; MDP uses PULL-DEPTH's P
(nested-fold leak and known-frame target, unmeasured, per the other session's review); the FAST path is hypothetical; ours simulated (C9);
the lane is the guard ladder, not the frozen v3 (C11).
Usage (run_light, PF_SLOT=claude): pf_move_down.py selftest | day <day> | report; driver run_move_down.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_queue_place as QP                               # noqa: E402  (sim4, run_at)
import pf_timing_split as TS                              # noqa: E402  (WS flag)
import pf_best_gap as BG                                  # noqa: E402  (FAST path)
import pf_pull_depth as PD                                # noqa: E402  (events, P)
import pf_side_moves as SM                                # noqa: E402  (ours_at)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/move_down_20261113'
ARMS = ('LDG', 'FAST', 'MD', 'MDP', 'MDF')
CFGS = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
SIZE = 30.0; TAU_MDP = 0.25; LEVEL_MS = PD.LEVEL_MS; MIN_N = 30
AGEB = (('< 100 ms', -1.0, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, 1e18))
INF = float('inf'); NAN = float('nan')
MUT_NO_TAIL = MUT_ANY_EPISODE = MUT_MOVE_SIGN = 0          # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def entry_line(vg, tg, cool=None):
    """LDG's entry block without the simulator's own wait: blocked if the guard was on at any step in (t - cool, t]."""
    cool = (0.0 if MUT_NO_TAIL else K.REST_COOL) if cool is None else cool
    out = []; last = -INF
    for k in range(len(vg)):
        if vg[k]:
            last = tg[k]
        out.append(bool(tg[k] - last < cool) if last > -INF else False)
    return out


def episodes(vg):
    ep = []; e = -1
    for k in range(len(vg)):
        if vg[k] and (k == 0 or not vg[k - 1]):
            e += 1
        ep.append(e if vg[k] else -1)
    return ep


def moved(vg, best, side):
    """Steps where the guard is on and the known best is worse than when this guard episode began (the move has happened)."""
    out = []; bon = -1; lastv = -1
    for k in range(len(vg)):
        if best[k] >= 0:
            lastv = best[k]
        if not vg[k]:
            bon = -1; out.append(False); continue
        if k == 0 or not vg[k - 1]:
            bon = best[k] if best[k] >= 0 else lastv
        b = best[k]
        worse = (b > bon) if (side == 0) == bool(MUT_MOVE_SIGN) else (b < bon)
        out.append(bool(bon >= 0 and b >= 0 and worse))
    return out


def md_dec(vg, tg, ep, P1=None, tau=None):
    """While the guard is on pull, except an order entered during this episode (MD: keep; MDP: pull if P(move >= 1) >= tau)."""
    tgl = list(tg)

    def dec(k, t, cur):
        if not vg[k]:
            return False
        kp = bisect.bisect_left(tgl, cur['ts'] - 1e-6)
        if kp < len(tgl) and abs(tgl[kp] - cur['ts']) < 1e-6 and vg[kp] and (MUT_ANY_EPISODE or ep[kp] == ep[k]):
            if P1 is None:
                return False
            p = P1[k]
            return bool((not (p == p)) or p >= tau)
        return True
    return dec


def selftest():
    bad = []
    tg = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    vg = [False, True, True, False, False, False, False, False, False]
    if entry_line(vg, tg) != [False, True, True, True, True, True, True, False, False]:   # last guard step 100 -> clear from 350
        bad.append(f'entry line {entry_line(vg, tg)}')
    best = [50, 50, 49, 49, 49, 49, 49, 49, 49]
    if moved(vg, best, 0) != [False, False, True] + [False] * 6:
        bad.append(f'bid move {moved(vg, best, 0)}')
    if moved(vg, [50, 50, 51, 51, 51, 51, 51, 51, 51], 1) != [False, False, True] + [False] * 6:
        bad.append('ask move')
    if moved(vg, [50, 50, 51, 51, 51, 51, 51, 51, 51], 0) != [False] * 9:
        bad.append('a better bid is not a move')
    vg2 = [True, True, False, True, True]; tg2 = [0.0, 50.0, 100.0, 150.0, 200.0]; ep2 = episodes(vg2)
    if ep2 != [0, 0, -1, 1, 1]:
        bad.append(f'episodes {ep2}')
    d = md_dec(vg2, tg2, ep2)
    if d(1, 50.0, dict(ts=50.0)) is not False or d(4, 200.0, dict(ts=50.0)) is not True or d(4, 200.0, dict(ts=150.0)) is not False:
        bad.append('episode keep rule')
    if d(1, 50.0, dict(ts=-50.0)) is not True or d(2, 100.0, dict(ts=50.0)) is not False:
        bad.append('LDG pull / guard off')
    dp = md_dec(vg2, tg2, ep2, P1=[0.1, 0.3, 0.0, 0.2, np.nan], tau=0.25)
    if dp(1, 50.0, dict(ts=0.0)) is not True or dp(3, 150.0, dict(ts=150.0)) is not False or dp(4, 200.0, dict(ts=150.0)) is not True:
        bad.append('MDP rule')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_NO_TAIL', 'MUT_ANY_EPISODE', 'MUT_MOVE_SIGN'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'MOVE-DOWN SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


def run_arm(w, q, shf, arm, vg, P1, allow_off=False):
    """One rung-0 arm; returns (fills, orders, pulls, context best, L_PLACE used)."""
    if arm == 'LDG':
        F, O, p = K.sim_arm(w, q, shf, 'rest', aux=vg, size=SIZE)
        return F, O, p, w['best'], K.L_PLACE
    if arm == 'FAST':
        F, O, p = BG.run_lane(w, q, shf, vg, SIZE, 'FAST')
        return F, O, p, BG.fast_context(w, 'FAST')['best'], 5.0
    fast = arm == 'MDF'
    ctx = BG.fast_context(w, 'FAST') if fast else w
    blk = entry_line(vg, w['tg']); ep = episodes(vg)
    mv = [False] * len(vg) if allow_off else moved(vg, ctx['best'], w['side'])
    aux = [b_ and not a_ for b_, a_ in zip(blk, mv)]
    dec = md_dec(vg, w['tg'], ep, P1=P1 if arm == 'MDP' else None, tau=TAU_MDP)
    path = 5.0 if fast else BG.CLK0['L_PLACE']
    try:
        K.MUT_REST_NOCOOL = 1
        if fast:
            K.L_PLACE = 5.0; K.L_CANCEL = 5.0; K.L_PENDCANCEL = 5.0
        F, O, p = K.sim_arm(ctx, q, shf, 'rest', aux=aux, decide=dec, idle_only=1, size=SIZE)
    finally:
        K.MUT_REST_NOCOOL = 0
        K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL = BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL']
    return F, O, p, ctx['best'], path


def order_prices(O, tg, cbest, lplace):
    """Rung 0 places at the known best of its placement step: (ta, end, a0, ref, price); price -9 if the step is not found."""
    out = []; bad = 0
    for ta, end, a0, ref in O:
        k = int(np.searchsorted(tg, ta - lplace - 1e-6))
        if k < len(tg) and abs(tg[k] - (ta - lplace)) < 1e-6:
            out.append((ta, end, a0, ref, int(cbest[k])))
        else:
            out.append((ta, end, a0, ref, -9)); bad += 1
    return out, bad


def day_run(day):
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WS = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    GRP = pd.Series(grp, index=LT.index)
    TPG = pd.Series(LT.ptick.values).groupby(grp).agg(list).to_dict()
    PR = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet', columns=['m', 'side', 'k', 'P1']); PRI = PR.groupby(['m', 'side']).indices
    EV = pd.read_parquet(f'{PD.OUTD}/events_{day}.parquet', columns=['eid', 'kind', 'side', 'p', 'tb', 'life'])
    EVI = {}
    for e in EV.itertuples(index=False):
        EVI.setdefault((int(e.eid.split('|')[1]), int(e.side)), []).append((str(e.kind), int(e.p), float(e.tb), float(e.tb) + min(float(e.life), LEVEL_MS)))
    st.update(window_sides=0, ctrl_ldg=0, ctrl_fast=0, md_differs=0, sim4_differ=0, unmapped=0, clock_ok=1)
    fills = []; lev = []; pres = []; why = []

    def ttype(r_, x, p):
        return DT.trade_type(x, p, TPG[GRP.at[r_]]) if GRP.at[r_] in TPG else 'n/a'
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tg = w['tg']; P5, opp = w['P5'], w['opp']
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        P1 = np.full(n, np.nan)
        if (m, side) in PRI:
            x_ = PR.iloc[PRI[(m, side)]]; P1[x_.k.values] = x_.P1.values
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        evs = EVI.get((m, side), [])
        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)
        for q, shf in CFGS:
            cfg = f'{q}{int(shf)}'; tt = w['anc'] - shf
            res = {a_: run_arm(w, q, shf, a_, vg, P1) for a_ in ARMS}
            c0 = run_arm(w, q, shf, 'MD', vg, P1, allow_off=True); st['ctrl_ldg'] += int(c0[0] != res['LDG'][0] or c0[1] != res['LDG'][1])
            c1 = run_arm(w, q, shf, 'MDF', vg, P1, allow_off=True); st['ctrl_fast'] += int(c1[0] != res['FAST'][0] or c1[1] != res['FAST'][1])
            st['md_differs'] += int(res['MD'][0] != res['LDG'][0])
            if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL, K.MUT_REST_NOCOOL) != (BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL'], 0):
                st['clock_ok'] = 0
            for arm, (F, O, pulls, cb, lp) in res.items():
                st[f'pulls|{arm}|{cfg}'] = st.get(f'pulls|{arm}|{cfg}', 0) + int(pulls)
                for f in F:
                    i, pt, qq, thr, age, ph, a0 = f[0], int(f[1]), f[2], f[3], f[4], f[5], f[6]
                    t = float(w['anc'][i])
                    if not (w['ws'] <= t < w['ws'] + 61000.0):
                        continue
                    ty = 'THROUGH' if thr else (ttype(lrow[i], side, pt) if src[i] else 'n/a')
                    v1 = val(t, 1000.0, pt)
                    fills.append((int(day), cfg, arm, 'WS' if src[i] else 'E2', age < 100.0, ph == 'pend', float(qq), v1, val(t, 5000.0, pt), ty, float(age)))
                    if cfg == 'M50' and src[i]:
                        for kev, p, tb, tend in evs:
                            if kev in ('adverse', 'calm fallback') and p == pt and tb < t <= tend:
                                lev.append((int(day), kev, arm, float(qq), v1, float(tt[i] - age - tb), ty, age >= 100.0, float(a0)))
                                break
                if cfg == 'M50':
                    Op, bad = order_prices(O, tg, cb, lp); st['unmapped'] += bad
                    for kev, p, tb, tend in evs:
                        if kev not in ('adverse', 'calm fallback'):
                            continue
                        c = [o for o in Op if not o[3] and o[4] == p and o[0] <= tend and o[1] > tb]
                        kb = int(np.searchsorted(tg, tb, side='right')) - 1
                        gon = bool(kb >= 0 and vg[kb])
                        kc = kb + 1
                        while gon and kc < n and vg[kc]:
                            kc += 1
                        clr = (float(tg[kc]) - tb) if (gon and kc < n) else (0.0 if not gon else NAN)
                        pres.append((int(day), kev, arm, bool(c), float(c[0][0] - tb) if c else NAN, float(c[0][2]) if c else NAN, gon, clr))
        # PART 2: the other side, LDG rung 0, M 50, with order bookkeeping
        Fo, Ot, O4, _ = QP.sim4(side, tf, opp, P5, w['S5'], w.setdefault('cache', {}), tg, w['vkl'], -INF, w['best'], w['bopp'], w['anc'] - 50.0,
                                w['ptl'], w['shl'], 'M', w['te'], vg, 0, QP.SIZE)
        Fk, Ok, _ = K.sim_arm(w, 'M', 50.0, 'rest', aux=vg, offset=0, size=QP.SIZE)
        st['sim4_differ'] += int(Fk != Fo or Ok != Ot)
        b0 = P5[:, 0].astype('int64')
        for kev, p, tb, tend in evs:
            if kev not in ('other', 'calm improve'):
                continue
            jb = int(np.searchsorted(tf, tb, side='left')); po = int(b0[jb - 1]) if jb >= 1 else -1
            s0, _, _ = SM.ours_at(O4, p, po, tb, tend)
            kb = int(np.searchsorted(tg, tb, side='right')) - 1; gon = bool(kb >= 0 and vg[kb])
            live = [o for o in O4 if not o['ref'] and o['ta'] <= tb < min(o['tr'], o['tfill']) and o['pt'] == po]
            rsn = 'n/a'; dt = NAN
            if s0 == 'HELD-OLD' and live:
                o = live[-1]; e_ = min(o['tc'], o['tfill'])
                if e_ > tb + 1000.0:
                    rsn = 'still live'
                else:
                    rsn = 'filled' if o['tfill'] <= o['tc'] else (o['why'] or 'end'); dt = e_ - tb
            elif s0 == 'OUT':
                prev = [o for o in O4 if o['ts'] < tb]
                if prev:
                    o = prev[-1]; e_ = min(o['tc'], o['tfill'], o['tr'] if o['ref'] else INF)
                    rsn = 'refused' if o['ref'] else ('filled' if o['tfill'] <= o['tc'] else (o['why'] or 'end')); dt = tb - e_
                else:
                    rsn = 'none yet'
            why.append((int(day), kev, s0, rsn, dt, gon))
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['ctrl_ldg'] or st['ctrl_fast'] or not st['md_differs'] or st['sim4_differ'] or st['unmapped'] or not st['clock_ok']:
        sys.exit(f'MOVE-DOWN CONTROL FAILED on {day} -> no output: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)} }')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'src', 'young', 'pend', 'q', 'v1', 'v5', 'type', 'age']).to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(lev, columns=['day', 'kind', 'arm', 'q', 'v1', 'gap', 'type', 'aged', 'a0']).to_parquet(f'{OUTD}/levfills_{day}.parquet', index=False)
    pd.DataFrame(pres, columns=['day', 'kind', 'arm', 'present', 'gap', 'a0', 'guard_on', 'clear']).to_parquet(f'{OUTD}/presence_{day}.parquet', index=False)
    pd.DataFrame(why, columns=['day', 'kind', 'state', 'reason', 'dt', 'guard_on']).to_parquet(f'{OUTD}/whygone_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}; dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    rd = lambda nm: pd.concat([pd.read_parquet(f'{OUTD}/{nm}_{d}.parquet') for d in DAYS9], ignore_index=True)
    A, LV, PS, WG = rd('fills'), rd('levfills'), rd('presence'), rd('whygone')
    nb = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]; nl = int((~np.isfinite(LV.v1)).sum()); LV = LV[np.isfinite(LV.v1)]
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if len(x) and x.q.sum() else NAN
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    def cell(arm, cfg, srcs=('WS',), d=None, h='v1', aged=True):
        x = A[(A.arm == arm) & (A.cfg == cfg) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    R = {}
    L = [f'MOVE-DOWN + WHY-GONE, 9 consumed replay days, first 60 s; as of {asof}; controls passed (MD with entries off = LDG and = FAST lane,'
         f' sim4 = simulator, orders mapped, clocks restored, planted errors caught); statuses: fills without a 1 s value {nb}, level fills {nl}', '',
         'PART 1a: our rung-0 order at the protection side\'s new best (M 50): kind | arm | events | present | go-live after the birth, median (p25 / p75)'
         ' | <= 100 ms | <= 250 ms | queue ahead at go-live, median sh']
    for kev in ('adverse', 'calm fallback'):
        for arm in ARMS:
            x = PS[(PS.kind == kev) & (PS.arm == arm)]; y = x[x.present]; g = y.gap.values
            L.append(f'  {kev:<13} {arm:<4} {len(x):>6} | {x.present.mean():5.1%} | {np.median(g) if len(g) else NAN:6.0f} ({np.percentile(g, 25) if len(g) else NAN:5.0f} /'
                     f' {np.percentile(g, 75) if len(g) else NAN:5.0f}) | {(g <= 100).mean() if len(g) else NAN:5.1%} | {(g <= 250).mean() if len(g) else NAN:5.1%} |'
                     f' {np.nanmedian(y.a0) if len(y) else NAN:5.0f}')
        x = PS[(PS.kind == kev) & (PS.arm == 'LDG')]
        L.append(f'  {kev:<13} LDG delay parts: guard on at the birth {x.guard_on.mean():.0%}; birth to the guard clearing, median {np.nanmedian(x.clear[x.guard_on]):.0f} ms;'
                 f' then the 250 ms wait and the 40 ms order path')
    R['R520'] = {}
    for arm in ('MD', 'MDF', 'FAST'):
        lo_ = hi = 0
        for d in DAYS9:
            a_ = PS[(PS.kind == 'adverse') & (PS.arm == arm) & PS.present & (PS.day == int(d))]; b_ = PS[(PS.kind == 'adverse') & (PS.arm == 'LDG') & PS.present & (PS.day == int(d))]
            if len(a_) >= MIN_N and len(b_) >= MIN_N:
                lo_ += int(np.median(a_.gap) < np.median(b_.gap)); hi += int(np.median(a_.gap) > np.median(b_.gap))
        lab = 'FASTER' if lo_ >= 7 else ('SLOWER' if hi >= 7 else 'NOT-SHOWN')
        R['R520'][arm] = dict(label=lab, lower=lo_, higher=hi)
        L.append(f'  R520 {arm}: go-live lower than LDG on {lo_}, higher on {hi} of 9 -> {lab}')
    L += ['', 'PART 1b: our fills at the new best (M 50, WS): kind | arm | fills | shares | aged c/sh | STOP | our order live after the birth, median |'
          ' queue ahead at go-live, median (top 5 there: +0.30 c/sh, 82% STOP)']
    for kev in ('adverse', 'calm fallback'):
        for arm in ARMS:
            y = LV[(LV.kind == kev) & (LV.arm == arm)]; ya = y[y.aged]
            L.append(f'  {kev:<13} {arm:<4} {len(y):>6} | {y.q.sum():>7.0f} | {cps(ya):+.3f} | {ya.q[ya.type == "STOP"].sum() / max(ya.q.sum(), 1e-9):4.0%} |'
                     f' {np.median(y.gap) if len(y) else NAN:6.0f} | {np.nanmedian(y.a0) if len(y) else NAN:5.0f}')
    R['R522'] = {}
    adv = LV[(LV.kind == 'adverse') & LV.aged]
    for arm in ('MD', 'MDP', 'MDF'):
        up = dn = 0
        for d in DAYS9:
            a_ = adv[(adv.day == int(d)) & (adv.arm == arm)]; b_ = adv[(adv.day == int(d)) & (adv.arm == 'LDG')]
            if len(a_) >= MIN_N and len(b_) >= MIN_N:
                up += int(cps(a_) > cps(b_)); dn += int(cps(a_) < cps(b_))
        lab = 'HIGHER' if up >= 7 else ('LOWER' if dn >= 7 else 'NOT-SHOWN')
        R['R522'][arm] = dict(label=lab, up=up, down=dn, arm=cps(adv[adv.arm == arm]), ldg=cps(adv[adv.arm == 'LDG']))
        L.append(f'  R522 {arm}: {cps(adv[adv.arm == arm]):+.3f} vs LDG {cps(adv[adv.arm == "LDG"]):+.3f} c/sh (higher {up}, lower {dn} of 9) -> {lab}')
    L += ['', 'PART 1c: rung-0 $/day at 1 s, w/o young: WS | E2 | all trades || WS all ages | pulls a day | WS c/sh | cancel-pending share of WS shares']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            pl = sum(stats[d]['stat'].get(f'pulls|{arm}|{cfg}', 0) for d in DAYS9) / ndays
            x = A[(A.arm == arm) & (A.cfg == cfg) & (A.src == 'WS') & ~A.young]
            L.append(f'  {cfg:<4} {arm:<4} {cell(arm, cfg):+8.1f} | {cell(arm, cfg, ("E2",)):+8.1f} | {cell(arm, cfg, ("WS", "E2")):+8.1f} || {cell(arm, cfg, aged=False):+8.1f} |'
                     f' {pl:7.0f} | {cps(x):+.3f} | {x.q[x.pend].sum() / max(x.q.sum(), 1e-9):4.0%}')
        L.append('')
    R['R521'] = {}
    for arm in ('MD', 'MDP', 'MDF', 'FAST'):
        up = {c: sum(cell(arm, c, d=d) > cell('LDG', c, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
        dn = {c: sum(cell(arm, c, d=d) < cell('LDG', c, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
        diff = {c: cell(arm, c) - cell('LDG', c) for c in ('M0', 'M50', 'M250')}
        lab = 'HELPS' if min(up['M0'], up['M50']) >= 7 else ('HURTS' if min(dn['M0'], dn['M50']) >= 7 else 'NOT-SHOWN')
        alld = cell(arm, 'M50', ('WS', 'E2')) - cell('LDG', 'M50', ('WS', 'E2'))
        R['R521'][arm] = dict(label=lab, up=up, down=dn, diff=diff, all_trades_M50=alld, sign_holds_M250=bool((diff['M250'] > 0) == (diff['M50'] > 0)))
        L.append(f'  R521 {arm}: {lab} (WS $/day vs LDG: M 0 {diff["M0"]:+.1f} ({up["M0"]}/9), M 50 {diff["M50"]:+.1f} ({up["M50"]}/9), M 250 {diff["M250"]:+.1f}'
                 f' ({up["M250"]}/9, sign {"holds" if R["R521"][arm]["sign_holds_M250"] else "FLIPS"}); all trades M 50 {alld:+.1f})')
    L += ['', 'C1 / C12: rung-0 $/day by order age, M 50: WS (< 100 | 100-500 | >= 500 ms) || E2']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50')]
        parts = [' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+7.1f}' for _, lo, hi in AGEB) for y in (x[x.src == 'WS'], x[x.src == 'E2'])]
        L.append(f'  {arm:<4} {parts[0]} || {parts[1]}')
    L += ['', 'PART 2 WHY-GONE (other side: bids in a pump, asks in a dump; LDG rung 0, M 50): kind | events | state at the birth AT-NEW / HELD-OLD /'
          ' HELD-OTHER / PEND / OUT | guard on that side at the birth']
    for kev in ('other', 'calm improve'):
        x = WG[WG.kind == kev]
        L.append(f'  {kev:<13} {len(x):>6} | ' + ' / '.join(f'{(x.state == s_).mean():4.0%}' for s_ in ('AT-NEW', 'HELD-OLD', 'HELD-OTHER', 'PEND', 'OUT')) +
                 f' | {x.guard_on.mean():4.0%}')
        h = x[x.state == 'HELD-OLD']
        L.append(f'    HELD-OLD ({len(h)}): within 1 s ' + ', '.join(f'{r_} {(h.reason == r_).mean():.0%}' for r_ in h.reason.value_counts().index) +
                 f'; ended after the birth, median {np.nanmedian(h.dt) if np.isfinite(h.dt).any() else NAN:.0f} ms')
        o = x[x.state == 'OUT']
        L.append(f'    OUT ({len(o)}): the last order ' + ', '.join(f'{r_} {(o.reason == r_).mean():.0%}' for r_ in o.reason.value_counts().index) +
                 f'; ended before the birth, median {np.nanmedian(o.dt) if np.isfinite(o.dt).any() else NAN:.0f} ms; guard on at the birth {o.guard_on.mean():.0%}')
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/move_down.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/move_down.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
