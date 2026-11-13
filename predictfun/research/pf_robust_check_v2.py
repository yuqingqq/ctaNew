"""ROBUST-CHECK v2 (2026-11-13, session ce68ebe0): corrections after the independent code review of pf_side_moves.py / pf_move_down.py,
which came back after ROBUST-CHECK v1 (pf_robust_check.py, 1ae29bdb; post hoc aaa04b61). v1 stays as provenance; v2 supersedes the
numbers it re-measures. Same days, events, groups and legs as v1. Changes:
  E  events whose birth frame is invalid (QR.runs labels a run start by price alone), or whose new price equals the old one or is empty,
     are excluded (status, counted by kind). They entered PULL-DEPTH's event list too, so control (i) checks the full list first.
  K1 strict sweep, K2 proof with margins, K3 front by count: as v1 on the remaining events.
  K3b first trade: every hashed order filled by the first taker order trading at the new price after the birth (removes the row order of
     legs within one trade) - top-5 share of those orders against their share of all filled orders (count).
  K4 requote when visible: makers proven at the old best in [tb - 2 s, tb] (now including a fill stamped at tb, the first frame showing it).
  K5 kept PER ORDER (v1 counted a maker as kept if any of its proven orders filled again: more orders, more chances): every proven old
     order is one row; only events with tb <= window start + 51 s, so the 10 s window lies inside the loaded legs (others: status).
  K6 value of kept fills, per order.
  H  hash coverage by group (share of WS first-minute legs with a certain order hash), printed.
  K7 gated fast entry, corrected in two ways, and compared with the SAME path (v1's declared rule compared every arm with today's lane):
     - after the lane's own drift / cross kill of an early-entry order, the lane placed again at once (the simulator's wait off, the
       callback not consulted for the entry), skipping the move test; a decision wrapper now pulls such an order through the callback when
       the entry line is blocked, so no placement happens that step;
     - "moved" now compares the known best with the known best 100 ms before the guard came on (a fallback already known at the onset
       counted as no move, so the arm behaved as today's lane there).
     Arms: MD2 / MDF2 (ungated, today's / fast path), MDG10v2 / MDG25v2 / MDG10Fv2 / MDG25Fv2 (enter at once only if P(move >= 1 more) < tau,
     pulled while the guard is on if P >= tau). Each against the same path's plain lane (LDG rung 0 or FAST rung 0).
  K8 (fast ladder) is unaffected by the review's points (no events, no early entry): v1's result stands, not re-run.
Pass criteria (declared before the first v2 run; day rule >= 7 of 9, < 30 units in a compared cell do not count):
  K1, K2, K3, K4, K6 exactly as v1. K3b: ratio > 1 pooled and on >= 7 days -> PASS. K5 (per order): top 5 > field at >= 300 sh traded on
  >= 7 days -> PASS. K7: each arm against its same-path lane, WS $/day w/o young at 1 s: HELPS if higher at M 0 and M 50 on >= 7 days;
  HURTS if lower at both on >= 7; else NOT-SHOWN; C4 ROBUST if no gated arm HELPS.
Controls (any failure -> no output): (i) the full event list equals PULL-DEPTH's before the exclusion; (ii) every corrected arm with entry
off equals its plain lane (LDG rung 0 / FAST rung 0) fill for fill and order for order on every window-side and clock; (iii) synthetic
cases for the first-trade orders, the per-order kept rule, the pre-onset move test and the decision wrapper, with three planted errors (the
first trade taken as the first leg only, the move test using the onset's best, the wrapper switched off) that must each fail a case, plus
v1's and MOVE-DOWN's self-tests; clocks and the cooldown switch restored; (iv) memory self-guard.
Limits as v1 (fills are lower bounds; price-time priority assumed; replay rule M for ours; consumed days; PULL-DEPTH's P nested-fold leak).
Usage (run_light, PF_SLOT=claude): pf_robust_check_v2.py selftest | day <day> | report; driver run_robust_check_v2.sh
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
import pf_qr_queue_race as QR                             # noqa: E402
import pf_timing_split as TS                              # noqa: E402
import pf_best_gap as BG                                  # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_move_down as PMD                                # noqa: E402
import pf_robust_check as RC                              # noqa: E402  (v1: sweep_vol, proof_orders, gate_line)

DAYS9 = K.DAYS9
OUTD = f'{RC.OUTD}/v2'
CFGS = RC.CFGS; SIZE = RC.SIZE; MIN_N = RC.MIN_N
OLD_MS, KEEP_MS, LEVEL_MS = RC.OLD_MS, RC.KEEP_MS, RC.LEVEL_MS
PRE_MS = 100.0; KEEP_LAST = 51000.0
ARMS2 = {'MD2': (None, False), 'MDF2': (None, True), 'MDG10v2': (0.10, False), 'MDG25v2': (0.25, False),
         'MDG10Fv2': (0.10, True), 'MDG25Fv2': (0.25, True)}
INF = float('inf'); NAN = float('nan')
MUT_FIRST_LEG = MUT_PRE_IGNORED = MUT_WRAP_OFF = 0          # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def first_trade_orders(L, p_new, tb, tend):
    """Hashed orders filled at p_new in (tb, tend]: (group, filled by the first taker order trading there after tb)."""
    t = L['t']; idx = [i for i in range(bisect.bisect_right(t, tb), bisect.bisect_right(t, tend)) if L['p'][i] == p_new]
    if not idx:
        return []
    tk0 = L['tk'][idx[0]]
    first = {idx[0]} if (MUT_FIRST_LEG or tk0 < 0) else {i for i in idx if L['tk'][i] == tk0}
    fo = {L['oh'][i] for i in first if L['oh'][i]}
    allo = {}
    for i in idx:
        if L['oh'][i]:
            allo.setdefault(L['oh'][i], L['grp'][i])
    return [(g_, h in fo) for h, g_ in allo.items()]


def old_orders(L, p_new, p_old, tb, tend):
    """Proof at p_old in [tb - 2 s, tb]. Returns makers (group, JOIN, shares traded at p_new while it was the best) and orders (group,
    pre-move shares, shares traded at p_old in (tb, tb + 10 s], KEPT = this order fills again there, kept shares and value 1 s / 5 s)."""
    t = L['t']; i_tb = bisect.bisect_right(t, tb); i_end = bisect.bisect_right(t, tb + KEEP_MS); i_new = bisect.bisect_right(t, tend)
    od = {}
    for i in range(bisect.bisect_left(t, tb - OLD_MS), i_tb):
        if L['p'][i] == p_old and L['oh'][i]:
            o = od.setdefault(L['oh'][i], dict(sig=L['sig'][i], grp=L['grp'][i], pre_q=0.0)); o['pre_q'] += L['q'][i]
    obs_q = float(sum(L['q'][i] for i in range(i_tb, i_end) if L['p'][i] == p_old))
    new_q = float(sum(L['q'][i] for i in range(i_tb, i_new) if L['p'][i] == p_new))
    joined = {L['sig'][i] for i in range(i_tb, i_new) if L['p'][i] == p_new}
    mk = {}
    for o in od.values():
        mk.setdefault(o['sig'], o['grp'])
    makers = [dict(grp=g_, join=s_ in joined, new_q=new_q) for s_, g_ in mk.items()]
    orders = []
    for h, o in od.items():
        kix = [i for i in range(i_tb, i_end) if L['p'][i] == p_old and L['oh'][i] == h]
        orders.append(dict(grp=o['grp'], pre_q=o['pre_q'], obs_q=obs_q, kept=bool(kix), kept_q=float(sum(L['q'][i] for i in kix)),
                           kept_v1=float(sum(L['q'][i] * L['v1'][i] for i in kix)), kept_v5=float(sum(L['q'][i] * L['v5'][i] for i in kix))))
    return makers, orders


def moved_pre(vg, best, side, tg, pre_ms=PRE_MS):
    """Guard on and the known best worse than the known best pre_ms before the guard came on."""
    out = []; bon = -1; lastv = -1; lastv_at = []
    for k in range(len(vg)):
        if best[k] >= 0:
            lastv = best[k]
        lastv_at.append(lastv)
        if not vg[k]:
            bon = -1; out.append(False); continue
        if k == 0 or not vg[k - 1]:
            kp = bisect.bisect_right(tg, tg[k] - pre_ms) - 1
            bon = lastv if (MUT_PRE_IGNORED or kp < 0) else lastv_at[kp]
        b = best[k]
        out.append(bool(bon >= 0 and b >= 0 and ((b < bon) if side == 0 else (b > bon))))
    return out


def wrap_dec(base, blocked, best, bopp, side):
    """Pull through the decision when the lane would kill the order by drift / cross while entry is blocked (no re-entry that step)."""
    def dec(k, t, cur):
        r = base(k, t, cur)
        if r or not blocked[k] or MUT_WRAP_OFF:
            return r
        b = best[k]
        if b < 0:
            return False
        beh = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
        cross = not ((cur['pt'] < bopp[k]) if side == 0 else (cur['pt'] > bopp[k]))
        return bool(beh > K.HOLD_K or cross)
    return dec


def selftest():
    bad = []
    rows = [(1050, 51, 5, 4, 'field', 'b', 8), (1050, 51, 3, 1, 'top 5', 'a', 8), (1100, 51, 2, 3, 'field', 'c', 9)]
    L = {k: [r[i] for r in rows] for i, k in enumerate(('t', 'p', 'q', 'sig', 'grp', 'oh', 'tk'))}
    ft = sorted(first_trade_orders(L, 51, 1000.0, 2000.0))
    if ft != [('field', False), ('field', True), ('top 5', True)]:
        bad.append(f'first trade {ft}')
    rows2 = [(500, 49, 8, 1, 'top 5', 'h1', 1), (600, 49, 2, 1, 'top 5', 'h5', 6), (1000, 49, 3, 2, 'field', 'h2', 2),
             (1100, 50, 5, 1, 'top 5', 'h3', 3), (4000, 49, 6, 1, 'top 5', 'h1', 4), (4100, 49, 6, 2, 'field', 'h4', 5)]
    L2 = {k: [r[i] for r in rows2] for i, k in enumerate(('t', 'p', 'q', 'sig', 'grp', 'oh', 'tk'))}; L2['v1'] = [1.0] * 6; L2['v5'] = [2.0] * 6
    mk, od = old_orders(L2, 50, 49, 1000.0, 2000.0)
    if sorted((m['grp'], m['join']) for m in mk) != [('field', False), ('top 5', True)]:
        bad.append(f'makers {mk}')
    if sorted((o['grp'], o['pre_q'], o['kept']) for o in od) != [('field', 3.0, False), ('top 5', 2.0, False), ('top 5', 8.0, True)]:
        bad.append(f'orders {od}')
    tg = [0.0, 50.0, 100.0, 150.0, 200.0]; vg = [False, False, True, True, False]
    if moved_pre(vg, [50, 49, 49, 49, 49], 0, tg) != [False, False, True, True, False]:      # fallback known before the onset
        bad.append(f'pre-onset move {moved_pre(vg, [50, 49, 49, 49, 49], 0, tg)}')
    base = lambda k, t, cur: False
    d = wrap_dec(base, [True, False], [55, 55], [60, 60], 0)
    if d(0, 0.0, dict(pt=50)) is not True or d(1, 50.0, dict(pt=50)) is not False or d(0, 0.0, dict(pt=54)) is not False:
        bad.append('wrapper')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_FIRST_LEG', 'MUT_PRE_IGNORED', 'MUT_WRAP_OFF'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    RC.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ROBUST-CHECK v2 SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


def run_arm2(w, q, shf, vg, P1, tau, fast, entry_off=False):
    """MD (tau None) or a gated arm, corrected (pre-onset move test, decision wrapper)."""
    ctx = BG.fast_context(w, 'FAST') if fast else w
    blk = PMD.entry_line(vg, w['tg']); ep = PMD.episodes(vg)
    mv = [False] * len(vg) if entry_off else moved_pre(vg, ctx['best'], w['side'], w['tg'])
    aux = [b_ and not m_ for b_, m_ in zip(blk, mv)] if tau is None else RC.gate_line(blk, mv, P1, tau)
    base = PMD.md_dec(vg, w['tg'], ep, P1=None if tau is None else P1, tau=tau)
    dec = wrap_dec(base, aux, ctx['best'], ctx['bopp'], w['side'])
    try:
        K.MUT_REST_NOCOOL = 1
        if fast:
            K.L_PLACE = 5.0; K.L_CANCEL = 5.0; K.L_PENDCANCEL = 5.0
        return K.sim_arm(ctx, q, shf, 'rest', aux=aux, decide=dec, idle_only=1, size=SIZE)
    finally:
        K.MUT_REST_NOCOOL = 0
        K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL = BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL']


def day_run(day):
    import pf_gap_top_makers as GP
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WS = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    TK = pd.Series(np.where(tkf[ix] >= 0, tkf[ix], -1), index=LT.index)
    SIG = pd.Series(full.signer.values[ix].astype('int64'), index=LT.index)
    MP = pd.read_parquet(f'{PD.ORD}/map_{day}.parquet').set_index('row')
    OH = pd.Series(np.where(MP.hash_ok.reindex(LT.index).eq(True).values, MP.ohash.reindex(LT.index).fillna('').values, ''), index=LT.index)
    gname = lambda s_: 'top 5' if G.get(int(s_), 'rest') == 'top 5' else 'field'
    PE = pd.read_parquet(f'{PD.OUTD}/events_{day}.parquet', columns=['eid', 'kind', 'side', 'p', 'tb'])
    PR = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet', columns=['m', 'side', 'k', 'P1']); PRI = PR.groupby(['m', 'side']).indices
    st.update(window_sides=0, ctrl_off=0, arms_differ=0, clock_ok=1, real_atbest_q={}, hash_legs={}, excluded={}, keep_truncated=0)
    evs = []; swr = []; orr = []; ftr = []; mkr = []; odr = []; fills = []; pend = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        P1 = np.full(n, np.nan)
        if (m, side) in PRI:
            x_ = PR.iloc[PRI[(m, side)]]; P1[x_.k.values] = x_.P1.values
        runs = QR.runs(tf, P5, opp, side)
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        anc = w['anc']; ptl = w['ptl']; shl = w['shl']
        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)
        Lw = dict(t=[], p=[], q=[], sig=[], grp=[], oh=[], tk=[], v1=[], v5=[])
        for i, r_ in enumerate(lrow):
            if not src[i]:
                continue
            t = float(anc[i]); s_ = int(SIG.at[r_]); pt = int(ptl[i]); g_ = gname(s_); oh = OH.at[r_]
            Lw['t'].append(t); Lw['p'].append(pt); Lw['q'].append(float(shl[i])); Lw['sig'].append(s_); Lw['grp'].append(g_)
            Lw['oh'].append(oh); Lw['tk'].append(int(TK.at[r_])); Lw['v1'].append(val(t, 1000.0, pt)); Lw['v5'].append(val(t, 5000.0, pt))
            if w['ws'] <= t < w['ws'] + 60000.0:
                hl = st['hash_legs'].setdefault(g_, [0, 0]); hl[0] += 1; hl[1] += int(bool(oh))
                if GP.depth_of(tfl, P5, opp, side, t, pt) == 0:
                    st['real_atbest_q'][g_] = st['real_atbest_q'].get(g_, 0.0) + float(shl[i])
        for q, shf in CFGS:
            cfg = f'{q}{int(shf)}'
            res = {'LDG0': K.sim_arm(w, q, shf, 'rest', aux=vg, size=SIZE), 'FAST0': BG.run_lane(w, q, shf, vg, SIZE, 'FAST')}
            for fast, ref in ((False, 'LDG0'), (True, 'FAST0')):
                for tau in (None, 0.10):
                    c_ = run_arm2(w, q, shf, vg, P1, tau, fast, entry_off=True)
                    st['ctrl_off'] += int(c_[0] != res[ref][0] or c_[1] != res[ref][1])
            for a_, (tau, fast) in ARMS2.items():
                res[a_] = run_arm2(w, q, shf, vg, P1, tau, fast)
                st['arms_differ'] += int(res[a_][0] != res['FAST0' if fast else 'LDG0'][0])
            if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL, K.MUT_REST_NOCOOL) != (BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL'], 0):
                st['clock_ok'] = 0
            for a_, (F, O, pulls) in res.items():
                for f in F:
                    i, pt, qq, age = f[0], int(f[1]), f[2], f[4]
                    t = float(anc[i])
                    if w['ws'] <= t < w['ws'] + 61000.0:
                        fills.append((int(day), cfg, a_, 'WS' if src[i] else 'E2', age < 100.0, float(qq), val(t, 1000.0, pt)))
        pend.setdefault(m, {})[side] = dict(w=w, vg=vg, runs=runs, tfl=tfl, L=Lw)
        if len(pend[m]) == 2:
            pair = pend.pop(m)
            ons = {s_: [float(pair[s_]['w']['tg'][k]) for k in PD.onsets(pair[s_]['vg'])] for s_ in (0, 1)}
            allon = sorted(ons[0] + ons[1]); used = set()

            def starts(s_, kd):
                b0_, valid_, rs_, kind_ = pair[s_]['runs']; tl = pair[s_]['tfl']; out = []
                for j in range(1, len(tl)):
                    if rs_[j] == j and kind_[j] == kd:
                        jn = j + 1
                        while jn < len(tl) and rs_[jn] == j:
                            jn += 1
                        out.append((j, int(b0_[j]), tl[j], tl[jn] if jn < len(tl) else INF, int(b0_[j - 1])))
                return out
            ST = {(s_, kd): starts(s_, kd) for s_ in (0, 1) for kd in (1, 2)}
            wsm = pair[0]['w']['ws']; E = []
            for s_ in (0, 1):
                for t0 in ons[s_]:
                    if not (wsm <= t0 < wsm + 60000.0):
                        continue
                    for x, kd, kev in ((s_, 2, 'adverse'), (1 - s_, 1, 'other')):
                        c = [e for e in ST[(x, kd)] if t0 - RC.WIN_PRE < e[2] <= t0 + RC.WIN_POST]
                        if not c or (x, c[0][0]) in used:
                            continue
                        j, p, tb, te_run, po = c[0]; used.add((x, j)); E.append((f'{day}|{m}|{x}|{j}', kev, x, j, p, tb, te_run, po))
            for x in (0, 1):
                for kd, kev in ((2, 'calm fallback'), (1, 'calm improve')):
                    for j, p, tb, te_run, po in ST[(x, kd)]:
                        if (x, j) in used or not (wsm <= tb < wsm + 60000.0):
                            continue
                        lo = bisect.bisect_left(allon, tb - RC.WIN_POST)
                        if lo < len(allon) and allon[lo] < tb + RC.WIN_PRE:
                            continue
                        E.append((f'{day}|{m}|{x}|{j}', kev, x, j, p, tb, te_run, po))
            for eid, kev, x, j, p, tb, te_run, po in E:
                evs.append((eid, kev, x, p, tb))
                why = ('invalid birth frame' if not pair[x]['runs'][1][j] else ('no new price' if p < 0 else ('same price' if p == po else '')))
                if why:
                    ex = st['excluded'].setdefault(kev, {}); ex[why] = ex.get(why, 0) + 1
                    continue
                tend = min(te_run, tb + LEVEL_MS); Lx = pair[x]['L']
                dr = 'UP' if ((kev in ('adverse', 'calm fallback') and x == 1) or (kev in ('other', 'calm improve') and x == 0)) else 'DOWN'
                if kev in ('adverse', 'calm fallback'):
                    for (g_, wh), v_ in RC.sweep_vol(Lx, p, po, tb).items():
                        swr.append((int(day), kev, dr, g_, wh, v_))
                    orr += [(int(day), kev, dr, o['grp'], o['rank'], o['q']) + tuple(o['proof']) for o in RC.proof_orders(Lx, p, tb, tend)]
                    ftr += [(int(day), kev, dr, g_, f_) for g_, f_ in first_trade_orders(Lx, p, tb, tend)]
                else:
                    if tb > wsm + KEEP_LAST:
                        st['keep_truncated'] += 1; continue
                    mk, od = old_orders(Lx, p, po, tb, tend)
                    mkr += [(int(day), kev, dr, z['grp'], z['join'], z['new_q']) for z in mk]
                    odr += [(int(day), kev, dr, z['grp'], z['pre_q'], z['obs_q'], z['kept'], z['kept_q'], z['kept_v1'], z['kept_v5']) for z in od]
            for s_ in (0, 1):
                pair[s_]['w'].pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    EV = pd.DataFrame(evs, columns=['eid', 'kind', 'side', 'p', 'tb'])
    st['events_differ'] = int(len(EV) != len(PE) or not (EV.eid.values.astype(str) == PE.eid.values.astype(str)).all()
                              or not (EV.p.values == PE.p.values).all() or not (EV.tb.values == PE.tb.values).all())
    st['unpaired_windows'] = len(pend)
    if st['events_differ'] or st['ctrl_off'] or not st['clock_ok'] or not st['arms_differ'] or st['unpaired_windows']:
        sys.exit(f'ROBUST-CHECK v2 CONTROL FAILED on {day} -> no output: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)} }')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    W = lambda rows, cols, nm: pd.DataFrame(rows, columns=cols).to_parquet(f'{OUTD}/{nm}_{day}.parquet', index=False)
    W(fills, ['day', 'cfg', 'arm', 'src', 'young', 'q', 'v1'], 'fills')
    W(swr, ['day', 'kind', 'dir', 'grp', 'which', 'q'], 'sweeps')
    W(orr, ['day', 'kind', 'dir', 'grp', 'rank', 'q', 'p0', 'p50', 'p250'], 'orders')
    W(ftr, ['day', 'kind', 'dir', 'grp', 'first'], 'firsttrade')
    W(mkr, ['day', 'kind', 'dir', 'grp', 'join', 'new_q'], 'makers')
    W(odr, ['day', 'kind', 'dir', 'grp', 'pre_q', 'obs_q', 'kept', 'kept_q', 'kept_v1', 'kept_v5'], 'oldorders')
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} excluded {st["excluded"]} in {time.time() - T0:.0f} s', flush=True)


def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}; dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    rd = lambda nm: pd.concat([pd.read_parquet(f'{OUTD}/{nm}_{d}.parquet') for d in DAYS9], ignore_index=True)
    SW, OR, FT, MK, OD, A = rd('sweeps'), rd('orders'), rd('firsttrade'), rd('makers'), rd('oldorders'), rd('fills')
    nb = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    ab = {d: stats[d]['stat']['real_atbest_q'] for d in DAYS9}
    sh_ab = sum(ab[d].get('top 5', 0.0) for d in DAYS9) / max(sum(sum(ab[d].values()) for d in DAYS9), 1e-9)
    shd = {d: ab[d].get('top 5', 0.0) / max(sum(ab[d].values()), 1e-9) for d in DAYS9}
    exc = {}
    for d in DAYS9:
        for k_, v_ in stats[d]['stat']['excluded'].items():
            for w_, c_ in v_.items():
                exc.setdefault(k_, {}); exc[k_][w_] = exc[k_].get(w_, 0) + c_
    hl = {g_: [sum(stats[d]['stat']['hash_legs'].get(g_, [0, 0])[i] for d in DAYS9) for i in (0, 1)] for g_ in ('top 5', 'field')}
    V = {}
    L = [f'ROBUST-CHECK v2 (after the code review), 9 consumed replay days, first 60 s; as of {asof}; controls passed (events = PULL-DEPTH before'
         f' the exclusion, corrected arms with entry off = their plain lanes, clocks restored, planted errors caught); fills without a 1 s value {nb}',
         f'statuses: events excluded {json.dumps(exc)}; maker events with a truncated 10 s window {sum(stats[d]["stat"]["keep_truncated"] for d in DAYS9)};'
         f' hash coverage of WS first-minute legs: top 5 {hl["top 5"][1] / max(hl["top 5"][0], 1):.1%}, field {hl["field"][1] / max(hl["field"][0], 1):.1%};'
         f' top-5 share of all real at-best WS fill shares {sh_ab:.1%}', '']
    L.append('K1 STRICT SWEEP: top-5 share of the volume at the new price | at the old best | shares')
    k1 = {}
    for kev in ('adverse', 'calm fallback'):
        for dr in ('UP', 'DOWN'):
            x = SW[(SW.kind == kev) & (SW.dir == dr)]; nw = x[x.which == 'new']; od = x[x.which == 'old']
            k1[(kev, dr)] = nw.q[nw.grp == 'top 5'].sum() / max(nw.q.sum(), 1e-9)
            L.append(f'  {kev:<13} {dr:<4} {k1[(kev, dr)]:5.1%} | {od.q[od.grp == "top 5"].sum() / max(od.q.sum(), 1e-9):5.1%} | {nw.q.sum():.0f}')
    x = SW[(SW.kind == 'adverse') & (SW.which == 'new')]; up_d = 0
    for d in DAYS9:
        y = x[x.day == int(d)]
        if y.q.sum() >= MIN_N:
            up_d += int(y.q[y.grp == 'top 5'].sum() / y.q.sum() > shd[d])
    V['K1'] = 'PASS' if (k1[('adverse', 'UP')] >= 1.2 * sh_ab and k1[('adverse', 'DOWN')] >= 1.2 * sh_ab and up_d >= 7) else 'FAIL'
    L.append(f'  K1: UP {k1[("adverse", "UP")]:.1%}, DOWN {k1[("adverse", "DOWN")]:.1%} vs {1.2 * sh_ab:.1%}; above the day\'s share on {up_d} of 9 -> {V["K1"]}')
    L += ['', 'K2 PROOF WITH A MARGIN: group\'s shares at the new best from orders filled there at t <= birth - 0 / 50 / 250 ms']
    pk = {}
    for dr in ('UP', 'DOWN'):
        x = OR[(OR.kind == 'adverse') & (OR.dir == dr)]
        for g_ in ('top 5', 'field'):
            y = x[x.grp == g_]
            L.append(f'  adverse {dr:<4} {g_:<6} ' + ' / '.join(f'{y.q[y[c]].sum() / max(y.q.sum(), 1e-9):5.1%}' for c in ('p0', 'p50', 'p250')) + f' ({len(y)} orders)')
        hi = 0
        for d in DAYS9:
            y = x[x.day == int(d)]; a_ = y[y.grp == 'top 5']; b_ = y[y.grp == 'field']
            if len(a_) >= MIN_N and len(b_) >= MIN_N:
                hi += int(a_.q[a_.p250].sum() / max(a_.q.sum(), 1e-9) > b_.q[b_.p250].sum() / max(b_.q.sum(), 1e-9))
        pk[dr] = hi
    V['K2'] = 'PASS' if min(pk.values()) >= 7 else 'FAIL'
    L.append(f'  K2: top 5 > field at 250 ms on UP {pk["UP"]} / DOWN {pk["DOWN"]} of 9 -> {V["K2"]}')
    L += ['', 'K3 / K3b FRONT: top-5 share of first-filled orders (K3) / of orders filled by the first trade (K3b) | of all filled orders | ratio']
    for nm, X, col in (('K3', OR, None), ('K3b', FT, 'first')):
        x = X[X.kind == 'adverse']
        f1 = (x[x['rank'] == 1].grp == 'top 5').mean() if col is None else (x[x[col]].grp == 'top 5').mean()
        fa = (x.grp == 'top 5').mean(); hi = 0
        for d in DAYS9:
            y = x[x.day == int(d)]; yf = y[y['rank'] == 1] if col is None else y[y[col]]
            if len(y) >= MIN_N and len(yf) >= MIN_N:
                hi += int((yf.grp == 'top 5').mean() > (y.grp == 'top 5').mean())
        V[nm] = 'PASS' if f1 / max(fa, 1e-9) > 1 and hi >= 7 else 'FAIL'
        L.append(f'  {nm}: {f1:5.1%} | {fa:5.1%} | {f1 / max(fa, 1e-9):.2f}; higher on {hi} of 9 -> {V[nm]}')
    L += ['', 'K4 REQUOTE WHEN VISIBLE (other side): JOIN rate of makers proven at the old best, all | >= 100 sh traded at the new price']
    k4 = {}
    for dr in ('UP', 'DOWN'):
        x = MK[(MK.kind == 'other') & (MK.dir == dr)]
        for g_ in ('top 5', 'field'):
            y = x[x.grp == g_]; h = y[y.new_q >= 100]; k4[(dr, g_)] = float(h['join'].mean()) if len(h) else NAN
            L.append(f'  other {dr:<4} {g_:<6} {y["join"].mean():5.1%} ({len(y)}) | {k4[(dr, g_)]:5.1%} ({len(h)})')
    V['K4'] = 'PASS' if k4[('UP', 'top 5')] <= 1 / 3 and k4[('DOWN', 'top 5')] <= 1 / 3 else 'FAIL'
    L.append(f'  K4: top 5 UP {k4[("UP", "top 5")]:.1%}, DOWN {k4[("DOWN", "top 5")]:.1%} (line 33%) -> {V["K4"]}; field UP {k4[("UP", "field")]:.1%},'
             f' DOWN {k4[("DOWN", "field")]:.1%}')
    L += ['', 'K5 / K6 KEPT PER ORDER (other side; 10 s window complete): KEPT | >= 100 / >= 300 sh traded at the old price || at >= 300 sh:'
          ' pre-move fills < 10 / >= 10 sh || kept fills c/sh 1 s / 5 s']
    for dr in ('UP', 'DOWN'):
        x = OD[(OD.kind == 'other') & (OD.dir == dr)]
        for g_ in ('top 5', 'field'):
            y = x[x.grp == g_]; h1 = y[y.obs_q >= 100]; h3 = y[y.obs_q >= 300]; kk = y[y.kept]
            L.append(f'  other {dr:<4} {g_:<6} {h1.kept.mean():5.1%} ({len(h1)}) / {h3.kept.mean():5.1%} ({len(h3)}) || {h3[h3.pre_q < 10].kept.mean():5.1%} /'
                     f' {h3[h3.pre_q >= 10].kept.mean():5.1%} || {kk.kept_v1.sum() / max(kk.kept_q.sum(), 1e-9):+.3f} / {kk.kept_v5.sum() / max(kk.kept_q.sum(), 1e-9):+.3f}')
    x = OD[OD.kind == 'other']; hi5 = pos6 = 0
    for d in DAYS9:
        y = x[(x.day == int(d)) & (x.obs_q >= 300)]; a_ = y[y.grp == 'top 5']; b_ = y[y.grp == 'field']
        if len(a_) >= MIN_N and len(b_) >= MIN_N:
            hi5 += int(a_.kept.mean() > b_.kept.mean())
        kk = x[(x.day == int(d)) & x.kept & (x.grp == 'top 5')]
        if len(kk) >= MIN_N:
            pos6 += int(kk.kept_v1.sum() > 0)
    kt = x[x.kept & (x.grp == 'top 5')]
    V['K5'] = 'PASS' if hi5 >= 7 else 'FAIL'; V['K6'] = 'PASS' if pos6 >= 7 and kt.kept_v5.sum() > 0 else 'FAIL'
    L.append(f'  K5: top 5 > field at >= 300 sh on {hi5} of 9 -> {V["K5"]}; K6: kept fills > 0 at 1 s on {pos6} of 9, 5 s pooled'
             f' {kt.kept_v5.sum() / max(kt.kept_q.sum(), 1e-9):+.3f} -> {V["K6"]}')

    def cell(arm, cfg, d=None, srcs=('WS',)):
        y = A[(A.arm == arm) & (A.cfg == cfg) & A.src.isin(srcs) & ~A.young]
        if d:
            y = y[y.day == int(d)]
        return float((y.q * y.v1).sum()) / 100.0 / (dd[d] if d else ndays)
    L += ['', 'K7 EARLY ENTRY, corrected, against the same path (rung 0, WS $/day w/o young at 1 s): arm - lane, M 0 / M 50 / M 250 (days higher /'
          ' lower) | all trades M 50']
    k7 = {}
    for a_, (tau, fast) in ARMS2.items():
        ref = 'FAST0' if fast else 'LDG0'; parts = []
        for cfg in ('M0', 'M50', 'M250'):
            hi = sum(cell(a_, cfg, d) > cell(ref, cfg, d) for d in DAYS9); lo = sum(cell(a_, cfg, d) < cell(ref, cfg, d) for d in DAYS9)
            parts.append(f'{cell(a_, cfg) - cell(ref, cfg):+8.1f} ({hi} / {lo})')
        up = {c: sum(cell(a_, c, d) > cell(ref, c, d) for d in DAYS9) for c in ('M0', 'M50')}
        dn = {c: sum(cell(a_, c, d) < cell(ref, c, d) for d in DAYS9) for c in ('M0', 'M50')}
        k7[a_] = 'HELPS' if min(up.values()) >= 7 else ('HURTS' if min(dn.values()) >= 7 else 'NOT-SHOWN')
        L.append(f'  {a_:<8} vs {ref:<5} ' + ' / '.join(parts) + f' | {cell(a_, "M50", srcs=("WS", "E2")) - cell(ref, "M50", srcs=("WS", "E2")):+8.1f} -> {k7[a_]}')
    V['K7'] = 'C4 REVISED' if any(k7[a_] == 'HELPS' for a_ in ('MDG10v2', 'MDG25v2', 'MDG10Fv2', 'MDG25Fv2')) else 'C4 ROBUST'
    L.append(f'  K7: {V["K7"]}')
    L += ['', 'VERDICTS v2: ' + json.dumps(V) + '  (K8 from v1: PASS)']
    json.dump(V, open(f'{OUTD}/robust_check_v2.json', 'w'), indent=1)
    open(f'{OUTD}/robust_check_v2.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


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
