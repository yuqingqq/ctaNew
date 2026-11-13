"""ROBUST-CHECK (2026-11-13, session ce68ebe0). USER: "Can you review then make sure your conclusion is correct then give me some robust
results". Stricter tests of the conclusions given to the USER after SIDE-MOVES (+ post hoc, corrected) and MOVE-DOWN:
  C1 the side the price moves into (asks in a pump): the top makers' orders are mostly already at the next price before the move
  C2 the side the price moves away from (bids in a pump): they rarely requote to the better price
  C3 there their old order stays more often than others', and the kept fills pay
  C4 posting at the new best sooner after the move does not pay (MOVE-DOWN MD / MDF; but MD also held through the guard episode and the
     model-braked arm MDP was flawed: its entry was not gated, it churned)
  C5 the fast order path helps (rung 0, every clock)
  C6 the same in pumps and dumps
9 consumed replay days, first 60 s; PULL-DEPTH's events (recomputed, checked); real WS-timed legs with ORDERS' maker / order hash / taker
order (legs_tk); groups top 5 by daily maker volume vs the field. Direction: UP = pump (protection side = asks, other side = bids), DOWN.

Checks and pass criteria, DECLARED BEFORE THE FIRST RUN (day rule >= 7 of 9; a day with < 30 units in a compared cell does not count):
 K1 (C1) STRICT SWEEP: taker orders (certain id) with legs at BOTH the old best and the new price of a protection-side event within
    (tb - 1 s, tb + 100 ms] (one taker crossing both prices: its legs at the new price hit orders resting there before the price moved):
    the top-5 share of the volume filled at the new price. PASS if pooled >= 1.2 x the top-5 share of all real at-best WS fill shares, in
    UP and in DOWN, and above that day's at-best share on >= 7 days (UP and DOWN pooled per day).
 K2 (C1) PROOF WITH A MARGIN: of each group's fill shares at the new best (fills in (tb, tend]), the share from orders that had a fill at
    the new price at t <= tb - 250 ms (well before the birth, no frame-timing doubt). PASS if top 5 > field on >= 7 days in UP and in DOWN.
    Margins 0 / 50 ms printed.
 K3 (C1) FRONT: at protection-side new bests, the top-5 share of first-filled orders against their share of all filled orders (count,
    certain hashes). PASS (FRONT-BIASED) if ratio > 1 pooled and on >= 7 days.
 K4 (C2) REQUOTE WHEN VISIBLE: makers proven at the old best (other side): JOIN rate among events where >= 100 sh traded at the new price
    while it was the best (an order placed there would likely be reached). PASS (RARE) if the top-5 rate <= 1/3 pooled in UP and DOWN.
 K5 (C3) KEPT AT HIGH VOLUME: KEPT among maker-events with >= 300 sh traded at the old price after the move: PASS if top 5 > field on
    >= 7 days; also printed for old orders whose proven pre-move fills were < 10 sh (less likely used up) and >= 10 sh.
 K6 (C3) VALUE OF KEPT FILLS: c/sh at 1 s and 5 s: PASS if top 5 > 0 on >= 7 days (1 s) and the pooled 5 s value > 0.
 K7 (C4) GATED FAST ENTRY on rung 0 (the fair version of MDP): enter the new best at once only if the move has happened AND PULL-DEPTH's
    out-of-day P(move >= 1 more) < tau; pull it while the guard is on if P >= tau; tau 0.10 / 0.25 (PULL-DEPTH's values); today's path
    (MDG10, MDG25) and the fast path (MDG10F, MDG25F). Each against LDG rung 0: WS $/day w/o young at 1 s, HELPS if higher at M 0 and
    M 50 on >= 7 days, HURTS if lower at both on >= 7, else NOT-SHOWN (M 250 printed). C4 ROBUST if no gated arm HELPS; REVISED if one
    does.
 K8 (C5) FAST LADDER: the full guard ladder (rungs 0-5) with the fast path against today: WS $/day w/o young at 1 s higher at M 0, M 50
    and M 250 on >= 7 days each -> ROBUST; all-trade M 50 and E2 printed.
 C6 is read from K1, K2, K4, K5 (UP and DOWN each).
Controls (any failure -> no output): (i) events equal PULL-DEPTH's row for row; (ii) a gated arm with tau = 0 (never enters early) equals
LDG rung 0 fill for fill and order for order on every window-side and clock; (iii) the fast ladder's rung 0 equals BEST-GAP's FAST lane;
the LDG ladder equals FUTURE-BEST's LDG fills row for row; (iv) synthetic cases for the strict sweep, the margin proof and the gate with
three planted errors (the sweep counting any leg at the new price, the margin ignored, the gate ignored) that must each fail a case; clocks
and the cooldown switch restored after every run; (v) memory self-guard.
Limits fixed in advance: fills are lower bounds (no order creation / cancel times in public data); inferences from queue order assume
price-time priority, which live fills only partly follow (B1a: a fifth contradict the shown-size queue); replay rule M for ours (C5);
consumed days (C8); PULL-DEPTH's P carries the nested-fold leak noted by the other session's review.
Usage (run_light, PF_SLOT=claude): pf_robust_check.py selftest | day <day> | report | posthoc (added after the report); driver
run_robust_check.sh
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

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/robust_check_20261113'
FBD = f'{K.IO}/future_best_20261113'
WIN_PRE, WIN_POST, LEVEL_MS, PROOF_MS = PD.WIN_PRE, PD.WIN_POST, PD.LEVEL_MS, PD.PROOF_MS
OLD_MS = 2000.0; KEEP_MS = 10000.0; SW_PRE = 1000.0; SW_POST = 100.0; MARGINS = (0.0, 50.0, 250.0)
CFGS = [('M', 0.0), ('M', 50.0), ('M', 250.0)]; SIZE = 30.0; RUNGS = range(6)
GATED = {'MDG10': (0.10, False), 'MDG25': (0.25, False), 'MDG10F': (0.10, True), 'MDG25F': (0.25, True)}
MIN_N = 30
INF = float('inf'); NAN = float('nan')
MUT_SWEEP_LOOSE = MUT_MARGIN_ZERO = MUT_GATE_OFF = 0       # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def sweep_vol(L, p_new, p_old, tb):
    """Volume by (group, 'new' / 'old') in taker orders with legs at both p_old and p_new within (tb - 1 s, tb + 100 ms]."""
    t = L['t']; lo = bisect.bisect_right(t, tb - SW_PRE); hi = bisect.bisect_right(t, tb + SW_POST)
    told = {L['tk'][i] for i in range(lo, hi) if L['p'][i] == p_old and L['tk'][i] >= 0}
    tnew = {L['tk'][i] for i in range(lo, hi) if L['p'][i] == p_new and L['tk'][i] >= 0}
    both = tnew if MUT_SWEEP_LOOSE else (told & tnew)
    out = {}
    for i in range(lo, hi):
        tk = L['tk'][i]
        if tk < 0 or tk not in both or L['p'][i] not in (p_new, p_old):
            continue
        key = (L['grp'][i], 'new' if L['p'][i] == p_new else 'old')
        out[key] = out.get(key, 0.0) + L['q'][i]
    return out


def proof_orders(L, p_new, tb, tend):
    """Orders (certain hash) filled at p_new in (tb, tend]: group, rank (all keys, fill order), shares, proof at each margin (a fill of the
    same order at p_new at t <= tb - margin)."""
    t = L['t']; i_tb = bisect.bisect_right(t, tb)
    first = {}
    for i in range(0, i_tb):
        if L['p'][i] == p_new and L['oh'][i] and L['oh'][i] not in first:
            first[L['oh'][i]] = t[i]
    seen = {}; out = []
    for i in range(i_tb, bisect.bisect_right(t, tend)):
        if L['p'][i] != p_new:
            continue
        key = L['oh'][i] or f'leg{i}'
        if key not in seen:
            seen[key] = len(seen) + 1
            if L['oh'][i]:
                e = first.get(L['oh'][i], INF)
                out.append(dict(grp=L['grp'][i], rank=seen[key], q=0.0, oh=L['oh'][i],
                                proof=[bool(e <= tb - (0.0 if MUT_MARGIN_ZERO else m)) for m in MARGINS]))
        for o in out:
            if o['oh'] == L['oh'][i]:
                o['q'] += L['q'][i]
                break
    return out


def maker_events(L, p_new, p_old, tb, tend):
    """Makers proven at p_old in [tb - 2 s, tb): join (fill at p_new in (tb, tend]), kept (a proven old order fills again at p_old in
    (tb, tb + 10 s]), shares traded at p_old after tb (obs_q) and at p_new while the best (new_q), the proven pre-move fills' shares (pre_q),
    kept fills' shares and value at 1 s / 5 s."""
    t = L['t']; i_tb = bisect.bisect_right(t, tb); i_end = bisect.bisect_right(t, tb + KEEP_MS); i_new = bisect.bisect_right(t, tend)
    old = {}; pre = {}; grp = {}
    for i in range(bisect.bisect_left(t, tb - OLD_MS), bisect.bisect_left(t, tb)):
        if L['p'][i] == p_old and L['oh'][i]:
            s_ = L['sig'][i]; old.setdefault(s_, set()).add(L['oh'][i]); pre[s_] = pre.get(s_, 0.0) + L['q'][i]; grp.setdefault(s_, L['grp'][i])
    obs_q = float(sum(L['q'][i] for i in range(i_tb, i_end) if L['p'][i] == p_old))
    new_q = float(sum(L['q'][i] for i in range(i_tb, i_new) if L['p'][i] == p_new))
    joined = {L['sig'][i] for i in range(i_tb, i_new) if L['p'][i] == p_new}
    out = []
    for s_, hs in old.items():
        kix = [i for i in range(i_tb, i_end) if L['p'][i] == p_old and L['sig'][i] == s_ and L['oh'][i] in hs]
        out.append(dict(grp=grp[s_], join=s_ in joined, obs_q=obs_q, new_q=new_q, pre_q=pre[s_], kept=bool(kix),
                        kept_q=float(sum(L['q'][i] for i in kix)), kept_v1=float(sum(L['q'][i] * L['v1'][i] for i in kix)),
                        kept_v5=float(sum(L['q'][i] * L['v5'][i] for i in kix))))
    return out


def gate_line(blk, mv, P1, tau):
    """Entry block for a gated arm: blocked unless the move has happened and P(move >= 1 more) < tau (or LDG's line says free)."""
    ok = [bool(m_) if MUT_GATE_OFF else bool(m_ and (P1[k] == P1[k]) and P1[k] < tau) for k, m_ in enumerate(mv)]
    return [bool(b_ and not o_) for b_, o_ in zip(blk, ok)]


def selftest():
    bad = []
    # ask side, pump: old best 50, new 51 born at tb = 1000. Taker X (tk 7) sweeps 50 (field) and 51 (top 5, hash a); taker Y (tk 8) only
    # 51 (field, hash b) at 1050; a fill of order 'c' at 51 at 600 (proof at margin 250: 600 <= 750), 'a' first seen at 990 (not at 250)
    rows = [(600, 51, 4, 3, 'field', 'c', 5), (990, 50, 10, 2, 'field', 'z', 7), (990, 51, 6, 1, 'top 5', 'a', 7), (1050, 51, 5, 4, 'field', 'b', 8),
            (1100, 51, 3, 3, 'field', 'c', 9), (1200, 51, 2, 1, 'top 5', 'a', 9)]
    L = {k: [r[i] for r in rows] for i, k in enumerate(('t', 'p', 'q', 'sig', 'grp', 'oh', 'tk'))}; L['v1'] = [0.5] * len(rows); L['v5'] = [0.2] * len(rows)
    sv = sweep_vol(L, 51, 50, 1000.0)
    if sv != {('field', 'old'): 10, ('top 5', 'new'): 6}:
        bad.append(f'strict sweep {sv}')
    po = proof_orders(L, 51, 1000.0, 2000.0)
    got = [(o['oh'], o['rank'], o['q'], o['proof']) for o in po]
    if got != [('b', 1, 5, [False, False, False]), ('c', 2, 3, [True, True, True]), ('a', 3, 2, [True, False, False])]:
        bad.append(f'proof {got}')
    # other side (bid, pump): old best 49 (maker 1 top 5 order h1, maker 2 field order h2), new best 50 at tb = 1000; 49 trades again
    rows2 = [(500, 49, 8, 1, 'top 5', 'h1', 1), (700, 49, 3, 2, 'field', 'h2', 2), (1100, 50, 5, 1, 'top 5', 'h3', 3), (4000, 49, 6, 1, 'top 5', 'h1', 4),
             (4100, 49, 6, 2, 'field', 'h4', 5)]
    L2 = {k: [r[i] for r in rows2] for i, k in enumerate(('t', 'p', 'q', 'sig', 'grp', 'oh', 'tk'))}; L2['v1'] = [1.0] * 5; L2['v5'] = [2.0] * 5
    me = sorted((m['grp'], m['join'], m['kept'], m['obs_q'], m['new_q'], m['pre_q'], m['kept_v5']) for m in maker_events(L2, 50, 49, 1000.0, 2000.0))
    if me != [('field', False, False, 12.0, 5.0, 3.0, 0.0), ('top 5', True, True, 12.0, 5.0, 8.0, 12.0)]:
        bad.append(f'maker events {me}')
    g = gate_line([True, True, True], [False, True, True], [0.1, 0.1, 0.3], 0.25)
    if g != [True, False, True]:
        bad.append(f'gate {g}')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_SWEEP_LOOSE', 'MUT_MARGIN_ZERO', 'MUT_GATE_OFF'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    PMD.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ROBUST-CHECK SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


def run_gated(w, q, shf, vg, P1, tau, fast):
    ctx = BG.fast_context(w, 'FAST') if fast else w
    blk = PMD.entry_line(vg, w['tg']); ep = PMD.episodes(vg); mv = PMD.moved(vg, ctx['best'], w['side'])
    aux = gate_line(blk, mv, P1, tau)
    dec = PMD.md_dec(vg, w['tg'], ep, P1=P1, tau=tau)
    try:
        K.MUT_REST_NOCOOL = 1
        if fast:
            K.L_PLACE = 5.0; K.L_CANCEL = 5.0; K.L_PENDCANCEL = 5.0
        return K.sim_arm(ctx, q, shf, 'rest', aux=aux, decide=dec, idle_only=1, size=SIZE)
    finally:
        K.MUT_REST_NOCOOL = 0
        K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL = BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL']


def run_fast_rung(w, q, shf, vg, d):
    try:
        K.L_PLACE = 5.0; K.L_CANCEL = 5.0; K.L_PENDCANCEL = 5.0
        return K.sim_arm(BG.fast_context(w, 'FAST'), q, shf, 'rest', aux=vg, offset=d, size=SIZE)
    finally:
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
    st.update(window_sides=0, ctrl_gate0=0, ctrl_fast0=0, gated_differs=0, clock_ok=1, real_atbest_q={})
    evs = []; swr = []; orr = []; mkr = []; fills = []; pend = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        P1 = np.full(n, np.nan)
        if (m, side) in PRI:
            x_ = PR.iloc[PRI[(m, side)]]; P1[x_.k.values] = x_.P1.values
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
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
            t = float(anc[i]); s_ = int(SIG.at[r_]); pt = int(ptl[i])
            Lw['t'].append(t); Lw['p'].append(pt); Lw['q'].append(float(shl[i])); Lw['sig'].append(s_); Lw['grp'].append(gname(s_))
            Lw['oh'].append(OH.at[r_]); Lw['tk'].append(int(TK.at[r_])); Lw['v1'].append(val(t, 1000.0, pt)); Lw['v5'].append(val(t, 5000.0, pt))
            if w['ws'] <= t < w['ws'] + 60000.0 and GP.depth_of(tfl, P5, opp, side, t, pt) == 0:
                g_ = gname(s_); st['real_atbest_q'][g_] = st['real_atbest_q'].get(g_, 0.0) + float(shl[i])
        # part 2: our arms, all clocks
        for q, shf in CFGS:
            cfg = f'{q}{int(shf)}'
            res = {}
            for d in RUNGS:
                res[('LDG', d)] = K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=SIZE)
                res[('FAST', d)] = run_fast_rung(w, q, shf, vg, d)
            fb = BG.run_lane(w, q, shf, vg, SIZE, 'FAST'); st['ctrl_fast0'] += int(fb[0] != res[('FAST', 0)][0] or fb[1] != res[('FAST', 0)][1])
            c0 = run_gated(w, q, shf, vg, P1, 0.0, False); st['ctrl_gate0'] += int(c0[0] != res[('LDG', 0)][0] or c0[1] != res[('LDG', 0)][1])
            c1 = run_gated(w, q, shf, vg, P1, 0.0, True); st['ctrl_gate0'] += int(c1[0] != res[('FAST', 0)][0] or c1[1] != res[('FAST', 0)][1])
            for a_, (tau, fast) in GATED.items():
                res[(a_, 0)] = run_gated(w, q, shf, vg, P1, tau, fast)
                st['gated_differs'] += int(res[(a_, 0)][0] != res[('LDG', 0)][0])
            if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL, K.MUT_REST_NOCOOL) != (BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL'], 0):
                st['clock_ok'] = 0
            for (a_, d), (F, O, pulls) in res.items():
                for f in F:
                    i, pt, qq, age = f[0], int(f[1]), f[2], f[4]
                    t = float(anc[i])
                    if not (w['ws'] <= t < w['ws'] + 61000.0):
                        continue
                    fills.append((int(day), cfg, a_, d, 'WS' if src[i] else 'E2', age < 100.0, float(qq), val(t, 1000.0, pt)))
        pend.setdefault(m, {})[side] = dict(w=w, vg=vg, runs=(b0, valid, rs, kind), tfl=tfl, L=Lw)
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
                        c = [e for e in ST[(x, kd)] if t0 - WIN_PRE < e[2] <= t0 + WIN_POST]
                        if not c or (x, c[0][0]) in used:
                            continue
                        j, p, tb, te_run, po = c[0]; used.add((x, j)); E.append((f'{day}|{m}|{x}|{j}', kev, x, p, tb, te_run, po))
            for x in (0, 1):
                for kd, kev in ((2, 'calm fallback'), (1, 'calm improve')):
                    for j, p, tb, te_run, po in ST[(x, kd)]:
                        if (x, j) in used or not (wsm <= tb < wsm + 60000.0):
                            continue
                        lo = bisect.bisect_left(allon, tb - WIN_POST)
                        if lo < len(allon) and allon[lo] < tb + WIN_PRE:
                            continue
                        E.append((f'{day}|{m}|{x}|{j}', kev, x, p, tb, te_run, po))
            for eid, kev, x, p, tb, te_run, po in E:
                evs.append((eid, kev, x, p, tb))
                tend = min(te_run, tb + LEVEL_MS); Lx = pair[x]['L']
                dr = 'UP' if ((kev in ('adverse', 'calm fallback') and x == 1) or (kev in ('other', 'calm improve') and x == 0)) else 'DOWN'
                if kev in ('adverse', 'calm fallback'):
                    for (g_, wh), v_ in sweep_vol(Lx, p, po, tb).items():
                        swr.append((int(day), kev, dr, g_, wh, v_))
                    orr += [(int(day), kev, dr, o['grp'], o['rank'], o['q']) + tuple(o['proof']) for o in proof_orders(Lx, p, tb, tend)]
                else:
                    mkr += [(int(day), kev, dr, z['grp'], z['join'], z['obs_q'], z['new_q'], z['pre_q'], z['kept'], z['kept_q'], z['kept_v1'], z['kept_v5'])
                            for z in maker_events(Lx, p, po, tb, tend)]
            for s_ in (0, 1):
                pair[s_]['w'].pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    EV = pd.DataFrame(evs, columns=['eid', 'kind', 'side', 'p', 'tb'])
    st['events_differ'] = int(len(EV) != len(PE) or not (EV.eid.values.astype(str) == PE.eid.values.astype(str)).all()
                              or not (EV.p.values == PE.p.values).all() or not (EV.tb.values == PE.tb.values).all())
    F = pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'rung', 'src', 'young', 'q', 'v1'])
    R0 = pd.read_parquet(f'{FBD}/fills_{day}.parquet', columns=['cfg', 'arm', 'rung', 'q', 'v1'])
    ref_bad = {}
    for cfg in ('M0', 'M50', 'M250'):
        a_ = F[(F.arm == 'LDG') & (F.cfg == cfg)]; b_ = R0[(R0.arm == 'LDG') & (R0.cfg == cfg)]
        if len(a_) != len(b_) or not np.array_equal(a_.q.values, b_.q.values) or not np.array_equal(a_.v1.values, b_.v1.values, equal_nan=True):
            ref_bad[cfg] = (len(a_), len(b_))
    st['ref_bad'] = ref_bad; st['unpaired_windows'] = len(pend)
    if st['events_differ'] or st['ctrl_gate0'] or st['ctrl_fast0'] or ref_bad or not st['clock_ok'] or not st['gated_differs'] or st['unpaired_windows']:
        sys.exit(f'ROBUST-CHECK CONTROL FAILED on {day} -> no output: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict) or k_ == "ref_bad"} }')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    F.to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(swr, columns=['day', 'kind', 'dir', 'grp', 'which', 'q']).to_parquet(f'{OUTD}/sweeps_{day}.parquet', index=False)
    pd.DataFrame(orr, columns=['day', 'kind', 'dir', 'grp', 'rank', 'q', 'p0', 'p50', 'p250']).to_parquet(f'{OUTD}/orders_{day}.parquet', index=False)
    pd.DataFrame(mkr, columns=['day', 'kind', 'dir', 'grp', 'join', 'obs_q', 'new_q', 'pre_q', 'kept', 'kept_q', 'kept_v1', 'kept_v5']).to_parquet(
        f'{OUTD}/makers_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}; dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    rd = lambda nm: pd.concat([pd.read_parquet(f'{OUTD}/{nm}_{d}.parquet') for d in DAYS9], ignore_index=True)
    SW, OR, MK, A = rd('sweeps'), rd('orders'), rd('makers'), rd('fills')
    nb = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    ab = {d: stats[d]['stat']['real_atbest_q'] for d in DAYS9}
    sh_ab = sum(ab[d].get('top 5', 0.0) for d in DAYS9) / max(sum(sum(ab[d].values()) for d in DAYS9), 1e-9)
    shd = {d: ab[d].get('top 5', 0.0) / max(sum(ab[d].values()), 1e-9) for d in DAYS9}
    V = {}; L = [f'ROBUST-CHECK, 9 consumed replay days, first 60 s; as of {asof}; controls passed (events = PULL-DEPTH, gated tau 0 = LDG / FAST,'
                 f' fast rung 0 = BEST-GAP, LDG ladder = FUTURE-BEST, clocks restored, planted errors caught); fills without a 1 s value {nb};'
                 f' top-5 share of all real at-best WS fill shares {sh_ab:.1%}', '']
    # K1
    L.append('K1 STRICT SWEEP (protection side, taker orders crossing the old best into the new price): top-5 share of the volume filled at the'
             ' new price | at the old best | units (shares)')
    k1 = {}
    for kev in ('adverse', 'calm fallback'):
        for dr in ('UP', 'DOWN'):
            x = SW[(SW.kind == kev) & (SW.dir == dr)]; nw = x[x.which == 'new']; od = x[x.which == 'old']
            s_new = nw.q[nw.grp == 'top 5'].sum() / max(nw.q.sum(), 1e-9); s_old = od.q[od.grp == 'top 5'].sum() / max(od.q.sum(), 1e-9)
            k1[(kev, dr)] = s_new
            L.append(f'  {kev:<13} {dr:<4} {s_new:5.1%} | {s_old:5.1%} | {nw.q.sum():.0f}')
    x = SW[(SW.kind == 'adverse') & (SW.which == 'new')]; up_d = 0
    for d in DAYS9:
        y = x[x.day == int(d)]
        if y.q.sum() >= MIN_N:
            up_d += int(y.q[y.grp == 'top 5'].sum() / y.q.sum() > shd[d])
    V['K1'] = 'PASS' if (k1[('adverse', 'UP')] >= 1.2 * sh_ab and k1[('adverse', 'DOWN')] >= 1.2 * sh_ab and up_d >= 7) else 'FAIL'
    L.append(f'  K1: adverse UP {k1[("adverse", "UP")]:.1%}, DOWN {k1[("adverse", "DOWN")]:.1%} vs 1.2 x {sh_ab:.1%} = {1.2 * sh_ab:.1%}; above the day\'s'
             f' at-best share on {up_d} of 9 days -> {V["K1"]}')
    # K2
    L += ['', 'K2 PROOF WITH A MARGIN (protection-side new best): share of the group\'s fill shares there from orders filled at that price'
          ' at t <= birth - margin: margin 0 / 50 / 250 ms']
    pk2 = {}
    for kev in ('adverse', 'calm fallback'):
        for dr in ('UP', 'DOWN'):
            x = OR[(OR.kind == kev) & (OR.dir == dr)]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]
                L.append(f'  {kev:<13} {dr:<4} {g_:<6} ' + ' / '.join(f'{y.q[y[c]].sum() / max(y.q.sum(), 1e-9):5.1%}' for c in ('p0', 'p50', 'p250')) + f' ({len(y)} orders)')
            if kev == 'adverse':
                hi = 0
                for d in DAYS9:
                    y = x[x.day == int(d)]; a_ = y[y.grp == 'top 5']; b_ = y[y.grp == 'field']
                    if len(a_) >= MIN_N and len(b_) >= MIN_N:
                        hi += int(a_.q[a_.p250].sum() / max(a_.q.sum(), 1e-9) > b_.q[b_.p250].sum() / max(b_.q.sum(), 1e-9))
                pk2[dr] = hi
    V['K2'] = 'PASS' if min(pk2.values()) >= 7 else 'FAIL'
    L.append(f'  K2: top 5 > field at 250 ms margin on UP {pk2["UP"]} / DOWN {pk2["DOWN"]} of 9 days -> {V["K2"]}')
    # K3
    L += ['', 'K3 FRONT (protection-side new best): top-5 share of first-filled orders | of all filled orders (count) | ratio']
    for kev in ('adverse', 'calm fallback'):
        x = OR[OR.kind == kev]; f1 = (x[x['rank'] == 1].grp == 'top 5').mean(); fa = (x.grp == 'top 5').mean()
        L.append(f'  {kev:<13} {f1:5.1%} | {fa:5.1%} | {f1 / fa if fa else NAN:.2f}')
    x = OR[OR.kind == 'adverse']; hi = 0
    for d in DAYS9:
        y = x[x.day == int(d)]
        if len(y) >= MIN_N and (y['rank'] == 1).sum() >= MIN_N:
            hi += int((y[y['rank'] == 1].grp == 'top 5').mean() > (y.grp == 'top 5').mean())
    r3 = (x[x['rank'] == 1].grp == 'top 5').mean() / max((x.grp == 'top 5').mean(), 1e-9)
    V['K3'] = 'PASS' if r3 > 1 and hi >= 7 else 'FAIL'
    L.append(f'  K3: ratio {r3:.2f}, higher on {hi} of 9 days -> {V["K3"]}')
    # K4
    L += ['', 'K4 REQUOTE WHEN VISIBLE (other side, makers proven at the old best): JOIN rate, all events | events with >= 100 sh traded at the'
          ' new price while it was the best']
    k4 = {}
    for kev in ('other', 'calm improve'):
        for dr in ('UP', 'DOWN'):
            x = MK[(MK.kind == kev) & (MK.dir == dr)]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]; h = y[y.new_q >= 100]
                if kev == 'other':
                    k4[(dr, g_)] = float(h['join'].mean()) if len(h) else NAN
                L.append(f'  {kev:<13} {dr:<4} {g_:<6} {y["join"].mean():5.1%} ({len(y)}) | {h["join"].mean() if len(h) else NAN:5.1%} ({len(h)})')
    V['K4'] = 'PASS' if k4[('UP', 'top 5')] <= 1 / 3 and k4[('DOWN', 'top 5')] <= 1 / 3 else 'FAIL'
    L.append(f'  K4: top-5 JOIN when visible UP {k4[("UP", "top 5")]:.1%}, DOWN {k4[("DOWN", "top 5")]:.1%} (RARE if <= 33%) -> {V["K4"]}')
    # K5, K6
    L += ['', 'K5 / K6 KEPT (other side): KEPT | traded at the old price after the move >= 100 / >= 300 sh || at >= 300 sh, pre-move fills < 10 /'
          ' >= 10 sh || kept fills c/sh at 1 s / 5 s']
    k5 = {}; k6 = {}
    for kev in ('other', 'calm improve'):
        for dr in ('UP', 'DOWN'):
            x = MK[(MK.kind == kev) & (MK.dir == dr)]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]; h1 = y[y.obs_q >= 100]; h3 = y[y.obs_q >= 300]; kk = y[y.kept]
                L.append(f'  {kev:<13} {dr:<4} {g_:<6} {h1.kept.mean() if len(h1) else NAN:5.1%} ({len(h1)}) / {h3.kept.mean() if len(h3) else NAN:5.1%} ({len(h3)}) ||'
                         f' {h3[h3.pre_q < 10].kept.mean() if (h3.pre_q < 10).any() else NAN:5.1%} / {h3[h3.pre_q >= 10].kept.mean() if (h3.pre_q >= 10).any() else NAN:5.1%} ||'
                         f' {kk.kept_v1.sum() / max(kk.kept_q.sum(), 1e-9):+.3f} / {kk.kept_v5.sum() / max(kk.kept_q.sum(), 1e-9):+.3f}')
    x = MK[MK.kind == 'other']; hi5 = 0; pos6 = 0
    for d in DAYS9:
        y = x[(x.day == int(d)) & (x.obs_q >= 300)]; a_ = y[y.grp == 'top 5']; b_ = y[y.grp == 'field']
        if len(a_) >= MIN_N and len(b_) >= MIN_N:
            hi5 += int(a_.kept.mean() > b_.kept.mean())
        kk = x[(x.day == int(d)) & x.kept & (x.grp == 'top 5')]
        if len(kk) >= MIN_N:
            pos6 += int(kk.kept_v1.sum() > 0)
    kt = x[x.kept & (x.grp == 'top 5')]
    V['K5'] = 'PASS' if hi5 >= 7 else 'FAIL'
    V['K6'] = 'PASS' if pos6 >= 7 and kt.kept_v5.sum() > 0 else 'FAIL'
    L.append(f'  K5: top 5 KEPT > field at >= 300 sh on {hi5} of 9 days -> {V["K5"]}; K6: top-5 kept fills > 0 at 1 s on {pos6} of 9 days, pooled 5 s'
             f' {kt.kept_v5.sum() / max(kt.kept_q.sum(), 1e-9):+.3f} c/sh -> {V["K6"]}')
    # K7, K8
    def cell(arm, cfg, d=None, srcs=('WS',), rungs=None):
        y = A[(A.arm == arm) & (A.cfg == cfg) & A.src.isin(srcs) & ~A.young]
        if rungs is not None:
            y = y[y.rung.isin(rungs)]
        if d:
            y = y[y.day == int(d)]
        return float((y.q * y.v1).sum()) / 100.0 / (dd[d] if d else ndays)
    L += ['', 'K7 GATED FAST ENTRY (rung 0): WS $/day w/o young at 1 s, M 0 / M 50 / M 250 (days higher than LDG rung 0) | all trades M 50']
    for a_ in ('LDG', 'FAST') + tuple(GATED):
        rr = [0]
        parts = []
        for cfg in ('M0', 'M50', 'M250'):
            hi = sum(cell(a_, cfg, d, rungs=rr) > cell('LDG', cfg, d, rungs=rr) for d in DAYS9)
            parts.append(f'{cell(a_, cfg, rungs=rr):+8.1f} ({hi})')
        L.append(f'  {a_:<7} ' + ' / '.join(parts) + f' | {cell(a_, "M50", srcs=("WS", "E2"), rungs=rr):+8.1f}')
    k7 = {}
    for a_ in GATED:
        up = {c: sum(cell(a_, c, d, rungs=[0]) > cell('LDG', c, d, rungs=[0]) for d in DAYS9) for c in ('M0', 'M50')}
        dn = {c: sum(cell(a_, c, d, rungs=[0]) < cell('LDG', c, d, rungs=[0]) for d in DAYS9) for c in ('M0', 'M50')}
        k7[a_] = 'HELPS' if min(up.values()) >= 7 else ('HURTS' if min(dn.values()) >= 7 else 'NOT-SHOWN')
    V['K7'] = 'C4 REVISED' if any(v_ == 'HELPS' for v_ in k7.values()) else 'C4 ROBUST'
    L.append(f'  K7: {k7} -> {V["K7"]}')
    L += ['', 'K8 FAST LADDER (rungs 0-5): WS $/day w/o young at 1 s, LDG | FAST (days higher) per clock; E2; all trades']
    k8 = {}
    for cfg in ('M0', 'M50', 'M250'):
        hi = sum(cell('FAST', cfg, d) > cell('LDG', cfg, d) for d in DAYS9); k8[cfg] = hi
        L.append(f'  {cfg:<4} {cell("LDG", cfg):+8.1f} | {cell("FAST", cfg):+8.1f} ({hi}/9) || E2 {cell("LDG", cfg, srcs=("E2",)):+8.1f} | {cell("FAST", cfg, srcs=("E2",)):+8.1f}'
                 f' || all {cell("LDG", cfg, srcs=("WS", "E2")):+8.1f} | {cell("FAST", cfg, srcs=("WS", "E2")):+8.1f}')
    V['K8'] = 'PASS' if min(k8.values()) >= 7 else 'FAIL'
    L.append(f'  K8: FAST ladder higher on {k8} -> {V["K8"]}')
    L += ['', 'VERDICTS: ' + json.dumps(V)]
    os.makedirs(OUTD, exist_ok=True)
    json.dump(V, open(f'{OUTD}/robust_check.json', 'w'), indent=1)
    open(f'{OUTD}/robust_check.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def posthoc():
    """POST HOC (added after the report, descriptive): K7's declared rule compared each gated arm with LDG rung 0, so the fast-path arms
    (MDG10F, MDG25F) inherit the fast path's own gain. Matched comparisons isolate the gated early entry: MDG10 / MDG25 against LDG rung 0,
    MDG10F / MDG25F against FAST rung 0 (same path): WS $/day w/o young at 1 s and all trades, per clock, days higher / lower."""
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}; dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True); A = A[np.isfinite(A.v1) & (A.rung == 0) & ~A.young]

    def cell(arm, cfg, d=None, srcs=('WS',)):
        y = A[(A.arm == arm) & (A.cfg == cfg) & A.src.isin(srcs)]
        if d:
            y = y[y.day == int(d)]
        return float((y.q * y.v1).sum()) / 100.0 / (dd[d] if d else ndays)
    L = ['ROBUST-CHECK post hoc (descriptive): gated early entry against the same path, rung 0, w/o young: WS $/day difference (days higher /'
         ' lower of 9) per clock | all trades M 50 difference']
    for a_, ref in (('MDG10', 'LDG'), ('MDG25', 'LDG'), ('MDG10F', 'FAST'), ('MDG25F', 'FAST')):
        parts = []
        for cfg in ('M0', 'M50', 'M250'):
            hi = sum(cell(a_, cfg, d) > cell(ref, cfg, d) for d in DAYS9); lo = sum(cell(a_, cfg, d) < cell(ref, cfg, d) for d in DAYS9)
            parts.append(f'{cfg} {cell(a_, cfg) - cell(ref, cfg):+7.1f} ({hi} / {lo})')
        L.append(f'  {a_:<6} vs {ref:<4} ' + ' | '.join(parts) + f' | all M 50 {cell(a_, "M50", srcs=("WS", "E2")) - cell(ref, "M50", srcs=("WS", "E2")):+7.1f}')
    open(f'{OUTD}/posthoc.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    elif cmd == 'posthoc':
        posthoc()
    else:
        sys.exit(__doc__)
