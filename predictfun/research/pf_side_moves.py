"""SIDE-MOVES (2026-11-13, session ce68ebe0). USER: "Can you still check both sides for top makers, on protection side, it looks we have
similar latency performance, but for the other quoting side, can you check what they do, do they also requote or what".
After PULL-DEPTH (bc4aa5a2) and FUTURE-BEST (bdc7868b): after a guard onset the other side's new best is a race the top 5 do not win (20% of
first fills); the protection side's new best is queued before it forms. Here: what the top makers do with their QUOTES on each side,
order by order (ORDERS' maker / order-hash map; real legs, WS-timed only).

Events (PULL-DEPTH's, recomputed here and checked row for row): guard onset on side s at t0 (first 50 ms step with v <= -0.10), first
60 s, 9 consumed replay days.
  protection side s: the first level that becomes s's best by FALLBACK in (t0 - 100 ms, t0 + 1 s]: p_new = the landing price,
                     p_old = the best that went away
  other side 1-s:    the first level that becomes its best by IMPROVEMENT in the same interval: p_new = the requote price,
                     p_old = the best it improved on
  calm:              other fallback / improve births of the first minute with no onset on either side in [birth - 1 s, birth + 100 ms)
  Births at tb; a level counts while it stays the best, at most 2 s (tend).
Measures (orders are seen only through fills: every share below is a lower bound, the same for every group):
  A. at p_new, fills in (tb, tend]: share of fill shares by group; the order of first fills = arrival order under FIFO: FIRST share,
     shares filled ahead of each order (queue position, lower bound); each order's origin, in this order:
       RESTED      the order itself had a fill at p_new before t_ref (onset; calm: birth - 250 ms): it was at that price before
       FROM-OLD    its maker had a fill at p_old in [tb - 2 s, tb): the maker was quoting the old best (moved or added)
       OTHER       its maker had a fill on this side at another price in [tb - 5 s, tb)
       FRESH       none of these
  B. at p_old, makers with an order PROVEN there (a fill at p_old in [tb - 2 s, tb), certain hash): per maker and event
       JOIN        the maker has a fill at p_new in (tb, tend] (moved or added a quote at the new price)
       OBS         p_old trades again (anyone) in (tb, tb + 10 s]
       KEPT        one of the maker's proven old orders fills again at p_old in (tb, tb + 10 s] (the old quote stayed)
     among OBS: MOVE (JOIN, not KEPT), ADD (JOIN and KEPT), KEEP (KEPT, not JOIN), NONE
  C. protection side only, run over: fills at p_old in (tb - 1 s, tb] (the old best's last second) by group, c/sh at 1 s
  D. ours (SIMULATED, exact, C9): the guard ladder's rung 0 (LDG, M 50, 30 sh), at tb: its order AT-NEW / HELD-OLD (live at p_old) /
     HELD-OTHER / PEND / OUT; JOIN = an order of ours live at p_new by tend; the held old order still live 1 s after tb.
Groups: top 5 by daily maker volume (B12, GP.rank_groups) vs the field (everyone else).

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule >= 7 of 9; a day with < 30 maker-events (B) / orders (A) in either
group does not count; ratio = top 5 rate / field rate, pooled):
 R510 OTHER-REQUOTE: other side after a signal, JOIN rate of makers proven at p_old: ratio >= 1.2 and higher on >= 7 days ->
      TOP-REQUOTE-MORE; <= 1/1.2 and lower on >= 7 -> TOP-REQUOTE-LESS; else SIMILAR.
 R511 OTHER-KEEP: other side after a signal, KEPT rate among OBS maker-events: ratio >= 1.2 and higher on >= 7 -> TOP-KEEP-MORE;
      <= 1/1.2 and lower on >= 7 -> TOP-KEEP-LESS; else SIMILAR.
 R512 OTHER-QUEUE: other side after a signal, median shares filled ahead of an order at p_new, top 5 minus field < 0 on >= 7 days ->
      TOP-AHEAD; > 0 on >= 7 -> TOP-BEHIND; else SIMILAR.
 R513 PROTECTION-RUNOVER: top 5 share of the fill shares at the old best's last second against their share of all real at-best WS fill
      shares: ratio <= 1/1.2 and lower on >= 7 -> TOP-AVOIDS; >= 1.2 and higher on >= 7 -> TOP-RUN-OVER-MORE; else PROPORTIONAL.
 R514 PROTECTION (descriptive): A and B on the protection side; calm births for every table; ours beside.
What the outcomes mean, fixed in advance: TOP-REQUOTE-MORE = when the other side's price improves after a signal, the top makers put a
quote at the new price more than others; TOP-KEEP-MORE = they leave their old quote in place more than others (with JOIN: keep-and-add);
TOP-AHEAD = their orders at the new price are earlier in its queue; TOP-AVOIDS = they are out of the old best before it is run over.
Controls (any failure -> no output): (i) the events equal PULL-DEPTH's (event id, price, birth) row for row on every day; (ii) sim4 rung 0
equals the simulator (sim_arm) fill for fill and order for order on every window-side; (iii) synthetic cases for A, B, C with three planted
errors (an order's own first fill counted in its shares ahead; KEPT credited to a new order of the same maker; JOIN counting fills before
the birth) that must each fail a case; (iv) memory self-guard.
Limits fixed in advance: orders are seen only through fills (lower bounds; an order used up before tb cannot show KEPT, which favours
larger orders); WS-timed legs only (C12); the guard is OUR signal, the top makers' own signals are unknown; consumed days (C8); ours
simulated with replay rule M (C5, C9).
Usage (run_light, PF_SLOT=claude): pf_side_moves.py selftest | day <day> | report; driver run_side_moves.sh. POST HOC (added after the
report, descriptive): dayposthoc <day> | posthoc; driver run_side_moves_posthoc.sh
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
import pf_queue_place as QP                               # noqa: E402  (sim4)
import pf_timing_split as TS                              # noqa: E402  (WS flag)
import pf_pull_depth as PD                                # noqa: E402  (events, onsets)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/side_moves_20261113'
ORD = PD.ORD
WIN_PRE, WIN_POST, LEVEL_MS, PROOF_MS = PD.WIN_PRE, PD.WIN_POST, PD.LEVEL_MS, PD.PROOF_MS
OLD_MS = 2000.0; SIDE_MS = 5000.0; KEEP_MS = 10000.0; RUN_MS = 1000.0
MIN_N = 30
KINDS = ('adverse', 'other', 'calm fallback', 'calm improve')
INF = float('inf'); NAN = float('nan')
MUT_AHEAD_SELF = MUT_KEPT_ANY = MUT_JOIN_EARLY = 0         # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def measures(L, p_new, p_old, tb, tend, t_ref, adverse, extra=False):
    """L: one side's WS legs sorted by time: t, p, q, sig, grp, oh ('' = no certain hash), v1 (lists). Returns (orders at p_new,
    maker-events at p_old, run-over fills); extra=True (POST HOC readout, 2026-11-13 after the report) adds per order 'prebirth' (the order
    had a fill at p_new at or before the birth: it was resting there before the price became the best), per maker-event the shares traded
    at p_old after the birth (obs_q) and its kept fills' shares / value / first time, and returns a 4th list: fills at p_new in
    (tb - 1 s, tb] (the trade that made it the best)."""
    t = L['t']; n = len(t)
    i0 = bisect.bisect_left(t, tb - SIDE_MS); i_tb = bisect.bisect_right(t, tb); i_end = bisect.bisect_right(t, tb + KEEP_MS)
    # A: orders at p_new
    seen = {}; ahead = 0.0; orders = []
    lo = i0 if MUT_JOIN_EARLY else i_tb
    for i in range(lo, bisect.bisect_right(t, tend)):
        if L['p'][i] != p_new:
            continue
        key = L['oh'][i] or f'leg{i}'
        if key not in seen:
            seen[key] = len(orders)
            orders.append(dict(grp=L['grp'][i], sig=L['sig'][i], oh=L['oh'][i], rank=len(orders) + 1,
                               ahead=ahead + (L['q'][i] if MUT_AHEAD_SELF else 0.0), gap=t[i] - tb, q=0.0, v=0.0))
        o = orders[seen[key]]; o['q'] += L['q'][i]; o['v'] += L['q'][i] * L['v1'][i]
        ahead += L['q'][i]
    pre_new = {L['oh'][i] for i in range(0, bisect.bisect_left(t, t_ref)) if L['p'][i] == p_new and L['oh'][i]}
    old_sig = {}; side_sig = set()
    for i in range(i0, bisect.bisect_left(t, tb)):
        side_sig.add(L['sig'][i])
        if L['p'][i] == p_old and t[i] >= tb - OLD_MS and L['oh'][i]:
            old_sig.setdefault(L['sig'][i], set()).add(L['oh'][i])
    pb = {L['oh'][i] for i in range(0, i_tb) if L['p'][i] == p_new and L['oh'][i]}
    for o in orders:
        o['prebirth'] = bool(o['oh'] and o['oh'] in pb)
        o['origin'] = ('RESTED' if o['oh'] and o['oh'] in pre_new else ('FROM-OLD' if o['sig'] in old_sig else
                                                                         ('OTHER' if o['sig'] in side_sig else 'FRESH')))
    # B: makers proven at p_old
    jn = {}
    for o in orders:
        jn.setdefault(o['sig'], o['gap'])
    obs = any(L['p'][i] == p_old for i in range(i_tb, i_end))
    obs_q = float(sum(L['q'][i] for i in range(i_tb, i_end) if L['p'][i] == p_old))
    mk = []
    for s_, hs in old_sig.items():
        kix = [i for i in range(i_tb, i_end) if L['p'][i] == p_old and L['sig'][i] == s_ and (MUT_KEPT_ANY or L['oh'][i] in hs)]
        kept = bool(kix)
        g_ = next(L['grp'][i] for i in range(i0, i_tb) if L['sig'][i] == s_)
        mk.append(dict(grp=g_, join=s_ in jn, jgap=jn.get(s_, NAN), obs=obs, kept=kept, obs_q=obs_q,
                       kept_q=float(sum(L['q'][i] for i in kix)), kept_v=float(sum(L['q'][i] * L['v1'][i] for i in kix)),
                       kept_gap=(t[kix[0]] - tb) if kix else NAN))
    # C: run over (protection side)
    ro = []
    if adverse:
        for i in range(bisect.bisect_right(t, tb - RUN_MS), i_tb):
            if L['p'][i] == p_old:
                ro.append(dict(grp=L['grp'][i], q=L['q'][i], v1=L['v1'][i]))
    if extra:
        sw = [dict(grp=L['grp'][i], q=L['q'][i]) for i in range(bisect.bisect_right(t, tb - RUN_MS), i_tb) if L['p'][i] == p_new]
        return orders, mk, ro, sw
    return orders, mk, ro


def selftest():
    bad = []
    # ask side; old best 52 (A top 5: hash a1; B field: b1), improves to 51 at tb = 1000; A joins 51 (a2) at 1100 behind C (c1, first);
    # 52 trades again at 3000: a1 fills (kept), b1 does not (B's new order b2 fills there instead); B's fill at 51 before tb is no join
    rows = [(500, 52, 10, 1, 'top 5', 'a1'), (600, 52, 5, 2, 'field', 'b1'), (800, 51, 2, 2, 'field', 'b9'), (900, 50, 3, 1, 'top 5', 'a0'),
            (1050, 51, 4, 3, 'field', 'c1'), (1100, 51, 6, 1, 'top 5', 'a2'), (1150, 51, 2, 3, 'field', 'c1'),
            (3000, 52, 7, 1, 'top 5', 'a1'), (3100, 52, 2, 2, 'field', 'b2')]
    L = {k: [r[i] for r in rows] for i, k in enumerate(('t', 'p', 'q', 'sig', 'grp', 'oh'))}; L['v1'] = [0.5] * len(rows)
    o, mk, ro = measures(L, 51, 52, 1000.0, 2000.0, 1000.0, False)
    got = [(x['oh'], x['rank'], x['ahead'], x['q'], x['origin']) for x in o]
    if got != [('c1', 1, 0.0, 6, 'FRESH'), ('a2', 2, 4.0, 6, 'FROM-OLD')]:
        bad.append(f'A {got}')
    mm = sorted((x['grp'], x['join'], x['obs'], x['kept']) for x in mk)
    if mm != [('field', False, True, False), ('top 5', True, True, True)]:
        bad.append(f'B {mm}')
    # a fill at the new price before the birth must not count as a join; RESTED proof; run over on the protection side
    rows2 = [(400, 49, 5, 4, 'top 5', 'd1'), (700, 50, 8, 2, 'field', 'b1'), (950, 50, 3, 4, 'top 5', 'd0'), (1200, 49, 5, 4, 'top 5', 'd1')]
    L2 = {k: [r[i] for r in rows2] for i, k in enumerate(('t', 'p', 'q', 'sig', 'grp', 'oh'))}; L2['v1'] = [-1.0] * len(rows2)
    o2, mk2, ro2 = measures(L2, 49, 50, 1000.0, 2000.0, 800.0, True)
    if [(x['oh'], x['origin'], x['ahead']) for x in o2] != [('d1', 'RESTED', 0.0)]:
        bad.append(f'A2 {[(x["oh"], x["origin"]) for x in o2]}')
    if sorted((x['grp'], x['join']) for x in mk2) != [('field', False), ('top 5', True)]:
        bad.append('B2')
    if [(x['grp'], x['q']) for x in ro2] != [('field', 8), ('top 5', 3)]:
        bad.append(f'C {ro2}')
    o3, mk3, _ = measures(L, 51, 52, 1000.0, 1000.0, 1000.0, False)          # tend = tb: nothing at the new price, no join
    if o3 or any(x['join'] for x in mk3):
        bad.append('empty level')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_AHEAD_SELF', 'MUT_KEPT_ANY', 'MUT_JOIN_EARLY'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'SIDE-MOVES SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


def ours_at(O, p_new, p_old, tb, tend):
    """Our rung-0 lane at tb (order dicts from sim4)."""
    live = [o for o in O if not o['ref'] and o['ta'] <= tb < min(o['tr'], o['tfill'])]
    if any(o['pt'] == p_new for o in live):
        st_ = 'AT-NEW'
    elif any(o['pt'] == p_old for o in live):
        st_ = 'HELD-OLD'
    elif live:
        st_ = 'HELD-OTHER'
    elif any(o['ts'] <= tb < o['ta'] or o['tc'] <= tb < o['tr'] for o in O):
        st_ = 'PEND'
    else:
        st_ = 'OUT'
    jt = [o['ta'] for o in O if not o['ref'] and o['pt'] == p_new and o['ta'] <= tend and min(o['tr'], o['tfill']) > tb]
    held = [o for o in live if o['pt'] == p_old]
    return st_, (max(min(jt) - tb, 0.0) if jt else NAN), (any(min(o['tr'], o['tfill']) > tb + 1000.0 for o in held) if held else None)


def day_run(day, posthoc=False):
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
    SIG = pd.Series(full.signer.values[ix].astype('int64'), index=LT.index)
    MP = pd.read_parquet(f'{ORD}/map_{day}.parquet').set_index('row')
    OH = pd.Series(np.where(MP.hash_ok.reindex(LT.index).eq(True).values, MP.ohash.reindex(LT.index).fillna('').values, ''), index=LT.index)
    gname = lambda s_: 'top 5' if G.get(int(s_), 'rest') == 'top 5' else 'field'
    PE = pd.read_parquet(f'{PD.OUTD}/events_{day}.parquet', columns=['eid', 'kind', 'side', 'p', 'tb'])
    st.update(window_sides=0, sim_differ=0, real_atbest_q={})
    evs = []; ordr = []; mkr = []; ror = []; our = []; pend = {}; ordx = []; mkx = []; swx = []
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg = w['m'], w['side'], w['sg']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        anc = w['anc']; ptl = w['ptl']; shl = w['shl']
        Lw = dict(t=[], p=[], q=[], sig=[], grp=[], oh=[], v1=[])
        for i, r_ in enumerate(lrow):
            if not src[i]:
                continue
            t = float(anc[i]); s_ = int(SIG.at[r_])
            Lw['t'].append(t); Lw['p'].append(int(ptl[i])); Lw['q'].append(float(shl[i])); Lw['sig'].append(s_); Lw['grp'].append(gname(s_))
            Lw['oh'].append(OH.at[r_]); Lw['v1'].append(sg * (mff[max(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1, 0)] - int(ptl[i])))
        for i, r_ in enumerate(lrow):                    # real at-best WS shares by group (R513's denominator, as PULL-DEPTH)
            t = float(anc[i])
            if src[i] and w['ws'] <= t < w['ws'] + 60000.0 and GP.depth_of(tfl, P5, opp, side, t, int(ptl[i])) == 0:
                g_ = gname(SIG.at[r_]); st['real_atbest_q'][g_] = st['real_atbest_q'].get(g_, 0.0) + float(shl[i])
        tt = anc - 50.0
        Fo, Ot, O, _ = QP.sim4(side, tf, opp, P5, w['S5'], w.setdefault('cache', {}), w['tg'], w['vkl'], -INF, w['best'], w['bopp'], tt, ptl, shl,
                               'M', w['te'], vg, 0, QP.SIZE)
        Fk, Ok, _ = K.sim_arm(w, 'M', 50.0, 'rest', aux=vg, offset=0, size=QP.SIZE)
        st['sim_differ'] += int(Fk != Fo or Ok != Ot)
        pend.setdefault(m, {})[side] = dict(w=w, vg=vg, runs=(b0, valid, rs, kind), tfl=tfl, L=Lw, O=O)
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
                        j, p, tb, te_run, po = c[0]; used.add((x, j)); E.append((f'{day}|{m}|{x}|{j}', kev, x, p, tb, te_run, po, t0))
            for x in (0, 1):
                for kd, kev in ((2, 'calm fallback'), (1, 'calm improve')):
                    for j, p, tb, te_run, po in ST[(x, kd)]:
                        if (x, j) in used or not (wsm <= tb < wsm + 60000.0):
                            continue
                        lo = bisect.bisect_left(allon, tb - WIN_POST)
                        if lo < len(allon) and allon[lo] < tb + WIN_PRE:
                            continue
                        E.append((f'{day}|{m}|{x}|{j}', kev, x, p, tb, te_run, po, tb - PROOF_MS))
            for eid, kev, x, p, tb, te_run, po, tref in E:
                tend = min(te_run, tb + LEVEL_MS)
                res_ = measures(pair[x]['L'], p, po, tb, tend, tref, kev in ('adverse', 'calm fallback'), extra=posthoc)
                o_l, mk, ro = res_[:3]
                if posthoc:
                    ordx += [(int(day), kev, x, z['grp'], z['rank'], z['ahead'], z['q'], z['v'], z['origin'], z['prebirth']) for z in o_l if z['oh']]
                    mkx += [(int(day), kev, x, z['grp'], z['join'], z['obs'], z['kept'], z['obs_q'], z['kept_q'], z['kept_v'], z['kept_gap']) for z in mk]
                    swx += [(int(day), kev, x, z['grp'], z['q']) for z in res_[3]]
                evs.append((eid, kev, x, p, tb))
                ordr += [(int(day), kev, z['grp'], z['rank'], z['ahead'], z['gap'], z['q'], z['v'], z['origin']) for z in o_l if z['oh']]
                mkr += [(int(day), kev, z['grp'], z['join'], z['jgap'], z['obs'], z['kept']) for z in mk]
                ror += [(int(day), kev, z['grp'], z['q'], z['v1']) for z in ro]
                s0, jg, hk = ours_at(pair[x]['O'], p, po, tb, tend)
                our.append((int(day), kev, s0, jg, -1 if hk is None else int(hk)))
            for s_ in (0, 1):
                pair[s_]['w'].pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    EV = pd.DataFrame(evs, columns=['eid', 'kind', 'side', 'p', 'tb'])
    st['events_differ'] = int(len(EV) != len(PE) or not (EV.eid.values.astype(str) == PE.eid.values.astype(str)).all()
                              or not (EV.kind.values.astype(str) == PE.kind.values.astype(str)).all() or not (EV.p.values == PE.p.values).all()
                              or not (EV.tb.values == PE.tb.values).all())
    st['unpaired_windows'] = len(pend); st['events'] = EV.kind.value_counts().to_dict()
    if st['sim_differ'] or st['events_differ'] or st['unpaired_windows']:
        sys.exit(f'SIDE-MOVES CONTROL FAILED on {day} -> no output: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)} }')
    st['planted_caught'] = mut
    od = f'{OUTD}/posthoc' if posthoc else OUTD
    os.makedirs(od, exist_ok=True)
    if posthoc:
        pd.DataFrame(ordx, columns=['day', 'kind', 'side', 'grp', 'rank', 'ahead', 'q', 'v', 'origin', 'prebirth']).to_parquet(f'{od}/ordx_{day}.parquet', index=False)
        pd.DataFrame(mkx, columns=['day', 'kind', 'side', 'grp', 'join', 'obs', 'kept', 'obs_q', 'kept_q', 'kept_v', 'kept_gap']).to_parquet(f'{od}/mkx_{day}.parquet', index=False)
        pd.DataFrame(swx, columns=['day', 'kind', 'side', 'grp', 'q']).to_parquet(f'{od}/swx_{day}.parquet', index=False)
    pd.DataFrame(ordr, columns=['day', 'kind', 'grp', 'rank', 'ahead', 'gap', 'q', 'v', 'origin']).to_parquet(f'{od}/orders_{day}.parquet', index=False)
    pd.DataFrame(mkr, columns=['day', 'kind', 'grp', 'join', 'jgap', 'obs', 'kept']).to_parquet(f'{od}/makers_{day}.parquet', index=False)
    pd.DataFrame(ror, columns=['day', 'kind', 'grp', 'q', 'v1']).to_parquet(f'{od}/runover_{day}.parquet', index=False)
    pd.DataFrame(our, columns=['day', 'kind', 'state', 'jgap', 'held_1s']).to_parquet(f'{od}/ours_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{od}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} events {st["events"]} in {time.time() - T0:.0f} s', flush=True)


def report(od=OUTD, write=True):
    stats = {d: json.load(open(f'{od}/day_{d}.json')) for d in DAYS9}
    rd = lambda nm: pd.concat([pd.read_parquet(f'{od}/{nm}_{d}.parquet') for d in DAYS9], ignore_index=True)
    OR, MK, RO, OU = rd('orders'), rd('makers'), rd('runover'), rd('ours')
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    cps = lambda x, qc='q', vc='v': float(x[vc].sum() / x[qc].sum()) if len(x) and x[qc].sum() else NAN
    R = {}
    L = [f'SIDE-MOVES, 9 consumed replay days, first 60 s, WS-timed real legs; as of {asof}; controls passed (events = PULL-DEPTH row for row,'
         f' sim4 = simulator, planted errors caught); events ' + json.dumps({k_: int(sum(stats[d]['stat']['events'].get(k_, 0) for d in DAYS9)) for k_ in KINDS}), '']
    for title, kinds in (('OTHER SIDE (the side that requotes after the signal)', ('other', 'calm improve')),
                         ('PROTECTION SIDE (the side the signal protects)', ('adverse', 'calm fallback'))):
        L += [title, '  A. orders at the new price (filled): group | share of fill shares | FIRST share | shares ahead median (p75) | first fill ms after'
              ' the birth, median | c/sh | origin: RESTED / FROM-OLD / OTHER / FRESH (share of the group\'s orders)']
        for kev in kinds:
            x = OR[OR.kind == kev]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]
                org = ' / '.join(f'{(y.origin == o_).mean() if len(y) else NAN:4.0%}' for o_ in ('RESTED', 'FROM-OLD', 'OTHER', 'FRESH'))
                L.append(f'    {kev:<13} {g_:<6} {y.q.sum() / max(x.q.sum(), 1e-9):5.1%} | {((x['rank'] == 1) & (x.grp == g_)).sum() / max((x['rank'] == 1).sum(), 1):5.1%} |'
                         f' {np.median(y.ahead) if len(y) else NAN:5.0f} ({np.percentile(y.ahead, 75) if len(y) else NAN:5.0f}) | {np.median(y.gap) if len(y) else NAN:5.0f} |'
                         f' {cps(y):+.3f} | {org}')
        L.append('  B. makers proven at the old best just before the birth: group | maker-events | JOIN (quote at the new price) | JOIN ms median |'
                 ' OBS share | KEPT | OBS)  || among OBS: MOVE / ADD / KEEP / NONE')
        for kev in kinds:
            x = MK[MK.kind == kev]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]; ob = y[y.obs]
                oj = ob['join']; yj = y['join']                # ['join']: DataFrame.join is a method
                sp = ' / '.join(f'{v_:4.0%}' for v_ in ((oj & ~ob.kept).mean(), (oj & ob.kept).mean(), (~oj & ob.kept).mean(),
                                                        (~oj & ~ob.kept).mean())) if len(ob) else 'n/a'
                L.append(f'    {kev:<13} {g_:<6} {len(y):>7} | {yj.mean() if len(y) else NAN:5.1%} | {np.nanmedian(y.jgap) if yj.any() else NAN:5.0f} |'
                         f' {y.obs.mean() if len(y) else NAN:4.0%} | {ob.kept.mean() if len(ob) else NAN:5.1%} || {sp}')
        if kinds[0] == 'adverse':
            ab = {g_: sum(stats[d]['stat']['real_atbest_q'].get(g_, 0.0) for d in DAYS9) for g_ in ('top 5', 'field')}
            sh_ab = ab['top 5'] / max(sum(ab.values()), 1e-9)
            L.append(f'  C. run over: fills at the old best in its last 1 s before the fallback: group | share of shares | c/sh  (top 5 share of all'
                     f' at-best WS fill shares {sh_ab:.1%})')
            for kev in kinds:
                x = RO[RO.kind == kev]
                for g_ in ('top 5', 'field'):
                    y = x[x.grp == g_]
                    L.append(f'    {kev:<13} {g_:<6} {y.q.sum() / max(x.q.sum(), 1e-9):5.1%} | {float((y.q * y.v1).sum() / y.q.sum()) if y.q.sum() else NAN:+.3f}')
        L.append('  D. ours (SIMULATED, LDG rung 0, M 50): state at the birth AT-NEW / HELD-OLD / HELD-OTHER / PEND / OUT | an order at the new'
                 ' price by the level\'s end | ms median | the held old order still live 1 s later')
        for kev in kinds:
            x = OU[OU.kind == kev]; hl = x[x.held_1s >= 0]
            L.append(f'    {kev:<13} ' + ' / '.join(f'{(x.state == s_).mean():4.0%}' for s_ in ('AT-NEW', 'HELD-OLD', 'HELD-OTHER', 'PEND', 'OUT')) +
                     f' | {np.isfinite(x.jgap).mean():5.1%} | {np.nanmedian(x.jgap) if np.isfinite(x.jgap).any() else NAN:5.0f} | {hl.held_1s.mean() if len(hl) else NAN:4.0%}')
        L.append('')
    # readings
    def rate_days(df, kev, col, cond=None):
        x = df[df.kind == kev]
        if cond is not None:
            x = x[x[cond]]
        pooled = {g_: float(x[x.grp == g_][col].mean()) for g_ in ('top 5', 'field')}
        hi = lo_ = 0
        for d in DAYS9:
            y = x[x.day == int(d)]; a_ = y[y.grp == 'top 5']; b_ = y[y.grp == 'field']
            if len(a_) >= MIN_N and len(b_) >= MIN_N:
                hi += int(a_[col].mean() > b_[col].mean()); lo_ += int(a_[col].mean() < b_[col].mean())
        return pooled, hi, lo_
    for key, col, cond, more, less in (('R510', 'join', None, 'TOP-REQUOTE-MORE', 'TOP-REQUOTE-LESS'), ('R511', 'kept', 'obs', 'TOP-KEEP-MORE', 'TOP-KEEP-LESS')):
        pooled, hi, lo_ = rate_days(MK, 'other', col, cond)
        ratio = pooled['top 5'] / pooled['field'] if pooled['field'] else NAN
        lab = more if ratio >= 1.2 and hi >= 7 else (less if ratio <= 1 / 1.2 and lo_ >= 7 else 'SIMILAR')
        R[key] = dict(top5=pooled['top 5'], field=pooled['field'], ratio=ratio, higher=hi, lower=lo_, label=lab)
        L.append(f'{key}: other side, {col} rate top 5 {pooled["top 5"]:.1%} vs field {pooled["field"]:.1%} (ratio {ratio:.2f}; higher {hi}, lower {lo_} of 9) -> {lab}')
    x = OR[OR.kind == 'other']; hi = lo_ = 0
    for d in DAYS9:
        y = x[x.day == int(d)]; a_ = y[y.grp == 'top 5']; b_ = y[y.grp == 'field']
        if len(a_) >= MIN_N and len(b_) >= MIN_N:
            hi += int(np.median(a_.ahead) > np.median(b_.ahead)); lo_ += int(np.median(a_.ahead) < np.median(b_.ahead))
    lab = 'TOP-AHEAD' if lo_ >= 7 else ('TOP-BEHIND' if hi >= 7 else 'SIMILAR')
    R['R512'] = dict(top5=float(np.median(x[x.grp == 'top 5'].ahead)), field=float(np.median(x[x.grp == 'field'].ahead)), ahead_days=lo_, behind_days=hi, label=lab)
    L.append(f'R512: other side, shares ahead median top 5 {R["R512"]["top5"]:.0f} vs field {R["R512"]["field"]:.0f} (top 5 ahead on {lo_}, behind on {hi} of 9) -> {lab}')
    ab = {g_: sum(stats[d]['stat']['real_atbest_q'].get(g_, 0.0) for d in DAYS9) for g_ in ('top 5', 'field')}; sh_ab = ab['top 5'] / max(sum(ab.values()), 1e-9)
    x = RO[RO.kind == 'adverse']; sh = float(x.q[x.grp == 'top 5'].sum() / max(x.q.sum(), 1e-9)); hi = lo_ = 0
    for d in DAYS9:
        y = x[x.day == int(d)]; a_d = stats[d]['stat']['real_atbest_q']; s_d = a_d.get('top 5', 0.0) / max(sum(a_d.values()), 1e-9)
        if (y.grp == 'top 5').sum() >= MIN_N and (y.grp == 'field').sum() >= MIN_N:
            v_ = float(y.q[y.grp == 'top 5'].sum() / y.q.sum()); hi += int(v_ > s_d); lo_ += int(v_ < s_d)
    ratio = sh / sh_ab if sh_ab else NAN
    lab = 'TOP-AVOIDS' if ratio <= 1 / 1.2 and lo_ >= 7 else ('TOP-RUN-OVER-MORE' if ratio >= 1.2 and hi >= 7 else 'PROPORTIONAL')
    R['R513'] = dict(share=sh, atbest_share=sh_ab, ratio=ratio, higher=hi, lower=lo_, label=lab)
    L.append(f'R513: protection side, top 5 share of the run-over shares {sh:.1%} vs at-best {sh_ab:.1%} (ratio {ratio:.2f}; higher {hi}, lower {lo_} of 9) -> {lab}')
    if not write:
        return R
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/side_moves.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/side_moves.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    return R


def posthoc_report():
    """POST HOC (2026-11-13, after the report; USER: "So they don't requote, their orders are already there ... top makers don't pull bid
    side, then what they do? They wait to get filled? Or they requote in higher price? ... Can you give me some solid conclusions").
    Descriptive, no readings. Control: the declared readings recomputed from the post hoc rerun equal side_moves.json. Split by direction:
    UP (a pump: the protection side is the ask, the other side the bid) / DOWN (a dump). Fills are lower bounds (public data has no order
    creation or cancel times: the match records carry only hash, signer, price and amount; checked on a consumed day)."""
    od = f'{OUTD}/posthoc'
    R0 = json.load(open(f'{OUTD}/side_moves.json')); R1 = report(od, write=False)
    bad = {k: (R0[k], R1[k]) for k in R0 if json.dumps(R0[k], sort_keys=True, default=float) != json.dumps(json.loads(json.dumps(R1[k], default=float)), sort_keys=True, default=float)}
    if bad:
        sys.exit(f'SIDE-MOVES POSTHOC CONTROL FAILED (rerun readings differ) -> no output: {bad}')
    rd = lambda nm: pd.concat([pd.read_parquet(f'{od}/{nm}_{d}.parquet') for d in DAYS9], ignore_index=True)
    OX, MX, SW = rd('ordx'), rd('mkx'), rd('swx')
    up = lambda df: ((df.kind.isin(['adverse', 'calm fallback']) & (df.side == 1)) | (df.kind.isin(['other', 'calm improve']) & (df.side == 0)))
    for df in (OX, MX, SW):
        df['dir'] = np.where(up(df), 'UP', 'DOWN')
    cps = lambda q, v: float(v.sum() / q.sum()) if q.sum() else NAN
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'SIDE-MOVES post hoc (descriptive), as of {asof}; control passed (readings from the rerun = side_moves.json)', '',
         'PROTECTION SIDE new best (UP: the new best ask after a pump; DOWN: the new best bid after a dump): group | share of fill shares |'
         ' share of the group\'s fill shares there from orders PROVEN resting at that price at or before it became the best | first-fill share |'
         ' shares ahead median | c/sh || share of the fills at that price in the trade(s) that made it the best (last 1 s up to the birth)']
    for kev in ('adverse', 'calm fallback'):
        for dr in ('UP', 'DOWN'):
            x = OX[(OX.kind == kev) & (OX.dir == dr)]; s_ = SW[(SW.kind == kev) & (SW.dir == dr)]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]
                L.append(f'  {kev:<13} {dr:<4} {g_:<6} {y.q.sum() / max(x.q.sum(), 1e-9):5.1%} | {y.q[y.prebirth].sum() / max(y.q.sum(), 1e-9):5.1%} |'
                         f' {((x["rank"] == 1) & (x.grp == g_)).sum() / max((x["rank"] == 1).sum(), 1):5.1%} | {np.median(y.ahead) if len(y) else NAN:4.0f} |'
                         f' {cps(y.q, y.v):+.3f} || {s_.q[s_.grp == g_].sum() / max(s_.q.sum(), 1e-9):5.1%}')
        y5 = OX[(OX.kind == kev) & (OX.grp == 'top 5')]; yf = OX[(OX.kind == kev) & (OX.grp == 'field')]
        hi = sum(int(y5[y5.day == int(d)].q[y5[y5.day == int(d)].prebirth].sum() / max(y5[y5.day == int(d)].q.sum(), 1e-9) >
                     yf[yf.day == int(d)].q[yf[yf.day == int(d)].prebirth].sum() / max(yf[yf.day == int(d)].q.sum(), 1e-9)) for d in DAYS9)
        L.append(f'    {kev}: top 5 PROVEN-before share higher than the field on {hi} of 9 days')
    L += ['', 'MAKERS AT THE OLD BEST when the price moves (proven by a fill there in the 2 s before): group | maker-events | quote at the new price'
          ' (JOIN) | old order filled again at the old price in 10 s, given it traded again (KEPT | OBS) | KEPT | OBS by shares traded at the old'
          ' price after the move: < 30 / 30-100 / >= 100 sh | kept fills: c/sh at 1 s, first one ms after the move (median)']
    for kev in ('other', 'calm improve', 'adverse', 'calm fallback'):
        for dr in ('UP', 'DOWN'):
            x = MX[(MX.kind == kev) & (MX.dir == dr)]
            for g_ in ('top 5', 'field'):
                y = x[x.grp == g_]; ob = y[y.obs]
                bk = ' / '.join(f'{z.kept.mean() if len(z) else NAN:5.1%} ({len(z)})' for z in (ob[ob.obs_q < 30], ob[(ob.obs_q >= 30) & (ob.obs_q < 100)], ob[ob.obs_q >= 100]))
                kk = y[y.kept]
                L.append(f'  {kev:<13} {dr:<4} {g_:<6} {len(y):>6} | {y["join"].mean() if len(y) else NAN:5.1%} | {ob.kept.mean() if len(ob) else NAN:5.1%} | {bk} |'
                         f' {cps(kk.kept_q, kk.kept_v):+.3f}, {np.nanmedian(kk.kept_gap) if len(kk) else NAN:5.0f}')
        y5 = MX[(MX.kind == kev) & (MX.grp == 'top 5') & MX.obs]; yf = MX[(MX.kind == kev) & (MX.grp == 'field') & MX.obs]
        hk = sum(int(y5[y5.day == int(d)].kept.mean() > yf[yf.day == int(d)].kept.mean()) for d in DAYS9)
        hj = sum(int(MX[(MX.kind == kev) & (MX.grp == 'top 5') & (MX.day == int(d))]['join'].mean() > MX[(MX.kind == kev) & (MX.grp == 'field') & (MX.day == int(d))]['join'].mean()) for d in DAYS9)
        L.append(f'    {kev}: top 5 KEPT higher than the field on {hk} of 9 days; JOIN higher on {hj} of 9')
    open(f'{OUTD}/posthoc.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    elif cmd == 'dayposthoc' and len(sys.argv) > 2:
        day_run(sys.argv[2], posthoc=True)
    elif cmd == 'posthoc':
        posthoc_report()
    else:
        sys.exit(__doc__)
