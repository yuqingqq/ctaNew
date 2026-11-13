"""MOVE-CASES (2026-11-13, session ce68ebe0). USER: "You mean they don't pull the trending side, when price pumps they still have their ask
there, don't they have adverse selection in this case? When price pumps a lot, how do they behavior? ... if price pumps a lot, they will
suffer from loss right? Can you give me the logic there, the data you show are clearly incorrect" and "You can check different cases, if
price pumps a lot, let me know their logic chains, or a series of actions, they can't rest all their orders there to be taken adversely.
When price reverts, there will also be a different series of actions".
Why: ROBUST-CHECK's K1 (51% of the volume at the new price in crossing trades) measured only the price where the price STOPPED after the
cross (the new best, by construction a level that held); prices run over in a bigger move were not measured, and no fill there was valued.

Events: PULL-DEPTH's protection-side events (guard onset, then the first fallback birth within 1 s) and calm fallback births; invalid
births excluded (as ROBUST-CHECK v2); 9 consumed replay days, first 60 s; real WS-timed legs with ORDERS' maker / order hash / taker address.
INTO side = the side the price moves into (asks in a pump, bids in a dump); AWAY side = the other side. p_old = the INTO side's best before
the birth (tb). The path of the INTO side's best over valid frames:
  M       the largest move against the INTO side in [tb, tb + 2 s], ticks from p_old (>= 1 by construction); t_pk the first frame reaching M
  F       the INTO side's best at t_pk + 3 s, ticks from p_old (what was kept of the move)
  cases   '1 tick, holds' (M = 1, F >= 1) | '1 tick, reverts' (M = 1, F <= 0) | '2 ticks' (M = 2) | 'big, trends' (M >= 3, F >= M - 1) |
          'big, reverts' (M >= 3, F <= M // 2) | 'big, partial' (M >= 3, otherwise); events with tb > window start + 55 s: status
Segments (maker legs; value at 5 s and 1 s after each fill, the maker's view; a taker's value is the negative):
  into_old      INTO side at p_old, (tb - 1 s, t_pk]: the old best being run over
  into_runover  INTO side at levels strictly between p_old and the peak, (tb - 1 s, t_pk]: prices run over during the move
  into_peak     INTO side at the peak price, (tb - 1 s, t_pk]: the price where the move stopped (filled on the way)
  into_after    INTO side, any price, (t_pk, t_pk + 3 s]
  away_chase    AWAY side better than its best just before tb (bids above it in a pump), (tb - 1 s, t_pk]: quotes that followed the move
  away_old      AWAY side at or behind that best, (tb - 1 s, t_pk]
  away_after    AWAY side, any price, (t_pk, t_pk + 3 s] (in a reversion: the side being hit on the way back)
Takers (legs by the taker's address; groups: top 5 = a top-5 maker's address, other maker, non-maker, unknown):
  take_with     INTO-side legs in (tb - 1 s, t_pk]: buying into a pump (selling into a dump)
  take_after    INTO-side legs in (t_pk, t_pk + 3 s]
  take_against  AWAY-side legs in (t_pk, t_pk + 3 s]: selling after a pump's peak (buying after a dump's trough)
Ours (SIMULATED, C9): the guard ladder LDG (6 rungs, 30 sh, M 50) fills on WS-timed trades in the same segments.
Groups: top 5 by daily maker volume (B12) vs the field.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule >= 7 of 9; < 30 legs in a compared cell: the day does not count):
 R540 RUN-OVER EXPOSURE (big, trends): the top-5 share of into_runover volume against their share of all real at-best WS fill volume:
      lower pooled and on >= 7 days -> THIN-WHERE-RUN-OVER; higher pooled and on >= 7 -> THICK-WHERE-RUN-OVER; else PROPORTIONAL.
      into_old and into_peak shares printed beside, every case.
 R541 BIG-TREND VALUE: the top-5 into-side fills (into_old + into_runover + into_peak) in 'big, trends', c/sh at 5 s: < 0 on >= 7 days ->
      THEY-LOSE-THERE; and top 5 minus field > 0 on >= 7 -> LESS-THAN-THE-FIELD (< 0 on >= 7 -> MORE).
 R542 TAKING THE TREND: the top-5 share of take_with volume in 'big, trends' against '1 tick, holds': ratio >= 1.2 pooled and higher on
      >= 7 days -> TAKE-INTO-BIG-MOVES; <= 1 / 1.2 and lower on >= 7 -> AVOID-TAKING-BIG-MOVES; else SIMILAR.
 R543 REVERSION (big, reverts): the top-5 share of away_after volume and its c/sh at 5 s, against their usual share (descriptive) and the
      same in 'big, trends' (printed).
 R544 NET OVER MOVES: the top-5 maker $ / day at 5 s over all segments and cases > 0 on >= 7 days -> NET-POSITIVE; < 0 -> NET-NEGATIVE.
What the outcomes mean, fixed in advance: THIN-WHERE-RUN-OVER = in moves that keep going, little of the volume run over is theirs (they are
not resting there then, whether pulled or never placed - fills cannot tell); THEY-LOSE-THERE = their fills in big trends lose even so;
TAKE-INTO-BIG-MOVES = they are disproportionately the buyers in big pumps (sellers in big dumps).
Controls (any failure -> no output): (i) the event list equals PULL-DEPTH's before the exclusion; (ii) the LDG ladder equals FUTURE-BEST's
LDG fills (M 50) row for row; (iii) synthetic cases for the move path (M, t_pk, F, case) and the segment classifier on both sides with three
planted errors (the move measured with the wrong sign, run-over levels including the peak, AWAY 'chase' reversed) that must each fail a case;
(iv) memory self-guard.
Limits fixed in advance: fills only (orders that are pulled are invisible: a low share is "not filled there", never proof of a pull); WS-timed
legs only (C12); the guard is OUR signal; consumed days (C8); ours replay rule M (C5); taker addresses 90% known.
Usage (run_light, PF_SLOT=claude): pf_move_cases.py selftest | day <day> | report; driver run_move_cases.sh
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
import pf_pull_depth as PD                                # noqa: E402
import pf_robust_check as RC                              # noqa: E402

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/move_cases_20261113'
FBD = f'{K.IO}/future_best_20261113'
PRE = 1000.0; MOVE_MS = 2000.0; AFTER_MS = 3000.0; LAST_TB = 55000.0; SIZE = 30.0; RUNGS = range(6)
CASES = ('1 tick, holds', '1 tick, reverts', '2 ticks', 'big, trends', 'big, partial', 'big, reverts')
MSEG = ('into_old', 'into_runover', 'into_peak', 'into_after', 'away_chase', 'away_old', 'away_after')
TSEG = ('take_with', 'take_after', 'take_against')
MIN_N = 30
INF = float('inf'); NAN = float('nan')
MUT_PATH_SIGN = MUT_RUNOVER_PEAK = MUT_CHASE_REV = 0       # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def move_path(tfl, b0, valid, x, p_old, tb):
    """M (largest move against the INTO side x in [tb, tb + 2 s], ticks), t_pk (first frame reaching M), F (ticks at t_pk + 3 s)."""
    sgn = 1 if (x == 1) != bool(MUT_PATH_SIGN) else -1
    lo = bisect.bisect_left(tfl, tb); hi = bisect.bisect_right(tfl, tb + MOVE_MS)
    M = 0; t_pk = NAN
    for j in range(lo, hi):
        if valid[j]:
            e = sgn * (int(b0[j]) - p_old)
            if e > M:
                M = e; t_pk = tfl[j]
    if M < 1:
        return 0, NAN, NAN
    k = bisect.bisect_right(tfl, t_pk + AFTER_MS) - 1
    while k >= 0 and not valid[k]:
        k -= 1
    F = sgn * (int(b0[k]) - p_old) if k >= 0 else NAN
    return int(M), float(t_pk), F


def case_of(M, F):
    if M == 1:
        return '1 tick, holds' if F >= 1 else '1 tick, reverts'
    if M == 2:
        return '2 ticks'
    if F >= M - 1:
        return 'big, trends'
    if F <= M // 2:
        return 'big, reverts'
    return 'big, partial'


def seg_into(p, t, x, p_old, M, tb, t_pk):
    lev = (p - p_old) if x == 1 else (p_old - p)
    if tb - PRE < t <= t_pk:
        if lev == 0:
            return 'into_old'
        if 0 < lev < M or (MUT_RUNOVER_PEAK and lev == M):
            return 'into_runover'
        if lev == M:
            return 'into_peak'
        return None
    if t_pk < t <= t_pk + AFTER_MS:
        return 'into_after'
    return None


def seg_away(p, t, xa, a0, tb, t_pk):
    better = (p > a0) if xa == 0 else (p < a0)
    if MUT_CHASE_REV:
        better = not better
    if tb - PRE < t <= t_pk:
        return 'away_chase' if better else 'away_old'
    if t_pk < t <= t_pk + AFTER_MS:
        return 'away_after'
    return None


def selftest():
    bad = []
    tfl = [900.0, 1000.0, 1200.0, 1500.0, 2000.0, 4000.0, 4400.0]
    b0 = np.array([50, 51, 52, 54, 54, 52, 51]); valid = np.array([True] * 7)
    if move_path(tfl, b0, valid, 1, 50, 1000.0) != (4, 1500.0, 1):                  # ask pump to 54 at 1500, back to 51 by 4500
        bad.append(f'path {move_path(tfl, b0, valid, 1, 50, 1000.0)}')
    if case_of(4, 1) != 'big, reverts' or case_of(4, 3) != 'big, trends' or case_of(1, 0) != '1 tick, reverts' or case_of(5, 3) != 'big, partial':
        bad.append('cases')
    bd = np.array([50, 49, 47, 47, 47, 47, 47])
    if move_path(tfl, bd, valid, 0, 50, 1000.0) != (3, 1200.0, 3):                  # bid dump to 47
        bad.append(f'bid path {move_path(tfl, bd, valid, 0, 50, 1000.0)}')
    sg = [seg_into(p, t, 1, 50, 4, 1000.0, 1500.0) for p, t in ((50, 950.0), (52, 1100.0), (54, 1500.0), (53, 2000.0), (55, 1400.0))]
    if sg != ['into_old', 'into_runover', 'into_peak', 'into_after', None]:
        bad.append(f'into segments {sg}')
    sa = [seg_away(p, t, 0, 48, 1000.0, 1500.0) for p, t in ((49, 1100.0), (48, 1100.0), (47, 3000.0))]
    if sa != ['away_chase', 'away_old', 'away_after']:
        bad.append(f'away segments {sa}')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_PATH_SIGN', 'MUT_RUNOVER_PEAK', 'MUT_CHASE_REV'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'MOVE-CASES SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


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
    SIG = pd.Series(full.signer.values[ix].astype('int64'), index=LT.index)
    MP = pd.read_parquet(f'{PD.ORD}/map_{day}.parquet', columns=['row', 'maker', 'taker', 'maker_ok']).set_index('row')
    mk_ok = MP.maker_ok.eq(True).values & MP.maker.notna().values
    rows_mk = MP.index.values[mk_ok]
    addr2sig = dict(zip(MP.maker.values[mk_ok], full.signer.values[rows_mk].astype('int64')))
    TADDR = MP.taker.reindex(LT.index)
    gname = lambda s_: 'top 5' if G.get(int(s_), 'rest') == 'top 5' else 'field'

    def tgroup(a_):
        if not isinstance(a_, str):
            return 'unknown'
        s_ = addr2sig.get(a_)
        if s_ is None:
            return 'non-maker'
        return 'top 5' if G.get(int(s_), 'rest') == 'top 5' else 'other maker'
    PE = pd.read_parquet(f'{PD.OUTD}/events_{day}.parquet', columns=['eid', 'kind', 'side', 'p', 'tb'])
    st.update(window_sides=0, real_atbest_q={}, excluded=0, truncated=0, zero_move=0, ref_rows=0)
    evs = []; ev_rows = []; seg_rows = []; our_rows = []; ldg_chk = []; pend = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg = w['m'], w['side'], w['sg']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        runs = QR.runs(tf, P5, opp, side)
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        anc = w['anc']; ptl = w['ptl']; shl = w['shl']
        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)
        Lw = []
        for i, r_ in enumerate(lrow):
            if not src[i]:
                continue
            t = float(anc[i]); pt = int(ptl[i]); g_ = gname(SIG.at[r_])
            Lw.append((t, pt, float(shl[i]), g_, tgroup(TADDR.at[r_]), val(t, 1000.0, pt), val(t, 5000.0, pt)))
            if w['ws'] <= t < w['ws'] + 60000.0 and GP.depth_of(tfl, P5, opp, side, t, pt) == 0:
                st['real_atbest_q'][g_] = st['real_atbest_q'].get(g_, 0.0) + float(shl[i])
        ours = []
        for d in RUNGS:
            F_, O_, _ = K.sim_arm(w, 'M', 50.0, 'rest', aux=vg, offset=d, size=SIZE)
            for f in F_:
                i, pt, qq = f[0], int(f[1]), f[2]
                t = float(anc[i])
                if not (w['ws'] <= t < w['ws'] + 61000.0):
                    continue
                ldg_chk.append((float(qq), val(t, 1000.0, pt)))
                if src[i]:
                    ours.append((t, pt, float(qq), val(t, 5000.0, pt)))
        pend.setdefault(m, {})[side] = dict(w=w, vg=vg, runs=runs, tfl=tfl, L=Lw, ours=ours)
        w.pop('cache', None)
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
                        j, p, tb, te_run, po = c[0]; used.add((x, j)); E.append((f'{day}|{m}|{x}|{j}', kev, x, j, p, tb, po))
            for x in (0, 1):
                for kd, kev in ((2, 'calm fallback'), (1, 'calm improve')):
                    for j, p, tb, te_run, po in ST[(x, kd)]:
                        if (x, j) in used or not (wsm <= tb < wsm + 60000.0):
                            continue
                        lo = bisect.bisect_left(allon, tb - RC.WIN_POST)
                        if lo < len(allon) and allon[lo] < tb + RC.WIN_PRE:
                            continue
                        E.append((f'{day}|{m}|{x}|{j}', kev, x, j, p, tb, po))
            for eid, kev, x, j, p, tb, po in E:
                evs.append((eid, kev, x, p, tb))
                if kev not in ('adverse', 'calm fallback'):
                    continue
                b0x, validx, _, _ = pair[x]['runs']
                if not validx[j] or p < 0 or p == po:
                    st['excluded'] += 1; continue
                if tb > wsm + LAST_TB:
                    st['truncated'] += 1; continue
                M, t_pk, Fk = move_path(pair[x]['tfl'], b0x, validx, x, po, tb)
                if M < 1 or not (Fk == Fk):
                    st['zero_move'] += 1; continue
                cs = case_of(M, Fk); dr = 'UP' if x == 1 else 'DOWN'; xa = 1 - x
                b0a, valida, _, _ = pair[xa]['runs']; tla = pair[xa]['tfl']
                ka = bisect.bisect_left(tla, tb) - 1
                while ka >= 0 and not valida[ka]:
                    ka -= 1
                a0 = int(b0a[ka]) if ka >= 0 else -1
                ev_rows.append((int(day), kev, dr, cs, M, Fk, t_pk - tb))
                acc = {}
                for (t, pt, q, g_, tg_, v1, v5) in pair[x]['L']:
                    if t <= tb - PRE or t > t_pk + AFTER_MS:
                        continue
                    sgm = seg_into(pt, t, x, po, M, tb, t_pk)
                    if sgm is not None:
                        a = acc.setdefault((sgm, 'maker', g_), [0.0, 0.0, 0.0]); a[0] += q; a[1] += q * v5; a[2] += q * v1
                        tsg = 'take_with' if t <= t_pk else 'take_after'
                        a = acc.setdefault((tsg, 'taker', tg_), [0.0, 0.0, 0.0]); a[0] += q; a[1] -= q * v5; a[2] -= q * v1
                if a0 >= 0:
                    for (t, pt, q, g_, tg_, v1, v5) in pair[xa]['L']:
                        if t <= tb - PRE or t > t_pk + AFTER_MS:
                            continue
                        sgm = seg_away(pt, t, xa, a0, tb, t_pk)
                        if sgm is not None:
                            a = acc.setdefault((sgm, 'maker', g_), [0.0, 0.0, 0.0]); a[0] += q; a[1] += q * v5; a[2] += q * v1
                            if sgm == 'away_after':
                                a = acc.setdefault(('take_against', 'taker', tg_), [0.0, 0.0, 0.0]); a[0] += q; a[1] -= q * v5; a[2] -= q * v1
                for (sgm, role, g_), a in acc.items():
                    seg_rows.append((int(day), kev, dr, cs, sgm, role, g_, a[0], a[1], a[2]))
                oa = {}
                for (t, pt, q, v5) in pair[x]['ours']:
                    if tb - PRE < t <= t_pk + AFTER_MS:
                        sgm = seg_into(pt, t, x, po, M, tb, t_pk)
                        if sgm is not None:
                            o = oa.setdefault(sgm, [0.0, 0.0]); o[0] += q; o[1] += q * v5
                if a0 >= 0:
                    for (t, pt, q, v5) in pair[xa]['ours']:
                        if tb - PRE < t <= t_pk + AFTER_MS:
                            sgm = seg_away(pt, t, xa, a0, tb, t_pk)
                            if sgm is not None:
                                o = oa.setdefault(sgm, [0.0, 0.0]); o[0] += q; o[1] += q * v5
                for sgm, o in oa.items():
                    our_rows.append((int(day), kev, dr, cs, sgm, o[0], o[1]))
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    EV = pd.DataFrame(evs, columns=['eid', 'kind', 'side', 'p', 'tb'])
    st['events_differ'] = int(len(EV) != len(PE) or not (EV.eid.values.astype(str) == PE.eid.values.astype(str)).all()
                              or not (EV.p.values == PE.p.values).all() or not (EV.tb.values == PE.tb.values).all())
    R0 = pd.read_parquet(f'{FBD}/fills_{day}.parquet', columns=['cfg', 'arm', 'q', 'v1'])
    R0 = R0[(R0.arm == 'LDG') & (R0.cfg == 'M50')]
    st['ref_rows'] = len(R0)
    st['ldg_differ'] = int(len(R0) != len(ldg_chk) or not np.array_equal(R0.q.values, np.array([a for a, _ in ldg_chk]))
                           or not np.array_equal(R0.v1.values, np.array([b for _, b in ldg_chk]), equal_nan=True))
    st['unpaired_windows'] = len(pend)
    if st['events_differ'] or st['ldg_differ'] or st['unpaired_windows']:
        sys.exit(f'MOVE-CASES CONTROL FAILED on {day} -> no output: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)} }')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(ev_rows, columns=['day', 'kind', 'dir', 'case', 'M', 'F', 'pk_ms']).to_parquet(f'{OUTD}/events_{day}.parquet', index=False)
    pd.DataFrame(seg_rows, columns=['day', 'kind', 'dir', 'case', 'seg', 'role', 'grp', 'q', 'v5q', 'v1q']).to_parquet(f'{OUTD}/segs_{day}.parquet', index=False)
    pd.DataFrame(our_rows, columns=['day', 'kind', 'dir', 'case', 'seg', 'q', 'v5q']).to_parquet(f'{OUTD}/ours_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}; dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    rd = lambda nm: pd.concat([pd.read_parquet(f'{OUTD}/{nm}_{d}.parquet') for d in DAYS9], ignore_index=True)
    EV, SG, OU = rd('events'), rd('segs'), rd('ours')
    ab = {d: stats[d]['stat']['real_atbest_q'] for d in DAYS9}
    sh_ab = sum(ab[d].get('top 5', 0.0) for d in DAYS9) / max(sum(sum(ab[d].values()) for d in DAYS9), 1e-9)
    shd = {d: ab[d].get('top 5', 0.0) / max(sum(ab[d].values()), 1e-9) for d in DAYS9}
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    st_ = {k_: sum(stats[d]['stat'][k_] for d in DAYS9) for k_ in ('excluded', 'truncated', 'zero_move')}
    L = [f'MOVE-CASES, 9 consumed replay days, first 60 s, real WS legs; as of {asof}; controls passed (events = PULL-DEPTH, LDG = FUTURE-BEST,'
         f' planted errors caught); statuses {json.dumps(st_)}; top-5 share of all real at-best WS fill volume {sh_ab:.1%}', '']
    L.append('Cases (protection-side events after a signal | calm fallbacks): share of events, UP / DOWN; median ms to the peak')
    for cs in CASES:
        parts = []
        for kev in ('adverse', 'calm fallback'):
            x = EV[EV.kind == kev]
            parts.append(f'{(x.case == cs).mean():5.1%} ({(x[x.dir == "UP"].case == cs).mean():4.1%} / {(x[x.dir == "DOWN"].case == cs).mean():4.1%})')
        y = EV[(EV.kind == 'adverse') & (EV.case == cs)]
        L.append(f'  {cs:<16} {parts[0]} | {parts[1]} | to peak {np.median(y.pk_ms) if len(y) else NAN:5.0f} ms')
    mk = SG[SG.role == 'maker']; tk = SG[SG.role == 'taker']
    cps = lambda x: float(x.v5q.sum() / x.q.sum()) if len(x) and x.q.sum() else NAN
    for kev in ('adverse',):
        L += ['', f'MAKERS ({kev}): per case and segment: top-5 share of the fill volume | top 5 c/sh at 5 s | field c/sh at 5 s | ours (SIMULATED'
              f' LDG) shares a day, c/sh at 5 s   (usual top-5 share {sh_ab:.1%})']
        for cs in CASES:
            L.append(f'  {cs}')
            for sgm in MSEG:
                x = mk[(mk.kind == kev) & (mk.case == cs) & (mk.seg == sgm)]
                if not len(x):
                    continue
                t5 = x[x.grp == 'top 5']; fd = x[x.grp == 'field']; o = OU[(OU.kind == kev) & (OU.case == cs) & (OU.seg == sgm)]
                L.append(f'    {sgm:<13} {t5.q.sum() / max(x.q.sum(), 1e-9):5.1%} | {cps(t5):+.2f} | {cps(fd):+.2f} | ours {o.q.sum() / ndays:7.0f} sh/day'
                         f' {float(o.v5q.sum() / o.q.sum()) if o.q.sum() else NAN:+.2f}')
        L += ['', f'TAKERS ({kev}): per case and segment: share of taker volume top 5 / other makers / non-makers / unknown | top-5 taker c/sh at 5 s']
        for cs in CASES:
            for sgm in TSEG:
                x = tk[(tk.kind == kev) & (tk.case == cs) & (tk.seg == sgm)]
                if not len(x):
                    continue
                tot = max(x.q.sum(), 1e-9)
                L.append(f'  {cs:<16} {sgm:<12} ' + ' / '.join(f'{x.q[x.grp == g_].sum() / tot:5.1%}' for g_ in ('top 5', 'other maker', 'non-maker', 'unknown')) +
                         f' | {cps(x[x.grp == "top 5"]):+.2f}')
        L += ['', f'TOP-5 MAKER $ / day at 5 s by case ({kev}): into side | away side | total;   field total;   ours (SIMULATED) total']
        for cs in CASES:
            x = mk[(mk.kind == kev) & (mk.case == cs)]; t5 = x[x.grp == 'top 5']; fd = x[x.grp == 'field']; o = OU[(OU.kind == kev) & (OU.case == cs)]
            ins = t5[t5.seg.str.startswith('into')].v5q.sum() / 100.0 / ndays; aws = t5[t5.seg.str.startswith('away')].v5q.sum() / 100.0 / ndays
            L.append(f'  {cs:<16} {ins:+8.1f} | {aws:+8.1f} | {ins + aws:+8.1f};   {fd.v5q.sum() / 100.0 / ndays:+8.1f};   {o.v5q.sum() / 100.0 / ndays:+8.1f}')
    R = {}
    a = mk[(mk.kind == 'adverse')]
    x = a[(a.case == 'big, trends') & (a.seg == 'into_runover')]; sh = float(x.q[x.grp == 'top 5'].sum() / max(x.q.sum(), 1e-9)); lo = hi = 0
    for d in DAYS9:
        y = x[x.day == int(d)]
        if y.q.sum() >= MIN_N:
            v_ = float(y.q[y.grp == 'top 5'].sum() / y.q.sum()); lo += int(v_ < shd[d]); hi += int(v_ > shd[d])
    R['R540'] = dict(share=sh, usual=sh_ab, lower=lo, higher=hi, label='THIN-WHERE-RUN-OVER' if sh < sh_ab and lo >= 7 else ('THICK-WHERE-RUN-OVER' if sh > sh_ab and hi >= 7 else 'PROPORTIONAL'))
    x = a[(a.case == 'big, trends') & a.seg.isin(['into_old', 'into_runover', 'into_peak'])]; neg = up = dn = 0
    for d in DAYS9:
        y = x[x.day == int(d)]; t5 = y[y.grp == 'top 5']; fd = y[y.grp == 'field']
        if t5.q.sum() >= MIN_N and fd.q.sum() >= MIN_N:
            neg += int(cps(t5) < 0); up += int(cps(t5) > cps(fd)); dn += int(cps(t5) < cps(fd))
    R['R541'] = dict(top5=cps(x[x.grp == 'top 5']), field=cps(x[x.grp == 'field']), neg_days=neg, better_days=up, worse_days=dn,
                     label=('THEY-LOSE-THERE' if neg >= 7 else 'NOT-SHOWN') + ' / ' + ('LESS-THAN-THE-FIELD' if up >= 7 else ('MORE-THAN-THE-FIELD' if dn >= 7 else 'SIMILAR-TO-FIELD')))
    tw = tk[(tk.kind == 'adverse') & (tk.seg == 'take_with')]
    sh_b = lambda y: float(y.q[y.grp == 'top 5'].sum() / max(y.q.sum(), 1e-9))
    big = tw[tw.case == 'big, trends']; small = tw[tw.case == '1 tick, holds']; ratio = sh_b(big) / max(sh_b(small), 1e-9); hi = lo = 0
    for d in DAYS9:
        b_ = big[big.day == int(d)]; s_ = small[small.day == int(d)]
        if b_.q.sum() >= MIN_N and s_.q.sum() >= MIN_N:
            hi += int(sh_b(b_) > sh_b(s_)); lo += int(sh_b(b_) < sh_b(s_))
    R['R542'] = dict(big=sh_b(big), small=sh_b(small), ratio=ratio, higher=hi, lower=lo,
                     label='TAKE-INTO-BIG-MOVES' if ratio >= 1.2 and hi >= 7 else ('AVOID-TAKING-BIG-MOVES' if ratio <= 1 / 1.2 and lo >= 7 else 'SIMILAR'))
    for cs in ('big, reverts', 'big, trends'):
        x = a[(a.case == cs) & (a.seg == 'away_after')]
        R[f'R543 {cs}'] = dict(share=sh_b(x), top5_cps=cps(x[x.grp == 'top 5']), field_cps=cps(x[x.grp == 'field']))
    pos = neg = 0
    for d in DAYS9:
        v_ = a[(a.day == int(d)) & (a.grp == 'top 5')].v5q.sum()
        pos += int(v_ > 0); neg += int(v_ < 0)
    R['R544'] = dict(usd_day=float(a[a.grp == 'top 5'].v5q.sum()) / 100.0 / ndays, pos=pos, neg=neg,
                     label='NET-POSITIVE' if pos >= 7 else ('NET-NEGATIVE' if neg >= 7 else 'MIXED'))
    L += ['', 'READINGS:']
    for k_, v_ in R.items():
        L.append(f'  {k_}: {json.dumps(v_, default=float)}')
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/move_cases.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/move_cases.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


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
