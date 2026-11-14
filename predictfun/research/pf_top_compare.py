"""TOP-COMPARE (2026-11-14, session ce68ebe0): our current lanes against the top makers, POST HOC and descriptive (no verdicts).
USER: "It will be a much larger cap, besides, how is current metrics compared with top makers".

OURS (simulated): JOINT-LADDER v3's lanes read from its saved fills, no re-run: LDG (pull every rung on the guard), FB25 (keep rule), OFB2
(perfect foresight), both sides, rungs 0-5 x 30 sh, caps hard 10 and none (the USER: the real cap will be much larger), fill clocks M 0 / 50 /
250; fills on orders < 100 ms old excluded (C1), all-age figures printed beside.
THEIRS (real): every maker leg on the public tape in the same windows' first 60 s, makers ranked per day by shares in t_rel 0-240 s (GAP / B12's
rule): top 5, ranks 6-20, rest.
SAME POPULATION AND VALUE FOR BOTH: the legs aligned to the simulator's (GAP's control i: the window-side's sorted anchors equal the lane's);
depth = ticks behind the maker side's best in the frame strictly before the anchor (pf_gap_top_makers.depth_of, the function v3 used for our
fills; 0 = at the best, < 0 = inside the spread); value sg x (mid(anchor + h) - price), h = 1 s / 5 s, the same mid series as v3; WS / E2 = the
anchor source of the taker order (pf_timing_split.ws_groups, v3's src).
THEIRS ALSO split (descriptive): by our guard state at the leg (the guard ladder's guard v <= -0.10 at the last 50 ms step <= anchor - 50 ms:
ON = our ladder pulls, OFF = it rests; a step older than 100 ms = none) and by trade type (pf_depth_timing.trade_type over the taker order's
legs on that side: STOP = its worst price is this leg's, THROUGH = it filled a worse one; legs without a taker id are their own order).
Ours by trade type is not in v3's fills (no leg index): BEST-GAP's rung-0 split is the reference.
Controls (any failure -> no output): legs aligned on every window-side; the WS 1 s cells for our arms recomputed here equal v3's printed PART 1
(hard 10, none; 18 cells); synthetic cases for the guard-state reader with planted errors (no lag; no staleness limit); upstream selftests
(GAP's depth classifier, the trade type, the WS groups).
Limits: ours simulated (rule M unvalidated, B1a; replay volumes inflated, C4), theirs real (C9); consumed days (C8); first minute only; signer
ids per day (our own account, if it traded, is among the ranked makers); fees and rebates not counted.
Usage (run_light, PF_SLOT=claude): pf_top_compare.py selftest | day <day> | report; driver run_top_compare.sh
"""
import bisect
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_timing_split as TS                              # noqa: E402
import pf_gap_top_makers as GP                            # noqa: E402
import pf_depth_timing as DT                              # noqa: E402
import pf_joint_ladder_v3 as J3                           # noqa: E402

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/top_compare_20261114'
LAG = 50.0; MAX_AGE = 100.0
OUR_ARMS = ('LDG', 'FB25', 'OFB2'); OUR_CAPS = ('hard10', 'none'); CFGS = ('M0', 'M50', 'M250')
GROUPS = ('top 5', '6-20', 'rest')
MEM_CAP_MIB = 2048.0
MUT_NOLAG = MUT_NOSTALE = 0                               # planted errors (selftest only)


def dclass(d):
    if d is None or d == -9:
        return 'no frame'
    return 'inside' if d < 0 else ('at best' if d == 0 else ('1 behind' if d == 1 else '2+ behind'))


def guard_state(tgl, vg, anc):
    """Our guard ladder's state at a leg: the last 50 ms step <= anc - 50 ms (B12's lag); 'none' if that step is > 100 ms old."""
    t_ = anc - (0.0 if MUT_NOLAG else LAG)
    g = bisect.bisect_right(tgl, t_) - 1
    if g < 0 or (not MUT_NOSTALE and t_ - tgl[g] > MAX_AGE):
        return 'none'
    return 'ON' if vg[g] else 'OFF'


def selftest():
    bad = []
    tgl = [0.0, 50.0, 100.0, 150.0]; vg = [False, False, True, True]
    cases = [(120.0, 'OFF'), (160.0, 'ON'), (40.0, 'none'), (370.0, 'none')]
    got = [guard_state(tgl, vg, a) for a, _ in cases]
    if got != [c for _, c in cases]:
        bad.append(f'guard state {got}')
    if [dclass(x) for x in (None, -1, 0, 1, 3)] != ['no frame', 'inside', 'at best', '1 behind', '2+ behind']:
        bad.append('depth class')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_NOLAG', 'MUT_NOSTALE'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    b = list(GP.selftest() or []) + selftest() + DT.selftest() + TS.selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'TOP-COMPARE SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'top-compare selftests passed; planted errors caught: {m}', flush=True)


def day_run(day):
    all_selftests()
    T0 = time.time(); D = K.load_day(day); S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tkf) != len(full):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    SIG = full.signer.values.astype('int64'); del full
    LT = D['LT']; ix_all = LT.index.values
    TKG = np.where(tkf[ix_all] >= 0, tkf[ix_all], -1 - ix_all)
    WSF = pd.Series(TS.ws_groups(LT.how.values, TKG), index=LT.index)
    TKS = pd.Series(TKG, index=LT.index)
    st = dict(window_sides=0, align_differ=0, legs=0, no_frame=0, nan1=0, no_taker_id=0)
    acc = {}; per_maker = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg = w['m'], w['side'], w['sg']
        if (m, side) not in D['LI']:
            continue
        l = LT.iloc[D['LI'][(m, side)]]
        o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        st['align_differ'] += int(not np.array_equal(l.anc.values[o_], w['anc']))
        tf, mff = w['tf_mff']; tfl = tf.tolist(); tgl = w['tg'].tolist()
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        tk = TKS.reindex(lrow).values; wsf = WSF.reindex(lrow).values.astype(bool)
        prices = {}
        for t_, p_ in zip(tk, w['ptl']):
            prices.setdefault(int(t_), []).append(int(p_))
        for i in range(len(lrow)):
            a = float(w['anc'][i])
            if not (w['ws'] <= a < w['te']):
                continue
            st['legs'] += 1; p = int(w['ptl'][i]); q = float(w['shl'][i])
            dc = dclass(GP.depth_of(tfl, w['P5'], w['opp'], side, a, p))
            if dc == 'no frame':
                st['no_frame'] += 1; continue
            v1 = sg * (mff[max(int(np.searchsorted(tf, a + 1000.0, side='right')) - 1, 0)] - p)
            v5 = sg * (mff[max(int(np.searchsorted(tf, a + 5000.0, side='right')) - 1, 0)] - p)
            if not np.isfinite(v1):
                st['nan1'] += 1; continue
            st['no_taker_id'] += int(tk[i] < 0)
            g_ = G.get(int(SIG[lrow[i]]), 'rest'); src = 'WS' if wsf[i] else 'E2'
            gs = guard_state(tgl, vg, a); typ = DT.trade_type(side, p, prices[int(tk[i])])
            e = acc.setdefault((g_, dc, src, gs, typ), [0, 0.0, 0.0, 0.0])
            e[0] += 1; e[1] += q; e[2] += q * v1; e[3] += q * (v5 if np.isfinite(v5) else 0.0)
            if g_ == 'top 5':
                pm = per_maker.setdefault((int(SIG[lrow[i]]), src), [0.0, 0.0]); pm[0] += q; pm[1] += q * v1
        w.pop('cache', None)
        if st['window_sides'] % 96 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['align_differ']:
        sys.exit(f'TOP-COMPARE CONTROL FAILED on {day} (legs not aligned) -> no output: {st}')
    rows = [dict(day=int(day), who=k[0], depth=k[1], src=k[2], guard=k[3], type=k[4], legs=v[0], q=v[1], v1=v[2], v5=v[3]) for k, v in acc.items()]
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f'{OUTD}/theirs_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0, per_maker=[[s_, src, v[0], v[1]] for (s_, src), v in per_maker.items()]),
              open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


def report():
    all_selftests()
    S = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    ndays = sum(S[d]['days'] for d in DAYS9)
    T = pd.concat([pd.read_parquet(f'{OUTD}/theirs_{d}.parquet') for d in DAYS9], ignore_index=True)
    O = pd.concat([pd.read_parquet(f'{J3.OUTD}/fills_{d}.parquet', columns=['day', 'cfg', 'arm', 'capk', 'src', 'depth', 'young', 'q', 'v1', 'v5'])
                   for d in DAYS9], ignore_index=True)
    O = O[O.arm.isin(OUR_ARMS) & O.capk.isin(OUR_CAPS) & np.isfinite(O.v1)]
    O['dc'] = [dclass(int(x)) for x in O.depth.values]
    # reconciliation: our WS 1 s w/o young cells equal v3's printed PART 1
    txt = open(f'{J3.OUTD}/joint_ladder_v3.txt').read(); bad = []; nrec = 0
    for line in txt.splitlines():
        mm = re.match(r'\s+(M0|M50|M250)\s+(LDG|FB25|OFB2)\s+(\S+)\s+([+-][\d.]+) \|', line)
        if not mm or mm.group(3) not in OUR_CAPS:
            continue
        x = O[(O.cfg == mm.group(1)) & (O.arm == mm.group(2)) & (O.capk == mm.group(3)) & (O.src == 'WS') & ~O.young]
        nrec += 1
        if abs(float((x.q * x.v1).sum()) / 100.0 / ndays - float(mm.group(4))) > 0.051:
            bad.append(mm.groups())
    if bad or nrec != len(CFGS) * len(OUR_ARMS) * len(OUR_CAPS):
        sys.exit(f'TOP-COMPARE RECONCILIATION FAILED -> no output: {nrec} cells, mismatches {bad}')

    def ours(arm, cn, cfg, src=None, aged=True, dc=None):
        x = O[(O.arm == arm) & (O.capk == cn) & (O.cfg == cfg)]
        if src:
            x = x[x.src == src]
        if aged:
            x = x[~x.young]
        if dc:
            x = x[x.dc == dc]
        q = float(x.q.sum()); v1 = float((x.q * x.v1).sum()); v5 = float((x.q * x.v5).sum())
        return q / ndays, v1 / 100.0 / ndays, (v1 / q if q > 0 else float('nan')), (v5 / q if q > 0 else float('nan'))

    def theirs(who, src=None, dc=None, guard=None, typ=None):
        x = T[T.who == who]
        if src:
            x = x[x.src == src]
        if dc:
            x = x[x.depth == dc]
        if guard:
            x = x[x.guard == guard]
        if typ:
            x = x[x.type == typ]
        q = float(x.q.sum()); v1 = float(x.v1.sum()); v5 = float(x.v5.sum())
        return q / ndays, v1 / 100.0 / ndays, (v1 / q if q > 0 else float('nan')), (v5 / q if q > 0 else float('nan'))
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    st_ = {k_: sum(S[d]['stat'][k_] for d in DAYS9) for k_ in ('legs', 'no_frame', 'nan1', 'no_taker_id')}
    L = [f'TOP-COMPARE (POST HOC, descriptive): our JOINT-LADDER v3 lanes (simulated) against the real makers, 9 consumed days, first 60 s; as of '
         f'{asof}; legs aligned on every window-side; our cells reconcile with v3 ({nrec} cells); their legs {st_["legs"]} (no frame {st_["no_frame"]}, '
         f'no 1 s mid {st_["nan1"]}, without a taker id {st_["no_taker_id"]})', '',
         'PART 1: per day. sh/day | $/day at 1 s | c/sh 1 s | c/sh 5 s || WS-timed: sh/day | $/day 1 s | c/sh 1 s   (ours: fills on orders < 100 ms '
         'old excluded; [all ages: WS $/day, c/sh])']
    for who in GROUPS:
        a, b = theirs(who), theirs(who, 'WS')
        L.append(f'  {who + " (real)":<22} {a[0]:10.0f} | {a[1]:+8.1f} | {a[2]:+6.3f} | {a[3]:+6.3f} || {b[0]:9.0f} | {b[1]:+8.1f} | {b[2]:+6.3f}')
    a, b = theirs('top 5'), theirs('top 5', 'WS')
    L.append(f'  {"top 5, per maker":<22} {a[0] / 5:10.0f} | {a[1] / 5:+8.1f} | {a[2]:+6.3f} | {a[3]:+6.3f} || {b[0] / 5:9.0f} | {b[1] / 5:+8.1f} | {b[2]:+6.3f}')
    for cfg in CFGS:
        for cn in OUR_CAPS:
            for arm in OUR_ARMS:
                a, b, c = ours(arm, cn, cfg), ours(arm, cn, cfg, 'WS'), ours(arm, cn, cfg, 'WS', aged=False)
                L.append(f'  {"ours " + arm + " " + cn + " " + cfg:<22} {a[0]:10.0f} | {a[1]:+8.1f} | {a[2]:+6.3f} | {a[3]:+6.3f} || {b[0]:9.0f} | '
                         f'{b[1]:+8.1f} | {b[2]:+6.3f}   [{c[1]:+7.1f}, {c[2]:+6.3f}]')
        L.append('')
    L.append('PART 2: by depth at the fill, WS-timed, 1 s: share of the group\'s shares | c/sh | $/day   (ours: M 50, w/o young)')
    for dc in ('at best', '1 behind', '2+ behind', 'inside'):
        parts = []
        for who in GROUPS:
            tot = theirs(who, 'WS')[0]; x = theirs(who, 'WS', dc)
            parts.append(f'{who} {x[0] / tot:5.1%} {x[2]:+6.3f} {x[1]:+7.1f}')
        for arm in OUR_ARMS:
            for cn in OUR_CAPS:
                tot = ours(arm, cn, 'M50', 'WS')[0]; x = ours(arm, cn, 'M50', 'WS', dc=dc)
                parts.append(f'{arm} {cn} {x[0] / max(tot, 1e-9):5.1%} {x[2]:+6.3f} {x[1]:+7.1f}')
        L.append(f'  {dc:<10}: ' + ' | '.join(parts))
    L += ['', 'PART 3: theirs by OUR guard ladder\'s state at the leg (lag 50 ms; ON = our ladder pulls, OFF = it rests), WS-timed, 1 s: share of '
          'their shares | c/sh | $/day']
    for who in GROUPS:
        tot = theirs(who, 'WS')[0]
        L.append(f'  {who:<6}: ' + ' | '.join(f'{gs} {theirs(who, "WS", guard=gs)[0] / tot:5.1%} {theirs(who, "WS", guard=gs)[2]:+6.3f} '
                                              f'{theirs(who, "WS", guard=gs)[1]:+7.1f}' for gs in ('ON', 'OFF', 'none')))
    L += ['', 'PART 4: theirs by trade type at each depth, WS-timed, 1 s: STOP share | STOP c/sh | THROUGH c/sh']
    for who in GROUPS:
        parts = []
        for dc in ('at best', '1 behind', '2+ behind'):
            t_ = theirs(who, 'WS', dc)[0]; s_ = theirs(who, 'WS', dc, typ='STOP'); h_ = theirs(who, 'WS', dc, typ='THROUGH')
            parts.append(f'{dc} {s_[0] / max(t_, 1e-9):5.1%} {s_[2]:+6.3f} {h_[2]:+6.3f}')
        L.append(f'  {who:<6}: ' + ' | '.join(parts))
    pm = pd.DataFrame([r for d in DAYS9 for r in S[d]['per_maker']], columns=['signer', 'src', 'q', 'v1'])
    L += ['', f'top-5 maker-days: {pm.signer.nunique()} signer ids over 9 days (ids are per day)']
    open(f'{OUTD}/top_compare.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_top_compare.py selftest | day <day> | report')
