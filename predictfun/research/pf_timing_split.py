"""TIMING-SPLIT (2026-11-13, session ce68ebe0; USER: "We can run it, it still looks we don't manage deep ladder well, the timing is also not
good"): FINDINGS C12 for the ladder studies - where exactly does the guard ladder lose, and does the loss sit on trades whose time is only
estimated (E2, +-250 ms) rather than WS-timed?

Simulator as DEEP-PULL (pf_deep_pull.py, ffc7e475): S4's rest lane, 9 consumed replay days, first 60 s, 30-sh orders kept through fills,
6 rungs per side. Arms: LDG (the guard trigger on every rung; the best ladder so far), LZ-none-30 (never pulled), J-v3 (5 sh). Fill clocks
M 0 / 50 / 250. Per simulated fill: the trade's anchor source (WS = its taker order has a WS-timed leg; E2 = estimated), rung, depth before
the trade (GP.depth_of), through our price or at it, order age at the fill (< 100 ms = young), filled while our cancel was pending, value at
1 s / 5 s (sg (mid(anchor + h) - price), the same function as DEPTH-TIMING), our P at the trade (FRONT-PULL's per-day model) in
DEPTH-TIMING's terciles. Real maker legs of the same windows (first minute), by group (top 5 / field), depth and anchor source, valued the
same way.
Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; LDG at M 50 unless stated; $/day w/o young = fills on orders >= 100 ms old):
 R450 SOURCE: the share of LDG's $/day w/o young loss that comes from E2-timed trades: >= 70% -> E2-DRIVEN; <= 30% -> WS-DRIVEN; else MIXED
      (if LDG is not negative: n/a). E2's share of LDG's shares printed beside.
 R451 WS-ONLY: LDG's $/day w/o young on WS-timed trades only: > 0 on >= 7 of 9 days -> WS-POSITIVE; < 0 on >= 7 -> WS-NEGATIVE; else NOT-SHOWN.
      Also at M 0 (printed).
 R452 REAL-BY-SOURCE (descriptive): the top 5's and the field's c/sh at 1 s on WS-timed and on E2-timed legs, at the best and behind it.
 R453 LOSS MAP (descriptive): LDG's $/day at 1 s by source x age (young / not) x pending (yes / no) x depth (0 / >= 1) x P tercile.
Controls (any failure -> no output): FRONT-PULL-SIZE's self-tests; the source flag on synthetic legs (a taker order with one WS-timed leg is
WS; none is E2); C1 at every rung; the sum of the loss-map cells equals each arm's total (1e-6); P aligned with FRONT-PULL's fit rows;
memory self-guard.
Limits: replay rule M (B1a); consumed days; the E2 anchor is an estimate, so E2 values carry its error for real legs too.
Usage (run_light, PF_SLOT=claude): pf_timing_split.py day <day> | report; driver run_timing_split.sh
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_front_pull as FP                                # noqa: E402
import pf_front_pull_size as FS                           # noqa: E402

DAYS9 = K.DAYS9
OUTT = f'{K.IO}/timing_split_20261113'
CFGS = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
TERC = (0.09040728596733005, 0.4236075378593026)
MEM_CAP_MIB = 2048.0


def ws_groups(how, grp):
    """Per leg: True if its taker order (grp) has a WS-timed leg (how > 0)."""
    return pd.Series(np.asarray(how) > 0).groupby(np.asarray(grp)).transform('any').values


def selftest():
    bad = []
    f = ws_groups([0, 1, 0, 0], [7, 7, 8, 8])
    if list(f) != [True, True, False, False]:
        bad.append(f'source flag {list(f)}')
    return bad


def day_run(day):
    import lightgbm as lgb
    import pf_gap_top_makers as GP
    FS.all_selftests(); b = selftest()
    if b:
        sys.exit(f'SELFTEST FAILED: {b}')
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)        # K.load_day's taker-order grouping (one anchor per group)
    WS = pd.Series(ws_groups(LT.how.values, grp), index=LT.index)
    SIG = pd.Series(full.signer.values[ix].astype('int64'), index=LT.index)
    fit = json.load(open(f'{FP.OUTF}/fit.json'))['days'][day]['lines']
    mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    z = np.load(f'{FP.MOVE_DIR}/day_{day}.npz'); zm = z['B|m'].astype(np.int64); zt = z['B|t']; zs = z['B|side'].astype(int)
    pz = np.load(f'{FP.OUTF}/p_rows_{day}.npy')
    ROWS = {(int(m_), int(s_)): (zt[ix_], pz[ix_]) for (m_, s_), ix_ in pd.DataFrame(dict(m=zm, s=zs)).groupby(['m', 's']).indices.items()}
    rows = []; st.update(window_sides=0, c1_differ=0, align_rows=0, align_bad=0, real=0, ours=0)
    nb = len(K.G32)
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']

        def mid_at(t):
            j = int(np.searchsorted(tf, t, side='right')) - 1
            return mff[j] if j >= 0 else np.nan
        Xw = np.column_stack([w['X'], np.full(n, float(side))]).astype(np.float32)
        P = mdl.predict(Xw) if n else np.zeros(0); vk = w['X'][:, nb] if n else np.zeros(0)
        P = np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)
        if (m, side) in ROWS:
            tz_, pz_ = ROWS[(m, side)]
            ii = np.searchsorted(w['tg'], tz_); ok = (ii < n) & (w['tg'][np.clip(ii, 0, n - 1)] == tz_)
            st['align_rows'] += len(tz_); st['align_bad'] += int((~ok).sum()) + int((~(np.abs(P[ii[ok]] - pz_[ok]) <= 1e-9)).sum())
        terc = lambda t: (lambda k: 'n/a' if k < 0 or not (P[k] == P[k]) else ('low' if P[k] <= TERC[0] else ('mid' if P[k] <= TERC[1] else 'high')))(
            int(np.searchsorted(w['tg'], t - 24.0, side='right')) - 1)
        if (m, side) not in D['LI']:
            continue
        l = D['LT'].iloc[D['LI'][(m, side)]]
        o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        src = WS.reindex(lrow).values
        for i, r_ in enumerate(lrow):                    # real legs of the first minute
            t = float(w['anc'][i])
            if not (w['ws'] <= t < w['ws'] + 60000.0):
                continue
            p_ = int(w['ptl'][i]); d_ = GP.depth_of(tfl, P5, opp, side, t, p_)
            if d_ is None or d_ < 0:
                continue
            rows.append((int(day), 'real', 'n/a', G.get(int(SIG.at[r_]), 'rest'), 'WS' if src[i] else 'E2', d_, -1, False, False, False,
                         float(w['shl'][i]), sg * (mid_at(t + 1000.0) - p_), sg * (mid_at(t + 5000.0) - p_), terc(t)))
            st['real'] += 1
        vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist()
        for q, shf in CFGS:
            tt = w['anc'] - shf
            runs = [('J-v3', 0, K.sim_arm(w, q, shf, 'J-v3'))]
            for d in FS.RUNGS:
                runs.append(('LDG', d, K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=FS.LAD_SIZE)))
                r0 = K.sim_arm(w, q, shf, 'rest', offset=d, size=FS.LAD_SIZE); runs.append(('LZ-none', d, r0))
                Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', decide=K.decide_record, offset=d, size=FS.LAD_SIZE)
                st['c1_differ'] += int(Fk != r0[0] or Ok != r0[1])
            for arm, d, (F, O, _) in runs:
                for f in F:
                    i, pt, qq, thr, age, ph = f[0], f[1], f[2], f[3], f[4], f[5]
                    t = float(w['anc'][i])                # valuation at the trade's anchor (unshifted), as S4 / DEPTH-TIMING
                    if not (w['ws'] <= t < w['ws'] + 60000.0 + 1000.0):
                        continue
                    d_ = GP.depth_of(tfl, P5, opp, side, t, int(pt))
                    rows.append((int(day), arm, f'{q}{int(shf)}', 'ours', 'WS' if src[i] else 'E2', -9 if d_ is None else d_, d, bool(thr), age < 100.0,
                                 ph == 'pend', float(qq), sg * (mid_at(t + 1000.0) - pt), sg * (mid_at(t + 5000.0) - pt), terc(t)))
                    st['ours'] += 1
        if st['window_sides'] % 80 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['c1_differ'] or st['align_bad'] or st['align_rows'] != len(zm):
        sys.exit(f'TIMING-SPLIT CONTROL FAILED: C1 {st["c1_differ"]}, alignment {st["align_bad"]} of {st["align_rows"]} ({len(zm)})')
    os.makedirs(OUTT, exist_ok=True)
    pd.DataFrame(rows, columns=['day', 'arm', 'cfg', 'grp', 'src', 'depth', 'rung', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'pt']).to_parquet(
        f'{OUTT}/fills_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTT}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


def report():
    A = pd.concat([pd.read_parquet(f'{OUTT}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    DY = {d: json.load(open(f'{OUTT}/day_{d}.json'))['days'] for d in DAYS9}; alld = sum(DY.values())
    A = A[np.isfinite(A.v1)].copy(); A['usd1'] = A.q * A.v1 / 100.0; A['usd5'] = A.q * A.v5 / 100.0
    cps = lambda x: float(x.usd1.sum() * 100.0 / x.q.sum()) if x.q.sum() else float('nan')
    usd = lambda x: float(x.usd1.sum() / alld)
    R = {}; L = []
    o = A[(A.grp == 'ours')]
    ldg = o[(o.arm == 'LDG') & (o.cfg == 'M50')]; ldg_wo = ldg[~ldg.young]
    cells = ldg.groupby(['src', 'young', 'pend', ldg.depth.clip(upper=1).rename('deep'), 'pt']).usd1.sum()
    if abs(cells.sum() - ldg.usd1.sum()) > 1e-6:
        sys.exit('TIMING-SPLIT REPORT CONTROL FAILED: loss-map cells do not add up')
    tot = usd(ldg_wo); e2 = usd(ldg_wo[ldg_wo.src == 'E2'])
    R['R450'] = 'n/a' if tot >= 0 else ('E2-DRIVEN' if e2 / tot >= 0.7 else ('WS-DRIVEN' if e2 / tot <= 0.3 else 'MIXED'))
    R['R450_detail'] = dict(total_wo=tot, e2_wo=e2, e2_loss_share=e2 / tot if tot else float('nan'),
                            e2_share_of_shares=float(ldg_wo.q[ldg_wo.src == 'E2'].sum() / ldg_wo.q.sum()))
    perd = []
    for d in DAYS9:
        x = ldg_wo[(ldg_wo.day == int(d)) & (ldg_wo.src == 'WS')]; perd.append(float(x.usd1.sum() / DY[d]))
    up = sum(v > 0 for v in perd); dn = sum(v < 0 for v in perd)
    R['R451'] = 'WS-POSITIVE' if up >= 7 else ('WS-NEGATIVE' if dn >= 7 else 'NOT-SHOWN')
    R['R451_detail'] = dict(ws_wo_M50=usd(ldg_wo[ldg_wo.src == 'WS']), per_day=perd,
                            ws_wo_M0=usd(o[(o.arm == 'LDG') & (o.cfg == 'M0') & ~o.young & (o.src == 'WS')]))
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L += [f'TIMING-SPLIT, 9 consumed replay days, first 60 s; as of {asof}; controls passed', 'Readings: ' + json.dumps(R, default=float), '',
          'ours: $/day at 1 s, all / w/o young, by trade timing (WS = exact, E2 = estimated); c/sh w/o young in brackets; share of shares']
    for arm in ('LDG', 'LZ-none', 'J-v3'):
        for cfg in ('M0', 'M50', 'M250'):
            x = o[(o.arm == arm) & (o.cfg == cfg)]
            seg = []
            for s_ in ('WS', 'E2'):
                y = x[x.src == s_]; yw = y[~y.young]
                seg.append(f'{s_} {usd(y):+9.2f} / {usd(yw):+9.2f} ({cps(yw):+.2f}) {y.q.sum() / max(x.q.sum(), 1e-9):.0%}')
            L.append(f'  {arm:<8} {cfg:<5} all {usd(x):+9.2f} / {usd(x[~x.young]):+9.2f} | ' + ' | '.join(seg))
    L += ['', 'R452 real legs, c/sh at 1 s (share of shares): at the best | behind it, by timing']
    rl = A[A.arm == 'real'].assign(who=lambda x: np.where(x.grp == 'top 5', 'top 5', 'field'))
    for who in ('top 5', 'field'):
        seg = []
        for dep, msk in (('best', rl.depth == 0), ('behind', rl.depth >= 1)):
            for s_ in ('WS', 'E2'):
                y = rl[(rl.who == who) & msk & (rl.src == s_)]
                seg.append(f'{dep} {s_} {cps(y):+.2f} ({y.q.sum() / max(rl[(rl.who == who) & msk].q.sum(), 1e-9):.0%})')
        L.append(f'  {who:<6} ' + ' | '.join(seg))
    L += ['', 'R453 LDG loss map at M 50, $/day at 1 s (c/sh), by timing x young x pending x depth (0 / >= 1); then by P tercile']
    for s_ in ('WS', 'E2'):
        for yg in (True, False):
            for pe in (True, False):
                for dp, msk in (('best', ldg.depth == 0), ('deep', ldg.depth >= 1)):
                    y = ldg[(ldg.src == s_) & (ldg.young == yg) & (ldg.pend == pe) & msk]
                    if y.q.sum() > 0:
                        L.append(f'  {s_} {"young" if yg else "aged":<5} {"pending" if pe else "live":<7} {dp:<4} {usd(y):+9.2f} ({cps(y):+.2f}) sh/day {y.q.sum() / alld:8.0f}')
    for s_ in ('WS', 'E2'):
        seg = []
        for tp in ('low', 'mid', 'high', 'n/a'):
            y = ldg_wo[(ldg_wo.src == s_) & (ldg_wo.pt == tp) & (ldg_wo.depth >= 1)]
            seg.append(f'{tp} {usd(y):+8.2f} ({cps(y):+.2f}, {y.q.sum() / alld:.0f} sh)')
        L.append(f'  deep w/o young {s_}: ' + ' | '.join(seg))
    for s_ in ('WS', 'E2'):
        seg = []
        for th in (False, True):
            y = ldg_wo[(ldg_wo.src == s_) & (ldg_wo.thr == th) & (ldg_wo.depth >= 1)]
            seg.append(f'{"through" if th else "at our price"} {usd(y):+8.2f} ({cps(y):+.2f})')
        L.append(f'  deep w/o young {s_}: ' + ' | '.join(seg))
    L += ['', 'R451 per day, LDG w/o young on WS-timed trades, M 50: ' + ' '.join(f'{v:+.1f}' for v in perd)]
    json.dump(dict(as_of=asof, readings=R), open(f'{OUTT}/timing_split.json', 'w'), indent=1, default=float)
    open(f'{OUTT}/timing_split.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
