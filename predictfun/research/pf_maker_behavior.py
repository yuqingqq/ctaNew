"""How do the top makers reach the front? Behaviour profile from their fills (declared in STATUS.yml 2026-11-03 22:22Z; user:
"we check their behavior first"). Descriptive, 8 read days, nothing selected.

Groups: makers ranked per day by live-window volume (t_rel 0-240 s; ids are per day) in bands top 5 / 6-20 / rest, and our
replay fills: the ladder arm of lad1q (K 3, z 5, S 60, TH 0.5) and the single quote S60 + CAP10 (round LC). Per fill (timed
legs only; 1 s-stamped legs and fills with no frame before, an empty side or inside the spread are statuses):
  depth      ticks behind the best on the maker's side in the frame strictly before the fill (0 = at best)
  level age  time since that price last had no resting size on that side (first frame of the current run); 'since first
             sight' if it rested from the market's first frame; 'censored' if the price left the 5 shown levels in between;
             'not resting' if the frame before shows no size there (for our fills: we would have been alone at that price)
  created    at-best fills: was the price already the best when its run began (posted as a new best = price improvement), or
             posted behind and reached when the better levels emptied
  binance    BTCUSDT mid move over the 1 s before the fill, signed for the maker (negative = the move was against the maker)
  net 1 s    spread + informed, c/sh: s (m0 - p) + s (m1 - m0), venue mids strictly before and 1 s after
Per maker-window (all live legs 0-300 s): leftover |net| / gross, P&L marked at the window's last venue mid vs carry to
settlement. Usage: python3 -m predictfun.research.pf_maker_behavior          (selftest: ... --selftest)
"""
import os, sys
import numpy as np, pandas as pd

AGE_EDGES = np.array([100.0, 1000.0, 10000.0, 60000.0]); AGE_LAB = ['<0.1 s', '0.1-1 s', '1-10 s', '10-60 s', '>60 s']
BN_EDGES = np.array([-2.0, -0.5, 0.5, 2.0]); BN_LAB = ['against >=2bp', 'against 0.5-2', 'flat', 'with 0.5-2', 'with >=2bp']


def level_status(P, S, side):
    """status[k, p], p = 0..100: 1 resting size > 0; 0 inside the shown range and empty; -1 beyond the 5 shown levels."""
    n = len(P); st = np.zeros((n, 101), np.int8); pr = np.arange(101)[None, :]; last = P[:, 4][:, None]
    st[(last >= 0) & ((pr < last) if side == 0 else (pr > last))] = -1
    for i in range(5):
        ok = (P[:, i] >= 0) & (S[:, i] > 0)
        st[np.nonzero(ok)[0], P[ok, i]] = 1
    return st


def run_start(st):
    """First frame of the current run of resting size at each (frame, price), and whether a frame with the price beyond the
    shown levels lies after the last empty frame (age censored)."""
    idx = np.arange(st.shape[0])[:, None]
    la = np.maximum.accumulate(np.where(st == 0, idx, -1), axis=0)
    lu = np.maximum.accumulate(np.where(st == -1, idx, -1), axis=0)
    return la + 1, lu > la


def classify(tf, P, st, rs, cz, ta, p, side):
    """Per fill: usable flag, depth, age label, created-as-best (1/0, -1 n/a) from the frame strictly before each fill time ta.
    Not usable: no frame before, the side empty, or the fill inside the spread (negative depth)."""
    j0 = np.searchsorted(tf, ta, side='left') - 1; ok = j0 >= 0; j = np.clip(j0, 0, None)
    best = P[j, 0]
    depth = np.where(best >= 0, (best - p) if side == 0 else (p - best), -1)
    s = st[j, p]; start = rs[j, p]; cens = cz[j, p]
    age = ta - tf[np.clip(start, 0, len(tf) - 1)]
    lab = np.array(AGE_LAB, dtype=object)[np.searchsorted(AGE_EDGES, age, side='right')]
    lab = np.where(s != 1, 'not resting', np.where(cens, 'censored', np.where(start == 0, 'since first sight', lab)))
    created = np.where((s == 1) & (depth == 0), (P[np.clip(start, 0, len(tf) - 1), 0] == p).astype(int), -1)
    return ok & (best >= 0) & (depth >= 0), depth, lab, created, start


def selftest():
    # frames: bids; price 50 rests at k 0-2, empty at k 3 (best 49), back at k 4-5 as the new best
    P = np.array([[51, 50, 49, 48, 47], [51, 50, 49, 48, 47], [50, 49, 48, 47, 46], [49, 48, 47, 46, 45], [50, 49, 48, 47, 46], [50, 49, 48, 47, 46]])
    S = np.ones((6, 5)); tf = np.array([0., 100., 200., 300., 400., 500.])
    st = level_status(P, S, 0); rs, cz = run_start(st)
    assert st[3, 50] == 0 and st[0, 50] == 1 and st[3, 44] == -1 and st[0, 52] == 0
    ok, dep, lab, cr, _ = classify(tf, P, st, rs, cz, np.array([450.0, 550.0, 250.0]), np.array([50, 50, 50]), 0)
    # 1) a fill at 450 sees frame 4: run began at frame 4 (t 400) -> age 50 ms, posted as a new best
    assert lab[0] == '<0.1 s' and cr[0] == 1 and dep[0] == 0, (lab, cr)
    # 2) at 550 (frame 5): age 150 ms; 3) at 250 (frame 2): resting since the first frame, reached by depletion of 51
    assert lab[1] == '0.1-1 s' and lab[2] == 'since first sight' and cr[2] == 0, (lab, cr)
    # 4) censored: price 46 beyond the shown levels at frame 1, then shown -> age unknown
    P2 = np.array([[50, 49, 48, 47, 46], [50, 49, 48, 47, 47], [50, 49, 48, 47, 46]]); st2 = level_status(P2, np.ones((3, 5)), 0); rs2, cz2 = run_start(st2)
    _, _, lab2, _, _ = classify(np.array([0., 100., 200.]), P2, st2, rs2, cz2, np.array([250.0]), np.array([46]), 0)
    assert st2[1, 46] == -1 and lab2[0] == 'censored', lab2
    # 5) asks mirror: beyond = above the 5th ask; 6) a fill where the frame before shows no size -> 'not resting'
    A = np.array([[52, 53, 54, 55, 56]]); sa = level_status(A, np.ones((1, 5)), 1)
    assert sa[0, 57] == -1 and sa[0, 51] == 0 and sa[0, 54] == 1
    _, _, lab3, _, _ = classify(tf, P, st, rs, cz, np.array([350.0]), np.array([50]), 0)
    assert lab3[0] == 'not resting', lab3
    ok4, _, _, _, _ = classify(tf, P, st, rs, cz, np.array([350.0]), np.array([50]), 0)   # bid 50 above the best bid 49: inside
    assert not ok4[0]
    print('selftest PASS (7 cases)')


if __name__ == '__main__' and '--selftest' in sys.argv:
    selftest(); sys.exit(0)

from . import pf_io as io

DAYS = os.environ.get('PF_MB_DAYS', '20261023,20261024,20261025,20261026,20261027,20261031,20261101,20261102').split(','); FRESH = ('20261101', '20261102')
S4 = f'{io.OUT}/step4'; OUTD = f"{S4}/maker_behavior{os.environ.get('PF_MB_SFX', '')}"; os.makedirs(OUTD, exist_ok=True)
COLS = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
OURS = [('ours: ladder', lambda d: f"{S4}/ladder_{'preds_c3' if d in FRESH else 'preds_o3'}_lad1q/{d}/fills/K3_z5_S60_TH0.5_CAP50_BUD75_RF1000_THIN30_CD500.parquet"),
        ('ours: single quote', lambda d: f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d in FRESH else 'preds_o3'}_lc/{d}/fills/"
                                         "MICRO_G4_G2_G6_G5_G3_G1_M025_XNA_XMIN1_XS0_XV_S60_CAP10__mid_szA1-5-25-0.parquet")]


def load_bn(day):
    ks, ms = [], []
    for h in io.hours_of(day, extra_after=1, extra_before=1):
        p_ = f'{io.BN_DIR}/{h}.csv.gz'
        if not os.path.exists(p_):
            continue
        df = pd.read_csv(p_, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'bid', 'ask'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)]
        k = df.E.values.astype(np.float64); m = (df.bid.values + df.ask.values) / 2.0; del df
        o = np.argsort(k, kind='stable'); k, m = k[o], m[o]; keep = np.r_[True, m[1:] != m[:-1]]
        ks.append(k[keep]); ms.append(m[keep])
    K = np.concatenate(ks); M = np.concatenate(ms); o = np.argsort(K, kind='stable')
    return K[o], M[o]


def bn_move(K, M, t):
    i1 = np.searchsorted(K, t, side='right') - 1; i0 = np.searchsorted(K, t - 1000.0, side='right') - 1
    ok = (i0 >= 0) & (i1 >= 0)
    return np.where(ok, 1e4 * np.log(M[np.clip(i1, 0, None)] / M[np.clip(i0, 0, None)]), np.nan)


AGG, INV, FRONT, st = [], [], [], dict(legs_1s_only=0, legs_unusable=0, ours_unusable=0, windows=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
    pc = [f'{s}{i}_p' for s in 'ba' for i in range(1, 6)]
    if not (np.all(bk[pc].values == np.round(bk[pc].values)) and bk[pc].values.min() >= -1 and bk[pc].values.max() <= 100):
        raise SystemExit(f'{d}: book prices are not integer cents in [-1, 100]; refusing (level_status indexes by price)')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'we', 'outcome']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    lt = lt[lt.mid.isin(reg.index) & (lt.t_rel >= 0) & (lt.t_rel < 300)].copy()
    lt['timed'] = (lt.how.values > 0) | np.isfinite(lt.t_e2.values)
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, np.where(np.isfinite(lt.t_e2.values), lt.t_e2.values, lt.exec_s.values * 1000.0))
    vol = lt[lt.t_rel < 240].groupby('signer').shares.sum().sort_values(ascending=False); rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = lt.signer.map(rk).fillna(1e9).values; lt['grp'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest'))
    st['legs_1s_only'] += int((~lt.timed & (lt.t_rel < 240)).sum())
    ours = []
    for g_, f_ in OURS:
        f = pd.read_parquet(f_(d), columns=['mid', 't', 'side', 'pt', 'q'])
        ws_ = reg.loc[f.mid.values].ws.values * 1000.0
        ours.append(pd.DataFrame(dict(mid=f.mid.values, side=f.side.values, ptick=f.pt.values.astype(int), shares=f.q.values, ta=ws_ + f.t.values * 1000.0,
                                      t_rel=f.t.values, timed=True, grp=g_, signer=-1)))
    U = pd.concat([lt[['mid', 'side', 'ptick', 'shares', 'ta', 't_rel', 'timed', 'grp', 'signer']]] + ours, ignore_index=True)
    K, M = load_bn(d)
    rows = []; UB = {m: x for m, x in U.groupby('mid', sort=False)}
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        tf = g.t_ms.values.astype(np.float64)
        b1 = g.b1_p.values; a1 = g.a1_p.values; vm = (b1 >= 0) & (a1 > b1); mids = np.where(vm, (b1 + a1) / 2.0, np.nan)
        tv = tf[vm]; mv = mids[vm]
        u = UB.get(mid)
        if u is None:
            continue
        st['windows'] += 1
        R = reg.loc[mid]; y = 100.0 if R.outcome == 'Up' else (0.0 if R.outcome == 'Down' else np.nan)
        we = float(R.we) * 1000.0; ie = np.searchsorted(tv, we, side='left') - 1; m_end = mv[ie] if ie >= 0 else np.nan
        for side in (0, 1):
            pre = 'b' if side == 0 else 'a'
            P = g[[f'{pre}{i}_p' for i in range(1, 6)]].values.astype(int); S = g[[f'{pre}{i}_s' for i in range(1, 6)]].values
            stt = level_status(P, S, side); rs, cz = run_start(stt)
            us = u[(u.side.values == side) & u.timed.values]
            if not len(us):
                continue
            ta = us.ta.values.astype(np.float64); p = us.ptick.values.astype(int)
            ok, dep, lab, cr, srt = classify(tf, P, stt, rs, cz, ta, p, side)
            i0 = np.searchsorted(tv, ta, side='left') - 1; i1 = np.searchsorted(tv, ta + 1000.0, side='right') - 1
            okm = (i0 >= 0) & (i1 >= 0); m0 = np.where(okm, mv[np.clip(i0, 0, None)], np.nan); m1 = np.where(okm, mv[np.clip(i1, 0, None)], np.nan)
            sg = 1.0 if side == 0 else -1.0
            rows.append(pd.DataFrame(dict(grp=us.grp.values, w60=us.t_rel.values < 60, w240=us.t_rel.values < 240, q=us.shares.values, ok=ok & okm,
                                          depth=np.clip(dep, 0, 2), age=lab, created=cr, bn=sg * bn_move(K, M, ta),
                                          net1s=sg * (m0 - p) + sg * (m1 - m0))))
            # front of the queue: the first trade at a price in each run of resting size (known start, market legs only);
            # in a FIFO queue its maker legs are the orders at the very front
            mk = ok & ~us.grp.str.startswith('ours').values & ~np.isin(lab, ['censored', 'not resting'])
            if mk.any():
                Fm = pd.DataFrame(dict(p=p[mk], start=srt[mk], ta=ta[mk], q=us.shares.values[mk], grp=us.grp.values[mk], age=lab[mk], depth=np.clip(dep[mk], 0, 2),
                                       w60=us.t_rel.values[mk] < 60, w240=us.t_rel.values[mk] < 240))
                first = Fm.groupby(['p', 'start']).ta.transform('min')
                FRONT.append(Fm[Fm.ta == first].assign(day=d))
        # per maker-window inventory (all live legs, incl. 1 s-stamped; ours: their own fills)
        sg = np.where(u.side.values == 0, 1.0, -1.0); q = u.shares.values; px = u.ptick.values / 100.0
        W = pd.DataFrame(dict(grp=u.grp.values, signer=np.where(u.grp.str.startswith('ours').values, u.grp.values, u.signer.astype(str).values),
                              net=sg * q, gross=q, cash=-sg * q * px))
        W = W.groupby(['grp', 'signer']).sum().reset_index()
        W['mtm'] = W.cash + W.net * m_end / 100.0; W['carry'] = W.net * (y - m_end) / 100.0; W['day'] = d; W['mid'] = mid
        INV.append(W[['day', 'mid', 'grp', 'net', 'gross', 'mtm', 'carry']])
    F = pd.concat(rows, ignore_index=True); st['legs_unusable'] += int((~F.ok & ~F.grp.str.startswith('ours') & F.w240).sum()); st['ours_unusable'] += int((~F.ok & F.grp.str.startswith('ours')).sum())
    F = F[F.ok]; F['bnbin'] = np.where(np.isnan(F.bn), 'no binance', np.array(BN_LAB, dtype=object)[np.searchsorted(BN_EDGES, np.nan_to_num(F.bn), side='right')])
    F['qn'] = F.q * F.net1s
    for part in ('w60', 'w240'):
        a = F[F[part]].groupby(['grp', 'depth', 'age', 'created', 'bnbin'], dropna=False)[['q', 'qn']].sum().reset_index(); a['part'] = part; a['day'] = d
        AGG.append(a)
    del bk, lt, U, F, K, M
    io.mem_check(f'day {d}')

A = pd.concat(AGG, ignore_index=True); I = pd.concat(INV, ignore_index=True); FR = pd.concat(FRONT, ignore_index=True)
FR.to_parquet(f'{OUTD}/front_fills.parquet', index=False)
A.to_parquet(f'{OUTD}/fills_agg.parquet', index=False); I.to_parquet(f'{OUTD}/maker_windows.parquet', index=False)
GR = ['top 5', '6-20', 'rest', 'ours: ladder', 'ours: single quote']
print(f'statuses {st}')
for part, lab in (('w60', 'fills in the first 60 s (our arms quote only then)'), ('w240', 'fills 0-240 s (makers only; ours are all < 60 s)')):
    a = A[A.part == part]; out = {}
    print(f'\n== {lab}; share of each group\'s timed volume; net 1 s in c/sh')
    print(f"{'group':20s} {'vol k':>7s} {'best':>5s} {'1bk':>5s} {'2+':>5s} | {'age<0.1':>7s} {'0.1-1':>6s} {'1-10':>6s} {'10-60':>6s} {'>60':>5s} {'1st':>5s} {'cens':>5s} {'notrst':>6s} | "
          f"{'new best':>8s} | {'bn ag2':>6s} {'bn w2':>6s} | {'net1s':>6s} {'ag2 net':>7s} {'<1s net':>7s} {'>10s net':>8s}")
    for gname in GR:
        g = a[a.grp == gname]
        if not len(g):
            continue
        v = g.q.sum(); sh = lambda m: g.q[m].sum() / v
        best = g[g.depth == 0]; nb = best.q[best.created == 1].sum() / max(best.q[best.created >= 0].sum(), 1e-9)
        net = lambda m: g.qn[m].sum() / max(g.q[m].sum(), 1e-9)
        young = g.age.isin(['<0.1 s', '0.1-1 s']); old = g.age.isin(['10-60 s', '>60 s', 'since first sight'])
        r = dict(vol_k=v / 1e3, best=sh(g.depth == 0), back1=sh(g.depth == 1), back2=sh(g.depth == 2), **{f'age {x}': sh(g.age == x) for x in AGE_LAB + ['since first sight', 'censored', 'not resting']},
                 new_best=nb, bn_against2=sh(g.bnbin == BN_LAB[0]), bn_with2=sh(g.bnbin == BN_LAB[4]), net1s=net(g.q > -1), net_against2=net(g.bnbin == BN_LAB[0]),
                 net_young=net(young), net_old=net(old))
        out[gname] = r
        print(f"{gname:20s} {r['vol_k']:7.0f} {r['best']:5.0%} {r['back1']:5.0%} {r['back2']:5.0%} | " + ' '.join(f"{r[f'age {x}']:6.0%}" for x in AGE_LAB) +
              f" {r['age since first sight']:5.0%} {r['age censored']:5.0%} {r['age not resting']:6.0%} | {nb:8.0%} | {r['bn_against2']:6.1%} {r['bn_with2']:6.1%} | "
              f"{r['net1s']:6.2f} {r['net_against2']:7.2f} {r['net_young']:7.2f} {r['net_old']:8.2f}")
    io.write_json(f'{OUTD}/report_{part}.json', out)
print('\n== per maker-window inventory (whole window 0-300 s; ours = the arm\'s fills): leftover and P&L split')
print(f"{'group':20s} {'maker-windows':>13s} {'|net|/gross':>11s} {'flat<=10%':>9s} | {'mark at end c/sh':>16s} {'carry c/sh':>10s} {'carry sd $/mw':>13s}")
inv = {}
for gname in GR:
    w = I[I.grp == gname]
    if not len(w):
        continue
    gs = w.gross.sum(); ok = w.mtm.notna() & w.carry.notna()
    r = dict(n=len(w), absnet_over_gross=float(w.net.abs().sum() / gs), flat_share=float((w.net.abs() <= 0.1 * w.gross).mean()),
             mtm_c=float(100 * w.mtm[ok].sum() / w.gross[ok].sum()), carry_c=float(100 * w.carry[ok].sum() / w.gross[ok].sum()), carry_sd=float(w.carry[ok].std()))
    inv[gname] = r
    print(f"{gname:20s} {r['n']:13d} {r['absnet_over_gross']:11.0%} {r['flat_share']:9.0%} | {r['mtm_c']:16.2f} {r['carry_c']:10.2f} {r['carry_sd']:13.2f}")
io.write_json(f'{OUTD}/report_inventory.json', dict(statuses=st, inventory=inv))
print('\n== front of the queue: the first trade at a price in each run of resting size (market makers; 0-240 s)')
print(f"   {len(FR[FR.w240])} front legs; share of front shares vs share of all timed market volume, and the level age at that first trade")
a = A[(A.part == 'w240') & ~A.grp.str.startswith('ours')]; allv = a.groupby('grp').q.sum() / a.q.sum()
fr = FR[FR.w240]; fv = fr.groupby('grp').q.sum() / fr.q.sum(); front = {}
for gname in ['top 5', '6-20', 'rest']:
    g = fr[fr.grp == gname]; v = g.q.sum()
    ages = {x: float(g.q[g.age == x].sum() / v) for x in AGE_LAB + ['since first sight']}
    front[gname] = dict(front_share=float(fv.get(gname, 0)), volume_share=float(allv.get(gname, 0)), at_best=float(g.q[g.depth == 0].sum() / v), age=ages)
    print(f"   {gname:6s} front {front[gname]['front_share']:5.1%} of front shares vs {front[gname]['volume_share']:5.1%} of volume | front fills at best {front[gname]['at_best']:4.0%} | age "
          + ' '.join(f"{k} {v_:4.0%}" for k, v_ in ages.items()))
io.write_json(f'{OUTD}/report_front.json', front)
