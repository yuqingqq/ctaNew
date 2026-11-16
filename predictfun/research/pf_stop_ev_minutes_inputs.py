"""Recorded-clock inputs for a continuous five-minute frozen STOP_EV replay.

The loader/window/fair-price routines below are derived from the frozen
pf_kp_s4 and pf_stop_model implementations. Their only economic extension
is time coverage. The first-minute input and replay parity checks in the
runner are mandatory. Untimed tape uses the already frozen day's imputation
offset, with its source retained; it is never a policy input.
"""
import glob
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_strategy_scorecard as S
import pf_stop_model_x5 as X5
np,pd,K,M,L=S.np,S.pd,S.K,S.M,S.L
IO=K.IO;G32=K.G32;_ms=K._ms;SEAL_MS=K.SEAL_MS;FWD_MS=K.FWD_MS
part_allowed=K.part_allowed;SKIP_FOREIGN_PARTS=True;PREROLL_MS=30000.
PF_MIN=K.PF_MIN;INF=float('inf');NAN=float('nan');capped=K.capped
relay_basis=M.relay_basis;MUT_FAIR_DUP=0
QUOTE_MS=299800.


def part_overlaps(path, windows):
    name=Path(path).name;d=name[5:13];h=int(name[14:16])
    start=_ms(f'{d[:4]}-{d[4:6]}-{d[6:]}')+h*3600000
    return start+3600000>min(windows.values())-PREROLL_MS and start<=max(windows.values())+300000.


def load_day(day, markets, impute_offset):
    """One replay day's inputs, windows inside the day only (statuses for the rest)."""
    S1d = f'{IO}/step1/{day}'; S5d = f'{IO}/step5/{day}'
    reg = pd.read_parquet(f'{S1d}/registry.parquet', columns=['mid', 'ws', 'outcome'])
    d0 = _ms(f'{day[:4]}-{day[4:6]}-{day[6:]}'); d1 = d0 + 86400000
    st = dict(registry=int(len(reg)), outside_day=0, sealed_or_forward=0, no_outcome=0)
    W0 = {}
    for m, ws, o in zip(reg.mid, reg.ws, reg.outcome):
        w = float(ws) * 1000.0
        if not (d0 <= w < d1):
            st['outside_day'] += 1; continue
        if SEAL_MS[0] <= w < SEAL_MS[1] or w >= FWD_MS:
            st['sealed_or_forward'] += 1; continue
        if o not in ('Up', 'Down'):
            st['no_outcome'] += 1; continue
        W0[int(m)] = w
    W0 = {m: w for m, w in W0.items() if m in markets}
    st['selected_windows'] = len(W0)
    YUP = {int(m): float(o == 'Up') for m, o in zip(reg.mid, reg.outcome) if o in ('Up', 'Down')}
    ck = json.load(open(f'{S1d}/clock_rl_d2.json'))
    if abs(ck['d_dec'] - 2.0) > 1e-9:
        sys.exit(f'REFUSED: {day} clock built with decision time {ck["d_dec"]}')
    P = pd.read_parquet(f'{S5d}/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask']); P = P[P.mid.isin(W0)]
    G = {int(m): (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in P.groupby('mid')}
    del P
    r = pd.read_parquet(f'{S5d}/rows_rl_d2_pk.parquet', columns=['mid', 'tau'] + G32); r = r[r.mid.isin(W0)]
    r = r[r.tau.values < r.mid.map(W0).values + 301000.0]   # all contract phases; trailing operations unchanged
    r = r.sort_values(['mid', 'tau'], kind='stable').reset_index(drop=True); g = r.groupby('mid', sort=False)
    r['d_lbs250'] = r.pf_lbs - g.pf_lbs.shift(5); r['d_las250'] = r.pf_las - g.pf_las.shift(5); r['d_bnimb250'] = r.bn_imb - g.bn_imb.shift(5)
    lt = np.log1p(r.bn_ntr1000.values); r['lt'] = lt
    r['act_rel'] = lt - r.groupby('mid', sort=False).lt.transform(lambda x: x.rolling(600, min_periods=20).mean()).values
    ROWS = {int(m_): x for m_, x in r.groupby('mid', sort=False)}
    del r
    cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    st['parts_skipped'] = 0
    for fn in sorted(glob.glob(f'{S1d}/books/part_*.parquet')):
        if SKIP_FOREIGN_PARTS and not part_allowed(fn, day):
            st['parts_skipped'] += 1; continue
        if not part_overlaps(fn, W0):
            continue
        d = pd.read_parquet(fn, columns=cols); d = d[d.mid.isin(W0)]
        rel = d.t_ms.values - d.mid.map(W0).values
        parts.append(d[(rel >= -PREROLL_MS) & (rel <= 300000.0)])
    B = pd.concat(parts, ignore_index=True); del parts
    LT = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    tk = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tk) != len(LT):
        sys.exit(f'REFUSED: {day} legs_tk not aligned with legs_timed2')
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(LT)))
    ws_t = np.where(LT.how.values > 0, LT.t_ms.values.astype(float), np.nan)
    e2_t = np.where((LT.how.values == 0) & np.isfinite(LT.t_e2.values), LT.t_e2.values, np.nan)
    a_ = pd.Series(ws_t).groupby(grp, sort=False).transform('min').values
    b_ = pd.Series(e2_t).groupby(grp, sort=False).transform('min').values
    LT['anc'] = np.where(np.isfinite(a_), a_, b_)
    LT['row_id'] = np.arange(len(LT))
    LT['group'] = grp
    LT['ws_timed'] = M.TS.ws_groups(LT.how.to_numpy(), grp)
    LT['imputed'] = ~np.isfinite(LT.anc)
    LT.loc[LT.imputed,'ws_timed']=False
    LT = LT[LT.mid.isin(W0)].copy(); LT['ws'] = LT.mid.map(W0)
    st['raw_selected_legs']=len(LT);st['raw_selected_shares']=float(LT.shares.sum())
    unt = LT[~np.isfinite(LT.anc.values) & (LT.exec_s * 1000.0 >= LT.ws) & (LT.exec_s * 1000.0 < LT.ws + 300000.0)]
    st.update(untimed_legs_0_300=int(len(unt)), untimed_shares_0_300=float(unt.shares.sum()))
    timed = LT[~LT.imputed].copy()
    imp = LT[LT.imputed & (LT.exec_s*1000 >= LT.ws-2000.) & (LT.exec_s*1000 < LT.ws+301000.)].copy()
    imp['anc'] = imp.groupby('group').exec_s.transform('min')*1000.+impute_offset
    LT = pd.concat([timed, imp]).sort_index()
    st['imputed_legs'] = len(imp)
    st['imputed_shares'] = float(imp.shares.sum())
    st['untimed_outside_context_legs']=st['raw_selected_legs']-len(LT)
    st['untimed_outside_context_shares']=st['raw_selected_shares']-float(LT.shares.sum())
    outside=~((LT.anc >= LT.ws-2000.) & (LT.anc <= LT.ws+301000.))
    st['anchored_outside_context_legs']=int(outside.sum())
    st['anchored_outside_context_shares']=float(LT.loc[outside,'shares'].sum())
    LT = LT[(LT.anc >= LT.ws - 2000.0) & (LT.anc <= LT.ws + 301000.0)]
    for label,mask in [('before_contract',LT.anc-50.<LT.ws),('after_contract',LT.anc-50.>=LT.ws+300000.)]:
        st[label+'_legs']=int(mask.sum());st[label+'_shares']=float(LT.loc[mask,'shares'].sum())
    st['windows'] = len(W0)
    return dict(day=day, W0=W0, YUP=YUP, SH=ck['shift_ms'], DD=ck['d_dec'], G=G, ROWS=ROWS, B=B, BI=B.groupby('mid', sort=False).indices,
                LT=LT, LI=LT.groupby(['mid', 'side'], sort=False).indices, stat=st)


def window_sides(D, quote_ms=QUOTE_MS):
    """Yields one context per window-side (full contract), with everything the simulator, the callbacks and the values need."""
    P90 = json.load(open(f'{IO}/step4/minute_lines_t60k.json'))['lines']['90'][0]
    for m in sorted(D['W0'], key=D['W0'].get):
        if m not in D['BI'] or m not in D['G'] or m not in D['ROWS']:
            D['stat']['no_frames_preds_rows'] = D['stat'].get('no_frames_preds_rows', 0) + 1; continue
        ws = D['W0'][m]; te = ws + quote_ms
        b = D['B'].iloc[D['BI'][m]].sort_values('t_ms', kind='stable')
        tf = b.t_ms.values.astype(float)
        kf = np.maximum.accumulate(np.maximum(tf + PF_MIN, b.recv_ns.values / 1e6 - D['SH']) + D['DD'])
        bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64'); valid = (bb >= 0) & (ba > bb)
        mff = pd.Series(np.where(valid, (ba + bb) / 2.0, np.nan)).ffill().values
        lev = {0: (b[[f'b{i}_p' for i in range(1, 6)]].values.astype('int64'), b[[f'b{i}_s' for i in range(1, 6)]].values.astype(float)),
               1: (b[[f'a{i}_p' for i in range(1, 6)]].values.astype('int64'), b[[f'a{i}_s' for i in range(1, 6)]].values.astype(float))}
        tau, vb, va = D['G'][m]; kk = (tau >= ws) & (tau < te); tg = tau[kk]; n = len(tg)
        ki = np.searchsorted(kf, tg, side='right') - 1; okk = (ki >= 0) & (bb[ki] >= 0) & (ba[ki] > bb[ki])
        kb_ = np.where(okk, bb[ki], -1); ka_ = np.where(okk, ba[ki], -1)
        best = {0: kb_.tolist(), 1: ka_.tolist()}; vk = {0: vb[kk], 1: va[kk]}
        x = D['ROWS'][m]; rt = x.tau.values; j = np.searchsorted(rt, tg); okr = (j < len(rt)) & (rt[np.clip(j, 0, len(rt) - 1)] == tg)
        jj = np.clip(j, 0, len(rt) - 1)
        D['stat']['grid_steps'] = D['stat'].get('grid_steps', 0) + n; D['stat']['grid_steps_without_row'] = D['stat'].get('grid_steps_without_row', 0) + int((~okr).sum())
        base = np.where(okr[:, None], x[G32].values[jj].astype(float), np.nan)
        for side in (0, 1):
            sg = 1.0 if side == 0 else -1.0
            own_d = (x.d_lbs250 if side == 0 else x.d_las250).values[jj]; opp_d = (x.d_las250 if side == 0 else x.d_lbs250).values[jj]
            extra = np.stack([vk[side], np.where(okr, own_d, np.nan), np.where(okr, opp_d, np.nan),
                              np.where(okr, x.act_rel.values[jj], np.nan), np.where(okr, x.d_bnimb250.values[jj], np.nan)], 1)
            if (m, side) in D['LI']:
                l = D['LT'].iloc[D['LI'][(m, side)]]
                o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values))
                anc = l.anc.values[o_]; pt = l.ptick.values[o_].astype('int64'); sh = l.shares.values[o_].astype(float)
            else:
                anc = np.zeros(0); pt = np.zeros(0, 'int64'); sh = np.zeros(0)
            yield dict(m=m, side=side, sg=sg, ws=ws, te=te, tf=tf, kf=kf, jk=np.searchsorted(kf, tg, side='right') - 1, tg=tg, n=n,
                       opp=(ba, bb)[side], P5=lev[side][0], S5=lev[side][1], best=best[side], bopp=best[1 - side],
                       vkl=vk[side].tolist(), stn=capped(tau, (vb, va)[side], -INF, -INF, INF)[0][kk].tolist(),
                       stv3=capped(tau, (vb, va)[side], P90, P90, INF)[0][kk].tolist(), P90=P90,
                       X=np.concatenate([base, extra], 1), anc=anc, ptl=pt.tolist(), shl=sh.tolist(), yup=D['YUP'][m], tf_mff=(tf, mff))


def fair_fix(day, W0):
    """Step 5's fair price with the relay basis rebuilt so that every relay print counts once (2026-11-15, s5check2: step 5's per-hour
    Binance passes include the prints of each hour's 10 s carry in two passes, biasing the 60 s median basis for steps within 60 s after an
    hour boundary - the first minute of every HH:00 window; up to 0.7 bps, a few cents of fair price). Same clocks as step 5: relay known at
    receipt - (72 - decision) ms, Binance at event + 1.5 + decision ms; same median window and minimum; bn_mid / sigma / time left / start
    price from step 5 and the registry (verified against raw data by s5check2). Returns ({market: (tau, p_bn, tau_rem, sigma, gap)}, stats)."""
    import pf_io as io
    from scipy.special import ndtr
    ck = json.load(open(f'{K.IO}/step1/{day}/clock_rl_d2.json')); dd = float(ck['d_dec']); b_in = float(ck['bn_in']) + dd; net = 72.0 - dd
    reg = pd.read_parquet(f'{K.IO}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'start_price']); SP = dict(zip(reg.mid.astype(int), reg.start_price.astype(float)))
    WSd = dict(zip(reg.mid.astype(int), reg.ws.astype(float)))
    S = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'bn_mid', 'tau_rem', 'sigma', 'p_bn', 'gap', 'basis_bps'])
    S = S[S.mid.isin(W0)]; S = S[S.tau.values < S.mid.map(WSd).values * 1000.0 + 300000.0].sort_values(['mid', 'tau'], kind='stable')
    d0 = pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]}', tz='UTC'); hours = []
    for i in range(-1, 24):
        h = (d0 + pd.Timedelta(hours=i)).strftime('%Y%m%d_%H')
        if K.part_allowed(f'part_{h}.parquet', day):
            hours.append(h)
    REF = np.array(sorted((r[0] / 1e6 - net, float(r[2])) for h in hours for r in io.ref_rows(h) if not r[3]))
    rv = np.full(len(REF), np.nan); prev = None
    for h in hours:                                        # stream Binance hours; each print gets the mid known at its time, once
        p_ = f'{io.BN_DIR}/{h}.csv.gz'
        if not os.path.exists(p_):
            continue
        df = pd.read_csv(p_, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'ask'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
        k = df.E.values.astype(np.float64) + b_in; o = np.argsort(k, kind='stable'); k = k[o]; m = ((df.bid.values + df.ask.values) / 2.0)[o]; del df
        if prev is not None:
            k = np.r_[prev[0], k]; m = np.r_[prev[1], m]
        sel = np.flatnonzero(np.isnan(rv) & (REF[:, 0] >= k[0]) & (REF[:, 0] <= k[-1]))
        if MUT_FAIR_DUP:
            sel = np.flatnonzero((REF[:, 0] >= k[0]) & (REF[:, 0] <= k[-1]))
        ii = np.searchsorted(k, REF[sel, 0], side='right') - 1; ok = ii >= 0
        rv[sel[ok]] = REF[sel[ok], 1] / m[ii[ok]]
        keep = k >= k[-1] - 10000.0; prev = (k[keep], m[keep])
    okr = np.isfinite(rv); rel_t = REF[okr, 0]; rel_v = rv[okr]
    bas = relay_basis(rel_t, rel_v, S.tau.values.astype(np.float64))
    sp = S.mid.map(SP).values.astype(np.float64); old = 1.0 + S.basis_bps.values.astype(np.float64) / 1e4
    den = S.sigma.values.astype(np.float64) * np.sqrt(np.maximum(S.tau_rem.values.astype(np.float64), 1e-3)); mid_ = S.bn_mid.values.astype(np.float64)
    # the correction is a DELTA on step 5's own p_bn (both evaluations from the same stored inputs), so rows whose basis is unchanged keep
    # step 5's value bit for bit (a full recomputation from float32-stored inputs moved every row by up to 0.0085 c - float32 noise)
    same = np.isfinite(bas) & np.isfinite(old) & (np.abs(bas - old) <= 1e-9)
    delta = 100.0 * (ndtr(np.log(mid_ * bas / sp) / den) - ndtr(np.log(mid_ * old / sp) / den))
    p5 = S.p_bn.values.astype(np.float64)
    pbn = np.where(same, p5, np.where(np.isfinite(old), p5 + delta, 100.0 * ndtr(np.log(mid_ * bas / sp) / den)))
    gapc = S.gap.values.astype(np.float64) + (pbn - p5)    # step 5's gap = p_bn - the known Predict mid
    dif = np.abs(pbn - S.p_bn.values) > 1e-4; hh = (S.mid.map(WSd).values % 3600.0) == 0.0
    st = dict(rows=int(len(S)), changed=float(np.nanmean(dif)), changed_hh00=float(np.nanmean(dif[hh])) if hh.any() else None,
              changed_other=float(np.nanmean(dif[~hh])) if (~hh).any() else None, max_abs_c=float(np.nanmax(np.abs(pbn - S.p_bn.values))) if len(S) else None,
              relay_prints=int(len(rel_t)))
    out = {}
    for mm_, g in pd.DataFrame({'mid': S.mid.values, 'tau': S.tau.values, 'p': pbn, 'tr': S.tau_rem.values, 'sg': S.sigma.values, 'gp': gapc}).groupby('mid', sort=False):
        out[int(mm_)] = (g.tau.values.astype(float).tolist(), g.p.values, g.tr.values, g.sg.values, g.gp.values)
    return out, st


def pairs(D, quote_ms=QUOTE_MS):
    pending={}
    for w in window_sides(D,quote_ms):
        pending[w['side']]=w
        if len(pending)==2:
            L.require(pending[0]['m']==pending[1]['m'],'pair identity')
            yield pending
            pending={}
    L.require(not pending,'unpaired market')


def market_inputs(W, fair, depth):
    scale,_=M.pair_scales(W)
    po=X5.pair_ofi(W);out={};summary=[]
    for s,w in W.items():
        _,valid,rs,_=M.QR.runs(w['tf'],w['P5'],w['opp'],s)
        rso=M.run_starts(np.asarray(w['opp'],'int64'),valid)
        ks=np.flatnonzero((w['jk']>=0)&(np.asarray(w['best'])>=0))
        nanp=np.full(w['n'],np.nan)
        raw=[M.all_feats(w,int(k),s,rs,rso,nanp,fair,bool(w['vkl'][k]==M.MASK_V)) for k in ks]
        frame=pd.DataFrame(np.asarray(raw,np.float32).reshape(-1,len(M.FEAT)),columns=M.FEAT)
        frame['scale_w30']=scale[ks].astype(np.float32)
        more=X5.side_inputs(po,X5.depth_feats(depth,w['tg'][ks]),scale,s,ks)
        for name,v in more.items():frame[name]=v.astype(np.float32)
        x=M.model_matrix(frame,'W30',S.V.F71)
        ix=np.full(w['n'],-1,int);ix[ks]=np.arange(len(ks))
        sc=frame.scale_w30.to_numpy(float)
        out[s]=(ix,x,sc,x[:,S.V.F71.index('pf_mid')])
        for minute in range(5):
            mask=((w['tg'][ks]-w['ws'])//60000).astype(int)==minute
            summary.append(dict(minute=minute+1,side=s,valid_rows=int(mask.sum()),
                missing_cells=int(np.isnan(x[mask]).sum()),market_cells=int(x[mask].size),
                outside_training_phase=int(mask.sum()) if minute else 0))
    return out,summary


def truncate(W,end_ms=60000.):
    out={}
    for s,w in W.items():
        n=int(np.searchsorted(w['tg'],w['ws']+end_ms))
        z=dict(w,te=w['ws']+end_ms,n=n,cache={})
        for key in ['tg','jk','best','bopp','vkl','stn','stv3','X']:z[key]=w[key][:n]
        out[s]=z
    return out
