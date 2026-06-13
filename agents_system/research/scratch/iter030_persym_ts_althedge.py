"""
iter-030 — Per-symbol TIME-SERIES strategy on the per-symbol pred, hedged with
the equal-weight ALT-INDEX (vs BTC, vs no hedge).

THE IDEA (fix vs iter-004): iter-004 traded each symbol on its own pred
(long pred>0 / short pred<0) and hedged net book beta with BTC -> LOST -1.97
Sharpe. The TS-IC of pred was real (+0.0116 / t4.5) but didn't monetize.
HYPOTHESIS (iter-006): the drawdown-driving risk is the ALT-COMPLEX factor
(alts fell -24% while BTC -7%); BTC UNDER-hedges alts. So neutralize NET
ALT-beta with the equal-weight alt index, isolating the idiosyncratic
beta-residual the pred targets.

PIT discipline:
  - pred at t is the model forecast (already PIT, lagged, walk-forward).
  - per-symbol TS weight at cycle t = f(pred_t). Held 24h via 6 sleeves.
  - hedge instrument weight sized to neutralize NET book beta to the hedge
    index, beta estimated on a TRAILING window ending t-1 (.shift(1)).
  - PnL realized t -> t+4h using return_pct (raw fwd return; the hedge removes
    the systematic part). Cost 4.5 bps/leg on turnover (incl hedge leg).

Compares standalone Sharpe (gross + net) on HL70 AND EXT for:
  (A) per-sym TS, NO hedge
  (B) per-sym TS, BTC hedge        (iter-004 replication)
  (C) per-sym TS, ALT-INDEX hedge  (THE FIX)
and the sizing variants {sign, clipped-pred}. Reports net-beta neutralization
check, turnover/cost, and corr to baseline book.
"""
import numpy as np, pandas as pd, sys, time
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"outputs/iter030"
KLINES = REPO/"data/ml/test/parquet/klines"
HOLD = 6
COST = 4.5e-4
BARS_PER_YR = 6*365
BETA_WIN = 180   # trailing 4h bars for beta (~30d), matches baseline

def ann(x):
    x = pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(BARS_PER_YR) if len(x)>2 and x.std()>0 else np.nan

def metrics(net_series, label):
    pb = net_series*1e4
    eq = pb.cumsum(); peak=eq.cummax(); dd=eq-peak
    sh = ann(net_series); maxdd=dd.min()
    cal = pb.mean()*BARS_PER_YR/abs(maxdd) if maxdd<0 else np.nan
    return dict(label=label, sharpe=sh, totpnl=eq.iloc[-1], maxdd=maxdd, calmar=cal,
                pos=(pb>0).mean()*100)

def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    c=df.set_index("open_time")["close"].astype(np.float64)
    return c[(c.index.hour%4==0)&(c.index.minute==0)]

def build(lab, preds_path):
    d = pd.read_parquet(preds_path)
    syms = sorted(d.symbol.unique())
    times = sorted(d.open_time.unique())
    tidx = pd.to_datetime(times)
    Tn = len(times); ti = {t:i for i,t in enumerate(times)}
    # pivot pred and return_pct to (T, sym) matrices
    PRED = d.pivot_table(index='open_time', columns='symbol', values='pred').reindex(tidx)
    RET  = d.pivot_table(index='open_time', columns='symbol', values='return_pct').reindex(tidx)
    syms = list(PRED.columns)

    # per-4h-bar return matrix for beta estimation (use the realized 4h return = return_pct,
    # but for trailing beta we need contemporaneous per-bar returns: reconstruct from klines 4h)
    # Build 4h close per symbol aligned to tidx
    closes = {}
    for s in syms:
        c = load_close(s)
        if c is not None: closes[s]=c.reindex(tidx)
    btc = load_close("BTCUSDT").reindex(tidx)
    btc_r = np.log(btc/btc.shift(1))
    # per-bar log returns matrix
    CL = pd.DataFrame(closes).reindex(columns=syms)
    BARR = np.log(CL/CL.shift(1))           # (T, sym) contemporaneous 4h log-ret
    # equal-weight ALT index per-bar return = mean across available syms (PIT, contemporaneous)
    alt_r = BARR.mean(axis=1)
    # forward 4h return for the hedge instruments (to realize hedge PnL t->t+4h),
    # = next-bar return = BARR.shift(-1); but we realize at same cycle as book's return_pct
    # (book return_pct is the fwd 4h return at t). Hedge fwd return = BARR.shift(-1).
    btc_fwd = btc_r.shift(-1)
    alt_fwd = alt_r.shift(-1)

    # trailing beta of each symbol to alt-index and to BTC (.shift(1), PIT)
    var_alt = alt_r.rolling(BETA_WIN, min_periods=42).var()
    var_btc = btc_r.rolling(BETA_WIN, min_periods=42).var()
    BETA_ALT = pd.DataFrame(index=tidx, columns=syms, dtype=float)
    BETA_BTC = pd.DataFrame(index=tidx, columns=syms, dtype=float)
    for s in syms:
        rs = BARR[s]
        BETA_ALT[s] = (rs.rolling(BETA_WIN,min_periods=42).cov(alt_r)/var_alt.replace(0,np.nan)).shift(1)
        BETA_BTC[s] = (rs.rolling(BETA_WIN,min_periods=42).cov(btc_r)/var_btc.replace(0,np.nan)).shift(1)
    return dict(lab=lab, syms=syms, tidx=tidx, PRED=PRED, RET=RET,
                BETA_ALT=BETA_ALT, BETA_BTC=BETA_BTC, alt_fwd=alt_fwd, btc_fwd=btc_fwd)

def run(data, sizing='sign', hedge='alt', K=None, cost=COST):
    """sizing: 'sign' (long pred>0, short pred<0, eq weight per active sym, scaled 1/n)
               'clip'  (weight = clip(pred,-1,1) cross-sec demeaned? no - raw pred z, /n)
       hedge:  'none' | 'btc' | 'alt'
       K: if set, only trade top-K/bottom-K by pred (concentrated TS). None=all syms.
    """
    PRED, RET = data['PRED'], data['RET']
    BETA = data['BETA_ALT'] if hedge=='alt' else data['BETA_BTC']
    HFWD = data['alt_fwd'] if hedge=='alt' else data['btc_fwd']
    tidx = data['tidx']; syms=data['syms']; Tn=len(tidx)
    Pv = PRED.values; Rv = RET.values; Bv = BETA.values
    hfwd = HFWD.values

    # per-cycle target weights (before sleeve averaging)
    cyc_w = []        # list of dict sym->weight
    cyc_netbeta = []  # the net book beta to hedge index (sum w_s * beta_s)
    for t in range(Tn):
        prow = Pv[t]; finite = np.isfinite(prow)
        idx = np.where(finite)[0]
        if len(idx) < 4:
            cyc_w.append({}); cyc_netbeta.append(0.0); continue
        if K is not None and len(idx) >= 2*K:
            order = idx[np.argsort(prow[idx])]
            shorts = order[:K]; longs = order[-K:]
            sel = list(longs)+list(shorts)
            sgn = {j:(1.0 if j in longs else -1.0) for j in sel}
        else:
            sel = list(idx)
            sgn = {j:(1.0 if prow[j]>0 else -1.0) for j in sel}
        n = len(sel)
        w = {}
        if sizing=='sign':
            for j in sel: w[j] = sgn[j]/n
        else:  # clip: size by |pred| capped at 1, signed
            mags = {j: min(abs(prow[j]),3.0) for j in sel}
            tot = sum(mags.values()) or 1.0
            for j in sel: w[j] = sgn[j]*mags[j]/tot
        cyc_w.append(w)
        # net beta to hedge index (PIT beta at t)
        nb = 0.0
        for j,wt in w.items():
            bj = Bv[t,j]
            if np.isfinite(bj): nb += wt*bj
        cyc_netbeta.append(nb)

    # held-book with sleeves; hedge weight applied at the NET (post-sleeve) level
    prev = {}; prev_h = 0.0
    pnl=[]; gross=[]; netbeta_held=[]; hedge_used=[]
    for t in range(Tn):
        active = cyc_w[max(0,t-HOLD+1):t+1]
        net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        # net book beta to hedge index at t (recompute on held net using current-bar PIT beta)
        nb=0.0
        for j,wt in net.items():
            bj=Bv[t,j]
            if np.isfinite(bj): nb+=wt*bj
        # hedge: short the index by nb (neutralize net beta). hedge fwd return realized t->t+4h.
        h = -nb if hedge!='none' else 0.0
        # turnover incl hedge leg
        alls=set(net)|set(prev)
        turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls) + abs(h-prev_h)
        # gross book pnl
        gp=0.0
        for j,wt in net.items():
            r=Rv[t,j]
            if np.isfinite(r): gp+=wt*r
        # hedge pnl
        hp = h*hfwd[t] if (hedge!='none' and np.isfinite(hfwd[t])) else 0.0
        gp_tot = gp + hp
        pnl.append(gp_tot - turn*0.5*cost)
        gross.append(gp_tot)
        netbeta_held.append(nb)  # net beta BEFORE hedge (to verify hedge would neutralize)
        hedge_used.append(h)
        prev=net; prev_h=h
    net_s = pd.Series(pnl, index=tidx)
    gross_s = pd.Series(gross, index=tidx)
    # residual net beta AFTER hedge: book net-beta + hedge*1 (index beta to itself=1) = nb + h = nb-nb=0
    # but verify empirically: regress net daily pnl-stream on alt_fwd
    return net_s, gross_s, pd.Series(netbeta_held,index=tidx), pd.Series(hedge_used,index=tidx)

if __name__=='__main__':
    t0=time.time()
    res=[]
    streams={}
    for lab in ['HL70','EXT']:
        print(f"\n==== building {lab} ...", flush=True)
        data = build(lab, OUT/f"preds4h_{lab}.parquet")
        print(f"  built in {time.time()-t0:.0f}s  syms={len(data['syms'])} T={len(data['tidx'])}", flush=True)
        for sizing in ['sign','clip']:
            for hedge in ['none','btc','alt']:
                net_s, gross_s, nb_s, h_s = run(data, sizing=sizing, hedge=hedge)
                # net-beta neutralization check: regress net pnl on hedge-index fwd return
                hfwd = (data['alt_fwd'] if hedge=='alt' else data['btc_fwd'])
                # beta of NET stream to ALT index (the risk we care about) - both pre/post
                alt = data['alt_fwd'].reindex(net_s.index)
                df = pd.concat([net_s.rename('p'), alt.rename('a')],axis=1).dropna()
                beta_to_alt = np.polyfit(df['a'],df['p'],1)[0] if len(df)>10 and df['a'].std()>0 else np.nan
                mg = metrics(net_s, f"{lab}_{sizing}_{hedge}")
                mgg= metrics(gross_s, f"{lab}_{sizing}_{hedge}_GROSS")
                mg['gross_sharpe']=mgg['sharpe']; mg['gross_pnl']=mgg['totpnl']
                mg['avg_netbeta_to_hedge']=nb_s.mean(); mg['beta_net_to_ALT']=beta_to_alt
                mg['turn_proxy']=h_s.abs().mean()
                res.append(mg)
                streams[f"{lab}_{sizing}_{hedge}"]=net_s
                print(f"  [{lab} {sizing:4s} {hedge:4s}] net Sh {mg['sharpe']:+.2f} "
                      f"gross Sh {mgg['sharpe']:+.2f} totPnL {mg['totpnl']:+.0f} maxDD {mg['maxdd']:+.0f} "
                      f"betaToALT {beta_to_alt:+.4f} avgNB {nb_s.mean():+.3f}", flush=True)
    R=pd.DataFrame(res)
    R.to_csv(OUT/"iter030_summary.csv", index=False)
    pd.DataFrame(streams).to_parquet(OUT/"iter030_net_streams.parquet")
    print("\n=== SUMMARY ===")
    print(R[['label','sharpe','gross_sharpe','totpnl','maxdd','beta_net_to_ALT','avg_netbeta_to_hedge']].to_string(index=False))
    print(f"\ndone {time.time()-t0:.0f}s")
