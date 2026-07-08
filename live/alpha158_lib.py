"""Microsoft Qlib Alpha158 feature set, adapted to the 4h crypto panel.

Systematic grid (not hand-crafted formulas): 9 KBAR candle-shape + 4 price-level ratios +
29 rolling operators x windows {5,10,20,30,60}. All from OHLCV+VWAP — fully implementable.
Definitions follow qlib/contrib/data/handler.py Alpha158 config; eps guards match qlib's +1e-12.
Same representation & memory-safe pattern as alpha191_lib. PIT shift(1) at the end.
compute_all(ohlc_df) -> long DataFrame (symbol, open_time, q158_*...).
"""
import sys; sys.path.insert(0, "/home/yuqing/ctaNew")
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.alpha191_lib import (DELAY, TSSUM, MA, STD, TSMIN, TSMAX, TSRANK, CORR, ABS, MAX, MIN,
                               HIGHDAY, LOWDAY, LOG)

EPS = 1e-12
WINDOWS = [5, 10, 20, 30, 60]

def QUANTILE(df, n, q):
    return df.rolling(n, min_periods=max(2, n//2)).quantile(q)

_REG_CACHE = {}
def _slope_rsq_resi(df, n):
    """Rolling OLS of y on t=0..n-1 within each FULL trailing window -> (slope, r^2, last-point residual).
    Fully vectorized: Sty built from n shifted adds (no python-level rolling.apply).
    Memoized per (df identity, n) — BETA/RSQR/RESI share one regression pass."""
    key = (id(df), n)
    if key in _REG_CACHE: return _REG_CACHE[key]
    y = df.astype("float64")
    Sy  = y.rolling(n, min_periods=n).sum()
    Syy = (y**2).rolling(n, min_periods=n).sum()
    Sty = sum(float(j) * y.shift(n-1-j) for j in range(1, n))   # sum_j j*y_{i-n+1+j}
    tm  = (n-1)/2.0
    Stt = n*(n*n-1)/12.0
    b = (Sty - tm*Sy) / Stt
    a = Sy/n - b*tm
    sst = Syy - Sy**2/n
    r2  = (b**2 * Stt) / sst.where(sst > 0)
    resi = y - a - b*(n-1)
    _REG_CACHE[key] = (b, r2, resi)
    return b, r2, resi

def _factors():
    F = {}
    def reg(name, fn): F[name] = fn

    # ---- KBAR (9) ----
    reg("q158_KMID",  lambda x: (x['C']-x['O'])/x['O'])
    reg("q158_KLEN",  lambda x: (x['H']-x['L'])/x['O'])
    reg("q158_KMID2", lambda x: (x['C']-x['O'])/(x['H']-x['L']+EPS))
    reg("q158_KUP",   lambda x: (x['H']-MAX(x['O'],x['C']))/x['O'])
    reg("q158_KUP2",  lambda x: (x['H']-MAX(x['O'],x['C']))/(x['H']-x['L']+EPS))
    reg("q158_KLOW",  lambda x: (MIN(x['O'],x['C'])-x['L'])/x['O'])
    reg("q158_KLOW2", lambda x: (MIN(x['O'],x['C'])-x['L'])/(x['H']-x['L']+EPS))
    reg("q158_KSFT",  lambda x: (2*x['C']-x['H']-x['L'])/x['O'])
    reg("q158_KSFT2", lambda x: (2*x['C']-x['H']-x['L'])/(x['H']-x['L']+EPS))
    # ---- price levels (4) ----
    reg("q158_OPEN0", lambda x: x['O']/x['C'])
    reg("q158_HIGH0", lambda x: x['H']/x['C'])
    reg("q158_LOW0",  lambda x: x['L']/x['C'])
    reg("q158_VWAP0", lambda x: x['VWAP']/x['C'])
    # ---- rolling x windows (29 x 5 = 145) ----
    for w in WINDOWS:
        def mk(w):
            C  = lambda x: x['C']; V = lambda x: x['V']
            dC = lambda x: x['C']-DELAY(x['C'],1)          # close diff
            dV = lambda x: x['V']-DELAY(x['V'],1)          # volume diff
            rv = lambda x: ABS(x['C']/DELAY(x['C'],1)-1)*x['V']   # |ret|*vol for WVMA
            reg(f"q158_ROC{w}",  lambda x: DELAY(x['C'],w)/x['C'])
            reg(f"q158_MA{w}",   lambda x: MA(x['C'],w)/x['C'])
            reg(f"q158_STD{w}",  lambda x: STD(x['C'],w)/x['C'])
            def beta_fn(x, w=w):
                s,_,_ = _slope_rsq_resi(x['C'], w); return s/x['C']
            def rsqr_fn(x, w=w):
                _,r,_ = _slope_rsq_resi(x['C'], w); return r
            def resi_fn(x, w=w):
                _,_,e = _slope_rsq_resi(x['C'], w); return e/x['C']
            reg(f"q158_BETA{w}", beta_fn)
            reg(f"q158_RSQR{w}", rsqr_fn)
            reg(f"q158_RESI{w}", resi_fn)
            reg(f"q158_MAX{w}",  lambda x: TSMAX(x['H'],w)/x['C'])
            reg(f"q158_MIN{w}",  lambda x: TSMIN(x['L'],w)/x['C'])
            reg(f"q158_QTLU{w}", lambda x: QUANTILE(x['C'],w,0.8)/x['C'])
            reg(f"q158_QTLD{w}", lambda x: QUANTILE(x['C'],w,0.2)/x['C'])
            reg(f"q158_RANK{w}", lambda x: TSRANK(x['C'],w))
            reg(f"q158_RSV{w}",  lambda x: (x['C']-TSMIN(x['L'],w))/(TSMAX(x['H'],w)-TSMIN(x['L'],w)+EPS))
            reg(f"q158_IMAX{w}", lambda x: HIGHDAY(x['H'],w)/w)
            reg(f"q158_IMIN{w}", lambda x: LOWDAY(x['L'],w)/w)
            reg(f"q158_IMXD{w}", lambda x: (LOWDAY(x['L'],w)-HIGHDAY(x['H'],w))/w)
            reg(f"q158_CORR{w}", lambda x: CORR(x['C'], LOG(x['V']+1), w))
            reg(f"q158_CORD{w}", lambda x: CORR(x['C']/DELAY(x['C'],1), LOG(x['V']/DELAY(x['V'],1)+1), w))
            reg(f"q158_CNTP{w}", lambda x: MA((dC(x)>0).astype(float), w))
            reg(f"q158_CNTN{w}", lambda x: MA((dC(x)<0).astype(float), w))
            reg(f"q158_CNTD{w}", lambda x: MA((dC(x)>0).astype(float), w) - MA((dC(x)<0).astype(float), w))
            reg(f"q158_SUMP{w}", lambda x: TSSUM(MAX(dC(x), dC(x)*0), w)/(TSSUM(ABS(dC(x)), w)+EPS))
            reg(f"q158_SUMN{w}", lambda x: TSSUM(MAX(-1*dC(x), dC(x)*0), w)/(TSSUM(ABS(dC(x)), w)+EPS))
            reg(f"q158_SUMD{w}", lambda x: (TSSUM(MAX(dC(x), dC(x)*0), w)-TSSUM(MAX(-1*dC(x), dC(x)*0), w))/(TSSUM(ABS(dC(x)), w)+EPS))
            reg(f"q158_VMA{w}",  lambda x: MA(x['V'],w)/(x['V']+EPS))
            reg(f"q158_VSTD{w}", lambda x: STD(x['V'],w)/(x['V']+EPS))
            reg(f"q158_WVMA{w}", lambda x: STD(rv(x),w)/(MA(rv(x),w)+EPS))
            reg(f"q158_VSUMP{w}",lambda x: TSSUM(MAX(dV(x), dV(x)*0), w)/(TSSUM(ABS(dV(x)), w)+EPS))
            reg(f"q158_VSUMN{w}",lambda x: TSSUM(MAX(-1*dV(x), dV(x)*0), w)/(TSSUM(ABS(dV(x)), w)+EPS))
            reg(f"q158_VSUMD{w}",lambda x: (TSSUM(MAX(dV(x), dV(x)*0), w)-TSSUM(MAX(-1*dV(x), dV(x)*0), w))/(TSSUM(ABS(dV(x)), w)+EPS))
        mk(w)
    return F

FACTORS = _factors()

def compute_all(ohlc):
    """Same memory-safe pattern as alpha191_lib.compute_all (fixed master index, per-factor loop, gc)."""
    import gc
    ohlc = ohlc.copy(); ohlc["open_time"] = pd.to_datetime(ohlc["open_time"], utc=True)
    ohlc = ohlc.sort_values(["open_time","symbol"]).reset_index(drop=True)
    master = pd.MultiIndex.from_frame(ohlc[["open_time","symbol"]])
    piv = lambda col: ohlc.pivot(index="open_time", columns="symbol", values=col).astype("float32")
    O,H,L,C,V = piv("open"),piv("high"),piv("low"),piv("close"),piv("volume")
    AMT = piv("quote_volume"); VWAP = (AMT/V.replace(0,np.nan))
    x = dict(O=O,H=H,L=L,C=C,V=V.replace(0,np.nan),AMT=AMT,VWAP=VWAP)
    cols = {}
    for i,(name, fn) in enumerate(FACTORS.items(),1):
        try:
            m = fn(x).replace([np.inf,-np.inf], np.nan)
            m = m.shift(1).astype("float32")             # PIT: only CLOSED bars <= t-1
            s = m.stack()
            cols[name] = s.reindex(master).to_numpy()
            del m, s
        except Exception as e:
            print(f"  [skip {name}: {e}]", flush=True)
        if i % 20 == 0: print(f"  {i}/{len(FACTORS)}", flush=True)
        gc.collect()
    out = ohlc[["open_time","symbol"]].copy()
    for name, arr in cols.items(): out[name] = arr
    return out

if __name__ == "__main__":
    df = pd.read_parquet("/home/yuqing/ctaNew/data/ml/cache/alpha191_ohlc4h.parquet")
    print(f"loaded {df['symbol'].nunique()} syms, {len(df)} rows; computing {len(FACTORS)} Alpha158 factors...", flush=True)
    feats = compute_all(df)
    feats.to_parquet("/home/yuqing/ctaNew/data/ml/cache/alpha158_factors.parquet")
    print(f"-> {len([c for c in feats.columns if c.startswith('q158_')])} factors, {len(feats)} rows saved.")
    print("COMPUTEDONE")
