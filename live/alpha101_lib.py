"""WorldQuant 101 Formulaic Alphas (Kakushadze 2015, arXiv 1601.00991), adapted to the 4h crypto panel.

Only the 57 alphas that are (a) implementable without IndNeutralize/cap and (b) NOT near-duplicates of
the GTJA191 port already screened (see live/ALPHA_SETS_SURVEY.md for the exclusion lists).
Same representation as alpha191_lib: time x symbol matrices, PIT shift(1) at the end, memory-safe loop.
Window adaptation: fractional windows rounded; 200/230/250-day windows -> 60 bars; adv120/adv180 -> adv60.
compute_all(ohlc_df) -> long DataFrame (symbol, open_time, wq###...).
"""
import sys; sys.path.insert(0, "/home/yuqing/ctaNew")
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.alpha191_lib import (RANK, DELAY, DELTA, TSSUM, MA, STD, TSMIN, TSMAX, PROD, CORR, COV,
                               TSRANK, DECAYLINEAR, SIGN, ABS, LOG, MAX, MIN, HIGHDAY, LOWDAY)

# ---------- extra operators ----------
def SCALE(df):                       # cross-sectional: rescale so sum(|x|)=1 per bar
    return df.div(df.abs().sum(axis=1).replace(0, np.nan), axis=0)
def TSARGMAX(df, n):                 # position of max within trailing window (higher = more recent max)
    return df.rolling(n, min_periods=max(2, n//2)).apply(lambda x: float(np.argmax(x))+1, raw=True)
def TSARGMIN(df, n):
    return df.rolling(n, min_periods=max(2, n//2)).apply(lambda x: float(np.argmin(x))+1, raw=True)
def SIGNEDPOWER(df, a): return SIGN(df) * (ABS(df) ** a)
def IF(cond, a, b):                  # elementwise conditional on matrices/scalars
    ref = cond if isinstance(cond, pd.DataFrame) else a
    return pd.DataFrame(np.where(cond, a, b), index=ref.index, columns=ref.columns)
def LT(a, b):                        # (a < b) as float with NaN where either side NaN
    out = (a < b).astype("float64")
    out[a.isna() | b.isna()] = np.nan
    return out

def _factors():
    F = {}
    def reg(name):
        def d(fn): F[name] = fn; return fn
        return d
    A = lambda x, n: MA(x['V'], n)   # adv{n} = trailing mean volume

    @reg("wq001")   # rank(ts_argmax(signedpower(ret<0? std(ret,20):close, 2),5))-.5   [vol-conditioned reversal]
    def _(x):
        base = IF(x['RET'] < 0, STD(x['RET'], 20), x['C'])
        return RANK(TSARGMAX(SIGNEDPOWER(base, 2.), 5)) - 0.5
    @reg("wq004")   # -ts_rank(rank(low),9)                                            [low-position reversal]
    def _(x): return -1 * TSRANK(RANK(x['L']), 9)
    @reg("wq007")   # adv20<vol ? -tsrank(|dclose7|,60)*sign(dclose7) : -1             [liquidity-gated reversal]
    def _(x):
        d7 = DELTA(x['C'], 7)
        return IF(A(x,20) < x['V'], -1*TSRANK(ABS(d7), 60)*SIGN(d7), -1.0)
    @reg("wq009")   # sign-consistent trend follow else reverse (window 5)
    def _(x):
        d = DELTA(x['C'], 1)
        return IF(TSMIN(d,5) > 0, d, IF(TSMAX(d,5) < 0, d, -1*d))
    @reg("wq010")   # rank(same, window 4)
    def _(x):
        d = DELTA(x['C'], 1)
        return RANK(IF(TSMIN(d,4) > 0, d, IF(TSMAX(d,4) < 0, d, -1*d)))
    @reg("wq012")   # sign(dvol)* -dclose                                              [volume-signed reversal]
    def _(x): return SIGN(DELTA(x['V'],1)) * (-1*DELTA(x['C'],1))
    @reg("wq013")   # -rank(cov(rank(close),rank(vol),5))                              [vol-price]
    def _(x): return -1*RANK(COV(RANK(x['C']), RANK(x['V']), 5))
    @reg("wq014")   # -rank(delta(ret,3))*corr(open,vol,10)
    def _(x): return -1*RANK(DELTA(x['RET'],3)) * CORR(x['O'], x['V'], 10)
    @reg("wq017")   # -rank(tsrank(close,10))*rank(ddclose)*rank(tsrank(vol/adv20,5))
    def _(x): return (-1*RANK(TSRANK(x['C'],10))) * RANK(DELTA(DELTA(x['C'],1),1)) * RANK(TSRANK(x['V']/A(x,20),5))
    @reg("wq019")   # -sign(mom7)*(1+rank(1+sum(ret,60)))                              [long-mom gated reversal]
    def _(x): return -1*SIGN((x['C']-DELAY(x['C'],7)) + DELTA(x['C'],7)) * (1 + RANK(1 + TSSUM(x['RET'],60)))
    @reg("wq020")   # -rank(open-d(high))*rank(open-d(close))*rank(open-d(low))        [gap anatomy]
    def _(x): return -1*RANK(x['O']-DELAY(x['H'],1)) * RANK(x['O']-DELAY(x['C'],1)) * RANK(x['O']-DELAY(x['L'],1))
    @reg("wq025")   # rank(-ret*adv20*vwap*(high-close))
    def _(x): return RANK((-1*x['RET']) * A(x,20) * x['VWAP'] * (x['H']-x['C']))
    @reg("wq027")   # .5<rank(mean(corr(rank(vol),rank(vwap),6),2)) ? -1 : 1
    def _(x):
        r = RANK(TSSUM(CORR(RANK(x['V']), RANK(x['VWAP']), 6), 2)/2.0)
        return IF(r > 0.5, -1.0, 1.0)
    @reg("wq028")   # scale(corr(adv20,low,5)+(h+l)/2-close)
    def _(x): return SCALE(CORR(A(x,20), x['L'], 5) + (x['H']+x['L'])/2 - x['C'])
    @reg("wq029")   # nested rank/scale pipeline + tsrank(delay(-ret,6),5)
    def _(x):
        inner = TSMIN(RANK(RANK(-1*RANK(DELTA(x['C']-1, 5)))), 2)
        p1 = TSMIN(RANK(RANK(SCALE(LOG(inner)))), 5)
        return p1 + TSRANK(DELAY(-1*x['RET'], 6), 5)
    @reg("wq030")   # (1-rank(3-bar sign streak))*sum(vol,5)/sum(vol,20)               [streak x volume]
    def _(x):
        c = x['C']
        streak = SIGN(c-DELAY(c,1)) + SIGN(DELAY(c,1)-DELAY(c,2)) + SIGN(DELAY(c,2)-DELAY(c,3))
        return (1.0 - RANK(streak)) * TSSUM(x['V'],5)/TSSUM(x['V'],20)
    @reg("wq031")   # rank^3(decay(-rank^2(dclose10),10)) + rank(-dclose3) + sign(scale(corr(adv20,low,12)))
    def _(x):
        return (RANK(RANK(RANK(DECAYLINEAR(-1*RANK(RANK(DELTA(x['C'],10))),10))))
                + RANK(-1*DELTA(x['C'],3)) + SIGN(SCALE(CORR(A(x,20), x['L'], 12))))
    @reg("wq032")   # scale(mean(close,7)-close)+20*scale(corr(vwap,delay(close,5),60))
    def _(x): return SCALE(TSSUM(x['C'],7)/7 - x['C']) + 20*SCALE(CORR(x['VWAP'], DELAY(x['C'],5), 60))
    @reg("wq033")   # rank(open/close - 1)                                             [candle body]
    def _(x): return RANK(-1*(1 - x['O']/x['C']))
    @reg("wq034")   # rank((1-rank(std(ret,2)/std(ret,5)))+(1-rank(dclose)))           [vol-ratio + reversal]
    def _(x): return RANK((1 - RANK(STD(x['RET'],2)/STD(x['RET'],5))) + (1 - RANK(DELTA(x['C'],1))))
    @reg("wq036")   # 5-term weighted rank blend
    def _(x):
        return (2.21*RANK(CORR(x['C']-x['O'], DELAY(x['V'],1), 15)) + 0.7*RANK(x['O']-x['C'])
                + 0.73*RANK(TSRANK(DELAY(-1*x['RET'],6),5)) + RANK(ABS(CORR(x['VWAP'], A(x,20), 6)))
                + 0.6*RANK((TSSUM(x['C'],60)/60 - x['O'])*(x['C']-x['O'])))
    @reg("wq038")   # -rank(tsrank(close,10))*rank(close/open)
    def _(x): return -1*RANK(TSRANK(x['C'],10)) * RANK(x['C']/x['O'])
    @reg("wq039")   # -rank(dclose7*(1-rank(decay(vol/adv20,9))))*(1+rank(sum(ret,60)))
    def _(x):
        return (-1*RANK(DELTA(x['C'],7)*(1 - RANK(DECAYLINEAR(x['V']/A(x,20), 9))))
                * (1 + RANK(TSSUM(x['RET'],60))))
    @reg("wq043")   # tsrank(vol/adv20,20)*tsrank(-dclose7,8)
    def _(x): return TSRANK(x['V']/A(x,20), 20) * TSRANK(-1*DELTA(x['C'],7), 8)
    @reg("wq044")   # -corr(high,rank(vol),5)
    def _(x): return -1*CORR(x['H'], RANK(x['V']), 5)
    @reg("wq045")   # -(rank(mean(delay(close,5),20))*corr(close,vol,2)*rank(corr(sum(close,5),sum(close,20),2)))
    def _(x):
        CORR2 = lambda a,b: pd.DataFrame({c: a[c].rolling(2, min_periods=2).corr(b[c]) for c in a.columns},
                                         index=a.index).replace([np.inf,-np.inf], np.nan)
        return -1*(RANK(TSSUM(DELAY(x['C'],5),20)/20) * CORR2(x['C'], x['V'])
                   * RANK(CORR2(TSSUM(x['C'],5), TSSUM(x['C'],20))))
    @reg("wq049")   # slope-gap < -0.1 ? 1 : -dclose                                   [trend-break reversal]
    def _(x):
        s = (DELAY(x['C'],20)-DELAY(x['C'],10))/10 - (DELAY(x['C'],10)-x['C'])/10
        return IF(s < -0.1, 1.0, -1*(x['C']-DELAY(x['C'],1)))
    @reg("wq050")   # -tsmax(rank(corr(rank(vol),rank(vwap),5)),5)
    def _(x): return -1*TSMAX(RANK(CORR(RANK(x['V']), RANK(x['VWAP']), 5)), 5)
    @reg("wq051")   # slope-gap < -0.05 ? 1 : -dclose
    def _(x):
        s = (DELAY(x['C'],20)-DELAY(x['C'],10))/10 - (DELAY(x['C'],10)-x['C'])/10
        return IF(s < -0.05, 1.0, -1*(x['C']-DELAY(x['C'],1)))
    @reg("wq053")   # -delta(((c-l)-(h-c))/(c-l),9)                                    [candle-position momentum]
    def _(x): return -1*DELTA(((x['C']-x['L'])-(x['H']-x['C']))/(x['C']-x['L']).replace(0,np.nan), 9)
    @reg("wq057")   # -(close-vwap)/decay(rank(tsargmax(close,30)),2)
    def _(x): return -1*(x['C']-x['VWAP']) / DECAYLINEAR(RANK(TSARGMAX(x['C'],30)), 2).replace(0,np.nan)
    @reg("wq060")   # -(2*scale(rank(candle-pos*vol)) - scale(rank(tsargmax(close,10))))
    def _(x):
        cp = ((x['C']-x['L'])-(x['H']-x['C']))/(x['H']-x['L'])*x['V']
        return -1*(2*SCALE(RANK(cp)) - SCALE(RANK(TSARGMAX(x['C'],10))))
    @reg("wq061")   # rank(vwap-tsmin(vwap,16)) < rank(corr(vwap,adv60,18))            [boolean]
    def _(x): return LT(RANK(x['VWAP']-TSMIN(x['VWAP'],16)), RANK(CORR(x['VWAP'], A(x,60), 18)))
    @reg("wq062")   # (rank(corr(vwap,sum(adv20,22),10)) < rank(2*rank(open) < rank(mid)+rank(high))) * -1
    def _(x):
        inner = LT(RANK(x['O'])*2, RANK((x['H']+x['L'])/2) + RANK(x['H']))
        return -1*LT(RANK(CORR(x['VWAP'], TSSUM(A(x,20),22), 10)), RANK(inner))
    @reg("wq064")   # (rank(corr(sum(.178*o+.822*l,13),sum(adv60,13),17)) < rank(delta(.178*mid+.822*vwap,4))) * -1
    def _(x):
        a = RANK(CORR(TSSUM(x['O']*0.178404 + x['L']*0.821596, 13), TSSUM(A(x,60),13), 17))
        b = RANK(DELTA(((x['H']+x['L'])/2)*0.178404 + x['VWAP']*0.821596, 4))
        return -1*LT(a, b)
    @reg("wq065")   # (rank(corr(.008*o+.992*vwap,sum(adv60,9),6)) < rank(open-tsmin(open,14))) * -1
    def _(x):
        a = RANK(CORR(x['O']*0.00817205 + x['VWAP']*0.99182795, TSSUM(A(x,60),9), 6))
        return -1*LT(a, RANK(x['O']-TSMIN(x['O'],14)))
    @reg("wq066")   # -(rank(decay(dvwap4,7)) + tsrank(decay((low-vwap)/(open-mid),11),7))
    def _(x):
        z = (x['L']-x['VWAP']) / (x['O']-(x['H']+x['L'])/2).replace(0,np.nan)
        return -1*(RANK(DECAYLINEAR(DELTA(x['VWAP'],4),7)) + TSRANK(DECAYLINEAR(z,11),7))
    @reg("wq068")   # (tsrank(corr(rank(high),rank(adv15),9),14) < rank(delta(.518*c+.482*l,1))) * -1
    def _(x):
        a = TSRANK(CORR(RANK(x['H']), RANK(A(x,15)), 9), 14)
        return -1*LT(a, RANK(DELTA(x['C']*0.518371 + x['L']*0.481629, 1)))
    @reg("wq071")   # max(tsrank(decay(corr(tsrank(c,3),tsrank(adv60,12),18),4),16), tsrank(decay(rank(l+o-2vwap)^2,16),4))
    def _(x):
        p1 = TSRANK(DECAYLINEAR(CORR(TSRANK(x['C'],3), TSRANK(A(x,60),12), 18), 4), 16)
        p2 = TSRANK(DECAYLINEAR(RANK(x['L']+x['O']-2*x['VWAP'])**2, 16), 4)
        return MAX(p1, p2)
    @reg("wq072")   # rank(decay(corr(mid,adv40,9),10)) / rank(decay(corr(tsrank(vwap,4),tsrank(vol,19),7),3))
    def _(x):
        num = RANK(DECAYLINEAR(CORR((x['H']+x['L'])/2, A(x,40), 9), 10))
        den = RANK(DECAYLINEAR(CORR(TSRANK(x['VWAP'],4), TSRANK(x['V'],19), 7), 3))
        return num/den.replace(0,np.nan)
    @reg("wq073")   # -max(rank(decay(dvwap5,3)), tsrank(decay(-dpct(.147*o+.853*l,2),3),17))
    def _(x):
        w = x['O']*0.147155 + x['L']*0.852845
        p1 = RANK(DECAYLINEAR(DELTA(x['VWAP'],5), 3))
        p2 = TSRANK(DECAYLINEAR(-1*DELTA(w,2)/w, 3), 17)
        return -1*MAX(p1, p2)
    @reg("wq074")   # (rank(corr(close,sum(adv30,37),15)) < rank(corr(rank(.026*h+.974*vwap),rank(vol),11))) * -1
    def _(x):
        a = RANK(CORR(x['C'], TSSUM(A(x,30),37), 15))
        b = RANK(CORR(RANK(x['H']*0.0261661 + x['VWAP']*0.9738339), RANK(x['V']), 11))
        return -1*LT(a, b)
    @reg("wq075")   # rank(corr(vwap,vol,4)) < rank(corr(rank(low),rank(adv50),12))    [boolean]
    def _(x): return LT(RANK(CORR(x['VWAP'], x['V'], 4)), RANK(CORR(RANK(x['L']), RANK(A(x,50)), 12)))
    @reg("wq077")   # min(rank(decay(mid-vwap,20)), rank(decay(corr(mid,adv40,3),6)))
    def _(x):
        mid = (x['H']+x['L'])/2
        return MIN(RANK(DECAYLINEAR(mid-x['VWAP'], 20)), RANK(DECAYLINEAR(CORR(mid, A(x,40), 3), 6)))
    @reg("wq078")   # rank(corr(sum(.352*l+.648*vwap,20),sum(adv40,20),7)) ^ rank(corr(rank(vwap),rank(vol),6))
    def _(x):
        a = RANK(CORR(TSSUM(x['L']*0.352233 + x['VWAP']*0.647767, 20), TSSUM(A(x,40),20), 7))
        return a ** RANK(CORR(RANK(x['VWAP']), RANK(x['V']), 6))
    @reg("wq081")   # (rank(sum(log(rank(rank(corr(vwap,sum(adv10,50),8))^4)),15)) < rank(corr(rank(vwap),rank(vol),5))) * -1
    def _(x):
        a = RANK(TSSUM(LOG(RANK(RANK(CORR(x['VWAP'], TSSUM(A(x,10),50), 8))**4)), 15))
        return -1*LT(a, RANK(CORR(RANK(x['VWAP']), RANK(x['V']), 5)))
    @reg("wq084")   # signedpower(tsrank(vwap-tsmax(vwap,15),21), dclose5)             [rank^delta interaction]
    def _(x): return SIGNEDPOWER(TSRANK(x['VWAP']-TSMAX(x['VWAP'],15), 21), DELTA(x['C'],5))
    @reg("wq085")   # rank(corr(.877*h+.123*c,adv30,10)) ^ rank(corr(tsrank(mid,4),tsrank(vol,10),7))
    def _(x):
        a = RANK(CORR(x['H']*0.876703 + x['C']*0.123297, A(x,30), 10))
        return a ** RANK(CORR(TSRANK((x['H']+x['L'])/2, 4), TSRANK(x['V'],10), 7))
    @reg("wq086")   # (tsrank(corr(close,sum(adv20,15),6),20) < rank(close-vwap)) * -1
    def _(x): return -1*LT(TSRANK(CORR(x['C'], TSSUM(A(x,20),15), 6), 20), RANK(x['C']-x['VWAP']))
    @reg("wq088")   # min(rank(decay(rank(o)+rank(l)-rank(h)-rank(c),8)), tsrank(decay(corr(tsrank(c,8),tsrank(adv60,21),8),7),3))
    def _(x):
        p1 = RANK(DECAYLINEAR(RANK(x['O'])+RANK(x['L'])-RANK(x['H'])-RANK(x['C']), 8))
        p2 = TSRANK(DECAYLINEAR(CORR(TSRANK(x['C'],8), TSRANK(A(x,60),21), 8), 7), 3)
        return MIN(p1, p2)
    @reg("wq092")   # min(tsrank(decay(mid+close<low+open,15),19), tsrank(decay(corr(rank(l),rank(adv30),8),7),7))
    def _(x):
        cond = LT((x['H']+x['L'])/2 + x['C'], x['L']+x['O'])
        p1 = TSRANK(DECAYLINEAR(cond, 15), 19)
        p2 = TSRANK(DECAYLINEAR(CORR(RANK(x['L']), RANK(A(x,30)), 8), 7), 7)
        return MIN(p1, p2)
    @reg("wq094")   # -(rank(vwap-tsmin(vwap,12)) ^ tsrank(corr(tsrank(vwap,20),tsrank(adv60,4),18),3))
    def _(x):
        a = RANK(x['VWAP']-TSMIN(x['VWAP'],12))
        return -1*(a ** TSRANK(CORR(TSRANK(x['VWAP'],20), TSRANK(A(x,60),4), 18), 3))
    @reg("wq095")   # rank(open-tsmin(open,12)) < tsrank(rank(corr(sum(mid,19),sum(adv40,19),13))^5,12)  [boolean]
    def _(x):
        b = TSRANK(RANK(CORR(TSSUM((x['H']+x['L'])/2, 19), TSSUM(A(x,40),19), 13))**5, 12)
        return LT(RANK(x['O']-TSMIN(x['O'],12)), b)
    @reg("wq096")   # -max(tsrank(decay(corr(rank(vwap),rank(vol),4),4),8), tsrank(decay(tsargmax(corr(tsrank(c,7),tsrank(adv60,4),4),13),14),13))
    def _(x):
        p1 = TSRANK(DECAYLINEAR(CORR(RANK(x['VWAP']), RANK(x['V']), 4), 4), 8)
        p2 = TSRANK(DECAYLINEAR(TSARGMAX(CORR(TSRANK(x['C'],7), TSRANK(A(x,60),4), 4), 13), 14), 13)
        return -1*MAX(p1, p2)
    @reg("wq098")   # rank(decay(corr(vwap,sum(adv5,26),5),7)) - rank(decay(tsrank(tsargmin(corr(rank(o),rank(adv15),21),9),7),8))
    def _(x):
        p1 = RANK(DECAYLINEAR(CORR(x['VWAP'], TSSUM(A(x,5),26), 5), 7))
        p2 = RANK(DECAYLINEAR(TSRANK(TSARGMIN(CORR(RANK(x['O']), RANK(A(x,15)), 21), 9), 7), 8))
        return p1 - p2
    @reg("wq099")   # (rank(corr(sum(mid,20),sum(adv60,20),9)) < rank(corr(low,vol,6))) * -1
    def _(x):
        a = RANK(CORR(TSSUM((x['H']+x['L'])/2, 20), TSSUM(A(x,60),20), 9))
        return -1*LT(a, RANK(CORR(x['L'], x['V'], 6)))
    @reg("wq101")   # (close-open)/(high-low+.001)                                     [candle body ratio]
    def _(x): return (x['C']-x['O'])/((x['H']-x['L'])+0.001)
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
    RET = C.pct_change()
    x = dict(O=O,H=H,L=L,C=C,V=V.replace(0,np.nan),AMT=AMT,VWAP=VWAP,RET=RET)
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
        if i % 10 == 0: print(f"  {i}/{len(FACTORS)}", flush=True)
        gc.collect()
    out = ohlc[["open_time","symbol"]].copy()
    for name, arr in cols.items(): out[name] = arr
    return out

if __name__ == "__main__":
    df = pd.read_parquet("/home/yuqing/ctaNew/data/ml/cache/alpha191_ohlc4h.parquet")
    print(f"loaded {df['symbol'].nunique()} syms, {len(df)} rows; computing {len(FACTORS)} WQ101 factors...", flush=True)
    feats = compute_all(df)
    feats.to_parquet("/home/yuqing/ctaNew/data/ml/cache/alpha101_factors.parquet")
    print(f"-> {len([c for c in feats.columns if c.startswith('wq')])} factors, {len(feats)} rows saved.")
    print("COMPUTEDONE")
