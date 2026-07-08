"""GTJA Alpha191 factor library, adapted to our 4h cross-sectional crypto panel.

Representation: each raw field is a time x symbol matrix (index=open_time, cols=symbol). TS operators act
along the time axis (per symbol); cross-sectional RANK acts along the symbol axis (per bar). This is the
standard Alpha101/191 vectorization. Benchmark = BTC (substitutes BANCHMARKINDEX*).

Faithful subset of the canonical GTJA set — operators are exact; factors implemented are the ones spanning
all structural families (vol-price CORR, DECAYLINEAR, TSRANK, reversal, intraday-range, momentum, STD).
compute_all(ohlc_df) -> long DataFrame (symbol, open_time, <alpha_name>...).
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

# ---------- operators (all operate on time x symbol matrices) ----------
def RANK(df):  return df.rank(axis=1, pct=True)                       # cross-sectional pct rank per bar
def DELAY(df, n): return df.shift(n)
def DELTA(df, n): return df.diff(n)
def TSSUM(df, n): return df.rolling(n, min_periods=max(2, n//2)).sum()
def MA(df, n):   return df.rolling(n, min_periods=max(2, n//2)).mean()
def STD(df, n):  return df.rolling(n, min_periods=max(2, n//2)).std()
def TSMIN(df, n):return df.rolling(n, min_periods=max(2, n//2)).min()
def TSMAX(df, n):return df.rolling(n, min_periods=max(2, n//2)).max()
def PROD(df, n): return df.rolling(n, min_periods=max(2, n//2)).apply(np.prod, raw=True)
def CORR(a, b, n):
    out = {c: a[c].rolling(n, min_periods=max(3, n//2)).corr(b[c]) for c in a.columns}
    return pd.DataFrame(out, index=a.index).replace([np.inf,-np.inf], np.nan)
def COV(a, b, n):
    out = {c: a[c].rolling(n, min_periods=max(3, n//2)).cov(b[c]) for c in a.columns}
    return pd.DataFrame(out, index=a.index)
def TSRANK(df, n):   # rolling rank of the last value within trailing window (per symbol), scaled 0..1
    return df.rolling(n, min_periods=max(3, n//2)).apply(lambda x: (x.argsort().argsort()[-1]+1)/len(x), raw=True)
def DECAYLINEAR(df, n):
    w = np.arange(1, n+1, dtype=float); w /= w.sum()
    return df.rolling(n, min_periods=max(2, n//2)).apply(lambda x: np.dot(x, w[-len(x):])/w[-len(x):].sum(), raw=True)
def SMA(df, n, m):   # Chinese SMA recursion == EWM with alpha=m/n
    return df.ewm(alpha=m/n, adjust=False).mean()
def WMA(df, n):
    w = 0.9 ** np.arange(n)[::-1]; w /= w.sum()
    return df.rolling(n, min_periods=max(2, n//2)).apply(lambda x: np.dot(x, w[-len(x):])/w[-len(x):].sum(), raw=True)
def HIGHDAY(df, n):  # bars since the max in trailing n
    return df.rolling(n, min_periods=max(2, n//2)).apply(lambda x: len(x)-1-int(np.argmax(x)), raw=True)
def LOWDAY(df, n):
    return df.rolling(n, min_periods=max(2, n//2)).apply(lambda x: len(x)-1-int(np.argmin(x)), raw=True)
def COUNT_GT0(df, n):
    return (df > 0).rolling(n, min_periods=max(2, n//2)).sum()
def SIGN(df): return np.sign(df)
def ABS(df):  return df.abs()
def LOG(df):  return np.log(df.clip(lower=1e-12))
def MAX(a, b): return a.where(a > b, b)
def MIN(a, b): return a.where(a < b, b)

# ---------- factor set (name -> fn(x) where x is a dict of matrices) ----------
def _factors():
    F = {}
    def reg(name):
        def d(fn): F[name] = fn; return fn
        return d

    @reg("alpha001")   # -corr(rank(delta(log(vol))), rank((close-open)/open), 6)   [vol-price]
    def _(x): return -1*CORR(RANK(DELTA(LOG(x['V']),1)), RANK((x['C']-x['O'])/x['O']), 6)
    @reg("alpha002")   # -delta((( close-low)-(high-close))/(high-low),1)            [intraday range reversal]
    def _(x): return -1*DELTA(((x['C']-x['L'])-(x['H']-x['C']))/(x['H']-x['L']), 1)
    @reg("alpha003")   # sum(close>delay? ...) momentum-ish
    def _(x):
        c,dc = x['C'], DELAY(x['C'],1)
        val = pd.DataFrame(np.where(c==dc,0,c-np.where(c>dc, MIN(x['L'],dc), MAX(x['H'],dc))), index=c.index, columns=c.columns)
        return TSSUM(val, 6)
    @reg("alpha004")   # reversal/breakout regime flag
    def _(x):
        cond = (TSSUM(x['C'],8)/8 + STD(x['C'],8)) < (TSSUM(x['C'],2)/2)
        cond2 = (TSSUM(x['C'],2)/2) < (TSSUM(x['C'],8)/8 - STD(x['C'],8))
        vr = x['V']/MA(x['V'],20)
        out = pd.DataFrame(np.where(cond,-1.0,np.where(cond2,1.0,np.where(vr>=1,1.0,-1.0))), index=x['C'].index, columns=x['C'].columns)
        return out
    @reg("alpha005")   # -tsmax(corr(tsrank(vol,5),tsrank(high,5),5),3)              [vol-price]
    def _(x): return -1*TSMAX(CORR(TSRANK(x['V'],5), TSRANK(x['H'],5), 5), 3)
    @reg("alpha006")   # -rank(sign(delta(open*0.85+high*0.15,4)))                    [intraday momentum]
    def _(x): return -1*RANK(SIGN(DELTA(x['O']*0.85 + x['H']*0.15, 4)))
    @reg("alpha007")   # rank(max(vwap-close,3))+rank(min(...))*rank(delta(vol))
    def _(x): return (RANK(TSMAX(x['VWAP']-x['C'],3)) + RANK(TSMIN(x['VWAP']-x['C'],3))) * RANK(DELTA(x['V'],3))
    @reg("alpha008")   # rank(delta(-(high+low)/2*0.2 + vwap*0.8,4))                  [decay-ish price]
    def _(x): return RANK(DELTA(-1*((x['H']+x['L'])/2*0.2 + x['VWAP']*0.8), 4))
    @reg("alpha009")   # sma(((high+low)/2 - delay ...)*(high-low)/vol,7,2)
    def _(x): return SMA(((x['H']+x['L'])/2 - (DELAY(x['H'],1)+DELAY(x['L'],1))/2) * (x['H']-x['L'])/x['V'], 7, 2)
    @reg("alpha010")   # rank(max(std(ret,20) or ret^2, 5))                           [vol]
    def _(x):
        r = x['RET']; cond = pd.DataFrame(np.where(r<0, STD(r,20), x['C']), index=r.index, columns=r.columns)
        return RANK(TSMAX(cond**2, 5))
    @reg("alpha012")   # rank(open-sum(vwap,10)/10)*(-rank(abs(close-vwap)))
    def _(x): return RANK(x['O'] - TSSUM(x['VWAP'],10)/10) * (-1*RANK(ABS(x['C']-x['VWAP'])))
    @reg("alpha013")   # (high*low)^0.5 - vwap                                        [intraday]
    def _(x): return ((x['H']*x['L'])**0.5) - x['VWAP']
    @reg("alpha014")   # close - delay(close,5)                                       [momentum]
    def _(x): return x['C'] - DELAY(x['C'],5)
    @reg("alpha015")   # open/delay(close,1) - 1                                      [overnight-gap reversal]
    def _(x): return x['O']/DELAY(x['C'],1) - 1
    @reg("alpha017")   # rank(vwap-max(vwap,15))^delta(close,5)
    def _(x): return RANK(x['VWAP'] - TSMAX(x['VWAP'],15)) * DELTA(x['C'],5)
    @reg("alpha018")   # close/delay(close,5)                                         [momentum]
    def _(x): return x['C']/DELAY(x['C'],5)
    @reg("alpha019")   # reversal vs 5-bar
    def _(x):
        c,d5 = x['C'], DELAY(x['C'],5)
        return pd.DataFrame(np.where(c<d5,(c-d5)/d5, np.where(c==d5,0,(c-d5)/c)), index=c.index, columns=c.columns)
    @reg("alpha020")   # (close-delay(close,6))/delay(close,6)*100                    [momentum]
    def _(x): return (x['C']-DELAY(x['C'],6))/DELAY(x['C'],6)*100
    @reg("alpha022")   # sma(((close-mean(close,6))/mean(close,6) - delay(...,3)),12,1)  [reversal]
    def _(x):
        z = (x['C']-MA(x['C'],6))/MA(x['C'],6); return SMA(z - DELAY(z,3), 12, 1)
    @reg("alpha023")   # std-based up/down asymmetry
    def _(x):
        c,dc = x['C'], DELAY(x['C'],1)
        up = pd.DataFrame(np.where(c>dc, STD(c,20), 0), index=c.index, columns=c.columns)
        dn = pd.DataFrame(np.where(c<=dc, STD(c,20), 0), index=c.index, columns=c.columns)
        return SMA(up,20,1)/(SMA(up,20,1)+SMA(dn,20,1))*100
    @reg("alpha024")   # sma(close-delay(close,5),5,1)                                [reversal]
    def _(x): return SMA(x['C']-DELAY(x['C'],5), 5, 1)
    @reg("alpha026")   # tssum(close,7)/7-close + corr(vwap, delay(close,5),230-ish)
    def _(x): return (TSSUM(x['C'],7)/7 - x['C']) + CORR(x['VWAP'], DELAY(x['C'],5), 60)
    @reg("alpha028")   # 3*sma((close-tsmin(low,9))/(tsmax(high,9)-tsmin(low,9))*100,3,1) - ...  [stochastic]
    def _(x):
        k = (x['C']-TSMIN(x['L'],9))/(TSMAX(x['H'],9)-TSMIN(x['L'],9))*100
        return 3*SMA(k,3,1) - 2*SMA(SMA(k,3,1),3,1)
    @reg("alpha029")   # (close-delay(close,6))/delay(close,6)*volume                 [vol-weighted momentum]
    def _(x): return (x['C']-DELAY(x['C'],6))/DELAY(x['C'],6)*x['V']
    @reg("alpha031")   # (close-mean(close,12))/mean(close,12)*100                    [reversal]
    def _(x): return (x['C']-MA(x['C'],12))/MA(x['C'],12)*100
    @reg("alpha032")   # -sum(rank(corr(rank(high),rank(vol),3)),3)                   [vol-price]
    def _(x): return -1*TSSUM(RANK(CORR(RANK(x['H']), RANK(x['V']), 3)), 3)
    @reg("alpha033")   # (-tsmin(low,5)+delay(tsmin(low,5),5))*rank(sum(ret,240)/220)*tsrank(vol,5)
    def _(x): return (-1*TSMIN(x['L'],5)+DELAY(TSMIN(x['L'],5),5)) * RANK(TSSUM(x['RET'],60)/55) * TSRANK(x['V'],5)
    @reg("alpha034")   # mean(close,12)/close                                         [reversal]
    def _(x): return MA(x['C'],12)/x['C']
    @reg("alpha035")   # decaylinear-based open/vwap                                  [decay]
    def _(x): return MIN(RANK(DECAYLINEAR(DELTA(x['O'],1),15)), RANK(DECAYLINEAR(CORR(x['V'], x['O']*0.65+x['O']*0.35, 17),7)))*-1
    @reg("alpha037")   # -rank(sum(open,5)*sum(ret,5) - delay(...,10))
    def _(x): return -1*RANK(TSSUM(x['O'],5)*TSSUM(x['RET'],5) - DELAY(TSSUM(x['O'],5)*TSSUM(x['RET'],5),10))
    @reg("alpha038")   # if sum(high,20)/20<high: -delta(high,2) else 0
    def _(x):
        cond = (TSSUM(x['H'],20)/20) < x['H']
        return pd.DataFrame(np.where(cond, -1*DELTA(x['H'],2), 0), index=x['H'].index, columns=x['H'].columns)
    @reg("alpha040")   # sum(vol if close>delay else 0,26)/sum(vol if close<=delay,26)*100  [vol-price]
    def _(x):
        c,dc=x['C'],DELAY(x['C'],1)
        up=pd.DataFrame(np.where(c>dc,x['V'],0),index=c.index,columns=c.columns)
        dn=pd.DataFrame(np.where(c<=dc,x['V'],0),index=c.index,columns=c.columns)
        return TSSUM(up,26)/TSSUM(dn,26)*100
    @reg("alpha041")   # rank(max(delta(vwap,3),5))*-1
    def _(x): return RANK(TSMAX(DELTA(x['VWAP'],3),5))*-1
    @reg("alpha042")   # -rank(std(high,10))*corr(high,vol,10)                        [vol-price]
    def _(x): return -1*RANK(STD(x['H'],10))*CORR(x['H'], x['V'], 10)
    @reg("alpha043")   # sum(vol signed by close move,6)                             [vol-price]
    def _(x):
        c,dc=x['C'],DELAY(x['C'],1)
        s=pd.DataFrame(np.where(c>dc,x['V'],np.where(c<dc,-x['V'],0)),index=c.index,columns=c.columns)
        return TSSUM(s,6)
    @reg("alpha044")   # tsrank(decaylinear(corr(low,mean(vol,10),7),6),4)+tsrank(decaylinear(delta(vwap,3),10),15) [decay/vol-price]
    def _(x): return TSRANK(DECAYLINEAR(CORR(x['L'], MA(x['V'],10), 7), 6), 4) + TSRANK(DECAYLINEAR(DELTA(x['VWAP'],3),10),15)
    @reg("alpha045")   # rank(delta(close*0.6+open*0.4,1))*rank(corr(vwap,mean(vol,15),15))  [vol-price]
    def _(x): return RANK(DELTA(x['C']*0.6+x['O']*0.4,1))*RANK(CORR(x['VWAP'], MA(x['V'],15), 15))
    @reg("alpha046")   # (mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24))/(4*close)  [reversal]
    def _(x): return (MA(x['C'],3)+MA(x['C'],6)+MA(x['C'],12)+MA(x['C'],24))/(4*x['C'])
    @reg("alpha047")   # sma((tsmax(high,6)-close)/(tsmax(high,6)-tsmin(low,6))*100,9,1)  [stochastic]
    def _(x): return SMA((TSMAX(x['H'],6)-x['C'])/(TSMAX(x['H'],6)-TSMIN(x['L'],6))*100, 9, 1)
    @reg("alpha052")   # sum(max(0,high-delay((h+l+c)/3)),26)/sum(max(0,delay-low),26)*100
    def _(x):
        tp=DELAY((x['H']+x['L']+x['C'])/3,1)
        a=MAX(x['H']-tp, x['H']*0); b=MAX(tp-x['L'], x['L']*0)
        return TSSUM(a,26)/TSSUM(b,26)*100
    @reg("alpha053")   # count(close>delay(close),12)/12*100                          [momentum]
    def _(x): return COUNT_GT0(x['C']-DELAY(x['C'],1),12)/12*100
    @reg("alpha054")   # -rank(std(abs(close-open))+ (close-open) + corr(close,open,10))  [vol-price]
    def _(x): return -1*RANK(STD(ABS(x['C']-x['O']),10) + (x['C']-x['O']) + CORR(x['C'], x['O'], 10))
    @reg("alpha057")   # sma((close-tsmin(low,9))/(tsmax(high,9)-tsmin(low,9))*100,3,1)  [stochastic]
    def _(x): return SMA((x['C']-TSMIN(x['L'],9))/(TSMAX(x['H'],9)-TSMIN(x['L'],9))*100, 3, 1)
    @reg("alpha058")   # count(close>delay,20)/20*100                                 [momentum]
    def _(x): return COUNT_GT0(x['C']-DELAY(x['C'],1),20)/20*100
    @reg("alpha059")   # sum(close-min/max signed,20)                                 [reversal]
    def _(x):
        c,dc=x['C'],DELAY(x['C'],1)
        v=pd.DataFrame(np.where(c==dc,0,c-np.where(c>dc,MIN(x['L'],dc),MAX(x['H'],dc))),index=c.index,columns=c.columns)
        return TSSUM(v,20)
    @reg("alpha060")   # sum(((close-low)-(high-close))/(high-low)*vol,20)            [vol-price]
    def _(x): return TSSUM(((x['C']-x['L'])-(x['H']-x['C']))/(x['H']-x['L'])*x['V'], 20)
    @reg("alpha063")   # sma(max(close-delay,0),6,1)/sma(abs(close-delay),6,1)*100    [RSI]
    def _(x):
        d=x['C']-DELAY(x['C'],1)
        return SMA(MAX(d, d*0),6,1)/SMA(ABS(d),6,1)*100
    @reg("alpha065")   # mean(close,6)/close                                          [reversal]
    def _(x): return MA(x['C'],6)/x['C']
    @reg("alpha066")   # (close-mean(close,6))/mean(close,6)*100                      [reversal]
    def _(x): return (x['C']-MA(x['C'],6))/MA(x['C'],6)*100
    @reg("alpha067")   # sma(max(close-delay,0),24,1)/sma(abs,24,1)*100               [RSI slow]
    def _(x):
        d=x['C']-DELAY(x['C'],1); return SMA(MAX(d,d*0),24,1)/SMA(ABS(d),24,1)*100
    @reg("alpha068")   # sma(((h+l)/2-delay((h+l)/2))*(h-l)/vol,15,2)                 [vol-price]
    def _(x): return SMA(((x['H']+x['L'])/2 - DELAY((x['H']+x['L'])/2,1))*(x['H']-x['L'])/x['V'], 15, 2)
    @reg("alpha070")   # std(amount,6)                                                [liquidity vol]
    def _(x): return STD(x['AMT'],6)
    @reg("alpha071")   # (close-mean(close,24))/mean(close,24)*100                    [reversal]
    def _(x): return (x['C']-MA(x['C'],24))/MA(x['C'],24)*100
    @reg("alpha072")   # sma((tsmax(high,6)-close)/(tsmax-tsmin)*100,15,1)            [stochastic]
    def _(x): return SMA((TSMAX(x['H'],6)-x['C'])/(TSMAX(x['H'],6)-TSMIN(x['L'],6))*100, 15, 1)
    @reg("alpha074")   # rank(corr(sum((low*0.35+vwap*0.65),20),sum(mean(vol,40),20),7))  [vol-price]
    def _(x): return RANK(CORR(TSSUM(x['L']*0.35+x['VWAP']*0.65,20), TSSUM(MA(x['V'],40),20), 7))
    @reg("alpha076")   # std(abs(close/delay-1)/vol,20)/mean(abs(close/delay-1)/vol,20)  [vol-price]
    def _(x):
        z=ABS(x['C']/DELAY(x['C'],1)-1)/x['V']; return STD(z,20)/MA(z,20)
    @reg("alpha078")   # ((h+l+c)/3-mean((h+l+c)/3,12))/(0.015*mean(abs(close-mean),12))  [CCI]
    def _(x):
        tp=(x['H']+x['L']+x['C'])/3; return (tp-MA(tp,12))/(0.015*MA(ABS(x['C']-MA(tp,12)),12))
    @reg("alpha081")   # sma(vol,21,2)                                                [vol smooth]
    def _(x): return SMA(x['V'],21,2)
    @reg("alpha082")   # sma((tsmax(high,6)-close)/(tsmax-tsmin)*100,20,1)            [stochastic]
    def _(x): return SMA((TSMAX(x['H'],6)-x['C'])/(TSMAX(x['H'],6)-TSMIN(x['L'],6))*100, 20, 1)
    @reg("alpha083")   # -rank(cov(rank(high),rank(vol),5))                           [vol-price]
    def _(x): return -1*RANK(COV(RANK(x['H']), RANK(x['V']), 5))
    @reg("alpha084")   # sum(vol signed by close vs delay,20)                         [vol-price]
    def _(x):
        c,dc=x['C'],DELAY(x['C'],1)
        s=pd.DataFrame(np.where(c>dc,x['V'],np.where(c<dc,-x['V'],0)),index=c.index,columns=c.columns)
        return TSSUM(s,20)
    @reg("alpha085")   # tsrank(vol/mean(vol,20),20)*tsrank(-delta(close,7),8)        [vol-price]
    def _(x): return TSRANK(x['V']/MA(x['V'],20),20)*TSRANK(-1*DELTA(x['C'],7),8)
    @reg("alpha086")   # triple-slope reversal flag
    def _(x):
        s=(DELAY(x['C'],20)-DELAY(x['C'],10))/10 - (DELAY(x['C'],10)-x['C'])/10
        return pd.DataFrame(np.where(s>0.25,-1.0,np.where(s<0,1.0,-1*(x['C']-DELAY(x['C'],1)))),index=x['C'].index,columns=x['C'].columns)
    @reg("alpha088")   # (close-delay(close,20))/delay(close,20)*100                  [momentum]
    def _(x): return (x['C']-DELAY(x['C'],20))/DELAY(x['C'],20)*100
    @reg("alpha089")   # 2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))  [MACD]
    def _(x):
        macd=SMA(x['C'],13,2)-SMA(x['C'],27,2); return 2*(macd-SMA(macd,10,2))
    @reg("alpha093")   # sum(max(open-low, open-delay(open)) if open<delay,20)        [downside]
    def _(x):
        cond=x['O']>=DELAY(x['O'],1)
        v=MAX(x['O']-x['L'], x['O']-DELAY(x['O'],1))
        return TSSUM(pd.DataFrame(np.where(cond,0,v),index=x['O'].index,columns=x['O'].columns),20)
    @reg("alpha095")   # std(amount,20)                                               [liquidity vol]
    def _(x): return STD(x['AMT'],20)
    @reg("alpha096")   # sma(sma((close-tsmin(low,9))/(tsmax-tsmin)*100,3,1),3,1)     [stochastic KDJ]
    def _(x):
        k=(x['C']-TSMIN(x['L'],9))/(TSMAX(x['H'],9)-TSMIN(x['L'],9))*100; return SMA(SMA(k,3,1),3,1)
    @reg("alpha098")   # regime switch on ma(close,100) slope
    def _(x):
        d=DELTA(TSSUM(x['C'],100)/100,100)/DELAY(x['C'],100)
        cond=(d<=0.05)
        return pd.DataFrame(np.where(cond,-(x['C']-TSMIN(x['C'],100)),-DELTA(x['C'],3)),index=x['C'].index,columns=x['C'].columns)
    @reg("alpha100")  # std(vol,20)                                                   [vol]
    def _(x): return STD(x['V'],20)
    @reg("alpha102")  # sma(max(vol-delay,0),6,1)/sma(abs(vol-delay),6,1)*100         [vol RSI]
    def _(x):
        d=x['V']-DELAY(x['V'],1); return SMA(MAX(d,d*0),6,1)/SMA(ABS(d),6,1)*100
    @reg("alpha103")  # (20-lowday(low,20))/20*100                                    [position of low]
    def _(x): return (20-LOWDAY(x['L'],20))/20*100
    @reg("alpha104")  # -delta(corr(high,vol,5),5)*rank(std(close,20))                [vol-price]
    def _(x): return -1*DELTA(CORR(x['H'], x['V'], 5),5)*RANK(STD(x['C'],20))
    @reg("alpha105")  # -corr(rank(open),rank(vol),10)                                [vol-price]
    def _(x): return -1*CORR(RANK(x['O']), RANK(x['V']), 10)
    @reg("alpha106")  # close-delay(close,20)                                         [momentum]
    def _(x): return x['C']-DELAY(x['C'],20)
    @reg("alpha109")  # sma(high-low,10,2)/sma(sma(high-low,10,2),10,2)               [range]
    def _(x): return SMA(x['H']-x['L'],10,2)/SMA(SMA(x['H']-x['L'],10,2),10,2)
    @reg("alpha110")  # sum(max(0,high-delay(close)),20)/sum(max(0,delay(close)-low),20)*100
    def _(x):
        dc=DELAY(x['C'],1); a=MAX(x['H']-dc,x['H']*0); b=MAX(dc-x['L'],x['L']*0)
        return TSSUM(a,20)/TSSUM(b,20)*100
    @reg("alpha111")  # sma(vol*((c-l)-(h-c))/(h-l),11,2)-sma(...,4,2)                [vol-price]
    def _(x):
        z=x['V']*((x['C']-x['L'])-(x['H']-x['C']))/(x['H']-x['L']); return SMA(z,11,2)-SMA(z,4,2)
    @reg("alpha112")  # (sum up - sum down)/(sum up + sum down)*100 on close moves     [momentum]
    def _(x):
        d=x['C']-DELAY(x['C'],1)
        up=TSSUM(pd.DataFrame(np.where(d>0,d,0),index=d.index,columns=d.columns),12)
        dn=TSSUM(pd.DataFrame(np.where(d<0,-d,0),index=d.index,columns=d.columns),12)
        return (up-dn)/(up+dn)*100
    @reg("alpha114")  # rank(delay((h-l)/(sum(close,5)/5),2))*rank(rank(vol))/((h-l)/(sum(close,5)/5)/(vwap-close)) [vol-price]
    def _(x):
        hl=(x['H']-x['L'])/(TSSUM(x['C'],5)/5); return RANK(DELAY(hl,2))*RANK(RANK(x['V']))/(hl/((x['VWAP']-x['C']).replace(0,np.nan)))
    @reg("alpha116")  # tsrank slope of close (proxy: -delta over 20)                 [momentum]
    def _(x): return -1*DELTA(x['C'],20)/DELAY(x['C'],20)
    @reg("alpha117")  # tsrank(vol,32)*(1-tsrank(close+high-low,16))*(1-tsrank(ret,32)) [vol-price]
    def _(x): return TSRANK(x['V'],32)*(1-TSRANK(x['C']+x['H']-x['L'],16))*(1-TSRANK(x['RET'],32))
    @reg("alpha118")  # sum(high-open,20)/sum(open-low,20)*100                        [intraday]
    def _(x): return TSSUM(x['H']-x['O'],20)/TSSUM(x['O']-x['L'],20)*100
    @reg("alpha120")  # rank(vwap-close)/rank(vwap+close)                             [vwap]
    def _(x): return RANK(x['VWAP']-x['C'])/RANK(x['VWAP']+x['C'])
    @reg("alpha122")  # sma^3(log(close)) momentum                                    [momentum]
    def _(x):
        s=SMA(SMA(SMA(LOG(x['C']),13,2),13,2),13,2); return (s-DELAY(s,1))/DELAY(s,1)
    @reg("alpha126")  # (close+high+low)/3                                            [typical price level, reversal input]
    def _(x): return ((x['C']+x['H']+x['L'])/3 - x['C'])/x['C']
    @reg("alpha129")  # sum(abs(close-delay) if close<delay,12)                       [downside vol]
    def _(x):
        d=x['C']-DELAY(x['C'],1); return TSSUM(pd.DataFrame(np.where(d<0,ABS(d),0),index=d.index,columns=d.columns),12)
    @reg("alpha133")  # (20-highday(high,20))/20*100 - (20-lowday(low,20))/20*100     [position]
    def _(x): return (20-HIGHDAY(x['H'],20))/20*100 - (20-LOWDAY(x['L'],20))/20*100
    @reg("alpha139")  # -corr(open,vol,10)                                            [vol-price]
    def _(x): return -1*CORR(x['O'], x['V'], 10)
    @reg("alpha145")  # (mean(vol,9)-mean(vol,26))/mean(vol,12)*100                   [vol trend]
    def _(x): return (MA(x['V'],9)-MA(x['V'],26))/MA(x['V'],12)*100
    @reg("alpha150")  # (close+high+low)/3*vol                                        [vol-weighted price]
    def _(x): return (x['C']+x['H']+x['L'])/3*x['V']
    @reg("alpha152")  # sma(mean(delay(sma(delay(close/delay(close,9),1),9,1),1),12)-mean(...,26),9,1)  [vol trend]
    def _(x):
        z=SMA(DELAY(x['C']/DELAY(x['C'],9),1),9,1); return SMA(MA(DELAY(z,1),12)-MA(DELAY(z,1),26),9,1)
    @reg("alpha158")  # (high-low)/close                                             [range]
    def _(x): return (x['H']-x['L'])/x['C']
    @reg("alpha159")  # money-flow style oscillator
    def _(x):
        tl=MIN(x['L'],DELAY(x['C'],1)); th=MAX(x['H'],DELAY(x['C'],1))
        return (x['C']-TSSUM(tl,6))/TSSUM(th-tl,6)*100
    @reg("alpha161")  # mean(max(max(h-l,abs(delay(close)-h)),abs(delay(close)-l)),12) [ATR]
    def _(x):
        dc=DELAY(x['C'],1); tr=MAX(MAX(x['H']-x['L'],ABS(dc-x['H'])),ABS(dc-x['L'])); return MA(tr,12)
    @reg("alpha167")  # sum(max(close-delay,0),12)                                    [upside accum]
    def _(x):
        d=x['C']-DELAY(x['C'],1); return TSSUM(pd.DataFrame(np.where(d>0,d,0),index=d.index,columns=d.columns),12)
    @reg("alpha169")  # sma of dif of sma(close-delay)                                [trend]
    def _(x):
        z=SMA(x['C']-DELAY(x['C'],1),9,1); return SMA(MA(DELAY(z,1),12)-MA(DELAY(z,1),26),10,1)
    @reg("alpha170")  # vol-price interaction
    def _(x): return (RANK(1/x['C'])*x['V']/MA(x['V'],20)) * (x['H']*RANK(x['H']-x['C'])/(TSSUM(x['H'],5)/5)) - RANK(x['VWAP']-DELAY(x['VWAP'],5))
    @reg("alpha171")  # -(low-close)*(open^5)/((close-high)*(close^5))                 [intraday]
    def _(x): return -1*(x['L']-x['C'])*(x['O']**5)/(((x['C']-x['H'])*(x['C']**5)).replace(0,np.nan))
    @reg("alpha175")  # mean(max(max(h-l,abs(delay(close)-h)),abs(delay(close)-l)),6)  [ATR6]
    def _(x):
        dc=DELAY(x['C'],1); tr=MAX(MAX(x['H']-x['L'],ABS(dc-x['H'])),ABS(dc-x['L'])); return MA(tr,6)
    @reg("alpha176")  # corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))), rank(vol),6)  [vol-price]
    def _(x):
        k=(x['C']-TSMIN(x['L'],12))/(TSMAX(x['H'],12)-TSMIN(x['L'],12)); return CORR(RANK(k), RANK(x['V']), 6)
    @reg("alpha177")  # (20-highday(high,20))/20*100                                  [position of high]
    def _(x): return (20-HIGHDAY(x['H'],20))/20*100
    @reg("alpha178")  # (close-delay(close,1))/delay(close,1)*volume                  [vol-weighted reversal]
    def _(x): return (x['C']-DELAY(x['C'],1))/DELAY(x['C'],1)*x['V']
    @reg("alpha184")  # rank(corr(delay(open-close,1),close,200-ish)) + rank(open-close)  [vol-price]
    def _(x): return RANK(CORR(DELAY(x['O']-x['C'],1), x['C'], 60)) + RANK(x['O']-x['C'])
    @reg("alpha187")  # sum(if open<=delay(open) 0 else max(high-open,open-delay(open)),20)  [upside]
    def _(x):
        cond=x['O']<=DELAY(x['O'],1); v=MAX(x['H']-x['O'],x['O']-DELAY(x['O'],1))
        return TSSUM(pd.DataFrame(np.where(cond,0,v),index=x['O'].index,columns=x['O'].columns),20)
    @reg("alpha189")  # mean(abs(close-mean(close,6)),6)                              [dispersion]
    def _(x): return MA(ABS(x['C']-MA(x['C'],6)),6)
    return F

FACTORS = _factors()

def compute_all(ohlc):
    """ohlc: long df with symbol, open_time, open, high, low, close, volume, quote_volume.
    Memory-safe: reindex every factor to a FIXED master (open_time,symbol) index, keep only value
    arrays, concat ONCE at the end (no incremental outer-joins). gc between factors."""
    import gc
    ohlc = ohlc.copy(); ohlc["open_time"] = pd.to_datetime(ohlc["open_time"], utc=True)
    ohlc = ohlc.sort_values(["open_time","symbol"]).reset_index(drop=True)
    master = pd.MultiIndex.from_frame(ohlc[["open_time","symbol"]])   # exact panel grid, fixed target
    piv = lambda col: ohlc.pivot(index="open_time", columns="symbol", values=col).astype("float32")
    O,H,L,C,V = piv("open"),piv("high"),piv("low"),piv("close"),piv("volume")
    AMT = piv("quote_volume"); VWAP = (AMT/V.replace(0,np.nan))
    RET = C.pct_change()
    x = dict(O=O,H=H,L=L,C=C,V=V.replace(0,np.nan),AMT=AMT,VWAP=VWAP,RET=RET)
    cols = {}
    for name, fn in FACTORS.items():
        try:
            m = fn(x).replace([np.inf,-np.inf], np.nan)
            m = m.shift(1).astype("float32")             # PIT: at decision open_time t, use only CLOSED bars <= t-1
            s = m.stack()                                # Series, MultiIndex (open_time,symbol); reindex re-adds NA
            cols[name] = s.reindex(master).to_numpy()     # align to fixed grid, keep values only
            del m, s
        except Exception as e:
            print(f"  [skip {name}: {e}]", flush=True)
        gc.collect()
    out = ohlc[["open_time","symbol"]].copy()
    for name, arr in cols.items(): out[name] = arr
    return out

if __name__ == "__main__":
    df = pd.read_parquet("/home/yuqing/ctaNew/data/ml/cache/alpha191_ohlc4h.parquet")
    print(f"loaded {df['symbol'].nunique()} syms, {len(df)} rows; computing {len(FACTORS)} factors...")
    feats = compute_all(df)
    feats.to_parquet("/home/yuqing/ctaNew/data/ml/cache/alpha191_factors.parquet")
    print(f"-> {len([c for c in feats.columns if c.startswith('alpha')])} factors, {len(feats)} rows saved.")
