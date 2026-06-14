"""Phase C (tail loop): is the bad-cycle tail PREDICTABLE from ANY free PIT feature? Decisive test.
Target y = cycle in worst-decile pnl. PIT features per cycle (all known at decision time t):
  squeeze_risk (=-mean lagged short-basket funding_z), btc_ret_30d (regime depth), |btc_ret_30d|,
  xs_ret_disp (cross-sectional std of return_1d, low=correlated regime), trailing realized portfolio vol (20-cyc),
  trailing portfolio return (20-cyc, momentum-of-strategy), pred_disp.
AUC per feature (univariate) + a logistic on all (walk-forward OOS). AUC≈0.5 => tail unpredictable.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
ROOT="/home/yuqing/ctaNew"; LAG=2

c=pd.read_csv(f"{ROOT}/live/state/v3loop/iter5_tilt0/cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
c=c.sort_values("open_time").reset_index(drop=True)
pred=pd.read_parquet(f"{ROOT}/live/state/v3loop/iter5_tilt0/predictions.parquet"); pred["open_time"]=pd.to_datetime(pred["open_time"],utc=True)
pan=pd.read_parquet(f"{ROOT}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","funding_rate_z_7d","return_1d"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fz_lag"]=pan.groupby("symbol")["funding_rate_z_7d"].shift(LAG)

sh=pred[pred["selected_short"]].merge(pan[["symbol","open_time","fz_lag"]],on=["symbol","open_time"],how="left")
c=c.merge(sh.groupby("open_time")["fz_lag"].mean().rename("short_fz"),on="open_time",how="left")
# xs return dispersion per cycle (PIT: return_1d uses trailing 1d, known at t)
disp=pan.groupby("open_time")["return_1d"].std().rename("xs_ret_disp")
c=c.merge(disp,on="open_time",how="left")
# build PIT features
c["squeeze_risk"]=-c["short_fz"]
c["abs_btc30"]=c["btc_ret_30d"].abs()
c["roll_vol20"]=(c["pnl_bps"]/1e4).shift(1).rolling(20).std()        # trailing realized portfolio vol (PIT)
c["roll_ret20"]=(c["pnl_bps"]/1e4).shift(1).rolling(20).sum()        # trailing strategy momentum (PIT)
y=(c["pnl_bps"] <= c["pnl_bps"].quantile(0.10)).astype(int)          # worst-decile cycle

FEATS=["squeeze_risk","btc_ret_30d","abs_btc30","xs_ret_disp","roll_vol20","roll_ret20","pred_disp"]
print(f"worst-decile base rate {y.mean():.2f}, n={y.sum()} bad cycles")
print("=== univariate AUC (>0.55 = some signal; ~0.50 = none) ===")
m=c[FEATS].notna().all(axis=1)
for f in FEATS:
    try:
        a=roc_auc_score(y[m], c[f][m]); a=max(a,1-a)
        print(f"  {f:14s} AUC {a:.3f}")
    except Exception as e: print(f"  {f}: {e}")

# walk-forward OOS logistic on all features (fit past, predict next block)
X=c[FEATS][m].values; yy=y[m].values
n=len(X); fold=n//5; oof=np.full(n,np.nan)
for i in range(1,5):
    tr=slice(0,i*fold); te=slice(i*fold,(i+1)*fold)
    Xtr=(X[tr]-X[tr].mean(0))/(X[tr].std(0)+1e-9); Xte=(X[te]-X[tr].mean(0))/(X[tr].std(0)+1e-9)
    lr=LogisticRegression(max_iter=500).fit(Xtr,yy[tr]); oof[te]=lr.predict_proba(Xte)[:,1]
mm=~np.isnan(oof)
print(f"=== multivariate walk-forward OOS AUC = {roc_auc_score(yy[mm],oof[mm]):.3f} ===")
