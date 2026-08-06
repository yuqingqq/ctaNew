"""Close the scope caveat: does the NONLINEAR (LGBM) stack extract anything from the 12 'dead' features
that the linear Ridge pipeline couldn't? Pooled LGBM (+sym_id), same V0 panel / cuts / purge / embargo as
the linear validation. Compare full-14 vs 2-factor (idio_vol + return_1d), both eras, day-clustered CIs +
paired diff. If full − 2factor spans 0 for LGBM too -> the 12 features are dead weight on the full stack.
Run: python3 -u -m live.validate_lgbm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from live.v0_feature_ablation import build_panel, perbar_ic, paired_ci, V0, RECENT_CUTS, OOS_CUTS

EMB = pd.Timedelta(days=1)
VOL, MOM = "idio_vol_to_btc_1d", "return_1d"
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.03, num_leaves=63,
              max_depth=8, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=5, lambda_l2=3.0, verbose=-1, seed=0, feature_fraction_seed=0,
              bagging_seed=0, data_random_seed=0)


def gen_lgbm(PAN, feats, cuts):
    cols = list(feats) + ["sym_id"]
    recs = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN.z_res.notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if len(tr) < 5000 or te.empty:
            continue
        ccal = tr.open_time.quantile(0.9)
        trn, cal = tr[tr.open_time <= ccal], tr[tr.open_time > ccal]
        dtr = lgb.Dataset(trn[cols], trn.z_res, categorical_feature=["sym_id"], free_raw_data=False)
        dcal = lgb.Dataset(cal[cols], cal.z_res, reference=dtr, free_raw_data=False)
        m = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dcal],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        recs.append(pd.DataFrame({"open_time": te.open_time.values, "pred": m.predict(te[cols]),
                                  "alpha_A": te.alpha_vs_btc_realized.values}))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def day_ci(ic, nb=3000, seed=1):
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True))
    s["d"] = s.index.floor("1D")
    g = [x["v"].values for _, x in s.groupby("d")]
    rng = np.random.default_rng(seed); k = len(g)
    b = [np.concatenate([g[i] for i in rng.integers(0, k, k)]).mean() for _ in range(nb)]
    return float(ic.mean()), *np.percentile(b, [2.5, 97.5])


def main():
    PAN = build_panel()
    PAN["sym_id"] = pd.factorize(PAN["symbol"])[0]
    print(f"panel {len(PAN):,} rows | pooled LGBM, full-14 vs 2-factor, both eras\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        f14 = perbar_ic(gen_lgbm(PAN, list(V0), cuts))
        f2 = perbar_ic(gen_lgbm(PAN, [VOL, MOM], cuts))
        m14, l14, h14 = day_ci(f14); m2, l2, h2 = day_ci(f2)
        d, lo, hi = paired_ci(f14, f2)
        verdict = "12 feats DEAD on LGBM too (validated)" if lo < 0 < hi else "LGBM USES the 12 feats"
        print(f"===== {era} =====", flush=True)
        print(f"    LGBM full-14  {m14:+.4f} [{l14:+.4f},{h14:+.4f}]", flush=True)
        print(f"    LGBM 2-factor {m2:+.4f} [{l2:+.4f},{h2:+.4f}]", flush=True)
        print(f"    full − 2factor: {d:+.4f} [{lo:+.4f},{hi:+.4f}]  → {verdict}\n", flush=True)
    print("LGBMDONE", flush=True)


if __name__ == "__main__":
    main()
