"""Why does the take model fit the days it is trained on and not another day? (USER 2026-11-26: "Can you check why it can't generalize on
untouched data"). Readout over take_model's labels and features; no new data. Target v_172_0 (the take at our 172 ms path).
Declared before the run:
1 capacity: a ridge regression (median-imputed, standardised, alpha 10), small trees (100 iterations, 4 leaves, >= 1000 per leaf), the
  pinned trees (take_model's), large trees (800 iterations, 63 leaves, >= 50 per leaf, no L2): top-1 % realised value and Spearman,
  in-sample (fitted and scored on all days) and leave-one-day-out.
2 splits (pinned trees): within a day across markets (each day's markets split 50 / 50 by a seeded draw, each half scored by the model of
  the other half); one day -> another day (4 x 4, the diagonal in-sample); leave-one-day-out.
3 what generalises (pinned trees, leave-one-day-out): the fill share, the Predict mid's move from the decision to 1 s after the match
  (mv_172, unconditional) and the value; held-out Spearman; the realised move and value of the top 1 % by predicted move.
4 concentration: distinct markets among the top 1 % (oracle, in-sample, leave-one-day-out) and the ten largest markets' share of it.
Output (write-once): take_model_why.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/take_model_why.py [<rehearsal dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_model as TM  # noqa: E402
import take_model_insample as TI  # noqa: E402

C, TR = TM.C, TM.TR
TG, FL, MV = 'v_172_0', 'fill_172_0', 'mv_172'
SEED = 20261126
MODELS = dict(ridge=lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.)),
              trees_small=lambda: HistGradientBoostingRegressor(random_state=0, max_iter=100, learning_rate=.05, max_leaf_nodes=4, min_samples_leaf=1000, l2_regularization=1.),
              trees_pinned=lambda: HistGradientBoostingRegressor(random_state=0, **TM.HGB),
              trees_large=lambda: HistGradientBoostingRegressor(random_state=0, max_iter=800, learning_rate=.05, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=0.))


def top1(pred, real, q=.01):
    o = np.argsort(-pred, kind='stable')[:max(1, int(round(q * len(pred))))]
    return float(np.mean(real[o])), o


def lodo(X, feats, y, mk):
    pred = np.full(len(X), np.nan)
    for dh in TM.DAYS:
        tr, te = (X.day != dh).to_numpy(), (X.day == dh).to_numpy()
        if te.sum() == 0:
            continue
        pred[te] = mk().fit(X.loc[tr, feats], y[tr]).predict(X.loc[te, feats])
    return pred


def conc(X, idx):
    mids = X.mid.to_numpy()[idx]
    vc = pd.Series(mids).value_counts()
    return dict(n=int(len(idx)), markets=int(vc.size), top10_markets_share=float(vc.iloc[:10].sum() / len(idx)))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'take_model_why.json'
    C.require(not fn.exists(), 'take_model_why.json exists (write-once)')
    X, feats = TI.frame()
    y, fill, mv = X[TG].to_numpy(float), X[FL].to_numpy(float), X[MV].to_numpy(float)
    out = dict(asof='2026-11-26', rows=int(len(X)), capacity={}, splits={}, components={}, concentration={})
    # 1 capacity
    for nm, mk in MODELS.items():
        pin = mk().fit(X[feats], y).predict(X[feats])
        plo = lodo(X, feats, y, mk)
        out['capacity'][nm] = dict(insample_top1=top1(pin, y)[0], insample_spearman=float(spearmanr(pin, y).correlation),
                                   lodo_top1=top1(plo, y)[0], lodo_spearman=float(spearmanr(plo, y).correlation))
        print(nm, out['capacity'][nm], flush=True)
    # 2 splits (pinned)
    mk = MODELS['trees_pinned']
    rng = np.random.default_rng(SEED)
    wd = np.full(len(X), np.nan)
    for dh in TM.DAYS:
        m = (X.day == dh).to_numpy()
        mids = np.unique(X.mid.to_numpy()[m])
        half = set(rng.choice(mids, len(mids) // 2, replace=False).tolist())
        a = m & X.mid.isin(half).to_numpy(); b = m & ~X.mid.isin(half).to_numpy()
        wd[b] = mk().fit(X.loc[a, feats], y[a]).predict(X.loc[b, feats])
        wd[a] = mk().fit(X.loc[b, feats], y[b]).predict(X.loc[a, feats])
    out['splits']['within_day_across_markets'] = dict(top1=top1(wd, y)[0], spearman=float(spearmanr(wd, y).correlation),
                                                      by_day={dh: top1(wd[(X.day == dh).to_numpy()], y[(X.day == dh).to_numpy()])[0] for dh in TM.DAYS})
    mat = {}
    for da in TM.DAYS:
        ma = (X.day == da).to_numpy()
        mdl = mk().fit(X.loc[ma, feats], y[ma])
        mat[da] = {}
        for db in TM.DAYS:
            mb = (X.day == db).to_numpy()
            pb = mdl.predict(X.loc[mb, feats])
            mat[da][db] = top1(pb, y[mb])[0]
    out['splits']['day_to_day_top1'] = mat
    plo = lodo(X, feats, y, mk)
    out['splits']['leave_one_day_out'] = dict(top1=top1(plo, y)[0], spearman=float(spearmanr(plo, y).correlation))
    # 3 components
    pf = lodo(X, feats, fill, mk)
    pm = lodo(X, feats, mv, mk)
    tm_, om = top1(pm, mv)
    out['components'] = dict(fill_spearman=float(spearmanr(pf, fill).correlation), move_spearman=float(spearmanr(pm, mv).correlation),
                             value_spearman=float(spearmanr(plo, y).correlation), top1_by_predicted_move=dict(realised_move_c=tm_, realised_value_c=float(np.mean(y[om])), fill=float(np.mean(fill[om]))),
                             move_all_c=float(mv.mean()), move_top1_oracle_c=top1(mv, mv)[0])
    pin = mk().fit(X[feats], y).predict(X[feats])
    out['concentration'] = dict(oracle=conc(X, top1(y, y)[1]), insample=conc(X, top1(pin, y)[1]), leave_one_day_out=conc(X, top1(plo, y)[1]), markets_total=int(X.mid.nunique()))
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
