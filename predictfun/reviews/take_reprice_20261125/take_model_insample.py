"""How good can the take model look in hindsight? (USER 2026-11-26: "You mean even in sample, the performance is not good"). Readout over
take_model's labels and features; no new data.
Declared before the run, target v_172_0 (and v_152_0): (1) the oracle: the realised value of the top 1 / 0.3 / 0.1 % of moments ranked by
the realised value itself (what perfect foresight would pick); (2) in-sample: the same trees fitted on all four days and scored on the same
rows; (3) per day in-sample: a model fitted on one day and scored on that day; (4) take_model's leave-one-day-out result, read.
Output (write-once): take_model_insample.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/take_model_insample.py [<rehearsal dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_model as TM  # noqa: E402

C, TR, WT = TM.C, TM.TR, TM.WT


def frame():
    L = pd.concat([pd.read_parquet(TM.OUT / f'{d}_labels.parquet') for d in TM.DAYS], ignore_index=True)
    Pz = pd.read_parquet(WT.OUT / 'public_score3_points.parquet')
    Pz['row'] = np.arange(len(Pz))
    Pz = Pz[Pz.row.isin(set(L.row))]
    NWp = pd.concat([pd.read_parquet(WT.OUT / 'norm_windows' / f'{d}.parquet', columns=['day', 't', 'd', 'kind'] + TM.NW_COLS) for d in TM.DAYS], ignore_index=True)
    NWp = NWp[NWp.kind == 'random'].drop_duplicates(['day', 't', 'd'])
    X = L.merge(Pz, on=['row'], how='inner', suffixes=('', '_p'))
    C.require(len(X) == len(L) and np.allclose(X.t.to_numpy(), X.t_p.to_numpy()), 'labels and features do not align')
    X = X.merge(NWp.drop(columns=['kind']), on=['day', 't', 'd'], how='left')
    feats = [c for c in Pz.columns if c not in TM.EXCL and c != 'row'] + [c for c in L.columns if c.startswith('pf_')] + TM.NW_COLS
    return X, feats


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'take_model_insample.json'
    C.require(not fn.exists(), 'take_model_insample.json exists (write-once)')
    X, feats = frame()
    lo = json.loads((H / 'take_model.json').read_text())
    C.require(lo['features'] == feats, 'not take_model\'s feature set')
    out = dict(asof='2026-11-26', rows=int(len(X)), targets={})
    for tg in ('v_172_0', 'v_152_0'):
        fl = tg.replace('v_', 'fill_')
        real, fill = X[tg].to_numpy(float), X[fl].to_numpy(float)
        o = dict(all=float(real.mean()), oracle={}, insample={}, per_day_insample={}, leave_one_day_out={q: lo['targets'][tg]['model'][q]['realised_c'] for q in ('0.01', '0.003', '0.001')})
        for q in TM.QTOP:
            o['oracle'][str(q)] = TM.topk_eval(real, real, fill, q)
        mdl = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X[feats], X[tg])
        pin = mdl.predict(X[feats])
        for q in TM.QTOP:
            o['insample'][str(q)] = TM.topk_eval(pin, real, fill, q)
        for dh in TM.DAYS:
            m = (X.day == dh).to_numpy()
            if m.sum() < 1000:
                continue
            md = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[m, feats], X.loc[m, tg])
            pd_ = md.predict(X.loc[m, feats])
            o['per_day_insample'][dh] = TM.topk_eval(pd_, real[m], fill[m], .01)
        out['targets'][tg] = o
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
