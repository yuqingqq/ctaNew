"""Freeze builder for candidate pm_ladder_v1 (USER 2026-11-06: "we freeze and tell the coding session to implement this version").
Candidate (ladder engine pf_step4_ladder.py): MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95; control
MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06 (docs sections 55, 60).
Builds the one artifact the replay never stored: the guard models. Same fit as pf_step5_guard_preds for a fresh day (LightGBM, pf_model_spec
PARAMS / ROUNDS pinned, features SETS['ALL'], target v per side clipped +-10, all rows of step4/badfill_info_tx_tk/grid.parquet = the 8
read days), threshold c = the 60% quantile of the in-sample prediction per side; the engine's guard value is v = prediction - c.
Writes predictfun/freeze/pm_ladder_v1/: guard_bid.txt, guard_ask.txt (LightGBM text models), params.json (thresholds, features in model
order, per-minute lines 90 (pull + ladder line) and 95 (size line), LightGBM params, input hashes).
Checks (each refuses): the grid holds exactly the 8 read days; the thresholds equal guard_preds_report_t_60.json's 11-03 fit; the saved and
reloaded models reproduce step5/20261103/preds_t60.parquet v_bid / v_ask exactly (float32, NaN pattern equal) on all 11-03 rows; falsifier:
a threshold moved by 1e-6 must fail that parity. Reads no day after 11-03.
Usage: python3 -m predictfun.research.pf_freeze_pm_ladder_v1
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; PAR_DAY = '20261103'; K = 60
GRID = f'{io.OUT}/step4/badfill_info_tx_tk/grid.parquet'; LINES = f'{io.OUT}/step4/minute_lines_t60k.json'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'freeze', 'pm_ladder_v1')
FE = SETS['ALL']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    G = pd.read_parquet(GRID, columns=['day', 'v_bid', 'v_ask'] + FE)
    if sorted(G.day.unique()) != READ:
        sys.exit(f'REFUSED: grid days {sorted(G.day.unique())} != the 8 read days')
    os.makedirs(OUT, exist_ok=True)
    X = G[FE].values.astype(np.float32); ms, cs = {}, {}
    for side in ('bid', 'ask'):
        ds = lgb.Dataset(X, G[f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
        m = lgb.train(PARAMS, ds, ROUNDS); del ds
        cs[side] = float(np.quantile(m.predict(X, num_threads=1), K / 100.0))
        m.save_model(f'{OUT}/guard_{side}.txt'); ms[side] = lgb.Booster(model_file=f'{OUT}/guard_{side}.txt')
    n_rows = len(G); del G, X
    ref = json.load(open(f'{io.OUT}/step5/guard_preds_report_t_60.json'))[PAR_DAY]['thresholds']
    if any(cs[s] != ref[s][str(K)] for s in cs):
        sys.exit(f'REFUSED: thresholds {cs} != guard_preds_report_t_60.json {ref}')
    r = pd.read_parquet(f'{io.OUT}/step5/{PAR_DAY}/rows_rl_d2_pk.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
    P = pd.read_parquet(f'{io.OUT}/step5/{PAR_DAY}/preds_t60.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    assert len(P) == len(r) and (P.mid.values == r.mid.values).all() and np.allclose(P.tau.values, r.tau.values), 'row alignment'
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values; Xd = r[FE].values.astype(np.float32)
    vh = {s: np.where(live, ms[s].predict(Xd, num_threads=1), np.nan) for s in ('bid', 'ask')}

    def parity(c):
        bad = {}
        for s in ('bid', 'ask'):
            a = (vh[s] - c[s]).astype(np.float32); b = P[f'v_{s}'].values
            if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])):
                ok = ~np.isnan(a) & ~np.isnan(b)
                bad[s] = dict(nan_equal=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs=float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None)
        return bad
    bad = parity(cs)
    if bad:
        sys.exit(f'REFUSED: saved models do not reproduce preds_t60 on {PAR_DAY}: {bad}')
    fz = parity({'bid': cs['bid'] + 1e-6, 'ask': cs['ask']})
    if not fz:
        sys.exit('FALSIFIER DID NOT FIRE: parity passed a moved threshold')
    L = json.load(open(LINES))
    out = dict(candidate='pm_ladder_v1', engine='predictfun/research/pf_step4_ladder.py',
               arm='MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95',
               control='MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06',
               guard=dict(models={s: f'guard_{s}.txt' for s in ms}, model_sha256={s: sha(f'{OUT}/guard_{s}.txt') for s in ms},
                          features=FE, feature_names_in_model={s: ms[s].feature_name() for s in ms}, threshold_c60=cs,
                          value='v_side = model_side(features) - threshold_c60[side]; live rows only (t_rel < 240 s and a known Predict mid); '
                                'stale after 100 ms; rows before the window start price is known: v = -1 (pf_preds_startknown)',
                          lgb_params=PARAMS, rounds=ROUNDS, target='v per side clipped +-10', train_rows=n_rows, train_days=READ),
               minute_lines=dict(source=LINES, sha256=sha(LINES), source_note=L.get('source'), minutes=L.get('minutes'),
                                 pull_and_ladder_line_90=L['lines']['90'], size_line_95=L['lines']['95']),
               inputs_sha256=dict(grid=sha(GRID)), parity=dict(day=PAR_DAY, rows=int(len(r)), live_rows=int(live.sum()), equal=True,
                                                               falsifier_fired=fz))
    io.write_json(f'{OUT}/params.json', out)
    print(json.dumps({k: out[k] for k in ('guard', 'minute_lines', 'parity')}, default=str)[:3000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
