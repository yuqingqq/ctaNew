"""B13a (2026-11-11, USER "We pick some days data to train and validate on other days, to use the imbalance features and
flow features, it looks the model does not reflect these changes well, need to emphasize on these features"). Does a
guard model that emphasises book pressure and flow predict Predict's next-second move better than the current one,
out of sample by day? Read-only on the data; writes models and a report.

Days: the 9 days with step5 rows that may be read: 10-23, 10-24, 10-25, 10-26, 10-27, 10-31, 11-01, 11-02, 11-03
(data/predict_fun/research/step5/<day>/rows.parquet; 288 windows x 6,000 grid points of 50 ms). Not used: the protected
days and every later day; 11-08 (the B12 day the pressure idea came from) is not a validation day. Blocked 3-fold
cross-validation by day: A = 10-23 / 10-24 / 10-25, B = 10-26 / 10-27 / 10-31, C = 11-01 / 11-02 / 11-03; each fold's
model is trained on the other six days and predicts its three, so every day is out of sample once (G = 9 days).
Models: LightGBM with the pinned step-5 PARAMS / ROUNDS (pf_model_spec), target y1000 (Predict mid change over the next
1 s, clipped to +-5 c), rows t_rel < 240 s with a known mid and target; training uses every 4th row (runtime on one
core), evaluation every 2nd row. BASE = the current feature set ALL (32). EMPH = ALL + 13 engineered pressure features,
computed per window from trailing data only:
  d_imb1_250, d_imb1_1000      change of Predict's L1 imbalance over 250 ms / 1 s
  d_bnimb_250, d_bnimb_1000    change of Binance's best bid / ask imbalance over 250 ms / 1 s
  thin_flag                    +1 if the ask L1 size <= half the bid's, -1 if the bid's <= half the ask's, else 0
  bid_rel, ask_rel             log1p(L1 size) minus its trailing 30 s mean (depletion of each side)
  l1_min_rel                   the thinner side's log1p(L1 size) minus its trailing 30 s mean
  bn_act_rel                   log1p(Binance trades in 1 s) minus its trailing 30 s mean
  agree                        pf_imb1 x bn_imb (both books pressing one way)
  imb_mom                      pf_imb1 x pf_dm250 (imbalance with Predict's last 250 ms move)
  bn_press                     bn_imb x log1p(bn_ntr1000) (Binance pressure with activity)
  flow_press                   bn_flow250 x bn_imb
Metrics per out-of-sample day: R2 = 1 - sum (y - p)^2 / sum y^2 (as pf_step5_apply), Pearson r. Pressure rows: |pf_imb1|
>= 0.6 or bn_ntr1000 >= that day's 90th percentile. Residual dependence: corr(y - p, pf_imb1) and corr(y - p, bn_imb).

Readings DECLARED BEFORE THE FIRST RUN (day-clustered, G = 9; 95% interval = mean +- t(8) x sd / 3):
 R90 skill      dR2 = R2(EMPH) - R2(BASE) per day, all evaluation rows: IMPROVES if the interval's low end > 0 and >= 7 of
                9 days are positive; WORSE if its high end < 0; else NO-CHANGE.
 R91 pressure   the same rule on the pressure rows: IMPROVES-UNDER-PRESSURE / WORSE-UNDER-PRESSURE / NO-CHANGE-UNDER-PRESSURE.
 R92 residual   mean over days of |corr(y - p, pf_imb1)| and of |corr(y - p, bn_imb)|: n/a (nothing to absorb) if BASE's are
                both < 0.02; ABSORBED if EMPH's are both <= 0.5 x BASE's; else NOT-ABSORBED.
Descriptive: per-day R2 and r for both models; LightGBM gain shares of the pressure features in BASE and of the 13
engineered features in EMPH.
Controls (any failure -> no table): the features on a planted frame (shifts and trailing means do not cross windows or
look ahead; thin_flag signs); the fold split (no day in both train and test); the pipeline on planted data (a signal
carried only by an engineered feature -> IMPROVES; no such signal -> NO-CHANGE); the label rules (NaN reads n/a); every
day has 1,728,000 rows and the spec's columns.
Limits fixed in advance: point estimates per day with a 9-day interval; one target (y1000); a better prediction of the
mid is not yet a better guard (the guard maps predictions to per-side values; B13 later parts); the 9 days were read by
earlier research (in-sample for the current model's design), but neither model's features or settings are chosen on
them here.
Usage: predictfun/research/run_light.sh b13a_pressure_model -- python3 -m predictfun.research.pf_b13a_pressure_model <out_dir>
       python3 -m predictfun.research.pf_b13a_pressure_model --selftest
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .pf_model_spec import SETS, PARAMS, ROUNDS

ROOT = '/home/yuqing/ctaNew/data/predict_fun/research/step5'
FOLDS = {'A': ['20261023', '20261024', '20261025'], 'B': ['20261026', '20261027', '20261031'], 'C': ['20261101', '20261102', '20261103']}
DAYS = [d for f in FOLDS.values() for d in f]
BASE = SETS['ALL']
EXTRA = ['d_imb1_250', 'd_imb1_1000', 'd_bnimb_250', 'd_bnimb_1000', 'thin_flag', 'bid_rel', 'ask_rel', 'l1_min_rel', 'bn_act_rel',
         'agree', 'imb_mom', 'bn_press', 'flow_press']
EMPH = BASE + EXTRA
PRESS_BASE = ['pf_imb1', 'pf_imb3', 'pf_lbs', 'pf_las', 'pf_micro_dev', 'bn_imb', 'bn_flow250', 'bn_flow1000', 'bn_flow5000', 'bn_ntr1000']
N_ROWS = 1728000; ROLL = 600; TRAIN_STEP = 4; EVAL_STEP = 2
NAN = float('nan')


def engineer(r):
    """The 13 engineered features, per window (column 'mid'), trailing only. r must be sorted by (mid, tau)."""
    g = r.groupby('mid', sort=False)
    out = pd.DataFrame(index=r.index)
    for c, lag, name in (('pf_imb1', 5, 'd_imb1_250'), ('pf_imb1', 20, 'd_imb1_1000'), ('bn_imb', 5, 'd_bnimb_250'), ('bn_imb', 20, 'd_bnimb_1000')):
        out[name] = r[c] - g[c].shift(lag)
    bs = np.expm1(r.pf_lbs.values); as_ = np.expm1(r.pf_las.values)
    out['thin_flag'] = np.where(as_ <= 0.5 * bs, 1.0, np.where(bs <= 0.5 * as_, -1.0, 0.0))
    out.loc[~(np.isfinite(bs) & np.isfinite(as_)), 'thin_flag'] = np.nan
    tmp = pd.DataFrame(dict(mid=r.mid.values, lb=r.pf_lbs.values, la=r.pf_las.values, lm=np.minimum(r.pf_lbs.values, r.pf_las.values),
                            lt=np.log1p(r.bn_ntr1000.values)), index=r.index)
    gt = tmp.groupby('mid', sort=False)
    for c, name in (('lb', 'bid_rel'), ('la', 'ask_rel'), ('lm', 'l1_min_rel'), ('lt', 'bn_act_rel')):
        out[name] = tmp[c] - gt[c].transform(lambda s: s.rolling(ROLL, min_periods=20).mean())
    out['agree'] = r.pf_imb1 * r.bn_imb; out['imb_mom'] = r.pf_imb1 * r.pf_dm250
    out['bn_press'] = r.bn_imb * np.log1p(r.bn_ntr1000); out['flow_press'] = r.bn_flow250 * r.bn_imb
    return out[EXTRA].astype(np.float32)


def load_day(d, step):
    cols = list(dict.fromkeys(['mid', 'tau'] + BASE + ['y1000', 't_rel', 'pf_mid']))
    r = pd.read_parquet(f'{ROOT}/{d}/rows.parquet', columns=cols)
    if len(r) != N_ROWS:
        raise ValueError(f'{d}: {len(r)} rows, expected {N_ROWS}')
    r = r.sort_values(['mid', 'tau'], kind='stable').reset_index(drop=True)
    ex = engineer(r)
    ok = ((r.t_rel < 240) & r.pf_mid.notna() & r.y1000.notna()).values
    idx = np.nonzero(ok)[0][::step]
    X = np.concatenate([r[BASE].values[idx].astype(np.float32), ex.values[idx]], axis=1)
    y = r.y1000.values[idx].clip(-5, 5).astype(np.float32)
    aux = dict(imb1=r.pf_imb1.values[idx].astype(np.float32), bnimb=r.bn_imb.values[idx].astype(np.float32), ntr=r.bn_ntr1000.values[idx].astype(np.float32))
    del r, ex
    return X, y, aux


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum(y ** 2)) if len(y) else NAN


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else NAN


def interval(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return NAN, NAN, NAN
    m = x.mean(); h = 2.306 * x.std(ddof=1) / np.sqrt(len(x))     # t(8) 97.5% for G = 9
    return float(m), float(m - h), float(m + h)


def label_skill(d, names):
    m, lo, hi = interval(d); npos = int(np.sum(np.asarray(d) > 0))
    if not np.isfinite(m):
        return 'n/a'
    return names[0] if (lo > 0 and npos >= 7) else (names[1] if hi < 0 else names[2])


def label_r92(b1, b2, e1, e2):
    if not all(np.isfinite(v) for v in (b1, b2, e1, e2)):
        return 'n/a'
    if b1 < 0.02 and b2 < 0.02:
        return 'n/a (nothing to absorb)'
    return 'ABSORBED' if (e1 <= 0.5 * b1 and e2 <= 0.5 * b2) else 'NOT-ABSORBED'


def gain_shares(m, names):
    g = m.feature_importance(importance_type='gain'); g = g / g.sum() if g.sum() > 0 else g
    return {n: float(v) for n, v in zip(names, g)}


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    # F1 engineered features on a planted frame: two windows, shifts must not cross them
    n = 30; r = pd.DataFrame({c: np.zeros(2 * n) for c in BASE})
    r['mid'] = np.repeat([1, 2], n); r['tau'] = np.tile(np.arange(n) * 50.0, 2)
    r['pf_imb1'] = np.tile(np.arange(n, dtype=float) / 100.0, 2); r['bn_imb'] = 0.5; r['pf_dm250'] = 2.0
    r['pf_lbs'] = np.log1p(np.r_[np.full(n, 100.0), np.full(n, 10.0)]); r['pf_las'] = np.log1p(np.r_[np.full(n, 10.0), np.full(n, 100.0)])
    r['bn_ntr1000'] = 3.0; r['bn_flow250'] = 4.0
    e = engineer(r)
    f1 = np.isnan(e.d_imb1_250.iloc[n]) and abs(e.d_imb1_250.iloc[n + 5] - 0.05) < 1e-6 and np.isnan(e.d_imb1_1000.iloc[19])
    f1 = f1 and e.thin_flag.iloc[0] == 1.0 and e.thin_flag.iloc[n] == -1.0 and abs(e.agree.iloc[3] - 0.015) < 1e-6
    f1 = f1 and abs(e.imb_mom.iloc[3] - 0.06) < 1e-6 and abs(e.bid_rel.iloc[25]) < 1e-6 and np.isnan(e.bid_rel.iloc[n + 5])
    chk('F1_features', f1)
    tr = {k: [d for d in DAYS if d not in v] for k, v in FOLDS.items()}
    chk('F2_folds', all(not (set(tr[k]) & set(v)) and len(tr[k]) == 6 for k, v in FOLDS.items()) and sorted(DAYS) == sorted(set(DAYS)) and len(DAYS) == 9)
    # F3 pipeline on planted data: the signal lives only in an engineered column
    rng = np.random.default_rng(3); res = []
    for sig in (1.0, 0.0):
        dd = []
        for day in range(9):
            Xb = rng.normal(0, 1, (6000, 5)).astype(np.float32); xe = rng.normal(0, 1, 6000).astype(np.float32)
            y = (0.3 * Xb[:, 0] + sig * np.tanh(2 * xe) + rng.normal(0, 1, 6000)).astype(np.float32)
            dd.append((Xb, xe, y))
        out = []
        for k in range(3):
            te = range(3 * k, 3 * k + 3); trn = [i for i in range(9) if i not in te]
            Xtr = np.concatenate([dd[i][0] for i in trn]); xtr = np.concatenate([dd[i][1] for i in trn]); ytr = np.concatenate([dd[i][2] for i in trn])
            pr = dict(PARAMS, min_data_in_leaf=50)
            mb = lgb.train(pr, lgb.Dataset(Xtr, ytr), 60); me = lgb.train(pr, lgb.Dataset(np.c_[Xtr, xtr], ytr), 60)
            for i in te:
                out.append(r2(dd[i][2], me.predict(np.c_[dd[i][0], dd[i][1]])) - r2(dd[i][2], mb.predict(dd[i][0])))
        res.append(label_skill(out, ('IMPROVES', 'WORSE', 'NO-CHANGE')))
    chk('F3_pipeline', res == ['IMPROVES', 'NO-CHANGE'])
    chk('F4_rules', label_skill([NAN] * 9, ('a', 'b', 'c')) == 'n/a' and label_skill([-0.01] * 9, ('a', 'b', 'c')) == 'b'
        and label_r92(0.01, 0.01, 0.0, 0.0) == 'n/a (nothing to absorb)' and label_r92(0.1, 0.1, 0.04, 0.05) == 'ABSORBED'
        and label_r92(0.1, 0.1, 0.06, 0.04) == 'NOT-ABSORBED' and label_r92(NAN, 0.1, 0.0, 0.0) == 'n/a')
    rep['F3_detail'] = res
    return ok, rep


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps(rep, indent=1)); print('selftest', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    nb = len(BASE); per_day = {}; gains = {}
    for k, test_days in FOLDS.items():
        train_days = [d for d in DAYS if d not in test_days]
        Xs, ys = [], []
        for d in train_days:
            X, y, _ = load_day(d, TRAIN_STEP); Xs.append(X); ys.append(y); print(f'fold {k}: loaded train {d} {len(y)}', flush=True)
        X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
        mb = lgb.train(PARAMS, lgb.Dataset(X[:, :nb], y, feature_name=BASE, free_raw_data=True), ROUNDS); print(f'fold {k}: BASE trained', flush=True)
        me = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=EMPH, free_raw_data=True), ROUNDS); print(f'fold {k}: EMPH trained', flush=True)
        del X, y
        mb.save_model(f'{out}/model_base_fold{k}.txt'); me.save_model(f'{out}/model_emph_fold{k}.txt')
        gains[k] = dict(base=gain_shares(mb, BASE), emph=gain_shares(me, EMPH))
        for d in test_days:
            X, y, aux = load_day(d, EVAL_STEP)
            pb = mb.predict(X[:, :nb], num_threads=1); pe = me.predict(X, num_threads=1)
            pr_ = (np.abs(aux['imb1']) >= 0.6) | (aux['ntr'] >= np.nanpercentile(aux['ntr'], 90))
            per_day[d] = dict(fold=k, rows=int(len(y)), pressure_rows=int(pr_.sum()),
                              r2_base=r2(y, pb), r2_emph=r2(y, pe), r_base=corr(y, pb), r_emph=corr(y, pe),
                              r2_base_press=r2(y[pr_], pb[pr_]), r2_emph_press=r2(y[pr_], pe[pr_]),
                              res_imb1_base=corr(y - pb, aux['imb1']), res_imb1_emph=corr(y - pe, aux['imb1']),
                              res_bnimb_base=corr(y - pb, aux['bnimb']), res_bnimb_emph=corr(y - pe, aux['bnimb']))
            print(f"{d}: R2 base {per_day[d]['r2_base']:.4f} emph {per_day[d]['r2_emph']:.4f}", flush=True)
            del X, y, aux
    dR2 = [per_day[d]['r2_emph'] - per_day[d]['r2_base'] for d in DAYS]
    dR2p = [per_day[d]['r2_emph_press'] - per_day[d]['r2_base_press'] for d in DAYS]
    b1 = float(np.mean([abs(per_day[d]['res_imb1_base']) for d in DAYS])); b2 = float(np.mean([abs(per_day[d]['res_bnimb_base']) for d in DAYS]))
    e1 = float(np.mean([abs(per_day[d]['res_imb1_emph']) for d in DAYS])); e2 = float(np.mean([abs(per_day[d]['res_bnimb_emph']) for d in DAYS]))
    readings = dict(R90=label_skill(dR2, ('IMPROVES', 'WORSE', 'NO-CHANGE')),
                    R91=label_skill(dR2p, ('IMPROVES-UNDER-PRESSURE', 'WORSE-UNDER-PRESSURE', 'NO-CHANGE-UNDER-PRESSURE')),
                    R92=label_r92(b1, b2, e1, e2))
    gsum = {k: dict(base_pressure_share=float(sum(v['base'][c] for c in PRESS_BASE)), emph_extra_share=float(sum(v['emph'][c] for c in EXTRA)),
                    emph_pressure_share=float(sum(v['emph'][c] for c in PRESS_BASE + EXTRA))) for k, v in gains.items()}
    res = dict(as_of_note='9 out-of-sample days by blocked 3-fold CV; day-clustered interval G = 9', controls=ctl, folds=FOLDS, per_day=per_day,
               dR2=dict(zip(DAYS, dR2)), dR2_interval=interval(dR2), dR2_pressure=dict(zip(DAYS, dR2p)), dR2_pressure_interval=interval(dR2p),
               residual_abs_corr=dict(base_imb1=b1, base_bnimb=b2, emph_imb1=e1, emph_bnimb=e2), gain=gains, gain_summary=gsum, readings=readings)
    json.dump(res, open(f'{out}/b13a.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not np.isfinite(v) else f'{v:+.4f}'
    L = ['B13a pressure-emphasised guard model against the current one (y1000), blocked 3-fold CV by day.',
         f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}", '',
         'day | fold | rows | R2 base | R2 emph | dR2 | r base | r emph | pressure rows | R2 base / emph on them | resid corr imb1 base / emph | resid corr bn_imb base / emph']
    for d in DAYS:
        x = per_day[d]
        L.append(f"{d} | {x['fold']} | {x['rows']} | {f(x['r2_base'])} | {f(x['r2_emph'])} | {f(x['r2_emph'] - x['r2_base'])} | {f(x['r_base'])} | {f(x['r_emph'])} | "
                 f"{x['pressure_rows']} | {f(x['r2_base_press'])} / {f(x['r2_emph_press'])} | {f(x['res_imb1_base'])} / {f(x['res_imb1_emph'])} | {f(x['res_bnimb_base'])} / {f(x['res_bnimb_emph'])}")
    m, lo, hi = res['dR2_interval']; mp, lop, hip = res['dR2_pressure_interval']
    L += [f"dR2 all rows: mean {f(m)} [{f(lo)}, {f(hi)}], positive days {sum(v > 0 for v in dR2)}/9",
          f"dR2 pressure rows: mean {f(mp)} [{f(lop)}, {f(hip)}], positive days {sum(v > 0 for v in dR2p)}/9",
          f"mean |resid corr|: imb1 base {b1:.4f} emph {e1:.4f}; bn_imb base {b2:.4f} emph {e2:.4f}",
          'gain shares: ' + '; '.join(f"fold {k}: BASE pressure features {v['base_pressure_share'] * 100:.1f}%, EMPH engineered {v['emph_extra_share'] * 100:.1f}% "
                                      f"(all pressure {v['emph_pressure_share'] * 100:.1f}%)" for k, v in gsum.items()),
          'top EMPH engineered features by gain (fold C): ' + ', '.join(f"{n} {v * 100:.1f}%" for n, v in sorted(((n, gains['C']['emph'][n]) for n in EXTRA), key=lambda z: -z[1])[:6]),
          '', 'Readings: ' + json.dumps(readings)]
    open(f'{out}/b13a.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
