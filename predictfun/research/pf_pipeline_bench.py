"""CHRONOLOGICAL PREDICTION BENCHMARK ON THE EXPORTED JOIN DECISIONS (2026-11-19, session aea5bfb9). USER: "proceed our pipeline model".
Steps 2-3 of the pipeline plan (handoffs/20261119_prediction_pipeline.md): "Freeze a chronological consumed-day benchmark (Oct23-25 train;
Oct26 calibration only when required by a declared candidate; Oct27 development evaluation). Separate bid/ask estimates, pinned LightGBM
parameters, explicit feature-only/mechanical and direct-value controls. Count all candidates. ... Judge fill/quantity calibration,
fill-weighted movement, severe loss amounts and net value separately, by side/day ... Gate mapping on amount calibration and economic
ranking, not fill AUC alone." Declared before any exported label is read; consumed days only; no economic cutoff is fitted.

DATA: pf_pipeline_export's days (every part's hash checked against its receipt). Rows: POST (status eligible or refused_post_only: a refusal
 is a zero-value outcome of posting) and RETAIN (status eligible: live at B). Training days Oct 23-25, every POST row and every 5th live step
 of each order's RETAIN rows ((j - J0) % 5 == 0: rows of one order overlap, so the thinning keeps the fit tractable and less redundant);
 evaluation day Oct 27 (the development evaluation), every row. ONE-DAY TEST (USER: "We run the whole pipeline for one day test first",
 declared before any exported label was read): Oct 27 is the only evaluation day; Oct 26 is neither fitted, calibrated nor read (no declared
 candidate needs a calibration step).
TARGETS at B (M50): q (affected quantity), qm (the fill-weighted signed move from the causal mid), value = s * (ref - p) * q + qm (exact,
 ref = pf_mid at the decision), severe excess (reported). M0 / M250 labels are never trained on.
CANDIDATES (per kind and side - separate bid / ask models; pinned pf_stop_ev_model.PARAMS, 200 rounds, objective binary or regression):
 J  JOINT TWO-PART (primary): P(fill) x E[q | fill], E[qm | fill] -> E[q] = P * E[q | fill] (E[q | fill] clipped to [0, the known remaining]),
    E[qm] = P * E[qm | fill]; mapped value = s * (ref - p) * E[q] + E[qm] (the contract's mapping: never E[q] * E[move]); plus a severe head
    (E[severe excess], reported only)
 D  DIRECT VALUE (control): one regression of value on J's inputs
 M  MECHANICAL (control): J on the mechanical inputs only (pf_flow_price_common.MECHANICAL: order / queue / time / book scalars)
 F  FAIR EDGE (the baseline's information): J on fair_edge and t_to_end only (A+0's input)
 Multiplicity: 4 candidates x 2 kinds.
READINGS (computed; per kind, side, evaluation day and candidate)
 Q1 AMOUNT: sum E[q] / sum q in [0.9, 1.1] -> CALIBRATED-IN-AMOUNT, else MISCALIBRATED-IN-AMOUNT (D has no quantity head: NOT-APPLICABLE)
 Q2 MOVE TERM: Spearman(E[qm], qm) > 0 and sum E[qm] / sum qm in [0.5, 2] -> MOVE-TERM-INFORMATIVE, else MOVE-TERM-NOT-SHOWN
 Q3 VALUE RANKING: Spearman(mapped value, value) above F's on the same rows -> RANKS-BEYOND-FAIR-EDGE, else NOT-BEYOND-FAIR-EDGE (F: BASELINE)
 Q4 LOSING DECILE: the realized mean value of the lowest predicted-value decile < 0 -> LOSING-DECILE-IDENTIFIED, else NOT-IDENTIFIED
 printed: P(fill) AUC, the severe amount ratio, value deciles (predicted vs realized means), counts
 MAPPING GATE (per kind, J): Q1 CALIBRATED-IN-AMOUNT and Q3 RANKS-BEYOND-FAIR-EDGE for both sides on the evaluation day -> PASS, else FAIL.
 The keep / pull policy test (pf_pipeline_policy.py, through the simulator's scenario bank) uses the rule declared here: pull at the first live
 step whose predicted RETAIN value (J) is below 0 - no threshold fitted. For the one-day test it runs whatever the gate reads (the USER asked
 for the whole pipeline end to end) and carries the gate's reading: under a FAIL its values are not attributable to a calibrated mapping.
CONTROLS (any failure -> no output): B1 chronology (training rows only from the training days, evaluation rows only from the evaluation
 days, every export part matching its receipt); B2 no label among the inputs (no q_ / value_ / qm_ / severe_ / nofill_ / live_ column, no
 mark / effective / endpoint field); B3 the mapping identity recomputed on every evaluation row; B4 every saved model reloads and reproduces
 its predictions on sampled rows; B5 bounds (0 <= P <= 1, 0 <= E[q] <= known remaining).
LIMITS: consumed days - development evidence, not validation (C8 / C16); per-decision statistics: the rows of one order overlap, so the
 calibration and ranking tables are per decision, not per action, and economic value is judged only through the simulator (one action per
 order); queue rule M (C5); replay fills not validated on live orders (C10); one order at a time.
SELFTEST (synthetic, no data): planted relationships recovered (J ranks value beyond an uninformative F; amounts calibrated), every reading
 on planted predictions (and edge cases), the gate on planted readings, the mapping identity; refusals by their own control with no output:
 an evaluation day in training (B1), a changed export part (B1), a label among the inputs (B2), the mapping replaced by a product of means
 (B3), a model file changed after fitting (B4), an expected quantity above the known remaining (B5).
 Mutations: pf_pipeline_bench_mutants.py.
v3 (review 1adf6a33 P2, declared before v3 ran): a forecast is coherent only if the two-atom forecast rebuilt from its moments - (P, E[q] / P,
 E[q * move] / E[q]) and (1 - P, 0, 0) - gives back E[q], E[q * move] and the mapped value. v2 set the move to 0 when E[q] = 0 and accepted a
 nonzero E[q * move] there (an impossible joint distribution acting on its value). Nothing else changes: the same exports, training rows,
 pinned parameters, seed and single thread, so the refit must reproduce v2's model files byte for byte (checked by
 reviews/pipeline_one_day_v3_20261119/compare_bench_v2_v3.py, declared before v3 ran). Counted per kind / side / day: the rows refused by the
 round trip alone (zero_quantity_refused). The selftest plants both signs (E[q * move] -100 and +100 c at E[q] = 0) and a coherent no-fill.
Usage (run_light, PF_SLOT=claude): pf_pipeline_bench.py selftest | run
"""
import contextlib
import gc
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_flow_price_common as FC                         # noqa: E402
import pf_pipeline_export as EX                           # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_bench_v3_20261119'        # v3 (review 1adf6a33 P2): coherence by the moments' round trip; v2 stays as provenance
MARK = 'PIPELINE BENCH'
TRAIN, EVAL = ('20261023', '20261024', '20261025'), ('20261026', '20261027')   # v2: Oct 26 is the economic gate's day of record (never evaluated
GATE_DAY = '20261026'                                                           # before); Oct 27 was seen in v1, so it is reported post hoc
KINDS, THIN = ('post', 'retain'), 5
SETS = {'J': list(EX.INPUTS), 'D': list(EX.INPUTS), 'M': list(FC.MECHANICAL), 'F': ['fair_edge', 't_to_end']}
LABELS = ('q_', 'value_', 'qm_', 'severe_', 'nofill_', 'live_')
FORBIDDEN = ('mark', 'effective_ms', 'endpoint_ms', 'decision_ms', 'oid', 't', 'status')
ROUNDS = 200
PARAMS = dict(KP.C.V.PARAMS)                              # the pipeline's pinned LightGBM parameters (pf_stop_ev_model.PARAMS)
MAP_PRODUCT = False                                       # planted-error switches (selftest only): the mapping as a product of means;
SWAP_SAVED = False                                        # a saved model file replaced after fitting
NO_CLIP = False                                           # E[q | fill] not clipped to the known remaining;
V2_COHERENCE = False                                      # v2's coherence flag, without the round trip (review 1adf6a33 P2)


COLS = ['day', 'oid', 'kind', 'j', 'side', 't', 'decision_ms', 'p', 'status', 'q_m50', 'qm_m50', 'value_m50', 'severe_m50']   # + every input


def rows_of(exp, days, kind, train, side=None):
    """The exported rows of a kind (and side) on the given days, only the columns used, inputs as float32 (their parts verified by
    verify_exports); training rows are thinned."""
    out = []; keep = ['eligible', 'refused_post_only'] if kind == 'post' else ['eligible']
    filt = [('kind', '==', kind), ('status', 'in', keep)] + ([('side', '==', int(side))] if side is not None else [])
    cols = list(dict.fromkeys(COLS + [x for fs in SETS.values() for x in fs]))
    for d in days:
        r = json.loads((exp / d / 'receipt.json').read_text())
        for x in r['outputs']:                                                              # (hashes verified once, before anything is written)
            g = pd.read_parquet(exp / d / x, columns=cols, filters=filt)
            if train and kind == 'retain':
                g = g[(g.j - KP.J0) % THIN == 0]
            fs_ = [c for c in cols if c not in COLS]; g[fs_] = g[fs_].astype(np.float32); out.append(g)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    C.require(len(df) > 0 and set(df.day.astype(str)) <= set(days), f'B1: {kind} rows outside the declared days {days}')
    return df


def verify_exports(exp, days):
    """B1: every export part of the days matches its receipt - checked before the benchmark writes anything."""
    for d in days:
        r = json.loads((exp / d / 'receipt.json').read_text())
        for x, h in r['outputs'].items():
            C.require(C.sha(exp / d / x) == h, f'B1 {d}: export part {x} changed since its receipt')


def fit_one(X, y, objective, feats, rounds=None):
    import lightgbm as lgb
    return lgb.train({**PARAMS, 'objective': objective}, lgb.Dataset(np.asarray(X, np.float32), label=np.asarray(y, float), feature_name=feats), num_boost_round=rounds or ROUNDS)


def fit(df, cand, rounds=None):
    """One candidate on one kind and side's training rows -> {head: booster}."""
    feats = SETS[cand]; X = df[feats].to_numpy(np.float32); q = df.q_m50.to_numpy(float); filled = q > 1e-12
    if cand == 'D':
        return {'value': fit_one(X, df.value_m50, 'regression', feats, rounds)}
    heads = {'p': fit_one(X, filled.astype(float), 'binary', feats, rounds), 'q': fit_one(X[filled], q[filled], 'regression', feats, rounds),
             'qm': fit_one(X[filled], df.qm_m50.to_numpy(float)[filled], 'regression', feats, rounds)}
    if cand == 'J':
        heads['severe'] = fit_one(X, df.severe_m50, 'regression', feats, rounds)
    return heads


def predict(models, df, cand):
    """Predicted moments and the mapped value on rows of one kind and side."""
    X = df[SETS[cand]].to_numpy(np.float32); sg = np.where(df.side.values == 0, 1., -1.); edge = sg * (df.pf_mid.values.astype(float) - df.p.values)
    if cand == 'D':
        v = models['value'].predict(X, num_threads=1); return dict(V=v)
    P = models['p'].predict(X, num_threads=1); qr = models['q'].predict(X, num_threads=1); qf = qr if NO_CLIP else np.clip(qr, 0., df.fp_remaining.values.astype(float))
    Eq = P * qf; Eqm = P * models['qm'].predict(X, num_threads=1)
    if MAP_PRODUCT:                                                                         # planted error: the product of marginal means
        Eqm = Eq * (models['qm'].predict(X, num_threads=1) / np.maximum(qf, 1e-9)) * P
    out = dict(P=P, Eq=Eq, Eqm=Eqm, V=edge * Eq + Eqm)
    mv = np.where(Eq > 0, Eqm / np.where(Eq > 0, Eq, 1.), 0.); end = df.pf_mid.values.astype(float) + sg * mv      # v2: the contract's two-atom check
    out['ok'] = ((P >= 0) & (P <= 1) & (qf >= 0) & (qf <= df.fp_remaining.values.astype(float)) & (end >= -1e-9) & (end <= 100. + 1e-9)
                 & np.isfinite(Eq) & np.isfinite(Eqm) & (df.pf_mid.values >= 0) & (df.pf_mid.values <= 100) & (df.fp_remaining.values > 0))
    # v3 (review 1adf6a33 P2): the two-atom forecast rebuilt from the moments - (P, E[q] / P, the move per share), (1 - P, 0, 0) - must give back
    # E[q], E[q * move] and the mapped value; a zero expected quantity therefore carries a zero E[q * move] (zq: the rows refused for that alone)
    rq = P * np.where(P > 0, Eq / np.where(P > 0, P, 1.), 0.); rqm = rq * mv; tol = lambda x: 1e-9 * np.maximum(1., np.abs(x))
    rt = (np.abs(rq - Eq) <= tol(Eq)) & (np.abs(rqm - Eqm) <= tol(Eqm)) & (np.abs(edge * rq + rqm - out['V']) <= tol(out['V']))
    out['zq'] = out['ok'] & ~rt
    if not V2_COHERENCE:
        out['ok'] = out['ok'] & rt
    if 'severe' in models:
        out['S'] = models['severe'].predict(X, num_threads=1)
    return out


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float)); return float(a.rank().corr(b.rank())) if len(a) > 2 else float('nan')


def auc(y, p):
    y = np.asarray(y, bool); r = pd.Series(np.asarray(p, float)).rank().values; n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def read_one(df, pr, base):
    """The readings of one candidate on one kind / side / day (base: F's predicted value on the same rows)."""
    q, qm, v = df.q_m50.values.astype(float), df.qm_m50.values.astype(float), df.value_m50.values.astype(float); out = {}
    if 'Eq' in pr:
        ratio = float(pr['Eq'].sum() / max(q.sum(), 1e-12)); out['Q1 AMOUNT'] = 'CALIBRATED-IN-AMOUNT' if .9 <= ratio <= 1.1 else 'MISCALIBRATED-IN-AMOUNT'
        sq = float(qm.sum()); rq = float(pr['Eqm'].sum() / sq) if abs(sq) > 1e-12 else float('nan'); rho_m = spearman(pr['Eqm'], qm)
        out['Q2 MOVE TERM'] = 'MOVE-TERM-INFORMATIVE' if rho_m > 0 and .5 <= rq <= 2. else 'MOVE-TERM-NOT-SHOWN'
        out.update(amount_ratio=ratio, move_ratio=rq, move_rank=rho_m, fill_auc=auc(q > 1e-12, pr['P']))
    else:
        out['Q1 AMOUNT'] = out['Q2 MOVE TERM'] = 'NOT-APPLICABLE'
    rho = spearman(pr['V'], v); rho_f = spearman(base, v)
    out['Q3 VALUE RANKING'] = 'BASELINE' if base is pr['V'] else ('RANKS-BEYOND-FAIR-EDGE' if rho > rho_f else 'NOT-BEYOND-FAIR-EDGE')
    dec = pd.qcut(pd.Series(pr['V']).rank(method='first'), 10, labels=False) if len(v) >= 10 else pd.Series(np.zeros(len(v), int))
    tab = pd.DataFrame(dict(d=dec.values, pv=pr['V'], rv=v)).groupby('d').agg(pv=('pv', 'mean'), rv=('rv', 'mean'), n=('rv', 'size'))
    out['Q4 LOSING DECILE'] = 'LOSING-DECILE-IDENTIFIED' if float(tab.rv.iloc[0]) < 0 else 'NOT-IDENTIFIED'
    out.update(value_rank=rho, fair_edge_rank=rho_f, deciles=tab.reset_index().to_dict('list'), n=int(len(v)),
               severe_ratio=float(pr['S'].sum() / max(df.severe_m50.sum(), 1e-12)) if 'S' in pr else None)
    return out


def bench(exp=None, out=None, rounds=None):
    """Fit on the training days, predict and read on the evaluation days -> out/bench.json / bench.txt / models / pred_<day>.parquet."""
    start = time.time(); exp = Path(exp or EX.OUT); out = Path(out or OUT); C.require(not (out / 'bench.json').exists(), 'bench output exists')
    bad = [f'B2 {c}: label or identity column {x} among the inputs' for c, fs in SETS.items() for x in fs if x.startswith(LABELS) or x in FORBIDDEN]
    bad += [f'B1: training day {d} is not before every evaluation day' for d in TRAIN if not all(d < e for e in EVAL)]
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad))
    verify_exports(exp, TRAIN + EVAL); part = out.parent / (out.name + '.partial')
    if part.exists():
        shutil.rmtree(part)
    (part / 'models').mkdir(parents=True); R = {}; Ec = {k: {} for k in KINDS}; preds = {d: [] for d in EVAL}; files = {}
    for kind in KINDS:                                                                     # one kind and side at a time (memory: RSS self-guard)
        R[kind] = {}
        for s in (0, 1):
            trs = rows_of(exp, TRAIN, kind, True, side=s); models = {}
            for cand in SETS:
                models[cand] = fit(trs, cand, rounds)
                for h, b in models[cand].items():
                    f = part / 'models' / f'{kind}_{s}_{cand}_{h}.txt'; b.save_model(str(f)); files[f.name] = f
                if SWAP_SAVED and cand == 'J':
                    models[cand]['qm'].save_model(str(part / 'models' / f'{kind}_{s}_J_q.txt'))
            del trs; gc.collect(); C.guard('pipeline bench fit')
            for d in EVAL:
                e = rows_of(exp, (d,), kind, False, side=s).reset_index(drop=True); pr = {cand: predict(models[cand], e, cand) for cand in SETS}
                sg = np.where(e.side.values == 0, 1., -1.); edge = sg * (e.pf_mid.values.astype(float) - e.p.values)
                for cand in ('J', 'M', 'F'):                                               # B3 the mapping re-derived from the raw heads; B5 bounds
                    p_ = pr[cand]; Xc = e[SETS[cand]].to_numpy(np.float32); m_ = models[cand]; P_ = m_['p'].predict(Xc, num_threads=1)
                    qf_ = np.clip(m_['q'].predict(Xc, num_threads=1), 0., e.fp_remaining.values.astype(float))
                    if not np.allclose(p_['V'], edge * P_ * qf_ + P_ * m_['qm'].predict(Xc, num_threads=1), rtol=0, atol=1e-9):
                        bad.append(f'B3 {kind} {s} {d} {cand}: the mapped value is not s * (ref - p) * E[q] + E[q * move] of its heads')
                    if not ((p_['P'] >= 0).all() and (p_['P'] <= 1).all() and (p_['Eq'] >= -1e-12).all() and (p_['Eq'] <= e.fp_remaining.values + 1e-9).all()):
                        bad.append(f'B5 {kind} {s} {d} {cand}: a probability or expected quantity out of bounds')
                R[kind].setdefault(s, {})[d] = {cand: read_one(e, pr[cand], pr['F']['V']) for cand in SETS}
                Ec[kind].setdefault(s, {})[d] = econ_one(kind, e, pr['J'])
                keep = e[['day', 'oid', 'kind', 'j', 'side', 't', 'decision_ms', 'p', 'pf_mid', 'fp_remaining', 'q_m50', 'qm_m50', 'value_m50']].copy()
                for cand in SETS:
                    for k_, v_ in pr[cand].items():
                        keep[f'{cand}_{k_}'] = v_
                preds[d].append(keep)
                ix = np.linspace(0, len(e) - 1, min(256, len(e)), dtype=int)                 # B4 saved models reload and reproduce
                for cand in SETS:
                    for h in models[cand]:
                        import lightgbm as lgb
                        b = lgb.Booster(model_file=str(part / 'models' / f'{kind}_{s}_{cand}_{h}.txt'))
                        xs = e[SETS[cand]].iloc[ix].to_numpy(np.float32)
                        if not np.allclose(b.predict(xs, num_threads=1), models[cand][h].predict(xs, num_threads=1), rtol=0, atol=1e-10):
                            bad.append(f'B4 {kind} {s} {cand} {h}: the saved model does not reproduce its predictions')
                del e, pr; gc.collect(); C.guard('pipeline bench evaluate')
            del models; gc.collect()
        print(json.dumps(dict(stage='bench', kind=kind, seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    if bad:
        shutil.rmtree(part); sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad[:12]))
    diag, gate = diag_of(R), gate_of(Ec)
    for d in EVAL:
        pd.concat(preds[d], ignore_index=True).to_parquet(part / f'pred_{d}.parquet', index=False)
    L = report_lines(R, gate, diag, Ec)
    (part / 'bench.txt').write_text('\n'.join(L) + '\n')
    C.dump(part / 'bench.json', dict(asof=C.stamp(), readings=R, economic=Ec, gate=gate, diagnostic=diag, gate_day=GATE_DAY, train=list(TRAIN), eval=list(EVAL), thin=THIN, params=PARAMS, rounds=ROUNDS,
                                     sets={k: v for k, v in SETS.items()}, multiplicity=len(SETS) * len(KINDS),
                                     exports={d: C.sha(exp / d / 'receipt.json') for d in TRAIN + EVAL}, models={k: C.sha(v) for k, v in files.items()},
                                     outputs={x: C.sha(part / x) for x in ['bench.txt'] + [f'pred_{d}.parquet' for d in EVAL]},
                                     source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    part.rename(out); print('\n'.join(L), flush=True)
    return R, gate, diag, Ec


def diag_of(R):
    """v2: the rank / quantity DIAGNOSTIC (v1's gate, review 89caa74c P2): J calibrated in amount and ranking beyond the fair edge, both sides,
    every evaluation day. Ranking ignores a constant offset, so this is not an economic gate."""
    return {kind: 'PASS' if all(R[kind][s][d]['J']['Q1 AMOUNT'] == 'CALIBRATED-IN-AMOUNT' and R[kind][s][d]['J']['Q3 VALUE RANKING'] == 'RANKS-BEYOND-FAIR-EDGE'
                                for s in (0, 1) for d in EVAL) else 'FAIL' for kind in KINDS}


def ratio_in(a, b, lo=.8, hi=1.25):
    return bool(b != 0 and np.sign(a) == np.sign(b) and lo <= a / b <= hi)


def econ_one(kind, df, pr):
    """v2 ECONOMIC MAPPING readings of J on one kind / side / day, at the rule's own boundary (declared before v2 ran): RETAIN acts (pulls)
    where the predicted keep value is below 0, POST acts (posts) where the predicted value is above 0; incoherent forecasts never act.
    E1 VALUE LEVEL: predicted / realized total value of the coherent decisions in [0.8, 1.25] (same sign).
    E2 BOUNDARY SIGN: RETAIN - the pulled decisions realize a loss (mean < 0) and the kept ones do not (mean >= 0); POST - the posted ones do not
       lose (mean >= 0) and the skipped ones would (mean < 0); an empty side of the boundary is NOT-EVALUABLE.
    E3 LOSS AMOUNT: predicted / realized total value of the decisions on the loss side (RETAIN pulled, POST skipped) in [0.8, 1.25]."""
    v = df.value_m50.values.astype(float); V = np.asarray(pr['V'], float); ok = np.asarray(pr['ok'], bool)
    act = ok & ((V < 0) if kind == 'retain' else (V > 0)); rest = ok & ~act; loss = act if kind == 'retain' else rest; gain = rest if kind == 'retain' else act
    e2 = ('NOT-EVALUABLE' if not (loss.any() and gain.any()) else
          'PASS' if float(v[loss].mean()) < 0 <= float(v[gain].mean()) else 'FAIL')
    return {'E1 VALUE LEVEL': 'PASS' if ratio_in(V[ok].sum(), v[ok].sum()) else 'FAIL', 'E2 BOUNDARY SIGN': e2,
            'E3 LOSS AMOUNT': 'PASS' if loss.any() and ratio_in(V[loss].sum(), v[loss].sum()) else 'FAIL',
            'acting': int(act.sum()), 'resting': int(rest.sum()), 'incoherent': int((~ok).sum()),
            'zero_quantity_refused': int(np.asarray(pr.get('zq', np.zeros(len(V), bool)), bool).sum()),   # v3: refused by the round trip alone
            'loss_side_realized_mean': float(v[loss].mean()) if loss.any() else None, 'gain_side_realized_mean': float(v[gain].mean()) if gain.any() else None,
            'value_level_ratio': float(V[ok].sum() / v[ok].sum()) if v[ok].sum() != 0 else None}


def gate_of(E):
    """v2: the ECONOMIC MAPPING GATE per kind - E1, E2 and E3 PASS for both sides on the gate's day of record (Oct 26)."""
    return {kind: 'PASS' if all(E[kind][s][GATE_DAY][k] == 'PASS' for s in (0, 1) for k in ('E1 VALUE LEVEL', 'E2 BOUNDARY SIGN', 'E3 LOSS AMOUNT'))
            else 'FAIL' for kind in KINDS}


def report_lines(R, gate, diag, Ec):
    L = ['CHRONOLOGICAL PREDICTION BENCHMARK v3 (pf_pipeline_bench.py): train ' + ', '.join(TRAIN) + '; evaluate ' + ', '.join(EVAL) + f' (consumed; per decision, not per action); '
         f'the economic gate\'s day of record {GATE_DAY}, the other day post hoc']
    for kind in KINDS:
        L.append(f'  {kind.upper()} - ECONOMIC MAPPING GATE (J, {GATE_DAY}): {gate[kind]}; rank / quantity diagnostic (v1\'s gate): {diag[kind]}')
        for s in (0, 1):
            for d in EVAL:
                e = Ec[kind][s][d]
                L.append(f"    side {s} {d} J economic{' (gate day)' if d == GATE_DAY else ' (post hoc)'}: {e['E1 VALUE LEVEL']} / {e['E2 BOUNDARY SIGN']} / {e['E3 LOSS AMOUNT']}; "
                         f"acting {e['acting']:,}, resting {e['resting']:,}, incoherent {e['incoherent']:,} (zero quantity {e['zero_quantity_refused']:,}); realized mean loss side "
                         f"{e['loss_side_realized_mean'] if e['loss_side_realized_mean'] is None else round(e['loss_side_realized_mean'], 3)}, gain side "
                         f"{e['gain_side_realized_mean'] if e['gain_side_realized_mean'] is None else round(e['gain_side_realized_mean'], 3)}; value level "
                         f"{e['value_level_ratio'] if e['value_level_ratio'] is None else round(e['value_level_ratio'], 3)}")
                for cand, x in R[kind][s][d].items():
                    L.append(f"    side {s} {d} {cand}: {x['Q1 AMOUNT']}, {x['Q2 MOVE TERM']}, {x['Q3 VALUE RANKING']}, {x['Q4 LOSING DECILE']}; value rank {x['value_rank']:+.4f} "
                             f"(fair edge {x['fair_edge_rank']:+.4f}); amount {x.get('amount_ratio', float('nan')):.3f}, move {x.get('move_ratio', float('nan')):.3f}, "
                             f"fill AUC {x.get('fill_auc', float('nan')):.3f}, n {x['n']:,}")
                dj = R[kind][s][d]['J']['deciles']
                L.append(f"    side {s} {d} J value deciles (predicted / realized mean c): " + ' | '.join(f'{a:+.2f}/{b:+.2f}' for a, b in zip(dj['pv'], dj['rv'])))
    L.append('LIMITS: consumed days; per-decision tables (rows of one order overlap); economic value only through the simulator; queue rule M; no live validation.')
    return L


def synthetic(exp, n=2000, seed=7):
    """Export-format parts for the five days with planted structure: P(fill) rises with input 0, the fill size with input 1, the move per
    share falls with input 2 (adverse); the fair edge and t_to_end are noise (an uninformative baseline)."""
    rng = np.random.default_rng(seed); names = list(EX.INPUTS)
    for d in TRAIN + EVAL:
        rows = []
        for kind in KINDS:
            X = rng.normal(0., 1., (n, len(names))).astype(np.float32); side = rng.integers(0, 2, n); sg = np.where(side == 0, 1., -1.)
            df = pd.DataFrame(X, columns=names); df['pf_mid'] = 50. + rng.normal(0., .2, n); p = df.pf_mid.values - sg * .5
            fill = rng.random(n) < 1. / (1. + np.exp(-2. * X[:, 0])); q = np.where(fill, np.where(X[:, 1] > 0, 20., 8.), 0.)
            move = -2. * X[:, 2] + rng.normal(0., .5, n); qm = q * move; value = sg * (df.pf_mid.values - p) * q + qm
            df = df.assign(day=d, oid=np.arange(n) + (0 if kind == 'post' else n), m=1, side=side, t=0., p=p, phase=1, mark=50., kind=kind,
                           j=np.where(kind == 'post', 0, KP.J0 + THIN * rng.integers(0, 19, n)), decision_ms=0., effective_ms=0., endpoint_ms=0.,
                           status='eligible', fp_remaining=30., q_m50=q, qm_m50=qm, value_m50=value, severe_m50=np.maximum(-value - KP.SEVERE, 0.),
                           nofill_m50=(q == 0).astype(float), live_m50=True)
            rows.append(df)
        (exp / d).mkdir(parents=True); pd.concat(rows, ignore_index=True).to_parquet(exp / d / 'rows_000.parquet', index=False)
        C.dump(exp / d / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(exp / d / 'rows_000.parquet')}))


def selftest():
    """Synthetic only (see the module's SELFTEST)."""
    global TRAIN, MAP_PRODUCT, SWAP_SAVED, NO_CLIP
    bad = []
    q = np.array([0., 10., 20., 0., 30.] * 20); qm = np.array([0., -5., 8., 0., -40.] * 20); v = qm + .5 * q
    df = pd.DataFrame(dict(q_m50=q, qm_m50=qm, value_m50=v, severe_m50=np.maximum(-v - 30., 0.)))
    good = read_one(df, dict(P=(q > 0) * .9 + .05, Eq=1.05 * q, Eqm=qm, V=v), np.random.default_rng(1).normal(0, 1, len(v)))
    poor = read_one(df, dict(P=np.full(len(q), .5), Eq=2. * q, Eqm=-qm, V=-v), v)
    if not ((good['Q1 AMOUNT'], good['Q2 MOVE TERM'], good['Q3 VALUE RANKING'], good['Q4 LOSING DECILE']) == ('CALIBRATED-IN-AMOUNT', 'MOVE-TERM-INFORMATIVE', 'RANKS-BEYOND-FAIR-EDGE', 'LOSING-DECILE-IDENTIFIED')
            and (poor['Q1 AMOUNT'], poor['Q2 MOVE TERM'], poor['Q3 VALUE RANKING'], poor['Q4 LOSING DECILE']) == ('MISCALIBRATED-IN-AMOUNT', 'MOVE-TERM-NOT-SHOWN', 'NOT-BEYOND-FAIR-EDGE', 'NOT-IDENTIFIED')
            and read_one(df, dict(V=v), v)['Q1 AMOUNT'] == 'NOT-APPLICABLE' and abs(auc(np.r_[0, 0, 1, 1], np.r_[.1, .2, .3, .4]) - 1.) < 1e-12):
        bad.append(f'planted readings: {good} / {poor}')
    up = np.arange(1., 101.); d2 = pd.DataFrame(dict(q_m50=np.ones(100), qm_m50=up, value_m50=np.r_[np.zeros(10), up[10:]], severe_m50=np.zeros(100)))
    base = np.r_[np.zeros(10), up[10:]]; tie = read_one(d2, dict(P=np.ones(100), Eq=np.ones(100), Eqm=3. * up, V=base.copy()), base)
    rev = read_one(d2, dict(P=np.ones(100), Eq=np.ones(100), Eqm=up[::-1].copy(), V=np.arange(100.)), base)
    if not (tie['Q2 MOVE TERM'] == 'MOVE-TERM-NOT-SHOWN' and rev['Q2 MOVE TERM'] == 'MOVE-TERM-NOT-SHOWN' and tie['Q3 VALUE RANKING'] == 'NOT-BEYOND-FAIR-EDGE'
            and rev['Q4 LOSING DECILE'] == 'NOT-IDENTIFIED'):
        bad.append(f"planted edge cases: Q2 {tie['Q2 MOVE TERM']} / {rev['Q2 MOVE TERM']}, Q3 tie {tie['Q3 VALUE RANKING']}, Q4 zero {rev['Q4 LOSING DECILE']}")
    ok_ = dict(J={'Q1 AMOUNT': 'CALIBRATED-IN-AMOUNT', 'Q3 VALUE RANKING': 'RANKS-BEYOND-FAIR-EDGE'})
    Rp = {k: {s_: {d: dict(ok_) for d in EVAL} for s_ in (0, 1)} for k in KINDS}
    g1 = diag_of(Rp); Rp['post'][1][EVAL[0]] = dict(J={'Q1 AMOUNT': 'MISCALIBRATED-IN-AMOUNT', 'Q3 VALUE RANKING': 'RANKS-BEYOND-FAIR-EDGE'})
    Rp['retain'][0][EVAL[-1]] = dict(J={'Q1 AMOUNT': 'CALIBRATED-IN-AMOUNT', 'Q3 VALUE RANKING': 'NOT-BEYOND-FAIR-EDGE'})
    if not (g1 == dict(post='PASS', retain='PASS') and diag_of(Rp) == dict(post='FAIL', retain='FAIL')):
        bad.append(f'the rank / quantity diagnostic on planted readings: {g1} / {diag_of(Rp)}')
    # v2 (review 89caa74c P3): the economic readings on planted cases; the reviewer's counterexample (every value shifted +20 c) must fail them
    tv = -np.arange(10., 0., -.5); qd = pd.DataFrame(dict(value_m50=tv)); E_ = lambda V, ok=None, kind='retain', d=qd: econ_one(kind, d, dict(V=np.asarray(V, float), ok=np.ones(len(d), bool) if ok is None else ok))
    shifted, good = E_(tv + 20.), E_(tv * 1.05)
    mix = pd.DataFrame(dict(value_m50=np.r_[-5., -3., 2., 1.])); keep_loses = E_(np.r_[-4., -3., 1., 2.], d=pd.DataFrame(dict(value_m50=np.r_[-5., -3., -2., -1.])))
    two_x, flipped = E_(tv * 2.), E_(-tv)
    rogue = E_(np.r_[-4., -3., -9., 2.], ok=np.r_[True, True, False, True], d=pd.DataFrame(dict(value_m50=np.r_[-5., -3., 40., 1.])))
    if not (shifted['E1 VALUE LEVEL'] == 'FAIL' and shifted['E2 BOUNDARY SIGN'] == 'NOT-EVALUABLE' and diag_of({k: {s_: {d: dict(J={'Q1 AMOUNT': 'CALIBRATED-IN-AMOUNT', 'Q3 VALUE RANKING': 'RANKS-BEYOND-FAIR-EDGE'}) for d in EVAL} for s_ in (0, 1)} for k in KINDS}) == dict(post='PASS', retain='PASS')
            and good['E1 VALUE LEVEL'] == good['E3 LOSS AMOUNT'] == 'PASS' and E_(np.r_[-4., -3., 1., 2.], d=mix)['E2 BOUNDARY SIGN'] == 'PASS'
            and keep_loses['E2 BOUNDARY SIGN'] == 'FAIL' and two_x['E1 VALUE LEVEL'] == 'FAIL' and flipped['E1 VALUE LEVEL'] == 'FAIL'
            and rogue['E2 BOUNDARY SIGN'] == 'PASS' and rogue['incoherent'] == 1):
        bad.append(f'the economic readings on planted cases: shifted {shifted}, good {good}, keep_loses {keep_loses}, 2x {two_x}, flipped {flipped}, rogue {rogue}')
    pE = {k: {s_: {EVAL[0]: {'E1 VALUE LEVEL': 'PASS', 'E2 BOUNDARY SIGN': 'PASS', 'E3 LOSS AMOUNT': 'PASS'}, EVAL[-1]: {'E1 VALUE LEVEL': 'FAIL', 'E2 BOUNDARY SIGN': 'FAIL', 'E3 LOSS AMOUNT': 'FAIL'}}
               for s_ in (0, 1)} for k in KINDS}
    gp = gate_of(pE); pE['retain'][1][GATE_DAY]['E2 BOUNDARY SIGN'] = 'FAIL'
    if not (GATE_DAY == EVAL[0] and gp == dict(post='PASS', retain='PASS') and gate_of(pE) == dict(post='PASS', retain='FAIL')):
        bad.append(f'the economic gate on planted readings: {gp} / {gate_of(pE)}')
    class Stub:
        def __init__(self, a):
            self.a = np.asarray(a, float)
        def predict(self, X, num_threads=1):
            return self.a
    cdf = pd.DataFrame({x: [0., 0.] for x in SETS['J']}).assign(side=[0, 0], p=[49.5, 49.5], pf_mid=[50., 50.], fp_remaining=[30., 30.])
    co = predict(dict(p=Stub([.5, .5]), q=Stub([10., 10.]), qm=Stub([1., -1000.])), cdf, 'J')['ok']
    if co.tolist() != [True, False]:
        bad.append(f'moment coherence: {co.tolist()} (an implied endpoint mid of -50 c must be refused)')
    # v3 (review 1adf6a33 P2): E[q | fill] -1 clips to 0 (E[q] = 0) with E[q * move] -100 c (the raw rule would pull) and +100 c (it would keep, the
    # value still impossible) - both refused, by the round trip alone; a certain no-fill (P = 0, E[q * move] = 0) and a valid row accepted
    c5 = pd.DataFrame({x: [0.] * 5 for x in SETS['J']}).assign(side=[0, 1, 0, 1, 0], p=[49.5, 50.5, 49.5, 50.5, 49.5], pf_mid=[50.] * 5, fp_remaining=[30.] * 5)
    pz = predict(dict(p=Stub([.5, .5, .5, 0., .5]), q=Stub([-1., -1., -1., 10., 10.]), qm=Stub([-200., 200., 0., 5., 1.])), c5, 'J')
    if not (pz['ok'].tolist() == [False, False, True, True, True] and pz['zq'].tolist() == [True, True, False, False, False]
            and np.allclose(pz['V'][:2], [-100., 100.], rtol=0, atol=1e-12)):
        bad.append(f"zero-quantity coherence: ok {pz['ok'].tolist()}, refused by the round trip alone {pz['zq'].tolist()} (E[q] = 0 with E[q * move] -100 / +100 c "
                   'must be refused; a zero-quantity zero-move forecast, a certain no-fill and a valid row accepted)')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp); exp = tmp / 'exp'; synthetic(exp)
        def run(out_):
            with contextlib.redirect_stdout(io.StringIO()):
                return bench(exp=exp, out=out_, rounds=30)
        R = None
        try:
            R, gate, diag, Ec = run(tmp / 'b')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic benchmark refused: {str(e)[:300]}')
        if R is not None:
            J = json.loads((tmp / 'b' / 'bench.json').read_text()); pr = pd.read_parquet(tmp / 'b' / 'pred_20261027.parquet')
            if not (diag == dict(post='PASS', retain='PASS') and set(gate.values()) <= {'PASS', 'FAIL'} and all(Ec[k][s][d]['incoherent'] >= 0 for k in KINDS for s in (0, 1) for d in EVAL)
                    and 'J_ok' in pr and all(R[k][s][d]['J']['Q3 VALUE RANKING'] == 'RANKS-BEYOND-FAIR-EDGE' for k in KINDS for s in (0, 1) for d in EVAL)
                    and all(R[k][s][d]['F']['Q3 VALUE RANKING'] == 'BASELINE' for k in KINDS for s in (0, 1) for d in EVAL)
                    and J['multiplicity'] == 8 and len(J['models']) == 2 * 2 * (4 + 1 + 3 + 3) and len(pr) == 2 * 2000
                    and all(R[k][s][d]['J']['fill_auc'] > .6 for k in KINDS for s in (0, 1) for d in EVAL)
                    and np.allclose(pr.J_V, np.where(pr.side == 0, 1., -1.) * (pr.pf_mid - pr.p) * pr.J_Eq + pr.J_Eqm, rtol=0, atol=1e-9)):
                bad.append(f'synthetic recovery: gate {gate}')
        def refused(name, msg, change, undo):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000)); change()
            try:
                run(o_); bad.append(f'the benchmark accepted {name}')
            except (SystemExit, ValueError) as e:
                if msg not in str(e):
                    bad.append(f'{name} refused by another control: {str(e)[:300]}')
            finally:
                undo()
            if o_.exists() or (o_.parent / (o_.name + '.partial')).exists():
                bad.append(f'a refused benchmark left output ({name})')
        keep_train, keep_m = TRAIN, list(SETS['M'])
        refused('an evaluation day in training', 'B1', lambda: globals().update(TRAIN=TRAIN + (EVAL[-1],)), lambda: globals().update(TRAIN=keep_train))
        refused('a label among the inputs', 'B2 M', lambda: SETS['M'].append('q_m50'), lambda: SETS.update(M=keep_m))
        refused('the mapping as a product of means', 'B3', lambda: globals().update(MAP_PRODUCT=True), lambda: globals().update(MAP_PRODUCT=False))
        refused('a saved model replaced after fitting', 'B4', lambda: globals().update(SWAP_SAVED=True), lambda: globals().update(SWAP_SAVED=False))
        f0 = exp / TRAIN[0] / 'rows_000.parquet'; raw = f0.read_bytes()
        def corrupt():
            g = pd.read_parquet(f0); g.loc[g.index[0], 'q_m50'] = g.q_m50.iloc[0] + 1.; g.to_parquet(f0, index=False)
        refused('an export part changed after its receipt', 'B1 20261023', corrupt, lambda: f0.write_bytes(raw))
        e0 = exp / EVAL[0] / 'rows_000.parquet'; raw_e = e0.read_bytes(); rec_e = (exp / EVAL[0] / 'receipt.json').read_bytes()
        def small_remaining():
            g = pd.read_parquet(e0); g.loc[g.index[:40], 'fp_remaining'] = 1.; g.to_parquet(e0, index=False)
            C.dump(exp / EVAL[0] / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(e0)})); globals().update(NO_CLIP=True)
        def restore_remaining():
            e0.write_bytes(raw_e); (exp / EVAL[0] / 'receipt.json').write_bytes(rec_e); globals().update(NO_CLIP=False)
        refused('an expected quantity above the known remaining (no clip)', 'B5', small_remaining, restore_remaining)
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline bench selftests passed (synthetic only): every reading on planted predictions; the planted structure recovered end to end (both '
          'mapping gates PASS, J ranks value beyond the uninformative fair-edge baseline, models / predictions / multiplicity recorded); refusals by their '
          'own control with no output: an evaluation day in training (B1), a label among the inputs (B2), the mapping as a product of means (B3), a '
          'saved model replaced (B4), an export part changed after its receipt (B1), an expected quantity above the known remaining (B5); the gate on '
          'planted readings; v3: zero-quantity moments with E[q * move] -100 / +100 c refused by the round trip, a certain no-fill accepted', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        bench()
    else:
        sys.exit('usage: pf_pipeline_bench.py selftest | run')
