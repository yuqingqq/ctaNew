"""ADV2-ENTRY-SMOOTH-v1: ONE DECLARED ENTRY CANDIDATE - TWO SMOOTHER POST HEADS WITH THE FROZEN PI0 RETENTION (P-2026-005; 2026-11-21, session
aea5bfb9). Review 371aab56's one next priority, items 1-3: does lower entry-model complexity improve useful selection on the existing labels,
with PI0 retention held fixed? A new declared candidate, not a repair of the R3 freeze (pf_adv2, c8655f75). Declared and frozen before any fit.
CANDIDATE (multiplicity: one more primary, two new fitted heads): ADV2-ENTRY-SMOOTH-v1 = two independent squared-error regressors of the paired
 POST label Y = R(POST now) - R(WAIT this tick) under the frozen pi0_50_v1 continuation, bid and ask. Inputs: RICH's 107 (pf_adv2.INPUTS['RICH']),
 float32 as stored, native NaN. Training rows: every POST row of R3's verified Oct 24-25 extraction (its files must equal the committed receipt
 manifest, predictfun/reviews/r3_adv2_20261121/fit/extract.json), zeros and tails kept, the stored w with ONE constant n / sum(w) per head.
PARAMETERS: pf_adv2.PARAMS with exactly ONE change, min_data_in_leaf 200 -> 1,000, and rounds 200 -> 50 (DELTA; everything else pinned:
 num_leaves 31, lambda_l2 10, learning_rate .05, seed 1, ...). One predeclared smoothing package, not a search. No FAIR / MECH refit, no second
 smoother, no calibration map, no gain / loss heads.
CONTROLLER: absent -> the candidate's supported POST prediction > 0: POST, else WAIT (an exact tie WAITs); live -> the unchanged frozen PI0-50
 teacher. Structural rules, pending / terminal states, cap, timing, size, END, zero costs: unchanged (pf_adv2.Candidate / Teacher0 / Hybrid).
 Candidate support: R3's training-derived POST support (fit/support.json, hash-checked), unchanged; no new generation cap, eligibility, cooldown,
 minimum lifetime or score threshold.
POLICIES (the bounded comparison): SMOOTH (this candidate's entry + PI0 retention), ENTRY_ONLY (R3's RICH POST entry + PI0 retention; the primary
 comparator: same retention, the old entry model), PI0 and N0 (benchmarks). Each on its own branch and state.
Usage (run_light): pf_adv2_entry.py selftest | fit"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402  frozen (freeze 2, c8655f75): inputs, readers, Candidate, Hybrid

C, pd, T, L = A.C, A.pd, A.T, A.L
VERSION = 'adv2_entry_smooth_v1'
NAME = 'SMOOTH'
HEADS = ((0, 'post'), (1, 'post'))
DELTA = dict(min_data_in_leaf=1000)
PARAMS, ROUNDS = dict(A.PARAMS, **DELTA), 50
PARAMS_SHA = 'ce5260158d2abced39df30ed9bdb0177d80da9bae11768a2bce198edb13a2c75'
R3_OUT = A.OUT
R3_RECEIPT = Path('predictfun/reviews/r3_adv2_20261121')
REPO = HERE.parent.parent
OUT = A.DATA / 'adv2_entry_smooth_v1_20261121'
POLICIES = ('N0', 'PI0', 'ENTRY_ONLY', 'SMOOTH')


def check_params():
    """The declared parameter delta: exactly min_data_in_leaf 1,000 against pf_adv2.PARAMS (objective regression), 50 rounds, the frozen hash."""
    diff = {k: (A.PARAMS.get(k), PARAMS.get(k)) for k in set(A.PARAMS) | set(PARAMS) if A.PARAMS.get(k) != PARAMS.get(k)}
    C.require(diff == {'min_data_in_leaf': (200, 1000)} and ROUNDS == 50 and A.ROUNDS == 200, f'the declared delta changed: {diff}, rounds {ROUNDS}')
    C.require(A.sha_of(PARAMS) == PARAMS_SHA, 'the resolved parameters changed')


def r3_train(out=R3_OUT, receipt=R3_RECEIPT):
    """R3's Oct 24-25 POST training rows: the extraction files must equal the committed receipt manifest byte for byte (extract.json) and its
    per-head hashes (keys, inputs, labels, weights) -> {(side, 'post'): rows}."""
    C.require(committed_equal(receipt / 'fit' / 'extract.json', Path(out) / 'train' / 'extract.json'), "R3's extraction manifest is not the committed one")
    ex = json.loads((Path(out) / 'train' / 'extract.json').read_text()); E = A.load_train(out)
    C.require(A.manifest_of(E, ex['statuses'])['heads'] == ex['heads'], "R3's training files differ from their extraction manifest")
    return {h: E[h] for h in HEADS}


def committed_equal(rel, local):
    """HEAD holds rel (repo-relative) and it equals local byte for byte."""
    import subprocess
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(local).read_bytes()


def fit_head(X, y, w, rounds=ROUNDS):
    """One POST head: the 107 RICH inputs in their order, the normalized weights, PARAMS, 50 rounds."""
    lg = A.lgb(); cols = [T.FEATS.index(c) for c in A.INPUTS['RICH']]
    ds = lg.Dataset(np.ascontiguousarray(X[:, cols]), label=np.asarray(y, float), weight=A.norm(w), feature_name=list(A.INPUTS['RICH']), free_raw_data=False)
    return lg.train(dict(PARAMS), ds, num_boost_round=rounds)


def predict(b, X):
    return A.predict(b, 'RICH', X)


def model_path(out, s):
    return Path(out) / 'models' / f'{NAME}_{s}_post.txt'


def fit_all(E, out, rounds=ROUNDS):
    """The two heads -> saved models (never overwritten), each reloaded and bit-equal on its training rows; the fit manifest."""
    out = Path(out); (out / 'models').mkdir(parents=True, exist_ok=True); man = {}; lg = A.lgb()
    for (s, h) in HEADS:
        e = E[(s, h)]; t0 = time.time(); b = fit_head(e['X'], e['y'], e['w'], rounds); p = model_path(out, s)
        C.require(not p.exists(), f'{p.name}: a model file exists (never overwritten)'); b.save_model(str(p))
        b2 = lg.Booster(model_file=str(p)); p1, p2 = predict(b, e['X']), predict(b2, e['X'])
        C.require(np.array_equal(p1, p2), f'{p.name}: the reloaded model does not predict bit-equal to the fitted booster')
        wn = A.norm(e['w']); ym = float(np.sum(wn * e['y']) / np.sum(wn))
        man[f'{NAME} {s} {h}'] = dict(file=p.name, sha256=C.sha(p), rows=int(len(e['y'])), trees=int(b2.num_trees()), seconds=round(time.time() - t0, 2),
                                      train_weighted_mean_y=ym, insample_wmse=float(np.sum(wn * (p1 - e['y']) ** 2) / np.sum(wn)),
                                      insample_wmse_zero=float(np.sum(wn * e['y'] ** 2) / np.sum(wn)))
    return man


def cmd_fit(out=OUT):
    """The two fits -> out/models + out/fit.json (the model hashes; committed before any evaluation reads them)."""
    check_params(); A.check_inputs(); out = Path(out); C.require(not (out / 'fit.json').exists(), 'the fit exists')
    r3fz = json.loads((R3_OUT / 'fit.json').read_text()); C.require(committed_equal(R3_RECEIPT / 'fit' / 'fit.json', R3_OUT / 'fit.json'), "R3's fit manifest is not the committed one")
    E = r3_train(); t0 = time.time(); man = fit_all(E, out)
    import lightgbm as lg_
    import platform
    fz = dict(version=VERSION, params=PARAMS, params_sha256=PARAMS_SHA, delta=DELTA, rounds=ROUNDS, inputs=dict(n=len(A.INPUTS['RICH']), sha256=A.INPUT_SHA['RICH']),
              r3_extract_sha256=C.sha(R3_OUT / 'train' / 'extract.json'), r3_support_sha256=r3fz['support_sha256'], r3_fit_sha256=C.sha(R3_OUT / 'fit.json'), models=man,
              libraries=dict(lightgbm=lg_.__version__, numpy=np.__version__, pandas=pd.__version__, python=platform.python_version()),
              source_sha256=C.sha(Path(__file__)), seconds=round(time.time() - t0, 1), asof=C.stamp())
    (out / 'fit.json').write_text(json.dumps(fz, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: dict(rows=v['rows'], trees=v['trees'], sha=v['sha256'][:12]) for k, v in man.items()}, indent=1))


def frozen(out=OUT):
    """fit.json and every model hash verified; R3's fit (RICH heads) and support verified -> (fit manifest, SMOOTH heads, R3 RICH heads, Support)."""
    out = Path(out); fz = json.loads((out / 'fit.json').read_text()); lg = A.lgb()
    C.require(fz['params_sha256'] == PARAMS_SHA and fz['rounds'] == ROUNDS and C.sha(R3_OUT / 'fit.json') == fz['r3_fit_sha256'], 'not this declaration\'s fit')
    import lightgbm as lg_
    C.require(lg_.__version__ == fz['libraries']['lightgbm'] and np.__version__ == fz['libraries']['numpy'], 'library versions differ from the fit')
    H = {}
    for (s, h) in HEADS:
        p = model_path(out, s); C.require(C.sha(p) == fz['models'][f'{NAME} {s} {h}']['sha256'], f'{p.name}: not the frozen model'); H[(s, h)] = lg.Booster(model_file=str(p))
    r3, rich, csup = A.frozen_fit(R3_OUT)
    C.require(C.sha(R3_OUT / 'train' / 'support.json') == fz['r3_support_sha256'], "R3's support changed")
    return fz, H, rich, csup


def make_policy(name, smooth, rich, csup, M, sup, causal=None):
    """A new, unbound policy of the declared name (header's POLICIES)."""
    C.require(name in POLICIES, f'an undeclared policy {name}')
    if name == NAME:
        ent = A.Candidate(smooth, csup, 'RICH'); ent.name = NAME
        p = A.Hybrid(ent, A.Teacher0(M, sup), NAME)
    else:
        p = A.make_policy(name, rich, csup, M, sup)
    p.causal = causal
    return p


def factory(name, smooth, rich, csup, M, sup, world, every=None):
    """A policy factory for pf_r2_labels.bound_fork: each branch gets its own policy bound to it and to the world."""
    made = []

    def make(B):
        probe, st = T.causal_probe(every) if every else (None, None)
        p = make_policy(name, smooth, rich, csup, M, sup, causal=probe); p.bind(B, world); made.append((p, st)); return p
    make.made = made
    return make


def selftest():
    import pf_adv2_entry_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['fit']:
        cmd_fit()
    else:
        sys.exit(__doc__)
