"""Export of the frozen chase line's entry rule for Tokyo's live pilot (USER 2026-11-28: "When the live session is recording. Can we export
model and commit what it needs from here first"; pmmmrust request docs/receipts/20261128_tokyo_chase_live_pilot_request, stage 2).
P-2026-005, no orders. No new days: the fixture rows come from 2026-11-20, a forward day this line already read (its stored forward rows,
scores and selections).
Exports (write-once, OUT/chase_live_export/):
  move_model.json      the frozen MOVE model (OUT/chase_forward/refit/models.pkl, the receipt's sha256), sklearn HistGradientBoostingRegressor
                       with numerical splits only, as per-tree node arrays (input, threshold, missing side, children, leaf, leaf value) and
                       the baseline; the 66 inputs in the model's order, which of them research rounds to float32, the cost term and the
                       MOVE_top10 threshold;
  move_fixtures.csv.gz complete moves of 11-20 (every row of each): N_ACT moves with a MOVE_top10 action and N_OTHER without, seeded; per row
                       the move key, t, d, mv_k, the 66 inputs, lim, md, the prediction, MOVE, above (MOVE >= threshold) and act (the
                       move's action); floats with 17 significant digits;
  eval_move.py         a standard-library scorer of the two files (live_export/eval_move.py);
  reference/           chase_forward.py and every research module it imports, copied at the pinned commit, with MANIFEST.json.
Checks (the run refuses to write otherwise):
  - the models' sha256 is the freeze receipt's, and the builder is the frozen one (chase_forward.frozen);
  - eval_trees (numpy, the trees summed in sklearn's order) equals sklearn's predict exactly on every row of the day;
  - sklearn's prediction and MOVE equal the stored forward scores (forward_scores.parquet) exactly on every row of the day;
  - the day's actions (the first row per move with a finite MOVE >= the threshold) equal forward_selections' MOVE_top10 rows of the day;
  - eval_move.py reproduces every fixture row and every fixture move's action (a subprocess, exit 0);
  - every copied file is tracked and unchanged at the pinned commit, and no file holds a credential-shaped string.
Stages: selftest | run.  Output: chase_live_export.json (write-once).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_live_export.py <stage>
"""
import datetime as dt
import gzip
import hashlib
import json
import os
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_forward as CF  # noqa: E402

C, TR, CT = CF.C, CF.TR, CF.CT
REPO = CF.REPO
OUTD = Path(os.environ['CLE_OUT']) if os.environ.get('CLE_OUT') else CF.WT.OUT / 'chase_live_export'
RESULT = OUTD / 'result.json' if os.environ.get('CLE_OUT') else H / 'chase_live_export.json'
EVAL = H / 'live_export' / 'eval_move.py'
DAY, SEL, N_ACT, N_OTHER, SEED = '20261120', 'MOVE_top10', 200, 200, 20261128
SECRET = (r'-----BEGIN [A-Z ]*PRIVATE KEY-----', r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}', r'AKIA[0-9A-Z]{16}',
          r'(?i)(api[_-]?key|secret|passw(or)?d|private[_-]?key)\s*[:=]\s*["\'][A-Za-z0-9+/=_-]{12,}["\']', r'\b0x[0-9a-fA-F]{64}\b')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ the trees
def export_trees(m, inputs, f32, thr):
    C.require(type(m).__name__ == 'HistGradientBoostingRegressor' and str(m.loss) == 'squared_error', 'the MOVE model is not a squared-error HGB regressor')
    C.require(list(m.feature_names_in_) == list(inputs), "the model's input order differs from the refit inputs")
    C.require(m.n_trees_per_iteration_ == 1 and getattr(m, 'is_categorical_', None) is None, 'one numerical tree per iteration expected')
    base = np.asarray(m._baseline_prediction, float).ravel()
    C.require(base.size == 1, 'a scalar baseline')
    trees = []
    for it in m._predictors:
        nd = it[0].nodes
        C.require(not nd['is_categorical'].any(), 'a categorical split')
        trees.append(dict(feature=nd['feature_idx'].astype(int).tolist(), threshold=nd['num_threshold'].astype(float).tolist(),
                          missing_left=nd['missing_go_to_left'].astype(bool).tolist(), left=nd['left'].astype(int).tolist(),
                          right=nd['right'].astype(int).tolist(), is_leaf=nd['is_leaf'].astype(bool).tolist(), value=nd['value'].astype(float).tolist()))
    return dict(model='MOVE (chase forward candidate, predictfun/freeze/chase_forward_1120_1126)', sklearn=sklearn.__version__, kind='HistGradientBoostingRegressor, squared error, numerical splits',
                rule=('prediction = baseline + the leaf value of each tree in order; from node 0, at a split take the missing side if the input is NaN, '
                      'else left iff input <= threshold'),
                inputs=list(inputs), float32_inputs=list(f32), baseline=float(base[0]), trees=trees,
                cost='d * (lim - md) + 0.02 * min(lim, 100 - lim); lim = the known Up ask (d = +1) or Up bid (d = -1), md = the known Up mid, cents',
                score='MOVE = prediction - cost', threshold=float(thr), action='the first decision point of a move with a finite MOVE >= threshold')


def eval_trees(J, X):
    """The exported trees on rows X (inputs in J's order) -> predictions; float64, summed in sklearn's order."""
    X = np.asarray(X, float)
    out = np.full(len(X), J['baseline'])
    for tr in J['trees']:
        f, th, ml, lf, rt, leaf, v = (np.asarray(tr[k]) for k in ('feature', 'threshold', 'missing_left', 'left', 'right', 'is_leaf', 'value'))
        node = np.zeros(len(X), int)
        act = ~leaf[node]
        while act.any():
            i = np.flatnonzero(act)
            nd = node[i]
            x = X[i, f[nd]]
            go = np.where(np.isnan(x), ml[nd], x <= th[nd])
            node[i] = np.where(go, lf[nd], rt[nd])
            act[i] = ~leaf[node[i]]
        out = out + v[node]
    return out


# ------------------------------------------------------------------------------------------------------------------ files
def ref_modules():
    """chase_forward.py and every research module it imports (loaded in this process), as repo-relative paths."""
    out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and Path(f).resolve().is_relative_to(REPO / 'predictfun'):
            out.add(Path(f).resolve().relative_to(REPO))
    return sorted(out)


def git_clean(rels):
    """-> (pinned commit, files not tracked or changed against it)."""
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    bad = []
    for r in rels:
        t = subprocess.run(['git', '-C', str(REPO), 'ls-files', '--error-unmatch', str(r)], capture_output=True)
        c = subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--', str(r)])
        if t.returncode or c.returncode:
            bad.append(str(r))
    return head, bad


def secrets(texts):
    """{name: text} -> [(name, pattern index, match)] for credential-shaped strings."""
    hits = []
    for nm, s in texts.items():
        for k, p in enumerate(SECRET):
            for mt in re.finditer(p, s):
                hits.append((nm, k, mt.group(0)[:24]))
    return hits


def fixtures(X, keep_keys, cols):
    F = X[X.key.isin(keep_keys)].sort_values(['key', 't'], kind='stable')
    return F[cols]


def write_csv_gz(F, p):
    with gzip.open(p, 'wt', newline='') as fh:
        F.to_csv(fh, index=False, float_format='%.17g', na_rep='')


# ------------------------------------------------------------------------------------------------------------------ run
def run():
    C.require(not RESULT.exists() and not (OUTD / 'move_model.json').exists(), f'{RESULT.name} or the export exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    rc = CF.frozen()
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    kf, thr = R['inputs'], float(R['thresholds'][SEL])
    mp = CF.FOUT / 'refit' / 'models.pkl'
    msha = sha(mp)
    C.require(msha == R['models_sha256'] and msha in rc, "the models' sha256 is not the receipt's")
    with open(mp, 'rb') as fh:
        Mo = pickle.load(fh)
    X = CF.fwd_frame((DAY,), kf)
    import pyarrow.parquet as pq
    sch = pq.read_schema(CF.FOUT / 'rows' / f'{DAY}_points.parquet')
    f32 = [c for c in kf if c in sch.names and str(sch.field(c).type) == 'float']
    C.require(all(str(X[c].dtype) == 'float32' for c in f32) and all(str(X[c].dtype) == 'float64' for c in kf if c not in f32), 'input dtypes')
    J = export_trees(Mo['move'], kf, f32, thr)
    pred = Mo['move'].predict(X[kf])
    mine = eval_trees(J, X[kf].to_numpy(float))
    n_eq = int(np.sum(mine == pred))
    C.require(n_eq == len(X), f'the exported trees differ from sklearn on {len(X) - n_eq} of {len(X)} rows')
    move = pred - CF.cost_of(X)
    S = pd.read_parquet(CF.FOUT / 'forward_scores.parquet', columns=['day', 't', 'd', 'pred_1s', 'MOVE'])
    S = S[S.day == DAY]
    mg = X[['day', 't', 'd']].assign(p=pred, mv=move).merge(S, on=['day', 't', 'd'], how='left', validate='1:1')
    C.require(len(mg) == len(X) and mg.pred_1s.notna().all(), 'a row without its stored score')
    C.require(np.array_equal(mg.p.to_numpy(), mg.pred_1s.to_numpy()) and np.array_equal(mg.mv.to_numpy(), mg.MOVE.to_numpy(), equal_nan=True),
              'the scores differ from forward_scores.parquet')
    key = X.key.to_numpy()
    act = CT.first_per_move(np.isfinite(move) & (move >= thr), key)
    G = pd.read_parquet(CF.FOUT / 'forward_selections.parquet', columns=['day', 't', 'd', SEL])
    G = G[(G.day == DAY) & G[SEL]]
    mine_a = set(zip(X.t.to_numpy()[act], X.d.to_numpy()[act]))
    C.require(mine_a == set(zip(G.t.to_numpy(), G.d.to_numpy())), f'the actions differ from forward_selections ({len(mine_a)} vs {len(G)})')
    X['pred'], X['MOVE'] = pred, move
    X['above'] = (np.isfinite(move) & (move >= thr)).astype(int)
    X['act'] = 0
    X.loc[act, 'act'] = 1
    rng = np.random.default_rng(SEED)
    ka = np.unique(key[act])
    ko = np.setdiff1d(np.unique(key), ka)
    keep = np.r_[rng.choice(ka, min(N_ACT, len(ka)), replace=False), rng.choice(ko, min(N_OTHER, len(ko)), replace=False)]
    cols = ['key', 'day', 't', 'd', 'mv_k'] + [c for c in kf if c != 'mv_k'] + ['lim', 'md', 'pred', 'MOVE', 'above', 'act']
    Fx = fixtures(X, keep, cols)
    rels = ref_modules() + [EVAL.relative_to(REPO), Path(__file__).resolve().relative_to(REPO)]
    rels = sorted(set(rels))
    head, dirty = git_clean(rels)
    C.require(not dirty, f'files not committed at HEAD: {dirty[:5]}')
    hits = secrets({str(r): (REPO / r).read_text(errors='replace') for r in rels})
    C.require(not hits, f'credential-shaped strings: {hits[:5]}')
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'move_model.json').write_text(json.dumps(J) + '\n')
    write_csv_gz(Fx, OUTD / 'move_fixtures.csv.gz')
    shutil.copy2(EVAL, OUTD / 'eval_move.py')
    rd = OUTD / 'reference'
    man = []
    for r in rels:
        dst = rd / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / r, dst)
        man.append(dict(path=str(r), sha256=sha(dst), bytes=dst.stat().st_size))
    TR.dump(rd / 'MANIFEST.json', dict(ctanew_commit=head, files=man))
    ev = subprocess.run([sys.executable, '-I', str(OUTD / 'eval_move.py'), str(OUTD / 'move_model.json'), str(OUTD / 'move_fixtures.csv.gz')], capture_output=True, text=True)
    C.require(ev.returncode == 0, f'eval_move.py does not reproduce the fixtures: {ev.stdout.strip()} {ev.stderr.strip()[-300:]}')
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), day=DAY, receipt=CF.receipt_path().name, models_sha256=msha, sklearn=sklearn.__version__,
               inputs=kf, float32_inputs=f32, threshold=thr, trees=len(J['trees']), nodes=sum(len(t['value']) for t in J['trees']), baseline=J['baseline'],
               day_rows=int(len(X)), trees_equal_sklearn_rows=n_eq, stored_scores_equal=True, day_actions=int(len(act)), actions_equal_selections=True,
               fixture_rows=int(len(Fx)), fixture_moves=int(Fx.key.nunique()), fixture_actions=int(Fx.act.sum()), fixture_nan_rows=int(Fx[kf].isna().any(axis=1).sum()),
               eval_move=ev.stdout.strip(), ctanew_commit=head, reference_files=len(man), reference_bytes=int(sum(m['bytes'] for m in man)),
               hashes={nm: sha(OUTD / nm) for nm in ('move_model.json', 'move_fixtures.csv.gz', 'eval_move.py')} | {'reference/MANIFEST.json': sha(rd / 'MANIFEST.json')},
               out_dir=str(OUTD), rehearsal=bool(os.environ.get('CLE_OUT')))
    TR.dump(RESULT, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('inputs', 'float32_inputs', 'hashes')}), flush=True)


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    # a two-tree model: positive control, then perturbations the evaluator must see
    J = dict(baseline=1.5, trees=[dict(feature=[0, 0, 0], threshold=[2., 0., 0.], missing_left=[True, False, False], left=[1, 0, 0], right=[2, 0, 0],
                                         is_leaf=[False, True, True], value=[0., -1., 1.]),
                                    dict(feature=[1, 0, 0], threshold=[0.5, 0., 0.], missing_left=[False, False, False], left=[1, 0, 0], right=[2, 0, 0],
                                         is_leaf=[False, True, True], value=[0., .25, .5])])
    X = np.array([[1., 0.], [2., 1.], [3., np.nan], [np.nan, .5]])
    C.require(np.array_equal(eval_trees(J, X), [1.5 - 1 + .25, 1.5 - 1 + .5, 1.5 + 1 + .5, 1.5 - 1 + .25]), f'eval_trees {eval_trees(J, X)}')
    J2 = json.loads(json.dumps(J)); J2['trees'][0]['threshold'][0] = 1.5
    C.require(not np.array_equal(eval_trees(J2, X), eval_trees(J, X)), 'a moved threshold went unseen')
    J3 = json.loads(json.dumps(J)); J3['trees'][0]['missing_left'][0] = False
    C.require(not np.array_equal(eval_trees(J3, X), eval_trees(J, X)), 'a flipped missing side went unseen')
    J4 = json.loads(json.dumps(J)); J4['trees'] = J4['trees'][:1]
    C.require(not np.array_equal(eval_trees(J4, X), eval_trees(J, X)), 'a dropped tree went unseen')
    # the standard-library scorer agrees with eval_trees, and refuses a corrupted fixture
    sys.path.insert(0, str(EVAL.parent))
    import eval_move as EM
    J5 = dict(J, inputs=['a', 'b'], threshold=0.6)
    rows = []
    for i, x in enumerate(X):
        p = float(eval_trees(J, x[None])[0])
        mv = p - EM.cost(1., 60., 59.5)
        rows.append(dict(key='k1' if i < 2 else 'k2', t=str(float(i)), d='1', lim='60', md='59.5', a='' if np.isnan(x[0]) else repr(float(x[0])), b='' if np.isnan(x[1]) else repr(float(x[1])),
                         pred=repr(p), MOVE=repr(mv), above='1' if mv >= .6 else '0', act='0'))
    for k in ('k1', 'k2'):
        ab = [r for r in rows if r['key'] == k and r['above'] == '1']
        if ab:
            ab[0]['act'] = '1'
    C.require(EM.check(J5, rows) == (4, 0, 2, 0), f'eval_move on the positive control: {EM.check(J5, rows)}')
    bad = [dict(r) for r in rows]; bad[1]['pred'] = repr(float(bad[1]['pred']) + 1e-12)
    C.require(EM.check(J5, bad)[1] == 1, 'eval_move missed a prediction off by 1e-12')
    bad = [dict(r) for r in rows]
    for r in bad:
        r['act'] = '0'
    C.require(EM.check(J5, bad)[3] >= 1, 'eval_move missed a missing action')
    with tempfile.TemporaryDirectory() as td:
        F = pd.DataFrame(rows)
        write_csv_gz(F, Path(td) / 'f.csv.gz')
        (Path(td) / 'm.json').write_text(json.dumps(J5))
        ok = subprocess.run([sys.executable, '-I', str(EVAL), str(Path(td) / 'm.json'), str(Path(td) / 'f.csv.gz')], capture_output=True, text=True)
        C.require(ok.returncode == 0, f'eval_move.py refused a correct fixture: {ok.stdout}')
        F.loc[0, 'MOVE'] = repr(float(F.loc[0, 'MOVE']) + 1.)
        write_csv_gz(F, Path(td) / 'f.csv.gz')
        no = subprocess.run([sys.executable, '-I', str(EVAL), str(Path(td) / 'm.json'), str(Path(td) / 'f.csv.gz')], capture_output=True, text=True)
        C.require(no.returncode == 1, 'eval_move.py accepted a corrupted fixture')
    # the credential scan flags a planted key and passes plain code
    pem, kv = '-----BEGIN RSA ' + 'PRIVATE KEY-----', 'api_' + 'key = "abcdefghijklmnop"'     # built here so this file holds neither
    C.require(secrets({'x': 'k = 1\n'}) == [] and len(secrets({'x': pem})) == 1 and len(secrets({'x': kv})) == 1, 'the credential scan')
    # write-once
    global RESULT
    keep = RESULT
    RESULT = Path(__file__)
    try:
        run()
        raise AssertionError('run overwrote its result')
    except ValueError as e_:
        assert 'write-once' in str(e_), e_
    finally:
        RESULT = keep
    print('chase_live_export selftest PASS')


if __name__ == '__main__':
    dict(selftest=selftest, run=run)[sys.argv[1]]()
