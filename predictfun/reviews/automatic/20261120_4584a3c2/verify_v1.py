"""Independent bounded R2-C review; no real fit, label generation or tape replay.
Run every mode through PF_SLOT=codex run_light.sh. Fixed committed source only.
"""
import ast
import copy
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-4584a3c2')
TARGET = '4584a3c2696625d27cf87f193a6260997348f85e'
OUT = ROOT / 'predictfun/reviews/automatic/20261120_4584a3c2'
RC = PIN / 'predictfun/reviews/r2c_teacher_20261120'
DATA = ROOT / 'data/predict_fun/research/r2_teacher_20261120'
OLD = PIN / 'predictfun/reviews/r2b_full_day_20261120'
CENS = ROOT / 'data/predict_fun/research/r2_shadow_20261119/census_20261023_full'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import numpy as np
import pandas as pd
import pf_r2_teacher as T


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def metadata():
    fz = read(DATA / 'freeze.json'); man = read(DATA / 'train/manifest.json')
    cfg = dict(T.REAL, out=DATA, census=CENS)
    assert T.frozen(cfg) == fz
    assert fz['version'] == 'pi0_50_v1' and fz['multiplicity'] == dict(teacher_versions=1, candidates=0)
    assert fz['rounds'] == 200 and fz['params'] == T.PARAMS and fz['objectives'] == T.OBJ
    assert fz['features'] == man['features'] == T.FEATS and len(T.FEATS) == 107
    assert fz['features_sha256'] == hashlib.sha256(json.dumps(T.FEATS).encode()).hexdigest()
    assert fz['params_sha256'] == hashlib.sha256(json.dumps(T.PARAMS, sort_keys=True).encode()).hexdigest()
    assert fz['plan_sha256'] == man['plan_sha256'] == sha(CENS / 'plan.json') == T.PLAN_SHA
    assert fz['records_sha256'] == man['records_sha256'] == sha(OLD / 'records_full.sha256')
    assert len(fz['sources']) == 76 and all(sha(PIN / 'predictfun/research' / n) == h for n, h in fz['sources'].items())
    libs = {k: importlib.metadata.version(k) for k in ('lightgbm', 'numpy', 'pandas', 'pyarrow')}
    libs['python'] = platform.python_version(); assert libs == fz['libraries']
    count = 0; total_bytes = 0
    for n, h in man['files'].items():
        p = DATA / 'train' / n; assert sha(p) == h, n
        count += 1; total_bytes += p.stat().st_size
    assert count == 96
    oldman = read(DATA / 'train_v0_pre_group_guard/manifest.json')
    assert oldman['files'] == man['files'] and oldman['groups'] == man['groups']
    # Actual archives, not just equal hash strings.
    for p in sorted((RC / 'checkpoint').rglob('*')):
        if not p.is_file():
            continue
        n = p.relative_to(RC / 'checkpoint').as_posix()
        raw = gzip.decompress(p.read_bytes()) if n.endswith('.gz') else p.read_bytes()
        n = n.removesuffix('.gz')
        dest = DATA / ('train/manifest.json' if n == 'train_manifest.json' else
                       'diag/' + n if n in ('diag.json', 'diag_horizon.csv') else
                       'rollout/' + n if n in ('worlds.csv', 'per_side.csv', 'decisions.csv', 'rollout.json') else n)
        assert raw == dest.read_bytes(), n
    models = T.load_models(cfg, fz)
    fits = [read(DATA / f'fit_{i}.json') for i in range(4)]
    assert fits == fz['fits']
    for i, (s, h) in enumerate(T.GROUPS):
        group = f'{s}_{h}'; fit = fits[i]; m = man['groups'][group]
        assert fit['group'] == group and fit['rows'] == m['rows'] and fit['filled'] == m['filled']
        for hd in T.HEADS:
            b = models[(s, h)][hd]; r = fit['heads'][hd]
            assert b.current_iteration() == r['iterations'] == 200 and b.feature_name() == T.FEATS
            assert r['rows'] == m['filled' if hd in ('q', 'qm') else 'rows']
            assert r['reload_equal'] and r['sha256'] == fz['models'][f'{group}_{hd}']
    snaps = []
    for p in sorted((OLD / 'accept_v1_outputs').iterdir()):
        if p.name != 'README.md':
            assert p.read_bytes() == (OLD / p.name).read_bytes(); snaps.append(p.name)
    # Validate each mutation really has one substitution site, without rerunning 69 tests.
    mut = {}
    for stem, expect in [('pf_r2_teacher', 44), ('pf_r2_runner', 25)]:
        tree = ast.parse((PIN / 'predictfun/research' / (stem + '_mutants.py')).read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MUTANTS' for t in n.targets))
        variants = ast.literal_eval(assignment.value); src = (PIN / 'predictfun/research' / (stem + '.py')).read_text()
        assert len(variants) == expect and all(src.count(old) == 1 for _, old, _ in variants)
        log = (RC / 'logs' / ('teacher_mutants.log' if stem.endswith('teacher') else 'runner_v3_mutants.log')).read_text()
        assert f'{expect} / {expect}' in log and 'rc=0' in log
        mut[stem] = dict(targets=expect, saved_log_pass=True)
    return dict(freeze_sha256=sha(DATA / 'freeze.json'), source_count=76, libraries=libs, train_files=count,
                train_bytes=total_bytes, groups=man['groups'], models_verified=16, reextraction_manifest_equal=True,
                saved_v1_v2_files_equal=snaps, mutants=mut)


def samples():
    man = read(DATA / 'train/manifest.json')
    picks = [0, 1, 2, 3, 116, 117, 118, 119, 236, 237, 238, 239, 356, 357, 358, 359, 473, 474, 475, 476]
    counts = np.asarray(man['per_part_rows']); starts = np.vstack([np.zeros(4, dtype=int), np.cumsum(counts, axis=0)])
    assert counts.shape == (477, 4)
    arrays = {}; weights = {}; positives = {}
    for j, (s, h) in enumerate(T.GROUPS):
        gn = f'{s}_{h}'; d = DATA / 'train' / gn; n = man['groups'][gn]['rows']; nf = man['groups'][gn]['filled']
        assert starts[-1, j] == n
        aa = {c: np.memmap(d / f'{c}_all.f64', mode='r', dtype=np.float64, shape=(n,)) for c in T.LAB}
        ff = {c: np.memmap(d / f'{c}_fil.f64', mode='r', dtype=np.float64, shape=(nf,)) for c in T.LAB}
        aa['X'] = np.memmap(d / 'X_all.f32', mode='r', dtype=np.float32, shape=(n, 107))
        ff['X'] = np.memmap(d / 'X_fil.f32', mode='r', dtype=np.float32, shape=(nf, 107))
        qpos = np.asarray(aa['q'] > 1e-12); assert qpos.sum() == nf
        # All conditional labels and weights must be exact subsets, not just equal moments.
        for c in T.LAB:
            assert same(np.asarray(aa[c])[qpos], np.asarray(ff[c])), (gn, c)
        oids, inv = np.unique(aa['original_episode_id'], return_inverse=True)
        sums = np.bincount(inv, weights=aa['weight']); assert np.allclose(sums, 1., atol=1e-12, rtol=0)
        assert np.isfinite(aa['weight']).all() and (aa['weight'] > 0).all()
        weights[gn] = dict(originals=len(oids), max_sum_error=float(np.max(np.abs(sums - 1))),
                           normalization_all=float(n / aa['weight'].sum()), normalization_filled=float(nf / ff['weight'].sum()))
        arrays[j] = (aa, ff); positives[j] = qpos
    audited = 0
    for i in picks:
        D = pd.read_parquet(CENS / 'parts' / f'part_{i:04d}.parquet', columns=list(dict.fromkeys(['status', 'sampled', 'side', 'head'] + T.FEATS + T.LAB)))
        D = D[(D.status == 'ok') & D.sampled.astype(bool)]; audited += len(D)
        assert sum(counts[i]) == len(D)
        for j, (s, h) in enumerate(T.GROUPS):
            G = D[(D.side == s) & (D['head'] == h)]; a, b = int(starts[i, j]), int(starts[i + 1, j]); aa, ff = arrays[j]
            assert len(G) == b - a
            gx = np.ascontiguousarray(G[T.FEATS].to_numpy(np.float32))
            assert same(gx, np.asarray(aa['X'][a:b])), (i, j, 'X')
            for c in T.LAB:
                assert same(np.ascontiguousarray(G[c].to_numpy(np.float64)), np.asarray(aa[c][a:b])), (i, j, c)
            fa = int(positives[j][:a].sum()); fb = fa + int(positives[j][a:b].sum())
            assert same(gx[G.q.to_numpy() > 1e-12], np.asarray(ff['X'][fa:fb])), (i, j, 'Xfil')
    # Positive/negative controls on the equality checker.
    z = np.array([[1., np.nan]], np.float32); bad = z.copy(); bad[0, 0] = 2
    assert same(z, z.copy()) and not same(z, bad) and not same(z, z[:, ::-1])
    return dict(part_indices=picks, sampled_ok_rows=audited, all_conditional_label_subsets_equal=True, weights=weights,
                sampled_inputs_labels_bit_equal=True, equality_controls_pass=True)


def outputs():
    W = pd.read_csv(DATA / 'rollout/worlds.csv'); E = pd.read_csv(DATA / 'rollout/per_side.csv'); D = pd.read_csv(DATA / 'rollout/decisions.csv')
    r = read(DATA / 'rollout/rollout.json'); diag = read(DATA / 'diag/diag.json'); chk = read(RC / 'checks.json')
    fz = read(DATA / 'freeze.json'); assert r['freeze_sha256'] == diag['freeze_sha256'] == sha(DATA / 'freeze.json')
    assert fz['asof'] < diag['asof'] and fz['asof'] < r['asof'] < chk['asof']
    sm = r['summary']; assert len(W) == sm['worlds'] == 128 and (W.status == 'ok').sum() == sm['worlds_ok'] == 127
    assert dict(W.status.value_counts()) == chk['readings']['world_statuses']
    assert len(D) == sm['decisions'] == 24541 and len(E) == 254 and not E.duplicated(['m', 't', 'side']).any()
    for fields, key in [(['side', 'action'], 'by_action'), (['side', 'status'], 'by_status')]:
        assert {str(k): int(v) for k, v in D.groupby(fields).size().items()} == sm[key]
    assert {str(k): int(v) for k, v in E.groupby(['side', 'entry']).size().items()} == sm['entries']
    for c in ('later_generations', 'pulls', 'fills'):
        assert int(E[c].sum()) == sm[c]
    assert int(E.posts.sum()) == int((D.action == 'POST').sum()) == chk['readings']['posts'] == 337
    assert int(E.refused.sum()) == chk['readings']['refused'] == 28
    assert E[[c for c in E if c.startswith('n_') and c not in ('n_model', 'n_cap_blocked', 'n_near_end')]].to_numpy().sum() == 0
    M = D[D.status == 'model']; assert np.isfinite(M.V).all()
    want = np.where(M.state == 'absent', np.where(M.V > 0, 'POST', 'WAIT'), np.where(M.V < 0, 'PULL', 'KEEP'))
    assert (M.action.to_numpy() == want).all()
    assert (D[D.status != 'model'].action == 'WAIT').all()
    # CSV has only implicit world identity: reconstruct using declared chronological concatenation + per-world log counts.
    cur = 0
    for w in W[W.status == 'ok'].itertuples():
        n = int(w.logged); dd = D.iloc[cur:cur+n]; cur += n
        assert w.ledger_ok and w.logged == w.engine_asked - w.engine_fallbacks
        assert dd.t.is_monotonic_increasing and not dd.duplicated(['t', 'side']).any()
        for s in (0, 1):
            ss = dd[dd.side == s]; ee = E[(E.m == w.m) & (E.t == w.t) & (E.side == s)].iloc[0]
            for status in T.STATUSES:
                assert (ss.status == status).sum() == ee['n_' + status]
            for action, col in [('POST', 'posts'), ('PULL', 'pulls')]:
                assert (ss.action == action).sum() == ee[col]
            first = ss.loc[ss.action == 'POST', 't'].min()
            assert (pd.isna(first) and pd.isna(ee.first_submit_ms)) or first == ee.first_submit_ms
    assert cur == len(D)
    cz = r['causal']; assert cz['checked'] == (len(D) + 6) // 7 == 3506
    assert cz['mismatch'] == cz['parity_mismatch'] == 0 and cz['parity_checked'] == 3384 and cz['delivered_changed'] == 3271
    assert sum(v['checked'] for v in cz['by_state'].values()) == cz['checked']
    A = pd.DataFrame(diag['rows']); S = read(T.REAL['support'])
    assert len(A) == 28 and diag['adapter']['categories_equal'] and diag['adapter']['cells_equal']
    for a in A.itertuples():
        cell = S['cells'][f'{a.side_head} {a.horizon}']
        assert (a.rows, a.originals, a.nonzero_originals) == (cell['rows'], cell['episodes'], cell['nonzero_episodes'])
        assert np.isclose(a.coherent_frac, (a.rows - a.incoherent) / a.rows)
    pd.testing.assert_frame_equal(A, pd.read_csv(DATA / 'diag/diag_horizon.csv'), check_dtype=False, atol=1e-10, rtol=1e-10)
    groups = {sh: dict(rows=int(g.rows.sum()), incoherent=int(g.incoherent.sum()),
                      pred_fill=float(np.average(g.pred_fill, weights=g.weight)), real_fill=float(np.average(g.real_fill, weights=g.weight)))
              for sh, g in A.groupby('side_head')}
    return dict(worlds=128, episodes_ok=127, decisions=len(D), rollout_summary=sm, diagnostic_groups=groups,
                sign_rule_all_decisions=True, reconstructed_world_log_identity_pass=True,
                diagnostic_support_counts_equal=True, no_new_predictions_or_tape_replay=True)


def controls():
    import pf_r2_runner as R
    # The disposition matches names, so replacing a status with a NEW stop inherits the old clearance.
    with tempfile.TemporaryDirectory() as t:
        p = Path(t); sp = p / 'run_status.json'
        sp.write_text(json.dumps(dict(state='STOP_RESOURCE', latch=dict(part=1, reason='first stop'))))
        old_hash = sha(sp); before = R.held_breaches(p, {})
        assert before['undisposed'] == ['run_status.json']
        (p / 'disposition.json').write_text(json.dumps(dict(parts=[], statuses=['run_status.json'], reason='first stop investigated')))
        approved = R.held_breaches(p, {}); assert approved['undisposed'] == []
        sp.write_text(json.dumps(dict(state='STOP_HARD_RSS', reason='new stop during a later process loading')))
        new_hash = sha(sp); after = R.held_breaches(p, {})
        assert new_hash != old_hash and after['undisposed'] == []
    # Denominator demonstration reproduces diagnostic accumulation's coherent mask and all-row denominator.
    w = np.array([1., 9.]); v = np.array([10., -20.]); coherent = np.array([True, False])
    reported = float((w[coherent] * v[coherent]).sum() / w.sum())
    conditional = float(np.average(v[coherent], weights=w[coherent])); all_rows = float(np.average(v, weights=w))
    assert (reported, conditional, all_rows) == (1., 10., -17.)
    # Actual published callback keeps its bound branch when presented a copied branch's view.
    from types import SimpleNamespace
    base = SimpleNamespace(marker='prefork'); arm = copy.deepcopy(base); arm.marker = 'post_arm'
    t = T.Teacher({}, None); t.bind(base, {})
    t.decide = lambda view, B: dict(action='WAIT', branch=B.marker)
    assert t.fn(dict(t=50, side=0, state='absent')) == 'WAIT'
    seen = t.log[-1]['branch']; assert seen == 'prefork'
    # Rebinding is the positive integration control (not an implementation fix).
    t.bind(arm, {}); t.fn(dict(t=50, side=0, state='absent')); assert t.log[-1]['branch'] == 'post_arm'
    return dict(resource_disposition=dict(before=before, approved=approved, new_stop=after, old_hash=old_hash, new_hash=new_hash,
                                         new_stop_inherits_clearance=True),
                value_diagnostic=dict(reported=reported, coherent_mean=conditional, all_row_mean=all_rows),
                callback_binding=dict(stale_callback_observed=seen, explicit_binding_observed=t.log[-1]['branch']))


if __name__ == '__main__':
    mode = sys.argv[1]; result = globals()[mode]()
    result.update(target=TARGET, asof=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), mode=mode)
    (OUT / (mode + '.json')).write_text(json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')
    print(json.dumps(dict(mode=mode, result='PASS', evidence=str(OUT / (mode + '.json')))))
