"""Independent pinned R2-A v2 review: archived artifacts and bounded synthetic cases."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib, copy, hashlib, io, json, math, re, sys, tempfile
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-b565f671')
OUT = ROOT / 'predictfun/reviews/automatic/20261120_b565f671'
TARGET = 'b565f6714ddf37216ab7061f4bf8a1f64eb08efe'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_r2_shadow as R

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, data):
    data.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(),
                script_sha256=sha(__file__), source_sha256=sha(R.__file__))
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')

def artifacts():
    d = ROOT / 'data/predict_fun/research/r2_shadow_20261119'
    rc = PIN / 'predictfun/reviews/r2a_shadow_census_v2_20261119'
    hashes = {}
    for line in (rc / 'sources.sha256').read_text().splitlines():
        h, rel = line.split(None, 1)
        p = ROOT / rel if rel.startswith('data/') else (PIN / 'predictfun/research' / rel if rel.startswith('pf_') else rc / rel)
        assert sha(p) == h, (rel, sha(p), h)
        hashes[rel] = h
    j = json.loads((d / 'census_20261023_r2av3.json').read_text())
    s = json.loads((d / 'support_20261023_r2av3.json').read_text())
    for tag in ['r2av2', 'r2av3']:
        for pre in ['census', 'support']:
            nm = f'{pre}_20261023_{tag}.json'
            assert sha(d / nm) == sha(rc / nm)
        jj = json.loads((d / f'census_20261023_{tag}.json').read_text())
        for rel, h in jj['outputs'].items():
            assert sha(d / rel) == h, rel
        assert sha(d / f'census_20261023_{tag}/manifest.json') == sha(rc / f'manifest_20261023_{tag}.json')
    assert j['source_sha256'] == sha(R.__file__)
    assert j['engine_sha256'] == sha(R.BE.__file__)
    assert j['inputs_list_sha256'] == hashlib.sha256(json.dumps(R.INPUTS + R.CTX).encode()).hexdigest()
    kp = ROOT / 'data/predict_fun/research/keep_pull_20261117/20261023'
    kj = json.loads((kp / 'receipt.json').read_text())
    assert sha(kp / 'receipt.json') == j['inputs']['kp_receipt']
    assert sha(kp / 'orders.parquet') == kj['outputs']['orders.parquet']
    orig = pd.read_parquet(kp / 'orders.parquet', columns=['oid', 'side', 'm', 'status', 't'])
    sel = pd.concat([orig[orig.side == a].sort_values('oid').head(128) for a in [0, 1]])
    ids = sorted((int(a), int(b)) for a, b in zip(sel.side, sel.oid))
    man = json.loads((d / 'census_20261023_r2av3/manifest.json').read_text())
    assert [list(x) for x in ids] == j['readings']['population']['selected_ids']
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == man['selected_ids_sha256']
    frames, seen = [], set()
    for i, p in enumerate(man['parts']):
        expect = sorted((int(a), int(b)) for a, b in zip(*[R.partitions(sel)[i][c] for c in ['side', 'oid']]))
        assert p['originals'] == [list(x) for x in expect] and p['part'] == i and len(expect) <= 64
        assert not (seen & set(expect)); seen |= set(expect)
        f = pd.read_parquet(d / 'census_20261023_r2av3' / p['file'])
        assert len(f) == p['rows']
        assert set(zip(f.side.astype(int), f.original_episode_id.astype(int))) == set(expect)
        assert not f.duplicated(R.SORT_KEY).any()
        frames.append(f)
    assert seen == set(ids)
    p = pd.concat(frames, ignore_index=True); del frames
    v3 = pd.read_parquet(d / 'census_20261023_v3.parquet')
    assert p.shape == v3.shape == (213349, 137) and set(p) == set(v3)
    assert p[R.SORT_KEY].equals(R.canonical(p[R.SORT_KEY]))
    changed = [c for c in v3 if not p[c].equals(v3[c])]
    assert changed == ['refused'], changed
    joint = p.source.str.startswith('joint') & (p['head'] == 'post') & (p.status == 'ok') & p.sampled.astype(bool)
    assert joint.sum() == 289 and p.refused[~joint].equals(v3.refused[~joint])
    assert v3.refused[joint].isna().all() and p.refused[joint].map(lambda x: isinstance(x, (bool, np.bool_))).all()
    del v3
    ok = (p.status == 'ok') & p.sampled.astype(bool); a = p[ok]
    assert len(a) == 44959
    stat = p[p.source == 'original']; epi = p[p.source != 'original']
    assert stat.groupby(['side', 'original_episode_id']).size().eq(1).all()
    assert not set(zip(stat.side, stat.original_episode_id)) & set(zip(epi.side, epi.original_episode_id))
    gw = a.groupby(['side', 'original_episode_id', 'head']).weight.sum()
    assert len(gw) == 508 and np.allclose(gw, 1., rtol=0, atol=1e-12) and (p.weight[~ok] == 0).all()
    assert all(str(p[c].dtype) == 'float32' for c in R.INPUTS + R.CTX) and a[R.CTX].notna().all().all()
    refs = {src: int((g.refused == True).sum()) for src, g in a[a['head'] == 'post'].groupby('source')}
    assert refs == j['readings']['refusals_by_source'] == dict(entry=312, joint_chase=12, joint_post_keep=6)
    assert p.refused[p['head'] == 'retain'].isna().all()
    sg = np.where(a.side == 0, 1., -1.)
    assert np.max(np.abs(a.value - sg * (a.end_mark - a.price) * a.q)) < 1e-7
    assert np.max(np.abs(a.qm - sg * (a.end_mark - a.pf_mid_ref) * a.q)) < 1e-7
    assert np.max(np.abs(a.severe - np.maximum(-a.value - R.KP.SEVERE, 0))) < 1e-7
    assert np.allclose(a.end_abs_ms - a.t0, R.END, rtol=0, atol=1e-6)
    assert np.allclose(a.decision_abs_ms - a.t0, a.decision_ms, rtol=0, atol=1e-6)
    assert (a.t_to_end == R.END - a.decision_ms).all()
    assert (a.q >= -1e-9).all() and (a.q <= a.fp_remaining + 1e-6).all()
    cz = j['readings']['causality']; expected = int(sum(math.ceil(n / 17) for n in a.groupby(['side', 'original_episode_id']).size()))
    assert cz['checked'] == expected == 2765 and cz['failed'] == 0 and cz['verdict'] == 'PASS' and not any(cz['components'].values())
    assert cz['retried'] == 2 and cz['draws'] == 8
    assert j['readings']['labels'] == dict(checked=44959, failed=0, no_fill=32902, verdict='PASS')
    peak = [int(x) for x in re.findall(r'peak_rss_mib=(\d+)', (rc / 'logs/pr2b_census_r2av3.log').read_text())]
    assert peak == [1246] and max(j['readings']['partitions']['seconds']) <= 900
    assert R.support_merge([R.support_part(p)]) == s
    # Independent support counts, ranges and category identities on the actual saved rows.
    category_rows = {(r['sh'], r['dim'], r['value']): r for r in s['categories']}
    cross_rows = {(r['sh'], r['hbin'], r['dim'], r['value']): r for r in s['cross_cells']}
    for (side, head), g in a.groupby(['side', 'head']):
        sh = f'{"bid" if side == 0 else "ask"} {head}'
        for lo, hi in zip(R.HORIZON_BINS[:-1], R.HORIZON_BINS[1:]):
            q = g[(g.t_to_end > lo) & (g.t_to_end <= hi)]; key = f'{sh} ({lo}, {hi}]'
            e, nz = q.original_episode_id.nunique(), q[q.q > 1e-12].original_episode_id.nunique()
            assert (len(q), e, nz) == tuple(s['cells'][key][c] for c in ['rows', 'episodes', 'nonzero_episodes'])
            assert s['cells'][key]['fallback'] == {'post': 'WAIT', 'retain': 'KEEP'}[head]
        for c in R.NUMERIC:
            assert s['ranges'][sh][c] == [float(g[c].min()), float(g[c].max())]
        cats = {
            'c_rem': g.fp_remaining.map(lambda x: '0' if x <= 0 else '(0,10]' if x <= 10 else '(10,20]' if x <= 20 else '(20,30]' if x <= 30 else 'n/a'),
            'c_queue': g.fp_queue.map(lambda x: 'unknown' if not np.isfinite(x) else '0' if x <= 0 else '(0,30]' if x <= 30 else '(30,150]' if x <= 150 else '>150'),
            'c_gen': g.generation.map(lambda x: '1' if x <= 1 else '2' if x <= 2 else '>=3'),
            'c_inv': g.ctx_inventory_shares.map(lambda x: 'long' if x > 1e-9 else 'short' if x < -1e-9 else 'flat'),
            'c_other': g[[f'ctx_other_state_{x}' for x in ['absent','post_pending','live','cancel_pending','terminal']]].idxmax(axis=1).str.removeprefix('ctx_other_state_'),
            'c_price': g.price.map(lambda x: '[1,5]' if x <= 5 else '(5,20]' if x <= 20 else '(20,80]' if x <= 80 else '(80,95]' if x <= 95 else '(95,99]')}
        for dim, values in cats.items():
            for value in R.DECLARED[dim]:
                q = g[values == value]; e = q.original_episode_id.nunique(); nz = q[q.q > 1e-12].original_episode_id.nunique()
                assert s['presence'][sh][dim][value] == e
                assert (f'{dim}={value}' in s['unsupported_state_category'][sh]) == (e == 0)
                if len(q):
                    r = category_rows[sh, dim, value]; assert (len(q), e, nz) == tuple(r[c] for c in ['rows','originals','nonzero_originals'])
                    for lo, hi in zip(R.HORIZON_BINS[:-1], R.HORIZON_BINS[1:]):
                        z = q[(q.t_to_end > lo) & (q.t_to_end <= hi)]
                        if len(z):
                            rr = cross_rows[sh, f'({lo}, {hi}]', dim, value]
                            assert (len(z), z.original_episode_id.nunique(), z[z.q > 1e-12].original_episode_id.nunique()) == tuple(rr[c] for c in ['rows','originals','nonzero_originals'])
    mut = [l for l in (rc / 'logs/pr2b_mutants4.log').read_text().splitlines() if ' -> ' in l]
    assert len(mut) == 47 and 'rc 0 -> passed' in mut[0] and all('rc 1 -> caught: R2 SHADOW SELFTEST FAILED' in l for l in mut[1:])
    import pf_r2_shadow_mutants as M
    for i in range(len(M.MUT)): M.source(i)
    # Prospective deterministic causality sample, budgeted by source/head rather
    # than relying on row-list cadence. Counts only; no new real-world replay.
    groups = a.groupby(['side', 'original_episode_id', 'source', 'head']).size()
    planned_checks = int(sum(max(1, math.ceil(n / 64)) for n in groups))
    predicted_checks = planned_checks * len(orig) / len(sel)
    save('artifacts.json', dict(all_checks_pass=True, hashes=hashes, saved_asof=j['asof'], census_rows=len(p), sampled_ok=len(a),
        full_originals=len(orig), selected_originals=len(sel), weight_groups=len(gw), changed_columns=changed, changed_joint_rows=int(joint.sum()),
        refusals=refs, causality=cz, labels=j['readings']['labels'], support_cells=len(s['cells']), supported=0,
        frame_memory_bytes=int(p.memory_usage(deep=True).sum()), full_frame_linear_gib=float(p.memory_usage(deep=True).sum()*len(orig)/len(sel)/2**30),
        saved_peak_rss_mib=peak[0], partition_seconds=j['readings']['partitions']['seconds'], developer_mutants_checked=46, mutation_targets_verified=len(M.MUT)-1,
        proposed_causality_rule='per original/source/head max(1, ceil(n_sampled_ok/64)), smallest declared identity SHA256',
        proposed_slice_checks=planned_checks, proposed_linear_full_checks=predicted_checks,
        proposed_linear_hours=477*25/3600+predicted_checks*.41/3600))
    print('Pinned hashes, 9 receipt checks, exact comparison and independent support/label algebra verified.')

def synthetic():
    day = R.DAY
    # Receipt-delayed venue-past book: preserve execution history, but a client
    # must still be unable to read the as-yet unreceived frame directly.
    src = R.KP.Syn(); raw = next(src.pairs(src.load(day, [101], src.offset(day))))
    w = {}
    for side, val in raw.items():
        v = dict(val); v['kf'] = np.asarray(v['tf'], float) + 500.
        v['jk'] = np.searchsorted(v['kf'], np.asarray(v['tg'], float), side='right') - 1
        v['cache'] = {}; w[side] = v
    t = float(w[0]['ws']) + 30000.
    o = pd.Series(dict(side=0, oid=1, m=101, t=t, phase=1))
    path, why = R.QP.build_path(w, t, t + R.END); assert path is not None, why
    fv = np.full(R.KP.NSTEP + 1, 51.)
    original_snapshot = R.client_snapshot
    def build_and_check():
        rows = R.probes(path, o, day, [])
        R.add_inputs(rows, w, o, None, None, R.X.syn_features(), lambda k: float(fv[k]))
        r = next(x for x in rows if x['source'] == 'entry' and x['status'] == 'ok' and x['decision_ms'] == 0.)
        chk = R.causal_rows([r], path, o, day, w, None, None, fv, every=1,
            feat=R.X.syn_features(), perturb=R.X.syn_perturb, fire=R.X.syn_fire)
        return r, chk
    clean, baseline = build_and_check()
    assert baseline[:2] == (1, 0) and baseline[3]['not_evaluable'] == 0, baseline
    def hidden_book(B, g, side):
        ctx, st, gid = original_snapshot(B, g, side)
        jv = int(np.searchsorted(B.E.f[:, 0], g, side='right') - 1)
        ctx['ctx_inventory_shares'] += float(B.E.f[jv, 2]) * .01
        return ctx, st, gid
    R.client_snapshot = hidden_book
    try:
        leaked, escaped = build_and_check()
        assert escaped[:2] == (1, 0) and escaped[3]['not_evaluable'] == 0, escaped
        jv = int(np.searchsorted(path['frames'][:, 0], 0., side='right') - 1)
        assert path['frames'][jv, 0] <= 0. < path['frames'][jv, 1]
        altered = copy.deepcopy(path); altered['frames'][jv, 2] += 1.
        a = R.rebuild_row(path, w, fv, o, day, leaked, None, None, R.X.syn_features())
        b = R.rebuild_row(altered, w, fv, o, day, leaked, None, None, R.X.syn_features())
        assert a['ctx_inventory_shares'] != b['ctx_inventory_shares']
        leak_result = dict(clean=baseline, planted_unreceived_book_leak=escaped,
            leaked_inventory=leaked['ctx_inventory_shares'], clean_inventory=clean['ctx_inventory_shares'],
            changed_hidden_book_inventory=b['ctx_inventory_shares'], frame_venue_ms=float(path['frames'][jv,0]),
            frame_receipt_ms=float(path['frames'][jv,1]), decision_ms=0.)
    finally:
        R.client_snapshot = original_snapshot
    # Minimal all-status population: no raw tape, labels, or feature build needed.
    with tempfile.TemporaryDirectory(prefix='pf-review-r2v2-') as tmp:
        td = Path(tmp); kp = td / 'kp'; (kp / day).mkdir(parents=True)
        o = pd.DataFrame([dict(oid=1, m=101, side=0, t=0., status='missing_endpoint'), dict(oid=2, m=102, side=1, t=0., status='missing_endpoint')])
        o.to_parquet(kp / day / 'orders.parquet', index=False)
        (kp / day / 'receipt.json').write_text(json.dumps(dict(outputs={'orders.parquet': sha(kp / day / 'orders.parquet')})))
        class Stub:
            def __init__(self, mode): self.mode = mode
            def markets(self, day): return [101] if self.mode == 'missing_market' else [101, 102]
            def offset(self, day): return 0.
            def load(self, *args): return None
            def pairs(self, data):
                yield {0: {'m':101}}
                if self.mode == 'duplicate': yield {0: {'m':101}}
                if self.mode != 'missing_pair': yield {0: {'m':102}}
        results = {}
        for mode in ['good', 'missing_pair', 'missing_market', 'duplicate']:
            with contextlib.redirect_stdout(io.StringIO()):
                d, s, r = R.census(day, per_side=1, out=td/mode, src=Stub(mode), kp=kp, inputs=R.X.syn_inputs, causal=False)
            assert r['population']['represented'] == 2
            results[mode] = dict(rows=len(d), reconciliation=r['population']['reconciliation'], statuses=d.status.tolist(),
                duplicate_original_status_rows=int(d.duplicated(['side','original_episode_id']).sum()))
        assert results['good']['rows'] == 2
        assert 'missing_source:pair' in results['missing_pair']['statuses']
        assert 'missing_source:market' in results['missing_market']['statuses']
        assert results['duplicate']['rows'] == 3 and results['duplicate']['reconciliation'] == 'EXACT'
        # Safe sentinel: prove the production census reaches the whole-day concat,
        # after receipts exist, without allocating a large frame.
        called = {}; old = R.read_parts
        class WholeRead(Exception): pass
        def sentinel(d):
            called['manifest_exists'] = (Path(d)/'manifest.json').exists()
            called['receipt_exists'] = (td/'whole'/f'census_{day}.json').exists()
            called['parts'] = len(json.loads((Path(d)/'manifest.json').read_text())['parts'])
            raise WholeRead('whole-census materialization reached')
        R.read_parts = sentinel
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                R.census(day, per_side=1, out=td/'whole', src=Stub('good'), kp=kp, inputs=R.X.syn_inputs, causal=False, part_size=1)
            assert False, 'read_parts not reached'
        except WholeRead:
            pass
        finally:
            R.read_parts = old
        assert called == dict(manifest_exists=True, receipt_exists=True, parts=2)
    save('synthetic.json', dict(population=results, whole_day_read=called, clock_visibility=leak_result, source='synthetic_only'))
    print('Missing-source fixes pass; within-part duplicate accepted; whole-census read reached after receipt write; unreceived venue-past book leak escapes.')

if __name__ == '__main__':
    {'artifacts': artifacts, 'synthetic': synthetic}[sys.argv[1]]()
