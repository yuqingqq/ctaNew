"""Pinned R2-D artifact audit. No new real labels, fit, replay or untouched day.
Every mode must run through PF_SLOT=codex run_light.sh.
"""
import ast
import collections
import copy
import gc
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
import weakref
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-f96c38d0')
TARGET = 'f96c38d01a7f27ed4fd39ddf2ddadeecf162b84b'
OUT = ROOT / 'predictfun/reviews/automatic/20261120_f96c38d0'
RC = PIN / 'predictfun/reviews/r2d_labels_20261120'
DATA = ROOT / 'data/predict_fun/research/r2_labels_20261120'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import numpy as np
import pandas as pd
import pf_r2_labels as L


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def key(r):
    # JSON census ints and pandas-promoted selected floats identify the same row.
    return tuple(r.get(c) for c in L.KEY)


def metadata():
    A, B = DATA / 'A', DATA / 'B'; man = read(A / 'manifest.json')
    assert (A / 'manifest.json').read_bytes() == (B / 'manifest.json').read_bytes() == (RC / 'checkpoint/manifest.json').read_bytes()
    assert (A / 'candidates.jsonl.gz').read_bytes() == (B / 'candidates.jsonl.gz').read_bytes() == (RC / 'checkpoint/candidates.jsonl.gz').read_bytes()
    assert sha(A / 'candidates.jsonl.gz') == man['candidates_sha256']
    assert man['freeze_sha256'] == L.FREEZE_SHA and man['settings'] == L.BE.SETTINGS and man['sources'] == list(L.SOURCES)
    assert man['sources_sha256'] == {n: sha(PIN / 'predictfun/research' / n) for n in L.DEPS}
    for line in (RC / 'sources_at_build.sha256').read_text().splitlines():
        h, p = line.split(); assert h == sha(PIN / 'predictfun/research' / Path(p).name)
    fz = L.T.frozen(); L.libraries_enforced(fz)
    assert sha(L.T.REAL['out'] / 'freeze.json') == L.FREEZE_SHA
    assert len(fz['sources']) == 76
    # All model hashes, without generating any predictions.
    for n, h in fz['models'].items():
        assert sha(L.T.REAL['out'] / 'models' / (n + '.txt')) == h
    cnt = collections.Counter(); ok = collections.defaultdict(lambda: collections.defaultdict(list)); seen = set(); n = 0
    with gzip.open(A / 'candidates.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line); n += 1; rk = key(r)
            assert rk not in seen, ('duplicate candidate', rk); seen.add(rk)
            assert not set(r) & set(L.OUTCOMES)
            assert 0 <= r['decision_ms'] < L.END and r['decision_ms'] % L.STEP == 0
            cnt[(r['day'], r['side'], r['kind'], r['source'], r['status'])] += 1
            if r['status'] == 'ok':
                assert r['sampled'] and r['kind'] in L.KINDS
                ok[(r['day'], r['side'], r['kind'])][r['source']].append(r)
    want = {(r['day'], r['side'], r['kind'], r['source'], r['status']): r['n'] for r in man['candidate_counts']}
    assert dict(cnt) == want and n == man['candidate_rows'] == 129868
    expected = []
    for cell, sources in sorted(ok.items()):
        for src, rows in sources.items():
            rows.sort(key=lambda r: (r['world'], r['decision_ms'], -1 if r['offset_ms'] is None else r['offset_ms'], r['generation']))
        chosen = []
        for rank in range(max(map(len, sources.values()))):
            for source in L.SOURCES:
                if rank < len(sources.get(source, [])):
                    chosen.append(sources[source][rank])
            if len(chosen) >= man['cap']:
                break
        expected.extend(chosen[:man['cap']])
    assert [key(r) for r in expected] == [key(r) for r in man['selected']]
    assert len({key(r) for r in man['selected']}) == len(man['selected']) == 128
    assert all(r['fork_equals_census'] for r in man['selected'] if r['source'] == 'pi0')
    assert all(v['pi0_binding_bad'] == 0 and v['pi0_binding_checked'] == v['pi0_calls'] > 0 for v in man['census_teacher'].values())
    kp = Path(read(L.T.REAL['population'])['inputs']['kp_dir']); populations = {}
    for day in man['days']:
        O = pd.read_parquet(kp / day / 'orders.parquet', columns=['m', 't', 'side', 'oid'])
        worlds = O.groupby(['m', 't'], sort=True).size(); assert worlds.eq(2).all()
        wm = [w for w in man['worlds'] if w['day'] == day]
        assert [(w['m'], w['t']) for w in wm] == list(worlds.index[:32])
        sr = [r for r in man['selected'] if r['day'] == day]
        populations[day] = dict(day_worlds=len(worlds), prefix_worlds=len(wm), selected_pairs=len(sr),
            selected_worlds=sorted({r['world'] for r in sr}), selected_markets=sorted({r['m'] for r in sr}),
            selected_first_t=min(r['t'] for r in sr), selected_last_t=max(r['t'] for r in sr),
            distinct_side_originals=len({(r['side'], r['original_episode_id']) for r in sr}),
            selected_sources=dict(collections.Counter(r['source'] for r in sr)),
            prefix_ok=sum(v for (d, s, k, src, st), v in cnt.items() if d == day and st == 'ok'))
    # Saved harness identities and catches, without rerunning the expensive mutation suites.
    src = (PIN / 'predictfun/research/pf_r2_labels.py').read_text()
    tree = ast.parse((PIN / 'predictfun/research/pf_r2_labels_mutants.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MUTANTS' for t in n.targets))
    muts = ast.literal_eval(node.value); assert len(muts) == 29 and all(src.count(m[1]) == 1 for m in muts)
    sh = (RC / 'logs/mutants/labels_sources_at_mutants.txt').read_text().split()
    assert sh == [sha(PIN / 'predictfun/research' / n)[:16] for n in ('pf_r2_labels.py', 'pf_r2_labels_selftest.py', 'pf_r2_labels_mutants.py')]
    for shard, count in [(0, 15), (1, 14)]:
        log = (RC / f'logs/mutants/labels_mutants_{shard}.log').read_text()
        assert f'{count} / {count}' in log and f'(shard {shard}/2) passed' in log and 'rc=0' in log
    return dict(manifest_sha256=sha(A / 'manifest.json'), candidates_sha256=sha(A / 'candidates.jsonl.gz'),
                all_candidate_rows=n, statuses=dict(collections.Counter({st: sum(v for k, v in cnt.items() if k[-1] == st) for st in {k[-1] for k in cnt}})),
                independent_selection_equal=True, duplicates=0, source_closure_and_frozen_models_match=True, populations=populations,
                saved_mutants_caught=29, selected_pi0_fingerprint_equal=sum(r['source'] == 'pi0' for r in man['selected']))


def records():
    A, B = DATA / 'A', DATA / 'B'; man = read(A / 'manifest.json'); msha = sha(A / 'manifest.json')
    hB = {Path(p).name: h for h, p in (x.split() for x in (RC / 'checkpoint/B_pairs.sha256').read_text().splitlines())}
    strip = lambda r: {k: v for k, v in r.items() if k not in ('asof', 'elapsed_s', 'hwm_mib', 'breach')}
    summary = []; arms = []; causal = collections.Counter(); binding = 0; nonzero = []
    labels = pd.read_csv(A / 'labels.csv'); assert len(labels) == 128
    assert (A / 'labels.csv').read_bytes() == (RC / 'checkpoint/labels.csv').read_bytes()
    for i, sel in enumerate(man['selected']):
        name = L.rname(sel); pa, pb = A / 'pairs' / name, B / 'pairs' / name; a, b = read(pa), read(pb)
        assert pa.read_bytes() == (RC / 'checkpoint/A_pairs' / name).read_bytes() and sha(pb) == hB[name]
        assert strip(a) == strip(b) and a['candidate'] == b['candidate'] == sel
        assert a['manifest_sha256'] == msha and a['key'] == L.rkey(sel)
        assert all(not any(r['breach'].values()) and r['elapsed_s'] <= 120 and r['hwm_mib'] <= 1536 for r in (a, b))
        r = a['result']; assert r['status'] == 'ok' and r['reconciled'] and r['common_prefork_identical']
        assert r['fork_fingerprint'] == sel['fork_fingerprint'] and r['side'] == sel['side'] and r['kind'] == sel['kind']
        assert r['k'] * 50 == sel['decision_ms'] and r['end'] == 5000 and r['settings'] == man['settings']
        for k in ('day', 'instrument_id', 'original_episode_id', 'side'):
            assert r['ident'][k] == sel[k]
        assert r['ident']['start_abs_ms'] == sel['t'] and r['ident']['end_abs_ms'] == sel['t'] + 5000
        names = ['POST', 'WAIT'] if r['kind'] == 'post' else ['KEEP', 'PULL']
        assert set(r['arms']) == set(names) and np.isclose(r['Y'], r['arms'][names[0]]['value'] - r['arms'][names[1]]['value'], atol=1e-9, rtol=0)
        assert labels.iloc[i].record_sha256 == sha(pa) and np.isclose(labels.iloc[i].Y, r['Y'], atol=1e-9, rtol=0)
        for n in names:
            z = r['arms'][n]; t = z['teacher']; assert z['reconciled'] and z['mark'] == r['mark']
            assert np.isclose(z['value'], z['cash'] + z['position'] * z['mark'], atol=1e-6, rtol=0)
            assert t['binding_bad'] == 0 and t['binding_checked'] == t['calls'] > 0
            assert sum(t['by_status'].values()) == sum(t['by_action'].values()) == t['calls']; binding += t['binding_checked']
            assert all(z[k] >= 0 for k in ('fills', 'generations', 'gens_bid', 'gens_ask'))
            assert abs(z['position']) <= 30 + 1e-9 and sum(z['open_at_end'].values()) <= 60 + 1e-9
            assert all(len(z[k]) == 64 for k in ('fills_sha256', 'decisions_sha256', 'requests_sha256'))
            fb = sum(v for k, v in t['by_status'].items() if k not in ('model', 'cap_blocked', 'near_end'))
            arms.append(dict(day=sel['day'], side=sel['side'], kind=sel['kind'], source=sel['source'], arm=n, calls=t['calls'], fallback=fb))
        assert len(a['causal']) == 2
        for cz in a['causal']:
            assert cz['checked'] == (cz['n'] + 6) // 7 and cz['mismatch'] == cz['parity_mismatch'] == 0
            assert sum(v['checked'] for v in cz['by_state'].values()) == cz['checked']
            for k in ('n', 'checked', 'mismatch', 'parity_checked', 'parity_mismatch', 'delivered_changed'):
                causal[k] += cz[k]
        summary.append(dict(day=sel['day'], side=sel['side'], kind=sel['kind'], source=sel['source'], Y=r['Y'], elapsed_s=a['elapsed_s'], hwm_mib=a['hwm_mib']))
        if abs(r['Y']) > 1e-9:
            nonzero.append(dict(key=a['key'], world=sel['world'], original=sel['original_episode_id'], Y=r['Y']))
    S = pd.DataFrame(summary); saved = pd.read_csv(RC / 'labels_summary.csv', dtype={'day': str})
    pd.testing.assert_frame_equal(S, saved, check_dtype=False, atol=1e-9, rtol=1e-12)
    arm_csv = pd.read_csv(RC / 'arms.csv', dtype={'day': str})
    pd.testing.assert_frame_equal(pd.DataFrame(arms).sort_values(['day','side','kind','source','arm','calls']).reset_index(drop=True),
                                 arm_csv[list(arms[0])].sort_values(['day','side','kind','source','arm','calls']).reset_index(drop=True), check_dtype=False)
    checks = read(RC / 'checks.json'); expected = {'A_manifest': True, 'B_resume_identity': True, 'C_labels': True, 'D_resources': False, 'E_controls': True, 'F_suite_on_build_sources': True}
    assert checks['predicates'] == expected and not checks['all_true']
    for d in (A, B):
        st = read(d / 'label_status.json'); assert st['state'] == 'PAIRS_DONE' and st['manifest_sha256'] == msha and st['done'] == 128
        assert not (d / 'stops.jsonl').exists() and not (d / 'disposition.json').exists()
    ctl = read(RC / 'controls.json'); assert ctl['all_true'] and ctl['manifest_sha256'] == msha and all(ctl['holds'].values())
    logs = {p.name: dict(peak=int(re.findall(r'peak_rss_mib=(\d+)', p.read_text())[-1]), rc=int(re.findall(r'rc=(-?\d+)', p.read_text())[-1]))
            for p in (RC / 'logs').glob('r2d_*.log') if re.findall(r'peak_rss_mib=(\d+)', p.read_text())}
    maxpair = float(S.elapsed_s.max()); meanpair = float(S.elapsed_s.mean())
    assert np.isclose(meanpair, checks['readings']['seconds_per_pair']['mean'])
    assert {k for k,v in logs.items() if v['peak'] > 1536} == {k for k,v in checks['readings']['peaks_mib'].items() if v > 1536}
    projection = checks['readings']['projection_h']
    for d, x in projection.items():
        assert np.isclose(x['prefix_hours'], x['prefix_ok_candidates'] * meanpair / 3600)
        assert np.isclose(x['full_day_hours'], x['prefix_ok_candidates'] / 32 * x['day_worlds'] * meanpair / 3600)
    return dict(pairs=128, per_arm_identities_and_value_arithmetic_pass=True, all_checkpoint_record_bytes_match=True,
                independent_resumed_records_equal=True, binding_checked=binding, causal=dict(causal),
                nonzero_pairs=nonzero, nonzero_count=len(nonzero), mean_pair_seconds=meanpair, max_pair_seconds=maxpair,
                resource_peaks_over_1536={k:v for k,v in logs.items() if v['peak']>1536}, saved_predicates=expected,
                projections_arithmetic_correct=projection, raw_fill_traces_available=False)


def diagnostics():
    import pf_r2_teacher_diag2 as D
    D.selftest()
    old = read(L.T.REAL['out'] / 'diag/diag.json'); new = read(L.T.REAL['out'] / 'diag2/diag2.json')
    assert new == read(PIN / 'predictfun/reviews/r2c_teacher_20261120/diag_v2/diag2.json')
    assert new['freeze_sha256'] == old['freeze_sha256'] == L.FREEZE_SHA
    A, B = pd.DataFrame(old['rows']), pd.DataFrame(new['rows'])
    fields = ['rows','weight','pred_fill','real_fill','cond_q_pred','cond_q_real','cond_qm_pred','cond_qm_real','severe_pred','severe_real']
    err = 0.
    for a in A.itertuples():
        b = B[(B.side_head == a.side_head) & (B.horizon == a.horizon)].iloc[0]
        for k in fields:
            v = float(getattr(a,k)); err = max(err, abs(v-float(b[k]))); assert np.isclose(v,b[k],atol=5e-12,rtol=1e-12)
        assert b.incoherent_rows == a.incoherent
        assert np.isclose(b.V_pred_coh_mean_c * b.coherent_weight / b.weight, a.V_pred_wmean_c, atol=1e-10)
        assert np.isclose(b.V_real_coh_mean_c * b.coherent_weight / b.weight, a.V_real_wmean_c, atol=1e-10)
    man = read(L.T.REAL['out'] / 'train/manifest.json'); allmean = {}
    for s,h in L.T.GROUPS:
        group = f'{s}_{h}'; d = L.T.REAL['out'] / 'train' / group; n = man['groups'][group]['rows']
        vals = np.memmap(d/'value_all.f64',dtype='float64',mode='r',shape=(n,)); w=np.memmap(d/'weight_all.f64',dtype='float64',mode='r',shape=(n,))
        mean = float(np.average(vals,weights=w)); b=B[(B.side_head==L.T.sh_of(s,h))&(B.horizon=='all')].iloc[0]
        assert np.isclose(mean,b.V_real_all_mean_c,atol=1e-10); allmean[group]=mean
    return dict(cells=28, all_groups=4, unchanged_columns_max_absolute_error=err, corrected_denominators_equal_old_numerators=True,
                full_population_realized_means_from_actual_training_arrays=allmean, source_selftest_pass=True)


def controls():
    RN = L.RN; out = {}
    # Temp-only copies of all real records: no actual artifact is modified or replayed.
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp); base=tmp/'base'; base.mkdir(); shutil.copy(DATA/'A/manifest.json',base/'manifest.json'); shutil.copytree(DATA/'A/pairs',base/'pairs')
        man=read(base/'manifest.json'); first=man['selected'][0]; p=base/'pairs'/L.rname(first); original=p.read_bytes(); msha=sha(base/'manifest.json')
        assert L.valid_record(base/'pairs',first,msha) and len(L.finalize(base))==128
        corruptions={}
        for name, change in [('Y_plus_one',lambda r:r['result'].__setitem__('Y',r['result']['Y']+1)),
                             ('wrong_result_fork',lambda r:r['result'].__setitem__('fork_fingerprint','0'*64)),
                             ('failed_reconciliation',lambda r:r['result'].__setitem__('reconciled',False)),
                             ('binding_mismatch',lambda r:next(iter(r['result']['arms'].values()))['teacher'].__setitem__('binding_bad',1))]:
            r=json.loads(original);change(r);p.write_text(json.dumps(r)); accepted=L.valid_record(base/'pairs',first,msha)
            rows=L.finalize(base); corruptions[name]=dict(valid_record=accepted, exported_rows=len(rows))
            assert accepted and len(rows)==128
        p.write_bytes(original)
        (base/'label_status.json').write_text(json.dumps(dict(state='STOP_HARD_RSS',manifest_sha256=msha,rss_mib=2100)))
        held=L.held_stops(base,msha);assert held['undisposed'];assert len(L.finalize(base))==128
        out['record_admission']=dict(positive_rows=128, corruptions=corruptions, undisposed_stop_still_finalizes=True)
        # New orchestration source is recorded but not enforced even by pairs' resume or finalize.
        stale=tmp/'stale';shutil.copytree(base,stale);(stale/'label_status.json').unlink();(stale/'stops.jsonl').unlink()
        mm=read(stale/'manifest.json');mm['sources_sha256']['pf_r2_labels.py']='0'*64;(stale/'manifest.json').write_text(json.dumps(mm));newsha=sha(stale/'manifest.json')
        for pp in (stale/'pairs').glob('*.json'):
            r=read(pp);r['manifest_sha256']=newsha;pp.write_text(json.dumps(r))
        # Actual frozen teacher preflight is retained; all records already complete so no raw input or label work occurs.
        state=L.pairs(stale);assert state=='PAIRS_DONE' and len(L.finalize(stale))==128
        out['stale_manifest_source']=dict(declared_label_source='0'*64, actual_label_source=sha(PIN/'predictfun/research/pf_r2_labels.py'), resume_state=state, finalized_rows=128)
    # The watchdog can overwrite the previous stop BEFORE archive_stops is reached.
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);sp=p/'run_status.json';sp.write_text(json.dumps(dict(state='STOP_RESOURCE',note='old unarchived loading stop')));oldsha=sha(sp)
        prev=RN.rss;ex=[]
        try:
            RN.rss=lambda:(2100.,2100.);mon=RN.Sampler(sp,2048.,exit=ex.append);assert mon.poll()
        finally:
            RN.rss=prev
        hist=RN.archive_stops(p);assert ex==[3] and oldsha not in {e['id'] for e in hist}
        out['watchdog_history']=dict(old_event=oldsha, recorded_events=[e['id'] for e in hist], old_stop_lost=True)
    # Actual census control flow with small synthetic sentinels: the loop's world keeps day 1 depth alive while day 2 loads.
    attrs=['worlds_of_day','world_inputs','world_bind','census_world'];saved={k:getattr(L,k) for k in attrs};refs=[];at_load=[]
    class Depth: pass
    def inputs(day,*a,**kw):
        gc.collect();at_load.append(dict(day=day,previous_depth_alive=bool(refs and refs[-1]() is not None)));dep=Depth();refs.append(weakref.ref(dep));return None,{}, {},dep,None
    try:
        L.worlds_of_day=lambda day,kp,n:[dict(m=1,t=0)]
        L.world_inputs=inputs
        L.world_bind=lambda day,w,pairs,fair,depth,ff:(dict(depth=depth),'ok')
        L.census_world=lambda *a:([],dict(pi0_calls=0,pi0_binding_bad=0,pi0_binding_checked=0))
        with tempfile.TemporaryDirectory() as temp:
            L.census(Path(temp),days=('first','second'),prefix=1,cap=1,kp=Path(temp),assets=(None,None,None),freeze_sha='synthetic')
        gc.collect();assert at_load[1]['previous_depth_alive'] and all(r() is None for r in refs)
    finally:
        for k,v in saved.items():setattr(L,k,v)
    out['census_day_lifetime']=dict(at_load=at_load, released_after_function_returns=True)
    return out


if __name__=='__main__':
    mode=sys.argv[1];result=globals()[mode]()
    result.update(target=TARGET,asof=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),mode=mode)
    (OUT/(mode+'.json')).write_text(json.dumps(result,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
    print(json.dumps(dict(mode=mode,result='PASS',evidence=str(OUT/(mode+'.json')))))
