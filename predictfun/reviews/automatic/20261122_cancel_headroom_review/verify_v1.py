"""Bounded pinned-source/saved-evidence review; no replay, fitting or new labels."""
from pathlib import Path
import contextlib, gzip, hashlib, importlib.util, io, json, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-ch-review-438ce3a8')
OUT = Path(__file__).resolve().parent
TARGET = '438ce3a889e8f2b27b5622315e0454ce949a4af9'
RR = Path('predictfun/reviews/ladder_cancel_headroom_20261122')
sys.path.insert(0, str(PIN / 'predictfun/research'))

def module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m

def sha(b):
    return hashlib.sha256(b).hexdigest()

def blob(ref, path):
    return subprocess.check_output(['git', '-C', str(PIN), 'show', f'{ref}:{path}'])

assert subprocess.check_output(['git', '-C', str(PIN), 'rev-parse', 'HEAD'], text=True).strip() == TARGET
import pf_ladder_cancel_headroom as H
import pf_ladder_cancel_headroom_selftest as ST
import pf_ladder_cancel_headroom_mutants as MH
import pf_mutant_one as MO
import pf_mutant_summary as MS
ST.main(H)
ld = PIN / RR / 'logs/mutants_try5'
rec = {'control': MS.last_record(ld / 'lch5_control.log')}
rec.update({i: MS.last_record(ld / f'lch5_{i}.log') for i in range(len(MH.MUTANTS))})
mut_ok, mut_bad = MS.judge(len(MH.MUTANTS), rec, MO.sources(MH), [x[0] for x in MH.MUTANTS])
assert mut_ok, mut_bad
C = module('ch_checks_review', PIN / RR / 'ch_checks.py')
C.REPO = PIN; C.R = PIN / RR  # data roots stay in the shared data directory
E = C.load(); E['mine'], E['pan_mine'] = C.mine(E)
base = C.run_checks(E)
assert base['all_ok'], base
fz = C.falsify(E, base)
assert fz['falsifier_ok'], fz
(OUT / 'checks.json').write_text(json.dumps(base, indent=2, default=str)+'\n')
(OUT / 'falsifier.json').write_text(json.dumps(fz, indent=2)+'\n')

# Verify freeze closure, committed output copies and all raw data hashes.
closure = list(C.COMMITTED)
assert all(blob('2a2f78da', x) == blob(TARGET, x) for x in closure)
for f in (PIN / RR / 'outputs').glob('*.csv.gz'):
    assert gzip.decompress(f.read_bytes()) == (C.D / f.name[:-3]).read_bytes(), f
assert (PIN / RR / 'outputs/audit.json').read_bytes() == (C.D / 'audit.json').read_bytes()

# Independently checked economics above; reconstruct ALL exported ancillary
# fields from source records too, so untested checker fields are not trusted.
rows, lines, panels = [], {}, []
source_sanity = []
for (day, market), (record, cms) in sorted(E['markets'].items()):
    end = float(record['path_info']['end_rel'])
    for arm in H.ARMS:
        r = record[arm]; cm = cms[arm]
        assert r['profile']['cancel'] == 10 and r['profile']['fill_notice'] == 100
        assert len(r['fills']) == len(r['fill_mids']) == len(r['fill_class'])
        assert len({int(o['gid']) for o in r['orders']}) == len(r['orders'])
        for f, mm in zip(r['fills'], r['fill_mids']):
            for ix, h in [(6, 1000), (8, 5000), (10, 30000)]:
                assert mm[ix] is None or np.isfinite(mm[ix])
                assert f[0] + h <= end or mm[ix] is None
        fl, kp = H.fills_of(r, end), H.keeps_of(cm)
        assert sum(len(f['t']) for f in fl.values()) == len(r['fills'])
        for o in r['orders']:
            gid = int(o['gid']); F = fl.get(gid, H.EMPTY); ks, fb = kp.get(gid, H.NO_KEEPS)
            assert len(np.unique(ks)) == len(ks)
            assert np.all(ks >= o['act'])
            assert o['req'] is None or np.all(ks <= o['req'])
            g = H.generation(day, market, arm, o, F, (ks, fb), r, end)
            rows.append(g); lines[(arm, day, market, gid)] = H.timeline(F, ks)
            pr = pd.DataFrame(H.panel_rows(F, o, ks, fb), columns=H.PANEL_COLS)
            if len(pr):
                pr = pr.assign(arm=arm, day=day, m=market, gid=gid, side=int(o['side']))
                pr['cls'] = H.panel_class(pr.G.values, pr.L.values, pr.n_unv.values)
                panels.append(pr)
G = pd.DataFrame(rows)
P = pd.concat(panels, ignore_index=True)

def compare_frame(actual, expected, name):
    assert list(actual.columns) == list(expected.columns), (name, 'columns')
    assert len(actual) == len(expected), (name, 'length')
    for col in actual:
        a, b = actual[col], expected[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.allclose(a.astype(float), b.astype(float), rtol=1e-10, atol=1e-7, equal_nan=True), (name, col)
        else:
            clean = lambda x: x.map(lambda v: '' if pd.isna(v) else str(v)).tolist()
            assert clean(a) == clean(b), (name, col)

def saved(name):
    return pd.read_csv(C.D / name, dtype={'day':str, 'side':str, 'why':str}, float_precision='round_trip')

for f, new in [('generations.csv', G), ('panel.csv.gz', P), ('summary.csv', H.summarize(G)[0]), ('classes.csv', H.class_table(G)),
               ('markets.csv', H.market_table(G)), ('sequences.csv', H.sequence_table(G)), ('panel_summary.csv', H.panel_summary(P)), ('cases.csv', H.casebook(G, lines))]:
    old = saved(f)
    if 'side' in new and 'side' in old:
        old['side'] = old['side'].astype(new['side'].dtype)
    compare_frame(old, new, f)

# Prove the ancillary-check coverage limitation without changing any artifact.
coverage = {}
for name, key, col, family, target in [('native_30s_summary', 'sum', 'b_v30', 'S', 'S_summary'),
                                     ('case_timeline', 'cases', 'timeline', 'C', 'C_casebook')]:
    e = dict(E); e[key] = E[key].copy()
    e[key].loc[0, col] = 'CORRUPT' if col == 'timeline' else e[key].loc[0, col] + 10000
    coverage[name] = {'clean': base['checks'][target], 'corruption_passes_named_check': C.FAMS[family](e)[0][target]}

# Report anchor distinctions; not a new trading experiment.
interpret = {}
for arm, g in G.groupby('arm'):
    cut = g[g.b_cut.notna()]
    pure = cut[cut.b_cls == 'pure_observed_loss']
    summary = dict(generations=len(g), feasible=int(g.n_keeps.gt(0).sum()), cuts=len(cut),
        cuts_at_first=int(cut.b_cut.eq(cut.a_cut).sum()), pure_at_chosen_cut=len(pure),
        chosen_pure_also_pure_at_first=int(pure.a_cls.eq('pure_observed_loss').sum()),
        all_first_pure=int(g.a_cls.eq('pure_observed_loss').sum()),
        keep_1s=float(g.v_keep.sum()), oracle_1s=float(g.b_v.sum()), keep_END=float(g.end_keep.sum()),
        same_cut_END=float(g.b_end.sum()), same_cut_END_gain=float((g.b_end-g.end_keep).sum()),
        common_removed_END=float(g.b_suf_cc_vend.sum()), removed_good_1s=float(g.b_suf_G.sum()), removed_bad_1s=float(g.b_suf_L.sum()),
        first_vs_chosen_classes=pure.a_cls.value_counts().to_dict(),
        day_END_gain={d:float((h.b_end-h.end_keep).sum()) for d,h in g.groupby('day')})
    interpret[arm] = summary

# Exact readout / identity outputs are generated into review-owned directories.
toolout = OUT / 'readback'; toolout.mkdir(exist_ok=True)
R = module('headroom_read_review', PIN / RR / 'tools/read_headroom.py'); R.REPO=PIN; R.OUT=toolout
I = module('async_identity_review', PIN / RR / 'tools/async_identity.py'); I.REPO=PIN; I.OUT=toolout
with contextlib.redirect_stdout(io.StringIO()):
    R.main(); I.main()
for p in toolout.iterdir():
    assert p.read_bytes() == (PIN / RR / 'outputs' / p.name).read_bytes(), p.name
evidence = dict(target=TARGET, selftest='8 groups pass', mutation_records={'count':len(MH.MUTANTS),'accepted':mut_ok},
    checker=base['all_ok'], falsifier_cases=len(fz['cases']), freeze_closure=len(closure),
    full_field_reproduction={'generations':len(G), 'panel':len(P), 'tables':8},
    interpretation=interpret, checker_coverage=coverage, resources=base['details']['I'],
    limits='saved evidence and synthetic controls only; no tape replay, new target optimization, fit or new day')
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else str(x))+'\n')
print(json.dumps(evidence,indent=2,default=str))
