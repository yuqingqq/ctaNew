"""Pinned review: synthetic controls plus saved-artifact reconciliation, no tape replay/fit.
Run through PF_SLOT=codex run_light.sh. All evidence is written beside this script.
"""
from pathlib import Path
import gzip, hashlib, importlib.util, json, os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-la-review-6b14fbf9')
TARGET = '6b14fbf951a96baebd885f96a93ecd10a7744bd8'
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', '-C', str(PIN), 'rev-parse', 'HEAD']).decode().strip() == TARGET
os.environ['PF_TEACHER_SYN'] = str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
sys.path.insert(0, str(PIN/'predictfun/research'))
import pf_ladder_audit as A
import pf_ladder_audit_selftest as ST
import pf_ladder_rest_selftest_v2 as V2
ST.main(A)
V2.main(A.M)

rp = PIN/'predictfun/reviews/ladder_audit_20261121'
spec = importlib.util.spec_from_file_location('audit_checks_review', rp/'audit_checks_v3.py')
CK = importlib.util.module_from_spec(spec); spec.loader.exec_module(CK)
CK.REPO = PIN; CK.R = rp; CK.REV = PIN/'predictfun/reviews/automatic/20261121_ladder_rest_review'
E = CK.load()
res = CK.run_checks(E); assert res['all_ok'], res
out = dict(target=TARGET, selftests='audit and ladder v2 passed', checks=res)
print('Pinned selftests and 23 saved checks passed', flush=True)

# Bind every receipt copy, including compressed outputs, to the saved data.
sha = lambda b: hashlib.sha256(b).hexdigest()
hashes = {}
for line in (rp/'outputs/SHA256SUMS').read_text().splitlines():
    h, n = line.split('  ', 1); p = rp/'outputs'/n
    assert sha(p.read_bytes()) == h, n
    raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
    orig = CK.D/(n[:-3] if p.suffix == '.gz' else n)
    assert raw == orig.read_bytes(), n
    hashes[n] = h
out['receipt_hashes'] = hashes

# All snapshot identities and dictionary inputs, then ALL windows, not only every 25th.
keys = ['day', 'm', 'kind', 'side', 'gid', 't']
S = E['states']; F = E['features']
assert not S.duplicated(keys).any() and not F.duplicated(keys).any()
J = S.merge(F, on=keys, suffixes=('_audit', '_saved'), validate='one_to_one')
assert len(J) == len(S) == len(F) == 14968
for c in A.M.DICT_COLS:
    if c == 'inv_signed':
        want = np.where(J.side == 0, 1., -1.) * J.ctx_inventory_shares
        got = J[c]
    else:
        want, got = J[c+'_saved'], J[c+'_audit']
    assert np.array_equal(got.astype(float), want.astype(float), equal_nan=True), c
rec = {(r['day'], int(r['m'])): r for r in E['recs'] if r['status'] == 'complete'}
fv, orders = {}, {}
for k, r in rec.items():
    for o in r['ledger']['orders']: orders[k+(int(o['gid']),)] = o
    for f, mm in zip(r['ledger']['fills'], r['ledger']['fill_mids']): fv.setdefault(k+(int(f[1]),), []).append((f, mm))
for x in S.to_dict('records'):
    k = (x['day'], int(x['m']), int(x['gid'])); o = orders[k]
    e = o['act'] if x['kind'] == 'entry' else max(x['t']+37., o['submit']+95.) if x['t'] < o['act'] else x['t']+37.
    win = [(f, mm) for f, mm in fv.get(k, []) if x['kind'] == 'entry' or f[0] >= e]
    pre = [(f, mm) for f, mm in fv.get(k, []) if x['kind'] != 'entry' and x['t'] <= f[0] < e]
    cp = [(f, mm) for f, mm in win if o['req'] is not None and o['req'] <= f[0] < o['remove']]
    assert abs(x['effect']-e) < 1e-9
    for pref, rows in [('win', win), ('pre', pre), ('cp', cp)]:
        assert len(rows) == x[pref+'_n']
        assert abs(sum(f[3] for f, mm in rows)-x[pref+'_q']) < 1e-6
        cens = any(mm[6] is None for f, mm in rows)
        expected = np.nan if cens else sum((1 if f[2] == 0 else -1)*f[3]*(mm[6]-f[4]) for f, mm in rows)
        assert np.isclose(x[pref+'_mo1s'], expected, atol=1e-6, rtol=0, equal_nan=True)
    assert bool(x['censored_1s']) == any(mm[6] is None for f, mm in win)
out['all_rows'] = dict(membership_inputs_windows_checked=len(S), kinds=S.kind.value_counts().to_dict())

# Recompute every tail contrast directly from states, independently of saved means/lead flags.
leads, contrasts, normalized = [], [], []
for kind in A.KINDS:
    for c in A.M.DICT_COLS:
        g = S[S.kind == kind].copy()
        g['bn'] = [CK.bin_of(v, E['edges'].get(f'{kind}|{s}|{c}', [])) for v, s in zip(g[c], g.side)]
        un = g[~g.censored_1s & g.win_mo1s.notna()]
        cells, support = [], True
        for day in E['days']:
            for side in (0, 1):
                a, b = [un[(un.day == day) & (un.side == side) & (un.bn == bn)] for bn in ('q1', 'q5')]
                support &= min(len(a), len(b)) >= 10
                cells.append(float(b.win_mo1s.mean()-a.win_mo1s.mean()))
                if kind == 'live' and c == 'ctx_bid_reserved_shares':
                    alltails = [g[(g.day == day) & (g.side == side) & (g.bn == bn)] for bn in ('q1','q5')]
                    normalized.append(dict(day=day, side=side, n1=len(a), n5=len(b),
                        q1_valid=float(a.win_q.sum()), q5_valid=float(b.win_q.sum()),
                        q1_all=float(alltails[0].win_q.sum()), q5_all=float(alltails[1].win_q.sum()),
                        per_share_valid_difference=float(b.win_mo1s.sum()/b.win_q.sum()-a.win_mo1s.sum()/a.win_q.sum())))
        a, b = [un[un.bn == bn] for bn in ('q1','q5')]
        pooled = float(b.win_mo1s.mean()-a.win_mo1s.mean())
        flag = bool(support and np.isfinite(cells).all() and len(set(np.sign(cells))) == 1 and cells[0] != 0)
        saved = E['leads'][(E['leads'].kind == kind) & (E['leads'].feature == c)].iloc[0]
        assert bool(saved.lead) == flag
        assert np.isclose(saved.pooled_q5_minus_q1, pooled, equal_nan=True)
        row = dict(kind=kind, feature=c, lead=flag, cells=cells, pooled=pooled)
        contrasts.append(row)
        if flag: leads.append(row)
out['leads'] = leads; out['lead_per_share_common_population'] = normalized
G = E['generations_v2']
out['censor_counter'] = dict(reported=E['audit']['cross']['generation_tails']['mo1s_censored'],
    actual=int(G.lifetime_mo1s.isna().sum()), no_fill=int((G.n_fills == 0).sum()))

# Verify persisted causal sample identities and counters; this does NOT replay raw tapes.
cr = {}
for day, c in E['causal'].items():
    want = A.sample_snapshots(F, day)
    ks = ['m', 'kind', 'side', 'gid', 't']
    assert sorted(tuple(r[k] for k in ks) for r in c['rows']) == sorted(tuple(r[k] for k in ks) for r in want.to_dict('records'))
    for nm in ('market_same','market_fired','order_same','order_fire_eligible','order_fired'):
        assert c[nm] == sum(bool(r[nm]) for r in c['rows'])
    assert c['sources'] == {n:sha((PIN/'predictfun/research'/n).read_bytes()) for n in A.SOURCES}
    cr[day] = {k:c[k] for k in ('n','market_same','market_fired','order_same','order_fire_eligible','order_fired')}
out['saved_causal_receipts'] = cr
out['falsifier'] = CK.falsify(E)
assert out['falsifier']['falsifier_ok']
(OUT/'evidence.json').write_text(json.dumps(out, indent=2, default=str)+'\n')
print(json.dumps(dict(all_rows=out['all_rows'], leads=leads, censor_counter=out['censor_counter'], falsifier_ok=True), indent=2))
