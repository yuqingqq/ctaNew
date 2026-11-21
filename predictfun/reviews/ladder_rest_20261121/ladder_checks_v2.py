"""INDEPENDENT CHECKS v2 of the LADDER_REST_INVENTORY_v1 run (declaration.md, Checks; session aea5bfb9, 2026-11-21). SUPERSEDES ladder_checks.py (v1,
frozen in 47d2eec1, kept with its checks.json as provenance): v1 read the measurement CSVs with pandas' default float converter, which is ~1 ulp
off, and recounted 940 generations below -1 c per submitted share where the exact values (and the summary) give 941 - one generation sits
on the -1 c boundary. v2 reads every CSV with float_precision='round_trip' (exact); no check, threshold or reading changed. Recomputed from the stored
evidence (the day records, arrays and feature files, the measurement CSVs and audit.json) without the runner's measurement code:
 A identity: every frozen file committed and byte-equal; every day record bound to the manifest and the current sources; every output hash;
 B membership: the records are the manifest's selected markets once each, statuses declared, complete ones with a ledger; the pilot = the
   manifest's pilot; markets.csv = the records;
 C ledger: fills inside their order's activation / removal and END, within their order's size and their public trade's size and side, at or
   through the order's price; whole-share sizes 1..100, rungs 0 / 1, prices 1..99; cash, position and value recomputed; per tick at most two
   unclosed orders per side and the reservation inequalities;
 D strategy: the mode path recomputed from the client positions (the hysteresis S / R); every POST's quantity recomputed from its entry
   snapshot (floor(min(q0 f, D - B_side, room))), never on a restricted side; every order has exactly one entry snapshot; every inventory
   pull on a side the mode restricts at its request; no two unclosed same-side orders share a price; no new order crosses an unclosed
   opposite order; every structural subreason flag and primary reason consistent with its logged book;
 E measurement: the summary, per-day and per-market values, fills, shares and generations; severe counts; the minute increments telescope and
   split; markout = edge + movement; no markout past END; the strategy-value predicate recomputed;
 F the legacy reproduction (engine parity, the 107 inputs) and the reconstruction of every snapshot;
 G the pilot records equal the full run's for the pilot markets (ledger fingerprints);
 H resources: this round's CPU <= the cap; peaks and high-waters <= 1,536 MiB.
falsify(): each corruption of a copy of the evidence must fail its check while the clean evidence passes them all. selftest(): the
recomputation helpers on hand data. Usage: ladder_checks.py [selftest]"""
import copy
import glob
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_rest_20261121'; D = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'
HERE = REPO / 'predictfun/research'
COMMITTED = ['predictfun/research/pf_ladder_rest.py', 'predictfun/research/pf_ladder_rest_selftest.py', 'predictfun/research/pf_branch_ladder.py',
             'predictfun/research/pf_branch_ladder_mutants.py', 'predictfun/research/pf_ladder_rest_mutants.py', 'predictfun/reviews/ladder_rest_20261121/declaration.md',
             'predictfun/reviews/ladder_rest_20261121/manifest.json', 'predictfun/reviews/ladder_rest_20261121/launch_ladder_rest.sh',
             'predictfun/reviews/ladder_rest_20261121/ladder_checks.py', 'predictfun/reviews/ladder_rest_20261121/ladder_checks_v2.py',
             'predictfun/reviews/ladder_rest_20261121/tools/pilot_projection_lr.py']
TOL = 1e-6
MODES = {'UNRESTRICTED': 0, 'LONG_RESTRICTED': 1, 'SHORT_RESTRICTED': 2}
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == (REPO / rel).read_bytes()


def gz_rows(b):
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines() if x]


def load(sub='days'):
    """The evidence of one stage (days / pilot) and the measurement."""
    man = json.loads((R / 'manifest.json').read_text()); recs, arrs, days = [], {}, {}
    for day in man['days']:
        p = D / sub / f'{day}.json'
        if not p.exists():
            continue
        rec = json.loads(p.read_text()); days[day] = rec; b = (D / sub / f'{day}.jsonl.gz').read_bytes(); rr = gz_rows(b)
        rec['_hash_ok'] = bool(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and sha(D / sub / f'{day}_arrays.npz') == rec['sha256']['npz']
                               and sha(D / sub / f'{day}_features.parquet') == rec['sha256']['feat'])
        recs += rr; z = np.load(D / sub / f'{day}_arrays.npz'); arrs.update({f'{day}|{k}': z[k] for k in z.files})
    E = dict(man=man, man_sha=sha(R / 'manifest.json'), recs=recs, arrs=arrs, days=days, sub=sub)
    md = D / 'measure'
    if sub == 'days' and (md / 'audit.json').exists():
        E['audit'] = json.loads((md / 'audit.json').read_text())
        for nm in ('markets', 'generations', 'fills', 'minutes', 'summary'):
            E[nm] = pd.read_csv(md / f'{nm}.csv', float_precision='round_trip'); E[f'{nm}_sha'] = sha(md / f'{nm}.csv')
    return E


def sg(s):
    return 1. if int(s) == 0 else -1.


def next_mode(old, I, S, Rb):
    """The declared hysteresis, re-implemented."""
    return 'LONG_RESTRICTED' if I >= S - 1e-9 else 'SHORT_RESTRICTED' if I <= -S + 1e-9 else 'UNRESTRICTED' if abs(I) <= Rb + 1e-9 else old


def expected_q(snap, cfg):
    """An entry's quantity from its snapshot: floor(min(q0 f, D - B_side, room) + 1e-9)."""
    s, I = int(snap['side']), float(snap['position']); inv = cfg['inventory']
    f = min(1., max(0., (1. - I / inv['S']) if s == 0 else (1. + I / inv['S']))); Dd = inv['Qside'] * f; Bs = snap['B'] if s == 0 else snap['A']
    room = (cfg['cap'] - I - snap['B']) if s == 0 else (cfg['cap'] + I - snap['A'])
    return float(math.floor(min(cfg['q0'] * f, max(0., Dd - Bs), max(0., room)) + 1e-9))


def overlaps_same_price(orders, closed, end):
    """Pairs of same-side orders unclosed at the same time at the same price (intervals [submit, closed) on the client clock)."""
    bad = 0; by = {}
    for o in orders:
        c = closed.get(str(o['gid'])); by.setdefault((int(o['side']), float(o['price'])), []).append((float(o['submit']), float(c[1]) if c else end + 1.))
    for iv in by.values():
        iv.sort()
        bad += sum(1 for (a0, a1), (b0, b1) in zip(iv[:-1], iv[1:]) if b0 < a1)
    return bad


def own_cross(orders, closed, end):
    """New orders priced at / through an opposite order unclosed at their submission (same-tick: the opposite counts when its side came first)."""
    bad = 0
    for o in orders:
        t, s, p = float(o['submit']), int(o['side']), float(o['price']); k = int(round(t / 50.)); first = k % 2
        for x in orders:
            if int(x['side']) == s:
                continue
            c = closed.get(str(x['gid'])); xs = float(x['submit']); alive = (xs < t or (xs == t and first == int(x['side']))) and not (c and float(c[1]) <= t)
            if alive and ((s == 0 and p >= float(x['price'])) or (s == 1 and p <= float(x['price']))):
                bad += 1
    return bad


def struct_ok(e):
    behind = e['best'] - e['price'] if int(e['side']) == 0 else e['price'] - e['best']; cross = e['price'] >= e['opp'] if int(e['side']) == 0 else e['price'] <= e['opp']
    fl = dict(no_book=e['best'] < 0, distance=behind > 3. + e['rung'], cross=bool(cross))
    return bool(all(bool(e[k]) == v for k, v in fl.items()) and any(fl.values()) and e['primary'] == next(k for k in ('no_book', 'distance', 'cross') if fl[k]))


def run_checks(E):  # noqa: C901
    man = E['man']; cfg = man['config']; inv = cfg['inventory']; chk, det = {}, {}
    comp = [r for r in E['recs'] if r['status'] == 'complete']
    # A identity
    sys.path.insert(0, str(HERE)); import pf_ladder_rest as M
    srcs = {n: sha(HERE / n) for n in M.SOURCES}
    chk['A_committed'] = all(committed(p) for p in COMMITTED)
    chk['A_bound'] = bool(E['days'] and all(d['manifest_sha256'] == E['man_sha'] and d['sources'] == srcs and d['_hash_ok'] for d in E['days'].values()))
    if 'audit' in E:
        chk['A_measure_bound'] = bool(E['audit']['manifest_sha256'] == E['man_sha'] and all(E['audit']['outputs'].get(f'{nm}.csv') == E[f'{nm}_sha'] for nm in ('markets', 'generations', 'fills', 'minutes', 'summary')))
    # B membership
    want = sorted((x['day'], int(x['m'])) for x in man['selected'] if E['sub'] == 'days' or int(x['m']) in {int(m) for m in man['pilot'][x['day']]})
    got = sorted((r['day'], int(r['m'])) for r in E['recs']); det['statuses'] = {}
    for r in E['recs']:
        det['statuses'][r['status']] = det['statuses'].get(r['status'], 0) + 1
    chk['B_membership'] = got == want
    chk['B_statuses'] = all(r['status'] == 'complete' or r['status'].startswith(('incomplete:', 'ledger:')) or r['status'] == 'missing_market_tape' for r in E['recs'])
    chk['B_complete_have_ledgers'] = all(('ledger' in r) == (r['status'] == 'complete' or r['status'].startswith('ledger:')) for r in E['recs'])
    if 'markets' in E:
        mk = E['markets']; chk['B_markets_csv'] = sorted(zip(mk.day.astype(str), mk.m.astype(int), mk.status)) == sorted((r['day'], int(r['m']), r['status']) for r in E['recs'])
    # C ledger
    c_bad = dict(fill_window=0, order_size=0, trade=0, price=0, value=0, levels=0, reservation=0, shape=0)
    for r in comp:
        L_ = r['ledger']; od = {int(o['gid']): o for o in L_['orders']}; end = float(L_['end']); used = {}
        for f in L_['fills']:
            t, g, s, q, p, i = float(f[0]), int(f[1]), int(f[2]), float(f[3]), float(f[4]), int(f[5]); o = od[g]
            c_bad['fill_window'] += int(not (float(o['act']) <= t < float(o['remove']) and t <= end + 1e-9 and q > 0 and int(o['side']) == s))
            tr = L_['trade_rows'][str(i)]; used[i] = used.get(i, 0.) + q
            c_bad['trade'] += int(int(tr[1]) != s); c_bad['price'] += int(not ((tr[2] <= o['price']) if s == 0 else (tr[2] >= o['price'])))
        c_bad['trade'] += sum(1 for i, q in used.items() if q > float(L_['trade_rows'][str(i)][3]) + 1e-9)
        for o in L_['orders']:
            fq = sum(float(f[3]) for f in L_['fills'] if int(f[1]) == int(o['gid']))
            c_bad['order_size'] += int(not (fq <= float(o['size']) + 1e-9 and 1. <= float(o['size']) <= cfg['q0'] and float(o['size']) == round(float(o['size']))))
            c_bad['shape'] += int(not (float(o['rung']) in (0., 1.) and 1. <= float(o['price']) <= 99. and float(o['price']) == round(float(o['price']))))
        cash = -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in L_['fills']); pos = sum(sg(f[2]) * float(f[3]) for f in L_['fills'])
        c_bad['value'] += int(not (abs(cash - L_['cash']) < TOL and abs(pos - L_['position']) < TOL and abs(cash + pos * L_['mark'] - L_['value']) < TOL))
        a = lambda nm: E['arrs'][f"{r['day']}|{int(r['m'])}__{nm}"]
        c_bad['levels'] += int(((a('npp') + a('nlive') + a('ncp')) > cfg['levels']).any())
        c_bad['reservation'] += int(((a('posc') + a('res')[:, 0]) > cfg['cap'] + 1e-9).any() or ((-a('posc') + a('res')[:, 1]) > cfg['cap'] + 1e-9).any())
    det['C'] = c_bad
    for k, v in c_bad.items():
        chk[f'C_{k}'] = v == 0
    # D strategy
    d_bad = dict(mode=0, size=0, restricted_post=0, entry_membership=0, pull_side=0, dup_price=0, own_cross=0, struct=0)
    for r in comp:
        L_ = r['ledger']; a = lambda nm: E['arrs'][f"{r['day']}|{int(r['m'])}__{nm}"]; pc, md = a('posc'), a('mode'); m = 'UNRESTRICTED'; codes = []
        for I in pc:
            m = next_mode(m, float(I), inv['S'], inv['R']); codes.append(MODES[m])
        d_bad['mode'] += int(not np.array_equal(np.asarray(codes), md.astype(int)))
        ent = [s_ for s_ in L_['snapshots'] if s_['kind'] == 'entry']; byk = {}
        for s_ in ent:
            byk.setdefault(s_['gid'], []).append(s_)
            d_bad['size'] += int(expected_q(s_, cfg) != float(s_['q'])); d_bad['restricted_post'] += int((s_['side'] == 0 and s_['mode'] == 'LONG_RESTRICTED') or (s_['side'] == 1 and s_['mode'] == 'SHORT_RESTRICTED'))
        for o in L_['orders']:
            e_ = byk.get(int(o['gid']), [])
            d_bad['entry_membership'] += int(not (len(e_) == 1 and float(e_[0]['q']) == float(o['size']) and float(e_[0]['t']) == float(o['submit']) and float(e_[0]['price']) == float(o['price'])
                                              and float(e_[0]['rung']) == float(o['rung'])))
            if o['why'] == 'policy':
                mk = int(md[int(round(float(o['req']) / 50.))]); d_bad['pull_side'] += int(not ((int(o['side']) == 0 and mk == 1) or (int(o['side']) == 1 and mk == 2)))
        d_bad['entry_membership'] += int(None in byk)
        d_bad['dup_price'] += overlaps_same_price(L_['orders'], L_['closed'], float(L_['end'])); d_bad['own_cross'] += own_cross(L_['orders'], L_['closed'], float(L_['end']))
        d_bad['struct'] += sum(0 if struct_ok(e) else 1 for e in L_['struct_log'])
    det['D'] = d_bad
    for k, v in d_bad.items():
        chk[f'D_{k}'] = v == 0
    # E measurement
    if 'summary' in E:
        S = E['summary']; G = E['generations']; F = E['fills']; Mi = E['minutes']
        row = S[(S.day.astype(str) == 'all') & (S.side == 'both') & (S.rung.astype(str) == 'all')].iloc[0]
        tv = sum(float(r['ledger']['value']) for r in comp); nf = sum(len(r['ledger']['fills']) for r in comp); fq = sum(float(f[3]) for r in comp for f in r['ledger']['fills'])
        ng = sum(len(r['ledger']['orders']) for r in comp)
        chk['E_summary_totals'] = bool(abs(float(row['value']) - tv) < 1e-4 and int(row['fills']) == nf and abs(float(row['filled_shares']) - fq) < 1e-6 and int(row['generations']) == ng)
        perday = {d: sum(float(r['ledger']['value']) for r in comp if r['day'] == d) for d in man['days']}
        sd = S[(S.side == 'both') & (S.rung.astype(str) == 'all') & (S.day.astype(str) != 'all')]
        chk['E_per_day'] = all(abs(float(x.value) - perday[str(x.day)]) < 1e-4 for x in sd.itertuples())
        gm = G.groupby([G.day.astype(str), G.m.astype(int)]).value.sum().to_dict()
        chk['E_generations_sum'] = all(abs(gm.get((r['day'], int(r['m'])), 0.) - float(r['ledger']['value'])) < 1e-4 for r in comp)
        chk['E_severe'] = bool(int(row['n_severe']) == int((G.value < -30.).sum()) and int(row['n_severe_per_share']) == int((G.value / G['size'] < -1.).sum()))
        mm = Mi.groupby([Mi.day.astype(str), Mi.m.astype(int)]).dv.sum().to_dict()
        chk['E_minutes'] = bool(all(abs(mm.get((r['day'], int(r['m'])), 0.) - float(r['ledger']['value'])) < 1e-4 for r in comp) and ((Mi.dv - Mi.carry - Mi.new).abs() < 1e-4).all())
        v = F.dropna(subset=['edge', 'mo1000', 'mv1000'])
        chk['E_markout_identity'] = bool(((v.mo1000 - v.edge - v.mv1000).abs() < 1e-6).all())
        ends = {(r['day'], int(r['m'])): float(r['ledger']['end']) for r in comp}
        chk['E_no_markout_past_end'] = bool(all(pd.isna(x.mo30000) for x in F.itertuples() if x.t + 30000. > ends[(str(x.day), int(x.m))]))
        pr = E['audit']['predicates']; days = [d for d in man['days'] if perday[d] != 0 or any(r['day'] == d for r in comp)]
        sv = [perday[d] for d in days]
        want_sv = 'INCOMPLETE' if len(days) < len(man['days']) else 'POSITIVE_EVERY_DAY' if all(x > 0 for x in sv) else 'NEGATIVE_EVERY_DAY' if all(x < 0 for x in sv) else 'MIXED'
        chk['E_strategy_value_predicate'] = pr['strategy_value']['answer'] == want_sv
    # F legacy and reconstruction
    lg = [r['legacy'] for r in E['recs'] if 'legacy' in r and 'status' not in r['legacy']]
    det['F'] = dict(legacy_worlds=len(lg), features_equal=sum(x['features_equal'] for x in lg), reconstruct_checked=sum(r['ledger']['reconstruct']['checked'] for r in comp))
    chk['F_legacy_engine'] = bool(lg and all(x['engine_all'] for x in lg))
    chk['F_legacy_features'] = bool(lg and all(x['features_bad'] == 0 and x['points_unmatched'] == 0 for x in lg) and det['F']['features_equal'] > 0)
    chk['F_reconstruction'] = all(r['ledger']['reconstruct']['failed'] == 0 and r['ledger']['reconstruct']['checked'] == len(r['ledger']['snapshots']) for r in comp)
    # G pilot determinism
    if E['sub'] == 'days' and E.get('pilot') is not None:
        pf = {(r['day'], int(r['m'])): (r['status'], (r.get('ledger') or {}).get('fingerprint')) for r in E['pilot']['recs']}
        full = {(r['day'], int(r['m'])): (r['status'], (r.get('ledger') or {}).get('fingerprint')) for r in E['recs']}
        chk['G_pilot_equals_full'] = bool(pf and all(full.get(k) == v for k, v in pf.items()))
    # H resources
    cpu, peak = 0., 0
    for p in glob.glob(str(R / 'logs' / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                cpu += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    pj = json.loads((R / 'logs' / 'pilot_projection.json').read_text()) if (R / 'logs' / 'pilot_projection.json').exists() else {}
    det['H'] = dict(core_hours=cpu / 3600., peak_mib=peak, cap=pj.get('cap'), hwm=max([float(d['hwm_mib']) for d in E['days'].values()] or [0.]))
    chk['H_resources'] = bool(pj.get('cap') is not None and cpu / 3600. <= float(pj['cap']) and peak < 1536 and det['H']['hwm'] <= 1536.)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


CASES = {
    'C_value': lambda E: _comp(E)[0]['ledger'].update(value=float(_comp(E)[0]['ledger']['value']) + 1.),
    'C_trade': lambda E: _first_fill(E).__setitem__(3, float(_first_fill(E)[3]) + 1e6),
    'B_membership': lambda E: E['recs'].pop(),
    'D_size': lambda E: _first_entry(E).update(q=float(_first_entry(E)['q']) + 1.),
    'D_mode': lambda E: _first_mode(E),
    'D_dup_price': lambda E: _dup(E),
    'D_struct': lambda E: _struct(E),
    'E_summary_totals': lambda E: E.update(summary=E['summary'].assign(value=E['summary'].value + 1.)),
    'F_legacy_engine': lambda E: next(r['legacy'] for r in E['recs'] if 'legacy' in r and 'status' not in r['legacy']).update(engine_all=False),
    'G_pilot_equals_full': lambda E: next(r for r in E['pilot']['recs'] if 'ledger' in r)['ledger'].update(fingerprint='x'),
}


def _comp(E):
    return [r for r in E['recs'] if r['status'] == 'complete']


def _first_fill(E):
    return next(r for r in _comp(E) if r['ledger']['fills'])['ledger']['fills'][0]


def _first_entry(E):
    return next(s_ for r in _comp(E) for s_ in r['ledger']['snapshots'] if s_['kind'] == 'entry')


def _first_mode(E):
    r = _comp(E)[0]; k = f"{r['day']}|{int(r['m'])}__mode"; a = np.array(E['arrs'][k], copy=True); a[len(a) // 2] = 2 if a[len(a) // 2] != 2 else 0; E['arrs'][k] = a


def _dup(E):
    for r in _comp(E):
        os_ = r['ledger']['orders']
        for i, o in enumerate(os_):
            for x in os_[i + 1:]:
                if x['side'] == o['side'] and x['price'] != o['price']:
                    c = r['ledger']['closed'].get(str(o['gid']))
                    if float(x['submit']) < (float(c[1]) if c else 1e18):
                        x['price'] = o['price']; return


def _struct(E):
    for r in _comp(E):
        if r['ledger']['struct_log']:
            e = r['ledger']['struct_log'][0]; e['primary'] = 'cross' if e['primary'] != 'cross' else 'distance'; return


def falsify(E):
    """Every case: the clean evidence passes its check; the corrupted copy fails it."""
    base = run_checks(E); out = {}
    for name, fn in CASES.items():
        if name.startswith('G_') and E.get('pilot') is None:
            continue
        X_ = copy.deepcopy(E); fn(X_); r = run_checks(X_)
        out[name] = dict(clean=bool(base['checks'].get(name)), corrupted=bool(r['checks'].get(name)), applied=name in r['checks'])
    return dict(cases=out, falsifier_ok=bool(out and all(v['clean'] and not v['corrupted'] and v['applied'] for v in out.values())))


def selftest():
    bad = []
    tab = [('UNRESTRICTED', 100., 'LONG_RESTRICTED'), ('LONG_RESTRICTED', 51., 'LONG_RESTRICTED'), ('LONG_RESTRICTED', 50., 'UNRESTRICTED'), ('LONG_RESTRICTED', -100., 'SHORT_RESTRICTED')]
    if not all(next_mode(a, I, 100., 50.) == b for a, I, b in tab):
        bad.append('next_mode')
    cfg = dict(q0=100., cap=200., inventory=dict(Qside=200., S=100., R=50.))
    if not (expected_q(dict(side=0, position=40., B=0., A=0.), cfg) == 60. and expected_q(dict(side=0, position=40., B=60., A=0.), cfg) == 60.
            and expected_q(dict(side=1, position=40., B=0., A=0.), cfg) == 100. and expected_q(dict(side=0, position=100., B=0., A=0.), cfg) == 0.):
        bad.append('expected_q')
    od = [dict(gid=0, side=0, price=50., submit=0.), dict(gid=1, side=0, price=50., submit=100.), dict(gid=2, side=0, price=49., submit=100.)]
    if not (overlaps_same_price(od, {'0': ['filled', 150.]}, 1000.) == 1 and overlaps_same_price(od, {'0': ['filled', 100.]}, 1000.) == 0):
        bad.append('overlaps_same_price')
    oc = [dict(gid=0, side=1, price=52., submit=0.), dict(gid=1, side=0, price=52., submit=200.)]
    if not (own_cross(oc, {}, 1000.) == 1 and own_cross(oc, {'0': ['removed', 200.]}, 1000.) == 0):
        bad.append('own_cross')
    e = dict(side=0, price=50., best=54., opp=56., rung=0., no_book=False, distance=True, cross=False, primary='distance')
    if not (struct_ok(e) and not struct_ok(dict(e, primary='cross')) and not struct_ok(dict(e, best=52., distance=False))):
        bad.append('struct_ok')
    import io
    xs = [0.1 * k + 1e-13 * k for k in range(1, 4000)]; buf = io.StringIO(); pd.DataFrame(dict(v=xs)).to_csv(buf, index=False)
    if not np.array_equal(pd.read_csv(io.StringIO(buf.getvalue()), float_precision='round_trip').v.values, np.asarray(xs)):
        bad.append('round_trip parse')
    print('CHECKS SELFTEST ' + ('FAILED: ' + ', '.join(bad) if bad else 'PASSED'))
    return not bad


def main():
    E = load('days'); E['pilot'] = load('pilot') if (D / 'pilot').exists() else None
    res = run_checks(E); (R / 'checks_v2.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E)
    (R / 'checks_falsifier_v2.json').write_text(json.dumps(fz, indent=1) + '\n')
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok']), indent=1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        sys.exit(0 if selftest() else 1)
    main()
