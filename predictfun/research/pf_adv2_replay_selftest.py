"""SELFTEST of pf_adv2_replay (the bounded own-state R4 diagnostic; 2026-11-21, session aea5bfb9). run(M) checks the module object M (the
file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only:
 1. population: per (day, hour) the first eight worlds by rank (ties by the canonical key), a smaller stratum whole, partitions of at most 32
    of one day in canonical order;
 2. the reached selection: the minimum rank per side / head among model decisions in absent / live states; a fallback decision is never
    chosen; the choice ignores the predictions (an outcome-free key);
 3. intervals: union, intersection and length on hand cases; active_intervals on a hand-built engine (activation to the removal effect or
    the full fill, the at-best part from the latest venue frame, a refused order excluded);
 4. end to end on the real engine and the cached synthetic world (stub heads that change sign with the tick, the stub teacher):
    - every policy's episode reconciles with zero binding failures; the recorder holds one entry per callback;
    - the readings' identities: value = value_bid + value_ask = the END markout; best time <= any time per side; both <= each side <= either;
      N0 has no generation, no quote time and value 0; ENTRY_PA pulls every reviewed generation at its first review; a keep-always RICH
      never does;
    - the recorded focal fingerprint of RICH decisions (both sides, an early and a late one) equals pf_r2_labels.bound_fork's context
      rebuilt with RICH as the roll-in; with PI0 as the roll-in it differs (the positive control);
    - reached_pairs: each selected state's pair is ok, reconciled and bound, its fork fingerprint the recorded one, its inclusion count the
      number of eligible decisions; a tampered recorded fingerprint is refused (fork_mismatch); one entry per side / head, the missing ones
      no_eligible_decision (an empty selection: four such entries and no pair run);
    - churn: one record per reviewed generation, the creating POST's time (the branch's decision log) and the first live review's action;
      the input changes equal review - post; RICH's values are its predictions; ENTRY_PA's first reviews are PULLs; cap_ok on every policy;
    - exposure: RICH's 'state|status' counts total its callbacks, every status's count is its own, and every dimension / horizon bin
      totals its decisions with inputs;
 5. run_part refuses before reading any outcome when the fit manifest is not committed."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ADV2 REPLAY SELFTEST FAILED'


class _Head:
    def __init__(self, f):
        self.f = f

    def predict(self, X, num_threads=1):
        return np.asarray([self.f(x) for x in X], float)


def _refused(fn, msg=None):
    try:
        fn()
    except ValueError as e:
        return msg is None or msg in str(e)
    except Exception:                                                                       # noqa: BLE001  (a crash is not the check)
        return False
    return False


class _E:
    pass


def run(M):  # noqa: C901
    bad = []; A, L, BE, R2, T, LS = M.A, M.L, M.BE, M.R2, M.T, M.LS
    import pf_r2_labels_selftest as LT
    import pf_r2_teacher_selftest as TS
    I, KP, X = R2.I, R2.KP, R2.X
    # ---- 1 population
    sel = []
    for day in ('20261026', '20261027'):
        for h in range(3):
            n_ = 11 if h != 2 else 5
            for i in range(n_):
                sel.append(dict(day=day, m=100 * h + i, t_ms=1000 * i + h, hour=h, rank=f'{(7 * i + h) % 11:02d}', pi=.5))
    plan = dict(days=['20261026', '20261027'], sample={d: dict(selected=[w for w in sel if w['day'] == d]) for d in ('20261026', '20261027')})
    saved = M.PART
    try:
        M.PART = 7; parts = M.population(plan)
    finally:
        M.PART = saved
    got = [(d, w['hour'], w['rank']) for d, ws in parts for w in ws]
    want = []
    for d in ('20261026', '20261027'):
        for h in range(3):
            ws = sorted([w for w in sel if w['day'] == d and w['hour'] == h], key=lambda w: (w['rank'], LS.ckey(LS.wkey(w))))[:8]
            want += [(d, h, w['rank']) for w in ws]
    if sorted(got) != sorted(want) or any(len(ws) > 7 for _, ws in parts) or any(w['day'] != d for d, ws in parts for w in ws):
        bad.append('population: not the first eight by rank per (day, hour) in partitions of one day')
    for d, ws in parts:
        if [(w['m'], w['t_ms']) for w in ws] != sorted((w['m'], w['t_ms']) for w in ws):
            bad.append('population: a partition is not in canonical order'); break
    # ---- 2 the reached selection
    rec = [dict(k=k, t=50. * k, side=s, state=st, generation=1, fingerprint='f', status=stt, action='POST', A=float(k), row={}, oid=7 + s)
           for k in range(20) for s in (0, 1) for st, stt in (('absent', 'model'), ('live', 'model'), ('absent', 'outside_training_range'))]
    s1 = M.select_reached(rec, '20261026', 5); rec2 = [dict(r, A=-r['A'] - 3.) for r in rec]; s2 = M.select_reached(rec2, '20261026', 5)
    if set(s1) != {(0, 'post'), (0, 'keep'), (1, 'post'), (1, 'keep')} or {k: v['key'] for k, v in s1.items()} != {k: v['key'] for k, v in s2.items()}:
        bad.append('selection: not one per side / head, or it depends on the predictions')
    if any(v['eligible'] != 20 for v in s1.values()):
        bad.append(f"selection: the inclusion counts are not the eligible decisions ({[v['eligible'] for v in s1.values()]})")
    for (s, head), v in s1.items():
        cands = [r for r in rec if r['side'] == s and r['status'] == 'model' and r['state'] == ('absent' if head == 'post' else 'live')]
        keys = [dict(day='20261026', instrument_id=5, original_episode_id=7 + s, side=s, head=head, generation=1, decision_ms=int(r['t'])) for r in cands]
        if v['key'] != min(keys, key=lambda k_: (M.rank_of(k_), A.ck(k_))) or v['dec']['status'] != 'model':
            bad.append(f'selection: {s} {head} is not the minimum-rank model decision')
    # ---- 3 intervals
    if M._union([[0, 5], [3, 8], [10, 12], [12, 13]]) != [[0, 8], [10, 13]] or M._inter([[0, 8], [10, 13]], [[5, 11]]) != [[5, 8], [10, 11]] or M._len([[0, 8], [10, 13]]) != 11.:
        bad.append('intervals: union / intersection / length')
    E = _E(); f = np.full((4, 22), -1.); f[:, 0] = [0., 100., 200., 300.]; f[:, 2] = [40., 41., 40., 40.]; f[:, 12] = [45., 45., 44., 44.]; E.f = f
    E.o = [dict(gid=0, side=0, price=40., size=30., act=58., remove=250., refused=False), dict(gid=1, side=1, price=45., size=30., act=60., remove=1e30, refused=False),
           dict(gid=2, side=0, price=40., size=30., act=10., remove=1e30, refused=True)]
    E.fills = [(150., 1, 1, 30., 45., 0)]
    an, bs = M.active_intervals(E, 5000.)
    if an != {0: [[58., 250.]], 1: [[60., 150.]]} or bs != {0: [[58., 100.], [200., 250.]], 1: [[60., 150.]]}:
        bad.append(f'active_intervals: {an} / {bs}')
    # ---- 4 end to end
    wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else None
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        if wroot is None:
            wroot = TS.build_world(Path(tmp) / 'world')
        wj = json.loads((wroot / 'world.json').read_text()); day = '20261024'; kp = Path(wj['kp']); src = KP.Syn(); ff = X.syn_features()
        W = L.worlds_of_day(day, kp, 3); _, prs, fair, depth, ff = L.world_inputs(day, W, src, X.syn_inputs, ff)
        worlds = [(w, L.world_bind(day, w, prs, fair, depth, ff)[0]) for w in W]; worlds = [(w, wb) for w, wb in worlds if wb is not None]
        Mst, sst = LT.stubs(-12.), LT.OpenSupport(); OS = type('OS', (), {'check': lambda self, s, h, r: None})()
        alt = {c: {h: _Head(lambda x: (float(x[2]) % 4.) - 1.5) for h in A.HEADS} for c in A.CANDS}
        keepall = {c: {h: _Head(lambda x: 1.) for h in A.HEADS} for c in A.CANDS}
        if not worlds:
            bad.append('end to end: no synthetic world')
        cover = set()
        for w, wb in worlds:
            for name in A.POLICIES:
                B, Lx, pol, cst, rcd = M.episode(name, alt, OS, Mst, sst, day, w, wb, every=7, record=(name == 'RICH'))
                if Lx['status'] != 'ok' or not L.reconcile_arm(Lx, B) or pol.bind_bad or not pol.bind_checked:
                    bad.append(f'end to end: {name} not reconciled / bound'); continue
                dec = [{k_: v_ for k_, v_ in d_.items() if k_ != 'row'} for d_ in pol.log]; mt = M.metrics(B, Lx, dec)
                if abs(mt['value'] - mt['value_bid'] - mt['value_ask']) > 1e-6 or abs(mt['mend'] - mt['value']) > 1e-6:
                    bad.append(f'metrics: {name} value identities')
                if not all(mt[f'best_ms_{s}'] <= mt[f'quote_ms_{s}'] + 1e-9 for s in ('bid', 'ask')) or not (mt['quote_ms_both'] <= min(mt['quote_ms_bid'], mt['quote_ms_ask']) + 1e-9
                                                                                                         and max(mt['quote_ms_bid'], mt['quote_ms_ask']) <= mt['quote_ms_either'] + 1e-9 <= 5000. + 1e-9):
                    bad.append(f'metrics: {name} quote-time identities')
                first = {}
                for d_ in dec:
                    if d_.get('state') == 'live' and d_.get('gid') is not None and d_['gid'] not in first:
                        first[d_['gid']] = d_['action']
                if mt['first_review_pulls'] != sum(1 for a_ in first.values() if a_ == 'PULL') or mt['reviewed_generations'] != len(first):
                    bad.append(f'metrics: {name} first-review pulls are not the first live decision of each generation')
                cover |= {'later-review pull'} if mt['policy_pulls'] > mt['first_review_pulls'] else set()
                if name == 'N0' and (mt['generations'] or mt['quote_ms_either'] or mt['value']):
                    bad.append('metrics: N0 has an order, quote time or value')
                if name == 'ENTRY_PA' and mt['first_review_pulls'] != mt['reviewed_generations']:
                    bad.append('metrics: ENTRY_PA did not pull every reviewed generation at its first review')
                cover |= {f'{name} pulls'} if mt['policy_pulls'] else set(); cover |= {f'{name} fills'} if mt['fills'] else set()
                ch = M.churn(pol.log, B); fr = {}
                for d_ in pol.log:
                    if d_.get('state') == 'live' and d_.get('gid') is not None and d_['gid'] not in fr:
                        fr[d_['gid']] = d_
                posts = {gid: g_ for g_, s_, st_, a_, gid in B.decisions if st_ == 'absent' and a_ == 'POST'}
                if len(ch) != mt['reviewed_generations'] or any(c_['action'] != fr[c_['gid']]['action'] or c_['t_post'] != posts.get(c_['gid']) or not c_['t_review'] > c_['t_post']
                                                                for c_ in ch):
                    bad.append(f"churn: {name} records are not the generations' POST and first live review")
                for c_ in ch:
                    pd_ = [d_ for d_ in pol.log if d_.get('state') == 'absent' and float(d_['t']) == c_['t_post'] and int(d_['side']) == c_['side']]
                    if not pd_:
                        bad.append(f'churn: {name} has no absent decision at a recorded post time'); break
                    rr = fr[c_['gid']].get('row'); rp = pd_[0].get('row')
                    if rp is not None and rr is not None and any(abs(c_[f'd_{f}'] - (float(rr[f]) - float(rp[f]))) > 1e-9 or c_[f'{f}_post'] != float(rp[f])
                                                                 for f in M.CHURN if np.isfinite(float(rp[f])) and np.isfinite(float(rr[f]))):
                        bad.append(f'churn: {name} input changes are not review - post'); break
                    if name == 'RICH' and (c_['value_post'] != pd_[0].get('A') or c_['value_review'] != fr[c_['gid']].get('A')):
                        bad.append("churn: RICH's values are not its decisions' predictions"); break
                    cover |= {'churn input change'} if any(c_[f'd_{f}'] is not None and np.isfinite(c_[f'd_{f}']) and c_[f'd_{f}'] != 0. for f in ('pf_mid', 'fair_edge', 'fp_queue')) else set()
                cover |= {'churn record'} if ch else set()
                if name == 'ENTRY_PA' and any(c_['action'] != 'PULL' for c_ in ch):
                    bad.append('churn: an ENTRY_PA first review that is not a PULL')
                if mt.get('cap_ok') is not True:
                    bad.append(f'metrics: {name} cap_ok is not true on a valid episode')
                if name == 'RICH':
                    ex = M.exposure(pol.log); nr = sum(1 for d_ in pol.log if d_.get('row') is not None)
                    top = sum(v for k_, v in ex.items() if k_.count('|') == 1)
                    per = {dm: sum(v for k_, v in ex.items() if k_.count('|') == 2 and k_.split('|')[2].split('=')[0] == dm) for dm in ('hbin',) + tuple(R2.DIMS)}
                    st_ok = all(sum(v for k_, v in ex.items() if k_.count('|') == 1 and k_.split('|')[1] == st_) == sum(1 for d_ in pol.log if d_['status'] == st_)
                                for st_ in {d_['status'] for d_ in pol.log})
                    if top != len(pol.log) or any(v != nr for v in per.values()) or not st_ok or not nr:
                        bad.append(f'exposure: RICH counts ({top} / {len(pol.log)}, {per}, {nr})')
                    if len(rcd.rec) != len(pol.log):
                        bad.append('recorder: not one entry per callback')
                    elig = [r for r in rcd.rec if r['state'] in ('absent', 'live')]
                    probe = [r for s in (0, 1) for r in (lambda e: e[:1] + e[-1:])([r for r in elig if r['side'] == s])]
                    for r in probe:
                        ctx = L.bound_fork(wb['path'], A.factory('RICH', alt, OS, Mst, sst, wb), r['k'], r['side'], R2.episode_ident(day, w['o'][r['side']]))
                        if ctx['fingerprint'] != r['fingerprint']:
                            bad.append(f"fork rebuild: RICH's roll-in does not reproduce the recorded state at k {r['k']} side {r['side']}")
                    late = [r for r in elig if r['k'] >= 40]
                    if late:
                        r = late[-1]; ctx = L.bound_fork(wb['path'], A.factory('PI0', alt, OS, Mst, sst, wb), r['k'], r['side'], R2.episode_ident(day, w['o'][r['side']]))
                        cover |= {'pi0 roll-in differs'} if ctx['fingerprint'] != r['fingerprint'] else set()
                    sel_ = M.select_reached(rcd.rec, day, w['m']); pr = M.reached_pairs(sel_, day, w, wb, alt, OS, Mst, sst, alt)
                    if sorted((x['key']['side'], x['key']['head']) for x in pr) != sorted(A.HEADS):
                        bad.append('reached: not one entry per side / head')
                    for x in pr:
                        sk = (x['key']['side'], x['key']['head'])
                        if sk not in sel_:
                            if x['status'] != 'no_eligible_decision' or x['eligible'] != 0:
                                bad.append('reached: a side / head without a selection is not no_eligible_decision')
                            continue
                        n_el = sum(1 for r in rcd.rec if r['side'] == sk[0] and r['status'] == 'model' and r['state'] == ('absent' if sk[1] == 'post' else 'live'))
                        if x['status'] not in ('ok', 'missing_mark') or x['eligible'] != n_el or (x['status'] == 'ok' and not (x['reconciled'] and x['binding_bad'] == 0
                                                                                                and x['binding_checked'] > 0 and x['fork_fingerprint'] == sel_[sk]['dec']['fingerprint'])):
                            bad.append(f"reached: a pair not ok / reconciled / bound / counted ({x['status']})")
                        cover |= {f"reached {x['key']['head']}"} if x['status'] == 'ok' else set()
                    if sel_:
                        k0 = sorted(sel_)[0]; tam = {k0: dict(sel_[k0], dec=dict(sel_[k0]['dec'], fingerprint='0' * 64))}
                        if [x['status'] for x in M.reached_pairs(tam, day, w, wb, alt, OS, Mst, sst, alt) if (x['key']['side'], x['key']['head']) == k0] != ['fork_mismatch']:
                            bad.append('reached: a tampered recorded fingerprint was not refused')
                    e0 = M.reached_pairs({}, day, w, wb, alt, OS, Mst, sst, alt)
                    if sorted((x['key']['side'], x['key']['head'], x['status'], x['eligible']) for x in e0) != sorted((s, h, 'no_eligible_decision', 0) for s, h in A.HEADS):
                        bad.append('reached: an empty selection is not four no_eligible_decision entries')
            B, Lx, pol, _, _ = M.episode('RICH', keepall, OS, Mst, sst, day, w, wb)
            mt = M.metrics(B, Lx, [{k_: v_ for k_, v_ in d_.items() if k_ != 'row'} for d_ in pol.log])
            if mt['first_review_pulls'] != 0 or mt['policy_pulls'] != 0:
                bad.append('metrics: a keep-always RICH pulled')
        if (M.PER_HOUR, M.PART, M.AUDIT_EVERY, M.DOMAIN) != (8, 32, 35, 'adv2_rich_v1_reached_20261121'):
            bad.append('constants: the declared population / audit / domain changed')
        for need in ('ENTRY_PA pulls', 'RICH pulls', 'reached post', 'pi0 roll-in differs', 'later-review pull', 'churn record', 'churn input change'):
            if need not in cover:
                bad.append(f'end to end: the synthetic run never covered "{need}"')
    # ---- 5 the refusal
    sc = M.AE.committed
    try:
        M.AE.committed = lambda rel, local: False
        with tempfile.TemporaryDirectory() as td:
            if not _refused(lambda: M.run_part(0, out=Path(td) / 'r'), 'not committed'):
                bad.append('run_part: ran without a committed fit manifest')
    finally:
        M.AE.committed = sc
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 replay selftest passed: population (rank, nesting, partitions); selection (minimum rank, model decisions only, outcome-free); '
          'intervals and active / at-best time; end to end (every policy reconciled and bound; value / markout / quote identities; N0, ENTRY_PA, '
          'keep-always; fork rebuild = the recorded state, PI0 roll-in differs; reached pairs ok, tampered fingerprint refused; churn records; '
          'cap_ok; RICH exposure counts); refusal '
          'without the committed fit')
