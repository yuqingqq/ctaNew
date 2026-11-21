"""SELFTEST of pf_adv2_entry_replay (the ADV2-ENTRY-SMOOTH-v1 bounded own-state comparison and reached entry states; 2026-11-21, session
aea5bfb9). run(M) checks the module object M (the file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic:
 1. select_entry: per side the minimum rank under the declared domain among absent model decisions with a finite prediction; a fallback, a
    live state and a nonfinite prediction are never chosen; the choice ignores the predictions; the inclusion count is the eligible number;
 2. end to end on the real engine and the cached synthetic world (stub heads that change sign with an input, the stub teacher), per policy:
    the episode reconciles and is bound; its retained trace passes pf_r2_label_sample._ledger_bad and _arm_calls_bad (the required callbacks
    recomputed from the decision log equal the calls and binding checks) and the probe ran at its declared rate; the trace's hashes are the
    ledger's;
 3. the reproduction control: each control policy's episode equals the one pf_adv2.make_policy builds (R3's construction) by value and hashes
    (r3_match true), and a perturbed reference is false;
 4. reached entry states: every selected pair is ok, its row passes pf_r2_label_sample._pair_bad with no failure (the shared invariants: END,
    mark, fill times, context, Y, ledgers, pre-fork fills, fingerprint, required callbacks, audit rate, causal fork inputs), its fork inputs
    equal the recorded decision's, both heads predicted on those inputs; a tampered recorded fingerprint is fork_mismatch; an empty selection
    is two no_eligible_decision rows;
 5. part identity: part_record_ok is true for this round's record and false for other members, sources, fit or bytes; frames refuses a part
    whose rows are not exactly its members; run_part refuses before any outcome without the committed fit manifest."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ADV2 ENTRY REPLAY SELFTEST FAILED'


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


def run(M):  # noqa: C901
    bad = []; A, N, L, BE, R2, T, LS = M.A, M.N, M.L, M.BE, M.R2, M.T, M.LS
    import pf_r2_labels_selftest as LT
    import pf_r2_teacher_selftest as TS
    I, KP, X = R2.I, R2.KP, R2.X
    # ---- 1 select_entry
    rec = [dict(k=k, t=50. * k, side=s, state=st, generation=1, fingerprint='f', status=stt, action='POST', A=a_, row={}, oid=7 + s)
           for k in range(20) for s in (0, 1) for st, stt, a_ in (('absent', 'model', float(k)), ('live', 'model', 1.), ('absent', 'outside_training_range', 1.),
                                                                  ('absent', 'model', float('nan')))]
    s1 = M.select_entry(rec, '20261026', 5); s2 = M.select_entry([dict(r, A=-r['A'] - 3.) for r in rec], '20261026', 5)
    if set(s1) != {0, 1} or {k: v['key'] for k, v in s1.items()} != {k: v['key'] for k, v in s2.items()} or any(v['eligible'] != 20 for v in s1.values()):
        bad.append('select_entry: not one per side, prediction-dependent, or the inclusion count is not the eligible number')
    for s, v in s1.items():
        keys = [dict(day='20261026', instrument_id=5, original_episode_id=7 + s, side=s, head='post', generation=1, decision_ms=50 * k) for k in range(20)]
        if v['key'] != min(keys, key=lambda k_: (hashlib.sha256((M.DOMAIN + '|' + A.ck(k_)).encode()).hexdigest(), A.ck(k_))) or v['dec']['state'] != 'absent':
            bad.append(f'select_entry: side {s} is not the minimum-rank absent model decision')
    if M.DOMAIN != 'adv2_entry_smooth_v1_reached_20261121' or M.AUDIT_EVERY != 35 or M.CONTROLS != ('N0', 'PI0', 'ENTRY_ONLY'):
        bad.append('constants: the declared domain / audit rate / controls changed')
    # ---- 2-4 end to end
    wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else None
    cover = set()
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        if wroot is None:
            wroot = TS.build_world(Path(tmp) / 'world')
        wj = json.loads((wroot / 'world.json').read_text()); day = '20261024'; kp = Path(wj['kp']); src = KP.Syn(); ff = X.syn_features()
        W = L.worlds_of_day(day, kp, 3); _, prs, fair, depth, ff = L.world_inputs(day, W, src, X.syn_inputs, ff)
        worlds = [(w, L.world_bind(day, w, prs, fair, depth, ff)[0]) for w in W]; worlds = [(w, wb) for w, wb in worlds if wb is not None]
        Mst, sst = LT.stubs(-12.), LT.OpenSupport(); OS = type('OS', (), {'check': lambda self, s, h, r: None})()
        alt = {h: _Head(lambda x: (float(x[2]) % 4.) - 1.5) for h in N.HEADS}
        rich = {'RICH': {h: _Head(lambda x: (float(x[2]) % 3.) - 1.) for h in A.HEADS}}
        if not worlds:
            bad.append('end to end: no synthetic world')
        for w, wb in worlds:
            base = dict(day=day, m=int(w['m']), t_ms=float(w['t']), hour=0, rank='r', pi=.5, oid0=int(w['o'][0].oid), oid1=int(w['o'][1].oid))
            sel = {}
            for name in N.POLICIES:
                try:
                    B, Lx, pol, cst, rcd = M.episode(name, alt, rich, OS, Mst, sst, day, w, wb, every=7, record=(name == N.NAME))
                except Exception as e:                                                      # noqa: BLE001
                    bad.append(f'end to end: the {name} episode stopped ({e!r})'[:200]); continue
                if Lx['status'] != 'ok' or not L.reconcile_arm(Lx, B) or pol.bind_bad or not pol.bind_checked:
                    bad.append(f'end to end: {name} not reconciled / bound'); continue
                tr = M.trace(Lx, B, pol, cst)
                lb, cb = LS._ledger_bad(tr), LS._arm_calls_bad(tr, -1., 0, 0)
                if lb or cb or tr['causal'].get('checked') != -(-tr['teacher']['calls'] // 7) or tr['fills_sha256'] != L._h(Lx['fills']) or tr['decisions_sha256'] != L._h(Lx['decisions']):
                    bad.append(f'trace: {name} ({lb}; {cb}; probe {tr["causal"].get("checked")} of {tr["teacher"]["calls"]} calls)')
                cover |= {'fills in a trace'} if tr['fill_list'] else set(); cover |= {f'{name} posts'} if Lx.get('generations') else set()
                if name in M.CONTROLS:
                    try:
                        probe, _ = T.causal_probe(7); p2 = A.make_policy(name, rich, OS, Mst, sst, causal=probe); B2 = BE.Branch(wb['path'], 0, ident=R2.episode_ident(day, w['o'][0]))
                        p2.bind(B2, wb)
                        for k in range(L.NT):
                            BE.step(B2, k, p2)
                        L2 = BE.finish(B2); ref = dict(value=float(L2['value']), decisions_sha256=L._h(L2['decisions']), fills_sha256=L._h(L2['fills']))
                        e = dict(value=float(Lx['value']), trace=tr)
                        if not M.r3_match(e, ref) or M.r3_match(e, dict(ref, value=ref['value'] + 1e-9)) or M.r3_match(e, dict(ref, fills_sha256='0')) or M.r3_match(e, None):
                            bad.append(f'reproduction: {name} does not match its pf_adv2 construction, or a perturbed reference matched')
                    except Exception as e_:                                                 # noqa: BLE001
                        bad.append(f'reproduction: {name} stopped ({e_!r})'[:200])
                if name == N.NAME:
                    sel = M.select_entry(rcd.rec, day, w['m'])
            try:
                pr = M.reached_entry(sel, day, dict(m=w['m'], t=w['t'], o=w['o']), wb, alt, rich, OS, Mst, sst, base)
            except Exception as e:                                                          # noqa: BLE001
                bad.append(f'reached: reached_entry stopped ({e!r})'[:200]); pr = []
            if sorted(x['side'] for x in pr) != [0, 1]:
                bad.append('reached: not one row per side')
            for x in pr:
                if x['side'] not in sel:
                    if x['status'] != 'no_eligible_decision':
                        bad.append('reached: a side without a selection is not no_eligible_decision')
                    continue
                if x['status'] != 'ok':
                    bad.append(f"reached: a pair is {x['status']}"); continue
                plan = dict(settings=x['result']['settings'], audit_every=M.AUDIT_EVERY, feature_schema=dict(n=len(T.FEATS)))
                pb = LS._pair_bad(plan, day, base, x)
                xr = np.asarray([x['fork']['inputs']], np.float32)
                why = (pb[:3] + (['inputs not the recorded ones'] if x['inputs_equal_recorded'] is not True else [])
                       + (['the SMOOTH prediction'] if x['pred'][N.NAME] != float(N.predict(alt[(x['side'], 'post')], xr)[0]) else [])
                       + (['the RICH prediction'] if x['pred']['RICH'] != float(A.predict(rich['RICH'][(x['side'], 'post')], 'RICH', xr)[0]) else [])
                       + (['the inclusion count'] if x['eligible'] != sel[x['side']]['eligible'] else []))
                if why:
                    bad.append(f'reached: a pair fails {why}')
                cover |= {'reached ok'}; cover |= {'reached nonzero'} if x['result']['Y'] != 0 else set()
            if sel:
                s0 = sorted(sel)[0]; tam = {s0: dict(sel[s0], dec=dict(sel[s0]['dec'], fingerprint='0' * 64))}
                if [x['status'] for x in M.reached_entry(tam, day, dict(m=w['m'], t=w['t'], o=w['o']), wb, alt, rich, OS, Mst, sst, base) if x['side'] == s0] != ['fork_mismatch']:
                    bad.append('reached: a tampered recorded fingerprint was not refused')
            if sel:
                s0 = sorted(sel)[0]; row2 = dict(sel[s0]['dec']['row']); f0 = T.FEATS[0]; row2[f0] = np.float32(float(row2[f0]) + 1.)
                mis = {s0: dict(sel[s0], dec=dict(sel[s0]['dec'], row=row2))}
                if [x['status'] for x in M.reached_entry(mis, day, dict(m=w['m'], t=w['t'], o=w['o']), wb, alt, rich, OS, Mst, sst, base) if x['side'] == s0] != ['input_mismatch']:
                    bad.append('reached: recorded inputs that differ from the rebuilt fork inputs were not input_mismatch')
            e0 = M.reached_entry({}, day, dict(m=w['m'], t=w['t'], o=w['o']), wb, alt, rich, OS, Mst, sst, base)
            if sorted((x['side'], x['status']) for x in e0) != [(0, 'no_eligible_decision'), (1, 'no_eligible_decision')]:
                bad.append('reached: an empty selection is not two no_eligible_decision rows')
    for need in ('SMOOTH posts', 'ENTRY_ONLY posts', 'reached ok', 'fills in a trace'):
        if need not in cover:
            bad.append(f'end to end: the synthetic run never covered "{need}"')
    # ---- 5 part identity
    members = [dict(day='20261026', m=1, t_ms=1000), dict(day='20261026', m=2, t_ms=2000)]; b = b'part bytes'; fz = 'f' * 64
    good = dict(members=[LS.wkey(x) for x in members], sources=M.sources(), fit_sha256=fz, eval_plan_sha256=M.AE.EVAL_PLAN_SHA, sha256=hashlib.sha256(b).hexdigest())
    if not M.part_record_ok(good, members, b, fz) or M.part_record_ok(good, members, b, None):
        bad.append("part identity: this round's record was refused, or a record was accepted before any fit")
    for ch in (dict(members=good['members'][:1]), dict(sources=dict(good['sources'], **{'pf_adv2.py': '0'})), dict(fit_sha256='0' * 64), dict(sha256='0' * 64)):
        if M.part_record_ok(dict(good, **ch), members, b, fz):
            bad.append(f'part identity: a record with another {sorted(ch)[0]} was accepted')
    with tempfile.TemporaryDirectory() as td:
        out = Path(td); (out / 'parts').mkdir(); rows = [dict(day='20261026', m=1, t_ms=1000, status='world:x')]; bb = LS.gz_bytes(rows)
        (out / 'parts' / 'part_000.jsonl.gz').write_bytes(bb); (out / 'parts' / 'part_000.json').write_text(json.dumps(dict(good, sha256=hashlib.sha256(bb).hexdigest())))
        sv = M.fit_sha
        try:
            M.fit_sha = lambda: fz
            if not _refused(lambda: M.frames(out, [('20261026', members)]), 'not exactly its members'):
                bad.append('frames: a part whose rows are not exactly its members was not refused')
        finally:
            M.fit_sha = sv
    sc, sr = M.N.committed_equal, M.r3_reference

    def sentinel(k):                                                                        # never a real partition from a selftest
        raise RuntimeError('sentinel: no real partition in a selftest')
    try:
        M.N.committed_equal = lambda rel, local: False; M.r3_reference = sentinel
        with tempfile.TemporaryDirectory() as td:
            if not _refused(lambda: M.run_part(0, out=Path(td) / 'r'), 'not committed'):
                bad.append('run_part: ran without a committed fit manifest')
            (Path(td) / 'x' / 'parts').mkdir(parents=True); (Path(td) / 'x' / 'parts' / 'part_000.json').write_text(json.dumps(dict(good, members=[])))
            (Path(td) / 'x' / 'parts' / 'part_000.jsonl.gz').write_bytes(b'x')
            if not _refused(lambda: M.run_part(0, out=Path(td) / 'x'), 'another identity'):
                bad.append('run_part: an existing record of another identity was accepted')
    finally:
        M.N.committed_equal, M.r3_reference = sc, sr
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 entry replay selftest passed: select_entry (minimum rank, absent model decisions only, outcome-free, inclusion counts); end to '
          'end (every policy reconciled, bound, its trace passing the shared ledger / callback / audit-rate checks); the reproduction control '
          '(equal to the pf_adv2 construction, perturbed references refused); reached entry pairs (the shared pair invariants, recorded inputs, '
          'both heads; tampered fingerprint; empty selection); part identity, membership and the run refusal')
