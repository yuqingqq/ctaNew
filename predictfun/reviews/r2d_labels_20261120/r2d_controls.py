"""The R2-D real-row controls, declared 2026-11-20 before the bounded build's outputs were read (review 8cc1dc49's next priority):
 (i) stale binding on real rows: for the first selected pair of each day of build A's manifest, the pair is re-run with ONE binding-checked
     teacher bound to a copy of the fork state for both arms; the binding check must report failures (> 0). The instrument fires on real
     data;
 (ii) reversal on real rows: for the first 2 selected pairs of each day, the arms executed in reverse order must equal build A's recorded
     result (pf_r2_labels.same_pair), compared like with like: the reversed result passes through the record writer's own JSON
     serialization first. v1 of this script compared a live dict with its JSON round trip, where the integer keys of open_at_end become
     strings, so every pair compared unequal (controls_v1.json, kept). Falsifier: the recorded result with Y + 1 must compare unequal;
 (iii) the single branch on consumed Oct 23: the binding-checked teacher on the first 4 episodes of the R2-C rollout population must
     reproduce the frozen rollout's decisions (status and action, in order). The R2-C decisions.csv is sliced into world blocks by
     worlds.csv's logged counts.
Writes controls.json beside this script; exit 0 only when every control holds. Usage: r2d_controls.py <build A directory>"""
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_branch_engine as BE                             # noqa: E402
import pf_r2_labels as L                                  # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher as T                                 # noqa: E402

HERE = Path(__file__).resolve().parent; A = Path(sys.argv[1]); C, pd, KP, X, QP = T.C, T.pd, T.KP, T.X, T.QP
norm = lambda x: json.loads(json.dumps(x, sort_keys=True, default=str))                  # the record writer's own serialization
man = json.loads((A / 'manifest.json').read_text()); msha = C.sha(A / 'manifest.json'); M, sup, fz = L.teacher_assets()
kp = Path(json.loads(Path(T.REAL['population']).read_text())['inputs']['kp_dir']); res = dict(stale=[], reversal=[], single=[])
for day in man['days']:
    sel = [r for r in man['selected'] if r['day'] == day]; W = L.worlds_of_day(day, kp, man['prefix'])
    src, prs, fair, depth, ff = L.world_inputs(day, W); bound = {}
    for j, r in enumerate(sel[:2]):
        wi = int(r['world']); bound.setdefault(wi, L.world_bind(day, W[wi], prs, fair, depth, ff)[0]); world = bound[wi]
        ctx = L.fork_for(r, world, M, sup); C.require(ctx['fingerprint'] == r['fork_fingerprint'], 'a control fork differs from the manifest')
        rec = json.loads((A / 'pairs' / L.rname(r)).read_text())['result']
        rv = norm(L.pair_bound(ctx, L.teacher_factory(M, sup, world), r['kind'], reverse=True))
        fals = not L.same_pair(dict(rec, Y=rec['Y'] + 1.), rec) if rec.get('status') == 'ok' else True
        res['reversal'].append(dict(day=day, key=L.rkey(r), equal=bool(L.same_pair(rv, rec) if rec.get('status') == 'ok' else rv.get('status') == rec.get('status')),
                                    falsifier_unequal=bool(fals)))
        if j == 0:
            stale = L.ArmTeacher(M, sup); stale.bind(copy.deepcopy(ctx['state']), world)
            rs = L.pair_bound(ctx, lambda B: stale, r['kind'])
            res['stale'].append(dict(day=day, key=L.rkey(r), status=rs.get('status'), binding_bad=stale.bind_bad, binding_checked=stale.bind_checked))
    del prs, fair, depth, bound, src, W, sel; gc.collect()                                  # one day's inputs at a time
D = pd.read_csv(T.REAL['out'] / 'rollout' / 'decisions.csv'); Wc = pd.read_csv(T.REAL['out'] / 'rollout' / 'worlds.csv')
worlds, _ = T.worlds_of(T.REAL, kp); src = KP.Tape(); mk = sorted({w['m'] for w in worlds[:6]}); fair, depth, _, _ = X.real_inputs(T.DAY, mk)
data = src.load(T.DAY, mk, src.offset(T.DAY)); pairs = {int(P_[0]['m']): P_ for P_ in src.pairs(data)}; at = 0; done = 0
for i, (w, wr) in enumerate(zip(worlds, Wc.itertuples())):
    n_ = int(wr.logged) if wr.status == 'ok' else 0; block = D.iloc[at:at + n_]; at += n_
    if wr.status != 'ok' or done >= 4:
        continue
    world, st = L.world_bind(T.DAY, w, pairs, fair, depth, X.real_features); C.require(world is not None, f'an R2-C world without its episode ({st})')
    t_ = L.ArmTeacher(M, sup); B = BE.Branch(world['path'], 0, ident=R2.episode_ident(T.DAY, w['o'][0])); t_.bind(B, world)
    for k in range(L.NT):
        BE.step(B, k, t_)
    got = [(float(d_['t']), int(d_['side']), d_['status'], d_['action']) for d_ in t_.log]
    want = [(float(x.t), int(x.side), x.status, x.action) for x in block.itertuples()]
    res['single'].append(dict(world=i, m=w['m'], t=w['t'], decisions=len(got), equal=got == want, binding_bad=t_.bind_bad)); done += 1
ok = dict(stale=bool(len(res['stale']) == len(man['days']) and all(x['status'] == 'ok' and x['binding_bad'] > 0 for x in res['stale'])),
          reversal=bool(len(res['reversal']) == 2 * len(man['days']) and all(x['equal'] and x['falsifier_unequal'] for x in res['reversal'])),
          single=bool(len(res['single']) == 4 and all(x['equal'] and x['binding_bad'] == 0 for x in res['single'])))
out = dict(asof=C.stamp(), manifest_sha256=msha, controls=res, holds=ok, all_true=all(ok.values()))
(HERE / 'controls.json').write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(dict(holds=ok, all_true=all(ok.values())))); sys.exit(0 if all(ok.values()) else 1)
