"""ENTRY-EXPOSURE-v1 diagnosis of check C's failure (2026-11-21, session aea5bfb9): the KEEP wrapper's `overrides` counter is incremented in
decide(), which the causal probe also calls on its hidden / delivered copies; the decision log records only the branch's own callbacks.
On one failing fork (argv: day m t_ms side), the KEEP arm is rerun from the same saved fork with the probe OFF and ON. Predicted: OFF the
counter equals the logged overrides and the decisions / fills / value equal the saved KEEP arm (the probe never changes the branch); ON the
counter reproduces the saved inflated counter and the same branch. Uses the frozen module's own functions; writes a JSON line."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_entry_exposure as M                            # noqa: E402

day, m, t_ms, side = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out = M.OUT; man = json.loads((out / 'manifest.json').read_text()); kind = 'train' if day in ('20261024', '20261025') else 'eval'
f = next(x for x in man['sets'][kind]['forks'] if (x['day'], x['m'], x['t_ms'], x['side']) == (day, m, t_ms, side))
k = next(i for i, pt in enumerate(man['parts']) if any((w['day'], w['m'], w['t_ms']) == (day, m, t_ms) for w in pt['worlds']))
saved = next(r for r in M.LS.gz_rows((out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()) if (r['day'], r['m'], r['t_ms'], r['side']) == (day, m, t_ms, side))
plan = json.loads((Path(M.PLANS[kind]['dir']) / 'plan.json').read_text()); kp = Path(plan['kp_dir']); src = M.KP.Tape(); ff = M.files_fn(kind)
M.C.require(M.LS.identity_now(day, kp, src, ff, M.LS.hours_tracked) == plan['sample'][day]['inputs'], 'inputs differ from the plan')
aw = M.LS.worlds_all(day, kp, keys=[(m, t_ms)])[0]; fair, depth, hrs, trk = M.X.real_inputs(day, [m])
M.C.require(M.LS.day_identity(day, kp, src, hrs, trk, ff) == plan['sample'][day]['inputs'], 'real_inputs differ from the plan')
data = src.load(day, [m], src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None
world, wst = M.L.world_bind(day, dict(m=m, t=aw['t'], o=aw['o']), prs, fair, depth, M.X.real_features)
Mo, sup, _ = M.L.teacher_assets(); ctx = M.L.fork_for(f['sel'], world, Mo, sup); M.C.require(ctx['fingerprint'] == f['fork_fingerprint'], 'fork fingerprint')
sk = saved['arms']['POST_KEEP_INITIAL']; res = {}
for name, every in (('probe_off', None), ('probe_on', M.AUDIT_EVERY)):
    def inner(B):
        t_ = M.L.ArmTeacher(Mo, sup); t_.bind(B, world); return t_
    t0 = time.time(); kf = M.keep_factory(inner, side, every=every); Bk, Lk, pk = M.run_arm(ctx, kf, keep=True)
    n_log = sum(1 for d in pk.log if d.get('status') == 'keep_initial_override')
    res[name] = dict(counter=pk.overrides, logged=n_log, fills_equal_saved=M.L._h(Lk['fills']) == sk['fills_sha256'], decisions_equal_saved=M.L._h(Lk['decisions']) == sk['decisions_sha256'],
                     value_equal_saved=float(Lk['value']) == float(sk['value']), seconds=round(time.time() - t0, 2))
pred = dict(off_counter_equals_log=res['probe_off']['counter'] == res['probe_off']['logged'] == len(saved['shadow']),
            on_counter_equals_saved=res['probe_on']['counter'] == int(saved['overrides']),
            branch_unchanged_by_probe=all(res[n][k_] for n in res for k_ in ('fills_equal_saved', 'decisions_equal_saved', 'value_equal_saved')))
print(json.dumps(dict(fork=[day, m, t_ms, side], saved_counter=int(saved['overrides']), saved_logged=len(saved['shadow']), runs=res, predicted=pred,
                      confirmed=all(pred.values()), asof=M.C.stamp())))
