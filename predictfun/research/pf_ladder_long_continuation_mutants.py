"""MUTATION CHECK of pf_ladder_long_continuation (LONG-CONTINUATION-PAIRED-VALUE-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_long_continuation_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once.
Not planted (equivalent on every fixture, or exercised on saved data only and recomputed by lc_checks.py): the census (saved command arrays),
the pre / post split at exactly the fork time (continuous synthetic trade times), the context-equality and pre-fork-identity gate
expressions (always true on a correct fork), the bookkeeping-only signal() call of the forced branch, the horizon markouts, the
time-weighted |I| tail term, and the day runner's write-once and memory self-guard (run_part's checks before them refuse first)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_long_continuation.py'
SELFTEST = 'pf_ladder_long_continuation_selftest'
FAIL = 'LONG CONTINUATION SELFTEST FAILED'
MUTANTS = [
    ('the forced action at every later tick', "if f['kind'] == 'REVIEW' and t == f['t'] and int(v['side']) == f['side'] and int(o['gid']) == f['gid'] and not self.applied:",
     "if f['kind'] == 'REVIEW' and t >= f['t'] and int(v['side']) == f['side'] and int(o['gid']) == f['gid']:"),
    ('no exemption', "        return bool(f['exempt'] and side == f['side'] and ((f['kind'] == 'REVIEW' and int(o['gid']) == f['gid'])", "        return False and bool(f['exempt'] and side == f['side'] and ((f['kind'] == 'REVIEW' and int(o['gid']) == f['gid'])"),
    ('the exemption on the opposite side', "        return bool(f['exempt'] and side == f['side'] and", "        return bool(f['exempt'] and side != f['side'] and"),
    ('the focal gid outside the lineage', "((f['kind'] == 'REVIEW' and int(o['gid']) == f['gid'])\n", "((False and int(o['gid']) == f['gid'])\n"),
    ('the lineage by another rung', "or (float(o['rung']) == f['rung'] and float(o['submit']) >= f['t'])))", "or (float(o['rung']) != f['rung'] and float(o['submit']) >= f['t'])))"),
    ('every KEEP counted as suppressed', "            if a == 'KEEP' and self.signal(v)[0]:", "            if a == 'KEEP':"),
    ('a forced KEEP with a new reason', "return f['action'], ('keep' if f['action'] == 'KEEP' else 'forced_pull')", "return f['action'], ('forced_keep' if f['action'] == 'KEEP' else 'forced_pull')"),
    ('a forced WAIT that posts', "            return 'WAIT', None, 'forced_wait'", "            return a, q, why"),
    ('the forced slot at another rung', "and int(v['side']) == f['side'] and float(r) == f['rung'] and not self.applied:", "and int(v['side']) == f['side'] and float(r) != f['rung'] and not self.applied:"),
    ('the carried mark one second late', "cash0 = -sum(SG[x[2]] * x[3] * x[4] for x in pre); M0 = mid(f, g)", "cash0 = -sum(SG[x[2]] * x[3] * x[4] for x in pre); M0 = mid(f, g + 1000.)"),
    ('the carried inventory double-counted', "post_c = sum(c(x) for x in post); Q = post_c + (I0 * (mark - M0) if M0 is not None else np.nan)",
     "post_c = sum(c(x) for x in post); Q = post_c + (I0 * mark if M0 is not None else np.nan)"),
    ('the reconciliation sign', "return bool(M0 is None or abs(value - (cash0 + I0 * M0) - Q) <= 1e-6 * max(1., abs(value)))", "return bool(M0 is None or abs(value - (cash0 - I0 * M0) - Q) <= 1e-6 * max(1., abs(value)))"),
    ('the activation frame from the left', "j = max(0, int(np.searchsorted(f[:, 0], act, side='right')) - 1); p = 2 + side * 10", "j = max(0, int(np.searchsorted(f[:, 0], act, side='left')) - 1); p = 2 + side * 10"),
    ('beyond the shown levels not zero', "        return 0., j\n", "        return 5., j\n"),
    ('the queue reset without a tolerance', "q_ok = all(r['refused'] or r['ahead0'] is None or abs(r['ahead0'] - r['shown']) <= 1e-9 for r in reps)",
     "q_ok = all(r['refused'] or r['ahead0'] is None or abs(r['ahead0'] - r['shown']) <= 1e9 for r in reps)"),
    ('FIFO matching the same side', "        while q > 1e-12 and lots and lots[0][0] == -d:", "        while q > 1e-12 and lots and lots[0][0] == d:"),
    ('FIFO holding from zero', "x = min(q, lots[0][1]); matched += x; held += x * (t - lots[0][2])", "x = min(q, lots[0][1]); matched += x; held += x * t"),
    ('both sides as either side', "        tot['both'] += (b - a) * (on[0] and on[1])", "        tot['both'] += (b - a) * (on[0] or on[1])"),
    ('an early gain as an early loss', "return ('early_loss' if early < -TOL else 'no_early_loss')", "return ('early_loss' if early > TOL else 'no_early_loss')"),
    ('the early group counted later', "later = sum(c(x) for x in post if x[1] in lin and x[0] > t1)", "later = sum(c(x) for x in post if x[1] in lin and x[0] >= t1)"),
    ('a tie as retain_better', "econ = 'retain_better' if Y > TOL else ('alternative_better' if Y < -TOL else 'tie')", "econ = 'retain_better' if Y > 0. else ('alternative_better' if Y < -TOL else 'tie')"),
    ('the mechanism from the alternative arm', "mech = ('retention_beneficial_with_early_loss' if R_['seq'].startswith('early_loss')", "mech = ('retention_beneficial_with_early_loss' if A_['seq'].startswith('early_loss')"),
    ('Y as a sum', "Y = R_['Q'] - A_['Q'] if (R_['M_fork'] is not None) else R_['post_c'] - A_['post_c']", "Y = R_['Q'] + A_['Q'] if (R_['M_fork'] is not None) else R_['post_c'] - A_['post_c']"),
    ('Y_focal as a sum', "Y_focal=R_['focal_c'] - A_['focal_c']", "Y_focal=R_['focal_c'] + A_['focal_c']"),
    ('the lineage replacement at the fork tick dropped', "(o['rung'] == st['rung'] and o['submit'] >= g)))}", "(o['rung'] == st['rung'] and o['submit'] > g)))}"),
    ('the fills list shared between arms', "memo = {id(B.E.f): B.E.f, id(B.E.tr): B.E.tr, id(pol.cache): pol.cache, id(pol.edges): pol.edges}",
     "memo = {id(B.E.f): B.E.f, id(B.E.tr): B.E.tr, id(pol.cache): pol.cache, id(pol.edges): pol.edges, id(B.E.fills): B.E.fills}"),
    ('the unused-exemption gate dropped', "            pr['status'] = pr['status'] if pr['gate_unused_exemption'] else 'gate_failed'", "            pr['gate_unused_exemption'] = True"),
    ('no gate on lineage pulls', "no_lineage_pull=bool(R_['lineage_pressure_pulls'] == 0 and A_['lineage_pressure_pulls'] == 0))", "no_lineage_pull=True)"),
    ('no gate on the queue reset', "queue_reset=bool(R_['queue_reset_ok'] and A_['queue_reset_ok'])", "queue_reset=True"),
    ('the base parity without the guard log', "par.update(guard_log=[list(x) for x in pol.log] == rec['guard_log']); par = dict(", "par.update(guard_log=False); par = dict("),
    ('a material threshold too high', "VARIED_SHARE, VARIED_ABS_C = 0.10, 1000.", "VARIED_SHARE, VARIED_ABS_C = 0.10, 100000."),
    ('a gate failure still reliable', "reliable = bool(allr.status_callback_not_reached == 0 and allr.status_gate_failed == 0)", "reliable = bool(allr.status_callback_not_reached == 0)"),
    ('the rest without the projection', "need(part == 'subset' or (pp.exists() and json.loads(pp.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')",
     "need(True, 'the rest needs the write-once projection with run_rest true')"),
    ('measure without the freeze', "        need(frozen(), 'the continuation builder / selftest / declaration / manifest / launcher / checker are not the committed ones')\n    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); od = Path(out or OUT)",
     "        pass\n    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); od = Path(out or OUT)"),
    ('build_manifest without write-once', "need(not path.exists() and not cpath.exists(), 'the manifest / census exists (write-once)')", "need(True, 'the manifest / census exists (write-once)')"),
]
