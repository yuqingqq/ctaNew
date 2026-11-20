"""MUTATION CHECK of the r2_label_sample_v2 runner's selftest (review 4393ad19's next priority; 2026-11-20, session aea5bfb9). Each planted error is
a textual replacement in a copy of pf_r2_label_sample.py; the file on disk is never touched. The copy is executed as the module under its real
path, in a fresh process, and its selftest (pf_r2_label_sample_selftest, synthetic only, fail fast) is run.
Required:
 - the unchanged copy passes;
 - every mutant is caught BY A CHECK: the selftest's own failure message. A crash or a timeout is not a catch;
 - every target occurs exactly once.
Guards over one property mask each other's mutants (FINDINGS P80). The selftest isolates them:
 - the validator's corruptions run without a record, so the record counts cannot mask the row checks;
 - each ledger corruption keeps every other identity consistent: R with Y, a self-consistent cash / position change, a zero fill, a
   consistent pre-fork fill, a consistent overfill;
 - the high-water is checked at each stage by a staged reading (loading, census, world);
 - the byte pre-check and the validator's byte check refuse with different messages;
 - identity refusals are matched by message, and changed inputs refuse before any day input is read.
Not mutated (equivalent on intact data, documented): the arm-to-teacher mapping hash check (an inversion is equivalent when both arms' fills
coincide); the fork-state copy before the inputs (the builders do not mutate the state); the running worst-pair maximum (the pair that
exceeds the limit latches at its own check).
Usage: pf_r2_label_sample_mutants.py [k/n] (shard k of n; the unchanged control runs in every shard). Per-mutant jobs (one process each):
pf_mutant_queue.sh with pf_mutant_one.py / pf_mutant_summary.py, which import MUTANTS, SRC, SELFTEST and FAIL from here."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_r2_label_sample.py'
SELFTEST = 'pf_r2_label_sample_selftest'
FAIL = 'R2 LABEL SAMPLE SELFTEST FAILED'
MUTANTS = [
    ("the stratum cap ignored", "n = min(per_hour, len(ranked)); pi = n / len(ranked)", "n = len(ranked); pi = n / len(ranked)"),
    ("pi fixed at 1", "n = min(per_hour, len(ranked)); pi = n / len(ranked)", "n = min(per_hour, len(ranked)); pi = 1."),
    ("the pilot not the first-ranked", "pilot=(i == 0)", "pilot=(i == n - 1)"),
    ("worlds ranked by key only", "ranked = sorted(ws, key=lambda w: (rank('world', wkey(w)), ckey(wkey(w))))", "ranked = sorted(ws, key=lambda w: (ckey(wkey(w)),))"),
    ("world ties not broken by the key", "ranked = sorted(ws, key=lambda w: (rank('world', wkey(w)), ckey(wkey(w))))", "ranked = sorted(ws, key=lambda w: (rank('world', wkey(w)),))"),
    ("the canonical key unsorted", "return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True)", "return json.dumps(x, sort_keys=False, separators=(',', ':'), ensure_ascii=True)"),
    ("not processed in canonical order", "sel.sort(key=lambda x: (x['day'], x['m'], x['t_ms']))", "sel.sort(key=lambda x: x['rank'])"),
    ("the UTC day not enforced", "C.require(not STRICT_DAY or d0 <= t_ms < d0 + DAY_MS, f'{day}: a world t0 {t_ms} outside its UTC day')", "pass"),
    ("the hour shifted by one", "return int(t_ms // HOUR_MS % 24)", "return int((t_ms // HOUR_MS + 1) % 24)"),
    ("the closure misses lazy imports", "for node in ast.walk(ast.parse((HERE / n).read_text())):", "for node in ast.parse((HERE / n).read_text()).body:"),
    ("the checks in the closure", "_CLOSURE.extend(sorted(n for n in seen if not n.endswith(CHECKS)))", "_CLOSURE.extend(sorted(seen))"),
    ("rows ranked by key only", "pool = sorted(keys, key=lambda k: (rank('row', k), ckey(k)))", "pool = sorted(keys, key=lambda k: (ckey(k),))"),
    ("duplicate keys accepted", "C.require(len({ckey(k) for k in pool}) == N, 'two eligible census rows share one canonical key')", "pass"),
    ("two mandatory rows accepted", "C.require(len(mand) <= 1, 'two neutral original-time POST rows')", "None"),
    ("no mandatory POST", "mand = [k for k in pool if is_mandatory(k)]", "mand = []"),
    ("the mandatory rule ignores the decision", "return k['source'] == 'neutral' and k['shadow_source'] == 'entry' and k['decision_ms'] == 0", "return k['source'] == 'neutral' and k['shadow_source'] == 'entry'"),
    ("the mandatory rule ignores the shadow source", "return k['source'] == 'neutral' and k['shadow_source'] == 'entry' and k['decision_ms'] == 0", "return k['source'] == 'neutral' and k['decision_ms'] == 0"),
    ("the mandatory rule ignores the source", "return k['source'] == 'neutral' and k['shadow_source'] == 'entry' and k['decision_ms'] == 0", "return k['shadow_source'] == 'entry' and k['decision_ms'] == 0"),
    ("the extra q 1 / N", "('extra', rest[0], N, 1. / (N - 1))", "('extra', rest[0], N, 1. / N)"),
    ("the keep q 1", "out = [('keep', pool[0], N, 1. / N)] if pool else", "out = [('keep', pool[0], N, 1.)] if pool else"),
    ("the only q 1", "out = [('only', pool[0], N, 1. / N)] if pool else", "out = [('only', pool[0], N, 1.)] if pool else"),
    ("the weights without pi", "(1. / (pi * n * q) if k is not None else None)", "(1. / (n * q) if k is not None else None)"),
    ("the extra may redraw the mandatory", "rest = [k for k in pool if k is not mand[0]]", "rest = list(pool)"),
    ("the census check not called", "world, M, sup); guard.check('census')", "world, M, sup)"),
    ("the pair check not called", "guard.check('pair', time.time() - t0)", "pass"),
    ("the pool keeps ineligible rows", "elig = [c for c in mine if c['status'] == 'ok']", "elig = list(mine)"),
    ("the hidden copy perturbs the delivered frame", "B2 = R2.hidden_copy(Bc, g, rng, delivered=dl)", "B2 = R2.hidden_copy(Bc, g, rng, delivered=True)"),
    ("the delivered control never perturbs", "B2 = R2.hidden_copy(Bc, g, rng, delivered=dl)", "B2 = R2.hidden_copy(Bc, g, rng, delivered=False)"),
    ("the arm fills not stored", "res['arms'][n_].update(v_)", "pass"),
    ("the stop rule disabled", "if self.worst > self.pair_s or hw > PEAK or el > self.bound_s:", "if False:"),
    ("the pair limit ignored", "if self.worst > self.pair_s or hw > PEAK or el > self.bound_s:", "if hw > PEAK or el > self.bound_s:"),
    ("the high-water ignored", "if self.worst > self.pair_s or hw > PEAK or el > self.bound_s:", "if self.worst > self.pair_s or el > self.bound_s:"),
    ("the bound ignored", "if self.worst > self.pair_s or hw > PEAK or el > self.bound_s:", "if self.worst > self.pair_s or hw > PEAK:"),
    ("the plan rewritten", "C.require(not pp.exists(), 'the plan exists (immutable)')", "pass"),
    ("a stop history reused", "C.require(not RN.archive_stops(out), 'a plan directory with a stop history: use a new namespace')", "pass"),
    ("the plan checkpoint removed", "        if hwm() > PEAK:\n            RN.atomic_json(out / 'plan_status.json'", "        if False:\n            RN.atomic_json(out / 'plan_status.json'"),
    ("the pilot run set ignored", "run = [w for w in sel if (w['pilot'] or not pilot)]", "run = list(sel)"),
    ("partitions overlap", "members=[wkey(w) for w in run[i:i + ps]]", "members=[wkey(w) for w in run[i:i + ps + 1]]"),
    ("the source check skipped", "C.require(sources_now() == plan['sources'], 'stale source: the source closure differs from the plan')", "pass"),
    ("the feature schema unchecked", "C.require(plan['feature_schema'] == dict(n=len(T.FEATS), sha256=hashlib.sha256(ckey(list(T.FEATS)).encode()).hexdigest()), 'the feature schema differs from the plan')", "pass"),
    ("the day inputs unchecked", "C.require(identity_now(d, kp, src, files_fn, ht_fn) == plan['sample'][d]['inputs'], f'changed input: {d}\\'s inputs differ from the plan')", "pass"),
    ("verified after the skip", "        M, sup = verify_current(plan, [day], kp, src, files_fn, ht_fn, assets, cfg)        # before a part is skipped or run\n        if accepted_part(out, plan, psha, part) is not None:\n            return 'DONE'", "        if accepted_part(out, plan, psha, part) is not None:\n            return 'DONE'\n        M, sup = verify_current(plan, [day], kp, src, files_fn, ht_fn, assets, cfg)"),
    ("kp outputs unchecked", "    C.require(all(C.sha(Path(kp) / day / x) == h for x, h in r_['outputs'].items()), f'{day}: keep-pull outputs changed since their receipt')", "    pass"),
    ("the offset not bound", "offset=src.offset(day), files=files_fn(day)", "offset=0., files=files_fn(day)"),
    ("held stops ignored", "        if und:\n            RN.atomic_json(stp, dict(state='HELD'", "        if False:\n            RN.atomic_json(stp, dict(state='HELD'"),
    ("the keyed world build ignores its keys", "want = {(int(m), int(t)) for m, t in keys}; O = O[[(int(m), int(round(float(t)))) in want for m, t in zip(O.m, O.t)]]", "want = None"),
    ("the member worlds unchecked", "C.require(sorted(allw) == sorted(ckey(wkey(w)) for w in members), 'a member world is not one of the day\\'s worlds')", "pass"),
    ("the partition builds every world", "worlds_all(day, kp, keys=[(w['m'], w['t_ms']) for w in members])}", "worlds_all(day, kp)}"),
    ("timing not recorded", "timing=timing(G.log),", "timing=None,"),
    ("the guard log not kept", "; self.log.append((stage, el, pair_s))", ""),
    ("the day inputs read twice", "mk = sorted({w['m'] for w in members}); fair, depth, hrs, trk = inputs(day, mk)", "mk = sorted({w['m'] for w in members}); inputs(day, mk); fair, depth, hrs, trk = inputs(day, mk)"),
    ("the producing identity unchecked", "C.require(day_identity(day, kp, src, hrs, trk, files_fn) == plan['sample'][day]['inputs'], 'changed input: real_inputs\\' hours / tracked differ from the plan')", "pass"),
    ("the load check not called", "G.check('load'); rows, pools, n_w = [], [], 0", "rows, pools, n_w = [], [], 0"),
    ("the world check not called", "            G.check('world')\n", ""),
    ("the planted interruption ignored", "if limit is not None and n_w >= limit:", "if False:"),
    ("a part promoted without validation", "C.require(not bad, f'part {k}: refused at promotion ({bad[:3]})')", "pass"),
    ("non-deterministic part bytes", "with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0) as g:", "time.sleep(1.1)\n    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=None) as g:"),
    ("a disposition of another plan accepted", "if disp.get('plan_sha256') == psha else set()", "if True else set()"),
    ("the byte pre-check removed", "    C.require(hashlib.sha256(b).hexdigest() == rec.get('sha256') and hashlib.sha256(pb).hexdigest() == rec.get('pools_sha256'),\n              f\"part {part['part']}: a promoted part fails validation (its bytes do not hash to the record)\")", "    pass"),
    ("resume skips an invalid part", "    C.require(not bad, f\"part {part['part']}: a promoted part fails validation ({bad[:3]})\")", "    pass"),
    ("finalize ignores held stops", "C.require(not held(out, psha), 'finalize: an undisposed stop')", "pass"),
    ("finalize skips the current identity", "        verify_current(plan, plan['days'], Path(plan['kp_dir'])", "        0 and verify_current(plan, plan['days'], Path(plan['kp_dir'])"),
    ("finalize accepts an unpromoted part", "C.require(rec is not None, f\"finalize: part {part['part']} is not promoted\")", "pass"),
    ("finalize ignores the high-water", "        C.require(hwm() <= PEAK, f'finalize: the process high-water passed {PEAK} MiB')", "        pass"),
    ("finalize ignores the partition cover", "                  'finalize: the partitions do not cover the run set exactly once')", "                  'finalize: the partitions do not cover the run set exactly once') if False else None"),
    ("the record bytes unchecked", "        if raw is not None and (hashlib.sha256(raw[0]).hexdigest() != rec.get('sha256') or hashlib.sha256(raw[1]).hexdigest() != rec.get('pools_sha256')):", "        if False:"),
    ("the record plan unchecked", "        if rec.get('plan_sha256') != psha or rec.get('part') != part['part'] or rec.get('members') != part['members']:", "        if False:"),
    ("the record sources unchecked", "        if rec.get('sources') != plan['sources']:", "        if False:"),
    ("the record inputs unchecked", "        if rec.get('inputs') != plan['sample'][day]['inputs']:", "        if False:"),
    ("the record breach unchecked", "        if rec.get('breach') != dict(seconds=False, rss=False) or not (isinstance(rec.get('hwm_mib'), (int, float)) and rec['hwm_mib'] <= PEAK):", "        if False:"),
    ("the record counts unchecked", "        if rec.get('rows') != len(rows) or rec.get('pools') != len(pools):", "        if False:"),
    ("the executing sources unchecked", "    if enforce_current and sources_now() != plan['sources']:", "    if False:"),
    ("members outside the sample accepted", "    if not all(m in sel for m in members):", "    if False:"),
    ("extra rows accepted", "        if wk not in members or r.get('head') not in PICKS or r.get('pick') not in PICKS.get(r.get('head'), ()):", "        if False:"),
    ("duplicate rows accepted", "        if key in got:\n            bad.append(f'a duplicate row {key}'); continue", "        if False:\n            bad.append(f'a duplicate row {key}'); continue"),
    ("duplicate pool lines accepted", "        if wk not in members or key in pl:", "        if wk not in members:"),
    ("the world status check removed", "            if (len(wst) != 1 or set(mine) != want", "            if False and (len(wst) != 1 or set(mine) != want"),
    ("a missing pool line accepted", "                    bad.append(f'{wk} side {s} {h}: a missing pool line'); continue", "                    continue"),
    ("pool consistency unchecked", "                    bad.append(f'{wk} side {s} {h}: the pool line is inconsistent'); continue", "                    pass"),
    ("the stored picks unchecked", "                if {(wk, s, h, x[0]) for x in exp} != {k_ for k_ in mine if k_[1:3] == (s, h)}:", "                if False:"),
    ("N / q / w unchecked", "                    if not (r.get('pi') == w['pi'] and r.get('world_rank') == w['rank'] and r.get('oid') == w[f'oid{s}'] and r.get('N') == N and same_w):", "                    if False:"),
    ("the empty-pick status unchecked", "                        if st != ('no_additional_post' if pick == 'extra' else 'empty_pool'):", "                        if False:"),
    ("the picked row identity unchecked", "                    if st not in PICKED or r.get('row_key') != key or r.get('row_rank') != rank('row', key) or r.get('source') != key['source']:", "                    if st not in PICKED:"),
    ("ok pairs not checked", "                    bad += [f'{wk} side {s} {pick}: {x}' for x in _pair_bad(plan, day, w, r, inputs_required=(st == 'ok'))]", "                    pass"),
    ("finite values unchecked", " or len(vals) != 9 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):", " or len(vals) != 9:"),
    ("the context unchecked", "    if not (res.get('kind') == r['head'] and res.get('side') == r['side']", "    if False and not (res.get('kind') == r['head'] and res.get('side') == r['side']"),
    ("Y unchecked", "    if abs(res['Y'] - (arms[names[0]]['value'] - arms[names[1]]['value'])) > TOL_V:", "    if False:"),
    ("R unchecked", "    if any(abs(arms[n]['cash'] + arms[n]['position'] * arms[n]['mark'] - arms[n]['value']) > TOL_V for n in names):", "    if False:"),
    ("the ledger unchecked", "    lb = [x for x in (_ledger_bad(arms[n]) for n in names) if x]", "    lb = []"),
    ("the fills hash / count unchecked", "    if not (isinstance(fl, list) and isinstance(sz, list) and len(fl) == A.get('fills') and L._h(fl) == A.get('fills_sha256')):", "    if not (isinstance(fl, list) and isinstance(sz, list)):"),
    ("cash / position unchecked", "    if abs(cash - A['cash']) > TOL_V or abs(pos - A['position']) > TOL_P:", "    if False:"),
    ("generation sizes unchecked", "    if not all(0 <= g_ < len(sz) and q_ <= sz[g_] + TOL_P for g_, q_ in gen.items()):", "    if False:"),
    ("the pre-fork fills unchecked", "        if pre(arms[names[0]]['fill_list']) != pre(arms[names[1]]['fill_list']):", "        if False:"),
    ("the stored flags unchecked", "    if not (res.get('reconciled') is True and res.get('common_prefork_identical') is True and all(arms[n].get('reconciled') is True for n in names)):", "    if False:"),
    ("the result fork unchecked", "    if res.get('fork_fingerprint') != r.get('fork_fingerprint'):", "    if False:"),
    ("the arm callbacks unchecked", "    cb = [x for x in (_arm_calls_bad(arms[n], g, r['side'], first) for n in names) if x]", "    cb = []"),
    ("the arm causal probe unchecked", "    if len(cz) != 2 or not all(c.get('mismatch', 1) == 0 and c.get('parity_mismatch', 1) == 0 for c in cz):", "    if False:"),
    ("the hidden copy unchecked", " and fi.get('hidden_equal') is True\n", "\n"),
    ("the rebuild unchecked", "            and fi.get('rebuild_equal') is True and hashlib", "            and hashlib"),
    ("the inputs hash unchecked", " and hashlib.sha256(np.asarray(fi['inputs'], np.float32).tobytes()).hexdigest() == fi.get('inputs_sha256')):", "):"),
    ("the status not checked against its result", "                    if st != exp_st:\n                        bad.append(f'{wk} side {s} {pick}: the status {st} is not its result\\'s ({exp_st})'); continue", "                    if False:\n                        bad.append(f'{wk} side {s} {pick}: the status {st} is not its result\\'s ({exp_st})'); continue"),
    ("any failed pair is an exclusion", "                        if res_.get('status') not in PAIR_DATA_STATUSES or res_.get('fork_fingerprint') != r.get('fork_fingerprint'):", "                        if res_.get('fork_fingerprint') != r.get('fork_fingerprint'):"),
    ("feature exclusions skip the pair invariants", "_pair_bad(plan, day, w, r, inputs_required=(st == 'ok'))]", "(_pair_bad(plan, day, w, r) if st == 'ok' else [])]"),
    ("END unchecked", "    if not (float(res.get('end', -1.)) == end and abs(float(idn.get('end_abs_ms', -1.)) - float(idn.get('start_abs_ms', 0.)) - end) < 1e-6):", "    if False:"),
    ("the common mark unchecked", "    if not (isinstance(res.get('mark'), (int, float)) and math.isfinite(res['mark']) and all(arms[n]['mark'] == res['mark'] for n in names)):", "    if False:"),
    ("fill times unchecked", "    if not all(isinstance(arms[n].get('fill_list'), list) and all(0. <= float(f[0]) <= end for f in arms[n]['fill_list']) for n in names):", "    if False:"),
    ("callbacks not reconciled", "    if not (t_.get('calls') == n and t_.get('binding_checked') == n and t_.get('binding_bad') == 0):", "    if not (t_.get('binding_bad') == 0):"),
    ("zero callbacks refused", "    if not (t_.get('calls') == n and t_.get('binding_checked') == n and t_.get('binding_bad') == 0):", "    if not (t_.get('calls') == n and t_.get('binding_checked') == n and n > 0 and t_.get('binding_bad') == 0):"),
    ("the decision log hash unchecked", "    if not (isinstance(dl, list) and isinstance(fb, list) and L._h(dl) == A.get('decisions_sha256')):", "    if not (isinstance(dl, list) and isinstance(fb, list)):"),
    ("the pre-fork other side counted", "        if t < g or (t == g and (s == side or first != side)) or (t, s) in fb:", "        if t < g or (t == g and s == side) or (t, s) in fb:"),
    ("engine fallbacks counted as callbacks", "        if t < g or (t == g and (s == side or first != side)) or (t, s) in fb:", "        if t < g or (t == g and (s == side or first != side)):"),
    ("the audit rate unchecked", "    elif not cb and not all(c.get('checked') == -(-int((arms[n].get('teacher') or {}).get('calls', 0)) // ev) for c, n in zip(cz, names)):", "    elif False:"),
    ("the audit rate not passed to the teacher", "                fac = L.teacher_factory(M, sup, world, every=every); res = L.pair_bound(ctx, fac, head)", "                fac = L.teacher_factory(M, sup, world); res = L.pair_bound(ctx, fac, head)"),
    ("a feature exclusion without its reason", "    elif fi.get('status') != 'feature_unavailable':", "    elif False:"),
]
RUN = """import sys, types
sys.path.insert(0, {here!r})
mod = types.ModuleType('pf_r2_label_sample'); mod.__file__ = {src!r}; sys.modules['pf_r2_label_sample'] = mod
exec(compile(open({mut!r}).read(), {src!r}, 'exec'), mod.__dict__)
import pf_r2_label_sample_selftest as ST
ST.run(mod)
"""


def run_copy(text, tmp):
    p = Path(tmp) / 'mutant.py'; p.write_text(text)
    try:
        r = subprocess.run([sys.executable, '-c', RUN.format(here=str(HERE), src=str(SRC), mut=str(p))], capture_output=True, text=True, env=dict(os.environ), timeout=1200)
    except subprocess.TimeoutExpired:
        return None, 'TIMEOUT (not a catch)'
    return r.returncode, r.stdout + r.stderr


def main():
    k_, n_ = (int(x) for x in sys.argv[1].split('/')) if len(sys.argv) > 1 else (0, 1)
    base = SRC.read_text()
    bad = [n for n, old, _ in MUTANTS if base.count(old) != 1]
    if bad:
        sys.exit(f'r2 label sample mutation check REFUSED: targets not found exactly once: {bad}')
    mine = [(i, m) for i, m in enumerate(MUTANTS) if i % n_ == k_]
    with tempfile.TemporaryDirectory() as tmp:
        rc, out = run_copy(base, tmp)
        pre = rc == 0 and 'r2 label sample selftests passed' in out
        print(f'shard {k_}/{n_}: unchanged copy passes its selftest: {pre}', flush=True)
        if not pre:
            sys.exit(f'r2 label sample mutation check FAILED: the unchanged copy does not pass ({out[-400:]})')
        caught = []
        for i, (name, old, new) in mine:
            rc, out = run_copy(base.replace(old, new), tmp)
            hit = rc not in (0, None) and FAIL in out
            why = out[out.index(FAIL):][:170].replace('\n', ' ') if hit else ('PASSED (not caught)' if rc == 0 else out if rc is None else 'CRASHED: ' + out.strip().splitlines()[-1][:170])
            caught.append(hit); print(f'  {i} {name}: {"caught" if hit else "NOT CAUGHT"} - {why}', flush=True)
    ok = all(caught)
    print(f'r2 label sample mutation check (shard {k_}/{n_}) {"passed" if ok else "FAILED"}: the unchanged copy passes its selftest, {sum(caught)} / {len(mine)} planted '
          f'code errors are caught by a check')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
