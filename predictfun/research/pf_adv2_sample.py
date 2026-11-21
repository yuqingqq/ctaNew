"""ADV2-RICH v1: THE EVALUATION LABEL SAMPLE ON CONSUMED OCT 26-27 (plan v2 R3; P-2026-005; 2026-11-21, session aea5bfb9).
Review 315979d2's next priority, item 7: "extend the reviewed deterministic sample contract to each evaluation day: up to 64 worlds per UTC
hour, same canonical ranking/domain/ties, five-source census, mandatory POST + uniform extra POST + uniform KEEP, recomputing that day's
inclusion probabilities. Maximum 3,072 worlds / 18,432 selected labels across Oct26-27."
Written before any Oct 26-27 label exists; nothing in this module reads a label. The frozen runner pf_r2_label_sample (4cdeff0c, the R2-F
freeze d3d024fd) is imported unchanged and called with days = EVAL_DAYS: its plan, sample, census, pairs, validator, resume and finalize are
the R2-F ones (the domain, ranks, ties, picks, weights, audit every 35, limits and the frozen pi0_50_v1).

THE ONE CHANGE, the day rule (CLAUDE.md "padded hours reach sealed days"). The runner's input identity (pf_r2_label_sample.real_files)
hashes EVERY book part in step1/<day>/books. The Oct 27 directory also holds the next day's 00h part, which is Oct 28: a sealed date that
this program never opens. The day loader never reads that part (pf_stop_ev_minutes_inputs.load_day skips any part outside
[day 00:00 - 1 h, day 24:00): part_allowed, SKIP_FOREIGN_PARTS = True). guarded_files is real_files with the book parts filtered BY NAME
through the loader's own part_allowed before any file is opened, so the identity lists exactly the parts the loader can read. It is passed
to plan, run and finalize; real_files is never called on an evaluation day. The other readers of the chain were audited for padding:
real_inputs / depth_hours / fair_fix read hours -1..23 through part_allowed and hour_ok; the flow-price receipts' input lists hold no
date after the day; the census and the pairs read no file.
CHECKS (pf_adv2_sample_selftest, synthetic roots except the last): the guard never opens a part outside hours -1..23 (a recorder of every
 hashed path) while the frozen rule does (the positive control); a sealed date is refused before any hash; the identity's own name check
 and the loader flag hold; plan_ok flags seven corruptions; plan / run / finalize hand the guard and the declared days to the frozen runner
 and make refuses a plan failing its post-condition; on consumed Oct 24 guarded_files = real_files minus exactly the after-day 00h part.
 Mutation check: pf_adv2_sample_mutants.py (19 planted errors, each caught by a check).
Usage (run_light): pf_adv2_sample.py selftest | plan <out> | run <out> <part> | finalize <out>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_r2_label_sample as LS                           # noqa: E402  frozen (4cdeff0c), unchanged

C, KP, X = LS.C, LS.KP, LS.X
LOADER = KP.I                                              # the tape's day loader (pf_stop_ev_minutes_inputs.load_day)
VERSION = 'adv2_eval_sample_v1'
EVAL_DAYS = ('20261026', '20261027')                        # plan v2: consumed development evaluation (never validation)
FORBIDDEN = ('20261028',)                                   # sealed: never opened, never listed in an identity


def _files(day, io_root, imp_dir, fp_dir, keep, sha):
    """pf_r2_label_sample.real_files' list (same files, same order) with the book parts kept only if keep(path, day) -> {path: sha(path)}."""
    s1, s5 = Path(io_root) / 'step1' / day, Path(io_root) / 'step5' / day
    books = [p for p in sorted((s1 / 'books').glob('part_*.parquet')) if keep(str(p), day)]          # by name: nothing opened yet
    fs = [s1 / 'registry.parquet', s1 / 'clock_rl_d2.json', s1 / 'legs_timed2.parquet', s1 / 'legs_tk.parquet', s5 / 'preds_t60k.parquet',
          s5 / 'rows_rl_d2_pk.parquet', Path(imp_dir) / f'day_{day}_imp.json', Path(fp_dir) / day / 'receipt.json'] + books
    C.require(all(p.exists() for p in fs), f'{day}: a producing input file is missing')
    return {str(p): sha(p) for p in fs}


def _roots():
    return Path(KP.I.IO), Path(KP.QL.C.V.E.OUTD), Path(X.QP.JD.C.OUT)


def guarded_files(day, sha=None, roots=None):
    """The day's producing-input identity with only the book parts the loader can read (part_allowed by name, before any open).
    (sha / roots: a recorder and synthetic roots in the selftest only.)"""
    C.require(LOADER.SKIP_FOREIGN_PARTS is True, 'the day loader no longer skips foreign parts: the guard would not mirror it')
    C.require(day not in FORBIDDEN, f'{day}: a sealed date')
    io_, imp, fp = roots or _roots()
    out = _files(day, io_, imp, fp, LOADER.part_allowed, sha or C.sha)
    C.require(not any(f in Path(p).name for p in out for f in FORBIDDEN), f'{day}: a sealed date in the identity')
    return out


def plan_ok(plan):
    """The plan's post-condition (header) -> a list of failures (empty = ok)."""
    bad = []
    want = dict(version=LS.VERSION, domain=LS.DOMAIN, days=list(EVAL_DAYS), per_hour=LS.PER_HOUR, run_set='full', audit_every=LS.AUDIT_EVERY,
                part_size=LS.PART_WORLDS, strict_day=True, freeze_sha256=LS.L.FREEZE_SHA)
    bad += [f'{k}: {plan.get(k)!r} != {v!r}' for k, v in want.items() if plan.get(k) != v]
    if plan.get('limits') != dict(LS.LIMITS, peak_mib=LS.PEAK, guard_mib=LS.GUARD):
        bad.append(f"limits {plan.get('limits')}")
    for day in EVAL_DAYS:
        files = plan['sample'][day]['inputs']['files']
        bad += [f'{day}: a book part not allowed: {Path(p).name}' for p in files if '/books/' in p and not LOADER.part_allowed(p, day)]
        bad += [f'{day}: a sealed date in the identity' for p in list(files) + list(plan['sample'][day]['inputs']['tracked']) if any(f in Path(p).name for f in FORBIDDEN)]
        bad += [f'{day}: a member of another day' for w in plan['sample'][day]['selected'] if w['day'] != day]
    bad += [f"part {p['part']}: members of another day" for p in plan['partitions'] if any(m['day'] != p['day'] for m in p['members'])]
    return bad


def run_part(out, k):
    return LS.run(out, int(k), files_fn=guarded_files)


def finalize_out(out):
    return LS.finalize(out, files_fn=guarded_files)


def make(out):
    C.require(not Path(out).exists() or not (Path(out) / 'plan.json').exists(), 'the plan exists (immutable)')
    plan = LS.make_plan(out, days=EVAL_DAYS, files_fn=guarded_files)
    bad = plan_ok(json.loads((Path(out) / 'plan.json').read_text()))
    C.require(not bad, f'the evaluation plan fails its post-condition: {bad[:5]}')
    psha = C.sha(Path(out) / 'plan.json')
    side = dict(version=VERSION, plan_sha256=psha, wrapper_sha256=C.sha(Path(__file__)), runner_sha256=C.sha(HERE / 'pf_r2_label_sample.py'),
                days=list(EVAL_DAYS), partitions=len(plan['partitions']), worlds=sum(len(p['members']) for p in plan['partitions']),
                strata={d: plan['sample'][d]['strata'] for d in EVAL_DAYS}, files_rule='guarded_files (part_allowed by name)', asof=C.stamp())
    (Path(out) / 'eval_sample.json').write_text(json.dumps(side, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(plan_sha256=psha, partitions=side['partitions'], worlds=side['worlds'],
                          N_worlds={d: plan['sample'][d]['N_worlds'] for d in EVAL_DAYS}), indent=1))


def selftest():
    import pf_adv2_sample_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a[:1] == ['plan'] and len(a) == 2:
        make(a[1])
    elif a[:1] == ['run'] and len(a) == 3:
        print(run_part(a[1], a[2]))
    elif a[:1] == ['finalize'] and len(a) == 2:
        print(json.dumps({k: v for k, v in finalize_out(a[1]).items() if k != 'parts'}, indent=1, default=str))
    else:
        sys.exit(__doc__)
