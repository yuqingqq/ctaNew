"""MUTATION CHECK of pf_adv2_sample (the evaluation label sample's day guard; review 315979d2's round; 2026-11-21, session aea5bfb9). Each
planted error is a textual replacement in a copy of pf_adv2_sample.py (the file on disk is never touched), executed as the module under its
real path in a fresh process by pf_mutant_one.py, which runs pf_adv2_sample_selftest.run on it. Required: the unchanged copy passes; every
mutant is caught BY A CHECK (the selftest's FAIL message; a crash is not a catch); every target occurs exactly once.
Per-mutant jobs (one process each): pf_mutant_queue.sh with pf_mutant_one.py / pf_mutant_summary.py, which import MUTANTS, SRC, SELFTEST and
FAIL from here."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_sample.py'
SELFTEST = 'pf_adv2_sample_selftest'
FAIL = 'ADV2 SAMPLE SELFTEST FAILED'
MUTANTS = [
    ('the guard keeps every book part', "out = _files(day, io_, imp, fp, LOADER.part_allowed, sha or C.sha)", "out = _files(day, io_, imp, fp, lambda p, d: True, sha or C.sha)"),
    ('the book parts listed without the name filter', "if keep(str(p), day)]", "]"),
    ('the sealed-date refusal removed', "    C.require(day not in FORBIDDEN, f'{day}: a sealed date')\n", ""),
    ('the identity name check removed', "    C.require(not any(f in Path(p).name for p in out for f in FORBIDDEN), f'{day}: a sealed date in the identity')\n", ""),
    ('the loader-flag check removed', "    C.require(LOADER.SKIP_FOREIGN_PARTS is True, 'the day loader no longer skips foreign parts: the guard would not mirror it')\n", ""),
    ('an identity file dropped', "s1 / 'legs_tk.parquet', ", ""),
    ('plan_ok: days unchecked', "days=list(EVAL_DAYS), per_hour", "per_hour"),
    ('plan_ok: audit rate unchecked', "run_set='full', audit_every=LS.AUDIT_EVERY,", "run_set='full',"),
    ('plan_ok: limits unchecked', "    if plan.get('limits') != dict(LS.LIMITS, peak_mib=LS.PEAK, guard_mib=LS.GUARD):\n        bad.append(f\"limits {plan.get('limits')}\")\n", ""),
    ('plan_ok: book parts unchecked', "        bad += [f'{day}: a book part not allowed: {Path(p).name}' for p in files if '/books/' in p and not LOADER.part_allowed(p, day)]\n", ""),
    ('plan_ok: sealed names unchecked', "        bad += [f'{day}: a sealed date in the identity' for p in list(files) + list(plan['sample'][day]['inputs']['tracked']) if any(f in Path(p).name for f in FORBIDDEN)]\n", ""),
    ('plan_ok: selected days unchecked', "        bad += [f'{day}: a member of another day' for w in plan['sample'][day]['selected'] if w['day'] != day]\n", ""),
    ('plan_ok: partition days unchecked', "    bad += [f\"part {p['part']}: members of another day\" for p in plan['partitions'] if any(m['day'] != p['day'] for m in p['members'])]\n", ""),
    ('run without the guard', "    return LS.run(out, int(k), files_fn=guarded_files)", "    return LS.run(out, int(k))"),
    ('finalize without the guard', "    return LS.finalize(out, files_fn=guarded_files)", "    return LS.finalize(out)"),
    ('plan without the guard', "    plan = LS.make_plan(out, days=EVAL_DAYS, files_fn=guarded_files)", "    plan = LS.make_plan(out, days=EVAL_DAYS)"),
    ('plan on the default (training) days', "    plan = LS.make_plan(out, days=EVAL_DAYS, files_fn=guarded_files)", "    plan = LS.make_plan(out, files_fn=guarded_files)"),
    ('the post-condition not enforced', "    C.require(not bad, f'the evaluation plan fails its post-condition: {bad[:5]}')\n", ""),
    ('the declared days changed', "EVAL_DAYS = ('20261026', '20261027')", "EVAL_DAYS = ('20261026', '20261028')"),
]
