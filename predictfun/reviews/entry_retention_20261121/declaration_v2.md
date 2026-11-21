# ENTRY-RETENTION-DISCRIMINATION-v1 — freeze 2b: the falsifier's argument plumbing (committed before its output)

Session Claude aea5bfb9, 2026-11-21. It amends [declaration.md](declaration.md) (freeze `2ac6e084`) for the falsifier only.

**What happened.** The frozen checks A–H ran on the outputs and are all true (`checks.json`, kept). The frozen falsifier then raised
`TypeError: Value after * must be an iterable, not PosixPath` at its first case, before recording any case (`logs/retention_checks.log`).
The cause is one line: `args_c, args_x = (d, clean_e) if on == 'out' else (clean, d)` unpacks the chosen pair into two paths instead of one
argument tuple per case.

**The fix.** [retention_checks_v2.py](retention_checks_v2.py) is the frozen `falsify()` with that line replaced by
`args = (d, clean_e) if on == 'out' else (clean, d)`. The logs switch is made on the frozen module's global (`X.LOGS`). The cases,
corruptions and checks are the frozen ones, imported from `retention_checks.py`. It writes `checks_falsifier_v2.json`. No check, output
or module changes.
