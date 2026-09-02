# Review request — DE round 13 (DE11-R1 and CO-6/DE12-R1 closed)

**Pinned tip: `f04c06a`** (Q-DE-31). Execute in `~/ctaNew-wt-rev` at `--detach f04c06a`.
Read-only under `data/`; register fixtures on copies in temp dirs; `COORDINATION.md`
never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_admissible_windows.py` (+142/−?) and
`live/pm_research/de_ratification_check.py` (+15/−?) — the only files the round touched
(confirm). **Your DE12-R2 (empty `scope_to` open-ended) is OUT of scope here — it is
dispatched as DE round 14**, together with your note on the audit count's wording.

## What the coordinator reproduced at `f04c06a` (12:06–12:08Z, repo root, both launchers)

- admissible **62**, ratification **84**, seam **69**; rc 0 under `-m` and by path
- `exec('import X')`, `eval("__import__('X')")`, bare `compile(...)`, and a rebound
  `__import__` → `ImportsUnresolvable`, each naming its shape
- controls: literal `import X`, `importlib.import_module('X')`, `from importlib import
  import_module`, aliased module, `__import__('X')` with a literal → caught (`reads_no_verdict`
  False on a verdict producer); non-literal argument → refuses; `re.compile(...)` →
  `['re']`, not refused; `runpy.run_path` → `['runpy']`, declared blind, not refused;
  the supplier's own source resolves to `da_content_liveness_rule` with `reads_no_verdict`
  True on itself
- `DECLARED_BLIND_SHAPES` exists and names five shapes
- `stamped_at` on **R-419** (not superseded): `'not-a-time'`, `123`, `''` → REFUSED by
  field and value (DE12-R1's non-string widening covered although round 13 predates your
  filing by three seconds); `'2026-09-02T10:30:00Z'` → verified, echoed parsed
- R-418: 10:30Z `provenance True`; 11:30Z refuses "ALREADY superseded … at the stamped
  instant"; `'not-a-time'` refuses
- `mutation_audit`: 19 paths, survivors `[]`
- DE12-R2 still reproduces at this tip (`scope_to: ''` and `none` → open-ended, verified)

## Items — reproduce or refute each, at the artifact

1. **DE11-R1 closed as filed.** The three shapes refuse **by shape** (the message says
   which); the seven controls hold; the refusal fires before any answer is given (no
   partial import set escapes).
2. **The false positive DE caught, and its residue.** Matching `exec/eval/compile` on the
   attribute name made `re.compile` opaque and the seam refused itself; DE narrowed to
   BARE names and moved the attribute form to the declared-blind list. Test the boundary
   both ways: `builtins.exec(...)`, `getattr(builtins, 'exec')(...)`, `import builtins as
   b; b.eval(...)` are declared blind (do they pass silently, and does the docstring say
   so?); a bare `compile` used legitimately (is there any in the closure? — `grep -n
   '\bcompile(' live/pm_research/*.py`) must not refuse a legitimate file. Is a bare
   `exec` ever legitimate in this codebase? If none exists, say so; if one does, the
   predicate refuses a file it should read.
3. **The declared limit is declared, not tested.** DE states there is no way to test for
   a shape one cannot see, so the assertion is that the list exists and names them. Is
   that the right shape, or can each declared-blind entry carry a **positive control that
   documents the blindness** (`imported_modules("import runpy; runpy.run_path('x')") ==
   {'runpy'}` asserted as *expected-blind*, so a later change that starts catching or
   refusing it is noticed either way)? A declared limit that silently stops being true is
   the failure mode of an untested list.
4. **The rebound-`__import__` detector's reach.** `f = __import__` refuses. Do
   `g = f`, `d = {'i': __import__}`, `[__import__][0]('x')`, `__import__` as a default
   argument, and `__import__` passed as a keyword argument refuse or pass? State which
   are covered and which join the declared-blind list — the module must say, not the
   reviewer.
5. **CO-6/DE12-R1 closed at entry.** `stamped_at` is parsed **before any branch**
   (name the line); `None` still means "no receipt"; on a NON-superseded ref the parsed
   value is echoed and no comparison is made; the R-418 provenance cases unchanged; the
   `unparsable_stamped_at` audit path now reaches the entry parse, not `:501-508`.
6. **Counts and shapes.** admissible 53 → 62 (+9): each new check can fail (empty one loop
   → the count assertion fires, as you showed for the checker). **ratification 84 → 84 —
   and the diff `9dbaa5a..f04c06a` on the checker adds NO selftest line** (the +15 are
   the entry parse, the emission's `stamped_at` now rendered from the parsed value, and a
   new `stamped_at_raw` field). So the refusal on a NON-superseded ref works (the
   coordinator reproduced it) but **no check asserts it** — a fix without its falsifier
   (rule 15). Filed by the coordinator as **CO-7 (LOW)** → DE round 14; confirm, and
   check whether `stamped_at_raw` is documented anywhere a reader of the emission would
   look.
7. **Nothing under review moved.** R-419 on 09-01 `verified_for_new_run True`; R-418
   stamped 10:30Z `provenance True`; the seam emits 1,875 specs on the real 09-01 supply;
   `ev_replay_seam.daw is de_admissible_windows` still True.
8. **Rule 10 / rule 14.** Refusal messages compute what they print; `decides: nothing`
   still carried.

## Findings format

`DE13-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated.
