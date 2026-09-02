# Review request — DE round 12 (DE10-R1: every temporal comparison parsed, not sorted)

**Pinned tip: `9dbaa5a`** (Q-DE-30). Execute in `~/ctaNew-wt-rev` at `--detach 9dbaa5a`.
Read-only under `data/`; register fixtures on copies in temp dirs; `COORDINATION.md`
never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_ratification_check.py` only (the only file the round
touched — confirm). The seam and the supplier are unchanged at this tip; do not re-review
them. **DE11-R1 (your own finding, `exec`/`eval`/rebound `__import__`) is OUT of scope
here — it is dispatched as DE round 13.**

## What the coordinator reproduced at `9dbaa5a` (12:0xZ, repo root, both launchers)

- selftest **84 / 84**, rc 0 under `-m` and by path
- `now_utc="zzzz"`, `"aaaa"`, `""` → REFUSED by field and value, naming the accepted
  formats; `now_utc=123` → REFUSED as "int, not a string" (a refusal, not a TypeError)
- default `now_utc` → `now_utc_source: wall_clock`; injected → `injected`
- boundary on 09-01: `23:59:59Z` → `day_closed False, verified False`; `00:00:00Z` and
  `00:00:01Z` → True
- `scope_to` ∈ {`not-a-date`, `aaaa`} and `scope_from` ∈ {`not-a-date`, `zzzz`} → REFUSED
  by field and value (both permissive and restrictive garbage refuse); `scope_to: null`
  → open-ended, verified
- a superseder whose heading carries the well-shaped `2026-99-99T99:99Z` → REFUSED,
  naming the entry and the value
- `mutation_audit`: 19 paths, survivors `[]`
- R-419 / R-418@10:30Z unchanged (`provenance True`)

## Items — reproduce or refute each, at the artifact

1. **DE10-R1 closed in BOTH directions.** For each of `now_utc`, `scope_from`,
   `scope_to`, and the superseder heading timestamp: a value that would sort PERMISSIVE
   as a string and one that would sort RESTRICTIVE both refuse **by field and value**.
   Then the direction the round could have missed: is there any temporal field the
   checker still compares as a string? Enumerate every `<`/`>`/`>=`/`<=` between
   timestamp-like values in the module and name what each operand is at that line.
2. **The refusal is a refusal, not a crash wearing one.** For every parsed field: a
   non-string (int, None, list) refuses with the checker's own message, never a
   `TypeError`/`AttributeError`. **`stamped_at` is parsed only on the superseded
   branch** (`:501-508`): on R-418 `stamped_at="not-a-time"` refuses by name (selftest
   `:972`), but on R-419 — not superseded — the coordinator's `stamped_at="not-a-time"`
   returned `verified True` and the emission carries the garbage verbatim at `:641`,
   never parsed. State whether that is a finding: a stamp supplied is a claim about a
   receipt whether or not a superseder exists today, and a stamp that is not parsed
   until the day a superseder appears is a value that sorts nowhere until it matters
   (DE10-R1's shape, one branch over). Severity is yours; the coordinator reads it as
   LOW and would route it to DE round 13.
3. **`now_utc_source` is honest.** Injected vs wall-clock is recorded from what actually
   happened, not from whether the kwarg was passed with a truthy value — try `now_utc=None`
   explicitly.
4. **`null` open-ended is the only open-ended spelling.** An absent `scope_to` line
   (CO-5, closed in round 10) still reads `unverifiable ['day_in_scope']`, not
   open-ended; `scope_to: ` (empty after the colon), `scope_to: None`, `scope_to: ~` each
   either refuse or say what they are — none reads open-ended silently.
5. **The control that ran nothing.** DE disclosed that one selftest control was written
   as a conditional expression and silently ran nothing for one case; it was rewritten.
   Read the selftest section for any remaining shape where a loop/branch can pass with
   zero assertions executed (a comprehension used for side effects, a `for` over an
   empty list, an `if` whose body is the only assertion). Each check should be able to
   fail; name any that cannot.
6. **Audit at 19.** The five new paths: each `refuses_when_live` and, where the audit
   has a skip form, `refuses_when_disabled: false`; where it drives a bad input against a
   control (the `check()` audit has no `skip_guard` — your round-11 note), confirm the
   new cases are on uniquely-owned refusals, as you did for MISSING.
7. **Nothing under review moved.** R-419 on 09-01 `verified_for_new_run True`; R-418
   stamped 10:30Z `provenance True`; the seam still emits 1,875 specs on the real 09-01
   supply (the checker is in its closure); `de_admissible_windows` selftest count
   unchanged from your round-11 figure.
8. **Rule 10 / rule 14.** Refusal messages compute what they print; the emission still
   carries `decides: nothing`.

## Findings format

`DE12-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated.
