# Review — BE round 8 (CO-12: attribution moved from the transcript to the stderr failure line; CO-13: the double count) — the closure of my own two BE-7 findings
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `c54e48e`** (on top of `fcafe9f`; row Q-BE-233 at `256fef4`); one pathspec
commit, `be_forward_day.py` only, **+118/−11**; driver sha **`ab65b026e3093cad`** = BE's row;
2,687 lines.
**Request of record:** `REQUEST_BE_ROUND_8_2026-09-02.md`. **Round 8 alone**, as dispatched.
**Composed 2026-09-02T18:46:44Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach c54e48e`; `~/ctaNew-wt-be`
and `~/ctaNew-wt-de` never read. **The main tree's `be_forward_day.py` was never read, run,
counted or cited** (standing_rule 9) — every number below is from the committed blob or from a copy
the module itself builds. `__pycache__` cleared before every execution; both launchers; streams
captured separately; the file **restored byte-identical** (sha `ab65b026e3093cad` after the whole
battery) and `git status --short` **0 lines**. Nothing under `data/`: the canonical `derived/`
listing is **identical** before and after, **173 entries**. No unit, timer, scope or anchor;
`DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run; `git worktree list` unchanged.
**BE7-R1..R4 and BE6-R1..R7 are round 9's items and are not re-found here** — I confirmed only that
nothing in this diff touches them.

## 0. What executed

| launcher | rc | `  PASS` lines | printed total | wall |
|---|---|---|---|---|
| `python3 -m live.pm_research.be_forward_day --selftest` | 0 | **106** | `106 checks OK` | 5 m 13 s |
| `python3 live/pm_research/be_forward_day.py --selftest` | 0 | **106** | `106 checks OK` | 5 m 15 s |
| `--no-such-flag` | **2** | — | — | — |

stderr two lines under each (the numpy-reload warning, round 6's item 6, round 9's to close). The
shipped audit run on its own: **12 cases, 0 survivors, baseline green, 3 m 37 s**.

## 1. The attribution predicate, driven four ways — and the substring count (item 1)

`_audit_failure_line` (`:1496-1506`) takes the text after the **last** `"AssertionError: "` on
**stderr**; `_audit_attributed` (`:1509-1514`) asks `want in` that line. Driven in-process:

| drive | failure line | attributed |
|---|---|---|
| (a) the `want` appears only on a `  PASS  ` line | `None` | **False** |
| (b) empty stderr | `None` | **False** |
| (c) chained traceback, handled first / fatal last | `'SECOND label, the one it DIED on'` | **True** for the last label |
| (c′) the same stderr, asking for the FIRST label | same | **False** — which is `rfind`'s job |
| (d) death with no AssertionError (`KeyError`) | `None` | **False** → survivor |
| (d′) a named refusal, no traceback | `None` | **False** → survivor |

**The substring question, counted as asked.** For each of the **12** shipped `want`s I counted the
check labels it matches in this tip's own 106-line green transcript:

- **8 wants match exactly one label** (cases 1, 2, 3, 4, 5, 9, 10, 11 — 3 and 4 share one `want`
  and one target check, which is two edits aimed at one assertion, not an ambiguity);
- **3 wants match ZERO labels** (cases 6, 7, 8) — and this is the **strongest** form, not a gap:
  their target is `ok(False, f"R5(5) a decision-shaped key ({_k}) must REFUSE")` (`:2202`) and its
  sibling, labels that **only ever exist on a red line**. The audit's own table confirms it: those
  three cases' `died_at` values are exactly those strings;
- **1 want matches two labels** — `BE34-R4 a usage error RETURNS 2` matches its own check
  (`:2317`) **and** the CO-12 end-to-end control (`:2549-2565`), whose message interpolates
  `died_at[:60]` and therefore quotes the case's own failure line. Filed **BE8-R1 (LOW)**.

**So the count the request asks for is 1 of 12** — one `want` that could be satisfied by a check
other than its target.

One boundary worth recording because it is what keeps a second collision unreachable: the launch
check's label interpolates the child's tail (`Child tail: {…[-300:]}`), which for a failing child
would contain that child's own `AssertionError: <label>` — and `rfind` would then attribute to the
grandchild's label. It cannot happen inside the audit because the audit's children carry
`BE_FORWARD_LAUNCH_CHECK=1` and return from `_selftest_launch` before that check runs. It is a
dependency, not a defect, and worth a comment beside the flag.

## 2. The falsifier in both directions, end to end (item 2)

The control at `:2549-2565` takes the **one** usage edit from `AUDIT_CASES`, names it twice, and
drives both through the **real** `mutation_audit`: the mis-named copy must be in `survivors`, the
correctly-named one must have `died_at_named_check True`, **and both `died_at` values must be
equal** — that last conjunct is what makes it a falsifier of the **attribution** rather than of the
edit, because it proves the two runs died at the same place and only the name differed. It passes
in both launchers' runs (transcript line 112).

**And it would not have discriminated at `fcafe9f`.** I did not need to re-checkout to show it: the
round-7 predicate was `want in (stdout + stderr)`, so on this tip's own bytes —

```
want = "R5(1) KNOWN-BAD: a SECOND run into the same outdir"   (the WRONG name)
round-7 predicate  (want in stdout+stderr)      -> True     # would have read as KILLED
round-8 predicate  (_audit_attributed, stderr)  -> False    # SURVIVOR, correctly
round-8 with the right name                     -> True
```

— which reproduces, on the released tip, exactly what I measured at `fcafe9f` in BE round 7
(`0f34aad`, §4: the mis-named usage case read `rc 1, died_at_named_check True`). **CO-12's
mechanism is closed at the predicate, and the closure has its own falsifier.**

## 3. The three (four) in-process controls, verified by mutation (item 3)

BE's claim — that leaving these **ungated** is what lets a mutation of the predicate be killed by
the shipped audit rather than merely noticed — is **verified through the audit itself**, which is
the drive the request asked for:

| shipped case (the predicate mutated) | rc | `died_at` |
|---|---|---|
| `CO-12 the attribution matches the TRANSCRIPT…` | 1 | `CO-12 KNOWN-BAD: a `want` that appears only on a `  PASS  ` line` |
| `CO-12 the failure line is taken from the FIRST…` (`rfind`→`find`) | 1 | `CO-12 with a CHAINED traceback the attribution takes the LAST As…` |

Each dies **at the control written for it**, by the stderr line — not incidentally. A count note,
not a finding: the block at `:2416-2455` holds **four** `ok`s, not three — the positive (the label
the child died on), the PASS-line known-bad, the chained-traceback one, and the no-AssertionError
one. All four run inside the audit's children, which is the property the design rests on.

## 4. The twelve cases by the line each died on (item 4)

Reproduced from the audit's own `per_case[...]["died_at"]`: **12 cases, 0 survivors, every
`rc = 1`, every `died_at_named_check` True**, and each `died_at` the line Q-BE-233's table names —
including the three red-only `R5(5)` labels of §1 and `R-421(2)/BE34-R5 the closure DECLARES its
method` for case 9.

**The `find`/`rfind` case is killable only because of the chained fixture**, measured directly:

```
single AssertionError : rfind -> 'ONLY label'   find -> 'ONLY label'    (identical)
chained (two)         : rfind -> 'SECOND'       find -> 'FIRST'         (differ)
```

So the case is a **no-op** unless stderr carries two AssertionErrors — the fixture at `:2439-2444`
is what distinguishes the two spellings, exactly as the row states, and renaming would not have
done it.

## 5. CO-13 (item 5)

**106 assertions, 106 `  PASS  ` lines, summary 106, under both launchers** — the figure of record
and the assertions that ran are the same number. The stray increment is gone (`:2566-2569` now
carries only the comment explaining what it was). **CO-13: CONFIRMED CLOSED.**

The launch-parity arithmetic is untouched by the removal: `_before` is captured before
`_selftest_launch`, and the guard that the launch check contributed at least one check still
stands. One residual, filed **BE8-R2 (LOW)**: `checks` is still incremented in two places —
`ok` (`:1632`, nonlocal) and `_selftest_launch` (`:2649`, on its parameter, returned and assigned
back over `ok`'s increment). The two cancel, which is why 106 = 106 today; nothing asserts they
must. A second `ok` inside that function would print two PASS lines and add one, which is CO-13's
shape in the one function whose arithmetic is already delicate.

## 6. Discipline and layout (item 6)

`_audit_tree` **copies** importable siblings and the comment now says so **and why**
(`:1386-1392`: `resolve()` follows a symlink, so symlinked siblings put the real tree back at the
front of `sys.path`) — the round-7 mismatch between that comment and the code is closed. The
audit's children run in a throwaway tree; nothing was written under `data/`; the register was not
read for this round beyond the request.

The layout fact holds as R-460 §1 states it: my worktree mirrors `data/pm_5min/` per entry, and the
suite runs green there; a bare detached worktree would refuse at check 24 because the driver reads
the ledger from the hardcoded `REPO` while `flow_intensity` resolves `data/pm_5min` tree-relative.
That refusal writes nothing and is correct. It is the BE7-R4 class from the other side and is
**round 9's**, so it is named here and not re-filed.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| BE8-R1 | LOW | `:1514`, `:2560-2565` | one shipped `want` (1 of 12) also matches a **second** check's label, because that check quotes `died_at` in its message |
| BE8-R2 | LOW | `:1632`, `:2649` | CO-13 is closed by removal; nothing asserts that `ok` is the only incrementer, and two increments still cancel by arrangement |

**BE8-R1.** `_audit_attributed` ends in `want in line` — a substring test on the failure line. At
this tip exactly one `want`, `BE34-R4 a usage error RETURNS 2`, appears in a second label: the
CO-12 end-to-end control's own message, which interpolates the `died_at` it just measured. A mutant
that killed **that** control would therefore be credited to case 12. It is not reachable by any
shipped case (the BE34-R4 check runs first), and the error direction is a false KILL, not a false
survivor. Closure, one word: attribute with `line.startswith(want)` — the control's own failure
line begins `CO-12 the attribution HAS a falsifier…`, so the collision disappears while every real
case still matches, since each `want` is a prefix of its target label.

**BE8-R2.** Measured: `ok` increments `checks` and prints one PASS line together, so those two can
never diverge; the only other increment is `_selftest_launch`'s (`:2649`), whose returned value the
caller assigns back over `ok`'s nonlocal increment, cancelling it. Correct today. Closure in the
module's own idiom: count the PASS lines the process printed and assert that count equals the
summary before printing it — the invariant CO-13 violated, made checkable rather than restored by
deletion.

## Disposition

**RELEASE `c54e48e`.** **CO-12: CONFIRMED CLOSED** — the attribution is taken from the stderr
failure line, refuses a `want` seen only on a PASS line, takes the last label of a chained
traceback, and treats a death with no AssertionError as a survivor; all four driven by me, and the
end-to-end control is a falsifier of the attribution because it asserts both runs died at the same
line under different names. **CO-13: CONFIRMED CLOSED** — 106 assertions, 106 PASS lines, 106 in
the summary, under both launchers. The shipped audit reproduces at 12 cases and 0 survivors with
every `died_at` the line the row names, and the two predicate mutants are killed by the controls
written for them. Two LOW findings, both one-line closures, for round 9 alongside BE7-R1..R4 and
BE6-R1..R7.
