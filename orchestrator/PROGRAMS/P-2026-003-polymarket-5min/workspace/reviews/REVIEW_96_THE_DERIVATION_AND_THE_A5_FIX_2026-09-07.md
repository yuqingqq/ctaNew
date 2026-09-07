# REVIEW 96 — the phase4 count, REV 95 §A5's fix, the two minors, and the held re-point

**Reviewer (pm-codex), 2026-09-07T09:4xZ. Read at `2bc4e4d`, and at `fe76d83` (the frozen
runs' bytes) by refreshing my own worktree to that ref and back — wt-de untouched. Read-only:
no heavy unit, no lock, nothing written under `data/`, never `--open`, no economic value read.
CHECKED = I went to the artifact or ran the code; AGREED = I read the same summary.**

> **GO E4 MAY PROCEED on this cell** — but **REV 95 §A5 is NOT closed**, and I drove it: the
> fix widened the `ok()` and added an all-done branch, then left the cell below it calling
> `rehearse(next_unread)` unconditionally. After E4 that raises the *identical*
> `KeyError: 'preconditions'` that was REV 94's NO-GO. E4 launches because at its launch an
> unread day still exists; the module's battery aborts the moment E4 finishes.

---

## §1 The by-design red, and the frozen tree

**At the tip the refusal names exactly one module** — the widening earning its keep:

```
BE_CASCADE_DIFFERS: 1 of 10 cited cascade modules do not match their declared pair --
[{'path': 'live/pm_research/de_phase4_diag_runner.py',
  'declared': 'ee4034c15c274982…', 'actual': 'ce9cc466782fb8f6…', 'status': 'DIGEST_DIFFERS'}]
```

**At `fe76d83` — the bytes E2/E3/E4 launch from — everything is green:** runner
**PASS 362 / 0 / 0**, early read **PASS 21 / 0 / 0** (**CHECKED**, both run by me in my own
worktree moved to that ref). And the frozen tree is *self-consistent*, which is the fact that
matters:

```
de_phase4_diag_runner.py @ fe76d83 (what wt-de holds) : ee4034c15c274982…   == v19's pin
de_phase4_diag_runner.py @ the tip (DE 127's landing) : ce9cc466782fb8f6…
```

**So DE 127's shared-tree landing does not reach E3/E4: they read code from wt-de, whose copy
still matches v19, so `verify_be_module` admits there** (**CHECKED**).

**And the freeze is load-bearing in a way worth naming.** The early-read CLI on a real day runs
**both batteries inside the launch** — `selftest(quiet=True)` at :33 and
`before_work=lambda: RUN.selftest(...)` at :35. So a wt-de refreshed to the tip would abort
**E3 and E4 at `before_work`** on the by-design red. R-775's freeze is not only about
provenance; it is what keeps the launch's own battery green. That is worth saying out loud
before anyone refreshes wt-de out of habit.

## §2 (1) The phase4 count — **the accounting is now honest; the derivation is still in a comment**

**The claim reproduces, and I nearly filed a false discrepancy against it.** My first AST count
found **176** `ok(` sites against DE's 209 and I went looking for the convention rather than
reporting a gap: the module has **two** check helpers, and

```
ok( call sites 176  +  refuses( call sites 33  =  209      <- DE's number, exactly
```

(**CHECKED**.) `209 sites + 7 executions from sites inside loops = 216 = n_run 212 +
n_conditional 4` is arithmetically sound.

**But it is not yet the property R-771 asked for.** The cell is

```python
_conditional_sites = { 5711: "…", 5722: "…", 5745: "…", 5765: "…" }   # typed
_n_conditional = len(_conditional_sites)
ok(n[0] + 1 + _n_conditional == EXPECTED_CHECKS, …)                    # EXPECTED_CHECKS = 216, typed
```

**Nothing in the code computes 209 or 7.** The assertion is an identity between an observed
count and two typed constants: add one check and `n[0]` becomes 213, the cell goes red, and the
repair is to edit `216 → 217` — **which is adjusting the constant to the observation, one
addend richer.** The derivation lives in the comment; the code carries the answer.

**What would make it the property, and I can name it because I just did it:** count the `ok`
and `refuses` call sites from the AST at run time — 176 + 33 — and assert
`n_run + n_conditional == n_sites + n_loop_extra`. Then adding a check moves *both* sides and
nobody edits a constant. That is the same move DA made for the importer census, and it is
available here for the same reason: the sites are in the file.

**In fairness, what did improve is large.** "~37 unreached checks", reported twice, was wrong
both times; it is now an enumerated four with their conditions named, and the 40 are correctly
identified as a constant that had outrun the code. That is a real advance in the accounting
even though the constant is still typed.

### Are the conditional four unreachable-by-design? **Yes in substance — and the recorded line numbers are wrong**

There are **exactly four** `ok(False, …)` arms in the file, and each is the failure arm of a
known-bad `try/except`, reachable only if the guard under test stops refusing. That is
unreachable in a passing run and reachable in a failing one, which is the whole idiom. The
identification is right.

**But `_conditional_sites`' keys point at the wrong lines — uniformly three early:**

```
recorded : 5711, 5722, 5745, 5765
actual   : 5714, 5725, 5748, 5768        (the ok(False, …) arms, by AST, at the landed bytes
                                          and at 61569a3 alike)
line 5711 is `_fake.write_text('{"planted": "not BE\'s bytes"}')`  -- a line that DOES execute
```

(**CHECKED**.) So the dict's only locator points at executing lines. The claim survives — four
arms, correctly characterised — but a reader who follows the numbers lands three lines short of
each, and **a line number is a literal that must track a moving thing**: the next edit above
those blocks moves them again. The AST already yields exactly four; deriving them fixes the
off-by-three and the drift in one stroke.

## §3 (2) REV 95 §A5 — **the fix does not hold, and I drove it**

The `ok()` at :739 was widened to `bool(_st["read"]) or bool(_st["unread"])` — always true —
and an all-done branch was added for the case `next_unread is None`. **Then the cell below it
was left unconditional:**

```python
if _st["next_unread"] is None:
    …the all-done branch, which does assert every read day refuses…
    _st_conditional_note = "the READY half is not reachable: no day is unread"
else:
    _st_conditional_note = None
…
reh = rehearse(_st["next_unread"])          # :767 -- runs whether or not next_unread is None
dc = reh["preconditions"]["digest_comparison"]   # :768 -- unguarded
```

Driven at the tip:

```
rehearse(None) -> {'status': 'NOT_READY', 'blocking': ['EARLY_READ_DAY_NOT_IN_THE_BAR']}
                  'preconditions' present: False
the nested read RAISES: KeyError: 'preconditions'
```

(**CHECKED**.) **That is the identical traceback that was REV 94's NO-GO**, in the same module,
one line further down, after a round whose purpose was to remove it. The all-done branch even
sets `_st_conditional_note` to say the READY half is unreachable — and then the code takes it
anyway.

**Does it gate GO E4? No — E4 launches.** At E4's launch the ledger holds read `[09-03, 09-04,
09-05]` and unread `[09-06]`, so `next_unread` is `2026-09-06`; both the tip's cell and the
frozen tree's older cell pass, and E4 runs its two batteries green. **GO E4 MAY PROCEED on this
cell.** What is blocked is the state *after* E4: the module's battery aborts from that moment,
and the abort is a traceback rather than a named check — so the module cannot report its own
state, which is the property REV 94's NO-GO was about.

**Closure:** put the READY drive inside the `else:` branch it already has. One indent.

## §4 (3) The two minors — both closed, one clause short

**`ruling.path` is repo-relative from now on**, with `path_is: "REPO-RELATIVE"`, and the landed
artifact is corrected **in band by a note that travels in every future emission**:

> *"…names its ruling at an ABSOLUTE path into `/home/yuqing/ctaNew-wt-de`, the worktree that
> produced it. That artifact is LANDED and is not edited (rule 13); this note is the
> correction, in band, and every emission from here writes the path repo-relative (REV 95)."*

(**CHECKED**.) That is the right mechanism — a correction that reaches a reader of the *family*,
not just a reader of the code.

**`verify_be_module` now names what it checked**: `n_modules_checked`, `modules_checked`, and a
`scope` string distinguishing one module from the cascade (**CHECKED** at the code; the tip's
call refuses by design, so I read the shape rather than a return value).

**One clause short, and it is my REV 95 routed item 2.** The NOTE covers the *path* only. The
landed 09-03 artifact also carries `status: DAY_RUN_SEALED` and
`the_economics_are_SEALED: true` against arm-days that are `sealed: false` with economics
present — the defect DE 127's own battery message describes (*"The landed 09-03 early read said
DAY_RUN_SEALED"*). That correction lives in a battery message and code comments; **the note
that travels with the artifacts does not carry it.** One clause in the same NOTE closes it, and
the mechanism is already built.

## §5 (4) What REV will need to see when the freeze lifts

The composed v20/v28 are unwritten, so nothing to verify yet. When they land I will check, in
this order, and none of it is new law — it is what the last two rounds established:

1. **params v20 from v19 by the pair**, and **design v28 from v27 by the pair**, both digests
   recomputed from disk; `PARAMS_REL` → v20; **`P3_design` holds** (v28 pins v20).
2. **All ten cascade pins equal their blob AT THE LANDING COMMIT and the disk** — the check
   that caught the uncommitted-bytes case at REV 94, and the one that matters most here because
   the module being re-pointed is the one that moved.
3. **The re-point justified by `be_module_repoint`'s own two axes** across DE 127's change to
   `de_phase4_diag_runner.py` — draw-path functions and module-level assignments/pinned
   constants, compared by AST, not by reading the diff. I will measure them independently as I
   did at REV 93 §A6.
4. **The shared tree's runner battery GREEN** — the by-design red gone, and the refusal
   reachable again (a guard that no longer fires must be shown to still fire).
5. **One thing new to this re-point, and it is the reason to be careful:** the module being
   re-pointed **has a failing selftest** — the pre-existing R-499 admission I bounded at
   REV 93 §B3, still failing at the tip. A re-point says *these are the bytes the null runs
   through*. DE should state whether that failing suite bears on the null, or say explicitly
   that it does not and why. Pinning bytes whose own suite is red is defensible; doing it
   silently is not.
6. **wt-de refreshed only after the last early read's receipt lands** — §1's point: the freeze
   is what keeps the launch's own battery green.

---

# §6 HOLDS AND ROUTING

**No holds.** GO E4 may proceed on the §3 cell.

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | **REV 95 §A5 is not closed**: `:767–768` runs unconditionally and raises `KeyError: 'preconditions'` after E4 — the same traceback as REV 94's NO-GO. Put the READY drive inside the `else:` it already has | routed, **re-opened** |
| 2 | DE | the phase4 count is still typed: derive it from the `ok`/`refuses` call sites (176 + 33 = 209) so adding a check moves both sides | routed (§2) |
| 3 | DE | `_conditional_sites`' four line numbers are three early and point at executing lines; derive them from the AST | routed (§2) |
| 4 | DE | extend the landed-artifact NOTE by one clause — the two false seal fields, not only the absolute path | routed (§4) |

**Closed this round:** REV 95's two minors (repo-relative path with an in-band note that
travels; `n_modules_checked` / `modules_checked` / `scope`); and the by-design red confirmed to
name exactly one module.

# §7 WHAT I DID NOT ESTABLISH

- **AGREED, not established:** DE's line-tracer measurement itself (205 of 209 executing) — I
  verified its *conclusion* (four `ok(False,…)` arms, exactly) and its *arithmetic* (209
  reproduces), not the tracer run; the 7 loop executions.
- **Not established:** E2's artifact and DA 126's read (REV 97, after ≈ 10:55Z); the R-499
  admission's bearing on the null.
- **Method note:** my first count of the check sites was 176 against DE's 209, and I went
  looking for the convention instead of filing the gap — the module has two check helpers and
  they sum exactly. Third round running that this habit has stopped a false finding; it is
  cheaper than the correction would have been.
