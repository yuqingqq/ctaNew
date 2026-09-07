# REVIEW 102 — the third composition: **GO E3 and GO E4 MAY PROCEED. GO #8: NO-GO.**

**Reviewer (pm-codex), 2026-09-07T11:2xZ. Read at `6c3a121` (`origin/mm-research-e3-composition`),
with my own worktree moved there via `wt_refresh.sh` and restored after — wt-de untouched.
Read-only: no heavy unit, no lock, nothing written under `data/`, never `--open`, no economic
value read. CHECKED = I went to the artifact or ran the code; AGREED = I read the same summary.**

> **GO E3 MAY PROCEED** and **GO E4 MAY PROCEED** at `6c3a121`:
> runner `eaa68ea55eadc4a07c9c6a28…`, early read `5aa544ef8d594efd807624fa…`,
> ledger `d78c370151cea431127bbb77…`, params **v19** `dd8db7ded9e6ed9723173a3a…`,
> design **v27** `3bcdf3c234cb7d4e4be116c2…`.
>
> **GO #8: NO-GO**, on one artifact — `de_multiday_gate1_runner.py`'s **real day path**, which
> supplies neither `ledger_anchor` nor `receipt_path`, so `assert_ledger_anchor` refuses
> `DECISION_LEDGER_HAS_NO_ANCHOR` before the day's work. **The guard is right; the day path was
> never given an anchor.** Tonight's 09-07 run would be the first real day under a
> ruling-carrying params, and it would refuse at the gate.

---

## §1 The composition verifies, and REVIEW 101's red is gone

**Exactly as prescribed** (full tree comparison, not a spot check):

```
paths differing from fe76d83 : 2
   live/pm_research/de_early_read.py             == ea06357's blob : True
   live/pm_research/de_multiday_gate1_runner.py  == ea06357's blob : True
path sets equal : True | every other path == fe76d83 : True
ea06357 is on origin/mm-research : YES   (DE 133)
```

**Ten cascade pins, 0 mismatched**, against this worktree's own root, and `verify_be_module`
ADMITS reporting `n_modules_checked 10`. `P3_design` holds (v27 pins v19). (**CHECKED**.)

**The check REVIEW 100 could not run and REVIEW 101 found red is now GREEN:**

```
de_multiday_gate1_runner --selftest   PASS -- 364 checks, 0 disarmed, 0 skipped   rc 0
                                      (32.5 s, 885,928 KiB)   -- 364 of 364 reached
de_early_read            --selftest   PASS --  25 checks, 0 disarmed, 0 skipped   rc 0
rehearse('2026-09-05')                READY, blocking [], G 4, EXPLORATORY,
                                      NONE_BELOW_FIVE_DAYS, is_a_full_pair True,
                                      bar_says == artifact_is (64 hex)
bar state                             read [09-03, 09-04] | unread [09-05, 09-06] | next 09-05
```

(**CHECKED**, all run by me at the real commit in a real worktree.)

## §2 DE 133's repair — driven four ways, and it records both of its own earlier failures

`assert_ledger_anchor` is a named function at `:5891`, called at `:6020`. I drove it:

```
1 no R-765 ruling                 owes=False,  no status
2 ruling + FIXTURE, no anchor     owes=False,  NO_LEDGER_FOR_A_FIXTURE_DAY     <- a named status
3 ruling + REAL day, no anchor    REFUSED DECISION_LEDGER_HAS_NO_ANCHOR
4 ruling + REAL day, with anchor  owes=True
```

(**CHECKED**.) **The placement claim holds at the code**: the call sits after
`assert_fixture_day_lock`, `assert_launch_form_at_runtime`, `assert_lock_form_at_runtime`,
`wrapper_observed()` and `assert_real_day_has_the_lock` — the guards it must not pre-empt — and
**before S0**, so a refusal costs nothing.

The docstring records both failure modes it went through, in DE's own words: DE 132 put it *"at
the ledger write — after ~90 minutes of draws — which is the shape R-610 exists to forbid"*, and
placed at the top it *"PRE-EMPTED the day-membership refusal and a cell testing that got this one
instead"*, with *"two versions of my own cell accepted THEIR refusal as if it were this one. A
check nobody can watch fire is not a check."* That is the right thing to leave behind: the
placement is now a stated constraint with both walls named, not a line someone moved twice.

## §3 The NO-GO for GO #8 — traced, then driven

**The real day path supplies no anchor.** Traced at the code (**CHECKED**):

```
_main_day                   -> day_split_residency_proof(day, book, params=params,
                                    fixture=..., n_days_complete=..., before_work=...)
                               ... no ledger_anchor, no receipt_path
day_split_residency_proof   -> run_day(day, book_path, params=params, **kw)
run_day :6020               -> assert_ledger_anchor(params, fixture=fixture,
                                    anchor=ledger_anchor if not None else receipt_path)
```

and **nothing anywhere in the module passes `receipt_path=` into `run_day`** — the only
occurrence of that name is the parameter's own default at `:5931`. So on a real day
`anchor` is `None`, `fixture` is `False`, and **params v19 carries `user_ruled_unsealed_emission`
(verified)** — the three conditions the refusal is written for. Driving exactly those arguments:
**`REFUSED DECISION_LEDGER_HAS_NO_ANCHOR`**.

**By contrast the early-read path is fine**, which is why E3/E4 clear: `de_early_read.py:414`
passes **`ledger_anchor=out`** — the artifact's own path — on the production call (`:946` and
`:964` are DE 132's green and red cells) (**CHECKED**).

**Why this has not bitten before, and why tonight is the first time.** GO #7 (the 09-06 day) ran
under params v15, which carries no R-765 ruling, so `assert_ledger_anchor` returns
`owes_a_ledger: False` on that path. The ruling arrived with v18. **Tonight's GO #8 would be the
first real day run under ruling-carrying params — and the first to meet this guard.**

**This is the guard working, not failing.** It refuses *before* the day's work, which is exactly
where DE moved it to; the cost of the NO-GO is a refused launch, not ninety minutes. **The fix is
small and is DE's:** `_main_day` already refuses without `--output` and already computes an
output directory, so it has an anchor to pass — it simply does not pass one. *(DE is being reset
at 98 % context; Q-DE-137 is its last row. This item should go to the next DE seat with the trace
above, since the reset means nobody currently holds it in context.)*

## §4 The re-pointed side branch — **not acceptable as it stands**, and the fix is cheap

The question is fair and the answer is measured, not stylistic:

```
5a34e722  reachable from any origin ref : NONE
9233b34   reachable from any origin ref : NONE
6c3a121   on origin/mm-research-e3-composition : YES
```

(**CHECKED**, scanned every `refs/remotes/origin/*`.) The two earlier heads survive **in my clone
only because I fetched them at REV 101**; on origin they are unreferenced and are gc candidates.

**And that already bites: REVIEW 101's NO-GO cites `5a34e722` by name, and that citation is
unresolvable from a fresh clone.** A review that names the commit it gated is the ordinary case,
not an unusual one — so a force-moved branch turns every superseded gate into a dangling
reference. This is the same property R-601 states for artifacts (*a cited artifact is locatable*)
applied to commits, and rule 12's `carrying_commit` depends on it.

**What is right and should not change:** re-composing after a NO-GO is correct, and keeping the
composition on a side branch rather than merging into `mm-research` is correct. **The defect is
only the force-move.** The fix costs nothing: give each attempt its own ref —
`mm-research-e3-composition-1/-2/-3`, or a tag per attempt — so every gate's citation stays
resolvable and the sequence of attempts is legible. **A carrying commit should be reachable for
as long as anything cites it, and reviews cite them by construction.**

---

## §5 HOLDS AND ROUTING

| id | artifact | what |
|---|---|---|
| **NO-GO** | `de_multiday_gate1_runner.py`'s real day path (`_main_day` → `day_split_residency_proof` → `run_day`) | **GO #8 blocked**: no `ledger_anchor` and no `receipt_path`, so `assert_ledger_anchor` refuses under v19's ruling. `_main_day` has an output directory and does not pass it |

**GO E3 and GO E4: MAY PROCEED** at `6c3a121`.

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE (next seat) | pass the day's output path as `ledger_anchor` from `_main_day`; the trace is in §3 | routed |
| 2 | coordinator | keep each composition attempt on its own ref or tag — `5a34e722` and `9233b34` are already unreachable and REVIEW 101 cites one of them | routed (§4) |
| 3 | DE | land the `ABSOLUTES_DO_NOT_RECONCILE` drive as a cell (REV 99 §A3, driven by me at REV 101 §4) | carried |

**Closed this round:** REVIEW 101's NO-GO — the runner's battery is green at the composition,
364 checks, and the fixture case is a named status rather than a refusal or a `null`.

## §6 WHAT I DID NOT ESTABLISH

- **Not established:** that GO #8's refusal would occur *in flight* — I traced the call chain and
  drove `assert_ledger_anchor` with the arguments that chain produces, but I did not launch a day
  run. The trace is three hops and each is quoted above; if any hop passes an anchor by a route I
  did not see, the NO-GO dissolves and I would want to be told so.
- **Deliberately not read:** every economic value in the early-read artifacts.
- **Process:** my worktree was moved to `6c3a121` and restored; wt-de untouched; nothing run that
  writes.
