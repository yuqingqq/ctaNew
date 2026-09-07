# REVIEW 100 (REV 99 part B) — E2's artifact, DE 132, DA 126, and the GO E3 gate

**Reviewer (pm-codex), 2026-09-07T11:0xZ. Read at `7ed070f` in `~/ctaNew-wt-rev`. Read-only: no
heavy unit, no lock, nothing written under `data/`, never `--open`. The early-read family is
USER-unsealed, so I censused MECHANICS — presence, shape, counts — and read no economic value.
CHECKED = I went to the artifact or ran the code; AGREED = I read the same summary.**

> **GO E3: NO-GO *at the tip as a refresh target*.** Refreshing wt-de to `ddda164`/`7ed070f`
> brings `de_phase4_diag_runner.py` at `309b98c7…` against v19's pin `ee4034c1…`; the runner's
> battery refuses `BE_CASCADE_DIFFERS`, and **the early-read CLI runs that battery *inside* the
> launch** — so E3 would abort at `before_work` **after taking the lock**.
>
> **The block is not DE 132.** DE 132 touches **no pinned cascade module**. A tree carrying
> `fe76d83`'s cascade **plus DE 132's two files** has **0 cascade mismatches** and a green
> early-read battery — I built it and checked. What drags in the mismatch is DE 129/130's
> unrelated edits to the phase4 module. **Clear that composition, not the tip.**

---

## §1 E2's artifact — censused by KEYS, and it confirms two earlier repairs in production

`p003_de_early_read_day_20260904__20260907T105906Z.json`, sha256 `b196bf32fe54c9184c98aa90…`
(**CHECKED**, recomputed). 15 top-level keys.

```
day_run.decision_ledger                       = null      <- and NO sibling reason field
ledger files for 09-04 on disk                = NONE
arm-days                                      = 2, both sealed=False, economic present
  econ keys (both): D_E0, Z, p_location, null_mean, null_sd, null_draws_summary
  `absolute` block                            = ABSENT on both   (DE 131 postdates E2)
day_run.status                                = DAY_RUN_UNSEALED
day_run.what_this_is_not.the_economics_are_SEALED = False
G_and_which_G_it_is                           = present (design_G_from_params 6, n_days_complete_at_this_emit 4)
is_a_validation false | G 4 | EXPLORATORY | NONE_BELOW_FIVE_DAYS
```

**Two of my own earlier findings are now closed at a REAL artifact rather than in a fixture.**
The 09-03 artifact said `DAY_RUN_SEALED` and `the_economics_are_SEALED: true` while carrying
unsealed arm-days with economics present (REV 95 §A3); **09-04 says `DAY_RUN_UNSEALED` and
`False`** — DE 126 phase 2's computed fields, working in production. And
`G_and_which_G_it_is` is present, which is REV 90 §A0(3)'s repair, likewise in production.

## §2 The unreachability, established at the code — and the stated intent was not what the code did

At `fe76d83` (the bytes E2 ran):

```
de_multiday_gate1_runner.py :6126   if _ruling765 and _ledger765 and receipt_path is not None:
de_early_read.py            :346    RUN.run_day(day, book, params=..., fixture=False,
                                                n_days_complete=..., early_read=..., before_work=...)
                                                                        ^ no receipt_path
```

(**CHECKED**.) `receipt_path` is `None` on the early-read path, so the ledger write is
**unreachable** — MEM 267's independent reading and the coordinator's notice both hold, and I
have now established it myself at both ends.

**And the sharper half.** Two lines above that guard the code says: *"A day whose ledger cannot
be written REFUSES — the ruling is that the numbers are kept, and a receipt promising a ledger
that is not there would be worse than no promise."* **The guard is an `if`, not a refusal.**
When `receipt_path` is `None` the block is silently skipped and `decision_ledger` is emitted as
`null` **with no status beside it** — an absence carried as a value. So the module's own stated
intent and its behaviour on this path had parted company, which is why **two runs (E1 and E2)
went by unnoticed**: nothing was ever going to say so.

## §3 DE 132 — the right fix, at both ends

```
de_early_read.py    ledger_anchor=out            (the artifact's own path, so the ledger lands beside it)
                    EARLY_READ_WROTE_NO_DECISION_LEDGER   -- the wrapper refuses on a null/blank block
de_multiday_gate1_runner.py
                    DECISION_LEDGER_HAS_NO_ANCHOR         -- the runner refuses instead of skipping
GREEN cell : with an anchor the ledger is written BESIDE the artifact; path + sha256 + rows +
             schema in the block, `Path(...).is_file()` and the parent directory asserted
RED cell   : with NO anchor the run REFUSES BY NAME instead of emitting `decision_ledger: null`
```

(**CHECKED** at the diff.) **Both ends is the right choice**: the runner no longer skips
silently, and the wrapper double-checks what it got back — so neither a future caller that
forgets the anchor nor a future runner that stops writing can reproduce the silent form. The
refusal's own words name E1 and E2 as the cases that went unnoticed, which is the disclosure
rule 4 asks for.

**`de_early_read --selftest` = PASS 24 checks, 0 disarmed, 0 skipped** in the composition E3
would run (**CHECKED**).

## §4 The GO E3 gate, measured

**What refreshing wt-de to the tip would do:**

```
v19's cascade pin vs the tip:  1 of 10 mismatched
   de_phase4_diag_runner.py    pinned ee4034c15c274982   actual 309b98c7a1045d1d
runner battery at the tip:     rc 1  (BE_CASCADE_DIFFERS)
the early-read CLI runs `before_work=lambda: RUN.selftest(...)` INSIDE the launch
   -> E3 aborts at before_work, after taking the lock, before doing any work
```

(**CHECKED**.)

**What the composition E3 needs looks like.** DE 132 touches only `de_early_read.py` and
`de_multiday_gate1_runner.py`; **neither is among the ten pinned modules** (**CHECKED**). I
built `fe76d83`'s tree with DE 132's two files substituted:

```
cascade pins mismatched: 0
de_early_read --selftest: PASS 24 checks, rc 0
```

**So the clean path is narrow and available**: a commit carrying the frozen cascade plus DE 132.
The alternative — params v20 re-pointing the cascade — is foreclosed by the standing version
freeze.

**What I could NOT verify, and it must be verified before the GO.** I could not run the
*runner's* battery in that composition: no such commit exists, and my scratch tree is a
`git archive` extraction, so `6_producing_code_is_locatable` fails there for want of git history
— **a scratch artefact, not a property of the composition** (the same class that made
`de_phase4_diag_runner` refuse out of an archive at REV 93 §B3). **The runner's battery must be
run at the real commit, in a real worktree, before E3 launches** — it is the battery the launch
itself runs.

## §5 DA 126 — the absence is a named status, and the reader refuses to approximate

DA's verdict on the **real** 09-04 artifact, obtained by calling `check_decision_ledger()` on it
(a pure read; I ran no emitting mode):

```
status                     : LEDGER_ABSENT
says                       : `day_run.decision_ledger` is present and NULL: no ledger was
                             written on the early-read path for this day
what_cannot_be_derived     : the 0-cancel BASELINE's own value for this day. The arm blocks
                             carry D(E0) -- a DIFFERENCE against that baseline -- and a
                             difference does not contain either of its terms
never_approximated         : this reader does not reconstruct the baseline from the arm blocks,
                             the fill counts or anything else. An approximation printed in a
                             table is read as a measurement
and_the_table_still_prints : True
```

(**CHECKED**.) Three things are right here and worth naming:

1. **`LEDGER_KEY_ABSENT` and `LEDGER_ABSENT` are separate statuses** — *"the artifact carries no
   key at all — not even a null. That is a different fact from a null block and is named
   separately."* A missing key and a null value fail differently and now read differently.
2. **`what_cannot_be_derived` states exactly why R-782 exists**: a difference does not contain
   either of its terms. That is the same sentence from the other side of the ledger.
3. ***"An approximation printed in a table is read as a measurement"*** — the strongest line of
   the round, and the reason the table prints `LEDGER_ABSENT` rather than a reconstructed
   baseline. The table still prints; only the line that cannot be computed is withheld.

**Did DA recompute D_E0/Z from the ledger? No — and it says so by name.** There is no ledger to
recompute from, and DA reports that as a status rather than falling back to the arm blocks
(**CHECKED**; no recompute path is reached, and the reader carries no reconstruction). That is
the right answer to the question, and it means R-765's *store the numbers* has not yet bought
anything on the early-read path — which DE 132 is what fixes.

DA's battery: **23 checks, 0 failures** (**CHECKED**, run by me).

---

## §6 HOLDS AND ROUTING

| id | artifact | what |
|---|---|---|
| **NO-GO** | `de_multiday_gate1_params_v19.json`'s `be_cascade` pin for `de_phase4_diag_runner.py` (`ee4034c1…`) against the tip's `309b98c7…` | **GO E3 blocked at the tip as a refresh target.** Clear instead a commit carrying `fe76d83`'s cascade + DE 132's two files, and **run the runner's battery there first** |

| # | to | finding | kind |
|---|---|---|---|
| 1 | coordinator | the composition to clear for E3 is not the tip; DE 132 is not the blocker, DE 129/130's phase4 edits are | routed (§4) |
| 2 | DE | `ABSOLUTES_DO_NOT_RECONCILE` still never driven (REV 99 §A3), before GO #9 | carried |

**Closed this round:** the E2 artifact confirms REV 95 §A3 (`DAY_RUN_UNSEALED` / `False`) and
REV 90 §A0(3) (`G_and_which_G_it_is`) **in production**; DE 132 converts the silent skip into a
refusal at both ends; DA 126 names the absence instead of approximating past it.

## §7 WHAT I DID NOT ESTABLISH

- **Not established:** the runner's battery in the composition E3 needs — no such commit exists
  and a `git archive` tree cannot run it (§4). This is the one thing standing between the NO-GO
  and a clearance.
- **Deliberately not read:** every economic value in either early-read artifact. I censused
  presence, shape and counts only.
- **Process:** my Q-REV-99 row was landed by the coordinator at `7002edc` together with
  Q-MEM-255 after each blocked the other's legacy-form landing (R-784). This round's row uses
  the new `--row` locked-insertion form.
