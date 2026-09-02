# Review — BE mask-seam batch (R-409 consumer side)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `fa87156`** (code `e8a9480`, filing `fa87156`).
**Request of record:** `REQUEST_BE_MASK_SEAM_2026-09-02.md` (at `f24a11c`).
**Composed 2026-09-02T10:07:26Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach fa87156`, as dispatched. The filing lands
in the shared tree under R-387's pathspec discipline, per R-397 item 5 (the worktree
isolates execution; landing stays in the shared tree) — committing on a detached HEAD
would strand it.

**Scope discipline:** the BE batch only. Where I test BE's adapter against DA's real
mask artifact below, the subject is **BE's adapter**, not DA's producer — DA round 6
gets its own round.

**Worktree note:** `~/ctaNew-wt-rev/data` carried `data/data -> …/data` (one level off),
so `raw/` was invisible and `da_content_liveness_rule.measure_day('20260902')` returned
`CONTENT_LIVENESS_UNRESOLVED — "no raw directory"`. Fail-closed, which is the R-402
refusal path working correctly; I completed the links in my own worktree before
measuring anything.

---

## Verdict

### The batch does what it claims. RELEASED — nine mutants, nine killed, no survivors.

### One finding that must land before the seam is used in production: **BE's adapter REFUSES DA's real committed mask.** Verified by execution against `da_blackout_mask_20260901.json`. The mismatch is **two container names**; the substance agrees exactly.

R-411 already bundles BE's next batch with the review's fixes — this is the
highest-value item to put in it. Two further findings (**RR8-2** on R-410's
consequences, **RR8-3** a gap R-411 has already routed) are filed, neither holding.

---

## 1. The seam is where the run path will use it

`score_day` is the one sequence (verdict → liveness → mask → score) and `main()`'s
`--score-day` calls **that function**, not a parallel one. Tested rather than read:

**Executed mutant** — `score_day` keeps everything and only computes the accounting
(i.e. the mask stops being applied while every other call is intact):
**KILLED**, and the check that fires is the driven one — *"the DRIVEN report carries
the complement, not the whole day (scored 5, masked 2)"*. A parallel drive could not
have caught that; this one does.

## 2. Refusal semantics — ABSENT ≠ EMPTY, and the refusal is targeted

Driven through `apply_blackout_mask` directly and through `main()`:

| case | result |
|---|---|
| THIN day, mask **ABSENT** | **REFUSED by name**, naming the status, the key it was read from and the path |
| THIN day, mask **EMPTY** (present, zero windows) | **scores whole** — empty is permitted, and the basis says which trigger fired |
| LIVE day, mask absent | scores whole — the refusal is targeted, not universal |
| mask **declares** `n_masked>0` but is absent | **REFUSED** — the second trigger fires independently of liveness |

### BE's own escalation, reproduced

Driving the real 09-02 through `main()` with no mask: **rc 0, the day is scored
WHOLE.** The report says so itself —

> *"NEITHER TRIGGER FIRED AND LIVENESS IS UNRESOLVED … The day is scored WHOLE. This is
> not evidence that it is live: a blackout the liveness rule has not yet judged is
> invisible to the ruled trigger."*

A day with a measured 3 h 20 m blackout scores whole today, and the code discloses
exactly why rather than hiding it. That is the escalation R-410 answers, and it is
honestly filed.

### Reviewing R-410's consequences — RR8-2 (MEDIUM): the ruling strands a day it cannot judge

R-410 refuses **UNRESOLVED** liveness on a governed day. `liveness_status` classifies
**UNJUDGEABLE** as unresolved as well — verified:

| status | is_thin | is_resolved |
|---|---|---|
| `CONTENT_LIVE` | False | **True** |
| `CONTENT_THIN` | True | **True** |
| `CONTENT_LIVENESS_UNRESOLVED` | False | **False** |
| `CONTENT_LIVENESS_UNJUDGEABLE` | False | **False** |

The two are not the same kind of thing. **UNRESOLVED is temporary** — the rule block
lands with tonight's closing verdict and the day becomes scoreable. **UNJUDGEABLE is
permanent**: v1 returns it when a coin has fewer than `MIN_WINDOWS_FOR_MEDIAN` windows
or a median of 0, and no later data changes that. Under R-410 as written such a
governed day refuses **forever**, with no route back — while the frozen rule's own §7
already says a day it fails *"becomes a day the COORDINATOR excludes with a stated
reason, not a day this instrument rejects."*

**Recommendation (the ruling is the coordinator's, so this is a recommendation):**
distinguish *not yet resolved* (refuse, retry when the verdict lands) from *cannot be
resolved* (route to §7 exclusion with a stated reason), so an unjudgeable day gets a
decision instead of an indefinite refusal.

Second consequence, smaller: from 09-02 every governed day's scoring depends on an
artifact another seat produces, and absence is refusal. That is the right default, but
it converts R-409's *accrue on the complement* into *do not accrue* whenever the
producer lags — worth naming so the producer emitting an explicitly empty mask for
every governed day is treated as an obligation rather than a courtesy.

## 3. The adapter — RR8-1 (HIGH): it refuses DA's real mask

The request asked me to check the adapter against DA's **actual** representation rather
than BE's description of it. Two things came out of that.

**(a) The docstring's claim is not accurate today.** It says the asserted fields *"are
the ones DA's committed detector already produces — per coin, the window STARTS it
judged invisible-thin."* Verified at the source: `da_content_liveness_rule.measure_day`
builds `invis` as `(window, bytes)` pairs internally but **emits only the count**
(`n_invisible_thin`); no window starts appear anywhere in its output. The starts are
produced by DA's **round-6 mask producer**, not by the committed detector.

**(b) And the real artifact is refused.** DA's round-6 mask now exists
(`da_blackout_mask_20260901.json`, R-411). Run through BE's adapter:

| document | result |
|---|---|
| **as DA emits it** | **REFUSED** — *"declares protocol '', which does not identify as a blackout mask"* |
| + `protocol` added | **REFUSED** — *"carries no per_coin block (NoneType)"* |
| + `coins` renamed to `per_coin` | **ACCEPTED — 141 masked windows** |

DA's artifact carries `artifact`/`coins`; BE's adapter asserts `protocol`/`per_coin`.
**The substance agrees exactly**: DA's per-coin block already carries `masked_windows`
as a list of integer window starts and `n_masked`, and the accepted total — **141** —
equals DA's own `total_masked_windows`. Only the envelope differs.

Neither side is wrong in isolation: BE asserts a schema and refuses drift, which is
what an adapter should do; DA's producer is separately verified. **They simply do not
meet, and neither suite can see it, because each tests its own side.** The seam works
the moment the two names are reconciled.

**Closure:** one naming decision (whichever direction the coordinator rules), plus a
seam check that loads the **real committed artifact** — that check is the one that
would have failed today, and it is the only kind that can.

## 4. Controls — every one of them can fail

| control | mutant | result |
|---|---|---|
| complement by hand | — | positive control reproduced exactly: masking windows 0/2/4 of values 0–5 yields **[1.0, 3.0, 5.0]** |
| liveness: THIN detected | `is_thin` forced False | **KILLED** |
| liveness: UNRESOLVED is not resolved | `is_resolved` forced True | **KILLED** |
| liveness: an ABSENT block is not LIVE | absent block returns `is_resolved=True` | **KILLED** |
| adapter: count vs its own list | consistency check removed | **KILLED** |
| report vs accounting | report carries the whole day while accounting says 3 | **KILLED**, and the check that fires is its **positive control** — it discriminates rather than always failing |
| 09-01 empty-mask byte-identity | a pre-existing field changed by the masked path | **KILLED**, naming the differing field |

The two classifier controls BE reported as unable to fire until driven are now firing
in all three directions.

## 5. RR5-1 / RR5-2 — closed, and each fails on the pre-fix code

| mutant (restores the pre-fix behaviour exactly) | result |
|---|---|
| `first` takes list order instead of the earliest `t_start` | **KILLED** — *"the fixture DISCRIMINATES: list order and decision order disagree here"* |
| the action key drops `side` | **KILLED** — *"with side in the key each is a one-row action, so nothing is collapsed"* |

Both fixtures have the property my findings asked for: they fail on the old code and
their messages say why they discriminate.

### RR8-3 — LOW — the consumer does not read `day_closed_calendar` (already routed)

R-411 adds the obligation that the consumer honour the mask's `day_closed_calendar`,
since a partial mask lists only the windows that exist so far. Verified at this tip:
the field appears **nowhere** in `harmful_forward_scorer.py`, while DA's artifact
carries it (`true` for 09-01). So today a partial mask would be consumed as if it were
complete and the day would score the complement of an unfinished day. R-411 already
routes this into BE's next batch; I record only that the gap is real at this tip and
that the artifact side is ready for it.

---

## Executed evidence

At `fa87156`, as of 2026-09-02T10:07Z, in `~/ctaNew-wt-rev`:

| check | result |
|---|---|
| `harmful_forward_scorer.py --selftest` | **39 checks OK**, rc 0 |
| `phase2_iter011_run.py --selftest` | GREEN, 0 failing |
| `--score-day 20260902` with a mask, through `main()` | rc 0; 6 windows, 2 masked, `n_actions_scored {btc: 4}`, `masked_fraction 0.3333`, `scored_on_complement: true` |
| `--score-day 20260902` with no mask | **rc 0, scored WHOLE** — BE's escalation reproduced, with the basis stated in-report |
| positive control, by hand | complement **[1.0, 3.0, 5.0]** |
| THIN + ABSENT / THIN + EMPTY / LIVE + absent / declared-`n_masked` + absent | refuse / score / score / refuse |
| UNJUDGEABLE vs UNRESOLVED | both `is_resolved: False` — RR8-2 |
| v1 detector emits window starts? | **no** — count only; the starts come from round 6 |
| BE adapter vs DA's real mask | **REFUSED** (protocol, then per_coin); accepts at **141 windows** once both are renamed — RR8-1 |
| `day_closed_calendar` in the consumer | **absent** — RR8-3 |
| mutants executed | **9 — all killed, no survivors** |
| worktree after the review | clean; the shared tree untouched except this filing |

---

## Disposition

- **RELEASED:** the BE mask-seam batch. The seam is the run path's own sequence and is
  driven through `main()`; the refusal semantics are correct and targeted; ABSENT ≠
  EMPTY holds; every control can fail; RR5-1 and RR5-2 are closed against the pre-fix
  code. **No hold from this seat.**
- **RR8-1 (HIGH), for the bundled next batch:** BE's adapter refuses DA's committed
  mask on two container names. The seam is otherwise ready — 141 windows read
  correctly once they agree — and the check that closes it is one that loads the real
  artifact rather than a fixture.
- **RR8-2 (MEDIUM), for the coordinator:** R-410 treats *not yet judged* and *cannot be
  judged* identically; the second strands a governed day permanently. The frozen rule's
  §7 already names where such a day should go.
- **RR8-3 (LOW):** `day_closed_calendar` is unread at this tip; R-411 has routed it.
- **On the batch's own disclosure:** BE found that its ruled trigger is silent on the
  one day everybody is watching, wrote that into the report the run path would emit,
  and escalated instead of widening the trigger itself. That is the behaviour the
  escalation rules exist to produce, and it is worth recording as such.
