# Review request — R-496 (C)+(D): the free-read premise and the pre-declaration, BEFORE any number exists

requested by: coordinator (pm-co) · 2026-09-03T04:02Z · reviewer: pm-codex
pinned tip: **`6badca0`** (R-496 at `70d2b45`, doc corrections at `6badca0`). Read `COORDINATION.md` R-496 and `RESULTS.md` §3 and §7.

**Why this and not BE round 12.** BE round 12's review is QUEUED and waits (R-377). This round produces the first economic number this programme has ever opened, and a declaration can only be strengthened *before* the result — afterwards every change looks like post-hoc adjustment. BE is executing now, so this filing is time-critical. **BE's results do not exist yet and you must not wait for them.**

## The two claims under review

**Claim 1 — the free-read premise (R-496 (C)).** That opening 08-29 costs the forward race **zero days**, because 08-29 is post-freeze (`post_freeze_pass` true; freeze epoch `1787897340` = 2026-08-28T06:09:00Z), era-pure (`eras_touched ["clob_v3_1"]`, `boundaries_inside_day []`), day-quality **passing**, and yet permanently non-accruable via `era_admissible: false` / `race_admissible_by_era: false` — for a *ruled collector property*, not a measured property of the feed.

**Attack it directly.** Is there ANY path by which 08-29 could later become race-admissible — an era re-ruling, a re-stamp, a `collector_runs.jsonl` correction, a change to the admissible-era set, a per-coin split, a superseding verdict? The verdict carries a `supersedes` field; 09-01 and 09-02 both have superseded predecessors on disk, so verdicts in this programme demonstrably do get rewritten. **If 08-29 can ever accrue, then opening it DOES spend a race day and the premise of this round is wrong.** Verify at `da_dayverdict_20260829.json`, at the collector-run ledger, and at the ruling that fixes the admissible-era set — not at R-496's account of them.

**Claim 2 — the pre-declaration (R-496 (D)) is complete.** It fixes: population (08-29 primary, 08-30 labelled secondary, never pooled); decision metric net cents vs the INCUMBENT, de-duplicated to actions, tranches valued at their own time and level and only after t+L; secondary rho = adverse/spread; control matched on action count, side, hour and compared on the decision metric; both nulls at minimum 200 draws, target 2,000 sign-flip permutations; cluster disclosure ruled-unit UTC day, G=1, unit used window, `weaker_than_ruled` true, `intervals_claimable` **false**, point estimate and NO interval; cell count declared before the read with Holm over it; skill incremental to the incumbent, never a base rate; exclusions as statuses with counts; predicates computed, no verdict strings; and 08-29/08-30 **named as consumed** with no parameter, threshold, horizon, budget, feature subset or candidate choosable on them.

## Items

1. **Every remaining degree of freedom.** After this declaration, what can BE still CHOOSE once it has seen the numbers? Enumerate them — that list is the review. Anything choosable post-hoc is a rule-11 hole whatever the declaration says elsewhere.
2. **The `clob_v3_1` caveat — is it merely a caveat?** R-496 calls 08-29 "two collector generations behind" the race's `clob_v4_1` and asserts nothing about direction. Establish at the collector-run ledger and the deploy runbooks (`V41_DEPLOY_RUNBOOK.md`, `V5_APPLICATION_HEARTBEAT_REPAIR_2026-08-31.md`, `O1_DEPLOY_RUNBOOK_*`) **what actually changed across v3_1 → v4 → v4_1, and whether any of it touches the fields the decision metric consumes.** If a change touches those fields, the read is not "a different surface" — it may be **uninterpretable**, and that is a HIGH finding, not a footnote.
3. **Is "named as consumed" enough?** Rule 11 says seen days are consumed. Does the declaration need to name which future analyses are barred from 08-29/08-30, or does naming the days suffice? State the failure mode you have in mind.
4. **The nulls.** Iteration 011's entire non-Q4 surviving set sits at the 1/501 floor. Is "minimum 200, target 2,000" the right construction here, and is a sign-flip permutation the right null for a ONE-day window-clustered population — or does the within-day dependence make it optimistic in a way the disclosure does not cover?
5. **The seal boundary.** The declaration opens 08-29/08-30 and keeps 09-01/09-02 sealed, with hard separation by outdir, receipt and protocol. Can an opened number leak into a sealed run's receipt, or the reverse, given one driver produces both? Name the mechanism if so.

## Discipline

Execution only in `~/ctaNew-wt-rev` at `--detach 6badca0` or reads at the blob. `~/ctaNew-wt-be`, `-da`, `-de` never read; `be_forward_day.py` **never run**; nothing under `data/` beyond reading; **no sealed file opened** — not `~/ctaNew_sealed_backup/`, not the `/tmp` original, not any relocation BE makes; the Phase-4 OUTDIR never passed to `--run`; no unit, timer or anchor; `DA_MIDNIGHT_MODE` never set; no plan file edited; `git worktree list` unchanged at quiescence (34).

## Disposition asked for

ONE filing (R-377), `REVIEW_R496_PREDECLARATION_2026-09-03.md` under `workspace/reviews/`, one pathspec commit, push. Findings numbered R496-Rn. Item 1 as an enumeration. Claim 1 as **HOLDS / DOES NOT HOLD**, and if it does not hold, say so first and loudly — the round is built on it. Overall as **RELEASE** (the declaration stands) or **AMEND** (with the exact clause). You estimate; the coordinator routes; the USER decides (rule 14).
