# Review — R-496 (C)+(D): the free-read premise and the pre-declaration, BEFORE any number exists

reviewer: pm-codex · filed 2026-09-03T04:22Z · pinned tip `75f68d2` (request at `75f68d2`, R-496 at `70d2b45`, doc corrections at `6badca0`)
executed in `~/ctaNew-wt-rev` at `--detach 75f68d2`. No sealed file opened. `be_forward_day.py` read, never run. Nothing written under `data/`. No other seat's worktree touched.

---

## CLAIM 1 — **DOES NOT HOLD.**

**Opening 08-29 does not cost the race zero days. It costs one contingent race day, and the thing standing between 08-29 and accrual is not a ruling — it is an unattributed seat default that a single USER word reverses.**

I drove it rather than argued it. One boolean changed, nothing else:

```
ERA_ADMISSIBLE (shipped)  = {'clob_v3_1': False, 'clob_v4': False, 'clob_v5': True, 'clob_v4_1': True}
A) as ruled today           race_admissible_by_era = False   (pure True, touched ['clob_v3_1'],
                                                              reconstructed False, unevidenced False)
B) clob_v3_1 -> True        race_admissible_by_era = True
C) split_verdict on the 08-29 artifact's OWN predicates + its own bar regime:
     as ruled today       quality=True post_freeze=True era=False -> race_accrual_eligible=False
     clob_v3_1 admitted   quality=True post_freeze=True era=True  -> race_accrual_eligible=TRUE
D) FALSIFIER 08-30, with BOTH clob_v3_1 and clob_v4 admitted -> race_admissible_by_era = False
     (boundaries_inside_day ['2026-08-30T05:30:02.114727Z'])
```

(D) is there because a demonstration that cannot fail proves nothing (rule 16 / SEAT_PROTOCOL 16): the flip is not a blanket mutation that admits everything — 08-30 still refuses on its mid-day boundary. **08-29 is a fully formed race day held out by one boolean.**

Five things make that boolean movable, each read at the artifact:

1. **It carries no ruling.** `da_forward_day_verify.py:894` reads `"clob_v3_1": False,   # pre-O1` — the only entry in the table with no authority. `clob_v4` cites R-340; `clob_v4_1` cites "USER 2026-08-31" with twelve lines of recorded rationale; `clob_v5` is ruled admissible in advance.
2. **The register's claim of authority does not verify.** COORDINATION.md states "Every existing entry in that table cites a USER ruling (`clob_v3_1`/`clob_v4` from R-232/R-340)". R-232 spans lines 16960–16975 and contains **zero** occurrences of `clob_v3_1`, `admissib*` or `accru*` as an admissibility ruling. What R-232 (2) *does* say is: *"target: before **08-29 begins accruing** tonight."* The USER-approved ruling of record anticipated 08-29 as a race day.
3. **A seat introduced it, after the fact.** `git log -S'"clob_v3_1": False'` → first at **`4e1133c`, 2026-08-31T03:20:38Z**, "DA: era-admission guard as code — and 08-29's final verdict was fully eligible on an inadmissible era". Set two days *after* the day it disqualifies.
4. **This exact field has already flipped once, at this exact day.** The verdict on disk carries `supersedes` → the artifact it replaced (`as_of` 2026-08-30T00:06:01.246972Z, sha256 `b808e603f3448a5…`) had **`race_accrual_eligible: true`**. Its own `write_reason` says the path "is a CACHE of the current verdict, not a receipt". The request asked whether verdicts get rewritten; this one was.
5. **The module implementing the guard argues against its own entry.** `da_forward_day_verify.py:1320-1334` records the **USER RULING 2026-09-01**, verbatim: *"i dont care about collector version, as long as the data quality is good, then we can use to test the model"*, and concludes "among RULED eras, **QUALITY ALONE DECIDES**"; `ACCRUAL_RULE` states "era carries no fidelity claim, since clob_v3_1/v4/v4_1 make NO row-stamping change". The only reason that reasoning does not already admit 08-29 is that v3_1 is not among the "ruled" eras — the step (1)–(3) show was never ruled.

And the day is attractive. By the bars that govern it (`day_bar_v2`, pre-registered `dfa0977` 2026-08-28T05:44:44Z + `368345b` 05:51:34Z — **before** 08-29 existed, so no rule-11 problem in the bar itself), 08-29 reads **btc P1 32.29 s/hr against a bar of 120** and **eth 0.44**. The day the USER accrued yesterday, 09-02, reads btc 73.71. **08-29 is the cleanest day in the record.**

So the failure mode is concrete and near: on ~09-05 the race stands at G=3 or G=4, the USER wants a fifth day, and the cleanest post-freeze day available is 08-29 — which by then has been read, and is consumed (rule 11). The premise "reading it consumes nothing the race was ever going to use" is an assumption about a future ruling, presented as a property of the day.

### R496-R1 — HIGH — the free read must be preceded by a ruling, not by an assumption

**Not a recommendation on which way to rule (rule 14).** The requirement is only that the ruling precede the read, because after the read only one of the two answers is still available.

> **AMEND clause.** Before any 08-29 number is computed, the USER rules on `ERA_ADMISSIBLE["clob_v3_1"]`:
> **(a) CLOSED** — ruled *never admissible for race accrual*, in the `clob_v4`/R-340 form, with the register cite written into the table entry at `da_forward_day_verify.py:895` and the `# pre-O1` comment replaced. On that ruling Claim 1 holds, the read is genuinely free, and I release it.
> **(b) ADMITTED** — 08-29 becomes a race day (driven above: `race_accrual_eligible` True, G would go 2 → 3) and **must not be opened**; the read moves to a day that is not a race candidate, or does not happen.
> Either way the entry stops being an unattributed default, which is the actual defect: the race's admissible set has one member that nobody ruled and everybody has been citing as ruled.

---

## Item 1 — every remaining degree of freedom

R-496 (D) is the strongest pre-declaration this programme has written: population primacy (08-29 primary, 08-30 secondary, never pooled), the decision metric as net cents **against the incumbent**, action de-duplication, own-time/own-level valuation, t+L, rho secondary, a control matched on the decision variable and compared on the decision metric, the cluster disclosure with `intervals_claimable` false and no interval, Holm, skill incremental to the incumbent, exclusions as statuses, predicates not verdict strings, and both days named as consumed. Those are closed and I do not re-litigate them.

What is still choosable **after** BE has seen numbers:

| # | degree of freedom | why it is open | closed by |
|---|---|---|---|
| 1 | **the cell count and each cell's identity** | (D) says the count "is declared BEFORE the read" and never states a number, a location or a writer. Holm's denominator is the whole multiplicity argument — iteration 011 turned on 0.0479 vs 0.1199 | R496-R4 |
| 2 | **coin set** | 08-29 carries 7 coins; the verdict's `per_coin` is btc+eth; iteration 011 is btc-only. (D) is silent. Reading two and reporting the better is selection | R496-R4 |
| 3 | **candidate and incumbent identity** (module, sha, params) | "the incumbent" is named by role, never by artifact. Rule 12's form (hash + commit ref in the receipt) exists and is not invoked | R496-R4 |
| 4 | **budget grid** | pinned `(0.05, 0.10, 0.15)` at `phase2_declaration.py:94` — not cited by (D) | R496-R4 |
| 5 | **L** | `TARGET_LATENCY_MS = 50` exists twice (`harmful_hazard_model.py:50`, `phase2_declaration.py:95`) and two *other* grids exist (`adverse_feature_rows.py:39` `(0,150,250,350,500)`, `adverse_feature_rows_fast.py:40` `(10,25,50,75,100,150,250)`). "the frozen latency axis" names none of them | R496-R4 |
| 6 | **threshold mode** | `CAUSAL_FROZEN_FROM_TRAIN` vs `RETROSPECTIVE_TOPK` (`:80-81`); not cited | R496-R4 |
| 7 | **draw count** | "minimum 200 draws, target 2,000" is a range. Which floor the p sits on (1/201 or 1/2001) would be settled after seeing cost or behaviour | R496-R3 |
| 8 | **sidedness of the p** | unstated. The programme has an *unresolved* instance: iteration 011's frozen design declares two-sided, adjudication went one-sided (R-286/R-288), amendment A2 is still a DRAFT | R496-R3 |
| 9 | **the null's unit** (window vs block) and block length | (D) says "sign-flip permutation" without a unit | R496-R3 |
| 10 | **the adverse markout horizon inside rho** | "rho = adverse / spread" fixes the ratio, not the offset at which adverse is measured | R496-R4 |
| 11 | **control matching tolerance** and draws per matched cell | "matched on action count, side and hour" — exact or nearest? how many? | R496-R4 |
| 12 | **the exclusion-status vocabulary** | rule 4 makes exclusions statuses; it does not close the *set*. A status invented after the read shrinks the population by a rule chosen on it. This repo already applies closed-vocabulary discipline elsewhere (`STATE_MARKERS`, `da_forward_day_verify.py:930-947`: "must ASSERT what it is from a CLOSED vocabulary, and a row that asserts nothing REFUSES") | R496-R4 |
| 13 | **masking treatment of 08-29** | no content-liveness rule governs that day | R496-R5 |
| 14 | **whether the 08-30 secondary is published** | (D) says labelled and never pooled; it does not say *reported regardless of what it shows* | R496-R4 |
| 15 | **number of run attempts, and which is of record** | `_flush` records a re-run as `.1`, `.2` with the prior chain hashed — recorded, not barred | R496-R6 |
| 16 | **the size of the era caveat, after the fact** | "whether the surface moves the economics is unknown and is not assumed in either direction" can be resized post-hoc in either direction | R496-R7 |
| 17 | **which downstream decisions may move on this read** | the largest one | R496-R2 |

### R496-R2 — HIGH — "named as consumed" bars the *day*; it does not bar the *decision* (item 3)

Naming 08-29/08-30 as consumed is necessary and it is not sufficient, and the gap is not hypothetical. Per `RESULTS.md` §7 there is now **exactly one** open USER decision: **the Phase-2 winner — "the race decides it."** A read that shows the candidate beating the incumbent on 08-29 speaks directly to that decision, and nothing in (D) prevents it from being used there. No 08-29 row need ever be touched again: the barred object is the day, the unbarred object is the choice.

The same shape covers the quieter cases — the read shows the candidate winning on btc and losing on eth and the race quietly becomes btc-only; a budget looks dominated and drops out; a threshold "obviously" wants moving. Each is rule 11 without re-using a single consumed row.

> **AMEND clause.** The declaration enumerates, by name, the decisions frozen at the instant 08-29 is opened — candidate identity, incumbent identity, coin set, budget grid, L, threshold mode, and **the criterion by which the Phase-2 winner is decided** — and states that none of them may move on the basis of this read; any later change to one of them is a new declaration with its own multiplicity. Add the outcome vocabulary in advance, in the form the repo already uses (`phase2_declaration.py:111 DECLARED_OUTCOMES`), so "what would count as a negative" is fixed before the number exists.

### R496-R3 — MEDIUM-HIGH — the null is the one that produced iteration 011's floor, and on a one-day population it is optimistic in a way the disclosure does not cover (item 4)

Read at the code: `phase2_iter011.py:1127 sign_flip_null` and `phase2_increment_null.py:127 sign_flip_p` — both **window-level**, "each permutation flips each window's sign independently and re-sums", H0 "the per-window paired increments are symmetric about zero". Both already pin `N_PERM = 2000` (`phase2_increment_null.py:62`; `phase2_iter011.py:1019`) with seeds 20260827 / 20260828.

Three separate problems, only one of which the disclosure covers:

- **The range is not a declaration.** "minimum 200, target 2,000" leaves the achieved floor to be settled later. Rule 6 wants a number. `N_PERM = 2000` already exists — cite it and commit to it, so the p is a measurement rather than a floor. This is the specific defect iteration 011 shipped (all 18 non-Q4 cells at 1/501, `at_permutation_floor: true`).
- **Within-day dependence is not covered by the interval disclosure.** `intervals_claimable: false` correctly disowns *intervals*. The **p** is a separate object, and a window-level sign flip inside a single UTC day treats serially dependent windows — same coin, same regime, same book state, one day — as exchangeable. It understates the null variance. With G=1 there is **no** valid cluster-level null available at all: you cannot permute days when you have one.
- **Sidedness is unstated**, in a programme that currently has an unresolved one-sided/two-sided amendment on exactly this metric.

> **AMEND clause.** Declare: `N_PERM = 2000` exactly (cite the module and seed); the p is **one-sided or two-sided, stated now**; and report the window-level p as the **optimistic bound** beside a **block-level** sign flip over contiguous windows with the block length declared before the read. If no defensible block length can be declared in advance, then declare that the primary quantity is the **point estimate** and that the p is descriptive — which is the honest reading of a one-day population and is compatible with everything else in (D).

### R496-R4 — MEDIUM — (D) fixes the procedure but not the numbers, and most of the numbers are already pinned in a committed module it does not cite

`phase2_declaration.py` already pins `ARMS` (:69), `THRESHOLD_MODES`/`THRESHOLD_PRIMARY` (:80-81), `EMBARGO_S` (:91), `N_RANDOM = 200` (:93), `BUDGETS` (:94), `TARGET_LATENCY_MS = 50` (:95), `DECISION_METRIC = "net_cents"` (:96), `DECLARED_OUTCOMES` (:111). R-496 (D) cites none of it, so rows 1–6, 10, 11, 14 of the table above are nominally re-choosable although the programme long ago chose them.

> **AMEND clause.** One line closes most of it: *"the read runs under `phase2_declaration.py` at sha `<X>`, with `POPULATION` overridden to 08-29 (primary) / 08-30 (secondary) and nothing else changed."* Then add what that module does not carry: the coin set; candidate and incumbent artifact shas; the adverse offset inside rho; the control's matching tolerance and draws per cell; the **closed** exclusion-status vocabulary, with the statement that an exclusion outside it REFUSES the run rather than shrinking the population; and that the 08-30 secondary is reported whatever it shows. Finally, the **cell count as an integer**, written in a committed artifact before the first score.

### R496-R5 — MEDIUM — no content-liveness rule governs 08-29, and 08-29 is the era where invisible holes are most likely

Read at the code (`da_forward_day_verify.py:2811-2823`, both asserted in the shipped suite): `da_content_liveness_rule.EFFECTIVE_FROM_DAY == "20260902"`, `da_content_liveness_v2_check.EFFECTIVE_FROM_DAY == "20260903"`, `governs("20260902") is False` for v2. **Neither governs 20260829.** Consistent with that, `data/pm_5min/derived/` holds `da_blackout_mask_20260901.json` and `da_blackout_mask_20260902.json` and no mask for 08-29.

That matters more on this day than on any other, because 08-29 predates **O1c**. Verified in the `6786a02` diff: before O1c, a socket that connects and never delivers raises nothing, so `open_gap` is never opened and no gap row is written — the diff's own words, "a silent no-subscribe was indistinguishable from a quiet market — an invisible-hole class". So on 08-29 that class leaves **no gap row, no duration, no mask and no governing content rule**. (Stated fairly: O1c is not sufficient either — the two unexplained 09-01 outages happened *with* O1c active, no gap rows, and are on the record. 08-29 simply has one more uncovered mechanism than the race days do.)

The consequence for the metric is **argued from the code path and not measured**, and I state it as such: during an undetected stall the last book persists, so the mid used after t+L has not moved, adverse selection is measured smaller than it was, and the spread is frozen — biasing **rho = adverse / spread** toward the flattering side. Direction argued; magnitude unknown; not asserted.

> **AMEND clause.** Declare 08-29's masking treatment before the read — either "no mask applied, and the day's unmasked status is disclosed in the result" or a named mask artifact built before the first score — and report a content-liveness status for 08-29 as a **REPORTED, non-governing** figure so a reader can see what the day looked like on the instrument built for this exact class. Whichever it is, it is chosen now.

### R496-R6 — LOW-MEDIUM — the seal boundary holds against files; the leak channel is code and attempts (item 5)

Verified by reading `be_forward_day.py` (never run): `run_forward_day(day, outdir)` is day-scoped and I found no cross-day read on the run path; scores go to `outdir/be_forward_day_SEALED_scores_{day}.json` (`:1002`); the receipt carries counts, identities and hashes only, and `:1094` **REFUSES** a receipt carrying decision-shaped fields; `_flush` (`:1109`) never overwrites a prior receipt — a second run supersedes into `.1`, `.2` with `supersedes_receipt`, `prior_receipts` and their sha256s. Distinct outdirs therefore give real separation, and an opened number cannot reach a sealed receipt through the driver.

Two channels remain, neither of them a file:

- **Code.** One driver scores the opened day and the sealed race days. Any change to `be_forward_day.py` or its dependencies motivated by what 08-29 showed retro-contaminates the race: 09-01 is already sealed under one `driver_sha256_prefix`, a later day would carry another. The receipt records the sha, so it is *visible* — visibility is not a bar.
- **Attempts.** "Run until it looks right" is recorded and not barred; nobody has undertaken to read the chain.

> **AMEND clause.** Declare the scoring stack **frozen for the duration of the race** — driver sha plus dependency shas recorded in the read's own receipt — with any post-read change to it requiring every already-accrued day to be re-scored on the new code and the fact recorded in the register. Declare that the **first completed run** of 08-29 is the result of record, and that every attempt is enumerated in the filing with its receipt path and sha. Declare distinct outdir **roots** for opened and sealed runs (the receipt already carries `roots`), not merely distinct filenames within one outdir.

### R496-R7 — LOW — item 2 answered: the fields are untouched, "two generations" overstates the distance, and the unquantified caveat is itself a degree of freedom

The request asked me to establish what actually changed and whether it touches the fields the decision metric consumes, with a HIGH if it does. **It does not**, and I checked it twice — once in the code, once on the tape.

**In the code.** `v3_1 → v4` is one commit, one file: `6786a02` (2026-08-28T05:48:23Z, `live/pm_research/collect_pm.py`, +74/−8) — O1a `ping_interval/timeout` 10/10→3/3, O1b `reconnect_delay` (cause-aware backoff with jitter), O1c the subscribe-confirmation branch, O1d `gap_start_ns` fallback, and the `COLLECTOR_VERSION` constant. Filtering the diff for the write path, the **only** ± line that touches it is `"gap_start_ns": last_recv_ns or err_ns` → `... or scope_start_ns` — the **gap ledger**, not the tape. `v4 → v4_1` is `1b35aa4` (2026-08-31T12:51:16Z, ping 3/3→10/10 rollback) and `168438a` (15:33:23Z, a `MODE_SPEC` mapping so the declared era and the keepalive that produces it cannot drift). Keepalive and bookkeeping; nothing on the row.

**On the tape** — because the ledger's "NO row-stamping change" is the deployer's own assertion, and rule 16 says verify at the artifact. 40 mid-day btc files, 60,000 rows each, on 08-29 (`clob_v3_1`), 09-01 and 09-02 (`clob_v4_1`):

| predicate | result |
|---|---|
| stamp form: one tab, 19-digit `recv_ns`, on **every** row | **True on all three days** (60,000/60,000 each) |
| payload schema sets equal across the three days | equal on all four common schemas (`price_change`, both `book` forms, `last_trade_price`) |
| only difference | 4 `tick_size_change` events in the 08-29 sample, 0 in the others — a **venue** event type, also present in the 08-19 sample; a sampling difference, not an era difference |
| falsifier: a planted extra field must break the equality predicate | **breaks it** (returns False) |

**And the label overstates the distance.** The code's own R-360/R-363 comment (removed from `collect_pm.py` at `168438a`, readable at `1b35aa4`) records the measurement: *pre-O1a **08-26..08-29**, 96.0 h → btc **100.2 s/hr** lost; post-O1a 08-30 05:30Z → **242.8 s/hr***. `clob_v4_1` is the **rollback to 10/10** — so on the one axis anybody measured, 08-29's era and the race's era are the **same configuration**, and `clob_v4` is the outlier between them. 08-29's own measured P1 (32.29 s/hr) beats both accrued days. The residual v3_1→v4_1 deltas are O1b/O1c/O1d, all of which concern **detecting and labelling loss**, never the values of rows that survive.

So this is not the HIGH the request contemplated. It is, however, a live degree of freedom in the shape it is currently written: "whether the surface moves the economics is unknown and is not assumed in either direction" is an escape hatch of adjustable size — an unfavourable number can be dismissed as "the old collector", a favourable one keeps it as a footnote.

> **AMEND clause.** Replace the open-ended caveat with its verified content, fixed before the read: *fields and stamping identical across `clob_v3_1`/`clob_v4_1` (verified at the code and on 180,000 tape rows); keepalive identical to the race's era, `clob_v4` being the outlier; the deltas confined to loss detection and labelling (O1b/O1c/O1d), of which O1c's absence is the material one and is handled under R496-R5.* Then the caveat cannot be resized after the number is seen.

### R496-R8 — LOW — one row of the candidate-day survey is not read from the artifact the document says it is

`RESULTS.md` §3 states "Every day below is read from its own `da_dayverdict_<day>.json`." There is **no `da_dayverdict_20260831.json`** — not on disk, and never committed (`git log --all --diff-filter=A` over that path is empty; the committed set is 08-28, 08-29, 08-29_v2, 08-30, 08-30_v2, 09-01 + superseded, 09-02 + superseded). The 08-31 row's `post_freeze` **false** is also surprising on its face: 08-31 lies three days after the freeze epoch 2026-08-28T06:09:00Z.

Not load-bearing — 08-31 is excluded on era mixing regardless — but the survey is quoted as artifact-anchored and one of its rows has no named basis. Rule 16's shape, in a document written to apply rule 16.

---

## Findings

| # | severity | finding | disposition |
|---|---|---|---|
| **R496-R1** | **HIGH** | 08-29 accrues on one boolean (driven, with a falsifier); `ERA_ADMISSIBLE["clob_v3_1"]` is an unattributed seat default (`4e1133c`, 2026-08-31T03:20Z), the R-232 cite does not verify, R-232 (2) anticipated 08-29 accruing, the verdict already superseded an `eligible: true` predecessor, and the module's own USER-2026-09-01 text argues for admission. The read is **not free** | **blocks the read**; USER ruling first, either direction |
| **R496-R2** | **HIGH** | "named as consumed" bars the day, not the decision — and the read speaks to the single open USER decision (Phase-2 winner) | AMEND before the read |
| **R496-R3** | MEDIUM-HIGH | window-level sign flip on a one-day population; draw count declared as a range; sidedness unstated | AMEND before the read |
| **R496-R4** | MEDIUM | procedure fixed, numbers not; most already pinned in `phase2_declaration.py`, uncited | AMEND before the read |
| **R496-R5** | MEDIUM | no content rule governs 08-29 and no mask exists for it, on the one day predating O1c | AMEND before the read |
| **R496-R6** | LOW-MEDIUM | seal boundary holds against files; the leak channel is the shared scoring code and unbarred re-attempts | AMEND before the read |
| **R496-R7** | LOW | item 2 answered: fields untouched (verified at code and on 180,000 rows); "two generations" overstates the distance; the unquantified caveat is a post-hoc lever | AMEND the caveat's wording |
| **R496-R8** | LOW | `RESULTS.md` §3's "every day read from its own verdict" is false for the 08-31 row | record correction |

## Disposition

**AMEND.** The exact clause, in the order it must be applied:

1. **R496-R1 first and alone.** The USER rules `ERA_ADMISSIBLE["clob_v3_1"]` — CLOSED (never admissible for race accrual, R-340 form, cite written into the table) or ADMITTED (08-29 becomes a race day and is not opened). **Nothing else in this filing matters if the answer is ADMITTED**, and after the read only one answer remains available.
2. On a CLOSED ruling, fold R496-R2 through R496-R6 into R-496 (D) as declaration text, and R496-R7 into the caveat, **before the first score of 08-29 is computed**. R496-R8 is a record correction and blocks nothing.

Claim 1: **DOES NOT HOLD.** Claim 2 (the pre-declaration is complete): **not complete** — seventeen degrees of freedom remain, of which item 17 (R496-R2) is the consequential one. The declaration is nonetheless the best this programme has written, and five of the six amendments are one or two sentences each.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed only in `~/ctaNew-wt-rev` at `--detach 75f68d2`; heavy steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No sealed file opened** — not `~/ctaNew_sealed_backup/`, not the `/tmp` original, not any relocation. `be_forward_day.py` read, never run. No Phase-4 OUTDIR passed to `--run`. No unit, timer or anchor; `DA_MIDNIGHT_MODE` never set. Nothing written under `data/`; the era-flip demonstration passed `admissible_table` as an argument and mutated no file. `~/ctaNew-wt-be`, `-da`, `-de` never read. No plan file edited. BE round 12 not touched (R-377).
