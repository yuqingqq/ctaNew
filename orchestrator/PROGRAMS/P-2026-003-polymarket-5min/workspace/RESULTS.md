# RESULTS — P-2026-003 Polymarket crypto 5-min

Consolidated 2026-09-03T03:23Z by the coordinator (ctanew-7e) on the USER's
instruction, with every seat halted. **Single writer: the coordinator.** This
file is the compact, artifact-anchored answer to "what has been tested and what
came out of it". `STATUS.yml` and `workspace/HANDOFF.md` remain the running
state; `COORDINATION.md` remains the append-only register. Read this first, then
`HANDOFF.md` for the live detail.

Every number below was read from the artifact named beside it during this
consolidation, not from a seat report or from an earlier summary. Where a
previous document disagrees, this file says so explicitly and the disagreement
is a **correction**, not a restatement.

---

## 1. The bottom line, in four sentences

The **ranking model works and the economics do not yet clear their own bar.**
Iteration 011's arrival head beats the incumbent hazard head by a wide margin
(AUC 0.8303 vs 0.7139), but every surviving cell sits **at its permutation
floor** with 500 draws on a null the artifact itself discloses as optimistic,
and the decision metric — net cents against the incumbent — does not survive
multiplicity (best Holm 0.1199). The **forward race**, which is the only thing
that could turn development evidence into validation, has **1 of the 5 required
complete days accrued**, with day 2 eligible and awaiting a USER call. The
**Phase-4 diagnostic the USER scheduled at R-459 has not run**, because the
runner's feature-assembly step had never been executed and the population's
split declaration is a USER ruling that is still open.

---

## 2. Iteration 011 — the conditional signed-value family

**Artifact:** `data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json`
(188,119 B, `as_of` 2026-09-02T05:21:34Z, written 05:35:04Z). BTC only.
**Class:** `development_evidence.is_a_validation = false` — both fitting and
evaluation populations are development. It selects; it never validates
(prereg 4).

| head | statistic | lgbm | linear | p | Holm | survives |
|---|---|---|---|---|---|---|
| **Q1_arrival** | AUC, 311,640 rows | **0.8303** | **0.7733** | 0.001996 | 0.0479 | **yes** |
| Q2_sign | AUC, 33,622 actions | 0.6003 | 0.5824 | 0.001996 | 0.0479 | no — `NO_INCUMBENT_COUNTERPART` |
| **Q3_magnitudes** | calibration slope, 15,912 rows | **0.6888** | **0.6437** | 0.001996 | 0.0479 | **yes** |
| **Q4_combined_ev** | **net cents vs incumbent**, 166 windows | +3,867.1 / +2,818.2 / +2,472.6 | +3,277.5 / +278.6 / +1,565.4 | 0.01999 best | **0.1199 best** | **no** |

**Q1 is a real increment.** Its gate — *"beats the matched-random null AND beats
the incumbent hazard head"* — has **both conjuncts computed and true**.
Incumbent hazard head AUC **0.7139077** (`incumbent_auc`, 2,000 permutations,
166 units), so the increments are **+0.1164** (lgbm) and **+0.0594** (linear),
on 166/166 windows with zero exclusions.

**Four qualifications, all read off the artifact, none of them optional:**

1. **Every surviving p is a floor, not a measurement.** All 18 non-Q4 cells
   carry the *identical* p = 0.001996 = **1/501** with
   `at_permutation_floor: true`, 500 draws. Holm 0.0479 sits 0.0021 under the
   0.05 line; **one draw the other way moves it to 0.0958 and the entire
   surviving set collapses.**
2. **"12 of 24 survive" overstates the evidence, and the artifact says so
   itself.** `distinct_results`: 24 declared cells → **12 distinct**, and
   **4 distinct surviving results** (Q1×2 arms, Q3×2 arms). Budgets select
   cancellations, not predictions, so Q1/Q2/Q3 carry one statistic replicated
   across three budgets. *Read the survivor count as distinct results, never as
   independent evidence.*
3. **The p-values are optimistic by declaration.** `cluster_disclosure`:
   `G_complete_utc_days: 0`, ruled unit **UTC day**, unit actually used
   **window**, `weaker_than_ruled: true`, `intervals_claimable: false` — units
   within a day are not independent. *Evidence, never a significance
   certificate.*
4. **Q3 passes a weaker gate than Q1.** Q3's frozen gate carries **no incumbent
   term** (`carries_incumbent_term` false for its conjunct set), so it passes a
   gate that never required beating anything. That is not Q1's achievement.

**Q4, the decision metric, in full.** All six point increments are positive
(+278.6 … +3,867.1 net cents; best cell REPORTED-not-adjudicated as candidate
+11,743.9c vs incumbent +8,466.4c). None survives: best one-sided p 0.01999
over 2,000 sign-flip permutations — **not** at the floor (floor 1/2001), so this
one is a real measurement — giving Holm 0.1199 over the family of 24. Status is
`GATE_PARTIALLY_EVALUATED`, and `passed` reads **null, not false**: the
structured conjuncts record `increment_beats_incumbent: false` and
`matched_random: null` — a conjunct nobody computed reads null (R-397 ruling 2).
The frozen design declares a **two-sided** p (0.04998, reported); the adjudicated
p is one-sided per R-286/R-288, and amendment A2 is a DRAFT awaiting the USER,
because only the USER amends a frozen design.

**One tension left open, deliberately.** That cell's prose `detail` says *"The
incumbent counterpart EXISTS (comparable=True) and was NOT COMPUTED, so only the
matched-random conjunct was evaluated"* — which is the **opposite way round**
from its own structured fields (`gate_conjuncts_unevaluated: ["matched_random"]`,
and an incumbent-increment p that plainly was computed). This is the rule-10
shape (prose beside a table). It is recorded here as an **observed tension to be
adjudicated**, not as a proven error, and nothing in this file is read off that
prose.

### Corrections to earlier documents (in-band, rule 13)

| where | said | the artifact says |
|---|---|---|
| `HANDOFF.md` §Current model state | `cells_by_status` = **18 OK + 6 NO_INCUMBENT_COUNTERPART** | **12 OK + 6 NO_INCUMBENT_COUNTERPART + 6 GATE_PARTIALLY_EVALUATED** (denominator 24). The six Q4 cells were counted as OK; they are not |
| coordinator's own report to the USER, 2026-09-03 ~03:08Z | "12 of 24 cells survive" quoted without the multiplicity disclosure | 12 cells = **4 distinct surviving results**, all at the 1/501 floor |

---

## 3. Forward race — the only path from development evidence to validation

**Bar:** ≥5 complete UTC days, each FINISHED ∧ AFTER ∧ ADMISSIBLE ∧ HEALTHY.
**State: G = 1 of 5.**

| day | verdict | note |
|---|---|---|
| 08-29 | not admissible | era |
| 08-30 | does not accrue | |
| 08-31 | does not accrue | mixed `clob_v4`/`clob_v4_1`; BTC P1 298.52 s/hr against a 120 s bar |
| **09-01** | **ACCRUED** | first day the race has ever counted; four conjuncts recomputed, not read back |
| **09-02** | **eligible, un-ruled** | first **governed** verdict; `race_accrual_eligible: true`; the accrual itself is the USER's (R-486 (6)) |
| 09-03 → | open | earliest possible G=5 is ~09-06, and only if every day accrues |

**09-02, at `data/pm_5min/derived/da_dayverdict_20260902.json`** (43,449 B,
sha256 `6f283262df463957…`, `as_of` 2026-09-03T00:06:01.399Z, written by the
scheduled unit — `ExecMainStatus=0`, `Result=success`, 00:06:00→00:06:06 UTC):

| conjunct | value |
|---|---|
| FINISHED | `day_closed_calendar` true (the tape selector reads false; its predicate lags the boundary by up to one window — disclosed in the row) |
| AFTER | `post_freeze_pass` true, 288/288 every coin |
| ADMISSIBLE | `clob_v4_1`, era-pure, no boundary inside the day — an interlock, not a quality grade |
| HEALTHY | `day_quality_pass` true under the governing `day_bar_v2`: btc P1 **73.71** s/hr (bar 120) · P2 **0.00 %** (bar 5 %) · P3 **219.7** s (bar 900); eth **1.85** · **0.00 %** · **15.5** s |

**Reported beside it and NOT governing:** btc `windows_gap_affected` **50.3 %**
coin-level (145/288 windows, 287 gap intervals, 1,769 s lost) against eth
**1.7 %**; the count bar `gap_rate_under_bar` fails (304 gaps, 12.67/hr, 8 hours
over the hourly bar); `tape_density` UNMEASURED (its receipt covers 13 days, not
this one).

**Content liveness governs for the first time and reads THIN.**
`content_liveness_rule` `governs: true`, `frozen_by_user: true` (R-386, module
`7196676840304f30`, effective from 20260902): status **CONTENT_THIN, 7 of 7
coins thin, 0 unjudgeable** (btc L1 0.138, longest thin run 40 windows; hype
0.055 passes L1). It **does not veto HEALTHY**
(`content_thin_vetoes_HEALTHY: false`, ruled by R-409): disclosed and masked,
not inadmissible. The blackout mask artifact is `WRITTEN`, 7 coins, **251 masked
windows**.

**09-01 was scored.** `be_forward_day_receipt_20260901.json` — outcome
**SCORED**, `as_of` 2026-09-02T13:24:05Z, **610,064 btc + 441,409 eth actions**,
masking applied at supply (141 windows masked across 7 coins) so the blackout is
excluded before rows are built and **counted**, never silently dropped.
**The scores themselves are SEALED and have not been read** (rule 11: no
forward metric is opened before ≥5 complete days). 09-02's receipt is a
**REFUSAL** — "not closed by calendar", written 13:50:22Z, before the day
closed; it will be re-run after the accrual call.

**Two unexplained outages remain on the record** (09-01): 00:00–01:05Z (65 min)
and 22:45–23:35Z (50 min) at 0.01–2.2 % of median window content, on all seven
coins, with **no gap rows** — invisible to every duration bar. Two independent
instruments (collector-log msgs/s and raw gzip-trailer bytes) agree to one
minute. This is the class the content-liveness rule exists for.

---

## 4. The Phase-4 diagnostic the USER scheduled (R-459) — NOT RUN

**Verified at 03:11Z** by importing `de_phase4_diag_runner` at the landed tip and
calling `preflight()` read-only (no `--run`, nothing written):

> `DiagRefused: the feature assembly's EXPENSIVE HALF is not wired, so
> q1_arrival_composed_lgbm/btc cannot be scored. What is missing is exactly one
> step: PA.tape_index(split) over the fit's own tape and
> PA._feature_pass(PA.FRAGMENT, 'phase4_diag', TAPE=...). … and the §3 population
> spans BOTH fit splits, which is a declaration nobody has made. Refused HERE,
> before any feed is built (DE34-C1).`

Two blockers, one of each kind:

1. **A producer step nobody had been dispatched to build** (coordinator's
   omission, stated as such): rounds 33–42 hardened the runner's instruments —
   necessary work, since round 33 would have fed for ~29 minutes and then
   crashed on a stub scorer feeding the booster one column against 106 — but the
   expensive half was never assigned. **Dispatched 03:13:51Z as DE round 43**;
   see §6 for what that immediately turned up.
2. **A declaration that is the USER's** (rule 14): the §3 population
   (08-24/08-25) spans **both** fit splits — 1,125,289 `train` rows and 638,917
   `score` rows — so every cell would score generations the heads were fitted
   on. Either the run is declared a MECHANICS diagnostic on a consumed
   population (splits labelled per cell), or it is restricted to the `score`
   split (smaller population, §3's counts change). **No seat may choose.**

**Cost, as far as it is known:** the feed is MEASURED at ~28.6 min for the §3
population, both coins. The feature assembly is **UNMEASURED** — a tape index
over `phase2_state_tape_v5.json` (3,170,987,711 B) and `_feature_pass` over
`harmful_exposure_rows_v3_eraB.json` (1,241,115,096 B, 1,135,943 rows). One
`arm_result` is unmeasured on real data with a floor of 0.007 s; 200 draws is
**800 replays**, plus rejected attempts.

---

## 5. What else has been settled, and what it cost to settle

| question | answer | where |
|---|---|---|
| What do these binaries settle on? | **Chainlink**, never Binance — verified in `data/pm_5min/markets.jsonl`. The exact settlement statistic is contested; the repo's own reconstruction favours **S60 endpoints (99.8 %)** over a TWAP-vs-open reading. **No settled form is asserted.** | R-253, Q-DA-142/146 |
| Is CLAUDE.md rule 9's parenthetical right? | **No — it is FALSE and must not be cited.** Rule 9 still binds this program through a different door: the PM book (`Identity`) already prices the event, so skill is reported incremental to `Identity`, never to a base rate | — |
| Sub-second Binance data | reliable **only** from 2026-08-24 13:48:54 UTC (`recv_ns >= 1787579334881534478`). Earlier tape is usable for ≥1 s bars only | `data/mm_hf/collector_runs.jsonl` |
| Fair-price Identity | built (typed). The **challenger protocol is not freeze-ready and no challenger has been scored** | `STATUS.yml: hazard-fair-price` |
| Skew semantics | `QR_SKEW_ONLY` user-frozen | `STATUS.yml: hazard-skew` |
| Seven-arm integrated replay | contracts, parity stubs and inert trajectories only — bit-identical parity against a real seven-arm replay, lifecycle economics and the integrated candidate freeze are all **open** | `STATUS.yml: hazard-integrated-replay` |

**The review machinery, measured:** 494 register entries, 579 filed seat rows,
57 adversarial review filings, **85 distinct numbered findings**. Two from the
last 24 hours show the shape of what it catches:

- **BE12-S1** — a selftest "positive control on a real emission" ran
  `run_forward_day("20260902")` under a comment asserting that day refuses in
  ~0 s. When the scheduled unit wrote the 09-02 verdict at 00:06Z, gate 1 began
  to PASS and the control silently became a **full closed-day scoring run inside
  the selftest** (measured: 14 min, ~16 GB, killed; wrote only to its own temp
  outdir). Its subject was the calendar, not the code. Now pinned to `21000101`
  and proved unscorable **before** the driver is called.
- **DA20-R2** — the R-486 `governs` stamping was suite-green but
  **unfalsifiable**: deleting *either* production call site left 254 checks
  passing. Rule 17's shape. Fixed in DA's rebuilt held chain (unpushed).

---

## 6. What the seats hold right now (all halted 2026-09-03 03:18Z)

| seat | worktree | held | state |
|---|---|---|---|
| BE | `~/ctaNew-wt-be` | nothing unpushed; clean | round 12 landed (`f47ceb7` code, `669ef72` row Q-BE-237). Coordinator verified 129/129 checks, rc 0, driver sha `0d688474a715e899` |
| DA | `~/ctaNew-wt-da` | **2 unpushed commits** `3c49cb7` (round 20 code) → `a36db71` (row Q-DA-216); clean | rebuilt HELD chain, ready to land after the 09-04 00:06Z run |
| DE | `~/ctaNew-wt-de` | **1 unpushed WIP commit `0d03902`** (+248/−27, one file), suite **RED** by design | round 43, §6a below |
| MEM | main tree | nothing unpushed | round 71 swept (`d9b85ee`); reports nothing lost |
| reviewer | `~/ctaNew-wt-rev` | nothing unpushed | DA-20 filing landed (`cc4cfb9`); its context had reached 100 % and auto-compacted during the stop |

### 6a. What DE round 43 turned up in four minutes — the producer half bites

Wiring the expensive half moved `phase2_arms.py` from **1 reached entry to 5**,
and the runner's own code pin went **BLOCKING** by name on `_stream_tape_rows`.
Measured from the blobs: that function **changed between the fit and the tip** —
sha `f0741bc4b170fabc` → `f0b3bccfb8ec5b88` at commit `2e1204f` ("BE: T2
fail-open readers", 2026-08-29). The diff is confined to one branch: EOF without
the rows array's closing `]` used to return and now raises; **the accepting path
is byte-for-byte unchanged**. The tape's last bytes are `...}}]}` — its rows
array is closed — so the new refusal branch cannot fire for this input, and DE
added `tape_rows_array_closed()` as the predicate behind that claim rather than
asserting it.

**Open question DE did not rule, and should not have:** whether declaring
`_stream_tape_rows` is a seat's call or the USER's. It needs no number and no
policy choice — only a computable statement about code and about this tape's
last bytes — but the judgement is unreviewed.

This is the value of running the producer: **one four-minute wiring attempt
surfaced a fit-vs-tip code drift that no instrument round had found in ten
rounds.**

---

## 7. Open USER decisions (five; none is a seat's to make — rule 14)

| # | decision | recommendation on record | blocking what |
|---|---|---|---|
| 1 | **09-02 accrual** on its non-blackout complement, per R-409 | **ACCRUE** (R-486) | G = 1 → 2 of 5 |
| 2 | **Phase-2 winner** | none — the race decides it | the forward race's meaning |
| 3 | **content-liveness v2 freeze** | open | whether 09-02+ days are judged on v1 or v2 |
| 4 | **addendum v2 package**, five asks (§1 horizon, §2 `theta_repost`, §3 `REPOST_DWELL_S`, §4 `inf` + the identity, §5 repost parity, and **§1a the split — ruled *with* §2 and §4, not after**) | §1a: declare **MECHANICS on both splits**, splits labelled per cell; §2/§3: keep the seat's values with sensitivity pairs reported and **neither selected**; §4/§5: adopt. The reviewer priced the alternative: the `score`-split answer costs a round **and** a re-declaration that comes back to the USER | **the R-459 diagnostic** |
| 5 | the 09-04 00:06Z run staying on the landed chain | no pin, no install | nothing — it fires either way |

The addendum DRAFT is at
`live/pm_research/plans/DE_PHASE4_DIAGNOSTIC_ADDENDUM_V2_DRAFT_2026-09-02.md`
(307 lines, sha `cb693000880c3d94`, the asks at §286-307). **Nobody edits it.**
It was released by the reviewer with a **+17 / −0** delta — purely additive, so
the released text is still line-for-line inside what the USER now reads.

---

## 8. Sister program P-2026-002 (HF market making) — a gate opened today

Read `orchestrator/PROGRAMS/P-2026-002-hf-market-making/workspace/HANDOFF.md`.
E1 is complete and the verdict is **overlay-only**:

- **E1-B, standalone Binance MM: no real pass.** Majors negative pre-fee
  (BTC −0.19, ETH −0.23 bps at 30 s). ADA's +2.44 screen pass is an H1
  fat-tick artifact — notional-weighted it is **−0.32 bps**; the same flip
  appears on every wide-tick name.
- **E1-A, passive-execution overlay for the XS book: PASS, audit-robust.**
  T_p=600 s: touch 3.45 [3.11, 3.79] / sweep 6.26 [5.76, 6.75] against an 8 bps
  capstone; stale-shadow 7.20; excl-ICP 6.15; per-symbol max 7.53. Still
  maker-optimistic (H1, no queue position).

**The E2.0 / E2-A gate is now MET and unworked.** It required 14 days of L2;
measured 2026-09-03 03:18Z: `data/mm_hf/raw/depth20/BTCUSDT` holds **351 hourly
files, 2026-08-19 12:00 → 2026-09-03 03:00Z**, and the Hyperliquid side
(`hl_raw/l2Book/BTC`) holds **350**, 16 symbols each, 44 GB total. Both
collectors are live. E2.0 is the true-mid recompute (notional-weighted) that
voids or settles ADA; E2-A resolves the overlay bracket with real books.
**Nothing has been dispatched against it** — this program has been calendar-
blocked since 2026-08-19 and the calendar has now moved.

---

## 9. How to reload this program's context

1. `workspace/SEAT_PROTOCOL.md` — who may write what. **One writer per state
   file.** `STATUS.yml` and `HANDOFF.md` are MEM's; `COORDINATION.md` R-entries
   are append-only; this file is the coordinator's.
2. `workspace/COORDINATION.md` — the register. Read the **last five R-entries**
   and the Q-filing table, not the whole file.
3. This file, then `workspace/HANDOFF.md` (11.8k lines — read the dated entries
   at the top and the sections you need).
4. `STATUS.yml` — 15 tasks with statuses, 317 flags, 10 standing rules.
5. `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` §10 — the
   governing TODO. The stateful cancel×skew worksheet's 47/113 checkbox count is
   **not** project completion.

**The reliability rules in `CLAUDE.md` are not style.** Each was bought with
dissolved work: rule 11 (choosing after seeing voids the test), rule 12 (a
freeze is a commit), rule 15 (every checker ships a falsifier), rule 16 (verify
at the artifact a claim names), rule 10 (compute predicates, never print
conclusions) — this consolidation applied rule 10 and rule 16 to the program's
own documents and found two errors in them (§2).
