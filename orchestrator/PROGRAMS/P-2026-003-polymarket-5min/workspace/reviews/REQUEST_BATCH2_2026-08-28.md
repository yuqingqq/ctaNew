# Review request — completed fix-batch 2 — 2026-08-28

**From:** coordinator (R-254). **To:** CODEX system reviewer.
**Protocol:** one round per completed batch (R-239/R-240). Review at the EXACT
batch tip — the commit introducing this file on `origin/mm-research` — and
commit+push your filing as ONE file in this `reviews/` directory. Every claim
is verified red-first by execution; holds release only on your explicit words.

## Batch-2 scope (since your batch-1 filing `641e326`)

Code: `b3f082e d2eed33 2db216b edee0db 3f69611 1020610 5e09a8e ae22312`
(DA Q-filings `bf2b14f` + Q-DA-117/119/120; register/state commits are
context, in scope only where a recorded claim contradicts an artifact.)

## 1. Day-bar v2 — your DB1 + DB2 (`2db216b`)

- DB2: consumer now validates and USES a finite, ordered producer-supplied
  `gap_end_ns`; synthesis is the FALLBACK (truly-missing end only), the two
  paths separately counted in the diag. **The REAL seam test you required
  exists** (`da_o1_daybar_seam_test.py`, 7 checks): drives the COMMITTED v4
  producer against a fake socket via the `c288ed1` harness, reads the row the
  producer actually wrote, feeds that file to `day_bar_v2`. Coordinator
  red→green: the exact R-251 probe row now charges the producer's 100.0 s;
  end<start still refuses.
- DB1: per-coin bars computed and appended PER COIN before composition; the
  per-coin artifact carries its own `day_bar_v2` block and quality/accrual
  split. Suite 63→68 (last two = Q-DA-120 isolation checks).

**ASK: re-execute your counterexamples; explicit HOLD RELEASED / MAINTAINED
for day-bar v2 governing days ≥ 08-29.** (An 08-29 verdict re-run after a
release is anticipated: the bar predates the day.)

## 2. Iteration 011 — your I11-1/2/3 (`edee0db`, plus `b3f082e`)

- I11-1: main's printed keys falsifier-pinned to report_arm's emitted keys.
- I11-2: evaluator WIRED into `main()` (runner references 0→12):
  `q4_economics` ranks ACTIONS under real budgets; `evaluate_family`
  adjudicates all 24 declared cells (permutation p-values, incremental nulls,
  NO_INCUMBENT_COUNTERPART statuses, fixed-denominator Holm, cluster
  disclosure); **`assert_receipt_has_all_cells` REFUSES a receipt lacking the
  declared 24 — the artifact-level guard you asked for.**
- I11-3: heads state their own unevaluability with a reason; STATUS
  PRECEDENCE ruled: UNDERPOWERED beats UNEVALUABLE (both fields survive).
- Q2 statistic: USER RULED min (worse side), recorded in the frozen A1 file
  (R-249); `min` adjudicates only when BOTH sides measurable.

**ASK: explicit HOLD RELEASED / MAINTAINED for 011 fit/score.**

## 3. O1 re-check for the 2026-08-30T00:00:00Z boundary

Your adverse finding was DB2. The producer is UNCHANGED (v4 as committed at
`6786a02`); the consumer accepts its rows; the seam test runs the integration
the way the launcher runs it. Q-DA-119 quantifies the postponement cost:
race-accrual-eligible days = ZERO, btc's collector is 99.2% of lost
feed-time — each slipped boundary is one more zero-accrual day.

**ASK: does the DB2 closure clear your adverse finding? A clearance re-arms
the 08-30 boundary per the runbook (v3_1 hold stays until then).**

## 4. Phase 2B amended draft (`1020610`) — pre-freeze review

On the CORRECTED settlement basis (R-253; verified 17,727+/0 —
Chainlink TWAP-vs-open, ties UP; Binance appears in zero records):
- estimand `P(TWAP_[t0,T] >= S_ref)` with the part-realized decomposition
  `A_t + R`; every challenger must carry `A_t` as PIT state from admissible
  records (a current-price-only challenger cannot price the event);
- three estimand falsifiers: terminal-up/TWAP-down → Down; exact tie → Up;
  two decisions must differ when `A_t` differs;
- your FP2 loose terms now NUMBERS: median freshness delta >50 ms;
  availability <95% of Identity on the DECLARED COMMON UNIVERSE (independent
  of either estimator producing a record); ≥2,000 paired instants/coin-day;
  Brier skill vs Identity; alpha 0.05 Holm; equal per-instant weighting;
  constant-lag falsifier moved to a controlled synthetic fixture;
- upstream aggregation explicitly excluded as a Chainlink-feed mechanism
  question, not the settlement rule.

**ASK: is the amended draft fit to put before the user for freeze? Note the
structural consequence (challengers carry part-realized TWAP state) is
flagged to the user as a build-cost input to their one-vs-two-challenger
choice.**

## 5. FP1 (`3f69611`) + parity hardening (`5e09a8e`) re-checks

- FP1: `__post_init__` enforces at the boundary; value bounded to [0,1] on
  record AND factory ("a PM binary's price IS a probability"); ordering
  enforced for OK only (a non-OK record is a REPORT about bad input and must
  be constructible carrying the offending values); freshness-consistency
  universal; `assert_no_double_count` + `assert_declared_before`
  (timestamp-checked); 12 direct-construction known-bads; suite 21→46
  (incl. the coordinator-flagged taxonomy fix: dollar/both-invalid books now
  OUT_OF_RANGE, not ONE_SIDED).
- Parity: your five findings closed — `matched_control`'s caller-choosable
  count DELETED (determined by the treated arm; seeded draw, sorts before
  drawing, refuses rather than clamps, strict-subset disjoint from
  treatment); evaluated receipt (absent required check → all_pass False AND
  named); zero-repost/permanent-hold anchor flagged on EXPOSURE with
  PLACE_WITHHELD as a STATUS; requested=effective+suppressed with the
  zero-limit bit-identity anchor; training-reuse two-empty-sets refusal;
  external-arm contract refusing missing AND undeclared-extra fields.
  Declared limitation: NO real BE trajectory has crossed the external
  interface yet.

## 6. Also in scope, self-reported

- `ae22312` (Q-DA-120): the launcher seam test had been OVERWRITING
  production verdict artifacts (LOG overridable, OUTDIR hardcoded). Fixed:
  self-locating log line (sha256+as_of); DA_MIDNIGHT_OUTDIR with
  both-or-neither refusal rc 5 (coordinator-verified); production mtimes
  unchanged across a full suite run. Content never corrupted (verdicts
  recompute deterministically); the defect was provenance.
- `d2eed33`: falsifier_count.sh fails LOUDLY and stamps its commit ref
  (follow-through on the bare-fallback removal you'll find in R-250).

## Governing documents

Frozen 011 prereg @ `3b71d3e` + A1 (FROZEN Option 1 + Q2-min ruled block);
DAY_BAR_V2_PREREGISTRATION @ `dfa0977`+`368345b`+`9bcc208`;
PHASE2B_CHALLENGER_PROTOCOL_DRAFT @ `1020610` (DRAFT, unfrozen);
freeze receipt v2 @ `68dca00`; SEAT_PROTOCOL rules 15–17; register
R-238..R-254 (R-253 = the settlement correction, R-250 = the coordinator
count-provenance correction).
