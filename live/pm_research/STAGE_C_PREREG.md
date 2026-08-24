# STAGE_C_PREREG — predictor-gated quoting, pre-registered before any forward number exists

REVISION: 1

**Authorizing ruling (in-file per R-126): R-135 — conditional
authorization, surface-freeze exception for THIS protocol only.**

**Status: DRAFT — pre-registered, NOT RUN, and carrying its own RUN
CONDITION (§2). Drafted blind: no gated cell, no forward number, and no
sight of BE's in-sample curve beyond the fact that it is being measured.**

**Why this is not the dead abstention axis, stated up front**: Stage A
proved that UNINFORMATIVE abstention scales the loss toward the WAIT-only
zero and cannot cross it (120/120, R-123). An INFORMATIVE gate has a
different break-even — `E[drift | kept] < spread` — because it does not
drop quoting time, it drops the fills the predictor says are toxic while
keeping the rest. Whether the predictor is informative ENOUGH is exactly
what this protocol tests, and the placebo control (§5) is what prevents
mere abstention from wearing information's clothes.

## §1. The gate — frozen score, placement time, retention-only grid

- The gate reads **BE's FROZEN candidate score** (hash-pinned by the
  re-frozen candidate receipt; the freeze instant and commit enter via
  `freeze_from_receipt`, R-132) **at placement time**, from
  knowledge-time-admissible inputs only. **No new model levers**: the
  candidate is immutable; nothing in this protocol may touch features,
  weights, horizons or thresholds inside the model.
- **The grid is RETENTION QUANTILES only, frozen now:**
  keep-fraction ∈ {0.90, 0.75, 0.50, 0.25} — thresholds expressed as
  retention quantiles of the score stream (scale-free; no raw-score
  tuning), plus the two anchors: retention 1.00 (ungated book — must
  equal the reference EXACTLY, the null-point gate) and retention 0.00
  (the wiring must-fail: zero fills or the run aborts).

## §2. RUN CONDITION — the protocol's own kill switch

**Stage C runs ONLY IF BE's in-sample `E[drift | kept]` curve crosses
zero at some retention.** If the in-sample curve — which is FLATTERED by
construction (in-sample, seen days) — never crosses, then the forward
test cannot be expected to and **this protocol SELF-VOIDS: the void is
recorded here in-file with the curve's receipt cited by hash and as-of,
and no cell ever runs.** An in-sample crossing is NECESSARY, not
sufficient — it licenses the forward run, it promotes nothing.

## §3. Population — forward days only, by the R-129 selector

Admission via `select_holdout` under the RE-FROZEN instant read from the
candidate v2 receipt (`freeze_from_receipt`; a receipt without a commit
hash is refused — R-132). Forward days only; partial days beside,
labeled, never deciding; every population with n and as-of (R-105).
**This protocol does not touch the re-freeze/split sequencing** — it
consumes the split that lands after DA's verification, whenever that is.

## §4. Metrics — matched windows, day unit

Primary: **fill-conditional markout (share-weighted M5) of the GATED
book vs the UNGATED book, at MATCHED windows** (same windows, all three
books — gated, ungated, placebo — replayed on the same tape). Beside:
total M5 PnL/window (the promotion quantity), fill shares, retention
achieved vs targeted. Day-clustered where G supports it; **point
estimate with NO interval otherwise** (the R-109 ruled standard).

## §5. The PLACEBO-GATE control — matched retention, no information

For every retention rung, a placebo gate drops the SAME FRACTION of
placements by seeded randomness (deterministic per (window, rung);
seed stamped). The placebo IS Stage A's lesson made into a control:
uninformative dropping scales toward zero. **Therefore the promotion
comparison is gated-vs-placebo at matched retention, not merely
gated-vs-ungated** — a gate that only matches its placebo is abstention,
whatever its score stream says. A doctored-score control (scores
shuffled within window) must be statistically indistinguishable from
the placebo — the association must-fail, run before any cell is read.

## §6. Promotion bar — forward days, day unit, placebo-beating

A retention rung PROMOTES iff, on EVERY complete forward day, both
coins: (a) the gated book's total M5 PnL/window is **> 0**, AND (b)
**> its placebo at matched retention** on that day. Signs and points at
the day unit; no intervals below supporting G. Anything else is
reported and nothing happens. If no rung promotes, the informative-gate
hypothesis fails on forward data and the result is reported with the
same standing as Stage A's 120/120 — an answer, not a disappointment.

## §7. Controls (before any cell is read) and results discipline

Null-point (retention 1.00 ≡ reference, every window, abort-on-break);
wiring must-fail (retention 0.00 ⇒ zero fills); placebo determinism
(same seed ⇒ identical placebo book); the doctored-score association
must-fail (§5); engine §4.3 controls inherited. Results file carries
the R-127 five up front: status line first, decision-eligibility
stated, model-gate state beside every number, populations with n and
as-of, authorizing ruling in-file.

## §8. Sequencing

1. This draft stands as the pre-registration. When BE's in-sample curve
   lands: evaluate §2 — self-void or proceed to the coordinator's
   freeze of this protocol (bar before data, as always).
2. On freeze + the post-verification split: run forward, read against
   §6, report cells first / promotion second, all rungs, all three
   books, gaps shown.
