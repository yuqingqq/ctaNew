# PAIRED-TURNOVER-ATTRIBUTION-v1 — declaration (frozen before the attribution is read)

2026-11-22, session aea5bfb9.
- **Specified by:** review `d799cb77`'s ONE next priority: a saved-evidence completion of LONG-CONTINUATION-PAIRED-VALUE-v1 (frozen `7407bdf1`, results `6b806c19`) on the same 64 pairs.
- **Builder:** `predictfun/research/pf_ladder_turnover_attribution.py`. Its docstring is the full rule.

**What it does not use:** no replay, label day, model or threshold.

**Budget:** inside the approved pilot allocation, so no USER decision is needed:
- this follow-up ≤ 0.04 core-hours plus a 0.02 review reserve;
- the batch ≤ 0.40, counting the pilot, the previous review's full 0.03 reserve, this follow-up and its reserve;
- the line ≤ 2.85.

## Rule

**Inputs:** the pilot's day records and per-pair ledgers (hash-bound) and its measure `pairs` / `arms` / `replacements` (= its audit). Every pair and status is used. Primary Y, rule, sampling and END are unchanged; the run refuses if any recomputed Y differs.

**FIFO lots** (a declared allocation convention, not causal necessity):
- Each arm's whole fill history is processed in (time, trade index) order.
- Lots carry side, price, quantity, actual acquisition time and arm-local gid.
- Lots open at the fork are **carried**: their incremental basis is the fork mark, so pre-fork gains and losses are never charged again, and their acquisition times are kept.

**Decomposition:** Q = matched closing profit + residual marked at END against basis. It is exact, refused otherwise; Y = ΔMatched + ΔResidual.

**Reported per arm:**
- matched shares, profit and profit per share;
- carried vs new, both matched and remaining;
- quantity-weighted holding time from the **actual** acquisition;
- lineage-involved vs other matched profit.

**Focal round trips:** each focal-lineage fill group (one gid at one time) is classed by its own END sign × its round trip (`profitable` / `loss` / `flat` / `none`). An own-END-negative fill with a profitable opposite close is never an uncompensated loss.

**Events:** retain vs alternative post-fork fills, keyed by execution event (time, trade index, side, price, quantity), never by shared gid. Y = Σ retain-only − Σ alternative-only (an identity, refused otherwise).

**Corrections (review `d799cb77` findings 2–3):**
- First replacement delay from the saved rows:
  - REVIEW: the first lineage submission;
  - ADMISSION POST arm: the first after the forked POST;
  - ADMISSION WAIT arm: the first at or after the fork, whatever its numeric gid.
- Attempts and accepted (non-refused) submissions are reported separately.
- Day signs per kind are computed: `POSITIVE_EVERY_DAY` / `NEGATIVE_EVERY_DAY` / `MIXED`.

**Outputs:** `arms` / `pairs` / `lots` / `focal` / `delays` / `summary` .csv and `audit.json`, all write-once.
- Summary per kind × day × side, kind and all: counts, Σ and Σ|·| of Y, ΔMatched, ΔResidual, the lineage / other split and the event value, Y quantiles, top-5 concentration, and the turnover share of Σ|ΔMatched| + Σ|ΔResidual|.

**What it does not do:** it declares no winner and learns no risk rule. Pairs overlap (repeated counterfactuals of 8 markets), so sums are never portfolio P&L. The queue mechanism stays `unidentified`.

## Verification (before the freeze)

- **Selftest:** 6 groups, using realizable fixed-price generations only:
  - profitable turnover despite a negative buy END mark;
  - a loss-making round trip;
  - residual long and short inventory;
  - carried inventory closed later, including a pre-fork partial round trip;
  - partial closes;
  - a corrupted Q refused;
  - focal classes;
  - events with the same trade under another gid;
  - gid reuse after WAIT;
  - the summary row and day signs;
  - the guards.
- **Mutation:** 22 / 22 planted errors caught.
- **Checker:** `ta_checks.py`, families A / D / Y / L / F / S / I, with an independent FIFO, events and delay rule:
  - L requires the arms that differ from the pilot's stored delay to be exactly the review's 12, at their corrected values;
  - a 10-case falsifier.
