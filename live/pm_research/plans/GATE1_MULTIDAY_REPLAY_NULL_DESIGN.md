# Gate 1 — the multi-day replay-null design (DE declaration, R-547)

**Status:** DESIGN DECLARATION. **No data is touched by this document or by
the module that emits its JSON.** The run is a separate, later act, and the
reviewer files on this design before it starts (R-547 item 6).

**Authority:** USER ruling R-547(A), verbatim in the register. **Emitter:**
`live/pm_research/de_multiday_design_declaration.py` (20 checks, falsifiers
both directions). **Declaration JSON:**
`data/pm_5min/derived/p003_de_multiday_gate1_design__20260906T031853Z.json`
(sha256 `89ac8b15b83c9197…`).

---

## 1. The days, and why exactly these five

`2026-09-01, 09-02, 09-03, 09-04, 09-05` — the only era-pure `clob_v4_1`
days in existence (R-547(C)). Admissibility is read from each day's
`da_dayverdict_<YYYYMMDD>.json` era-admission block. **It is NOT CLAUDE.md
rule 5's mm_hf Binance boundary**, which governs a different tape; that
clause of the draft was corrected in band at R-547(B).

Nothing has been chosen on these days: the arms' thetas were fixed on the
consumed 08-24 hour, and the forward race scored a **different object** on
them whose scores are sealed and unread.

**Consequence, stated now:** this run **consumes** all five. Gates 2–6, if
reached, need five further admissible days — 09-06 onward, earliest
complete set 09-10, readable 09-11.

## 2. What DE needs from BE, per day

| item | detail |
|---|---|
| object | the day's reference book, same shape as the 08-24 arms cache's `fr` |
| fields | `reference` (slug → side → generations with tranches), `statuses`, `population`, `n_slugs`, **`terminal_marks`** |
| why terminal_marks | the inventory leg is marked to each window's terminal price; a book without them reads `NO_TERMINAL_MARK` on every fill |
| digest | BE publishes `{day, path, sha256}` per day in its builder declaration |
| pinning | DE's per-day artifact carries `reference_book: {day, path, sha256}` and **recomputes the digest at read time**; a mismatch **refuses that day** — the whole day, not the offending draw |

DE does not build the book and does not open BE's pickle.

## 3. The arms, and where their thetas are pinned

| arm | theta | head | pinned at |
|---|---:|---|---|
| `CONDVALUE_X_SKEW` | 0.32450609461933483 | `q1_arrival_composed_lgbm` | `be_cancel_axis_null_v1.json` (`6951f57d2b8a23bd…`) `cells.CONDVALUE_X_SKEW.arm_filed.theta` |
| `HAZARD_OVER_SKEWED_REF` | 0.43525926488298716 | `incumbent_linear_d` | same artifact, `cells.HAZARD_OVER_SKEWED_REF.arm_filed.theta` |

Both were fixed on the **consumed** 2026-08-24 13:50–14:50Z hour and are
**not refitted on any of the five days**. That is what keeps the days
unconsumed for this test.

## 4. The decision population

The arm's **above-threshold events on day d at its fixed theta** — the set
a cancel decision is drawn from. On the consumed hour it was 1,154
(CONDVALUE; 586/568 by side) and 106 (HAZARD; 41/65). **The per-day values
are OUTPUTS and are unknown now**, so no expectation about them can later
become a filter. An arm with zero decisions on a day is a **counted
status**: it cannot be aggregated over G = 5 and the test does not
silently become a four-day test.

## 5. The null

Random decisions **matched to the arm's own count and side split on that
day**, drawn from that day's decision population, replayed through the
**same stateful cascade**. Matched on: decision count, side split. **Not**
matched on realised cancel count, realised cancel set, or fills lost — a
decision is not a cancel (CONDVALUE converts 1,154 → 333, HAZARD 106 → 48).

- **≥ 500 draws per arm per day.** Fewer refuses.
- **Machinery:** `live/pm_research/be_cancel_axis_null.py` (BE's). DE does
  not re-implement the cascade — two implementations of one cascade are
  two cascades, and the null must run through the policy the arm ran
  through or it is not a control for it.
- **Seed:** `int(sha256(day_book_sha256 ‖ arm ‖ 'P003_GATE1_MULTIDAY')[:8], 16)`.
  The seed **pins the data by digest**: a book that moved cannot reuse the
  draw sequence, and the sequence is reproducible from the artifact alone.

## 6. The metric

**Primary: `D(E0)`** — net value delta at maker fee zero (our signed rate),
arm minus `QR_SKEW_ONLY`, per day. **`D(E−R)`** at the rebate's identity
value is reported beside it as robustness and never substituted for it.
Unchanged from R-537 and the fee-endpoint receipt v1–v3.

## 7. Aggregation, and the §7 predicate — declared before any day is seen

Per day: `p_d = (1 + #{null ≥ observed}) / (1 + K)`, one-sided, larger is
better; floor `1/501` at K = 500. Per-day cluster value:
`Z_d = (D_d − mean(null_d)) / sd(null_d)`. Cluster unit = **UTC day**
(rule 8). Estimate = `mean_d Z_d` over G = 5. Draws are **not** pooled
across days — that would make the draw the cluster unit and inflate by a
free resource.

> **The §7 predicate.** Arm *a* **FAILS** iff `mean_d Z_d ≤ 0` **or** the
> five day signs are **not unanimous**.

**The asymmetry is deliberate and is the point.**

- **FAIL is cheap and needs no significance.** "Did not beat the null" is
  not a claim that needs power. A stopping rule should be easy to trigger.
- **PASS is capped by arithmetic.** At G = 5 the smallest attainable
  one-sided sign-test p is `2⁻⁵ = 0.03125`; Holm at m = 2 compares the
  smaller p against `0.025`. **No arm can clear Holm on this run even if
  every day goes its way.** A pass therefore means **DIRECTIONAL AND
  CONSISTENT, NEVER SIGNIFICANCE-BEARING** — the same limit R-529(A) ruled
  for the forward race, declared here *before* the run.

**Computed, not asserted:** the smallest clearing G is **6 at m = 2** and
**5 at m = 1**. So **one more admissible day** would make a unanimous pass
significance-bearing at m = 2. (I first wrote 7 and 6 by hand; the check
caught it. The field is computed for exactly that reason.)

Multiplicity **m = 2** (the two arms), declared here.

## 8. Falsifiers, both directions

| planted input | required behaviour |
|---|---|
| arm at the null's mean every day | **FAILS** |
| arm strongly positive on 4 of 5 days | **FAILS** — the rule is unanimity, not the mean, so a four-of-five arm cannot be talked into a pass afterwards |
| arm above every draw on every day | **does not fail**, and `clears_holm` is asserted **False** |
| day whose book digest ≠ its pin | **REFUSES that day** |
| < 500 draws | **REFUSES** |
| null with zero dispersion | **STATUS**, never a large Z |
| four days instead of five | **REFUSES** rather than testing at a smaller G |

## 9. What would refute this design

1. If an arm's per-day decision population is systematically empty or tiny
   on the admissible days, the matched null cannot be built — that refutes
   the **design**, not the arm.
2. If BE's cascade cannot be driven on a day's book without refitting
   anything, the "same cascade" premise is false.
3. If the five days are not independent in the way the day cluster assumes
   (one market event spanning days), the interval is wrong even at G = 5.
4. If any arm's theta is found to have been fitted on any of these five
   days, they are consumed and the run is void (rule 11).

## 10. Resources

Measured on the consumed hour: DE's arms replay **47.0 s / 0.61 GB** for 12
windows; BE's 500-draw null **290.9 s** for two arms over the same hour. A
UTC day is 288 windows against 12, so a linear extrapolation is **24×** —
roughly 19 minutes of replay and ~2 hours of null per arm-day. **That is an
estimate from one hour, not a measurement.**

Cap: one CPU, `MemoryMax=8G`, **never raised** (R-174). If a day exceeds
the cap, **the day refuses; the cap does not rise.** Recommendation: run
the first day alone as a smoke and publish its resource observation before
the remaining four.
