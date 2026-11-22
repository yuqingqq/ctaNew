# PAIRED-TURNOVER-ATTRIBUTION-v1 — receipt

2026-11-22, session aea5bfb9.
- **Specified by:** review `d799cb77`'s ONE next priority: a saved-evidence completion of LONG-CONTINUATION-PAIRED-VALUE-v1 (`7407bdf1` / `6b806c19`) on the same 64 pairs.
- **Frozen:** `db044afb`, before the attribution was read.

**What it is:** a descriptive FIFO attribution. No replay, model or threshold. The pairs are overlapping counterfactuals of 8 consumed markets, so sums are never portfolio P&L. No winner, no risk rule; the queue mechanism stays `unidentified`.

## Integrity

| Item | Result |
|---|---|
| Unchanged inputs | Every recomputed Y = the pilot's Y (64 / 64; the run refuses otherwise) |
| Decomposition | Every arm's Q = matched closing profit + residual, exactly |
| Event identity | Y = Σ retain-only − Σ alternative-only execution-event values on every pair |
| Independent checks | `ta_checks.py` A / D / Y / L / F / S / I all pass, with its own FIFO, events and delay rule. **L:** exactly the review's 12 ADMISSION-alternative delays differ from the pilot's stored field, each at its corrected value. Falsifier **10 / 10** |
| Mutation | 22 / 22 |
| Resources | Follow-up **≈ 0.0042 core-hours** (cap 0.04). Batch with both review reserves 0.277 of 0.40. Line **2.6634**; with the 0.02 reserve, **2.6834 of 2.85** |

## Result — completed turnover versus residual exposure

ΔMatched is the closing profit of post-fork round trips (carried lots at the fork-mark basis). ΔResidual is the open lots at END, marked against their basis. Both are retain − alternative, in c.

| Kind | Day | Σ Y | Σ ΔMatched (Σ\|·\|) | Σ ΔResidual (Σ\|·\|) | Turnover share of the \|components\| | Top-5 share of Σ\|Y\| |
|---|---|---:|---:|---:|---:|---:|
| REVIEW | Oct 24 | −544 | −654 (1,375) | +110 (405) | 0.77 | 0.99 |
| REVIEW | Oct 25 | −1,897 | −6,202 (10,015) | +4,305 (4,312) | 0.70 | 1.00 |
| REVIEW | Oct 26 | −2,563 | +863 (2,061) | −3,426 (4,095) | 0.33 | 1.00 |
| REVIEW | Oct 27 | −2,696 | −1,718 (2,923) | −978 (988) | 0.75 | 1.00 |
| **REVIEW** | **all** | **−7,699** | **−7,711 (16,374)** | **+12 (9,801)** | 0.63 | 0.86 |
| ADMISSION | Oct 24 | +2,491 | +2,451 (2,451) | +40 (72) | 0.97 | 1.00 |
| ADMISSION | Oct 25 | +4,547 | +531 (3,338) | +4,017 (4,017) | 0.45 | 1.00 |
| ADMISSION | Oct 26 | −901 | −912 (912) | +11 (11) | 0.99 | 1.00 |
| ADMISSION | Oct 27 | −1,898 | −1,888 (1,892) | −11 (32) | 0.98 | 1.00 |
| **ADMISSION** | **all** | **+4,239** | **+182 (8,593)** | **+4,057 (4,132)** | 0.68 | 0.97 |

**Computed day signs:** REVIEW `NEGATIVE_EVERY_DAY`; ADMISSION `MIXED`.

**REVIEW** (KEEP vs PULL under the common lineage hold): the aggregate deficit is **completed turnover**. ΔMatched −7.7k c against ΔResidual +0.01k.
- The lineage-involved part is −2.4k and the other orders' part −5.3k.
- On Oct 26 alone, the residual carries it.

**ADMISSION** (POST vs WAIT): the aggregate gain is **residual exposure**, +4.1k c, almost all on Oct 25. The turnover components are large in |·| but offset across pairs.

**Concentration:** the top 5 pairs carry 86 % (REVIEW) and 97 % (ADMISSION) of Σ|Y|. Most of each aggregate comes from a few pairs; 34 of 64 pairs are exact ties.

**The book itself, per arm** (means, REVIEW retain / alternative):

| Measure | Retain | Alternative |
|---|---:|---:|
| Matched closing profit, c | −4,348 | −4,107 |
| Matched shares | 1,278 | 1,291 |
| Profit per matched share, c | −2.90 | −2.88 |
| Holding time from actual acquisition | 8.6 s | 8.8 s |
| Residual, c | −490 | −490 |

Carried lots: 65 shares at the fork, −223 / −129 c when closed. ADMISSION is similar: about −2.0 to −2.2 c per matched share.

The book's completed round trips lose about 2–3 c per share in both arms. Under this FIFO lens, the continuation's loss is realized round-trip loss, not mainly open inventory.

**Focal round trips** (fill groups of the focal lineage, own END sign × round trip; REVIEW retain):

| Own END sign | Round trip | Groups | Own END, c | Round trip, c |
|---|---|---:|---:|---:|
| negative | profitable | 172 | −170.2k | +18.2k |
| negative | loss | 203 | −179.1k | −47.6k |
| positive | loss | 273 | +218.9k | −70.7k |
| positive | profitable | 197 | +181.6k | +29.1k |

Remaining classes are small: none / flat / zero. The two axes are largely independent. Many own-END-negative fills close at a profit (they are **not** uncompensated losses), and many own-END-positive fills sit on losing round trips. Both arms look alike.

**Events per pair** (mean, REVIEW): 226 common, 34 retain-only, 37 alternative-only.

**Corrected replacement delays** (review finding 2): the alternative-arm mean first-attempt delay over the 63 arms with a replacement is **869.8 ms**, not 1,361 ms. The 12 affected ADMISSION WAIT arms are at +50 ms. Accepted delays differ from attempt delays only where a refusal came first.

## What this shows and does not show

**Shows** (descriptively, on these 64 overlapping pairs):
- REVIEW's sampled aggregate favours the alternative on every day, through **completed turnover**: retaining under the lineage hold closed round trips at worse prices than pulling and re-entering.
- ADMISSION's sampled aggregate favours posting through **residual exposure** concentrated on one day.
- The one-sided END sign of a focal fill says little about its round-trip value.

**Does not show:**
- that the queue causes any of it: the mechanism is `unidentified`;
- a decision rule, predictability or profit;
- generality beyond 4 consumed days and this continuation.

FIFO is an allocation convention; another matching rule would split ΔMatched / ΔResidual differently, though never Y.

The pilot report's wording is superseded in band by `../ladder_long_continuation_20261122/report_interpretation_v2.md`.

## Files

- **`outputs/`:** `arms`, `pairs`, `lots` (every FIFO close), `focal`, `delays`, `summary` (.csv.gz; each = `audit.json`).
- **Receipt root:** `checks.json`, `checks_falsifier.json`.
- **`logs/`:** every job and the mutation runs.
