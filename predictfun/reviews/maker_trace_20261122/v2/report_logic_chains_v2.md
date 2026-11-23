# TOP-3 MAKER LOGIC CHAINS v2 — supersedes `report_logic_chains.md` in-band

2026-11-23, session aea5bfb9. Two parts:
- **Correction:** review 6bb458b30's next priority. Contract 27026e4a6, tools frozen 12d288cd2, runs `mt_x_cc_post_v1` / `mt_x_cc_pt_v1` rc 0.
- **TAKE-LOGIC v1:** the USER's request for solid proof of the take logic. Contract 3d9c6b082, tool frozen 5445e58f3, run `mt_x_tl_v1` rc 0, 12.1 CPU s.

Every number below is read from `v2/chains/correction_v1/` or `v2/chains/take_logic/`; the claims files hold the computed predicates.

**Population and limits:** ex-ante top 3 × the 19 BOOK markets, Oct 24–27 (4 consumed days).
- point estimates, no intervals (G = 4 < 5);
- p = circular-shift or reference-draw tail fraction, not independent-day confidence;
- descriptive of these markets, not a validated rule.

The first receipt stays as provenance.

## 1. Withdrawn or qualified statements of the first receipt

| First receipt | Now | Evidence |
|---|---|---|
| "≥ 70% of filled orders rested ≤ 10 s" and "not long-resting queue builders" | **Withdrawn.** 940 of 4,149 filled orders (22.7%) have a bound-constrained posting window ≤ 10 s (6 ≤ 1 s). The other 1,977 scenario-short orders are scenario estimates: 1,547 scenario-only, 413 conflict-flagged, 17 status-flagged. Their true rest is unknown. 834 bound-constrained windows are > 10 s (early or late, unidentified). | `claims_post.json` grades; all 14 reviewer recounts reproduced exactly |
| Rank-1 takes follow "BTC moving against its position" (5.1% vs 2.7%) | **Withdrawn: sign misread.** That statistic is BTC against the TAKE side, renamed `btc_against_take_side_5s`. BTC relative to inventory is measured in §3 on the decision-time population. | `claims_pull_take.json` renames |
| Pulls follow adverse BTC / inventory build-up "vs matched moments when the order kept resting" | **Qualified.** Details below. | `pull_families.csv`, `pull_membership.csv`, `claims_pull_take.json` |
| "57% re-quote within 5 s", reprice / refill / keep / two-sided / ladder counts | **Graded.** Details below. | `census_graded.csv` |
| Episode logic tables | **Regraded.** 176 POST rows (44 orders): `B_constrained` 176 → 88 B / 68 `C_scenario` / 20 `E_unresolved`; queue-priority "contradicted" 26 → 9. | `episodes_post_regraded.csv` |

**Pull qualification:**
- All 2,741 are cancel-compatible terminal windows; 0 pulls are identified.
- The presence reference is a union of same-side presence spans, not the same order: 908 matched, 1,826 active-span fallbacks, 7 unavailable.
- **Survives:** adverse BTC within 1 s is enriched under BOTH reference families on all 4 days at D = 250 ms. The fill-anchored family gives +9.70 / +9.38 / +10.43 / +4.17 pp (pooled +9.06, n = 1,401).
- **Timing-sensitive:** at D = 0, Oct 27 has p 0.119; at D = 1000 Oct 27 turns negative.
- **Withdrawn:**
  - a stable inventory trigger (fill-anchored Oct 25 −7.77 pp);
  - exhaustion as the weaker reference (Oct 27: exhausted +24.84 pp vs cancel-compatible +4.17).

**Census grades:**
- maker fills + takes: 19,658 observed executions;
- keep-after-fill: 4,931, continued execution, not a private KEEP;
- reprice: 4,023 = 2,954 tied + 1,069 sequential-under-scenarios;
- refill: 1,548 = 448 tied + 1,100 sequential;
- successors ≤ 5 s: later first-filled same-side hashes, not replacements;
- identified reposts: 0;
- two-sided / ladder: feasible or timing-sensitive overlaps (4,893).

**Takes with earlier excluded legs:** 248 claim-eligible takes. Their inventory classes are conditional on the retained ledger; tables are shown with and without them (`take_excluded_before.csv`).

## 2. Episode paths

`v2/chains/correction_v1/paths.md` gives 6 predetermined episodes: per rank, the first two in case order; not outcome-selected. Each runs post → fills → terminal tie → successor → takes → economics, with the alternative that would change each link's logic.

## 3. TAKE LOGIC — decision-time population (TAKE-LOGIC v1)

**Unit:** every market-second of each maker on the 19 markets: 5,700 per maker, 17,100 pooled. That is the whole 300-s window, not only the moments they took.

**Event:** a take whose earliest possible match lies in the next second: 1,359 event-seconds, 4,124 takes. Of these:
- reduce: 721 event-seconds, 2,076 takes;
- add: 473 / 1,195;
- timing-sensitive class: 247 / 815;
- other: 28 / 38;
- 23 takes precede the window (status).

State is measured at the second, from known-time data only.

**Rank 1 is most of it** (3,620 of 4,147 takes); ranks 2–3 take rarely.

### What the inventory is (rank 1; `inventory_at_take.csv`)

**Holding:** rank 1 holds inventory in **97.7%** of market-seconds (5,567 / 5,700): median |q| 150 shares (IQR 70–278; > 500 in 7%).

**Before a reducing take:** median |q| **209 shares** (IQR 101–351). Each reducing take unwinds a small slice:
- median **4.3%** of |q| (IQR 1.7–10%; p90 21%);
- 0 full closes.

**By inventory size** (rank 1, per 1,000 market-seconds):

| \|q\| (shares) | (0, 50] | (50, 200] | (200, 500] | > 500 |
|---|---|---|---|---|
| reduce-take rate | 85 | 110 | 116 | 146 |
| add-take rate | 89 | 79 | 65 | 42 |

### Verdicts (reduce type unless stated; D = 0; MH ratio; 500-shift null; `claims.json`)

| Logic tested | Rank 1 | Top-3 pooled | Reading |
|---|---|---|---|
| Reduce more when inventory is large (> 200 vs ≤ 50 sh) | RR 1.27, p 0.064, 11/12 markets: **timing_sensitive** (supported once pending legs count) | RR **1.48**, p **0.010**, **16/18** markets: **supported**, at every timing variant | **The one supported trigger** |
| Reduce right after ≥ 3 own maker fills in 5 s | RR 1.40, p 0.020, 15/19 markets: **timing_sensitive** (flips when the state is read 1 s earlier) | RR 1.41, p 0.016, 16/19: timing_sensitive | A fast reaction to fill bursts (within ~1 s) |
| **Take profit**: reduce when BTC (5 s) moved WITH the inventory | RR 1.12, p 0.10; only 4 markets with enough exposure: insufficient_support | RR 1.03, p 0.20, 4/8: **not_supported** | No evidence |
| **Stop loss**: reduce when BTC moved AGAINST the inventory | RR 0.90, p 0.52: insufficient_support | RR 0.82, p 0.73, 4/12: **not_supported** | No evidence |
| Unwind when the price beats fair value (edge ≥ 0.5 c) | RR 1.23, p 0.16: insufficient_support | RR 1.04, p 0.31: not_supported | No evidence |
| Add in the direction of the BTC move (sniping) | RR 0.97, p 0.56: insufficient_support | RR 0.90: not_supported | No evidence |
| Flatten in the last 30 s | RR **0.58**, 4/19 markets | RR 0.55, 3/19: not_supported | **Fewer** takes in the last 30 s |

Multiplicity: 7 mechanisms × 2 scopes, all listed.

### Does it happen once or a lot? (rank 1; `patterns.csv`)

| Pattern | Takes | Share of its takes | Markets with ≥ 1 (≥ 5) of 19 |
|---|---|---|---|
| P1 sell to reduce a long while BTC rises ("sells when BTC rises") | 81 | 2.2% | 7 (2) |
| P2 sell to reduce a long while BTC falls | 61 | 1.7% | 9 (2) |
| P3 / P4 buy to reduce a short while BTC falls / rises | 63 / 75 | 1.7% / 2.1% | 8 (0) / 14 (0) |
| P5 reduce while BTC is flat | **1,542** | **42.6%** | **19 (19)** |
| P8 reduce with ≤ 30 s left | 110 | 3.0% | 15 (2) |

**On BTC:**
- BTC was flat (|5-s move| < 1 bp relative to the inventory) in **86%** of rank 1's inventory seconds.
- The reduce rate is 134 / 106 / 105 per 1,000 s with / flat / against.
- "Sells when BTC rises" happens a handful of times, not as a habit.

### End of the 5-minute market (rank 1; `hazard_cells.csv`, `end_of_market.csv`)

| Time left (s) | 240–300 | 120–240 | 60–120 | 30–60 | 10–30 | 0–10 |
|---|---|---|---|---|---|---|
| reduce-take rate / 1,000 s | 88 | 100 | 131 | **172** | 74 | 37 |
| add-take rate / 1,000 s | 61 | 73 | 102 | 116 | **0** | 5 |

- **Peak:** taking builds through the market and peaks 30–60 s before the end.
- **Last 30 s:** adds stop and reduces fall. This profile is the declared marginal table; only the ≤ 30-s ratio was tested.
- **Inventory into the end:** median |q| falls from 195 shares 60 s before the end to 12 at the end. 10 of 19 markets end within 10% of the 60-s level, but 7 of 19 go into resolution holding ≥ 50 shares (up to 491).

### The take logic these numbers support (rank 1 dominates; descriptive, 4 consumed days)

1. Rank 1 carries inventory almost all the time and **takes mainly to trim it**: 2,076 reducing vs 1,195 adding takes pooled. Each take is small (median 4% of inventory).
2. **Trigger = inventory size**, plus a fast reaction to its own fill bursts:
   - the reduce rate rises steadily with |inventory| (85 → 146 per 1,000 s) while the add rate falls (89 → 42);
   - pooled this holds in 16 of 18 markets (p 0.010);
   - after ≥ 3 own fills in 5 s the reduce rate is 1.4× (timing-sensitive).
3. **BTC direction is not the trigger.** Neither take-profit nor stop-loss is supported; 43% of all its takes are reduces with BTC flat.
4. **Timing within the market:** trimming intensifies toward the end (peak 30–60 s left), then nearly stops in the last 30 s. Inventory still open then is mostly carried into resolution.

**Not identified here:**
- private decision times;
- unfilled take attempts;
- why taking stops in the last 30 s — book state near resolution is not measured in this packet.
