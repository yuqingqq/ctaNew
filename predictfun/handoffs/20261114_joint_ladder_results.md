# JOINT-LADDER: the guard ladder with both sides in one lane, a shared position and a size-scaled cap (results, 2026-11-14)

Session ce68ebe0. USER:
- "The economic test is incomplete. The current replay evaluates ladder levels independently. Preserving several orders needs evaluation
  under shared inventory limits, including adverse fills and cancellation/re-entry costs".
- "We should have larger cap when we increase the size".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_joint_ladder.py` + `run_joint_ladder.sh`, readings R610-R612 declared | 0ded5c49, before the first run | refused at 05:44Z by its self-test |
| Self-test fix: two wrong expected values. Own fills are known L_FILL = 100 ms later (not 20); a resting order is cancelled at the window's end | 879fe15a | 05:45-06:02Z, ~2 min a day |
| `parity`: a POST HOC control. Independent mode must equal the per-rung lanes everywhere | 0e60b8e4 | 06:02-06:05Z (10-23, 10-31, 11-03) |

## Design

- **One lane per window holds both sides of the guard ladder:** rungs 0-5, 30 sh each.
  - Each rung follows S4's rest lane line for line.
  - Our orders never share a price on one side.
  - Each trade leg's volume is shared among our orders in price priority.
  - Own fills are known 100 ms later.
- **Shared position:** + bid fills, - ask fills.
- **Cap = K x order size,** so a larger size gives a larger cap. K = 2 (v3's soft cap), 5 (v3's hard), 10, and no cap.
  - While |known position| >= cap, the side that would increase it is pulled and blocked, then waits 250 ms.
- **Arms:** LDG (pull every rung on the guard), FB25 (FUTURE-BEST's keep rule, old model), OFB2 (the same rule with the realized 2 s move,
  perfect foresight).
- **Clocks and values:**
  - fill clocks M 0 / 50 / 250;
  - fills valued at 1 s / 5 s and also marked at 60 s and at settlement, i.e. the position held;
  - inventory statistics.
- **Controls passed:**
  - R610: no cap equals the per-rung lanes on the 897 window-side x arm x clock cases where nothing interacted; 45,759 were skipped for an
    interaction.
  - Post hoc parity: with interactions switched off (each order gets the whole leg, no price-collision wait), the joint lane equals the
    per-rung lanes on all 15,552 checks of 3 days, 0 differ.
  - Synthetic cases: the shared sweep, the cap side and lag, a price collision; three planted errors caught.

## Results (WS $/day w/o young at 1 s, unless stated)

| Arm, cap | M 0 | M 50 | M 250 | All trades M 50 | All fills marked at 60 s, M 50 | Settlement, M 50 | Max position per window, median / p95 | Steps at the cap |
|---|---|---|---|---|---|---|---|---|
| LDG, 2 x size | +53 | +32 | -542 | -104 | -86 | +29 | 60 / 120 sh | 25% |
| LDG, none | +83 | +54 | -691 | -91 | -92 | +69 | 67 / 242 | - |
| FB25, 2 x size | +106 | **+74** | -882 | -167 | -201 | -184 | 65 / 123 | 30% |
| FB25, 5 x size | +145 | +95 | -1,044 | -177 | -240 | -83 | 90 / 193 | 8% |
| FB25, none | +170 | +121 | -1,160 | -156 | -254 | -146 | 90 / 293 | - |
| OFB2 (perfect), 2 x size | +303 | **+275** | -290 | +235 | +329 | +379 | 66 / 120 | 28% |
| OFB2, none | +394 | +366 | -375 | +336 | +416 | +534 | 91 / 272 | - |

**R611 KEEP-UNDER-CAP:**
- **FB25 PAYS against LDG at every cap:** M 0 on 9 of 9 days, M 50 on 7-8 of 9. It is +$41 a day at 2 x size and +$66 with no cap at M 50.
  At M 250 it is below LDG on 9 of 9 days at every cap: -$339 a day at 2 x size.
- **OFB2 PAYS at every cap and every clock:** +$242 to +$316 a day, 9 of 9 days, M 250 included.

**R612, what the cap costs** (2 x size against none):
- M 0 / M 50: LDG -30 / -22, FB25 -63 / -47, OFB2 -91 / -92.
- At M 250 the cap helps (+$85 to +$278).

**Compared with the independent rungs** (TARGET-TEST, M 50):
- LDG +54 vs +54, FB25 +121 vs +124.
- OFB2 +366 vs +435: shared trade legs and price collisions cost the oracle about $70 a day.

## What it means

1. **Shared inventory and a cap shrink the value of keeping, but on WS-timed trades it survives at M 0 / 50.** FB25 beats pulling everything
   by $41 a day at v3's soft cap (2 x size) against $66 uncapped. It still loses at M 250.
2. **The keep strategy itself is sound when we know the harm.** With perfect foresight it beats LDG by $240-310 a day at every cap and every
   fill clock, with the shared position. The limit is prediction, which is ORDER-HARM's job.
3. **Over all trades and held to 60 s, both LDG and FB25 lose money.**
   - FB25 at 2 x size: -$167 all trades, -$201 at the 60 s mark.
   - The estimated-time (E2) trades carry the loss, as in TIMING-SPLIT.
   - Only the oracle is positive there.
4. **The cap binds 25-30% of the time at 2 x size, and a sweep overshoots it to about twice the cap** (p95 120 sh for a 60-sh cap): several
   rungs fill in one taker order before the cap can react. Deeper ladders need either a larger cap with size, or fewer kept rungs.

## Next

- ORDER-HARM (running): the per-order label (probability and severity of harm after the cancel lands, keep minus pull with the stay-out),
  with rung 0 included; models per held-out day.
- Then the ORDER-HARM policy in this joint lane at the same caps and clocks, against LDG and FB25.

## Limits

- Replay rule M (B1a); inflated volumes (C4); consumed days (C8).
- The guard ladder, not the frozen v3 (C11).
- Settlement marks rest on one binary outcome per window.
- The post hoc parity covers 3 of 9 days.

Outputs:
- `data/predict_fun/research/joint_ladder_20261114/`: joint_ladder.txt / .json, fills_*, inv_*, day_*.json, parity_*.json, logs.
- Copies (txt / json / logs) in `reviews/joint_ladder_20261114/`.
