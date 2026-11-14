# JOINT-LADDER v3: the joint ladder on client-known decisions, with hard caps that hold capacity until a refusal is known (results, 2026-11-14)

Session ce68ebe0.

USER review of v2 (06:3xZ):
> "it releases capacity using a rejection that has not happened yet ... This can give the replacement order an unrealistically early queue
> position. All current conservation checks pass, because this is a decision-timing error, not a volume-accounting error. The fix is to reserve
> capacity from submission until the acceptance/rejection becomes known, keeping client-known state separate from venue execution state. Add a
> test requiring identical decisions when only future events change."

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| v2 `research/pf_joint_ladder_v2.py` + driver: hard pre-trade cap, conservation checks | 3040edf2 | stopped after the USER's review; partial outputs deleted unread. **VOID**, never reported |
| v3 `research/pf_joint_ladder_v3.py` + `run_joint_ladder_v3.sh`, readings R613 / R615 declared | eb735795, before its first run | 06:49-07:54Z |
| POST HOC exposure control `pf_joint_ladder_v3_exposure.py` + driver | 273bbe8f, before its run | 08:00-08:39Z |
| POST HOC readings `pf_joint_ladder_v3_posthoc.py` (all trades, day blocks) | 87ebe59c, before its run | 08:39Z |

## The repairs (v2 -> v3)

1. **Capacity is reserved from submission until the refusal is known.**
   - v2 dropped a refused order from the cap's count the moment it was submitted. The venue refuses it 40 ms later, so v2 used the future.
   - In v3 an order counts against the cap from submission until one of these is known to us:
     - its refusal, at activation (submission + 40 ms);
     - its full fill (+100 ms);
     - its cancel's confirmation (landing + 100 ms).
2. **Every lane decision uses client-known state.**
   - **The cap's capacity.** Known position + every share of ours still live to the client (size minus fills we know of) + the new order ≤
     K x size.
   - **"Our order rests."** v1 and v2 read the venue's remaining size, so a fill we had not yet seen changed decisions up to 100 ms early.
   - **The held-price rule.** v1 and v2 read the venue's live orders.
   - Fills, queues and refusals stay the venue's: the S4 rest lane's model is unchanged.
3. **A decision log** records every place, cancel, cap refusal and held-price wait with its time, side, rung and price.

## Controls (all passed)

| Control | Result |
|---|---|
| **Self-test.** Synthetic invariance cases: I1 is the USER's case (a 30-sh order refused or accepted at 40 ms under a 30-sh cap); I2 is an unseen fill; I3 is an unseen fill at a held price. v2's cases as well. Five planted errors: capacity released at submission (the USER's bug), "rests" on the venue's remaining, the held price on venue state, a leg's fills not conserved, priority broken | all five caught |
| **Decision invariance on real data.** Days 10-23, 10-31, 11-03; cuts at 20 s and 40 s. Every venue event after the cut is changed (frames one tick up and resized, trades doubled). Arms LDG / FB25, caps hard 2 / none, M 50. Every decision up to the cut must be identical | 6,912 checks, 9,835,924 decisions compared, **0 differ** |
| Conservation and accounting on every run: leg, order, live, priority, collision, position, cap | 163,296 runs, 0 violations |
| Regression: compat mode (v1's venue-state decisions) at soft 2 and no cap equals v1's fills row for row | 9 days x 3 arms x 3 clocks, 0 differ |
| Compat independent-mode parity with the per-rung lanes | 15,552 checks, 0 differ |
| Hard caps: max true position | exactly the cap: 60 / 150 / 300 sh |
| **POST HOC: true exposure under the hard caps** (see below) | 93,312 runs, 0 over the cap: **HARD-EXPOSURE-OK** |

**A correction to v3's printed header.**
- The header says "max true exposure under the hard caps". The lane records the largest true |position| only.
- The cap's claim is about exposure: the position plus every share of ours that can still fill.
- The post hoc control measures exposure directly. It counts each order from submission until its cancel lands or it is fully filled.
  - soft 2 is the positive control: it is known to overshoot, so the check must flag it.
  - In the self-test, a planted position-only statistic (what the header reported) must fail.

Result (9 days; client-state runs of every arm and clock; 13,152,371 accepted submissions read):

| Cap | Cap, sh | Max exposure, sh | Runs over the cap | Max position, sh | Per window-side max exposure, p50 / p95 / p99 |
|---|---|---|---|---|---|
| hard 2 | 60 | 60.0 | 0 of 23,328 | 60.0 | 60 / 60 / 60 |
| hard 5 | 150 | 150.0 | 0 of 23,328 | 150.0 | 150 / 150 / 150 |
| hard 10 | 300 | 300.0 | 0 of 23,328 | 300.0 | 300 / 300 / 300 |
| soft 2 (positive control) | 60 | 419.8 | 23,319 of 23,328 | 239.8 | 330 / 390 / 408 |

- **HARD-EXPOSURE-OK:** the hard caps bound true exposure, not just the position.
- The positive control fired on 9 of 9 days.
- In the median window-side the exposure reaches the cap at some moment, even at 10 x size: when the best moves, every rung is re-priced and
  new orders are placed while the old ones still count (until the cancel is confirmed). So the cap binds briefly in most windows.

## Design

- **Lane.** Both sides in one lane, rungs 0-5 x 30 sh, shared position. Arms:
  - LDG: pull every rung on the guard;
  - FB25: FUTURE-BEST's keep rule, with rung 0 at the best always pulled;
  - OFB2: the same rule with the realized 2 s move (perfect foresight).
- **Caps.**
  - hard K = 2, 5, 10 (pre-trade, client-known, above);
  - v1's soft 2 (a delayed trigger on the known position);
  - none.
- **What a hard cap means for a 6-rung ladder.** Every resting order counts, so at a flat position:
  - hard 2 (60 sh) allows two 30-sh orders per side;
  - hard 5 allows five;
  - hard 10 allows all six plus 120 sh of position.
  - A refused placement is retried at every 50 ms step. The "refused" column counts these rung-steps.
- **Clocks and values.**
  - Fill clocks M 0 / 50 / 250.
  - Each fill counted once, valued at 1 s and 5 s, at the 60 s mark and at settlement.
  - Headline: WS-timed trades without fills on orders < 100 ms old (C1 / C12).

## Results

WS $/day without young fills, at 1 s unless stated:

| Arm, cap | M 0 | M 50 | M 250 | E2 trades, M 50 | All trades, M 50 | All fills marked at 60 s, M 50 | Settlement, M 50 | Max position per window p50 / p95 / max, M 50 | Refused rung-steps a day, M 50 |
|---|---|---|---|---|---|---|---|---|---|
| LDG, hard 2 | +32 | +14 | -358 | -118 | -104 | -85 | -21 | 47 / 60 / 60 | 1.36 M |
| LDG, hard 5 | +58 | +30 | -576 | -146 | -117 | -123 | +52 | 60 / 149 / 150 | 0.44 M |
| LDG, hard 10 | +74 | +47 | -675 | -148 | -101 | -101 | +77 | 67 / 227 / 300 | 36 k |
| LDG, soft 2 | +53 | +32 | -542 | -137 | -104 | -87 | +21 | 60 / 120 / 210 | - |
| LDG, none | +83 | +54 | -691 | -146 | -92 | -92 | +77 | 67 / 242 / 928 | - |
| FB25, hard 2 | +49 | +26 | -498 | -156 | -130 | -143 | -121 | 47 / 60 / 60 | 1.27 M |
| FB25, hard 5 | +104 | +63 | -921 | -255 | -192 | -251 | -186 | 70 / 148 / 150 | 0.38 M |
| FB25, hard 10 | +147 | +100 | -1,100 | -279 | -179 | -271 | -124 | 90 / 248 / 300 | 43 k |
| FB25, soft 2 | +106 | +73 | -880 | -240 | -167 | -201 | -215 | 65 / 123 / 230 | - |
| FB25, none | +169 | +120 | -1,154 | -276 | -155 | -250 | -182 | 90 / 283 / 1,310 | - |
| OFB2 (perfect), hard 2 | +93 | +77 | -284 | -88 | -12 | +30 | +61 | 45 / 60 / 60 | 1.26 M |
| OFB2, hard 5 | +251 | +226 | -392 | -73 | +153 | +186 | +342 | 66 / 146 / 150 | 0.35 M |
| OFB2, hard 10 | +362 | +335 | -386 | -39 | +295 | +358 | +477 | 90 / 234 / 300 | 34 k |
| OFB2, soft 2 | +298 | +271 | -293 | -42 | +229 | +312 | +355 | 66 / 120 / 231 | - |
| OFB2, none | +386 | +358 | -381 | -30 | +328 | +395 | +454 | 90 / 269 / 1,116 | - |

**R613 KEEP-UNDER-HARD-CAP** (declared: above LDG at M 0 AND M 50 on ≥ 7 of 9 days):

| Pair, cap | Reading | M 0 ($/day, days up) | M 50 | M 250 |
|---|---|---|---|---|
| FB25 vs LDG, hard 2 | **NOT-SHOWN** | +17, 8/9 | +12, 4/9 | -140, 0/9 |
| FB25 vs LDG, hard 5 | **PAYS** | +46, 9/9 | +33, 7/9 | -345, 0/9 |
| FB25 vs LDG, hard 10 | **PAYS** | +73, 9/9 | +54, 7/9 | -425, 0/9 |
| OFB2 vs LDG, hard 2 | PAYS | +61, 9/9 | +63, 9/9 | +75, 9/9 |
| OFB2 vs LDG, hard 5 | PAYS | +193, 9/9 | +197, 9/9 | +184, 9/9 |
| OFB2 vs LDG, hard 10 | PAYS | +288, 9/9 | +288, 9/9 | +289, 9/9 |

**R615 CLIENT vs VENUE-STATE decisions** (printed): client-known minus v1's venue-state decisions, WS $/day w/o young at 1 s, M 0 / M 50 /
M 250:
- LDG: soft 2 -0.0 / -0.1 / +0.2; none -0.5 / -0.4 / -0.3.
- FB25: soft 2 -0.5 / -0.9 / +1.2; none -0.7 / -0.1 / +5.7.
- OFB2: soft 2 -4.6 / -4.2 / -2.7; none -8.2 / -8.2 / -5.2.

**Per day, M 50, hard 2** (WS $/day w/o young at 1 s; LDG | FB25 | OFB2):

| Day | LDG | FB25 | OFB2 |
|---|---|---|---|
| 10-23 | +24.4 | +18.6 | +85.0 |
| 10-24 | +52.3 | +94.6 | +136.3 |
| 10-25 | +62.0 | +116.5 | +133.8 |
| 10-26 | +13.6 | +11.3 | +82.4 |
| 10-27 | +2.2 | +1.0 | +77.1 |
| 10-31 | -2.1 | +14.6 | +39.7 |
| 11-01 | -21.7 | +7.9 | +27.4 |
| 11-02 | -2.3 | -4.9 | +62.2 |
| 11-03 | -4.4 | -25.0 | +44.9 |

**C1 / C12, M 50, hard 2** ($/day at 1 s by order age, WS: < 100 / 100-500 / ≥ 500 ms || E2, same):

| Arm | WS < 100 ms | WS 100-500 | WS ≥ 500 | E2 < 100 | E2 100-500 | E2 ≥ 500 |
|---|---|---|---|---|---|---|
| LDG | +1.1 | +3.9 | +9.8 | -8.6 | -29.3 | -88.2 |
| FB25 | +1.2 | +1.8 | +24.3 | -1.7 | -21.2 | -134.8 |
| OFB2 | -2.0 | -2.4 | +78.9 | -8.0 | -29.2 | -59.0 |

**POST HOC readings** (chosen after seeing the table above; reconciliation: all 135 printed client cells recomputed, 0 differ).

PH1 / PH2: FB25 minus LDG, $/day w/o young at 1 s (days FB25 higher / lower of 9):

| Cap | All trades, M 0 | All trades, M 50 | All trades, M 250 | E2 only, M 0 | E2 only, M 50 | E2 only, M 250 | Label, all trades at M 50 |
|---|---|---|---|---|---|---|---|
| hard 2 | -6.3 (4/5) | -26.2 (3/6) | -293.1 (0/9) | -23.5 (0/9) | -38.5 (0/9) | -153.2 (0/9) | MIXED |
| hard 5 | -21.9 (4/5) | -75.6 (2/7) | -745.5 (0/9) | -67.7 (0/9) | -108.7 (0/9) | -400.3 (0/9) | BELOW |
| hard 10 | -1.8 (4/5) | -77.3 (2/7) | -958.7 (0/9) | -75.2 (0/9) | -130.9 (0/9) | -533.2 (0/9) | BELOW |
| soft 2 | -5.4 (4/5) | -62.6 (2/7) | -755.3 (0/9) | -58.3 (0/9) | -103.0 (0/9) | -417.1 (0/9) | BELOW |
| none | +14.3 (5/4) | -63.4 (2/7) | -1,028.9 (0/9) | -71.8 (0/9) | -129.9 (0/9) | -566.1 (0/9) | BELOW |

PH3: WS $/day w/o young at 1 s, M 50, mean of 10-23..10-27 | mean of 10-31..11-03:

| Cap | LDG | FB25 | OFB2 |
|---|---|---|---|
| hard 2 | +30.9 / -7.6 | +48.4 / -1.8 | +102.9 / +43.6 |
| hard 5 | +57.3 / -4.9 | +115.4 / -3.1 | +286.8 / +151.1 |
| hard 10 | +80.4 / +4.5 | +164.3 / +20.1 | +407.3 / +243.5 |
| soft 2 | +60.3 / -2.7 | +125.7 / +6.3 | +339.7 / +184.0 |
| none | +86.1 / +13.6 | +183.8 / +41.1 | +432.9 / +264.4 |

## What it means

1. **The decision-timing leak is closed, and it was small in dollars.**
   - No decision changes when only future venue events change (0 of 9.8 M compared).
   - Re-deciding v1's runs on client-known state moves LDG and FB25 by under $1 a day at M 0 / 50 (−0.9 to 0.0) and the oracle by −$3 to
     −$8 (R615).
   - v1's numbers stand to that precision.
   - v2's refusal-release bug was in v2's hard mode only, which never produced a result.
2. **A cap that cannot be exceeded is much tighter than the soft cap, because every resting share counts.**
   - At 2 × size only two 30-sh orders per side can be out at a flat position: the ladder is effectively two rungs.
   - LDG earns +$14 a day at M 50 under hard 2 against +$32 under soft 2, which reached 210 sh of position.
   - **USER, after this run:** "We can use a much larger cap for testing, I mean there will be multiple ladders, so the potential cap can be
     much larger."
   - So hard 2 / 5 mostly test the cap. The relevant cells here are hard 10 (the full ladder plus 120 sh of position) and no cap. The
     ORDER-HARM policy replay runs at hard 10 / 20 / 50 and none.
3. **Where the cap is large enough, keeping (FB25) beats pulling everything at M 0 / 50 and loses at M 250.**
   - hard 10: +$54 a day at M 50 (7/9 days), +$73 at M 0 (9/9), −$425 at M 250 (0/9).
   - No cap (v1's comparison, re-decided here): FB25 +$120 vs LDG +$54 a day at M 50 (the printed cells).
   - At hard 2 the gain does not show (+$12 at M 50, 4/9).
4. **With perfect foresight keeping pays at every cap and every clock, M 250 included** (R613 over M 0 / 50 / 250: +$61 to +$75 a day at
   hard 2, +$184 to +$197 at hard 5, +$288 to +$289 at hard 10; 9/9).
   - Under a shared position and a strict cap, the keep strategy is right and its value grows with the cap; prediction is the limit.
5. **Over all trades keeping loses to pulling everything** (PH1, post hoc).
   - At M 50 FB25 is below LDG on 7 of 9 days at hard 5 / 10, soft 2 and no cap (-$63 to -$77 a day).
   - On E2-timed trades it is below on 9 of 9 days at every cap and clock.
   - Its WS gain (+$54 a day at hard 10) is smaller than its extra E2 loss (-$131 a day at hard 10, M 50).
   - Whether E2-timed fills are real is the timing question (C12): their anchors are uncertain by about +-250 ms, which is the same order as
     the keep decision.
6. **The WS gains sit mostly in the first five days** (PH3, post hoc).
   - At hard 10 FB25 makes +$164 a day on 10-23..10-27 and +$20 on 10-31..11-03; LDG +$80 and +$4.5.
   - The oracle keeps +$244 a day in the later block: the keep strategy still has room there, but the replay's simple pulling and FB25
     earn little.
7. **Young fills do not drive these numbers.**
   - WS fills on orders < 100 ms old are about $1 a day; the WS gains are on orders ≥ 500 ms old.
   - The E2-timed losses are on old orders too.

## Limits

- Replay rule M (B1a); inflated volumes (C4); consumed days (C8).
- The guard ladder, not the frozen v3 (C11).
- The lane's model of when a refusal is known (at activation) and the 100 ms notice for fills and cancel confirmations are fixed choices.
- Upstream inputs (the guard, P) are taken as known at their steps.
- Settlement marks rest on one binary outcome per window.

Outputs:
- `data/predict_fun/research/joint_ladder_v3_20261114/`:
  - joint_ladder_v3.txt / .json, fills_*, inv_*, day_*.json;
  - invariance_*.json, parity_*.json;
  - exposure_* and exposure.txt / .json (post hoc);
  - joint_ladder_v3_posthoc.txt / .json (post hoc);
  - logs.
- Copies (txt / json / logs) in `reviews/joint_ladder_v3_20261114/`.
