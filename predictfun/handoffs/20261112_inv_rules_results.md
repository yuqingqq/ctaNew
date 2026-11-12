# INV-RULES: does the frozen v3's inventory logic move the profit test? (post hoc, 2026-11-12)

Session ce68ebe0. USER: "Can we recheck our quoting logic, I remember there are skew/inventory related logics that will influence
our quoting price, will that influence our profit test, if we optimize the timing now, we can disable those first".

Script `research/pf_inv_rules_posthoc.py` (0a1262f0 + a day-count line), 16:58-16:59Z in the claude slot.
- It reads existing outputs only, with no new engine run.
- The frozen engine's in-sample runs i2J / i2X already carry four arms that differ only in the inventory rules.
- **POST HOC, in-sample and descriptive: no readings were declared.** The v3 build chose CAP10 and RO60 on these same 9 days.
- Control passed: the v3 arm reproduces FV3's table.

## 1. The quoting logic (the frozen v3 arm, read in `pf_step4_ladder_v3.py`)

| Part | Rule | Inventory-driven? |
|---|---|---|
| Price and side | MICRO: microprice from the best-level sizes; with a one-tick spread G4 keeps the bid only when the bid queue >= the ask queue, the ask only when the ask queue >= the bid queue. Price = the best (or one tick inside a wider spread) | No: book imbalance |
| Rung | LD2: one order one tick behind the best, kept while within 2 ticks | No |
| Size | AM:5:25:95: 5 sh, 25 sh when v >= the minute's 95th-percentile line | No: signal |
| Guards | PM90 (per-minute line), GS (stale value), G1 / G2 (Binance 1 s / 250 ms), G3 (gap behind the touch), G4 | No |
| **CAP10** | The side that would increase \|net\| is withheld once \|known net\| >= 10 sh (touch and rung) | **Yes** |
| **G6** | Soft cap 2 x 25 = 50 sh, hard 5 x 25 = 125 sh (binds only without CAP10) | **Yes** |
| **RO60** | From 60 s only the side that reduces \|net\| is quoted, sized <= \|net\| | **Yes** |
| G5 / S240 | No quotes from 240 s | Time |
| XMIN1 / XNA / XV | Venue: $1 minimum order value, no amend, request throttle | Venue constraints, not ours to disable |

**There is no inventory price skew in v3.** The skew band (SK), the reducing-side exemption (RX), the exposure limit (RL) and
inventory sizing (A4) exist as engine tokens, but none of them is in v3's arm. The inventory logic only withholds quotes.

The J-v3 lane used by QR, CS, FA and GAP has none of these rules (one 5-sh order per side at the known best, `pf_kp_s4.py` docstring):
their timing results carry no inventory logic.

## 2. Effect on the profit test (in-sample arms of the frozen engine, $/day, 9 days)

Fills in the first 60 s (RO60 acts only after 60 s, so S60 = v3 here):

| Timing, arm | 1 s all | Without young | B1a M 0 / M 50 | Settlement (ties 50) | Shares/day | Cancel-pending |
|---|---|---|---|---|---|---|
| J v3 | 50.27 | 9.73 | 8.76 / 7.61 | 90.60 | 3,418 | 41.3% |
| J without CAP10 | 61.28 | 12.44 (8 of 9 days above) | 10.92 / 9.43 | 71.04 (4 above, 5 below) | 4,095 | 39.8% |
| X v3 | 39.46 | 8.81 | 7.43 / 6.16 | 50.39 | 2,876 | 40.5% |
| X without CAP10 | 44.40 | 10.02 (4 above, 5 below) | 8.46 / 6.81 | 40.22 (4 above, 5 below) | 3,353 | 39.0% |

Fills from 60 s (the reduce-only phase, RO60):

| Timing, arm | 1 s all | Without young | Settlement | Shares/day |
|---|---|---|---|---|
| J v3 | 9.86 | 0.63 | 8.87 | 598 |
| X v3 | 6.00 | 0.75 | 14.92 | 439 |
| J / X S60 (no quotes after 60 s) | -0.01 / -0.10 | 0.01 / -0.10 | -1.47 / -0.42 | 5 / 7 |

Inventory per day:

| Arm (J) | Mean \|net\| at the stop per window | Decisions withheld by CAP10 / RO / G6 soft / XMIN1 |
|---|---|---|
| v3 | 6.67 sh | 20,905 / 318,031 / 4,852 / 17,956 |
| Without CAP10 | 8.39 sh | 0 / 310,147 / 15,393 / 18,498 |
| S60 | 8.54 sh | 20,905 / 0 / 1,507 / 3,067 |
| Without CAP10, S60 | 10.61 sh | 0 / 0 / 4,742 / 3,681 |

What it shows:
1. **CAP10 is the only inventory rule that touches the first minute.**
   - It withholds about a sixth of v3's shares there (3,418 against 4,095 a day at J).
   - Without it, the 1 s value rises by $11 (J) / $5 (X) a day, and by $2.7 / $1.2 without young fills.
   - The rise without young fills holds on 8 of 9 days at J and only 4 of 9 at X.
   - Settlement moves the other way ($20 / $10 lower) but splits 4 / 5 by day: noise.
2. **RO60 does not change the first minute at all.** After 60 s it only reduces inventory. Those fills are worth $0.6-0.75 a day at
   1 s without young fills, and $9-15 at settlement.
3. **G6 binds only once CAP10 is gone** (15k decisions a day at J). No in-sample arm runs without G6, or quotes freely after 60 s.
   Both would need new engine runs (PF_ARMS only; the engine stays unedited).

## 3. For the timing work

- The J-v3 timing studies (QR, CS, FA, GAP) are unaffected: they have no inventory logic.
- The frozen-v3 timing numbers (FV3's order path +$2-5 a day without young fills) were measured with CAP10 on. A cap that blocks
  about a sixth of first-minute shares can hide part of a timing gain.
- A timing base for the frozen engine would be:
  - first 60 s only;
  - CAP10 and G6 off; RO60 / S240 do not matter inside the minute;
  - MICRO / G4, the guards, the rung and the size rule kept (they are the timing and side logic);
  - the venue rules (XMIN1, XNA, XV) kept, because they bind live.
  Judge it on fill markouts (1 s, 5 s), and add inventory control back afterwards with settlement as its check.

## Limits

- In-sample and descriptive: CAP10 and RO60 were chosen on these 9 days.
- Replay fill model rule M; young and cancel-pending fills unvalidated (B1a).
- Settlement is noisy at 9 days.
- `carry_stop_usd` in summary.json uses the registry's raw outcomes (ties as Up); the settlement columns above have ties at 50.

Outputs: `data/predict_fun/research/inv_rules_20261112/` (inv_rules.json / .txt); copies in `reviews/inv_rules_20261112/`.
