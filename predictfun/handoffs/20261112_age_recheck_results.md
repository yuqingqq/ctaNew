# AGE: QR2 / GAP re-checked against B1a's young-order finding (results, 2026-11-12)

> **Causal-clock note (2026-11-12, FINDINGS.md C6; independent review handoffs/20261112_new_results_insights.md, 9053263d):** R312 / R313 for F30 and Y+F30 inherit QR2's acausal 30 ms schedule and are VOID pending a causal rerun. The young-order split itself (R310, R311 for v3 and Y) is unaffected.

Session ce68ebe0. USER: "The simulation has some issues, profits are mainly from short-lived orders, but in practice it's hard for
us to fill those orders, we have already checked this previously, can you check that again".

Script and run:
- Script `research/pf_qr2_age_check.py`, driver `research/run_age.sh` (64638440, committed before the first run).
- Run 11:40-11:41Z in the claude slot.
- QR2's arms on the 9 consumed replay days (first 60 s), split by the order's age at the fill.
- Post hoc diagnostic.

Controls passed:
- Every arm reproduces QR2's committed day totals.
- The age bins add up to the totals.
- A planted bin-edge error is caught.

B1a (20261110_b1a_live_orders.md) replayed live's own 11-08 orders through the same fill model. Of the replay's shares, the part that
came from the taker orders that really filled live:

| Order age at fill | M 0 | M 50 | M 250 |
|---|---|---|---|
| < 100 ms | 8.3% | 12.3% | 0.7% |
| 100-500 ms | 54.8% | 28.6% | 7.7% |
| >= 500 ms | 46.8% | 47.7% | 22.0% |

## Answer: confirmed. Three quarters of the replay profit is on orders under 100 ms old

$/day at 1 s by the order's age at the fill (c/sh):

| Arm, M 50 | All | < 100 ms | 100-500 ms | >= 500 ms | Without < 100 ms | B1a-weighted |
|---|---|---|---|---|---|---|
| v3 | 14.73 (+1.03) | 11.03 (+1.27) | 3.49 (+0.80) | 0.21 (+0.16) | 3.70 | 2.46 |
| Y (no old levels) | 14.88 (+1.34) | 10.94 (+1.53) | 3.58 (+1.09) | 0.36 (+0.55) | 3.94 | 2.54 |
| F30 (30 ms joins) | 23.30 (+0.92) | 17.20 (+0.98) | 5.52 (+0.88) | 0.59 (+0.35) | 6.11 | 3.98 |
| Y+F30 | 23.45 (+1.05) | 17.15 (+1.07) | 5.60 (+1.06) | 0.71 (+0.70) | 6.31 | 4.05 |

| Arm | M 0: all / without < 100 ms / B1a-weighted | M 250: all / without < 100 ms / B1a-weighted |
|---|---|---|
| v3 | 21.21 / 4.94 / 4.01 | -11.85 / -4.13 / -0.66 |
| Y+F30 | 31.69 / 8.53 / 6.52 | -5.57 / +0.50 / -0.04 |

Readings:
- **R310 YOUNG-DRIVEN for every arm:** 73-77% of the profit at M 0 and M 50 comes from orders under 100 ms old.
- **The profit that is left is small.**
  - R311: without those fills every arm stays positive (SURVIVES; v3 on 9 / 7 days at M 0 / M 50), but v3 falls to $3.7 a day at
    M 50 and $4.9 at M 0.
  - Weighted by B1a's real shares, v3 is worth $2.5-4.0 a day.
  - Orders 500 ms or older earn about nothing: v3 $0.2-0.5 a day (R311b FAILS for v3).
  - At M 250 nothing earns.
- **R312:** QR2's speed gain survives without the young fills, but it is a third of the headline.
  - Y+F30 beats v3 on 9 of 9 days at both clocks by +2.6 $/day at M 50 and +3.6 at M 0, against +8.7 / +10.5 with them.
  - The weighted comparison agrees (R313 WEIGHTED-ABOVE, 9 of 9).
  - The age cut's own gain (Y) is young only (GAIN-IS-YOUNG): without young fills it equals v3.
- **R314:** without young fills, our per-share value against the top 5's real +0.10..+0.31 c/sh is MIXED for v3 (above on 8 / 6
  days) and still higher for Y+F30 (9 / 8 days).
- The cancel-pending part is $2.1 of v3's $14.7 at M 50 (294 sh a day). B1a found the replay credits 2.5 times live's pending
  shares.

## What it means

- QR2's $/day and GAP's per-share lead mostly stand on young-order fills that the replay cannot be trusted on. QR's table by
  level age is likely affected too (an order on a young level is a young order), but it was not re-checked here.
- Without them, v3 is worth a few dollars a day in the first minute, and nothing if trades match 250 ms before they are reported.
- The QR2 ranking survives in sign, at about a third of the size: joining new levels fast (+2.6 $/day) beats v3 without counting
  any young fill. That is because an earlier order sits further forward when the older trades come.
- B1a's live check also found live's real fills on young orders were not adverse (+0.05 c/sh under 100 ms). So "phantom" means the
  replay credits young orders with the wrong trades and far too many of them, not that young orders never fill.
- Whether young fills are real for a fast-entering strategy can only be measured live (the A4b capture). B1a's rates come from
  another engine (the churning live engine) on one day.

## Limits

- B1a's rates are one day of a different engine's orders.
- Rule M is not validated.
- Consumed days; post hoc.
- First minute only.

Outputs:
- `data/predict_fun/research/age_20261112/`: age.json / age.txt, day_*.json, logs.
- Copies in `reviews/age_20261112/`.
