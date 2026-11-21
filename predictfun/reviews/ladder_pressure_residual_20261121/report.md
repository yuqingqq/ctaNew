# PRESSURE-RESIDUAL-TIMING-AUDIT-v1 — report

2026-11-21 20:49Z, session aea5bfb9. This is review `e40bd01d`'s ONE next priority: a saved-evidence timing partition of the full pressure rule's fills.
- **Source arm:** FULL_11 of PRESSURE-ACTION-ATTRIBUTION-v1 (`54eab545`), which equals the accepted pressure arm.
- **Population:** 19 complete markets, 7,395 fills and 4,811 generations. All 96 coverage statuses are kept upstream.
- **Freeze:** `a10eeea3`, before any table.
- **Status of the evidence:** consumed Oct 24-27 development days; no replay and no tape load. It is a diagnostic partition: no predictive-skill, avoided-loss, threshold or strategy claim.

## Reconciliation and integrity

- **Exact once:** every FULL_11 fill is assigned exactly once. Per market, the fill count, shares, END (= the ledger value), eligible +1 s and edge equal the accepted FULL_11 row (`reconcile` refuses otherwise).
- **Mismatches:** 0 event or action mismatches. A timely evaluable ON that was kept, or a fill at or after its removal, would be one.
- **Thin days:** 0.
- **Status counts:** pending_pressure 75, pending_inventory 5, pending_structural 321, before_feasible_effect 422, unflagged_later 6,572, nonevaluable 0, event_or_action_mismatch 0.
- **Generations:** filled 1,790, no_fill 2,745, refused 276.

## The reading (declared rule)

**UNFLAGGED_EXPOSURE_DOMINANT.** The rows below are shares of the gross negative eligible +1 s value. Timing-limited means pending_* plus before_feasible_effect; unflagged means unflagged_later. Positive value and quantities are shown alongside.

| day | gross negative +1 s | gross positive +1 s | timing-limited share | timing-limited negative | its positive | its shares | its fills | unflagged share | unflagged negative | its positive | its shares | its fills | other negative |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | -56,260.6 c | +23,903.1 c | 28.33 % | -15,937.0 c | +1,827.0 c | 3,341.6 | 377 | 71.67 % | -40,323.6 c | +22,076.1 c | 30,418.5 | 2,857 | +0.0 c |
| 20261025 | -39,686.3 c | +21,689.8 c | 26.27 % | -10,426.0 c | +2,208.8 c | 2,274.6 | 244 | 73.73 % | -29,260.3 c | +19,481.0 c | 21,878.2 | 2,059 | +0.0 c |
| 20261026 | -36,567.6 c | +6,713.4 c | 19.54 % | -7,143.9 c | +797.6 c | 1,575.7 | 172 | 80.46 % | -29,423.7 c | +5,915.9 c | 14,880.5 | 1,261 | +0.0 c |
| 20261027 | -7,780.5 c | +2,824.3 c | 35.17 % | -2,736.3 c | +2.5 c | 444.3 | 29 | 64.83 % | -5,044.2 c | +2,821.8 c | 4,240.8 | 359 | +0.0 c |
| all | -140,295.0 c | +55,130.7 c | 25.83 % | -36,243.2 c | +4,835.9 c | 7,636.1 | 822 | 74.17 % | -104,051.8 c | +50,294.8 c | 71,418.0 | 6,536 | +0.0 c |

## By status (all days)

| status | generations | fills | shares | generation live time | execution edge | +50 ms | +100 ms | eligible +1 s | +1 s > 0 | +1 s < 0 | valid +1 s shares | censored fills | movement to END | late / missing END | END |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pending_pressure | 26 | 75 | 874.8 | 94.0 s | -270.5 c | -2,045.9 c | -2,430.8 c | -6,186.2 c | +0.0 c | -6,186.2 c | 874.8 | 0 | +8,395.9 c | +0.0 c | +2,209.7 c |
| pending_inventory | 5 | 5 | 35.4 | 19.6 s | +2.6 c | +23.5 c | +23.5 c | +12.1 c | +43.4 c | -31.2 c | 35.4 | 0 | -734.7 c | +0.0 c | -722.5 c |
| pending_structural | 154 | 321 | 2,434.5 | 322.4 s | -2,680.9 c | -5,417.1 c | -5,388.3 c | -9,321.2 c | +1,715.8 c | -11,037.1 c | 2,427.7 | 1 | +29,210.1 c | -13.6 c | +19,875.3 c |
| before_feasible_effect | 149 | 422 | 4,298.2 | 65.6 s | +1,034.7 c | -5,986.3 c | -6,406.3 c | -15,912.0 c | +3,076.7 c | -18,988.7 c | 4,298.2 | 0 | +30,718.4 c | +0.0 c | +14,806.4 c |
| unflagged_later | 1,602 | 6,572 | 71,756.6 | 5,636.0 s | +52,458.4 c | -14,228.3 c | -14,508.7 c | -53,757.0 c | +50,294.8 c | -104,051.8 c | 71,418.0 | 36 | -69,006.5 c | +485.1 c | -122,278.4 c |
| all | 1,936 | 7,395 | 79,399.5 | 6,137.6 s | +50,544.4 c | -27,654.1 c | -28,710.7 c | -85,164.3 c | +55,130.7 c | -140,295.0 c | 79,054.1 | 37 | -1,416.8 c | +471.5 c | -86,109.6 c |

- Generations and their live time are counted once per status: a generation with fills in two statuses counts in both, so the all row is not a distinct-generation count.

### By status and side

| side | status | generations | fills | shares | generation live time | execution edge | +50 ms | +100 ms | eligible +1 s | +1 s > 0 | +1 s < 0 | valid +1 s shares | censored fills | movement to END | late / missing END | END |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bid | pending_pressure | 12 | 41 | 432.9 | 42.6 s | -382.2 c | -1,357.3 c | -1,676.3 c | -3,396.9 c | +0.0 c | -3,396.9 c | 432.9 | 0 | +8,205.0 c | +0.0 c | +4,808.1 c |
| ask | pending_pressure | 14 | 34 | 441.9 | 51.4 s | +111.7 c | -688.6 c | -754.5 c | -2,789.2 c | +0.0 c | -2,789.2 c | 441.9 | 0 | +190.9 c | +0.0 c | -2,598.4 c |
| bid | pending_inventory | 4 | 4 | 34.1 | 16.4 s | +2.6 c | +24.1 c | +24.1 c | +13.4 c | +43.4 c | -30.0 c | 34.1 | 0 | -711.5 c | +0.0 c | -698.2 c |
| ask | pending_inventory | 1 | 1 | 1.2 | 3.2 s | +0.0 c | -0.6 c | -0.6 c | -1.2 c | +0.0 c | -1.2 c | 1.2 | 0 | -23.1 c | +0.0 c | -24.4 c |
| bid | pending_structural | 72 | 157 | 1,118.4 | 148.9 s | -1,400.9 c | -2,718.8 c | -2,622.2 c | -4,460.4 c | +92.7 c | -4,553.0 c | 1,118.4 | 0 | +35,790.3 c | +0.0 c | +31,329.9 c |
| ask | pending_structural | 82 | 164 | 1,316.1 | 173.5 s | -1,280.0 c | -2,698.3 c | -2,766.2 c | -4,860.9 c | +1,623.2 c | -6,484.0 c | 1,309.3 | 1 | -6,580.2 c | -13.6 c | -11,454.6 c |
| bid | before_feasible_effect | 74 | 220 | 2,103.6 | 29.5 s | +547.8 c | -3,431.2 c | -3,615.3 c | -8,963.8 c | +1,259.4 c | -10,223.2 c | 2,103.6 | 0 | +55,520.1 c | +0.0 c | +46,556.3 c |
| ask | before_feasible_effect | 75 | 202 | 2,194.6 | 36.2 s | +486.9 c | -2,555.1 c | -2,791.0 c | -6,948.3 c | +1,817.3 c | -8,765.6 c | 2,194.6 | 0 | -24,801.7 c | +0.0 c | -31,749.9 c |
| bid | unflagged_later | 794 | 3,225 | 35,765.5 | 2,828.9 s | +24,820.8 c | -8,552.8 c | -9,655.9 c | -28,596.0 c | +24,865.4 c | -53,461.4 c | 35,563.5 | 16 | +333,213.5 c | +202.4 c | +304,819.9 c |
| ask | unflagged_later | 808 | 3,347 | 35,991.0 | 2,807.1 s | +27,637.6 c | -5,675.5 c | -4,852.8 c | -25,161.0 c | +25,429.4 c | -50,590.4 c | 35,854.5 | 20 | -402,220.1 c | +282.7 c | -427,098.3 c |

### By status and birth rung

| rung | status | generations | fills | shares | generation live time | execution edge | +50 ms | +100 ms | eligible +1 s | +1 s > 0 | +1 s < 0 | valid +1 s shares | censored fills | movement to END | late / missing END | END |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | pending_pressure | 16 | 49 | 552.1 | 51.1 s | -130.4 c | -1,442.8 c | -1,576.7 c | -4,193.0 c | +0.0 c | -4,193.0 c | 552.1 | 0 | -4,027.6 c | +0.0 c | -8,220.6 c |
| 1 | pending_pressure | 10 | 26 | 322.7 | 42.9 s | -140.1 c | -603.1 c | -854.2 c | -1,993.1 c | +0.0 c | -1,993.1 c | 322.7 | 0 | +12,423.5 c | +0.0 c | +10,430.4 c |
| 1 | pending_inventory | 5 | 5 | 35.4 | 19.6 s | +2.6 c | +23.5 c | +23.5 c | +12.1 c | +43.4 c | -31.2 c | 35.4 | 0 | -734.7 c | +0.0 c | -722.5 c |
| 0 | pending_structural | 96 | 198 | 1,547.3 | 139.6 s | -1,341.3 c | -3,317.2 c | -3,431.3 c | -5,156.9 c | +1,592.2 c | -6,749.1 c | 1,540.5 | 1 | +19,561.5 c | -13.6 c | +14,391.0 c |
| 1 | pending_structural | 58 | 123 | 887.2 | 182.8 s | -1,339.6 c | -2,100.0 c | -1,957.0 c | -4,164.3 c | +123.7 c | -4,288.0 c | 887.2 | 0 | +9,648.6 c | +0.0 c | +5,484.3 c |
| 0 | before_feasible_effect | 124 | 353 | 3,415.2 | 59.9 s | +1,548.9 c | -3,637.8 c | -4,111.8 c | -11,503.4 c | +2,574.7 c | -14,078.1 c | 3,415.2 | 0 | +8,273.7 c | +0.0 c | -3,229.7 c |
| 1 | before_feasible_effect | 25 | 69 | 883.0 | 5.7 s | -514.2 c | -2,348.5 c | -2,294.4 c | -4,408.7 c | +502.0 c | -4,910.7 c | 883.0 | 0 | +22,444.7 c | +0.0 c | +18,036.0 c |
| 0 | unflagged_later | 872 | 3,619 | 37,613.3 | 2,538.7 s | +24,532.4 c | -5,712.9 c | -7,588.5 c | -27,550.3 c | +27,577.2 c | -55,127.5 c | 37,382.6 | 29 | -65,991.2 c | +422.7 c | -93,118.7 c |
| 1 | unflagged_later | 730 | 2,953 | 34,143.3 | 3,097.3 s | +27,926.0 c | -8,515.4 c | -6,920.3 c | -26,206.7 c | +22,717.6 c | -48,924.3 c | 34,035.4 | 7 | -3,015.4 c | +62.4 c | -29,159.7 c |

### By status and day

| day | status | generations | fills | shares | generation live time | execution edge | +50 ms | +100 ms | eligible +1 s | +1 s > 0 | +1 s < 0 | valid +1 s shares | censored fills | movement to END | late / missing END | END |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | pending_pressure | 10 | 37 | 443.8 | 47.5 s | -487.1 c | -1,559.9 c | -1,900.9 c | -3,127.5 c | +0.0 c | -3,127.5 c | 443.8 | 0 | +13,507.0 c | +0.0 c | +10,379.5 c |
| 20261025 | pending_pressure | 6 | 12 | 142.8 | 7.5 s | -57.4 c | -626.1 c | -674.9 c | -1,080.6 c | +0.0 c | -1,080.6 c | 142.8 | 0 | -5,841.2 c | +0.0 c | -6,921.8 c |
| 20261026 | pending_pressure | 7 | 11 | 101.6 | 22.4 s | +21.8 c | -112.1 c | -107.1 c | -1,550.0 c | +0.0 c | -1,550.0 c | 101.6 | 0 | +1,887.5 c | +0.0 c | +337.5 c |
| 20261027 | pending_pressure | 3 | 15 | 186.6 | 16.6 s | +252.2 c | +252.2 c | +252.2 c | -428.0 c | +0.0 c | -428.0 c | 186.6 | 0 | -1,157.4 c | +0.0 c | -1,585.5 c |
| 20261024 | pending_inventory | 2 | 2 | 7.8 | 5.5 s | +6.6 c | +9.3 c | +9.3 c | +2.0 c | +3.3 c | -1.2 c | 7.8 | 0 | -89.1 c | +0.0 c | -87.1 c |
| 20261025 | pending_inventory | 2 | 2 | 24.5 | 5.5 s | -5.5 c | +12.7 c | +12.7 c | +8.6 c | +38.6 c | -30.0 c | 24.5 | 0 | -761.0 c | +0.0 c | -752.5 c |
| 20261026 | pending_inventory | 1 | 1 | 3.0 | 8.6 s | +1.5 c | +1.5 c | +1.5 c | +1.5 c | +1.5 c | +0.0 c | 3.0 | 0 | +115.5 c | +0.0 c | +117.0 c |
| 20261024 | pending_structural | 67 | 132 | 935.2 | 152.7 s | -1,101.9 c | -1,850.1 c | -1,858.1 c | -2,182.9 c | +1,165.4 c | -3,348.3 c | 935.2 | 0 | +2,695.6 c | +0.0 c | +512.7 c |
| 20261025 | pending_structural | 50 | 117 | 913.1 | 80.0 s | -878.8 c | -2,150.5 c | -2,060.8 c | -4,526.3 c | +499.0 c | -5,025.3 c | 913.1 | 0 | +16,319.3 c | +0.0 c | +11,793.0 c |
| 20261026 | pending_structural | 35 | 69 | 574.6 | 86.8 s | -694.4 c | -1,400.6 c | -1,453.6 c | -2,596.2 c | +51.4 c | -2,647.6 c | 567.8 | 1 | +10,338.5 c | -13.6 c | +7,728.7 c |
| 20261027 | pending_structural | 2 | 3 | 11.6 | 2.9 s | -5.8 c | -15.8 c | -15.8 c | -15.8 c | +0.0 c | -15.8 c | 11.6 | 0 | -143.3 c | +0.0 c | -159.1 c |
| 20261024 | before_feasible_effect | 70 | 206 | 1,954.8 | 29.9 s | +507.8 c | -4,279.9 c | -4,327.3 c | -8,801.6 c | +658.3 c | -9,459.9 c | 1,954.8 | 0 | +15,524.5 c | +0.0 c | +6,722.9 c |
| 20261025 | before_feasible_effect | 45 | 113 | 1,194.2 | 21.8 s | +359.7 c | -115.8 c | -237.1 c | -2,618.8 c | +1,671.3 c | -4,290.1 c | 1,194.2 | 0 | +5,104.8 c | +0.0 c | +2,486.0 c |
| 20261026 | before_feasible_effect | 29 | 92 | 903.2 | 13.5 s | +651.7 c | -22.7 c | -268.9 c | -2,201.6 c | +744.6 c | -2,946.3 c | 903.2 | 0 | -6,659.0 c | +0.0 c | -8,860.6 c |
| 20261027 | before_feasible_effect | 5 | 11 | 246.0 | 0.5 s | -484.5 c | -1,568.0 c | -1,573.0 c | -2,290.0 c | +2.5 c | -2,292.5 c | 246.0 | 0 | +16,748.0 c | +0.0 c | +14,458.0 c |
| 20261024 | unflagged_later | 705 | 2,867 | 30,451.8 | 2,486.6 s | +23,105.0 c | -6,681.6 c | -6,970.6 c | -18,247.5 c | +22,076.1 c | -40,323.6 c | 30,418.5 | 10 | -32,349.2 c | +57.6 c | -50,539.2 c |
| 20261025 | unflagged_later | 488 | 2,082 | 22,133.2 | 1,704.1 s | +18,625.2 c | -4,040.3 c | -2,311.5 c | -9,779.3 c | +19,481.0 c | -29,260.3 c | 21,878.2 | 23 | -26,935.5 c | +488.1 c | -36,226.6 c |
| 20261026 | unflagged_later | 316 | 1,264 | 14,930.8 | 1,150.7 s | +8,270.4 c | -3,595.8 c | -5,162.4 c | -23,507.8 c | +5,915.9 c | -29,423.7 c | 14,880.5 | 3 | +14,970.8 c | -60.6 c | -8,597.6 c |
| 20261027 | unflagged_later | 93 | 359 | 4,240.8 | 294.6 s | +2,457.9 c | +89.4 c | -64.2 c | -2,222.4 c | +2,821.8 c | -5,044.2 c | 4,240.8 | 0 | -24,692.7 c | +0.0 c | -26,915.0 c |

### Concentration and the signal diagnostics

| status | markets with negative +1 s | top-3 market share of its absolute negative +1 s |
|---|---|---|
| pending_pressure | 13 | 55.64 % |
| pending_inventory | 2 | 100.00 % |
| pending_structural | 19 | 53.27 % |
| before_feasible_effect | 17 | 43.32 % |
| unflagged_later | 19 | 24.96 % |

| status | fills | signal ON at submission | never live | ON at the first live callback | mean feasible ticks | ON at the latest feasible tick | median fill time after activation |
|---|---|---|---|---|---|---|---|
| pending_pressure | 75 | 0.00 % | 0.00 % | 2.67 % | 0.0 | 0.00 % | 3.7 s |
| pending_inventory | 5 | 0.00 % | 0.00 % | 0.00 % | 0.0 | 0.00 % | 3.2 s |
| pending_structural | 321 | 0.00 % | 2.49 % | 0.00 % | 0.0 | 0.00 % | 0.8 s |
| before_feasible_effect | 422 | 0.00 % | 4.03 % | 0.71 % | 0.0 | 0.00 % | 0.0 s |
| unflagged_later | 6,572 | 0.00 % | 0.00 % | 0.00 % | 63.9 | 0.00 % | 1.7 s |

- An ON at submission or at the first live callback is descriptive. The pressure rule acts only through a live callback whose cancel can take effect before the fill.

## Review e40bd01d finding

- The ablation checker's lost/added components and pooled +1 s total are not reused by this audit, so the limitation is retained. The frozen ablation receipt is not edited; any later reuse must extend that checker with the three refusals.

## Validity

- **Checks** A / B / F / I: 10 / 10 true. This checker uses its own status code. Falsifier: 7 / 7.
- **Checker rehearsal** on the saved-input pilot: 9 / 9 true; B_membership is false by design.
- **Mutants and selftest:** mutation suite 19 / 19; selftest groups 1-5 pass.
- **Cost:** projection 0.038 of 0.12 core-hours. The round used 0.010 core-hours; the ladder line was at 1.806 of 2.0 at the checks.

## Limits

- **What the partition is.** It covers the full rule's fills under its own feedback path. Statuses describe when a cancel could have taken effect, not what an earlier cancel would have saved.
- **Replay assumptions.** These carry through from the saved evidence: unchanged external flow, M50 timing, gross zero-cost values and the modelled Binance delay.
- **Coverage.** The population is 19 of 96 markets on consumed days. No day here is a validation day.
