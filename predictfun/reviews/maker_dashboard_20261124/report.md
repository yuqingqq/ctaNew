# Market microscope — three saved market comparisons

USER task: compare our actions, market conditions and top-maker behavior in a usable frontend, to investigate queue/churn and why we lose. This packet adds no strategy replay, fit, market day, validation or live activity. Three known examples were fixed before extraction. As of 2026-11-24.

Open [the portable interactive dashboard](dashboard.html) (works directly from disk, no network), or serve this directory and open `dashboard.html?market=3789484`. [Desktop preview](desktop.png). Select a market/maker/side, switch current/front assumptions, zoom, click fills/orders, and follow linked maker-order candidates. Marker hard windows, inferred post bounds, original sizes where available, our received-book structural reasons, queue samples, exact commands and inventory/value paths are retained. Histories are not turned into identified maker cancels.

Source HEAD: e8c3f3a70c8c246ecf4fc30b4a5e7ed68ca1c307. Initial extraction freeze fbce28223d06d9640193ad8a0ec06bdf57f75058; superseding pre-output adapter freeze d436b11d87d85b98c083d952657aad15d5ed200f. All underlying inputs have hashes in manifest_v2.json; standalone decoded data equals the checked projection. No external messaging or push.

## Material finding: inventory hysteresis blocks the wrong side

**P1 — outcome interpretation affected; fix before drawing stronger queue conclusions.** `pf_branch_ladder.py:55–65` resets only when `abs(I) <= R`, unless the opposite S threshold is reached. A batch of fills can jump entirely over that reset band. Old LONG_RESTRICTED can persist while inventory is short (or vice versa). `blocked` at lines 287–289 then prevents the inventory-reducing side; sizing at lines 294–307 can reduce the other side below the one-share minimum. This can leave neither side quoting. No implementation was edited.

`check_inventory.py` independently sums all notified fills at every stored tick in all six arms and reconciles the saved known position. It executes only the source's pure next_mode function: LONG_RESTRICTED with I=-99.92717 stays LONG; controls I=-30 reset and I=-101 switch. This is a post-hoc saved-evidence diagnosis, not a new strategy candidate or a claim about the corrected P&L.

Current arm, measured 50 ms grids (durations stop at supported END=298s):

| Market | Wrong-side restriction seconds | Of those, no live quotes seconds | Entry jump, shares; interval seconds |
|---|---:|---:|---|
| 3702960 | 268.15 | 263.95 | 29.85–298.00: +102.07 → -97.93 |
| 3789484 | 27.85 | 3.50 | 138.50–166.35: -83.19 → +57.53 |
| 3877472 | 165.65 | 157.95 | 132.35–298.00: +80.66 → -54.21 |

For 3702960, the known inventory jumps +102.07283 → -97.92717 at 29.85s, then reaches -99.92717; LONG_RESTRICTED persists through END. For 3877472 the same trap persists from132.35s toEND. The front-at-activation arms have different trap durations (268.15 /0 /15.35s), so their feedback-policy contrasts combine queue differences with this controller defect. Accounting remains the arithmetic of the policy that actually ran; it is not an estimate of the intended corrected policy.

## What the three markets show

USD, same supported numerical interval and common END mark. Recorded top-three includes maker+taker legs and self-trade fees, zero-start view. Our fills are simulated, maker fills recorded with different clock certainty. These are marked values, not settlement P&L or spread-only profit.

| Day / market | Our current END | Top 3 recorded END | Our hypothetical front END | Our generations / cancel requests |
|---|---:|---:|---:|---:|
| 20261024 / 3702960 | +29.42 | +6.54 | +29.19 | 11 / 8 |
| 20261026 / 3789484 | -17.90 | +657.68 | +47.90 | 554 / 482 |
| 20261027 / 3877472 | -6.54 | +100.88 | -50.80 | 169 / 130 |

- **3702960:** a profitable own replay, useful as a counterexample to “we always lose because of churn.” Maker #3 sells4sh at24¢ in hard window56.303–58.303s. Its10s and60s markouts are both+2.5¢/share. Our same/equal-better ask is absent throughout that window. Inventory is about-99.93 while the stale LONG mode remains; ask slot reasons are zero_desired. This missed overlap is not simply a queue race. G4 estimated posting window57.653–58.303s is a scenario, not proof of650ms continuous resting or priority.
- **3789484:** 554 generations and482 cancel requests (316 structural,161 pressure,5 inventory). Maker #1's10sh46¢ bid fill has hard window269.035–271.035s and G4 estimated post269.998–271.035s. Our bids traverse61/60/59/53/52/49¢ during that bound; no46¢ order overlaps. Our equal-or-better bid is present only part of the bound. The viewer links the observed32¢ successor candidate, without asserting an identified cancellation. Queue-front sensitivity improves this market, but includes different inventory/controller paths.
- **3877472:** maker #1's59¢ ask trades50.187169sh by29.176s, with further same-hash fills. Our own59¢ ask (gid49) is already live24.71–37.51s, covering the entire27.176–29.176s hard match window; the sidebar exposes its seeded/first-review queue. This is compatible with queue disadvantage, but public data do not identify the maker's true priority. Merely assuming front priority worsens this market's END value; it is not sufficient for profitability.

The examples are illustrative and outcome-aware navigation, not a representative or matched selection test. The whole supported market trace remains available. Missing marks remain missing. Good10/60s fills are not proof of profitable complete turnover. Maker zero-start flow is not actual wallet inventory.

## Verification and limitations

Independent artifact check: six arms,2,297 own fills,1,819 own generations,2,198 recorded top-maker legs;18 planted wrong-side/price/time/value/front/queue corruptions refused. Source check: all2,198 maker legs,1,685 hard maker match windows,1,280 enabled scenario post windows and450 public-book samples;three altered source markouts refused. Source counts reconcile (10 recorded legs outside the shown interval on Oct27). Extra structural fields equal all six raw logs;18 signed-order context records equal the hash-bound source. Raw known inventory is independently reconciled at every stored grid. Browser desktop/mobile tests exercise all markets, arms, examples, zoom, side controls, event drilldown and keyboard; no JS errors or horizontal overflow. Portable HTML makes no network requests.

Preserved failures: the first extraction refused before output because it excluded self-trade fees included by the accepted totals. Amendment v2 retains flagged self legs and handles verified empty ranks. The v2/v3 checker failures came from summing six-decimal display quantities with exact tolerances/full-fill thresholds. v4 uses explicit rounding bounds for sums and raw exact fills for completion; projection unchanged. Logs and old source are kept. These checks establish faithful display, not predictive skill or profitability.

Browser environment: Chromium cannot initialize within run_light's4GiB virtual-address limit. Research extraction/checks all used PF_SLOT=codex/run_light. Rendering-only Chromium used core8 separately with timed RSS and no new research calculation, while the Playwright driver stayed in run_light. A conservative browser charge includes its renderers; this is recorded in resources.json. No resource limits were changed on research jobs.

Largest limitation: the saved controller's sign-crossing restriction defect materially affects participation and feedback; in addition, maker private cancellations and continuous queue positions remain unidentified. The chart shows source facts/bounds, not a proven maker decision policy.

**ONE next priority:** the development session should fix sign-crossing hysteresis release, add a batch-fill control that jumps across the entire deadband, and rerun exactly these three fixed markets for CURRENT_ASYNC_FAST10 and FRONT_ASYNC_FAST10 with the same ledgers/metrics. Compare participation, quotes on the inventory-reducing side, turnover/residual and common-END value before interpreting a queue benefit. This is a new strategy replay and needs a separately bounded authorization; the current approved0.05h packet is saved-data analysis only. Do not tune thresholds or expand days to hide the failure. The dashboard and diagnosis are complete; no replay is launched here.
