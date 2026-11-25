# Market microscope

The [ten-market dashboard](dashboard.html?market=3838531) keeps the three original markets and adds seven with directional 30-second moves in the YES midpoint. Select an arrow-labelled market and click the signed price-change button to jump to that window. The [Maker #3 comparison](maker3-research.html?market=3838531&from=245&to=275) has baseline and wider-band overlays for the same ten markets. The seven windows were selected for illustration using public prices on a one-second grid. Each selected move covers at least 10¢ and its net change is at least 60% of the cumulative absolute movement.

2026-11-25: [Maker3 calm/volatile versus book-spread figure](maker3-regimes.png), [vector version](maker3-regimes.svg), and [full attribution receipt](../maker3_regime_spread_20261125/report.md). Uses the same19 complete markets. Profitable volatile exits are concentrated in two Oct26 markets; volatile-entry residuals lose. These are accounting associations, not a recovered quote rule.

2026-11-24: [Maker #3 trend, spread and inventory](maker3-research.html) adds all19 complete markets, a public-price-selected trend, the last two minutes of3789484, native take directions, execution-window book clocks, completed turnover versus residual risk, and the matched taker-condition screens. [Receipt](../maker3_spread_risk_20261124/report.md). Through the existing SSH tunnel open `http://127.0.0.1:8765/maker3-research.html`. The page is self-contained and downloadable; the existing service remains loopback-only.

Open `dashboard.html` directly in a modern browser. It is self-contained and was tested from `file://` with no network requests. Start with Oct26 /3789484 and “Inspect maker example”; follow the linked next order or click our fills/orders. All ten markets have a saved current and hypothetical front arm.

Optional local server:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory /home/yuqing/ctaNew/predictfun/reviews/maker_dashboard_20261124
```

Then open `http://127.0.0.1:8765/dashboard.html?market=3789484`.

The compact payload was independently checked against saved source artifacts. `index.html`, `app.js`, `style.css`, `data.js`, `extra.js` are the editable multi-file viewer. `pack.py` produces the portable file without changing decoded data. The old extractor/checker attempts remain as provenance; use extract_v2 / check_v4 for the final projection, plus check_sources and check_inventory. Do not overwrite the write-once extraction or blindly rerun publish.py.

Read report.md before using the pictures for a research conclusion. In particular, this packet found a material inventory-hysteresis defect in the saved controller. The quotes/values faithfully show that executed policy, not a corrected one. Maker timestamps remain bounded and inferred; their private pulls and queue positions are not observed.

2026-11-24 clock clarification: [the user-named 63c fill](fill_clock_report.md) is a recorded multi-level execution. The older 59c/61c depth snapshot was 449ms before its marker. The viewer now states the underlying snapshot time and age explicitly; refresh the page to load the wording correction. Research data and accounting are unchanged.

[61c follow-up](ask61_report.md): 24.59 shares traded earlier; the subsequent 205.41-share disappearance has no overlapping recorded execution and is a likely withdrawal, conditional on the existing clock/capture assumptions. Its owner and private cancel command remain unidentified.

[Maker3 first-minute turnover](first_minute_report.md): the user-observed first30s buy-low/sell-high pattern reconciles to completed passive turnover. The full minute includes losing cycles and carried inventory; its common-mark value is +$28.14 versus our simulated −$22.08. This is an illustrative selected minute, not proof of long rest or an identified posting rule.

[Maker3 quote reconstruction](maker3_quotes_report.md): all17 filled-order hashes in that minute, with estimated posting windows and51 fills. Same-price bid replenishment and a later58c ask are conditionally ordered; a simultaneous ladder, exact original sizes and private cancels remain unidentified. The conversation's quote map shows uncertainty bars, not resting durations.

[Opening-loss diagnosis](opening_fill_report.md): our50/51c asks fill before the executed pressure rule signals danger; no cancel is pending. Maker3's56c buys and later58–63c sales are different price/side opportunities. This establishes the saved replay's mechanism, not maker3's private warning signal or a measured latency advantage.
