# Market microscope

Open `dashboard.html` directly in a modern browser. It is self-contained and was tested from `file://` with no network requests. Start with Oct26 /3789484 and “Inspect maker example”; follow the linked next order or click our fills/orders. All three markets have a saved current and hypothetical front arm.

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
