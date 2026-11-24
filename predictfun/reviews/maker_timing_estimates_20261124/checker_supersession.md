# Checker v2 — before its result

The frozen checker completed its valid-data interval/cause/gap/summary checks, then the planted exhausted-as-pull corruption raised TypeError in `eq(1, None)` instead of the ValueError expected by the falsifier runner. It wrote no checks.json. Keep logs/check.log as the failed attempt.

`check_v2.py` changes only missing/nonmissing equality refusal (ValueError), runs two explicit controls, and delegates every original check and falsifier unchanged. It also checks the report-only signed-chain rendering against the frozen intervals and identities. No builder, input, output, timing label, predicate or sample changes. Commit this wrapper before its run.

The report-only readout joins all88 fixed signed rows and renders market-relative endpoints; no new cases or conditions are selected. Its output and script are retained. The first-check exception affects the instrument's corruption receipt, not the measurements.
