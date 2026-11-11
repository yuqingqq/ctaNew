# Quote-default / learned pull — v4 correction

Supersedes v3 before any model fit. The previous own-order/calm experiment remains complete at d475eccb with no promoted variant.

Review found that historical same-price depth of zero was classified as missing. A newly formed price level was therefore unquotable for about 250 ms, contrary to quote-default entry. v4 treats observed zero prior depth as a neutral no-thinning indicator. It still refuses unavailable, stale or out-of-depth observations; positive-to-zero thinning still triggers pressure rules. A regression test verifies immediate admission and shows that planting the old function delays it.

Six complete v3 neutral build days are retained as provenance (Oct23–27 and Oct31), each with 288 windows per arm and exact reference/accounting checks. No learned model was fit. All v4 neutral paths and labels are rebuilt; model features, parameters, thresholds, nulls, dates and execution scenarios are unchanged. This is a correctness correction, not a result-selected new threshold. The former v3 full-day cache equivalence remains evidence about the unchanged feature cache only, not policy-output equality after this correction.

The driver runs controls, an eight-window smoke, nine neutral build days, six expanding model folds, six primary replay days, a primary report, six stress replay days, a final report, the complete artifact audit and the comparison figure. Each stage writes progress to HANDOFF.md, STATUS.yml, the progress handoff and append-only data logs. A failed stage stops later stages and keeps its artifacts.

Compare default quoting, pressure rules, a guard floor and market/queue pull-value models against both the matched guard architecture and frozen ladder v3. Evaluate paired marked value, spread, settlement/carry, inventory-square exposure, final inventory, union of resting intervals and cancellation activity. The figure uses verified paired populations and day-cluster intervals; it is generated only after the complete artifact audit passes.

All nine dates are consumed research data; no validation or deployment claim. Public maker fills do not reveal their cancellation policy, and execution/queue calibration remains unvalidated. No live changes. One light numeric process under existing resource limits. Canonical output: data/predict_fun/research/default_pull_20261111_v4. Source/declaration committed before new outcomes; prior receipts remain unchanged.
