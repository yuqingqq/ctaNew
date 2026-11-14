# ORDER-HARM client-state / 50 ms repair

Implemented at **32da4014**, verified **2026-11-14 10:07:17 UTC**. Branch
`research/predict-order-harm-fix-20261114`, checkout
`/tmp/cta-order-harm-fix-20261114`. The three requested repairs are complete in
code. No historical experiment, training job, or forward-data read was started.

## Changes

1. **Labels and policy share client-state execution.**
   `pf_order_harm_client_lane.py` derives its engine from frozen JOINT-LADDER
   v3 and adds explicit active-side/rung-offset support for reference labels.
   Both the keep path and pull/re-entry fork now use it. The label population
   uses its complete order records instead of relying on callbacks from the
   legacy venue-state engine. A fill at 400 ms cannot remove the 450 ms
   decision when its notification arrives at 500 ms.

2. **Portfolio value includes the cancellation tail at a common terminal.**
   The replay stops quoting at 60 seconds, processes cancellation-pending
   executions, and values cash plus inventory at 60.2 seconds. It refuses
   fills after that terminal, outstanding executable orders, missing marks,
   and position mismatches. Every replay run reconciles the sum of fill
   values against independently accumulated cash plus marked inventory.
   The fill artifacts retain price, window, execution time and terminal;
   inventory artifacts retain terminal cash, position, mark and PnL.
   The former 60-second mark is kept only for the old-engine regression
   comparison, not for portfolio acceptance. Missing one-second markouts
   no longer remove otherwise valid terminal-PnL rows from the report.

3. **Assessment runs every 50 ms.**
   Both label decisions and model policy use `EVERY=1`. FB25C therefore
   matches FB25 at 50 ms, and the replay now requires their fills to agree.
   The generic cadence helper still has 250 ms control cases, including
   guard-episode reset checks. They are tests, not the model's operating
   cadence. The 250 ms re-entry cooldown and execution delays are unchanged.

Frozen JOINT-LADDER v3 and older ORDER-HARM sources/results remain as
provenance. The unrun v3 builder and policy v2 are updated in a new source
commit and use fresh output directories:

- `data/predict_fun/research/order_harm_v3_client50_20261114/`
- `data/predict_fun/research/order_harm_policy_v2_client50_20261114/`

Receipts carry `order-harm-client-50ms-terminal-tail-v1` and `decision_ms: 50`.
Fits, policy folds and reports reject incompatible receipts. Previous 250 ms
labels/models must not be reused. Both drivers run the new synthetic suite
before their existing controls and any data work.

## Verification

**Nine targeted tests passed**, including an actual label build on an invented
window with all data sources supplied in memory. The checks cover:

- the unobserved-fill decision prefix and a known-information positive control;
- both sides and multiple reference rung offsets, including pull-fork re-entry;
- 12 default-engine cases matching frozen v3 exactly;
- all five existing joint-engine planted defects caught in the new engine;
- four target and five policy planted defects caught;
- cancellation-boundary portfolio valuation, post-terminal-fill refusal,
  missing-mark/refused-open-order cases and independent position accounting;
- 50 ms scoring and rejection of old artifact contracts.

Both complete synthetic selftest entry points and shell syntax checks passed.
Verification installed a guard against shared historical-data reads; there
were zero attempts. Source identities, test output and limits are saved in
`reviews/order_harm_client50_fix_20261114/verification.json` and `validation.log`.

The earlier review receipts are unchanged. Their two concrete defects are
superseded by this implementation and these targeted controls, not by a claim
that historical performance has improved.

## Remaining evaluation

Full label construction, fitting and historical policy evaluation remain
stopped under the user's earlier no-run instruction. The repaired drivers
are ready for that later step. No 50 ms inference-runtime guarantee is made.
Upstream guard and FRONT-PULL predictions still use leave-one-day-out fits;
the full pipeline is not yet a clean forward validation. The fill model,
per-rung training population, bounded tail and three fitted-Q iterations
retain their disclosed research limitations.
