# Size model experiment frozen and queued, 2026-11-13 11:33 UTC

The goal remains active. The complete historical size pipeline is implemented,
committed, control-checked and frozen. Its driver is live, waiting for the
existing recovery experiment and the Codex/core8 slot. No historical size stage
has started, no larger-order model has been fitted on history, and no size or
model performance gain is established. No live trading changes.

## Authoritative identities and processes

- Source freeze: `8a8f4c18e352dc3ae9a88487fe613550e10730be`.
- Declaration as-of: `2026-11-13T11:30:43.831598+00:00`.
- Declaration SHA: `cea782c5c074158c44b52f302e79907f23808123535e6cbf01a77f5c40b5183c`.
- Root: `/home/yuqing/ctaNew/data/predict_fun/research/model_size_priority_20261113`.
- Mirrors: `predictfun/reviews/model_size_priority_20261113` in both trees.
-385 source identities,857 input identities,9 settlement registries,169 ordered
  stage tasks. All declaration/control/started mirrors agree.
- Controls receipt SHA: `71aa1798f2cd401636925cd9c0fddce9dc9dcec9b7fa0ba769d97bb61c48c503`.
- New pure driver1167074, started11:31:29.003843Z; confirmed live and sleeping
  at this checkpoint. Log: `logs/driver.log`; started receipt SHA
  `ba30efa265b40f8387c157d445791d42e7f87cc8b3c7a91204bb326fd488fcb5`.
- Existing recovery driver3159222 and Oct26/stress worker911474 remain live;
  recovery6/18 scenario-days complete, all six primary dates done.

The size driver waits for the entire preceding recovery driver, its final
verified evidence receipt, and a free Codex slot. It does not steal gaps between
the old driver's workers. Every engine job runs in a fresh process, protecting
against imported clock caches. Partial logs/output refuse automatic reruns.
All full historical numerical work uses core8. The declaration's bounded
hash/preparation run used core9,189MiB/15.23s; no outcome calculation ran there.
Its log SHA is `a5d241f3d4b5d0df9da194cd5db2a78bd784698280e0c4c1c74edbaa6145d1be`.

## Frozen scientific comparison

Five and50-share profiles share a reserved inventory cap50. Actual partial
initial sizes may be smaller when native capacity limits placement. Existing
late-cancel/notice reservations, price/data/minimum-notional constraints and
the60s reduce-only transition remain. One layer only; old five/cap-ten model
weights are not extrapolated to the larger profile.

At submission each neutral generation receives one of five equally weighted
rules: first accepted-age observation at0,100,500 or2000ms, or first notified
five-share partial milestone. Independent SHA bits select one quarter for
capture. Every generation and exclusion remains in the ledger, including orders
exhausted before their assigned observation. Untouched age-assigned orders stay
in scope; partial fills alone do not define training eligibility.

Captured alternatives are native KEEP, cancel/repost the notified residual,
and PULL until250ms of continuously clear causal pressure observations, with
maximum observation gap50ms. The10s target is KEEP-minus-PULL strict average
1s/5s fill marks minus0.01c endpoint inventory-square change. The short target
uses original-order fills during250ms after actual cancellation effectiveness.
Unavailable marks remain statuses. KEEP-minus-RESET and later-taker value are
diagnostics, not isolated causal queue premiums.

Three fits share the eligible pooled five/50 population and pinned300-round
parameters: STATIC+size predicts total value; HISTORY+size predicts total and
short value. Six expanding outer dates use earlier core days and the latest
preceding day for calibration; no refit on calibration or test. Original guard
fits and phase thresholds are authenticated earlier-only artifacts.

Initial quoting is default. Accepted known orders may be scored from0ms age,
at most once per100ms. HISTORY pulls on negative total value. The primary
SELECTIVE policy pulls only when both total and short values are negative.
Re-entry uses the identical pressure-clear protocol as captured PULL. There is
no forced resting lock or new threshold search.

Local comparisons include STATIC/HISTORY/SELECTIVE separately for each size.
Null controls preserve cancellation counts within day, side, hour, age bucket,
inventory sign, assigned action type and notified-partial state:500 draws.
Intervals use1000 UTC-day resamples. Local utilities overlap and are not portfolio
P&L; differing neutral size populations do not identify a pure size effect.

Full replay compares five-share DEFAULT/SELECTIVE and50-share DEFAULT/causal
guard/HISTORY/SELECTIVE under X, execution/queue stress and age500, across six
test days x288 windows. Four-window smokes include age0 exact native/evidence
parity. X defaults are reused as independent neutral construction references.
No local result chooses the sequential primary or skips a declared scenario.

Primary success requires50-share SELECTIVE to beat both50-share chronological
guard and HISTORY on common strict marked value, with nonincreasing all-window
inventory-square exposure and a positive lower95% day interval in every
scenario. Entry any-quote presence must not fall and learned pull decisions
must fall versus HISTORY. Size advantage is computed separately for50-versus5
under identical DEFAULT and SELECTIVE policies, also requiring realized risk
not to increase. Identical configured caps alone never establish matched risk.

## Implemented pipeline and evidence checks

`pf_size_priority_experiment_spec/stages` bind corrected grid/context receipts,
inputs, native arm identities, independently replayed defaults, capture/feature
joins, chronological fits, saved models and all stage artifacts. Model replays
retain original taker identity and the actual global fill order.

`..._local` implements matched comparisons and descriptive forecast calibration.
`..._report` reconstructs portfolio/settlement/presence, checks native fill-order
inventory, and reports repeat-taker value with unknown/missing statuses.
`..._verify` redoes both reports from their named artifacts, verifies source,
input and complete stage coverage, and refuses altered tables or verdicts.
`..._declare` requires the full committed pipeline and passing source-bound
controls. `model_size_priority_experiment_driver.py` executes the frozen order.

The initial five-test stage command found the auxiliary slot owned by another
research chain and refused before launching; no test attempt or stage receipt
was created. That chain completed before the final suite started.

Final12 controls PASS as-of `2026-11-13T11:28:40.361760+00:00`, source8a8f4c18:
six stage controls, four evaluator/freeze/schedule controls, two native runtime
controls. They cover real native capture-to-feature/target joins, corrected-grid
receipt and rehashed invalid-grid refusals, saved pooled chronological fits and
prediction reconstruction with exclusions, known-bad joins, full task coverage,
matched-count/day comparisons, separate size/model risk predicates, actual
success-log checks, native five/50 retention, independent neutral parity and
age0/500 behavior. These are synthetic implementation results.

Control receipt SHA:
`eac19559a35e20122dd043f712f37501a09ee3b24c480870a56c99fb67942948`.
All79 bound control source identities, committed contents, log and three mirrors
were verified. Wrapper256MiB/20.30s on free auxiliary core9. The original302
recovery source files remain unchanged. No control/tool session remains pending.

## Next

Monitor the existing recovery worker/driver and the queued size driver by their
actual process handles. Preserve any terminal failure before correction; never
restart because a log is quiet. When size jobs begin, review the first default
and captured smokes, especially full native schema/clock linkage, then let the
unchanged declared sequence proceed. Do not rebuild the pipeline or alter frozen
constants. Review both negative and positive results with the declared gates.
Only the nine consumed dates are authorized; no untouched-day or live claim is
available. Model improvement and goal completion remain unproven.
