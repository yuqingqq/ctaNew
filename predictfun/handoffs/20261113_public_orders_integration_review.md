# Public order study: implications for prediction and quoting

Read-only source/result review as of **2026-11-13T06:06:49.313461+00:00**.
The separate study's saved report is as of **2026-11-13T06:04:01Z**, on nine
consumed dates: Oct23/24/25/26/27/31 and Nov1/2/3. Its order cohort contains
911,353 observed order records with first observed fill in the first minute:
173,778 in the daily top-five-volume group, 283,317 in ranks 6–20, and 454,258
in the remainder. Fills through second 61 supply the recorded spans. Orders
that never fill are invisible. These are not neutral decision-time actions.

The report completed with wrapper rc=0 at 06:04 UTC; the driver records
`ALL DONE analyze`. This review reads the saved results and their source; it
does not independently rebuild identities, rerun the analysis or establish
new statistical significance.

Publication check at **2026-11-13T06:25:39Z**: the owner's
[published summary](20261113_orders_results.md) is now available, and its
review copy of `orders.json` matches the data copy exactly. The prose SHA is
`158b24cc764257d4a4d303b073888bf343e0058c4220246d827120e5672a3c14`.
The interpretations below apply to that publication as well as the raw report.

## What the saved result supports

For the population and as-of above, the report records:

| Measure | Top five (173,778 orders) | Field (737,575 orders) | Saved reading |
|---|---:|---:|---|
| Orders with at least three recorded fill IDs | 25.77% | 10.05% | `R400=PERSIST-MORE`; higher on all nine dates |
| Orders with at least two sole-maker-stop observations | 15.69% | 4.88% | `R401=FRONT-KEEP-MORE`; higher on all nine dates |

These results support examining the value of retaining a useful resting order.
They do not identify a causal gain from longer holding: order size, quote
placement, participation, market selection and filled-order selection differ
between groups. The groups rank daily maker shares over seconds 0–240, not
profitability or an identity known before that day's trading. The source does
not size-match these comparisons. Repeated fill IDs use certain taker-order
keys where available and an anchor-time fallback otherwise; they are not all
independently identified distinct taker orders.

The study also describes fills at multiple prices and overlapping same-hash
fill spans. Distinct filled prices over a minute do not count simultaneous
quote layers. Our frozen recovery experiment uses the existing placement
policy; it does not evaluate a new multilayer ladder. Even a successful
cancellation model would not establish top-maker parity.

## Limits that matter for this goal

- **Order life and re-entry:** `life = last_fill - first_fill`, conditional on
  repeated fills; it is not acceptance-to-cancel lifetime. The re-entry field
  is the gap between different hashes' last and first fills at the same price.
  It is not time out of the book or cancel-to-repost latency. These quantities
  must not set our re-entry timer or calibrate quote uptime.
- **Front retention and disappearance:** a sole-maker fill at the stopping
  price followed by that level still appearing is an inference about the
  queue. The subsequent book does not identify that same order's remaining
  size. `GONE` in the post-hoc EXIT-v2 analysis means the maker is absent from
  a later level-taking trade after an earlier inferred-presence observation;
  it is not an observed cancellation message or its effective timestamp.
- **Creator comparison:** despite the module description's broad statement
  about complete trades, `day_measures` builds creator candidates from all
  anchored `x60` legs. Its at-best denominator uses complete, finite-WS-time
  taker orders. Creator `hash_ok` is recorded but not a report filter. Their
  coverage differs, so the creator-share comparison does not cleanly identify
  who created a level or which maker has a placement-speed advantage.
- **Inventory:** the reconstructed position starts at zero within each market
  window and follows observable fills. The take classifier does not observe
  starting inventory, other accounts or external hedges. The report's
  `DIRECTIONAL` label therefore does not establish the maker's actual motive.
- **Prediction association:** EXIT-v2 uses the existing front-pull score,
  whose fits use the other eight dates and an inherited guard input with
  target-date exposure. The prior [front-pull review](20261113_front_pull_integration_review.md)
  documents this. Its post-hoc score/disappearance association does not prove
  earlier-only predictive skill, the maker's private signal, or a profitable
  cancellation policy. No score or threshold is imported into our experiment.

There is also a concrete coverage-reporting issue. `orders.json` prints
`stat.legs_unmapped = 0` by summing `ST[d].get(key, 0)`, but that field is absent
from **all nine** inspected daily receipts. This zero is not an identity
coverage check. For example, the Oct23 daily receipt reports 214,561 legs,
207,095 with certain hashes, 189,959 with certain taker orders and 163,880 in
complete trades. These populations differ. The original artifacts are
preserved; this review neither silently corrects them nor claims the other
owner has issued a superseding report.

## Consequence for the active experiment

The evidence is consistent with selective retention being worth testing, but
does not show that prediction alone explains the maker gap. Continue the
already frozen action-value experiment: price avoidable harm, forgone fills,
the continuation/re-entry path and inventory consequences, then measure full
policy value, presence and churn. Do not turn observed fill spans into quote
lifetimes or learn a cancel timer from them. No new model, policy, threshold,
success criterion or live change is introduced by this review.

Two claims in the published summary require narrower wording. The statement
that the pull signal is already right is not established by an association
between a score and inferred disappearance on these consumed dates. The
existing front-pull strategy result remains negative, and its score provenance
and policy limitations remain relevant. The reported gap from earlier
inferred presence to a later trade does not observe when a cancel was decided.

Likewise, a five-share order is not mechanically exhausted by every fill.
Our frozen engine credits the minimum of available trade volume and remaining
order quantity, subtracts that fill and marks completion only when the
remainder is exhausted. The current default-quote hook can retain an existing
order with positive remaining quantity. This source check establishes support
for partial fills; it does not measure how frequently they occur. Larger size
can sustain more fills, but the public comparison does not isolate its benefit
at equal inventory risk. The proposed larger-size replay would be a separate
experiment; no size or risk limit is changed here.

## Inspected artifact identities

The [saved machine report](/home/yuqing/ctaNew/data/predict_fun/research/orders_20261113/orders.json)
SHA-256 is `7aee33e2649bd46f63ecae57a033e62264c5b86d607748ac572a70b3982cdbd6`;
the [text report](/home/yuqing/ctaNew/data/predict_fun/research/orders_20261113/orders.txt)
SHA is `c6753b08805899abc3317c5c449bd3436cf4693931f7eca3ac092d2d8e07f8e5`.
The main-worktree source bytes match commit
`9f8ec33848123594cb153070f270e2e988789c67` for `pf_orders.py`.

| Inspected main-worktree source | SHA-256 |
|---|---|
| `predictfun/research/pf_orders.py` (`day_measures`, `report`, `exit_obs`) | `f822a5af3ecdfb2849bee2785776eee3508c1279f8eb01827292e72a8b3dab65` |
| `predictfun/research/pf_exit_check.py` (`classify`, `matched_diff`) | `dd5f426e42366af7b8588602c06a0c9aadec0d91f1d5af4e8f173b7f35c06f3b` |
| `predictfun/research/pf_gap_top_makers.py` (`rank_groups`) | `33feed2507215e0c8cd7b66d768337e97ea4a2a6504be67e7bbc7866f67a82e5` |
| `predictfun/research/pf_front_pull.py` (score provenance) | `42d832d8922b6eb5c7af4363d075b12c39851b29eb0b9f732db6d720ff0d76d6` |

The publication clarification also inspected these unchanged files in the
model worktree:

| Source | SHA-256 |
|---|---|
| `pf_step4_ladder_v3.py` (fill allocation and completion, lines 1172–1188) | `d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17` |
| `pf_order_policy_engine.py` (generated engine retains this fill allocation) | `09f37691da8f903083a63699680b61c1b3aa495ecddd8fb2d0e1e23a9ba0d999` |
| `pf_default_pull_hooks.py` (`remaining`, `Hook.targets`) | `d22348d05a491cc76cfb086a825262dd20dbf154268e9e739ea1448076787f4b` |

Daily receipts were read at
`data/predict_fun/research/orders_20261113/day_<date>.json`; the Oct23 extraction
receipt was also inspected. This is a bounded source review of reported
evidence, not a new validation claim.
