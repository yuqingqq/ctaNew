# SOURCE-EXTEND-v1 — declaration (step 0 of the six-date coverage build)

2026-11-21 14:33Z, session aea5bfb9.

| Decision | Content |
|---|---|
| USER 13:41Z, "Approve go" | Authorizes review 216ff603's coverage proposal. |
| USER 14:2xZ, choice "Extend source, all 6" | The metadata preflight found that the R2 / R3 world source exists for only two of the six approved dates. The USER chose to extend it to all six. |

The specification is the header of `predictfun/research/pf_source_extend.py`.

## What runs

The frozen neutral source chain runs unchanged for the consumed replay days Oct 31 and Nov 1–3:
1. `pf_queue_place_test.day_run` (v1): the queue-place grid rows.
2. `pf_keep_pull.day_run`: `orders.parquet`, the grid's JOIN states with their statuses, and `fills.parquet`.

The wrapper swaps only the builders' declared day lists and v1's output root, and restores them after the call. No frozen file is edited.
Outputs go to `data/predict_fun/research/source_extend_20261121/<builder>/<day>`. Nothing is written into the 11-17 directories.

## Boundaries

`allowed()` refuses the following before anything runs:
- any day in the loader's sealed block (Oct 28–30);
- any day ending after the forward boundary (Nov 4);
- any undeclared day.

Padding audit of the chain:
- One loader module (`pf_stop_ev_minutes_inputs`) serves v1, the ledger, keep / pull and the label sampler.
- `load_day` reads books through `part_allowed` with `SKIP_FOREIGN_PARTS`.
- `fair_fix` and `depth_hours` read hours −1..23 through `part_allowed`; `real_inputs` additionally checks `hour_ok` and the market window.
- Everything else is a day-named file.
- The preflight confirmed the effect: Oct 31's hour −1 (sealed Oct 30 23Z) and Nov 3's next 00h (forward Nov 4) are refused by name before any file opens.

## Positive control (before any new day)

The wrapper is run on consumed Oct 24 into `source_extend_20261121/control`. It must reproduce the 11-17 outputs:
- v1 `rows.parquet`, and keep / pull `orders.parquet` and `fills.parquet`, equal as frames (`DataFrame.equals`: shape, column order, dtypes and every value);
- the receipts' stat, status, reconciliation and fair_missing, and v1's markets / stat;
- keep / pull's recorded frozen-source hashes.

The result is written once to `control.json`. `run_day` refuses a new day unless that control passed and the wrapper is the committed one.

Controls on the wrapper itself:
- Selftest passes.
- 18 / 18 mutants are caught (`logs/mutants_src`). A first 16 / 18 run is kept: its two escapes were equivalent, because `DataFrame.equals` already compares dtypes and column order. `frames_equal` now relies on it alone, and the replacement mutants change what is compared.

## Use of the outputs

The coverage build reads only these orders' world keys and statuses to enumerate the worlds; this is what `pf_r2_label_sample.worlds_all` does on Oct 24–27. The KEEP-branch values in the orders are not read by the coverage build.

Cost (about 100 s per builder per day) counts toward the 8 core-hours. The coverage builder itself, its sampling manifest and its reading map are frozen in a later commit, before the pilot and before any label.
