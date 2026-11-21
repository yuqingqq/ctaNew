# ENTRY-TARGET-AUDIT-v1 — the freeze: the accounting of the existing POST / WAIT labels, before any output

Review **d6d5d55e254c507db9799ef2fc1c6129d4657d06** accepted R4E (`141e820e`) as a reliable negative result and named one next priority:
decompose the existing labels and fill traces before choosing a different entry target or decision unit. This batch has **no fit, no new
label, no tape access and no policy replay**. Tuning of the failed direct POST − WAIT regressors (R3 RICH, R4E SMOOTH) stops here.

| File | Role |
|---|---|
| [pf_entry_audit.py](../../research/pf_entry_audit.py) | the population, the accounting (class, decomposition, evidence), predictions, summaries and the declared predicates |
| `pf_entry_audit_selftest.py` / `pf_entry_audit_mutants.py` | the review's synthetic cases, the refusals, the predicates, the summaries and the key join / 15 planted errors |
| [audit_checks.py](audit_checks.py) | the declared checks A–G and their falsifier |

## Definitions (fixed here; the module header is the reference)

- **Population:**
  - every POST row of R2-F's Oct 24–25 dataset (`2c62fc9b`…), in-sample for the frozen models;
  - every POST row of R3's Oct 26–27 dataset (`902526da`…), adaptive development;
  - every status kept, with picks reported separately and as the weighted total.

  R4E's 760 reached SMOOTH entry pairs are a separate, balanced diagnostic.
- **Class**, exhaustive, in this order:
  1. `no_post_fork_fills`;
  2. `economic_identical`: equal (side, q, price) multisets after the fork;
  3. `same_cash_inventory`;
  4. `offset_by_mark`: |Y| ≤ 1e-6 c;
  5. `meaningful`.

  The timed comparison, exact zeros and arithmetic residues are reported beside it.
- **Decomposition:**
  - V_initial is the forced generation's fills (located by the fork tick, the focal side and its POST decision's gid) valued at the
    common END mark.
  - D_following splits into D_focal and D_opposite.
  - **Y = V_initial + D_following** must hold on every row within 1e-6 c.
  - Gids are arm-local. V_initial is not a deployable entry advantage.
- **Evidence:** the initial generation's fill status and the WAIT arm's next actual POST. No refusal or queue cause is asserted.
- **Predictions:** joined by exact key for Oct 26–27 (R4E's saved file); in-sample for Oct 24–25. Cross-tabs only: no threshold, fit
  or search.

## Declared predicates (per set, weighted over its ok rows)

- `rare_execution`: the meaningful share is below **0.10** and the initial generation fills nothing in at least **0.50** of rows.
- `cancellation`: Σ w |V_initial| > 0 and Σ w |Y| ≤ **0.5** × Σ w |V_initial|.
- `diagnosis`: `both`, `rare_execution`, `cancellation` or `not_identified`.

The receipt's recommendation is chosen from these predicates:
- **cancellation:** a larger decision unit is worth declaring next, with its alternative action and opportunity cost specified first.
- **rare execution only:** relabeling the initial-generation component alone will not help.
- **neither:** not identified.

## Review d6d5d55e's other items

- Item 2 (names): the receipt renames the R4E first-review table per leg and per policy. Entry uses the candidate's paired advantage A
  (SMOOTH, ENTRY_ONLY) or PI0's V_post, and retention uses PI0's V_retain. The R4E predicates' `realized` becomes the
  *mandatory-anchor control value*, distinct from the *all-sampled selected-POST weighted mean*.
- Item 3 (check F's contract): the replay checker is not reused in this batch, so the review's verified workaround stands. A future
  replay checker adds episode END, settings, fill-time bounds and common marks, with single-field falsifiers.

## Order and budget

1. This freeze, with the selftest and mutation suite.
2. The audit on one core.
3. The checks, the falsifier and the receipt.

The planning ceiling is **one core-hour**. A breach stops with an explicit incomplete status.

## Freeze amendment (before any output)

The first run was stopped by hand before it wrote any file. Writing the formatter showed that `cmd_run` wrote the per-row reached data to
`reached.csv` and then overwrote it with the per-day summary of the same name.

The amendment:
- every output now has its own declared name (`FILES`, checked distinct: `reached_rows.csv` and `reached_summary.csv`), with `audit.json`
  written last;
- the selftest writes every output to a temporary directory;
- one more mutant (the collision restored) is caught: 16 / 16;
- check D and the formatter read the new names.

No definition, predicate or threshold changed.
