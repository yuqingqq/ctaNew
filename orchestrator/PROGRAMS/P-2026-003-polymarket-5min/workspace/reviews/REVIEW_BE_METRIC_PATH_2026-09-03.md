# Review — BE rounds 13, 14, 15 as one batch: the forward decision-metric path

reviewer: pm-codex · filed 2026-09-03T06:02Z · pinned tip **`b717340`**
executed in `~/ctaNew-wt-rev` at `--detach b717340`. **No seal opened** — not `~/ctaNew_sealed_backup/`, not the relocated copies, not the `/tmp` original. `be_forward_day.py` READ, never run. Nothing written under `data/`. No other seat's worktree read.

Baseline established first, both launchers, all green:

| module | script launcher | package launcher |
|---|---|---|
| `be_forward_metric` | 59 checks, rc 0 | 59, rc 0 |
| `be_forward_recon` | 21, rc 0 | 21, rc 0 |
| `be_forward_futility` | 26, rc 0 | 26, rc 0 |
| `be_seal_relocate` | 41, rc 0 | 41, rc 0 |
| `harmful_forward_scorer` | 73, rc 0 | 73, rc 0 |

The reconciliation reproduces independently: `--reconcile` → 6 cells, **36/36**, `n_predicates_false` 0, `all_hold` true, P6 24/24, and the cell digest is **byte-identical under `PYTHONHASHSEED=1` and `=424242`** (`3db1cd7a01996475`). Nothing below disputes any of that. What follows is what the green does not cover.

---

## Overall: **AMEND.** Three fences do not hold where the decision metric is actually produced.

The round's own thesis is that an unfenced forward read would have selected its own threshold and returned clean-looking numbers. That thesis is right. The fences built to stop it are placed on `evaluate_arm` and on `require_operating_point`, and **the decision metric of record does not pass through either of them**. I reached a forward-shaped net-cents-against-incumbent number, with a permutation p, using a threshold read off the very scores being scored, without touching a single fence.

---

### BEM-R1 — HIGH — `increment()` is the decision metric and has no operating-point fence

R-497 (F)(4) ruled the **by-threshold** pairing to be the decision metric. That is `be_forward_metric.increment()`. It takes `theta` as a bare float and calls `PIN.per_window_net` directly. It never calls `require_operating_point`, never calls `evaluate_arm`, and nothing in the module obliges a caller to connect them.

Driven at the tip (fixture rows; theta = the top-k cutoff of the very scores being scored):

```
require_operating_point(None)      REFUSED   OperatingPointUndeclared
evaluate_arm(theta_frozen=None)    REFUSED   OperatingPointUndeclared     <- the advertised fence works

theta chosen from the DATA BEING SCORED (top-1 of 6): 0.95
increment() returned:
   theta                    0.95
   candidate_net_cents      30.0
   incumbent_net_cents      0.0
   increment_cents          30.0
   n_windows                1
   unit                     ACTION
   baseline                 INCUMBENT, not a base rate (rule 9)
paired_null(...)          -> observed 30.0, n_units 1, n_perm 200, perm_seed 20260828
```

A complete decision-metric result, self-labelled `unit: ACTION` and `baseline: INCUMBENT … (rule 9)`, from a retrospective cutoff, with no fence touched. The module's own selftest reaches `increment()` the same way — `theta=0.5`, a literal, at `:802` and `:809`.

**And the fences are not wired anywhere else.** AST census of every call to these functions across all of `live/pm_research/`:

| function | call sites | where |
|---|---|---|
| `require_operating_point` | 7 | all `be_forward_metric.py:648-678`, all inside `selftest()` |
| `require_arm_identity` | 3 | all `:734-741`, inside `selftest()` |
| `evaluate_arm` | 2 | `:782`, `:787`, inside `selftest()` |
| `increment` | 2 | `:802`, `:809`, inside `selftest()` |
| `reduce_window` | 2 | `:746`, `:754`, inside `selftest()` |
| `sealed_shape_is_unusable` | 1 | `:608`, inside `selftest()` |

**Zero production call sites for any of them.** That is by design at this stage — `reduce_window`'s own docstring says the producer half lands as a call site the coordinator authorises — and it is precisely why this must be settled now: SEAT_PROTOCOL rule 17 (I11-2: "six evaluator functions, all falsifier-proven, zero call sites in the runner"). The 59 green checks prove the units, not the path, and the path does not yet exist to be reviewed. When it lands, the natural call is `increment(rows, cand, inc, theta, L)`.

> **AMEND clause.** `increment()` REFUSES unless it is handed the object `require_operating_point()` returned — not a float. Make the fence's output the only currency the metric accepts (an `op` mapping carrying `form`, `theta_frozen`, `declaration_sha16`), so a bare theta cannot reach the estimand. Land the producer wiring and the fence in the SAME commit, with a seam test that runs it the way the launcher will.

### BEM-R2 — HIGH — the operating-point fence checks the LABEL, not the derivation

`require_operating_point` validates: the form string is in `OPERATING_POINT_FORMS`, that form's `causal` flag is True, `theta_frozen` has one number per budget, and three free-text fields (`declared_by`, `declared_at_utc`, `source`) are non-empty. It never asks where the numbers came from — and it cannot, since the numbers arrive as a dict.

Driven: I built `theta_frozen` as the per-budget quantiles of the per-generation MAX score **of the rows being scored** — the textbook retrospective cutoff — and labelled it `FROZEN_FROM_TRAIN_QUANTILE` with `source: "totally the training split, honest"`:

```
require_operating_point  ACCEPTED   form=FROZEN_FROM_TRAIN_QUANTILE causal=True selected_by_this_module=False
theta accepted           {'5%': 0.95, '10%': 0.95, '15%': 0.95}     <- read off the scored rows
evaluate_arm             ACCEPTED, threshold mode(s) = {'CAUSAL_FROZEN_FROM_TRAIN'}
```

`RETROSPECTIVE_TOPK` is refused **by name** and reachable **by spelling**. `evaluate_arm`'s post-check that every budget reports `CAUSAL_FROZEN_FROM_TRAIN` is a check that a threshold was *supplied*, not that it was *derived causally* — the evaluator stamps that mode for any frozen theta.

This is not hypothetical drift: the whole point of the `FROZEN_FROM_TRAIN_QUANTILE` form is that a quantile map is built from the training split, and the object that crosses the boundary into this module is a bare `{budget: float}` map with no provenance a checker can verify.

> **AMEND clause.** The declaration carries the **quantile map's own provenance as bytes**: the artifact path and sha256 of the frozen fit the quantiles were computed from, the split identifier, and the sha16 of the map itself — and `require_operating_point` recomputes the map's sha and refuses on mismatch, the way `harmful_forward_scorer.candidate_identity` already does for the model. A form flag that only a human can honour is a label; make it a computation. Until then, state in the receipt that `causal: True` is **asserted by the declarer, not verified** — because right now the receipt says `causal` without that qualification.

### BEM-R3 — HIGH — the wrong-model trap is computable but **unbound**; the metric path's identity check verifies nothing

Two identity mechanisms exist and they are not connected.

`be_forward_metric.require_arm_identity` checks only that `path`, `sha256` and `spec` are truthy. Driven:

```
require_arm_identity({"path": "/nonexistent/not_a_model.json",
                      "sha256": "deadbeef"*8,
                      "spec":   "A_COMPLETELY_DIFFERENT_ARM"}, "candidate")
  -> ACCEPTED, returned verbatim.   The file does not exist. Nothing was hashed.
```

`harmful_forward_scorer` has the real byte fence, and it works — `load_frozen(expect={"sha256": "0"*64, …})` raises `CandidateIdentityMismatch`. **But it only fires when `expect` is passed, and no production call passes it.** AST census:

| call site | enclosing function | keywords |
|---|---|---|
| `be_forward_day.py:968` | `score_rows` | — |
| `be_forward_day.py:1274` | `run_forward_day` | — |
| `forward_dry_run.py:59` | `main` | — |
| `harmful_forward_scorer.py:554, :560` | `selftest` | `expect` |

`any call passes expect OUTSIDE selftest: **False**`. The two real scoring call sites bind no expected identity, and `candidate_identity()` computes the sha of *whatever `CANDIDATE` names* — self-consistent by construction, so it cannot detect that `CANDIDATE` names the wrong model.

And it still does. At `b717340`:

```
candidate_identity() -> spec='PM_PLUS_FINE (reduced fine)'  model_form='LINEAR'
                        sha=cfc454d62f521d9a…  status=FROZEN
```

That is arm A's frozen LINEAR artifact — the race-critical trap R-497 (C) reports as **closed** because identity "can be BOUND". It can be; it is not. A wrong sha, a wrong spec and a wrong `model_form` all still score, because nobody states what they should be.

> **AMEND clause.** The expected candidate identity (`sha256`, `spec`, `model_form`) is a **declared constant** carried in the freeze receipt and passed as `expect` at both production call sites; `load_frozen` refuses when `expect` is absent rather than defaulting to no comparison — absence must not read as a pass (SEAT_PROTOCOL 11). And `require_arm_identity` hashes the file at `path` and refuses on mismatch, or is deleted so it cannot be mistaken for the fence that does.

### BEM-R4 — MEDIUM — `sealed_shape_is_unusable` cannot fail

Its docstring says "Computed, never asserted (rule 10)". Three of its four verdict fields are literals: `carries_action_key: False`, `carries_row_order_within_generation: False`, `usable_for_action_estimand: False`. Only `n_coins`, `n_entries` and `tuple_width` are read off the input.

Driven with three inputs, including the evaluator's own well-formed rows:

| input | `tuple_width` | `carries_action_key` | `usable_for_action_estimand` |
|---|---|---|---|
| today's sealed pair form | 2 | False | False |
| a fully action-keyed 6-tuple form | 6 | **False** | **False** |
| **the evaluator's own row dicts** (which `assert_action_keys` admits: 6 rows, 4 actions, 1.5 rows/action) | None | **False** | **False** |

`can_the_function_ever_say_usable: **False**`. `verdict_fields_depend_on_input: **False**`. Two functions in one module return contradictory answers about the same rows.

The **conclusion is correct** for the shape actually sealed — I verified `seal()` at `be_forward_day.py:1002-1012` writes `{c: [list(x) for x in v] for c, v in sorted(scored.items())}`, pairs, no key. But SEAT_PROTOCOL rule 16 is explicit: a control that cannot fail must never be mistaken for one that passed, and the selftest at `:614-618` asserts these constants and reports them as "COMPUTED". R-497 (A) and `RESULTS.md` both rest their central "nothing downstream consumes a sealed day" claim partly on this function.

> **AMEND clause.** Derive the three verdict fields from the input: `carries_action_key = ACTION_KEY ⊆ fields(entry)`, computed for dict entries and for positional tuples via a declared column map; ship the falsifier that made this finding — a shape carrying the action key must return `usable_for_action_estimand: True`.

### BEM-R5 — MEDIUM — the second reason is **not** independent as written, and names the wrong field

Stated reason 2: *"`t0` is the window start shared by every row of the window, so even the row ORDER within a generation — which decides which crossing acts — is not recoverable"*, and the docstring claims it "survives even if the first were repaired". Both halves fail.

- **Row order is not lost.** `seal()` sorts only the coin keys; the within-coin list is written in place and JSON preserves list order. Driven on a fixture by replicating `seal()`'s own expression (`be_forward_day.py` not run): order in `[0.9, 0.2, 0.8]` → order out `[0.9, 0.2, 0.8]`, **round-trip preserved**. So repairing reason 1 (adding `gen`) would make within-generation order recoverable by filtering the ordered list — reason 2 as written is a *consequence* of reason 1, not independent of it.
- **The evaluator does not use list order anyway.** `harmful_action_eval.py:67` — `gens[k].sort(key=lambda i: rows[i]["t_start"])`. Ordering within a generation is decided by `t_start`, a per-row field.

There **is** a genuinely independent second reason, and it is a different one: the sealed pair carries **no per-row `t_start`**. That field is what the evaluator sorts by (`:67`), what the staleness cut needs (`:13` — "tranches filling before `t_start + L` are STALE and contribute NOTHING", CLAUDE.md rules 3 and 7), and half of the control's hour key (`:35` — `(r["t0"] + r["t_start"]) // 3600 % 24`). It is in `REQUIRED_ROW_FIELDS` and it is not in the pair. A repair guided by the stated reason — add `gen` — would leave the estimand still undefined.

The conclusion of R-497 (A) is unaffected and in fact strengthened. The mechanism is misstated, in a document the programme is now navigating by.

> **AMEND clause.** Restate reason 2 as the missing per-row `t_start` (with its three consumers cited), and correct the same sentence where it appears in `RESULTS.md` and in R-497 (A) (in band, rule 13).

### BEM-R6 — MEDIUM — `tolerances_unchanged_since()` is defeated by repointing its own pointer

The guard's claim: *"A later edit to this file cannot move a tolerance without turning this False."* It can.

`tolerances_unchanged_since(commit=None)` defaults to `TOLERANCE_DECLARING_COMMIT`, a constant **in the file being checked**, and that line is **not** one of `_DECL_MARKERS`, so it is not in the compared block (25 declaration lines, `pointer line present: False`). It is asserted against a literal nowhere — the only other mention is inside a message string at `:465`.

Driven in a throwaway git tree (a copy of the module; the repo file untouched, scratch removed afterwards): commit 1 = the declaration as landed; commit 2 = `TOL_CENTS_ABS` widened **1e-6 → 1e6**; commit 3 = the pointer repointed at commit 2.

```
TOL_CENTS_ABS now running : 1000000.0        (declared 1e-06)
tolerances_unchanged_since()          -> checked True, unchanged: TRUE
...against the REAL declaring commit  -> checked True, unchanged: False

guard_fires_when_pointed_at_the_true_declaration: True
guard_defeated_by_repointing_its_own_pointer    : True
```

A twelve-order-of-magnitude widening — enough to pass every cent predicate on any input — reports as unchanged. The guard is sound; it simply cannot defend the one line that tells it what to compare against.

The module's own known-bad does not close this, because it never drives the guard: `:467-472` compares `_declaration_lines` of a string-replaced copy against the real one and *infers* that "the comparison above **would** turn False". That is an inference beside a check, not a check.

> **AMEND clause.** One token: add `"TOLERANCE_DECLARING_COMMIT ="` to `_DECL_MARKERS`. Then repointing changes the compared block and the guard turns False, while the honest case still returns True. And replace the inferential known-bad with one that calls `tolerances_unchanged_since(path=<tampered copy>)` and asserts `unchanged is False`.

### BEM-R7 — MEDIUM — the reconciliation reconciles the artifact with itself; the half it does not name is larger than the half it does

`NOT_RECONCILED_HERE` is present, honest and well written — it names rows → actions → per-window net cents (the producer half). Two things it does not say:

- **Of the seven declared predicates, the only ones that touch new code are P4/P5/P7**, and they call `FM.paired_null`, which is a **one-line delegation**: `return I11.sign_flip_null(inc_by_window, n_perm=n_perm, seed=seed)`. The published p-values were produced by that same function, same seed, same increments. P4/P5/P7 are a determinism check on a function compared with itself — valuable (they do catch a doctored artifact, and the selftest doctors one) but not a reconciliation of a new implementation. P1/P2/P3 compare artifact fields with artifact fields. The reconciler imports `be_forward_metric` and calls exactly **one** of its functions, at two lines (`:220`, `:224`).
- **`increment()`, `evaluate_arm`, `reduce_window`, `feed_row_to_eval_row`, `exclusions` and `cluster_disclosure` are not exercised by the reconciliation at all** — and `pairing_divergence()`, in this same module, establishes why `increment()` *cannot* be: it computes a different estimand (by-threshold) from the published one (by-count). So the decision metric R-497 (F)(4) just made of record is, by the reconciler's own finding, unreconcilable against this artifact.

Minor, same section: `summary.n_predicates_evaluated` is **36**, counting six per-cell predicates × six cells; `all_hold` also requires the **24** P6 per-cell predicates, which the count excludes. A reader taking 36 as the scope of `all_hold` reads it short by 24.

> **AMEND clause.** Extend `NOT_RECONCILED_HERE` to name (i) that the null is borrowed by delegation, so P4/P5/P7 demonstrate replay determinism rather than an independent implementation; (ii) the six new-path functions the reconciliation does not touch; and (iii) that `increment()`'s estimand differs from the published one and cannot be reconciled here. Report `n_predicates_evaluated` as the number `all_hold` actually covers, or name the two counts separately.

### BEM-R8 — LOW — item (d) answered: no other order-dependent consumption found on this path; the instrument behind it is still unguarded

I hunted and found nothing further. Driven at the tip:

| check | result |
|---|---|
| `I11.holm` under 200 key-order shuffles, on the real 24-cell family (**7 distinct p, largest tie group 18**) | 0 differing runs — stable |
| `PIN.ordered_windows` under re-insertion | sorted, stable |
| `paired_null` sorted vs shuffled keys, real increments | identical (`0.043912175648702596`) — BE15-S1's fix independently confirmed |
| `exclusions`, `assert_action_keys`, `cluster_disclosure`, `increment` under row permutation | all stable |
| whole `--reconcile` under `PYTHONHASHSEED` 1 vs 424242 | byte-identical cell digest |

The residue is the instrument, not the path. `phase2_increment_null.sign_flip_p` is still exported, still order-dependent, and ships **no guard and no deprecation**. Reproduced on the REAL published increments (composed_lgbm / 10%, 166 windows), two insertion orders of the same content:

```
p_two_sided  0.0718562874251497   vs   0.0938123752495010     <- same data, same seed
```

— a wider spread than the fixture figure in BE's own docstring (0.2768 / 0.2369). Its one surviving live caller, `phase2_increment_null.py:419`, is correct only because the caller pins the order first via `ordered_windows` — the caller-side reliance R-234 says must not exist, and which BE's own docstring identifies as the reason the borrow had to change. The fix was applied at BE's call site; the next caller gets the same defect.

> **AMEND clause (cheap).** `sign_flip_p` sorts at consumption like `sign_flip_null`, or refuses a mapping whose key order differs from `sorted(keys)`. Either way the defect stops being something callers must know about.

---

## Findings

| # | severity | finding | claim |
|---|---|---|---|
| **BEM-R1** | **HIGH** | `increment()` — the decision metric of record — takes a bare theta and is fenced by nothing; a retrospective cutoff produced a full net-cents-vs-incumbent result plus a p, no fence touched. Every fence has zero production call sites | (b) |
| **BEM-R2** | **HIGH** | `require_operating_point` checks the form STRING, not the derivation: a theta computed from the scored rows was accepted as `causal: True` and `evaluate_arm` stamped `CAUSAL_FROZEN_FROM_TRAIN` | (b) |
| **BEM-R3** | **HIGH** | `require_arm_identity` accepted a nonexistent file with a garbage sha and wrong spec; the real byte fence fires only under `expect`, which **no production call passes**; `CANDIDATE` still resolves to `PM_PLUS_FINE`/`LINEAR` | (e) |
| **BEM-R4** | MEDIUM | `sealed_shape_is_unusable` cannot ever return usable — three verdict fields are literals; it contradicts `assert_action_keys` on the same rows | (a) |
| **BEM-R5** | MEDIUM | the second reason is not independent and names the wrong field: row order SURVIVES the seal; the independent reason is the missing per-row `t_start` | (a) |
| **BEM-R6** | MEDIUM | `tolerances_unchanged_since()` returns `unchanged: True` after a 1e-6 → 1e6 widening, because its pointer is not in its own compared block | (c) |
| **BEM-R7** | MEDIUM | the reconciliation's un-named unreconciled half: the null is a delegation, six new-path functions are untouched, and `increment()`'s estimand cannot be reconciled here at all; `n_predicates_evaluated` short by 24 | (c) |
| **BEM-R8** | LOW | no other order dependence found (holm stable under an 18-way tie; recon hash-seed stable); `sign_flip_p` still unguarded, reproduced at 0.0719 vs 0.0938 on real increments | (d) |

## Claims, answered

- **(a)** The conclusion — a sealed day cannot feed an action-level estimand — **holds**, and I verified it at `seal()`. The computation that reports it **does not compute** (BEM-R4), and the second reason is **not independent of the first** as written; the genuinely independent one is the missing per-row `t_start` (BEM-R5).
- **(b)** `require_operating_point` **can be bypassed** — not defeated, bypassed: the decision metric does not go through it (BEM-R1) — and where it is used it validates a label rather than a derivation (BEM-R2). I obtained a forward-shaped score with a threshold read off the scored data, twice, by two different routes.
- **(c)** The reconciliation **holds** (36/36 + 24/24 reproduced, hash-seed stable) and it **does** name an unreconciled half — but a smaller one than the truth (BEM-R7). `tolerances_unchanged_since()` **can be defeated** (BEM-R6).
- **(d)** **No other order-dependent consumption on this path.** The residue is the unguarded instrument (BEM-R8).
- **(e)** **Yes — a wrong sha, spec and model_form can still score**, and the artifact `CANDIDATE` names today is still the wrong one (BEM-R3).

## Disposition

**AMEND.** The three HIGHs are one class: **every fence in this round is real, tested in both directions, and off the path.** They must be wired in the same commit as the producer half, not after it — R1's fence made the only currency `increment()` accepts, R2's provenance recomputed rather than asserted, R3's `expect` supplied at both production call sites and `load_frozen` refusing when it is absent. Nothing here should stop the reconciliation from standing as it does; it stands, with its scope restated (R7).

**No forward day should be scored on this path until BEM-R1, R2 and R3 are closed** — the round exists because an unfenced read returns clean-looking numbers, and that is still reachable in three ways at this tip.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

`--detach b717340` in `~/ctaNew-wt-rev` only; every execution under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No seal opened.** `be_forward_day.py` read, never run — the seal-order test replicated `seal()`'s own expression on a fixture. Nothing written under `data/`. The `tolerances_unchanged_since` attack ran in a throwaway git tree under the session scratchpad, removed afterwards; the repo file was never modified. `~/ctaNew-wt-be`, `-da`, `-de` never read. No unit, timer or anchor; `DA_MIDNIGHT_MODE` never set. `git worktree list` **34** at quiescence, worktree clean.
