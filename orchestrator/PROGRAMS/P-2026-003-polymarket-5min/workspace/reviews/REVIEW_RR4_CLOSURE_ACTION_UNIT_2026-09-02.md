# Review — RR4-3 (the action-unit measurement), RR4-1, RR4-2
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `c180061`** (BE's RR4 batch, code `b1f36e2`; artifact
re-emitted as-of 2026-09-02T05:21:34Z, 188,119 B).
**Continues** `REVIEW_R397_BATCH_AND_REGISTRY_2026-09-02.md` (`911f186`).
**Composed 2026-09-02T05:40:44Z.** One filing, per R-377.

Method unchanged: detached worktree at the pinned tip, ledgers symlinked
read-only, verdicts from execution; mutants applied in the lab and reverted; no
repository file modified and nothing written to any production ledger or tape.

---

## Verdict

### **WINNER-HOLD RELEASED.** RR4-3, RR4-1 and RR4-2 are all closed.

The hold I placed was narrow and it has been met exactly: Q1's statistic is now
reported at the **action** unit beside the row-level one, for **both sides of the
comparison**, under **all three collapse rules**, on **both arms**, with the
invariance computed rather than asserted. I verified all eight (arm × unit/rule)
comparisons myself: **the candidate beats the incumbent hazard head in every one**,
and the increment is **larger** at every action-unit rule than at the row level.
Deduplication did not manufacture the beat; it widened it.

**Releasing the hold does not answer the winner question** — that is the USER's,
and two facts below belong in front of whoever answers it: the *level* of Q1's
discrimination moves with the collapse rule (0.790–0.876 lgbm, 0.735–0.814 linear),
and the codebase's own two conventions point at different rules.

Two findings, both coverage of the new measurement, neither holding: **RR5-1** (the
`first` rule's earliest-instant selection is untested) and **RR5-2** (the action
key's `side` component is untested). Both are one fixture each.

---

## RR4-3 — CLOSED

### The eight comparisons, recomputed from the artifact

| arm | unit / rule | candidate | incumbent | increment | beats |
|---|---|---|---|---|---|
| composed_lgbm | ROW | 0.8303 | 0.7139 | **+0.1164** | yes |
| composed_lgbm | ACTION / max | 0.8760 | 0.7418 | **+0.1342** | yes |
| composed_lgbm | ACTION / mean | 0.8641 | 0.7041 | **+0.1600** | yes |
| composed_lgbm | ACTION / first | 0.7903 | 0.6136 | **+0.1767** | yes |
| composed_linear | ROW | 0.7733 | 0.7139 | **+0.0594** | yes |
| composed_linear | ACTION / max | 0.8144 | 0.7418 | **+0.0725** | yes |
| composed_linear | ACTION / mean | 0.7978 | 0.7041 | **+0.0937** | yes |
| composed_linear | ACTION / first | 0.7353 | 0.6136 | **+0.1217** | yes |

**8 of 8 favour the candidate**, `agrees_with_row_level: true` on both arms, and
both sides of every comparison are collapsed by the **same** rule — which is the
property that makes the comparison meaningful at all, and it is honoured.

Arithmetic I checked directly: `rows_per_action` recomputes to **1.754**
(311,640 / 177,674) as stated; the disagreeing-label disclosure is
**6,108 / 177,674 = 3.438%**, matching the dispatch's 3.44%; and `statistic_n` now
reads **311,640** with `statistic_n_unit: "rows"` and a basis naming the action-unit
value beside it — my round-4 field defect is fixed at the source.

### What I did NOT verify, said plainly

The action-unit AUCs are recomputations over the model's own predictions, which the
artifact does not emit. I verified the **collapse logic by reading and mutating it**
and the **comparison arithmetic by recomputation**; I did not recompute an AUC from
predictions, because the predictions are not in the artifact and re-running the fit
was not warranted for this question. What is verified is that the two sides are
collapsed identically, that the increment is what the emitted numbers say, and that
the beat holds under every rule.

### Two facts for whoever answers the winner question

**1. "Deduplication raises" is rule-dependent, and the dispatch states it flatly.**
Against the row-level 0.8303: `max` 0.8760 **raises**, `mean` 0.8641 **raises**,
`first` 0.7903 **lowers**. The row-level number sits **inside** the action-unit span,
so the honest reading is not "dedup raises" but *the level of the statistic carries
a ±0.04 band from the unit convention alone, and the row-level number is in the
middle of it.* What is rule-independent is the comparison.

**2. The codebase's own conventions point at different rules, and the artifact's
primary follows one of them.** `collapse_primary: "max"` is justified in the
docstring as reusing `q4_economics`'s existing ranking convention. **I verified that
premise in code**: `q4_economics._rank` does rank generations by `max(scores)`. So
`max` is not a rule chosen because it is the highest — it is the codebase's ranking
convention, and AUC is a ranking metric.

But the same code block says the other half out loud: it ranks by max and **values
at the EARLIEST crossing row**, because *"a cancel fires the first time the score
crosses… Valuing at the generation's max row credits the policy with a decision it
never made."* The `first` rule is the analogue of that valuing convention, and it
is the rule under which Q1 scores **lowest** (0.7903 / 0.7353). A reader deciding
whether Q1 is a Phase-2 winner should see that the programme's own ranking
convention and its own valuing convention give 0.876 and 0.790 for the same head.

Neither observation weakens the release: the gate's conjunct is a comparison, and
the comparison is invariant.

---

## RR4-1 — CLOSED

`gate_conjuncts_evaluated` is now derived from the conjuncts. Computed over the
artifact: **0 cells** claim `true` beside a null conjunct (12 did last round).
Q1 and Q3 carry `true`; **Q2 and Q4 now carry `false`**, and Q4's six cells have
moved to `GATE_PARTIALLY_EVALUATED` — so the latent case I named is closed at the
source: Q4 can no longer be published as surviving on an uncomputed conjunct even
if its p improves. Statuses: 12 OK, 6 GATE_PARTIALLY_EVALUATED, 6
NO_INCUMBENT_COUNTERPART; survivors unchanged at 12.

**Mutant:** the action label rule flipped from `any` to `all` → **KILLED** by an
RR4-3 known-bad naming the two-unit divergence.

## RR4-2 — CLOSED, and it matches my independent recomputation exactly

The cell now carries both numbers, each computed:

| field | artifact | my recomputation |
|---|---|---|
| `holm_at_floor` | 0.04790419161676646 | 0.047904 |
| `holm_if_ONE_draw_beat_it_in_THIS_CELL_ONLY` | **0.04790419161676646** | **0.047904** |
| `holm_if_ONE_draw_beat_it_in_EVERY_at_floor_CELL` | **0.09580838323353293** | **0.095808** |
| `survives_if_THIS_CELL_ONLY_moved` | **true** | true |
| `survives_if_EVERY_at_floor_cell_moved` | **false** | false |

plus `holm_if_EVERY_at_floor_cell_moved_together_multiplied` kept beside the
computed one, and a flag saying the numbers are computed. My round-1 framing —
which was wrong for the single-cell case and which the artifact and two register
entries had inherited — is now corrected everywhere it appeared.

---

## Findings

### RR5-1 — LOW/MEDIUM — the `first` rule's "earliest decision instant" is untested

`first` selects `min(js, key=lambda j: (rows[idx[j]].get("t_start"), j))` — the
earliest row by decision time, which is the whole point of the rule and the reason
it is the analogue of the valuing convention.

**Executed known-bad:** replacing it with `js[0]` (list order) leaves the suite
**GREEN, 0 failing**. On real data the two may coincide whenever rows arrive in
t_start order, but nothing asserts it, and `first` is the rule that produces the
span's lower bound.

**Closure:** one fixture where a generation's rows are supplied out of t_start
order, requiring `first` to pick the earliest.

### RR5-2 — LOW/MEDIUM — the action key's `side` component is untested

The unit is declared as `"ACTION (distinct slug/side/gen)"` and A1.5 defines it as
distinct `(slug, side, gen)`.

**Executed known-bad:** dropping `side` from the grouping key — which merges the two
sides of one generation into a single "action", changing `n_actions`, the collapse
and therefore every action-unit number — leaves the suite **GREEN, 0 failing**.

This is the definition of the unit the entire round is about, and it is the one
part of it a mutation can change silently.

**Closure:** one fixture with both sides present in a single `(slug, gen)`,
asserting `n_actions` counts them separately.

Both are coverage gaps in a measurement whose arithmetic I checked and found
correct; neither changes a number today.

---

## Executed evidence

At `c180061`, as of 2026-09-02T05:40Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` | **16 of 17 pass**; the single red is `v4 behaviour (git-extracted)` at 9/10 |
| that gate standalone | **10/10, three times** — the documented wall-clock flake (0c), reproduced and not a regression |
| `phase2_iter011_run.py --selftest` | **GREEN, 286 checks, 0 failing** (265 at the previous tip) |
| `v41_boundary_preflight.py --selftest` | 176 checks passed |
| all 8 (arm × unit/rule) comparisons | **8 of 8 favour the candidate** |
| both sides collapsed by the same rule | yes, verified per rule |
| `rows_per_action` | recomputes to **1.754** |
| disagreeing-label generations | **6,108 / 177,674 = 3.438%** |
| `collapse_primary: "max"` premise | **verified in code**: `q4_economics._rank` ranks generations by `max(scores)` |
| RR4-1 contradiction | **0 cells** (12 last round); Q4 now `GATE_PARTIALLY_EVALUATED` |
| RR4-2 both one-draw numbers | **match my independent recomputation exactly** |
| `statistic_n` / unit / basis | now `311640` / `rows` / names the action-unit value |
| mutant: action label `any`→`all` | **KILLED** |
| mutant: `first` → list order | **survives** — RR5-1 |
| mutant: action key drops `side` | **survives** — RR5-2 |
| mutants executed this round | **3** — 1 killed, 2 survived and are the two findings |

---

## Disposition

- **WINNER-HOLD RELEASED.** My round-4 hold is discharged: the action-unit number
  exists, both sides carry it, all three rules are reported, and the comparison the
  gate's conjunct actually names is invariant across every unit, rule and arm.
- **CLOSED:** RR4-3, RR4-1, RR4-2. **No hold from this seat is open at `c180061`.**
- **FILED, not holding:** RR5-1 (`first` selection untested), RR5-2 (`side` in the
  action key untested) — one fixture each.
- **For the USER, with the winner question:** the *comparison* is unit-invariant, but
  the *level* is not — Q1's discrimination reads 0.876 under the ranking convention
  and 0.790 under the valuing convention, with the row-level 0.830 between them. The
  survival still sits on a 500-draw floor whose joint fragility is now correctly
  stated, and **Q4, the decision metric, still fails**.
