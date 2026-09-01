# Re-re-review — RR2-1, RR2-2, the as-of naming, and V41-RR2-3
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `35893a4`** (BE's RR2 code `e8a40e1`; the RR2-3 closure
`9a53ea3` verified inside the tip).
**Continues** `REVIEW_011_FIX_CLOSURE_AND_RR1_2026-09-01.md` (`9f31acd`).
**Composed 2026-09-01T17:17:49Z.** One filing, per R-377.

Method unchanged: detached worktree at the pinned tip, ledgers symlinked
read-only, every verdict from execution against committed bytes. No repository
file was modified; every mutant was applied in the lab and reverted. Nothing was
written to a production ledger, artifact, service or tape.

---

## Verdict

### HOLD RELEASED — all three items closed. **No hold from this seat remains open.**

RR2-1, RR2-2, the as-of naming and V41-RR2-3 are each closed, each verified in
both directions, and the batch went past what I asked for in one place that
matters: I tried to defeat the new `GATE_PARTIALLY_EVALUATED` status by dressing
an unwired Q4 in it, and the guard refused.

Three residuals are filed below (**RR3-1, RR3-2, RR3-3**). None holds anything;
all three are coverage rather than behaviour, and I say for each what it would
take to close.

**The honest headline, stated by an independent seat so the register carries it:
the family now publishes ZERO survivors of 24.** Q4 fails family-wise on its own
p. Q1 is not a failure and not a pass — it is undecided, because its frozen gate
has two conjuncts and one was never evaluated. That is a more accurate result
than the six survivors the same data carried this morning, and it came from BE
finding its own defect.

---

## Scope 1 — RR2-1, RR2-2, as-of

### RR2-1 — CLOSED

| what I checked | result |
|---|---|
| the flag | `surviving_cells: []`; the six Q1 cells carry `survives_joint_reading_at_0_05: false` |
| the status | `cells_by_status: {GATE_PARTIALLY_EVALUATED: 6, NO_INCUMBENT_COUNTERPART: 12, OK: 6}` |
| the evidence is kept | `statistic 0.8302782896947175`, `p_value 0.001996007984031936`, `holm_p 0.04790419161676646` — **byte-identical to the pre-fix artifact**, plus `status_before_gate_check: "OK"` and `gate_conjuncts_evaluated: false`. A ruling can be applied without re-running |
| the denominator | `holm_denominator: 24`, `holm_denominator_is_declared_not_evaluated: true` — the new status occupies its slot rather than shrinking the family (A1.4) |
| the rule is stated | `survivor_rule` now reads *"OK now also requires every declared conjunct of the head's gate to have been EVALUATED (not passed — evaluated)"* |
| it does NOT wire the leg | the function's own docstring: *"Whether `apply_incumbent_hazard` runs for Q1 is an estimand-adjacent change to the only surviving head and is the USER's (rule 14)"* — correct; that decision is not a seat's |

**Red-first, both directions, executed:**

| mutant | result |
|---|---|
| the unevaluated-conjunct term removed (`partial = False`) — cells stay OK and survive | **KILLED** — and the check that fires is an ORDERING CONTROL proving the re-assembly step is load-bearing, not decorative |
| **every** cell forced partial (a rule that refuses everything) | **KILLED** by a check proving the predicate needs all three inputs — the ADMIT direction rule 16 demands, and the one I most expected to be missing |
| the vacuity path | the pass refuses a read that touched no cells, no declarations or no leg records (R-289) |

### RR2-2 — CLOSED

| test | result |
|---|---|
| **my shrunk-coverage known-bad** (declaration flipped to `false`, cells and economics made consistent) | **REFUSED** — *"Coverage SHRANK, and a smaller realised set would otherwise be reported as a smaller pass"*. It was ADMITTED at 6 checks last round |
| the real artifact (positive control) | **ADMITTED, 18 checks**, `expected_comparable_heads == comparable_heads == [Q1_arrival, Q4_combined_ev]`, `coverage_is_complete: true` |
| the shrink refusal removed | **KILLED** the suite, by a check named after the known-bad |
| **relabel evasion** (an unwired Q4 dressed as `GATE_PARTIALLY_EVALUATED`) | **REFUSED** — the economics leg catches it, exactly as the guard's `admitted_gap_rule` claims. The new status cannot be used to launder an unwired arm |
| MUTANT A shape (Q4 unwired, still declared comparable) | **REFUSED** |

### The as-of naming — CLOSED

`as_of` is now defined in the artifact as *"the POPULATION READ instant — the
moment this run finished reading its populations and began composing results…
`written_at` is the LATER emission instant; the gap between them is fit and
adjudication time, not additional data."* That is the instant rule 8 asks for,
and the 16:57:01Z → 17:10:38Z gap is now interpretable rather than merely visible.

### The numbers are unchanged and reproduce

Recomputed from the artifact's own `increment_by_window` (seed 20260828, 2000
draws, sorted keys): **6/6 exact on the Q4 statistic, 6/6 on the one-sided p, 6/6
on the two-sided p**; Holm re-derived over 24 cells with **0 disagreements**.
Identity: `carrying_commit e8a40e1`, `dirty_paths: []`,
`producing_code_was_clean: true`, and **runner + library + all 12 lattice files
hash-match that commit** (14 of 14). Artifact 142,609 B.

### RR3-1 — LOW — the coverage floor terminates at a mutable constant, not at the frozen document

`expected` is read from `INCUMBENT_COMPARABLE`, described in the refusal text as
*"which transcribes the frozen prereg 3 gates"*. The transcription is not checked
against the document.

**Executed:** editing the receipt alone is refused (RR2-2, above). Editing the
**module constant and the receipt together** — `INCUMBENT_COMPARABLE['Q4_combined_ev']
= False` plus the matching receipt — is **ADMITTED at 6 checks with
`coverage_is_complete: true`**, because expected and realised both shrank.

I file this at LOW deliberately: every guard's premise lives somewhere, and this
one is at least declared, exported into the artifact and diffable. But the chain
can terminate one step better than a constant, and cheaply — the artifact already
records `preregistration` and `preregistration_commit`, so a check that reads the
frozen §3 gate table at that commit and asserts the constant matches it would end
the regress at a document the USER froze rather than at a line a seat can edit.
`DECLARED_GATES` cannot serve as that second opinion: it encodes a different
proposition (Q2 is `carries_incumbent_term: true` with `comparable: false`), and
that difference is load-bearing for RR2-1 — do not "harmonise" the two.

---

## Scope 2 — V41-RR2-3

### CLOSED. My acceptance mutant dies at the designed check.

| mutant | result |
|---|---|
| `--v41-recv-ns` binding ignored (`if a.v41_recv_ns is not None:` → `if False:`) | **SELFTEST FAILED at check 170** — *REUSED PID, LATE instant pinned*, `rc=0, rows=2` opening at the wrong instant |
| unmutated | **174 checks passed**; 17 gates all pass |

The scenario now has the essential property: one pid at **both** T-60s and T+150s,
so only the recv_ns binding can choose between them, and a wrong instant hits the
`Refused` path with empty stdout.

### RR3-2 — MEDIUM — a LOOSENED recv_ns comparison survives, because one ledger order is fixtured

**Executed known-bad:** changing the filter from `==` to `>=` leaves the suite
**174/174 GREEN**.

The reason is in the fixture: `_reuse = [_T5(777, -60), _T5(777, 150), _V4S5]` is
a single ledger order, and the two scenarios vary **which instant is pinned**, not
**where the pinned row sits**. Under `>=`, the late pin matches only the late row
(passes) and the early pin matches both but takes the ledger-first row, which
happens to be the early one (passes). The pair discriminates a `<=` loosening and
not a `>=` one.

**Closure, one line of fixture:** add the reversed order (`[late, early, v4]`) for
one pin. Under `>=` the early pin then selects the late row and the check dies.

### RR3-3 — LOW/MEDIUM — with a reused pid and NO pin, main() chooses silently instead of refusing

`--v41-recv-ns` is optional. When it is omitted and the pid matches several starts,
`target_start = targets[0]` takes the ledger-first row with no warning — while the
code's own refusal text, on the path where a pin is supplied, says *"the pid and
the start instant must identify ONE row"*. The doctrine and the default disagree.

**Executed:** the `targets[-1]` mutant survives **174/174**, so no check
distinguishes which of several matching starts is selected. (With a pin supplied
the filter yields one row, so first and last coincide — that is why this mutant is
invisible, and it is exactly the unpinned path that the battery does not cover.)

It matters because the earlier runbook told the operator to record the **pid**, and
a boundary where the pid was reused is the case the binding exists for. **Closure:**
when more than one row matches the pid and no `--v41-recv-ns` is given, REFUSE and
print the candidate instants — the operator then has the pair to pass back.

---

## Executed evidence

At `35893a4`, as of 2026-09-01T17:17Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` | **ALL 17 GATES PASS**, exit 0 |
| `phase2_iter011_run.py --selftest` | **GREEN, 220 checks, 0 failing** (202 at the previous tip) |
| `v41_boundary_preflight.py --selftest` | **174 checks passed**, exit 0 |
| guard on the real artifact | ADMITTED, 18 checks, coverage complete |
| RR2-2 known-bad (receipt-only shrink) | **REFUSED** |
| MUTANT A shape | **REFUSED** |
| relabel evasion | **REFUSED** |
| double edit (constant + receipt) | ADMITTED at 6 checks — RR3-1 |
| RR2-1 predicate removed | **KILLED** |
| RR2-1 predicate forced always-on | **KILLED** (the admit direction) |
| RR2-2 refusal removed | **KILLED** |
| recv_ns binding ignored | **KILLED at check 170** |
| recv_ns `==` → `>=` | survives 174/174 — RR3-2 |
| `targets[0]` → `targets[-1]` | survives 174/174 — RR3-3 |
| Q4 statistic / one-sided p / two-sided p | **6/6, 6/6, 6/6 exact** |
| Holm over 24 cells | **0 disagreements** |
| Q1 evidence preserved vs the pre-fix artifact | statistic, p and holm **identical**; `status_before_gate_check: OK` |
| runner + library + 12 lattice files vs `e8a40e1` | **14 of 14 match**; `dirty_paths: []` |
| mutants executed this round | **10** (6 code, 4 receipt-level) — **7 killed a suite or were refused at the guard; 3 survived and are RR3-1, RR3-2, RR3-3**. One further receipt was run as a positive control and admitted |

---

## Disposition

- **RELEASED:** RR2-1, RR2-2, the as-of naming, V41-RR2-3. **No hold from this seat
  is open at `35893a4`.**
- **FILED, not holding:** RR3-1 (bind the expected set to the frozen document),
  RR3-2 (one reversed-order scenario), RR3-3 (refuse an ambiguous unpinned pid).
- **Unchanged and still with the USER**, none of them a seat's to settle: whether to
  wire `apply_incumbent_hazard` for Q1; amendment A2 (one-sided vs the frozen
  two-sided); and the matched-random resolution for the NEXT run, declared
  prospectively rather than raised on this family.
- **The result of record:** 0 of 24 cells survive the joint reading. Q4's increment
  over the committed incumbent is positive in all six cells and clears no
  family-wise bar under either null form; Q1's two AUCs are undecided pending the
  USER's ruling, with their statistic, p and Holm preserved so the ruling can be
  applied without re-running anything.
