# REVIEW 140 — the sweep: guards that hardcode one of two sibling blocks

**REV, 2026-09-09T15:05Z.** Read-only: no lock (DA holds it), no heavy unit, nothing written
under `data/`, no book unpickled.

**VERDICT: THE CLASS IS SIX SITES IN THREE MODULES, AND THREE OF THEM ARE NEW. THE WORST IS
NOT A GUARD AT ALL — `de_multiday_gate1_runner.py:2865` COMPUTES THE MULTI-DAY DAY-CLUSTER
VERDICT FROM `r["economic"]["Z"]`, THE DIAGNOSTIC ENDPOINT, WHILE R-801 MADE THE SETTLEMENT
P&L PRIMARY. THE SECOND IS THAT `da_gate1_day_verdict.py` CONTAINS *ZERO* OCCURRENCES OF
`economic_settlement`: DA's independent recompute — the point of the seat — never touches the
ruled endpoint, and its seal predicate reports `sealed: True` on a receipt carrying
`D_E_settle` and both arm totals. Driven. THE ENUMERATION IS CLOSED over the scope in §5.**

---

## 1. THE OPERATION THAT CONSTITUTES MEMBERSHIP

A site is in the class when it **names one of two sibling blocks literally** and then reads a
field **that exists in both**. So the membership test is mechanical, and the field list is
measured, not guessed:

```
shared field names across `economic` and `economic_settlement`:
    ['Z', 'null_draws_summary', 'null_mean', 'null_sd', 'p_location']
```

**Five names, each readable from the wrong block with no error.** Add block-level presence
tests (`is this arm sealed?`), which are in the class for the same reason: the quantity they
look for lives in two places.

## 2. THE SIX SITES

| # | site | reads | cited as protecting / producing | miss | new? |
|---|---|---|---|---|---|
| 1 | `de_multiday_gate1_runner.py:2865` | `r["economic"]["Z"]` | **the multi-day day-cluster verdict and the section-7 predicate** — *"the harmful-fill route STOPS if EITHER arm fails to beat the replay null at day-cluster level"* | **SILENT** — the key exists; it answers on the 5-second diagnostic | **NEW** |
| 2 | `da_gate1_day_verdict.py:925-926` | `recomputed["economic"]` vs `receipt_arm["economic"]`, fields `D_E0, Z, p_location, null_mean, null_sd` | **DA's independent recompute of the economics** | **SILENT** — reports `0 mismatches` having compared the diagnostic only | **NEW** |
| 3 | `da_gate1_day_verdict.py:876` | `arm_block.get("economic")` ∩ `ECONOMIC_FIELDS` | **`receipt_is_sealed` — "absence must not read as a pass"** | **SILENT** — driven below | **NEW** |
| 4 | `de_multiday_gate1_runner.py:1752` | `(result or {}).get("economic")` as the **value source** for the reasons leak guard | the sealed-value guard | **SILENT** | **NEW (the other half of a known defect)** |
| 5 | `de_multiday_gate1_runner.py:5945, 5954` | `(a.get("economic") or {}).get(field)` | `test_statistic_from` — *"asking a point-estimate receipt for a test statistic refuses"* | SILENT | known — REVIEW 139 |
| 6 | `ECONOMIC_FIELDS` (the name set, both modules) | eleven names, none from the settlement block | the seal / leak guard | SILENT | known — REVIEW 123, **de-prioritised by the user's ruling** |

### Site 3, driven

```
both blocks present                      sealed=False  fields_present=['D_E0','Z']
`economic` STRIPPED, settlement PRESENT  sealed=TRUE   fields_present=[]
both absent                              sealed=True   fields_present=[]
```

**A receipt carrying `D_E_settle = 30045.04`, both arm totals and a settlement `Z`, with the
`economic` block stripped, reads `sealed: True`.** That is the user's own planted-settlement
probe reproduced **independently in DA's verifier** — the same hole, in the seat whose job is
to catch it.

### Site 4, stated precisely because it changes the size of the fix

The reasons guard iterates `ECONOMIC_FIELDS` **over `(adm, result["economic"])`**. So the
defect is **two-sided**: even with `D_E_settle` added to `ECONOMIC_FIELDS`, the loop would
never find its **value**, because the settlement block is not a source. **REVIEW 123's fix is
two lines, not one** — the name set *and* the source tuple.

## 3. THE TWO MODELS — this is what the corrected form looks like

- **`de_point_estimate_day.py:273-282`** reads both blocks and iterates the pair
  `(("economic", economic), ("economic_settlement", settlement))`.
- **`da_early_read_verify.py:1408+`** carries a dedicated settlement re-derivation
  (`Z == (D_E_settle - null_mean)/null_sd`) beside its `economic` reader, and its `economic`
  read at `:1458` is *deliberate* — it is comparing against the D_E0 ledger draws.
  **Not instances. The pattern to copy.**

## 4. ONE WATCH ITEM, NAMED RATHER THAN COUNTED

`admissibility` also exists twice — at the arm's top level (`admissible, bar, n_decisions`)
and inside `economic_settlement` (`admissible, class, design_days`). They share **one** name,
`admissible`, and answer **different questions** (the decision-count bar vs the settlement
class). `da_gate1_day_verdict.py:2047` filters days on the top-level one. **That may be
correct; nothing in the artifact says which admissibility a reader should use**, so I record
it as a watch item and not as an instance.

## 5. SCOPE — so the closure is checkable rather than asserted

- **Closed over:** every literal read of `"economic"` or `"economic_settlement"` in every
  `.py` under `live/pm_research`, each classified into production vs battery/fixture, with the
  shared-field-name list measured from the real artifact rather than assumed.
- **Not closed over:** (a) sibling families other than the economic pair — I checked
  `admissibility` and report it above, and did not sweep every block in the receipt;
  (b) readers that reach a block through a **variable** rather than a literal key, which no
  key-literal sweep can see; (c) anything outside `live/pm_research`.
- **The class is CLOSED AT SIX over that scope**, of which three are new. It is not a
  suspicion of a seventh.

## 6. ROUTED — with the line and the block each misses

1. **DE — `de_multiday_gate1_runner.py:2865`**: the day-cluster verdict reads
   `economic.Z`; the ruled endpoint's Z is in `economic_settlement`. **Which endpoint the
   multi-day verdict aggregates is a RULING, not a patch** (rule 14) — but the aggregator
   should name the endpoint it used in its own output either way.
2. **DA — `da_gate1_day_verdict.py`**: add `economic_settlement` to the recompute
   (`:925-926`) and to `receipt_is_sealed` (`:876`). **Zero occurrences today.**
3. **DE — `de_multiday_gate1_runner.py:1752`**: the leak guard's source tuple, alongside
   REVIEW 123's name set. Two lines, not one.
4. **DE — `:5945/:5954`** (already routed at REVIEW 139).
