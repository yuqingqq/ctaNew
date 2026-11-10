# REVIEW 196 — your four PASS. My two FAIL, and one of them is REAL ON THE LANDING BRANCH RIGHT NOW.

**REV 155, 2026-09-11T08:55Z** (clock read separately). Read-only. Driven against
`origin/de-freeze-chain-v2:live/pm_research/be_score_neutrality.py` (`dcec80a`, 910 lines)
and that ref's declarations, in a scratch copy. **Reporting before 09:04 as instructed.**

## THE TWO FAILURES FIRST

### **F1 — THE LANDING BRANCH'S AMENDMENT CHAIN HAS A HOLE, AND THE RESOLVER IS SILENT**

```
origin/de-freeze-chain-v2 amendments : 1 2 3 4 5 6 7 10     <- v8, v9 ABSENT
origin/mm-research        amendments : 1 2 3 4 5 6 7 8 9    <- v10 absent

resolve_frozen_params_pin() on the branch's own chain:
   pin   -> de_multiday_gate1_params_v31.json   dd58223c…
   chain -> versions [1, 2, 3, 4, 5, 6, 7, 10]     it jumps 7 -> 10 and SAYS NOTHING
```

**Neither branch holds the whole chain.** The freeze's amendments are **split across two
branches**, and the enforcement path on the branch the valuation will run from is resolving a
chain that is missing two ruled amendments — **while emitting a `chain` provenance record that
lists 1–7 and 10 as though that were the chain.**

**THE SUBSTANCE IS UNAFFECTED AND I SAY SO PLAINLY:** I measured earlier that **none of
v2..v9 pins params** (0 params-ish fields in each), so the resolved pin is v31 either way.
**The pin is right. The provenance record is wrong, and it is wrong silently.**

### **F2 — A BROKEN CHAIN RESOLVES INSTEAD OF REFUSING** (REVIEW 194's check 6)

```
v7 removed from a scratch copy -> resolved anyway, chain [1,2,3,4,5,6,10]
```

The resolver `glob`s and sorts by version; **a hole is invisible to it.** R-711 already
requires a declaration chain's `supersedes` pair to verify — **the freeze chain is the
exception, and F1 is that exception firing on real data rather than a fixture.**

**Fix, one predicate: after sorting, assert the versions are contiguous from 1 to the head,
and REFUSE by name otherwise.** That single check turns F1 from silent into loud.

## YOUR FOUR — ALL PASS

| # | check | result |
|---|---|---|
| **1** | version order, last pin wins | **3/3** — amendments present → **v31**; removed → **v29**; synthetic v11 pinning v30 → **v30** (order honoured, v10 correctly overridden) |
| **2** | v10 pins v31's digest and nothing else | **PASS** — `dd58223c6e3654a2…` equals the file; **no other key** under `frozen_parameters` |
| **4** | not a bypass | **PASS** — a params file matching no freeze raises *"the frozen params digest moved: declared dd58223c…"* |
| **3** | driver at `dcec80a` | the driver is now **committed into the pin** — REVIEW 195's provenance objection is discharged by `Q-DE-270`; the import-direction property it rested on was already driven 6/6 |

*(My first pass reported check 1 as failing. **That was my harness, not the resolver** — the
API returns `{"pin", "chain"}` and I read `path` off the top level. Corrected and re-run.)*

## CHECK 5 (MINE) — REFINED: ONE SITE IS BENIGN, **ONE IS A LIVE BROKEN CELL**

Three lines build a path from `FREEZE_REL`:

- **line 109** — inside the resolver itself. Correct.
- **line 141** — `frozen_params` checks the base exists, then **delegates: `pair =
  resolve_frozen_params_pin(decl_dir)["pin"]`**. **Benign.**
- **line 650 — NOT benign.**

```python
frozen_name = Path(json.loads((Path(DECL) / FREEZE_REL).read_text())
                   ["frozen_parameters"]["params"]["path"]).name        # -> v29
note("a malformed later params file cannot move the frozen comparator",
     {v["params_file"] for v in AH.values()} == {frozen_name})
```

and `arm_heads()` → `frozen_params()` → the resolver → **v31**.

> **PREDICTION, TYPED BEFORE THE SELFTEST IS RUN: that cell now compares v31's name against
> v29's name and MUST FAIL.** If DE's selftest reports green, the cell is not testing what it
> says. **This is REVIEW 194 §3 exactly: the fix repaired the enforcement site and left the
> falsifier reading v1.** The cell's sentence also needs restating — once an amendment *can*
> move the comparator, the property is *"only a params file pinned by the resolved chain head
> can move it."*

## ACKNOWLEDGED

**REVIEW 188's unnamed refusal is fixed** — the base-absent path now raises
`UNNAMED_FREEZE_ABSENT`, with the reason recorded in the message. Good.

## WHAT I WOULD DO BEFORE `deRV0907go5` LAUNCHES

1. **Land v8 and v9 onto `de-freeze-chain-v2`** (or the branch onto mm-research) so the chain
   the enforcement reads is the chain the programme ruled. **Cheap, and it removes F1.**
2. **Run the selftest and look at the line-650 cell specifically** (§5's prediction).
3. **F2's contiguity predicate** — one line, and it converts every future hole into a refusal.

**None of the three changes the resolved pin, which is v31 by all six readings. They change
whether the artifact can be trusted to say so.**
