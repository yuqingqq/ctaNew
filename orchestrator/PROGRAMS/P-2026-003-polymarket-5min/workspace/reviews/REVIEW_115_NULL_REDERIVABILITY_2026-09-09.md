# REVIEW 115 — the null's machinery: can the reported statistics be re-derived from the persisted draws?

**REV, 2026-09-09T07:54Z.** Untargeted round. I took leg (iii) — but not the moving
rebuild. I took the **invariant that must hold across it**: *can a receipt's reported
statistics be re-derived from the draws the artifact persists?* That is precisely what
R-825 found violated, it survives whatever ruling B changes, and it is the property that
makes the surviving finding checkable by anyone downstream. Read-only: no lock, no heavy
unit, nothing written under `data/` (the fixture ledger went to scratch).

**RESULT: THE LEG IS SOUND. R-825 is closed, and I established that by RE-DERIVING the
numbers, not by reading the fix.** Two residual observations, neither blocking, and one
scope fact about what the finding actually rests on.

---

## 1. THE PROPERTY HOLDS — driven end to end, both endpoints, both arms

At the tip the ledger is **schema 4** and each `NULL_DRAW` row carries **`settle_value`
beside `value`** (`None` where the day was not valued — *"absent is a fact, not a zero"*).
Driven on a real run through `run_day`, then re-derived from the emitted ledger:

| arm | endpoint | receipt `null_mean` / `null_sd` / `Z` | re-derived from the persisted draws |
|---|---|---|---|
| CONDVALUE | D_E0 | −0.628600 / 91.444566 / 10.475386729 | identical |
| CONDVALUE | **settlement** | −2.600000 / 229.201309 / 0.011343740 | identical |
| HAZARD | D_E0 | 4.319875 / 99.466324 / −9.680046954 | identical |
| HAZARD | **settlement** | −8.600000 / 265.303675 / 0.032415684 | identical |

500 five-second draws and 500 settlement draws persisted per arm; every statistic matches
to the last digit.

**And on a LANDED artifact, not only a fixture:** the sealed 09-07 day run names its ledger
by digest, the digest verifies, and its `null_mean`, `null_sd` and `Z` re-derive from the
500 persisted draws with **zero difference on both arms**. That receipt predates the
settlement endpoint (schema 2), so it can only test the D_E0 half — which is why the
fixture run above matters: it is the only way to exercise both halves today.

## 2. THE CONTROL — my re-derivation can detect the R-825 pairing, and I made it do so

A re-derivation that agrees proves nothing unless it could have disagreed. So I
cross-derived **on purpose**: the settlement observed value against the **5-second**
moments, which is exactly R-825's defect.

```
CONDVALUE : correct (settlement moments) Z +0.011344  sd 229.201
            R-825's (5-second moments)   Z +0.006874  sd  91.445
HAZARD    : correct (settlement moments) Z +0.032416  sd 265.304
            R-825's (5-second moments)   Z -0.043431  sd  99.466
```

Different on both arms — **and on HAZARD the sign FLIPS**, +0.032 against −0.043. That is
the sharpest statement of why the pairing matters: **the wrong moments can change the
DIRECTION of the excess, not only its size.** R-825 reported values 2.2× to 5.4× more
extreme; on these numbers the operative factor is the sd ratio (229 vs 91, 265 vs 99 —
about 2.5×), and the magnitude here is small only because this fixture's settlement excess
is exactly 0.

## 3. THE SCHEMA CHAIN HELD ACROSS THE BUMP

`KNOWN_SCHEMA_VERSIONS = (1, 2, 3, 4)`, writer at 4, and driven: **a reader whose known set
is (1,2,3) REFUSES a v4 ledger with `LEDGER_SCHEMA_UNKNOWN`.** The property REVIEW 109
established for 2→3 was maintained through 3→4 rather than being a one-off.

## 4. TWO RESIDUAL OBSERVATIONS, NEITHER BLOCKING

- **The writer pads; only the reader refuses.** `settle_value: (_sv[i] if i < len(_sv) else
  None)` silently pads a short settlement list at the WRITE, and the mismatch is caught only
  at READ time by `recompute`'s `len(_sv) != n` refusal. The refusal exists and is correct;
  it fires long after the run that produced the file. Refuse at the write — it is cheaper to
  catch at production than after.
- **Alignment is by index and asserted nowhere.** The length check cannot see a
  mis-ORDERED pairing: two length-500 lists in the wrong order pass it and reproduce R-825
  exactly. Safe today, because both are appended in the same loop iteration of
  `null_draws_valued` — but that is an invariant of one function's shape, and it is the
  invariant R-825 violated. One assert at the write records it.

## 5. A SCOPE FACT ABOUT WHAT THE FINDING RESTS ON — not a defect

All **14** point-estimate artifacts carry `run_mode: "POINT_ESTIMATE"` and, in both economic
blocks, `Z: "NULL_NOT_DRAWN_POINT_ESTIMATE_RUN"` — a named status, never a number. That is
honest and it is right. **It also means the settlement numbers on the L250ms/L0ms days are
point estimates with NO control at all.** So when leg (iii) is described as "a null nobody
has driven": for those artifacts there is no null to drive. The nulls that exist are the
D_E0 nulls in the sealed day runs, and those re-derive.

## 6. MY OWN PROBE OVER-FLAGGED, AND I AM RECORDING IT

My first sweep tested `isinstance(Z, str)` and flagged 2 of the 14 point-estimate artifacts
as carrying a non-status Z. Both are benign: one (a superseded 09-05 L250ms attempt) has an
**empty** `economic_settlement` block, so `.get("Z")` returns `None`; the other (09-07)
carries `status: "NOT_VALUED_DAY_NOT_ADMISSIBLE"` — **rule 11 working exactly as it
should**, since 09-07 is not in the admissible set. Neither is a defect. I record it because
a probe that over-flags is how a false positive becomes a "finding", which is the failure
this round's predecessor was about. *(The one thing worth a line: the empty block carries no
status saying why it is empty — on a superseded artifact, so no exposure.)*

## 7. ROUTED

1. **DE — refuse at the write** when `len(null_settle_values)` does not match
   `len(null_values)`, and assert the pairing rather than inheriting it from the loop's
   shape. Two lines, and they close the class R-825 came from rather than the instance.
2. **Coordinator — leg (iii) is examined and it holds.** The surviving finding's statistics
   re-derive from the artifacts that carry them; where they cannot, the artifact says
   `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN` rather than offering a number.
3. **Nothing here invalidates a landed claim.**
