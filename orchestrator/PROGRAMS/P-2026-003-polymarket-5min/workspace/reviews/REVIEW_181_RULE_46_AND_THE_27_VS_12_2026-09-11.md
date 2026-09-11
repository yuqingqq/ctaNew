# REVIEW 181 — rule 46 drafted; and the 27 and the 12 are DISJOINT BY CONSTRUCTION, which corrects my own tripwire bound

**REV 138, 2026-09-11T06:54:06Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`. **SEAT_PROTOCOL is the coordinator's to commit** — §1
is a draft to paste, not an edit.

---

## 1. THE RULE, DRAFTED IN HOUSE STYLE FOR RULE 46

> 46. **Reconstruct from the record that STATES a fact, never from arithmetic over adjacent
>     fields.** (2026-09-11, REV, on its own work.) Asked to establish which bytes valued
>     day one, REV derived the process start by subtracting `elapsed_s` from two result
>     mtimes, reached ≈04:04Z, and filed **"probably clean, and I can only say probably."**
>     The heavy-run launch record already held
>     `{"event":"launch","utc":"2026-09-11T04:04:30Z","worktree":"…/ctaNew-wt-be","tip":"651a7b5…"}`
>     — the exact start, **the tree**, and **the commit** — and with the worktree's reflog it
>     closed the question outright: the run imported the FROZEN digest. **The hedge was not
>     caution; it was the cost of inferring what an artifact had already recorded.**
>     **THE CHECKABLE FORM — before deriving any fact about a run, ask "WHICH ARTIFACT IS
>     SUPPOSED TO STATE THIS?" and open it first.** A derived value is evidence only where
>     no record states it, and a reconstruction that never opened the stating record is
>     reported as a reconstruction, never as a limit of the evidence. **This is rule 42
>     applied to TIME and PROVENANCE: there the trap was a label standing in for a property;
>     here it is arithmetic standing in for a record.** The corollary is a claim on the
>     producer side: **a record that states a fact must be reachable from the artifact whose
>     reader needs it** — day one's provenance lives in the LAUNCHER's record and not on the
>     RESULT, which is why forty minutes of reconstruction were available to be spent.

---

## 2. THE 27 AND THE 12 — TWO QUANTITIES, AND THE SETS ARE **DISJOINT BY CONSTRUCTION**

Answered at the producers' own declared definitions, **before DA 230 and not instead of it**
(§2.3).

**THE 27** — `be_gate1_fragment_receipt_20260907_btc.json`:
`selection.n_gap_bearing_windows = 27`, against `population.n_windows = 287`,
`selection.era = clob_v4_1`. **BTC only. Predicate: the window BEARS A GAP.**

**THE 12** — `da_blackout_mask_20260907.json`: `total_masked_windows = 12`,
`total_coverage_absent_windows = 0`, `n_coins = 7`. **All seven coins pooled.** And the
detector's own definition settles it:

> `"below thin_frac x the SAME-DAY (day, coin) median **AND NOT OVERLAPPED BY A GAP-LEDGER
> INTERVAL**"` — `thin_frac 0.05`, `da_content_liveness_rule`, `v1_FROZEN`, authority R-386.

**The mask's predicate carries "not overlapped by a gap-ledger interval" as a CONJUNCT.**
So the 12 are, by construction, windows that are **thin and NOT gap-bearing** — and the 27
are gap-bearing. **The two sets cannot intersect.** They are not two instruments on one
quantity; they are two quantities, on different populations (7 coins vs BTC), under
predicates that are explicitly mutually exclusive. `coverage_accounting` in the same artifact
separates a third status again (`coverage_absent` = no file at all, count 0 here).

**AND A THIRD NUMBER IS HIDING BETWEEN THEM**, in the same fragment receipt:
`binance_gap_index.n_gaps = 3`, `n_that_would_be_excluded = 3`, `excluded_window_starts =
[1788407700, 1788424500, 1788438600]` — with
**`windows_excluded_binance_gap_STATUS = "NOT_APPLIED_ON_THE_DAY_PATH"`** and
`windows_excluded_binance_gap_reported_by_build_rows = 0`. **So on 09-07 there are 27
gap-bearing windows, 3 that a continuity rule WOULD exclude but which is not applied on this
path, and 12 thin-and-gapless windows masked by a different authority.** A reconciliation
that lands on "27 vs 12" has already lost the 3.

*(Also noted in passing: `population.n_windows = 287`, not 288.)*

### 2.1 THIS CORRECTS MY OWN TRIPWIRE, AND MATERIALLY

REVIEW 179 §1(d) computed the era-rebuild bound from the mask: *"12 masked windows … 0.6 % of
2,016 coin-windows (4.2 % if all BTC) … erasing 11,018c needs ~25× concentration."*

**If the rebuild excludes GAP-BEARING windows, that bound was computed on the wrong SET —
not merely the wrong scope.** The right denominator is then **27 of 287 BTC windows = 9.4 %**,
and the concentration needed to erase the deficit falls from ~25× to **~10.6×**.

**That matters, because this programme has measured ~10× concentration in a closely related
quantity:** Q-DA-58 found the worst 10 % of fills carried **77 %** of drift. **So under the
gap-bearing reading a rescue is not a remote tail — it is squarely inside the range already
observed here.** My REVIEW 179 wording ("not impossible") understated it.

**The tripwire therefore binds harder than I said, and its threshold should be set on the
set the rebuild actually uses:**

| if the rebuild excludes | excluded share (BTC) | concentration needed to erase −11,018c |
|---|---|---|
| DA's mask windows (12, 7 coins, thin-and-gapless) | ≤ 4.2 % | ~24× |
| **BE's gap-bearing windows (27 of 287)** | **9.4 %** | **~10.6×** |
| both, disjoint | ≤ 13.6 % | ~7.4× |

**Which row applies is exactly what DA 230 must settle, and the tripwire cannot be set until
it does.** The 25 % |ΔD| threshold I proposed stands as the *reporting* trigger either way;
what changes is how surprised anyone should be if it fires.

### 2.2 THE DISCRIMINATOR I WILL APPLY TO DA 230, DECLARED NOW

1. **Do the two numbers name the same POPULATION?** (BTC-only vs seven coins — measured: no.)
2. **Do their PREDICATES overlap?** (Measured: no — the mask excludes gap-overlap by
   definition.)
3. **Is either number a SUBSET of the other's universe?** 27 ⊂ 287 BTC windows; 12 ⊂ 2,016
   coin-windows. Not nested.
4. **Does the reconciliation account for the 3?** A reconciliation of two numbers that leaves
   a third, differently-statused number unmentioned has reconciled arithmetic, not
   quantities.
5. **Which set does the era rebuild actually exclude?** This is the one the tripwire needs,
   and it is not answered by reconciling 27 with 12 at all.

### 2.3 AND THE RULE-38 POINT, SINCE I HAVE NOW PRE-EMPTED PART OF DA'S ROUND

I have answered §2 **from the two producers' declared definitions**; DA 230 is reconciling
from its own instrument. **Those are different instruments on the same question, so
convergence would count — but only if DA's derivation is independent of this filing.** Under
REVIEW 176 §3 this is the DRIVEN case, so knowledge is harmless and **DA should be shown this
before finishing, not after**: sweeping prior art first costs nothing when the instrument
computes. If DA's answer differs from "disjoint by construction", **DA's is the one to
believe until the definitions are re-read together** — I have read two JSON fields and a
detector docstring, not the code paths that populate them.

## 3. HOLDING

Holding for DA 230 as instructed. Nothing further from me until it lands.

**Standing: REVIEW 178 `0c1f136`, 179 `066c800`, 180 `dfe3ef5` and this filing are stranded —
the shared tree has been dirty at every attempt and rule 21 forbids the retry.**

## 4. SCOPE

Read at the artifacts: both receipts' relevant fields, the blackout detector's frozen
definition, the fragment's era resolution and continuity block. **Not established:** which
predicate the era rebuild will use — the question the tripwire turns on; and the per-coin
split of the 12, whose key I did not resolve (the `coins` sub-dicts use a field name I did
not find, and I did not guess one).
