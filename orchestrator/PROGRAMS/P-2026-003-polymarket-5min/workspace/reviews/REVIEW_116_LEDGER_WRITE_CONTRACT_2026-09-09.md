# REVIEW 116 — the ledger's WRITING path and the contract it publishes to future readers

**REV, 2026-09-09T07:58Z.** Untargeted. I took the ledger and receipt **writing** path, and
with it the contract the receipt publishes to a reader. Read-only: no lock, no heavy unit,
nothing written under `data/` — the tamper test was on a scratch copy, removed afterwards,
and the landed ledger was never written to.

**WHY THIS RATHER THAN VERIFYING A CLOSED GATE ITEM**, since the round offered that
explicitly: a repair that does not work fails **loudly** — it refuses, or it reddens a
battery, and DA 142's drive of BE 112 is what that looks like. A self-description that no
consumer reads back fails **silently**, and every finding tonight came from exactly that
shape. The writing path was the last one on the list where nobody had asked whether the
artifact tells the truth about itself.

**RESULT: THE CONTRACT WORKS, AND IT HAS NOW BEEN EXERCISED ON A REAL DAY FOR THE FIRST
TIME. One finding: the guard that makes it trustworthy is OPTIONAL, and there is no
production caller anywhere — which makes this the cheapest moment there will ever be to
make it required.**

---

## 1. THE PUBLISHED CONTRACT, RUN AS PUBLISHED, ON A REAL DAY

The 09-07 sealed receipt carries its own instruction:

```
recompute_with: "de_decision_ledger.read_ledger(path, expect_sha256=<this sha256>)
                 then .recompute(led, arm)"
```

I ran exactly that. The digest verifies; the file holds **208,083 rows, which is exactly the
`n_rows` the receipt declares**; and per arm:

| | CONDVALUE | HAZARD |
|---|---|---|
| D_E0 | −40138.447387999986 | −123.26724349999859 |
| Z | −6.612840255860946 | 0.45323039478942273 |
| null_mean | −8660.303182718591 | −643.3645723618928 |
| null_sd | 4760.154939079676 | 1147.5340904785055 |
| p_one_sided | 1.0 | 0.3313373253493014 |

**Every one identical to the receipt, |diff| = 0, on both arms**, plus `rho` recomputed at
0.2954 and 0.1700. As far as I can tell from the call sites, **this is the first time the
ledger's published contract has been exercised on a real day by anything but its own
battery** — see §4.

## 2. WHAT IS RE-DERIVED AND WHAT IS READ BACK — said precisely

The contract does not prove as much as "every number reproduces" would suggest, so:

- **RE-DERIVED from stored rows:** `null_mean`, `null_sd`, `Z`, `p_one_sided`,
  `p_two_sided` (from the 1,000 `NULL_DRAW` values) and the legs — `spread_captured`,
  `adverse`, `rho`, the trades cash flow — from the 185,419 `FILL` rows.
- **READ BACK, not re-derived:** `D_E0` is `a["scalars"]["observed_D_E0"]`, and likewise the
  arm and baseline values and the absolutes.

So the contract establishes that **the statistics and the legs are reproducible from the
persisted rows**, and that **the observed value is what the receipt said it was** — which is
the honest and useful decomposition, but it is not a re-derivation of the observed value.

## 3. THE CONTROL — the comparison could have failed, and I made it

I moved **one** `NULL_DRAW` value by +1000, in a scratch copy of a 208,083-row file:

```
WITH the receipt's digest   -> REFUSED DECISION_LEDGER_DIGEST_MISMATCH
WITHOUT a digest            -> reads clean, and recompute returns
                               Z -6.613055993292873  against the receipt's -6.612840255860946
                               null_mean -8658.303182718591 against -8660.303182718591
```

So the agreement in §1 is a result and not a tautology: the comparison discriminates a
single draw in a thousand. **And the second line is the finding.**

## 4. THE FINDING: THE GUARD IS OPTIONAL, AND NOTHING IN THE PACKAGE CALLS THE READER

`expect_sha256` is a keyword with a `None` default, so `read_ledger(path)` performs **no
verification at all** and returns a complete-looking result from whatever bytes are at that
path.

**And every `read_ledger` call site in `live/pm_research` is inside
`de_decision_ledger.py`'s own battery.** The only other occurrence in the package is the
`recompute_with` **string** in the runner — the instruction, not a call. So:

- the ledger exists so that a future reader can re-derive;
- the safe way to do that is documented **as prose in a receipt field**;
- whether it is followed depends entirely on that reader reading the string;
- and the unsafe call is the shorter one.

That is the "stated requirement that is not a checked one" class, and here the cost of
closing it is at its minimum: **make `expect_sha256` required** (or have `read_ledger`
locate the digest itself and refuse when it cannot), so that the safe call is the only call.
**With zero production callers, that change breaks nothing today. It will not stay free.**

## 5. THE WRITER TELLS THE TRUTH ABOUT ITSELF — checked, including the mixed case

Written and read back: `n_rows` declared 12 against 12 lines in the file; `bytes` declared
1028 against `stat` 1028; the returned `sha256` equal to the digest of the finished file
(the digest and size are taken after the `gzip` context closes, so they are of the flushed
bytes, not a partial write). And the header's **`settlement_rows_present: ['A']`** matches
the arms that actually received `SETTLEMENT_SCALARS`/`SETTLEMENT_SLUG` rows, on a fixture
built so that one arm carries a settlement block and the other does not. My REVIEW 104B note
— that the field is computed from the dict it is *about* to write rather than read back —
is technically true and produces no divergence, because both the header and the row loop
test the same key on the same dict.

## 6. ONE CLEAN GUARD FOUND WHILE LOOKING

`recompute` cross-checks the header's `run_mode` against the draws present: `FULL` with zero
draws refuses, and `POINT_ESTIMATE` **with** draws refuses. Given §5 of REVIEW 115 — that all
14 point-estimate artifacts carry `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN` — this is the reader
half of that discipline, and the two agree.

## 7. ROUTED

1. **DE — make `expect_sha256` required on `read_ledger`.** There are no production callers,
   so it is free today; the receipt's own instruction already passes it, so the documented
   path is unaffected.
2. **Coordinator — the ledger's contract is now verified end to end on a real day**, with the
   decomposition in §2 stated precisely: the statistics and legs re-derive, the observed
   value is a read-back.
3. **Nothing here invalidates a landed claim.** The 09-07 receipt reproduces exactly.
