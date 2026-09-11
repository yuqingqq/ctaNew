# REVIEW 228 — the per-day standing order, armed; nothing to read yet, and the state a later reader needs

**REV, 2026-09-11T16:36Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

**None of the five records exists.** Filed so the order survives a context loss: the arming
condition, the per-day checklist, and — the part that cannot be reconstructed from the new
file alone — **the carried state each day's futility block must be consistent with**.

## 1. STATE AT THIS READ

```
fwd_v2/p003_de_forward_value_2026091{1,2,3}.json, …0909, …0910    ALL ABSENT
build frontier      09-09  frag tape ----      (book is the next build)
                    09-10  ---- ---- ----
                    09-11  ---- ---- ----      (the day has not closed: it is 16:36Z on 09-11)
                    09-12  ---- ---- ----      (future)
                    09-13  ---- ---- ----      (future)
running             the two collectors only; no build, no valuation in flight
```

**Schedule floor, from the calendar rather than from anyone's plan:** 09-09 and 09-10 are
buildable now; 09-11 cannot be built before ~00:00:05Z on 09-12 (day close + the 5 s markout);
09-12 before 09-13; 09-13 before 09-14. **The population cannot complete before
2026-09-14T00:00Z**, whatever the queue does.

## 2. THE CHECKLIST, PER DAY — ONE REVIEW EACH, ON THE FILE'S APPEARANCE

**(1) The four licensing criteria**, read at the record, not inferred:

- `cells[<arm>].book_receipt.admitted_by` — **`DESCENDANT`** on both arms (`EXACT` means the
  descendant arm was not exercised and is a finding, not a pass);
- **one oracle sha** across both cells, and equal to the day's standalone
  `winner_source_<day>.json`;
- the **unnamed scoring members by digest** — `identical: true`, the compared width, and
  `recorded_from`; and `ruled_lazy_exemption_NOT_WIDENED.outside_the_exemption` unchanged;
- **stage 0's verdict recorded — run-scoped this time.** Day two's was `structured: true` but
  sourced from `/tmp/stage0_freeze_20260908.json`, which is not run-scoped; DE's own note says
  future launches write it into the launch record before the offer. **That is the thing to
  check changed**, not merely that a `stage0` block is present.

**(2) The day's four fields per arm, as the file states them** — `observed_D_cents`,
`p_two_sided`, `n_draws`, `seed` (with `seed_derivation`) — reported as stated, not
recomputed.

**(3) The futility block computed and consistent with the prior days** — §3.

**(4) Anything that would make the record not stand** — the two that have bitten twice:
`STOP_ADVICE` null while an arm is dead, and the unconditional gap table carrying nulls. Both
were closed on day two's v2; a new day is a new emit.

## 3. THE CARRIED STATE — WHAT CONSISTENCY MEANS

```
                          2026-09-07            2026-09-08
CONDVALUE_X_SKEW          -14645.078818000005   -49303.579891000016      both non-positive
HAZARD_OVER_SKEWED_REF     +4925.363903000005   -23977.998804000017      positive, then negative
```

**Invariants every later record must satisfy**, and any breach is a finding:

- `per_day_D_by_arm` must carry **these exact values** for 09-07 and 09-08. A changed prior
  value means the assembly is re-deriving rather than carrying.
- `n_negative` is **monotone non-decreasing** per arm, and starts from **CONDVALUE 2,
  HAZARD 1**.
- `FUTILE` is **`true` for both arms in every later record** — futility does not reverse, and
  a later record showing `false` is an assembly defect, not news.
- `best_attainable_p` may only **rise or stay**: with `G_declared = 7` and `n_neg = k`, it is
  `2·Σ_{i≤k} C(7,i)/2⁷` — `0.453125` and `0.125` today; `0.7734375` at k = 3, `0.9375` at
  k = 4.
- `G_declared` stays **7** and `tolerance_negative_days` stays **0**. A change to either is a
  change to the declared design after the verdict and is the one thing that would make a
  record not stand on its face.
- The verdict is **fixed at day two** (R-910/R-914). Later days are descriptive; **no later
  record can revive an arm**, and I will not read one as doing so.

## 4. WHAT I WILL NOT DO

No licensing read — the path is licensed and code-frozen, and re-litigating that per day would
be re-opening a closed question. **One review per day, on the file's appearance**, and if a
day's record never appears I will say so rather than infer it. If a record appears for a day
whose book was never built, that is itself the finding.

## ROUTED

1. **Coordinator — armed.** Nothing to read at 16:36Z; the next thing that can produce a record
   is 09-09's book, and its inputs are ready.
2. **Whoever reads this after a context loss — §3 is the part you cannot get from the new
   file.** The four criteria are in REVIEW 225; the futility arithmetic is in REVIEW 226; the
   post-population queue is in REVIEW 227 §5.
