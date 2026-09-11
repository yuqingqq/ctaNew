# REVIEW 172 — addendum: the whole-history threshold search finished, and it moves theta eleven days earlier

**REV 129 addendum, 2026-09-11T03:16:02Z** (clock read separately). Read-only. In band
with REVIEW 170 and 171; **neither is edited** (rule 13). Tip at write time `f4b5bd8`.

**Why this exists:** REVIEW 170 §9 recorded a named scope gap — *"`git log -S` over the
whole history for two literals timed out at 120 s and was not retried."* That search had
been backgrounded and it has now returned. Leaving §9's sentence standing would leave a
false open residual in a landed artifact, so the closure is filed rather than the artifact
amended.

## THE RESULT — IT STRENGTHENS §3, IT DOES NOT CORRECT IT

Both frozen thresholds appear, under the key they are still pinned by, in

```
288df3f  2026-08-27T15:44:25Z  "BE R-216: Phase-2 four-arm receipt + the receipt-path
                                defect that overwrote v1"
  data/pm_5min/derived/phase2_four_arm_v2.json:6    "10%": 0.43525926488298716   (HAZARD)
  data/pm_5min/derived/phase2_four_arm_v2.json:144  "10%": 0.32450609461933483   (CONDVALUE)
```

REVIEW 170 §3 dated theta to **2026-09-05T16:25:47Z** (`be_cancel_axis_null_v1`'s own
`as_of`, the freeze's pin source). **The values are older than that: they existed
2026-08-27T15:44:25Z — eleven days and eight hours before 09-07 closed**, as the `10%`
entry of the four-arm receipt, which is the same `10%` causal-threshold key DA 212 traced
the frozen theta to.

**And the reproduction record is better than "unchanged".** The same two literals carry
through the commits that follow, whose own subjects are the claim: **R-225** *"enforcement
rerun, numbers identical to 8.3e-17"* (08-28T02:54Z), **R-228** *"numbers EXACTLY
identical"* (08-28T04:34Z), **R-230** *"numbers again EXACT"* (08-28T06:06Z), then the
manifest commit (08-28T15:01Z), then the three `be_cancel_axis_null_v{1,2,3}` derivations
(09-05T16:25Z, 09-06T02:46Z, 09-06T04:51Z) that REVIEW 170 §5 read bit-for-bit. **Seven
independent productions of the same two floats, the earliest eleven days before T0.** A
value re-derived after 09-07 would have had to land on all sixteen significant figures
seven times.

## THE CAVEAT, WHICH IS WHY THIS IS A BOUND AND NOT A CENSUS

The traversal emitted, three times:

```
fatal: packfile .git/objects/pack/pack-4411acf…pack cannot be mapped,
       check sys.vm.max_map_count and/or RLIMIT_DATA: Cannot allocate memory
```

**So the scan was PARTIAL and its commit counts (129 and 131) are not reliable — do not
quote them.** What survives the failure is the direction that matters: **a partial
traversal can only MISS older commits, never invent one.** The 2026-08-27 appearance is
therefore a sound *lower bound on the age* of both values, and an older true first
appearance would only strengthen the conclusion. This is the one case where an instrument
that could not see everything still answers the question, and it is worth saying why
rather than quietly reporting the number: **the failure mode and the claim point the same
way.**

## WHAT CHANGES

**Nothing in the verdict.** REVIEW 171 §1 stands unaltered: **09-07 STANDS IN THE
PRIMARY.** The only edits a superseding record should carry are the two dates:

| REVIEW 170 §3 said | the measurement now says |
|---|---|
| theta, both arms, fixed **2026-09-05T16:25:47Z** | **2026-08-27T15:44:25Z**, with seven bit-identical reproductions through 2026-09-06 |
| §9: the whole-history `-S` search "was not retried" | **run and returned; partial traversal, result is a lower bound on age** |

Margin to T0 grows from **2 days 7 hours** to **11 days 8 hours**.
