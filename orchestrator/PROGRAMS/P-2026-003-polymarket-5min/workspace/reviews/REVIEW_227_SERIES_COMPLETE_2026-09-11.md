# REVIEW 227 — both residuals closed on the v2 record, nothing else moved (176 paths, every one accounted for), and the forward-test review series is COMPLETE

**REV, 2026-09-11T16:04Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

`fwd_v2/p003_de_forward_value_20260908_v2.json`, 16:03, 28,523 bytes.

## 1. BOTH RESIDUALS CLOSED

```
STOP_ADVICE                "STOP_FOR_FUTILITY"                        (+ STOP_ADVICE_why)
per_window_table            43 rows, 43 non-null gap_seconds, 43 non-null utc
  sample                    {"window_start": 1788825900, "utc": "00:05:00Z",
                             "gap_seconds": 1.337, "n_gap_intervals": 1,
                             "share_of_day_gap_time": 0.005050104}
```

**REVIEW 190's defect is finally fixed** — it survived from day one's emit to day two's first
emit and is closed here with real values, not placeholders.

## 2. THE VERDICT IS UNCHANGED, AND "NOTHING ELSE MOVED" IS COMPUTED

```
CONDVALUE_X_SKEW        FUTILE=True  best_p=0.453125  n_neg=2  days ['2026-09-07','2026-09-08']
HAZARD_OVER_SKEWED_REF  FUTILE=True  best_p=0.125     n_neg=1  days ['2026-09-08']
ANY_ARM_ALREADY_DEAD / EVERY_ARM_ALREADY_DEAD:  True / True
```

I diffed the two records key by key rather than taking the claim: **176 differing paths, and
every one is one of the four changes**:

```
172   /emit/per_window_table_unconditional     (43 rows x 4 fields that were null)
  1   /emit/STOP_ADVICE
  1   /emit/STOP_ADVICE_why                    (added)
  1   /supersedes                              (added)
  1   /at_utc                                  (the emit time)
```

**The cells, the futility block, `stage0`, `cohort_agreement`, the tripwire block, the floors
and `per_day_D_by_arm` are byte-identical.** Nothing else moved.

**And the supersession is checkable, and checks:**

```
supersedes.path    …/p003_de_forward_value_20260908.json
supersedes.sha256  51decc84c8b368d857c6cdc5cb3ecb4a27ffced89bff98005e33fcd6295aabac
the file on disk   51decc84c8b368d857c6cdc5cb3ecb4a27ffced89bff98005e33fcd6295aabac      MATCH
kept_as            "provenance, unedited (rule 13)"
```

## 3. ONE NUMBER I CANNOT SETTLE, AND IT DOES NOT TOUCH THE VERDICT

The gap table's **window set is exactly right** — I checked it against BE's independent
artifact:

```
be137_gap_windows_20260908.json   n_gap_bearing 43,  43 starts listed
the emit's table                  43 rows
same window set?                  TRUE   (0 only-BE, 0 only-emit)   missing_interior_window: []
```

**The seconds do not match my own earlier measurement.** The table sums to **264.747 s**; my
REVIEW 141-era figure for 09-08 was **143.8 s (0.167 % of the day ≈ 144.3 s)**.

I am not calling that a defect, and the reason is my own record: **REVIEW 183's headline on
this exact quantity was withdrawn in REVIEW 184** when I found `gap_overlaps` is a boolean and
the two figures were the ends of a range rather than a correction and a truth. Summing
per-window gap seconds double-counts an interval that spans a window boundary; a union does
not. That is the live hypothesis and it would explain a factor in this neighbourhood, but
**I cannot settle it in a read-only round** and BE's artifact carries the window set, not the
seconds.

What it is not: an input to `D` or `p`. The table is the unconditional diagnostic the ruling
asked for, and both arms' futility rests on the two days' D signs, which are unaffected.
`share_of_day_gap_time` sums to exactly `1.000000`, so those shares are internally consistent
as shares **of the gap time** — the field name reads as "of the day" and means "of the day's
gap time", which is worth one word in the name.

## 4. THE SERIES IS COMPLETE

With both residuals closed and the verdict unchanged, **the review series for the forward test
is complete.** The record that stands is
`fwd_v2/p003_de_forward_value_20260908_v2.json`, superseding the 15:54 record which is kept
unedited as provenance under rule 13.

**What the series established, at the artifact:**

- **LICENSED** (REVIEW 225): the descendant arm exercised and recorded on both arms, one oracle
  read once, the three unnamed scoring members admitted on 64-hex digests rather than a typed
  name list, and stage 0's freeze verdict recorded structurally — `POPULATION_FREEZE_HOLDS`,
  67 of 67, with the verifier's own digest and the declaration it read.
- **THE RESULT STANDS** (REVIEW 226): both arms futile at G = 2 of 7, best attainable
  day-cluster p **0.453125** and **0.125** against a Holm step-one threshold of **0.025**,
  recomputed here from first principles and reproducing the record exactly.
- **Falsifier (a) green** on both arms to every printed digit, on a rebuilt book and a later
  ledger; **(c) green** since `2fff936`.
- **The outcome**: `NOT_ESTABLISHED_AT_THIS_POWER`, both arms — a statement about the
  instrument's reach, not about the world.
- **And the caveat that outlives it**: the design's own floor at G = 7 is 0.015625, which
  against the **69-candidate screen** that produced the arms is 1.078 — **even a perfect run
  could not have delivered evidence that survives its own selection.** That was computed before
  any forward day was valued and belongs beside the fail sentence, not after it.

**One correction of mine stands in the record** (REVIEW 226 §1): REVIEW 224/225 called
CONDVALUE live when the declared design had already killed it at day one. The emit had it
right and I did not.

## 5. TO THE POST-POPULATION QUEUE (R-913)

Nothing below blocks the result; each is filed and none is closed:

1. **The gap-seconds total** (§3) — 264.747 s summed vs 143.8 s measured; settle by computing
   the union of gap intervals, not the per-window sum, and record which the table means.
2. **`share_of_day_gap_time`** — the name says "of the day", the value is "of the day's gap
   time".
3. **The self-vouching residual on a direct invocation** and **`DE_VALUATION_PREFLIGHT_OFF`**,
   still an unrecorded bypass (REVIEW 221 §5, REVIEW 222 §3) — the role-swap inside the frozen
   modules was ruled post-population.
4. **`n_verified` vs the declared total in the freeze verifier's own return** — closed in the
   stage-0 block at 67/67, still worth reconciling in the function.
5. **The declared-digest cross-check for unnamed members** (REVIEW 223 §2.2) — one lookup that
   turns "unchanged since the build" into "and the build's copy was the declared one".
6. **`de_asymmetry_null_run`'s second per-arm oracle read** (REVIEW 212 §4), declared
   post-population.

## SCOPE

Closed over: the v2 record read key by key; a full structural diff against the superseded
record with every differing path classified; the supersession sha verified against the file on
disk; the futility arithmetic already recomputed in REVIEW 226; the gap table's window set
compared against BE's independent `be137` artifact; the gap totals and shares summed.
**Not closed over:** the 264.747 vs 143.8 question (§3), which needs the union computation;
conjunct (b), moot for these arms; and the five remaining days, which the futility stop makes
unnecessary to value for this test.

## ROUTED

1. **Coordinator — the series is complete**; the standing record is the v2 file and the outcome
   sentence is REVIEW 226 §5 with §6 beside it.
2. **R-913 — the six items in §5**, of which only the first was found in this round.
