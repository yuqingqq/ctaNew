# Whole maker action chains — signed terms plus graded context

All six previously selected examples, consumed Oct24. Relative times below are settlementId/recorded bounds from market opening, not submission or cancellation times. New signed terms resolve final budget, not continuous resting or private motives. Maker/taker FIFO closing roles are accounting assignments; they do not prove intent.

## Chain 1 — rank 1, market 3716727

Order `0x88c0095c1ea1e1350b55d25f19c83af8fac96c4b3a5d27f395b21f9bb52611a4`. **ASK 69c; signed 30shares.**

1. **Candidate market context — causal link unresolved.** Saved inferred interval 2.325–80.506s; grade `G4_scenario_only` (scenario union estimate only). Scenario/bound grade must be retained; this is not a timestamped posting command. Nearby saved state inside that interval: recorded inventory range -188.7483142857143..0.0shares; aligned BTC5s set `against|flat`; spread 1.0..5.0ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.
2. **Order/queue — terms known, queue unknown.** Signed size 30shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.
3. **Executions — observed.** 1recorded fills, 30.000000shares, first/last settlement records at 80.506/80.506s. The span is not pre-fill resting time.
4. **Ending/next order — graded.** `exhausted_by_observed_fills`. Old next-order link: `reprice` / `sequential_identified`, hash `0x89e05c25834f9256412d3e274d4231a6b9c83dd087ff42402414c8dd53019e36`. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.
5. **Inventory turnover — exact FIFO accounting.** 0.000000shares closed earlier inventory for $+0.0000net. New-entry completed closes: 30.000000shares, $+2.2704; maker closes 15.168828, taker closes 14.831172. Residual 0.000000shares. These components cannot be replaced by the whole account's P&L.
6. **Taking — observed executions, motive not proved.** 12 take record(s); D0 {'opposing': 3, 'q0': 2, 'same': 7}; clock-disagreeing 0; undetermined {'q0': 2} The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.

| Recorded settlement time after opening (s) | Shares | YES price |
|---:|---:|---:|
| 80.506 | 30.000000 | 69.00c |

The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.

## Chain 2 — rank 1, market 3702960

Order `0x02fb943dae46a833559094e61ce64e8ac3ffcbde63f0db8abdc4c0ee9cc8521a`. **ASK 13c; signed 30shares.**

1. **Candidate market context — causal link unresolved.** Saved inferred interval 211.237–213.303s; grade `G4_scenario_only` (scenario union estimate only). Scenario/bound grade must be retained; this is not a timestamped posting command. Nearby saved state inside that interval: recorded inventory range -125.32590813329404..-17.895114264614087shares; aligned BTC5s set `flat`; spread 1.0..5.0ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.
2. **Order/queue — terms known, queue unknown.** Signed size 30shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.
3. **Executions — observed.** 1recorded fills, 6.666600shares, first/last settlement records at 213.303/213.303s. The span is not pre-fill resting time.
4. **Ending/next order — graded.** `observed_budget_incomplete`. Old next-order link: `reprice` / `tied`, hash `0xdef88dc345de50793ca7ede2e8e40cf5e4fcb870460767ffb1f5645a0f055b49`. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.
5. **Inventory turnover — exact FIFO accounting.** 0.000000shares closed earlier inventory for $+0.0000net. New-entry completed closes: 0.000000shares, $+0.0000; maker closes 0.000000, taker closes 0.000000. Residual 6.666600shares. These components cannot be replaced by the whole account's P&L.
6. **Taking — observed executions, motive not proved.** 8 take record(s); D0 {'opposing': 3, 'same': 5}; clock-disagreeing 0; undetermined {} The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.

| Recorded settlement time after opening (s) | Shares | YES price |
|---:|---:|---:|
| 213.303 | 6.666600 | 13.00c |

The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.

## Chain 3 — rank 2, market 3702960

Order `0x084be2d8a41e7076c69f91fe7581a4089c4be1fda60a3a1b52069da39f6c58ca`. **BID 49c; signed 50shares.**

1. **Candidate market context — causal link unresolved.** Saved inferred interval 4.027–4.785s; grade `G1_conflict` (void: conflict flags). Rejected inferred interval; G1_conflict is not a valid posting-time bound. Nearby saved state inside that interval: recorded inventory range 0.0..4.0shares; aligned BTC5s set `flat`; spread 1.0..1.0ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.
2. **Order/queue — terms known, queue unknown.** Signed size 50shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.
3. **Executions — observed.** 3recorded fills, 50.000000shares, first/last settlement records at 4.785/7.030s. The span is not pre-fill resting time.
4. **Ending/next order — graded.** `exhausted_by_observed_fills`. Old next-order link: `reprice` / `sequential_identified`, hash `0xd0f156e83a9bb5552bf835511e4b1eb1c92072822a64c935cb0dd2d6260899d8`. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.
5. **Inventory turnover — exact FIFO accounting.** 0.000000shares closed earlier inventory for $+0.0000net. New-entry completed closes: 50.000000shares, $-3.4600; maker closes 50.000000, taker closes 0.000000. Residual 0.000000shares. These components cannot be replaced by the whole account's P&L.
6. **Taking — observed executions, motive not proved.** 5 take record(s); D0 {'opposing': 1, 'q0': 1, 'same': 3}; clock-disagreeing 0; undetermined {'q0': 1} The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.

| Recorded settlement time after opening (s) | Shares | YES price |
|---:|---:|---:|
| 4.785 | 13.138474 | 49.00c |
| 5.054 | 1.960784 | 49.00c |
| 7.030 | 34.900742 | 49.00c |

The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.

## Chain 4 — rank 2, market 3712725

Order `0xa46229318a1903bd49f5ac1cb2d7d155da80edcca4ca4af179fda9d61d6c4369`. **BID 42c; signed 50shares.**

1. **Candidate market context — causal link unresolved.** Saved inferred interval 62.538–65.084s; grade `G2_bound_constrained` (lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)). Scenario/bound grade must be retained; this is not a timestamped posting command. Nearby saved state inside that interval: recorded inventory range 146.7259086449646..146.7259086449646shares; aligned BTC5s set `flat`; spread 1.0..1.0ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.
2. **Order/queue — terms known, queue unknown.** Signed size 50shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.
3. **Executions — observed.** 1recorded fills, 50.000000shares, first/last settlement records at 65.084/65.084s. The span is not pre-fill resting time.
4. **Ending/next order — graded.** `exhausted_by_observed_fills`. Old next-order link: `reprice` / `tied`, hash `0x6f5fc55834cdd7d96d79be8decf0ed9b8f5aaa61b834c246de6719f4af4f094a`. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.
5. **Inventory turnover — exact FIFO accounting.** 0.000000shares closed earlier inventory for $+0.0000net. New-entry completed closes: 50.000000shares, $-4.4949; maker closes 50.000000, taker closes 0.000000. Residual 0.000000shares. These components cannot be replaced by the whole account's P&L.
6. **Taking — observed executions, motive not proved.** no take record in the episode window The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.

| Recorded settlement time after opening (s) | Shares | YES price |
|---:|---:|---:|
| 65.084 | 50.000000 | 42.00c |

The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.

## Chain 5 — rank 3, market 3712725

Order `0x01ebf81129c65381fd72b5b33ddf3f4aa5e0b84e52b146e789e224ee313344b6`. **ASK 32c; signed 220shares.**

1. **Candidate market context — causal link unresolved.** Saved inferred interval 186.382–187.544s; grade `G1_conflict` (void: conflict flags). Rejected inferred interval; G1_conflict is not a valid posting-time bound. Nearby saved state inside that interval: recorded inventory range -35.83450815862124..213.50688073026762shares; aligned BTC5s set `flat`; spread 1.0..1.0ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.
2. **Order/queue — terms known, queue unknown.** Signed size 220shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.
3. **Executions — observed.** 7recorded fills, 220.000000shares, first/last settlement records at 187.544/195.961s. The span is not pre-fill resting time.
4. **Ending/next order — graded.** `exhausted_by_observed_fills`. Old next-order link: `none` / `none`, hash ``. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.
5. **Inventory turnover — exact FIFO accounting.** 211.856953shares closed earlier inventory for $+6.6326net. New-entry completed closes: 8.143047shares, $+0.0000; maker closes 8.143047, taker closes 0.000000. Residual 0.000000shares. These components cannot be replaced by the whole account's P&L.
6. **Taking — observed executions, motive not proved.** 5 take record(s); D0 {'opposing': 5}; clock-disagreeing 0; undetermined {} The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.

| Recorded settlement time after opening (s) | Shares | YES price |
|---:|---:|---:|
| 187.544 | 54.285714 | 32.00c |
| 187.683 | 4.950312 | 32.00c |
| 188.168 | 10.210526 | 32.00c |
| 188.686 | 14.107500 | 32.00c |
| 189.550 | 125.525122 | 32.00c |
| 189.614 | 2.777778 | 32.00c |
| 195.961 | 8.143047 | 32.00c |

The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.

## Chain 6 — rank 3, market 3716727

Order `0x66e4f311be20759a83daa0ab94f757740294138102f04e5e433393540f759bd1`. **ASK 61c; signed 10shares.**

1. **Candidate market context — causal link unresolved.** Saved inferred interval 0.896–56.719s; grade `G2_bound_constrained` (lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)). Scenario/bound grade must be retained; this is not a timestamped posting command. Nearby saved state inside that interval: recorded inventory range -30.0..0.0shares; aligned BTC5s set `flat`; spread 1.0..3.0ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.
2. **Order/queue — terms known, queue unknown.** Signed size 10shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.
3. **Executions — observed.** 1recorded fills, 10.000000shares, first/last settlement records at 56.719/56.719s. The span is not pre-fill resting time.
4. **Ending/next order — graded.** `exhausted_by_observed_fills`. Old next-order link: `reprice` / `tied`, hash `0xc5d1f7e4998fa2c58342c37cae9b9fd4e75f8b67e650baa8121e4fb01883df35`. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.
5. **Inventory turnover — exact FIFO accounting.** 0.000000shares closed earlier inventory for $+0.0000net. New-entry completed closes: 10.000000shares, $-0.7070; maker closes 2.612786, taker closes 7.387214. Residual 0.000000shares. These components cannot be replaced by the whole account's P&L.
6. **Taking — observed executions, motive not proved.** 3 take record(s); D0 {'opposing': 3}; clock-disagreeing 0; undetermined {} The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.

| Recorded settlement time after opening (s) | Shares | YES price |
|---:|---:|---:|
| 56.719 | 10.000000 | 61.00c |

The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.

