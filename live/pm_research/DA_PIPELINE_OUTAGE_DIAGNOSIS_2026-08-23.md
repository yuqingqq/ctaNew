# DA pipeline outage — diagnosis

Written 2026-08-23 by the DA plane under dispatch **D-1a**, **before any repair
was applied**, because the first finding is a contracts-level fact and not an
implementation detail.

Scope: why both derivation lanes have been failing hourly since 2026-08-22.
This document **decides nothing**. It writes no admissibility rule, excludes no
row, and changes no guard — those are coordinator-gated (§2.4 R-ADMISS, §2.1).

> Sections 1.1-1.5, 2 and 3.1-3.3 were written **before any code was touched**,
> as D-1a requires. Sections 1.4a, 1.5a, 3.4 and 3.4a were appended after the
> repair and the census ran, and say so.
>
> Current status: **A repaired and verified** (`measurement_batch.py` only);
> **B characterized, proposal only, `normalize_clob` unchanged and still
> raising**; **C reported and untouched**; the run-record trap in 1.4a found and
> **not** fixed.

---

## 0. Summary — there are THREE failures, not two, and the third is the tallest

| | failure | site | status |
|---|---|---|---|
| **A** | `immutable JSON mismatch` — every re-verify of an already-written health artifact | `measurement_batch.py:229` | **REPAIRED, verified** (§1.5a) |
| **B** | `duplicate identity has conflicting payload` | `tier1_pipeline.py:1058` | **DA characterizes, coordinator ratifies** |
| **C** | `leak canary did not bind knowledge-time truncation` | `replay_canary.py:282` | **newly found — reported, not touched** |

**A is not what is stopping the lane.** It is what the lane hits *first* on a
retry. Repairing A alone does not restore the measurement lane: with A's
comparison bypassed in a read-only harness, day 2026-08-20 runs btc, eth, sol,
xrp and then aborts on **C** at doge. C is older than A and A has been masking
it.

This corrects the dispatch's blast-radius line. `tier1/` holds only
btc/eth/sol/xrp for `day=2026-08-20` **because C aborted the batch at the fifth
coin**, not because A did.

---

## 1. Failure A — the contracts-level fact

### 1.1 The mechanism

`_write_immutable_json` (`measurement_batch.py:217`) does two things with two
different notions of equality:

```python
actual = _content_hash(unhashed)          # sha256(canonical_json(payload - hash_field))
if not declared or declared != actual:    # (i) hash check, over the JSON PROJECTION
    raise ValueError(...)
if path.exists():
    existing = json.loads(path.read_text())
    if existing != document:              # (ii) re-verify, over PYTHON VALUES
        raise RuntimeError(f"immutable JSON mismatch at {path}")
```

`canonical_json` is `json.dumps(value, sort_keys=True, separators=(",",":"),
allow_nan=False)`. So **(i) hashes the JSON projection of the payload; (ii)
compares the pre-projection in-memory payload against the post-projection parsed
file.** Those are different equality relations, and they disagree exactly where
the projection is lossy.

### 1.2 The answer to the coordinator's question

**Yes — the health hash covers every field written.** Self-exclusion of
`hash_field` is necessary and is not the defect. The defect is one level down:

> **The hash covers every field, but it hashes the field's JSON PROJECTION, not
> its value. `R-BATCH` immutability is therefore a property of the artifact
> bytes, never of the payload that produced them.**

JSON projection is not injective on Python values:

| distinct in Python | identical after projection |
|---|---|
| `("A-TWAP-1", "ab09…")` | `["A-TWAP-1", "ab09…"]` |
| `{1: "x"}` | `{"1": "x"}` |

(sets and datetimes raise in `json.dumps`, so they are not a collision source.)

**Read both directions, because only one of them bit us.**

- **Liveness — the live defect.** A byte-identical artifact fails re-verification
  and the failure is fatal and permanent. The first write succeeds (no file to
  compare against); every subsequent re-verify of that same artifact raises.
  Because the unit runs `--catch-up --since 2026-08-20 --max-days 1`, catch-up
  re-verifies its earliest day forever and never advances. Measured: **21
  consecutive hourly failures**, spine frozen at `day=2026-08-20` for three
  days.
- **Safety — what R-BATCH may no longer be read to claim.** Two payloads that
  differ in Python but project to the same JSON receive the **same hash and the
  same content-addressed path**, and after the repair below they will re-verify
  as equal. That is defensible — the artifact *is* the JSON, and anything the
  JSON cannot express was never in the artifact — but it must be said out loud:
  `R-BATCH`'s `resumable` check ("absence of a receipt permits immutable
  per-coin artifacts to be **validated** and reused") validates the bytes. It
  does not certify that the producer's in-memory value was the same value.

**Recorded as a sixth instance of *the name is not the definition*:** the
function is named `_write_immutable_json` and reads as though it enforced
immutability of the payload. It enforces immutability of the artifact, and
compared the two as if they were one type.

### 1.3 The exact field, and the census behind it

Producer, `measurement_batch.py:431`:

```python
rule_pairs = {(rule.id, rule.spec_hash)}      # a set of TUPLES
check("FROZEN_RULE_BINDING",
      rule_pairs == {(rule.id, rule.spec_hash)},
      observed=sorted(rule_pairs),            # -> [ ('A-TWAP-1', 'ab098a55…') ]
      expected=[rule.id, rule.spec_hash])     # -> [ 'A-TWAP-1', 'ab098a55…' ]
```

A census that walked every immutable payload the measurement lane writes, for
every coin, looking for tuples, sets and non-string dict keys:

| payload | tuple / non-string key found |
|---|---|
| `health_hash` (4 coin payloads reached) | **1 each** — `checks/FROZEN_RULE_BINDING/evidence/observed[0]` |
| `run_hash` (`daily_pipeline` run record) | none |
| `batch_hash` (cross-coin receipt) | not reachable — the lane aborts on **C** before commit-last |

`observed` is a list-of-tuple and `expected` is a list-of-str; they are not even
the same shape. That asymmetry is cosmetic, but it is how the tuple survived
review.

### 1.4 Blast radius of the defect CLASS, not the instance

| site | comparison | verdict |
|---|---|---|
| `measurement_batch.py:229` | in-memory payload vs parsed JSON | **live defect** |
| `daily_pipeline.py:509` | in-memory payload vs parsed JSON | **same class, latent** — safe today only because its payload happens to contain no tuple; nothing enforces that |
| `tier1_pipeline.py:1376`, `:1458` | `sha256(file)` vs recorded `output_sha256` | **correct by construction** — bytes on both sides |

`daily_pipeline.py:509` also verifies the on-disk record's own declared hash
before comparing, which `measurement_batch.py` does not. The stronger check
lives at the weaker-typed site.

### 1.4a A second, sharper instance of the same class — found by trying to fix the first

The latent site was patched and then **reverted**, because patching it exposes a
worse trap.

`write_run_record` stores the run receipt at a **fixed** path
(`runs/day=…/coin=…/lane=…/run.json`) while binding its content to
`pipeline_code_sha256 = sha256(daily_pipeline.py)`. Measured:

```
run.json recorded pipeline sha   0640f41c2caf9d2d…   == HEAD daily_pipeline.py
after a one-line edit            b41ba9db26699ba3…   != recorded
```

So **any** edit to `daily_pipeline.py` — a bug fix included — makes every
already-derived coin-day raise `run-record merge-never-overwrite` forever. The
artifact is immutable, the path is not content-addressed, and the content
depends on mutable code. There is no way to change that file without either
orphaning existing run records or rewriting artifacts under `tier1/`, which is
prohibited without coordinator sign-off.

Contrast `health`, which binds the same kind of code hash but is stored at a
**content-addressed** path. It absorbed the change cleanly — 2026-08-20/btc now
holds three health artifacts from three validator versions, each valid, none in
conflict:

| `validator_code_sha256` | artifact |
|---|---|
| `a04255c2…` | `health=ec292ff9…` |
| `3b5a93da…` | `health=bd569509…` |
| (this repair) | `health=8854ccbe…` |

**Content-addressing is what makes an immutable artifact survive a code change;
a fixed path plus a code-derived field cannot.** `daily_pipeline.py` is
therefore effectively frozen until the coordinator rules on the run-record
layout. Flagged, not fixed.

### 1.5 The repair, stated before it is applied

Take the projection **once**, at entry, so the hash, the comparison and the
bytes on disk are all the same object:

```python
document = json.loads(canonical_json(dict(payload)))
```

Then `existing != document` compares list to list. This **relaxes nothing**:

- a genuinely different payload projects differently, hashes differently, and
  lands at a *different* content-addressed path — it was never reaching this
  comparison in the first place;
- an on-disk file that was edited or corrupted still projects differently and
  still raises;
- `declared != actual` is untouched, and since `existing`'s projection now
  equals `document`'s, the existing file's declared hash is implied to match.

**Applied to `measurement_batch.py` only.** The identical patch at
`daily_pipeline.py:509` was written, tested and **reverted** for the reason in
§1.4a; that file is restored to `0640f41c…`, byte-identical to `HEAD`.

### 1.5a Verified — two consecutive runs of the exact unit command

```
python3 -m live.pm_research.measurement_batch \
    --catch-up --since 2026-08-20 --lane measurement --max-days 1 --scheduled --json
```

| run | result |
|---|---|
| 1 | writes health for btc, eth, sol, xrp — **no `immutable JSON mismatch`** — aborts at doge on **C** |
| 2 | **re-verifies run 1's health artifacts without raising** — aborts at doge on **C** |

Run 2 is the test: before the repair, every second run of the same day raised.
The failure now reported is **C**, which is the pre-existing blocker A was
masking. **The measurement lane is still down, and A is no longer why.**

### 1.6 Contract wording — PROPOSED, not written

`contracts.yaml` changes are coordinator-gated (§2.2). Proposed, additive, for
`R-BATCH`:

> `projection: immutability is asserted over the canonical JSON projection of an
> artifact, never over the in-memory payload; a re-verify compares projections`

---

## 2. Failure C — a guard that fires on a routine sample property

**Not in the D-1a dispatch. Found while proving A. Nothing about it has been
changed, and the prohibition on relaxing a guard is why it is written up rather
than fixed.**

`replay_canary.py:282`:

```python
if event_only == 0 or disagreements == 0:
    status = "INVALID_UNBOUND_GUARD"
elif math.isclose(delta, 0.0, abs_tol=1e-15):
    # A score delta can cancel even when the selected states differ.  That
    # is a review flag, while event-only reads prove the guard is wired.
    status = "BOUND_ZERO_SCORE_DELTA"
```

and `run_from_partitions:422` raises on `INVALID_UNBOUND_GUARD`, aborting the
whole batch for every remaining coin.

### 2.1 Every canary on disk

| day | coin | status | disagreements | event-only reads | delta |
|---|---|---|---:|---:|---:|
| 2026-08-20 | btc | VALID_GUARD_BITES | 4 | 568 | 0.0139 |
| 2026-08-20 | eth | VALID_GUARD_BITES | 3 | 566 | 0.0104 |
| 2026-08-20 | sol | VALID_GUARD_BITES | 3 | 564 | 0.0104 |
| 2026-08-20 | xrp | VALID_GUARD_BITES | 2 | 568 | 0.0069 |
| 2026-08-20 | **doge** | **INVALID_UNBOUND_GUARD** | **0** | **568** | 0.0 |
| 2026-08-21 | btc | VALID_GUARD_BITES | 2 | 566 | 0.0069 |
| 2026-08-21 | eth | VALID_GUARD_BITES | 1 | 570 | 0.0035 |
| 2026-08-21 | **sol** | **INVALID_UNBOUND_GUARD** | **0** | **566** | 0.0 |

Coin order is btc, eth, sol, xrp, doge, bnb, hype. **doge is the fifth coin on
2026-08-20 and sol is the third on 2026-08-21** — which is exactly the set of
coins present on disk for each day, and therefore the direct cause of the
partial days.

### 2.2 The guard is wired in every one of those rows

`event_only ≈ 566` in the INVALID rows too. The deliberately-leaky twin read
past the knowledge boundary **568 times on doge** and the knowledge-time view
refused every one of them. Wiring is not in question. What is zero is
`decision_disagreements` — the look-ahead never flipped a *winner*.

That is a property of the **day's data**, not of the harness. Across the eight
coin-days above the disagreement rate is **15 / 2,304 windows = 0.0065 per
window**, so a 288-window coin-day comes up empty with probability
`(1 − 0.0065)^288 ≈ 0.15`:

| quantity | value |
|---|---:|
| P(one coin-day has zero disagreements) | **0.152** |
| expected INVALID coin-days per 7-coin day | **1.07** |
| P(at least one INVALID coin in a 7-coin day) | **0.686** |
| observed | **2 of 8 coin-days** |

**About seven days in ten will abort on this guard by chance.** The lane cannot
run a week without hitting it. The contract's own note
(`LeakCanary`: *"score delta may cancel on a small slice; event-only selected
reads and decision disagreements prove the guard is wired"*) and the code
comment two lines below the branch both say event-only reads are the wiring
evidence — but the `disagreements == 0` arm reaches `INVALID` before
`BOUND_ZERO_SCORE_DELTA` can be considered.

**This is the mirror of the corpus's own lesson.** Three times the programme has
recorded *a gate that cannot fire is not a gate*. This one fires on noise, and
fatally.

### 2.3 What DA proposes, and does not do

Coordinator decision, because it changes what evidence is admissible and what
halts a lane. DA has changed nothing and offers the options with their costs:

1. **Route `event_only > 0 and disagreements == 0` to `BOUND_ZERO_SCORE_DELTA`**
   — non-fatal, review-flagged, matching the contract's stated semantics.
   Cost: a coin-day whose guard genuinely stopped being wired *and* which
   happens to have zero disagreements would no longer halt; `event_only > 0`
   is what rules that out, and it is checked.
2. **Keep `INVALID` but make it non-fatal for the coin** — record the status,
   exclude that coin-day from the committed lane, continue the batch. This is an
   exclusion rule and therefore squarely R-ADMISS: it needs both arms and the
   excluded set reported beside the retained one.
3. **Leave it fatal.** Then the lane's expected uptime is ~31 % of days and the
   derivation backlog grows monotonically. Stating it so it is chosen rather
   than inherited.

DA's read: the guard's *question* is "is knowledge-time truncation wired", and
`event_only` answers it; `disagreements` answers a different question ("did the
look-ahead matter on this slice"), which is a finding, not a precondition. But
that is an argument, not a decision, and the decision is not DA's.

---

## 3. Failure B — the duplicate-identity conflict

Characterization and proposal only. **Which copy is authoritative and whether
any window is excluded is a selection decision under R-ADMISS and is not made
here.**

### 3.1 The key that aborted the run

`tier1_pipeline.normalize_clob` dedups by `_raw_message_key`. For a `book`
message that key is `(event_type, asset_id, hash, timestamp)` — it deliberately
excludes the book contents, because the venue's own `hash` identifies them. The
equality test that follows, however, digests the **whole message**:

```
slug        btc-updown-5m-1787184000        (2026-08-20 00:00:00 UTC window)
event       book
asset_id    70343184730245917298138381441659449428594466101007298184643586135827033537130
venue hash  88dedbc08d479e0f54a230d87812c946cd0f3e3d      (identical in both copies)
timestamp   1787184063582                                  (identical in both copies)

copy 1      recv_ns 1787184063654092900
copy 2      recv_ns 1787184063739317313      (+85.2 ms)

difference  the OPTIONAL fields `last_trade_price` and `tick_size` are present
            in one copy and absent in the other.  No shared field disagrees:
            bids, asks, market and asset_id are byte-identical.
```

So the two payloads describe **the same book**, carry the **same venue hash**,
and differ only by an envelope field-set. The guard is not reporting a data
conflict; it is reporting an envelope difference through a whole-message digest.

### 3.2 Which collector wrote each copy

Both copies are in the **same single shard**
(`raw/20260820/btc-updown-5m-1787184000.jsonl.gz`; no `.1.gz` exists for this
slug), so they were written by one process, 85 ms apart.

The versioned start/stop ledger does not reach back to them:

| event | UTC | version | pid |
|---|---|---|---|
| — | *conflict at 2026-08-20 **00:01:03.65*** | **no ledger coverage** | **unknown** |
| collector_start | 2026-08-20 14:50:21 | `clob_v2` | 2554828 |
| collector_stop | 2026-08-20 15:09:44 | `clob_v2` | 2554828 |
| collector_start | 2026-08-20 15:10:28 | `clob_v2_1` | 2565208 |
| collector_stop | 2026-08-20 16:30:50 | `clob_v2_1` | 2565208 |
| collector_start | 2026-08-20 16:31:26 | `clob_v3` | 2603653 |
| collector_stop | 2026-08-20 17:44:19 | `clob_v3` | 2603653 |
| collector_start | 2026-08-20 17:44:20 | `clob_v3_1` | 2639831 |
| collector_stop | 2026-08-21 01:42:15 | `clob_v3_1` | 2639831 |
| collector_start | 2026-08-21 01:42:58 | `clob_v3_1` | 2858536 |
| collector_start | 2026-08-21 01:45:19 | `clob_v3_1` | **2860318** (running now) |

**The conflicting records predate the ledger.** The `collector_start`/
`collector_stop` records were introduced by the same commit that created the
gap ledger, so the pre-`clob_v2` era that wrote 2026-08-20 00:01 has **no
version or pid record at all**. Its era cannot be named from the ledger, only
bounded: it is pre-`clob_v2`, i.e. inside the tape the audit already classes as
degraded.

**Two incidental findings, both reportable rather than actionable here:**

1. **`pid 2858536` has a `collector_start` and no `collector_stop`.** A second
   collector started 2 m 21 s later and is the one running now. Either the first
   died without draining or the two overlapped. The wire records cannot settle
   it — `recv_ns` and the JSON body carry no pid or `collector_version`; only
   the ledger does. Raised for OPS (D-1b) as a supervision fact.
2. **Restart overlap produced clean numbered shards, not interleaved writes.**
   `.jsonl.N.gz` counts are 22 / 134 / 14 / 0 / 0 for 2026-08-19…23, and the 14
   on 2026-08-21 are exactly windows `1787276400` and `1787276700` across all
   seven coins — the two windows spanning the 01:42:58 and 01:45:19 restarts.
   That is the restart mechanism behaving as designed.

### 3.3 Is this the known duplicate-collector overlap?

**No — not by the mechanism the audit describes, on the evidence above.** The
audit's overlap is *sockets*, not processes: `clob_v2_1` reported **21
overlapping sockets, three per coin**, during handover. Same-process multiple
subscriptions to the same market would deliver the same venue message more than
once — which is consistent with an 85 ms re-delivery inside one shard. But the
conflicting records sit **13 hours before** the first ledgered restart, so the
audit's measured instance is not this instance, and the mechanism is inferred
rather than measured. Stated as an inference, and labelled as one.

### 3.4 Population — corpus-wide census

A census over every `(day, coin, slug)` on the raw tape, applying the pipeline's
own `_raw_message_key` and payload digest (`da_duplicate_identity_scan.py`,
receipts `derived/da_duplicate_identity_v1.json` and
`derived/da_duplicate_identity_anatomy_v1.json`):

```
records scanned          264,851,295        slugs scanned      7,031
exact duplicate records      251,599 (0.095 %)   -- byte-identical, collapsed silently
CONFLICTING keys                 518            -- across 463 slugs (6.6 %)
```

| by day | keys | | by coin | keys | | by event type | keys |
|---|---:|---|---|---:|---|---|---:|
| 2026-08-19 | 64 | | bnb | 154 | | **`price_change`** | **499** |
| 2026-08-20 | 156 | | eth | 92 | | `book` | 19 |
| 2026-08-21 | 148 | | doge | 71 | | | |
| 2026-08-22 | 130 | | sol | 61 | | | |
| 2026-08-23 (partial) | 20 | | hype | 58 | | | |
| | | | btc | 43 | | | |
| | | | xrp | 39 | | | |

**Every day and every coin is affected, including 2026-08-22 and 2026-08-23,
which are wholly inside `clob_v3_1`.** This is not a legacy-era artifact and it
will not age out. **The key that aborted the lane is a `book` — the 3.7 %
minority.**

### 3.4a The two mechanisms are different findings, and only one is a duplicate

Anatomized across all 518 pairs:

**`book` — 19 keys — a genuine re-delivery.**

| | |
|---|---|
| shared fields that differ | **none, in all 19** |
| fields present in one copy only | `last_trade_price`, `tick_size`, in all 19 |
| same market | 19 / 19 |
| separation | p50 85 ms, max **15.09 s** |

Same venue `hash`, therefore the same book. One state, seen twice, with a
different envelope. This is the aborting key's class.

**`price_change` — 499 keys — NOT duplicates. Two different book states.**

| | |
|---|---|
| differing row fields | `best_bid`, `best_ask` — **and nothing else, in all 499** |
| row count equal | 499 / 499 |
| same market | 499 / 499 |
| `bid(Up) + ask(Down)` per copy | **1.0000 in all 1,996 sums computed** |
| top-of-book move between copies | 0.01 (398) · 0.02 (218) · … · **0.27 (2)** |
| separation | p50 **28.6 µs**, max 113 ms; same receive millisecond in only 329 / 499 |

Worked example, `bnb-updown-5m-1787160000`, both copies `timestamp` 1787160172898,
consecutive wire lines, 5.2 µs apart:

```
copy 1   Up   best_bid 0.74  best_ask 0.84     Down  best_bid 0.16  best_ask 0.26
copy 2   Up   best_bid 0.73  best_ask 0.84     Down  best_bid 0.16  best_ask 0.27
         ^ change rows (asset_id, hash, price, size, side) IDENTICAL in both
         ^ 0.74+0.26 = 1.00 and 0.73+0.27 = 1.00 -- each copy internally consistent
```

**The venue `timestamp` is not a unique event identity.** Two events sharing one
`timestamp` are separated by up to 113 ms of receive time and by up to 27 cents
of top-of-book. `_raw_message_key` for `price_change` is
`(event_type, timestamp, change-rows)` and **excludes `best_bid`/`best_ask`** —
the resulting state. So the key cannot separate two events that changed the same
levels, and the guard fires on the difference in the state they produced.

**This is the sixth instance of *the name is not the definition*.** The error
reads `duplicate identity has conflicting payload`. For 96.3 % of the population
there is no duplicate; there is a **key collision between distinct events**.

**One consequence beyond the crash, for BE (`f_r` counts, not state):** the
251,599 records collapsed as exact duplicates are an **upper bound** on true
re-deliveries. Where two distinct events in one millisecond changed the same
levels *and* produced the same top-of-book, the pair is byte-identical and is
silently collapsed — the state survives, the **event count does not**. At
0.095 % of records this is small, but it is a second, independent reason the
count layer is contaminated, on top of the micro-actor share already recorded in
`FLOW_MODEL_STATE.md` §5. Flagged for BE; no count is restated here.

### 3.5 What DA proposes, and does not do

**No rule is written and none is applied. `normalize_clob` still raises.**

**The dispatch asks which copy is authoritative. For B2 that question does not
have an answer, and DA reports that rather than picking one.** Both copies are
real, internally consistent, successive states of the same book. Choosing either
one deletes a top-of-book observation — and `best_bid`/`best_ask` on
`price_change` is precisely what the inherited rule says the whole corpus must
read book state from. Keeping the first would bias the tape toward stale tops;
keeping the last would drop intermediate states that a 113 ms separation says
were genuinely visible. **Neither is a selection between a good and a bad copy;
both are a deletion of data.**

For ratification, the two mechanisms want different treatments:

| | **B1 `book`** (19) | **B2 `price_change`** (499) |
|---|---|---|
| what it is | one state, two deliveries | **two states, one key** |
| authoritative copy | either — they agree on every shared field | **neither — both are real** |
| DA proposes | treat same-`hash` snapshots as one state; record the envelope-field difference as a coverage fact | **extend the identity key with the resulting top-of-book** so the two events stop colliding, and retain both |
| cost | none for state: the inherited rule already forbids reading state from `book` snapshots | true re-deliveries stay byte-identical and still dedup; nothing is retained that was not distinct |

- **Exclusion is available but looks unjustified by the data.** It would remove
  463 slugs across all five days and all seven coins — 6.6 % of the corpus,
  including days otherwise clean — to discard observations that the census says
  are valid. If the coordinator wants it anyway, R-ADMISS needs both arms and
  the excluded set reported beside the retained one; DA will measure it.
- **DA does not propose widening the digest to ignore differing fields.** That
  silences the case the guard exists for — a `bids`/`asks` disagreement under
  one venue `hash` — which the census shows has never occurred (0 of 19) and
  which we would then never learn about.
- Whether affected windows are excluded is a separate decision from which copy
  is authoritative. Both are the coordinator's.

---

## 4. Corrections to the dispatch's stated state

1. **`tier1/` no longer holds only `day=2026-08-20`.** Before this dispatch
   arrived, DA ran the batch under `--latest` while diagnosing A; it selected
   `2026-08-21` and built, for that day: `twap` for btc/eth/sol/xrp/doge,
   `windows`+`coverage`+`canary` for btc/eth/sol, `health`+`runs` for btc/eth —
   then aborted on **C** at sol. The two verification runs in §1.5a then added
   one new `health` artifact per coin for btc/eth/sol/xrp on 2026-08-20 (new
   validator code hash, new content-addressed path). All of it is the lane's
   normal immutable output; **nothing was deleted or overwritten**. But the
   partial day is there, and the coordinator's blast-radius line should be read
   with it. **Still true: no `full`-lane receipt, and no `batch` receipt of any
   lane, has ever committed.**
2. **The cause of the 4-coin day is C, not A** (§0).
3. **`MEASUREMENT_PLAN.md:33`** — "`DA-Normalize / DA-State / DA-Settlement` —
   none built" — is stale for the first two.
4. **`DA_INVENTORY_STATE_PLAN.md` §0** still asks the coordinator to rename a
   file that commit `f46379f` already renamed; all eleven references are
   consistent with the current name.
