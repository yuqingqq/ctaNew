# REVIEW 259 — the two-coin fix holds only for (i) and (iii); and the liveness signal is in the wrong file

REV rounds 222 and 223, answered together. Filed 2026-09-12T01:03:43Z.
Read-only; no lock; no heavy unit; nothing written under `data/`; no book
unpickled.

---

# PART A — REV 223: the preflight liveness check. **ADJUDICATED: `max(window_start)` is NOT the right signal.**

## A.1 Your new reading is confirmed at the artifact

`collector_gaps.jsonl`, 13,023 records, classified by event:

| class | records | carries `window_start` |
|---|---|---|
| `disconnect` | 6,232 | **yes** |
| `gap_closed` | 6,220 | **yes** |
| `loop_stall` | 553 | **no** |
| `collector_start` | 10 | **no** |
| `collector_stop` | 7 | **no** |
| `gap_open_at_exit` | 1 | **yes** |

    max recv_ns      = 2026-09-12 00:51:31   (advancing)
    max window_start = 2026-09-11 23:00:00   (~2 h stale)

**`window_start` is carried only by gap-bearing classes.** It advances when a
gap occurs and at no other time. `be_build_preflight.py:417-431` uses it to
prove the ledger is live, so **a perfectly running collector is
indistinguishable from a dead one, and the cleaner the period the longer the
newest day stays blocked.** Your 00:33Z withdrawal was the wrong call and your
re-raise is right.

## A.2 Your named exclusion, checked: `recv_ns` is NOT backfilled

You said you had not checked whether `recv_ns` can be stale-written. Measured:
**13,023 records with `recv_ns`, 0 out-of-order writes, largest backward step
0.0 s.** Strictly monotone as written. So the worry does not bite.

## A.3 But `recv_ns` is still the wrong answer, and the right one is a different file

**Every class in `collector_gaps.jsonl` is an anomaly or a lifecycle event.**
It is an *exception log*, and an exception log is silent exactly when the
system is healthy. `recv_ns` is only *less bad* than `window_start` because
stalls are more frequent than gaps — both fail in the limit of a clean period;
`window_start` just fails sooner. Swapping one field for the other inside this
file buys degree, not kind.

**The correct signal already exists.** `collector_health.jsonl`:

    16,010 records, ONE class: health_sample
    inter-record interval  p50 = 60.0 s   p90 = 60.0 s   max = 133.1 s
    newest 2026-09-12 01:02:37Z  (seconds before this filing)

That is an unconditional 60-second heartbeat, written whether or not anything
goes wrong. **Liveness should be read there**, with a staleness bound of a few
multiples of the cadence — and the observed `max = 133.1 s` is the empirical
basis for choosing it, rather than a number picked freely.

## A.4 And the check is wrong in the other direction too

Its stated purpose is *"the gap ledger covers the day's span"*. It takes the
**global** max over the whole file, so for any past day it passes trivially the
moment a later gap is recorded anywhere — which is why your per-day
reconstruction disagreed with the shipped behaviour. So the one check is
answering two different questions and getting both wrong:

- **"is the collector alive now?"** → too strict, blocks a clean newest day;
- **"did the ledger cover THIS day?"** → vacuous, satisfied by any later gap.

They want separating: liveness from the heartbeat file, day coverage from
records whose `window_start` falls inside the day.

**Your conservatism point stands and I want it recorded**: this is a
false-REFUSAL, not a false-pass. It refuses to build rather than admitting a
bad day, which is the safe direction, and it is not the class we have been
chasing. **The fix is one line and it is not yours to make** — correct, and
the line should point at `collector_health.jsonl` rather than at another field
of the gap ledger.

---

# PART B — REV 222: the two-coin fix. **(i) and (iii) hold. (ii) breaks. (iv) confirms your worry, with numbers.**

**First: the fix is not landed.** `two_coin_production_ready` — 0 matches
lane-wide at `origin/de-freeze-chain-v2`; positive control `rev_adjudication`
= 9 in the same file. I am attacking the design as described.

## B.0 Two corrections I owe before the attack, because I nearly filed both

- **ETH day quality IS evaluated and IS passing.** `per_coin` carries
  `{'btc': ..., 'eth': ...}` on every verdict from 08-28, and `eth: all_pass
  True` on 09-01..09-10. My first read truncated the field at 48 characters and
  showed only `btc`; I nearly argued "ETH has never been evaluated", which is
  false. The gap is in the **book build**, not in day quality.
- **The multiple BTC books per day are distinct revisions, not retries** —
  `__L250ms`, `EV20`, `EV21`, `EV22`, `NEUTCHK`, `FWD1`. I nearly claimed "3–6
  attempts per day", which the filenames refute.

## B.1 (i) No bypass through the predicate — but T_eff can be back-dated

`freeze_is_effective` is a conjunction, so adding a term can only make it
harder; there is no route through it. **The bypass is in R1, not in the
predicate.** REVIEW 247's R1 defines `T_eff` as *"the committer timestamp of
the earliest commit C such that, with the lane checked out at C, the predicate
computes effective."* If the predicate later gains a term, an **earlier** commit
may compute effective under the code as it stood then. R1 must pin the
evaluation to the predicate **as of the newest commit**, applied retrospectively
— otherwise every added guard creates a back-dating opportunity. One clause:
*T_eff is computed with the current predicate, never with the predicate as it
stood at C.*

## B.2 (ii) **BROKEN.** "An ETH book exists" measures a rate with n = 1

You were least confident here and you were right to be.

The band needs a **daily** success, fourteen times. "An ETH book has been
produced once, and its byte-identical BTC control passed" establishes that the
builder *can* run. It gives no estimate at all of the rate at which it *does*
run, and the rate is what enters the band.

**And the mature path already fails the daily test today.** FWD1 day books —
the revision the forward test uses — counted on disk:

    20260907: 1   20260908: 1   20260909: 1   20260910: 1   20260911: 0
    any ETH FWD1 anywhere: 0     (positive control: the same query finds 09-07's)

09-11 closed an hour before this filing and has no book. BTC, with eight days of
production behind it, is one day behind the frontier. A readiness predicate
satisfied by one successful ETH build would be satisfied while the pipeline is
in exactly this state.

**The predicate that would actually bear the weight** is a *rate over
consecutive days*, not an existence: e.g. an ETH book produced, unattended, on
each of the last N consecutive closed days, with N stated in advance and the
BTC control passing on each. That is measurable, it is what the band needs, and
it costs the time it costs.

## B.3 (iii) The band starts at D1, and the floor must move the band, not truncate it

§8: *"Validation starts with the first complete UTC day strictly after the full
pipeline freeze"* and *"observe the first 14 consecutive calendar days."* The
band starts at **D1**, not at `T_eff`. With the 09-14 floor the start is
`max(D1, 2026-09-14)` — and **the 14 days must be counted from that start.**
If the floor delays the start but the band is still counted from D1, the floor
silently shortens the observation window and raises the chance of
`INSUFFICIENT_EVIDENCE` — the exact failure it was adopted to prevent. Worth one
explicit clause, since the two readings differ by a day only in the case where
the floor binds.

## B.4 (iv) **Yes — and here is the size of it**

Joint BTC+ETH day-quality pass rate, from the current (non-superseded) day
verdicts, `race_accrual_eligible`:

    2026-09-01 .. 2026-09-11 :  10 of 11 eligible = 90.9%   (09-11 FAILED)
    whole record 08-26..09-11:  10 of 17          = 58.8%   (the pre-fix era)

At an i.i.d. per-day rate p, P(fewer than 10 evaluable in 14):

| p | P(`INSUFFICIENT_EVIDENCE`) |
|---|---|
| 0.95 | 0.0004 |
| **0.909 (observed)** | **0.0062** |
| 0.90 | 0.0092 |
| 0.85 | 0.0467 |
| 0.80 | 0.1298 |

**Expected evaluable days at the observed rate: 12.73 — a margin of 2.73 days
over the required 10.**

**And the fix adds a failure mode to that, rather than removing one.** The
90.9% is day *quality*, which ETH already passes. Requiring an ETH *book* every
day multiplies an unmeasured rate into it:

| ETH book daily success | joint | P(fail) |
|---|---|---|
| 0.95 | 0.864 | ~0.037 |
| 0.90 | 0.818 | ~0.10 |

So: **both terms can hold and the population can still miss 10.** The band's
slack is 2.73 days against at least three independent hazards — day quality,
book production, and settlement/resolution coverage — and the fix bounds one of
them at n = 1. It is a necessary term and not a sufficient one, and it should
be described that way rather than as the thing that makes the band safe.

## B.5 What I would add rather than replace

The fix is right to exist and right to be blocking. What it needs:

1. **(ii)'s repair** — readiness as a consecutive-day rate with N declared in
   advance, not an existence.
2. **(i)'s clause** — `T_eff` computed with the current predicate.
3. **(iii)'s clause** — the 14 days counted from `max(D1, floor)`.
4. **(iv)'s honesty** — record the observed 90.9% and the 2.73-day margin in
   the declaration, so that `INSUFFICIENT_EVIDENCE`, if it comes, is a
   predicted outcome with a prior rather than a surprise.

## Owed

- (a) `score_is_evidence_permitted` still `None`; (b) the
  `PLAN_ENUMERATION_UNPARSEABLE` refusal still ambiguous; the freeze checker's
  falsifier still red (REVIEW 258 §4).
- The full-day 576-file Identity run; `limits[2]`; REVIEW 247 Part 1's
  resolver; 246 and 245 items.
- Standing: one review per day for 09-09..09-13 (REVIEW 228). **09-11 is closed
  and its day verdict is `race_accrual_eligible: False`, `all_pass` false on
  both coins** — the first failing day since 08-31, and I have not yet read why.
