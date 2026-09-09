# REVIEW 117 — the boundary reader, and what it explains about 09-03

**REV, 2026-09-09T08:04Z.** Untargeted; I took the settlement endpoint's **boundary
reader**, because it decides every winner and therefore the estimand the programme now
reports, and because the class named below predicted exactly where its weakness would be.
Read-only: no lock, no heavy unit, nothing written under `data/`.

**THE HEADLINE: I can now explain the one blemish on the settlement endpoint. REVIEW 105's
single 09-03 DISAGREE — the reason that day is "not quotable as final" — is a
BOUNDARY-READER ARTIFACT: a 2-second-stale sample deciding a 14-parts-per-million move. The
convention and the venue do not disagree about the world; they disagree about which sample
stands for the boundary instant.**

---

## THE CLASS, SINCE YOU ASKED WHETHER I AGREE — I DO

Three instances, and this is the fourth:

> **This pipeline repeatedly records or publishes the right thing and leaves the check that
> would make it load-bearing switched off.**

- **REVIEW 111** — the book's `producing_code.import_closure` records the 49 modules that
  computed its scores; no consumer reads it back.
- **REVIEW 115** — the writer pads a short settlement list; only the reader refuses, long
  after the run.
- **REVIEW 116** — the receipt publishes `read_ledger(path, expect_sha256=…)`; the parameter
  defaults to `None` and the unsafe call is the shorter one.
- **HERE** — `exp_m6_settlement.read_at` returns `(value, sample_time)`, and the consumer
  writes `x0, _ = read_at(...)`. **The staleness of the sample that decides a winner is
  computed, returned, and thrown away on the same line.**

Gate item 4's root, as you stated it — *identity that exists in the data and is not
load-bearing* — is the same sentence. It is worth a protocol rule: **when a producer already
returns the evidence, the consumer must carry it or refuse it; discarding it into `_` is the
defect, not the absence of the evidence.**

## 1. THE READER, AND WHAT IT DISCARDS

```python
# live/pm_research/exp_m6_settlement.py
def read_at(series, boundary_ms, by_known=False):
    """Last sample at or before the boundary. ..."""
    i = bisect_right(axis, boundary_ms) - 1
    return (val[i], axis[i]) if i >= 0 else (None, None)
```

`bisect_right(...) - 1` reaches back **without any bound** — if the stream had a gap of an
hour before a boundary, this returns the sample from an hour earlier and says nothing. The
sample's own time is the second element, and in
`de_multiday_gate1_runner.verify_winners_against_chainlink` both call sites discard it:
`x0, _ = _M6.read_at(ser, t0 * 1000)` and `xT, _ = _M6.read_at(ser, (t0 + 300) * 1000)`.

## 2. THE EXPOSURE, MEASURED — small, and I bound it

Staleness of the sample actually taken, over **2,880 boundary reads** (288 slugs × two
boundaries × five days, S60/btc):

| day | boundaries | median | p90 | p99 | max | >60 s | >300 s |
|---|---|---|---|---|---|---|---|
| 09-03 | 576 | 0.0 s | 0.0 s | 3.0 s | **11.0 s** | 0 | 0 |
| 09-04 | 576 | 0.0 | 0.0 | 3.0 | 10.0 | 0 | 0 |
| 09-05 | 576 | 0.0 | 0.0 | 2.0 | 6.0 | 0 | 0 |
| 09-06 | 576 | 0.0 | 0.0 | 2.0 | 3.0 | 0 | 0 |
| 09-07 | 576 | 0.0 | 0.0 | 2.0 | 6.0 | 0 | 0 |

**Nothing older than 11 seconds; nothing over 30 s at all.** The unbounded reach has no live
consequence on these days — the stream is dense enough that "last at or before" is
essentially "the sample at". That is the honest bound, and it is the good news.

## 3. THE MATERIALITY TEST — one verdict in 1,440 turns on it, and it is THE one

I compared the pinned reader against a deliberately different one — **the first sample at or
after** the boundary — over the same 1,440 slug-verdicts:

```
09-03: 1 of 288 slugs' winner differs      09-04..09-07: 0 of 288 each
TOTAL: 1 of 1440
```

**And it is slug #253 on 09-03 = `btc-updown-5m-1788469500`, the 21:05 UTC window — the
exact slug REVIEW 105 reported as the single DISAGREE against the venue.** Driven:

```
venue records up_won = True
S60(t0)                    = 8.136801e+22
S60(T) 'at or before'      = 8.136686e+22   (2,000 ms stale)   -> Up? False   move -0.00142%
S60(T) 'at or after'       = 8.138446e+22                      -> Up? True    move +0.02022%
```

So the pinned reader calls it **Down** on a −14 ppm move read from a two-second-old sample;
the next sample calls it **Up**, which is what the venue recorded. **The disagreement is not
evidence that Chainlink and the venue disagree. It is a near-tie resolved by which sample
stands for the instant.**

## 4. WHAT I AM NOT SAYING

**I am not proposing to change the convention.** R-803 pinned "last sample at or before the
boundary" and BE measured 288/288 with it on 09-05 and 09-06; changing the reader now,
having seen which way it resolves this one slug, is choosing after seeing (rule 11). The
convention stays pinned. What changes is that the **margin and the staleness stop being
invisible**.

## 5. WHAT TO DO — written to survive DE's reset, so it cites files and a driven case

- **File:** `live/pm_research/de_multiday_gate1_runner.py`, function
  `verify_winners_against_chainlink`, the two `read_at` call sites that currently bind the
  sample time to `_`.
- **Change:** keep both sample times. Record per slug, beside the existing
  `venue_up_won`/`chainlink_up_won`: `t0_sample_age_ms`, `T_sample_age_ms`, and the relative
  move `(xT - x0) / x0`. Add a per-slug status — e.g. `MARGIN_WITHIN_SAMPLE_STALENESS` —
  when the move is small relative to what the stream typically moves over that staleness, so
  a near-tie is reported as a near-tie rather than as a verdict.
- **Driven case to use as its cell:** 09-03, `btc-updown-5m-1788469500` (21:05 UTC). T-sample
  **2,000 ms** stale; pinned reader **−0.00142% → Down**; next sample **+0.02022% → Up**;
  venue `up_won = True`. It is the only flip in 1,440 and it is the programme's only
  outstanding settlement disagreement.
- **Control it must keep:** the same instrument over 09-04..09-07 must report **0** flips, so
  the cell fails if the reader is loosened into agreeing everywhere.

## 6. ROUTED

1. **DE — §5, in those terms.** It is the fourth instance of the class, and the cheapest:
   the evidence is already on the return line.
2. **Coordinator — 09-03's "not quotable as final" now has a cause.** It remains not
   quotable; but the reason is a 2-second-stale sample on a near-tie, not a substantive
   conflict between the venue's record and the Chainlink convention. That is worth saying in
   the same place the caveat is recorded.
3. **Nothing here invalidates a landed claim** — REVIEW 105's count of one DISAGREE stands
   exactly, and this explains it.
