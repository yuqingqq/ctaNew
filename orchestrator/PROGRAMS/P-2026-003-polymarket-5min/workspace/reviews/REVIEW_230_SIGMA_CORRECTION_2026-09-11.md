# REVIEW 230 — REVIEW 229 §3 item 1 is WITHDRAWN: I reported a seat's in-flight module as prior art. BE's table is correct, confirmed at the code. The other four findings are untouched, with the evidence.

**REV, 2026-09-11T18:31Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

## 1. THE ERROR, AND ITS TIMING

```
my tree read (the greps that found the file)   ~18:22-18:26Z
REVIEW 229 filed                                18:27:48Z
be_sigma_30m.py ADDED by 8fe2a2e                18:28:38Z   "BE 194: the 30-minute sigma producer"
```

**The file was uncommitted in the working tree while I read it, and was committed 50 seconds
after I filed.** `git log --diff-filter=A` shows `8fe2a2e` as its *first and only* commit. I
reported **BE's step-2 work, landing against the plan I was auditing, as an existing asset BE
should defer to.** REVIEW 229's header item 2 and §2 row 1 and §5.2 are withdrawn.

**And the provenance was only half of it.** I inferred "a producer exists" from a **file name
and a signature** — `sigma_30m(ticks, decision_recv_ns)` looked point-in-time, so I called the
plan's missing item overstated without reading what it computes or where it came from. Two
checks, either of which would have caught it:

- `git log -1 --format=%cI -- <file>` — one line, and it would have read 18:28:38Z;
- read the **estimand**, not the signature.

This is the third time I have made the same shape of mistake and it is worth naming as one
class: **REVIEW 120** (searched one spelling, stopped at three), **REVIEW 162** (ranked growing
inputs by mtime instead of by the property), and now this. Each time I substituted a cheap
proxy — a name, a timestamp, a signature — for the operation that defines membership. The
standing rule I wrote for other people's enumerations applies to my own inventories:
**decide membership by the property, and for "does this already exist" the property has two
parts — what it computes, and when it landed.**

## 2. BE'S TABLE, CONFIRMED INDEPENDENTLY AT THE CODE

I read both sides rather than take the table.

**Route-A / `sigma_kernels.py` (5 commits, first 2026-08-20 — genuine prior art):**

```
HORIZON_GRID = (30, 60, 120, 180, 240, 270)              sigma_kernels.py:38
HORIZONS     = (30, 60, 120, 180, 240, 270)              exp_sigma_route_a.py:36
":108  Route A regresses observed S_fast/S_slow on observed x_T"
":213  '… not cross-fitted; in-sample residual …'"
```

A **regression residual variance across a six-horizon grid** — exactly the r ∈ {30,60,120,
180,240,270}s BE names.

**`be_sigma_30m.py` (BE 194, today):**

```
Binance USDM bookTicker MIDPOINT log returns, PER-SQUARE-ROOT-SECOND, no annualisation
WINDOW_S = 1800   N_EXPECTED_RETURNS = 1800   "1801 grid points -> 1800 returns"
grid_end_sec = decision_sec - 1        <- SHIFTED BY ONE COMPLETE OBSERVATION
grid rule: the latest midpoint whose LOCAL-KNOWLEDGE time (recv_ns) is at or before the
           grid instant; no interpolation; no later tick may fill it
sigma_local_knowledge_ns <= decision_recv_ns     "structural, not checked"
ERA_FLOOR_NS = 1787579334881534478  (2026-08-24 13:48:54Z, hf_ws_v2) -> PRE_ERA refusal
```

**Different estimand, different input, different window, different population.** One is a
regression residual variance over a horizon grid; the other is a realized volatility of
one-second log returns over a 1,800-second trailing window with a local-knowledge grid rule and
an era floor. **BE's field-by-field table is correct on every element I checked. There is
nothing to wire, and the two-validators warning does not apply** — two estimators of *different*
estimands are not two validators of one.

*(One thing worth keeping from the wreckage: the era floor BE uses is the same
`1787579334881534478` boundary the programme's own reliability rules name as the start of
sub-second-reliable Binance data. That is the right floor and it is applied as a refusal, not a
filter.)*

## 3. DID THE SAME ERROR TOUCH THE OTHER FOUR? — NO, AND HERE IS THE EVIDENCE

I checked provenance for **every** file REVIEW 229 cited, not only the one that was wrong:

| finding | source | provenance | touched? |
|---|---|---|---|
| the two-regime estimand at `T-60` | `da_fair_price_identity.py` | **8 commits, first 2026-08-28, last 2026-08-29** | **no** — two weeks old |
| "the exporter explicitly refuses" is not what the code does | `be_trajectory_export.py` | **4 commits, last 2026-08-28** | **no** — two weeks old |
| the 187 ppm cross-venue basis | `data/pm_5min/prices/*` | collector output, not seat work; and the payload's own `window_s: 60` corroborates the stream claim independently of any module | **no** |
| §8's 8-vs-10 tolerance | the plan text | arithmetic; no code read at all | **no** |

**All four stand as filed.** The error was confined to the one file that landed in the session,
and the other citations are from a fortnight ago.

*One honest qualification on the 187 ppm, unrelated to this error and already in REVIEW 229's
scope: it is one hour, one coin, one day. It establishes that the raw stream is not the TWAP's
underlying — which is the load-bearing claim — and it does not establish that 187 ppm is the
basis's typical size.*

## 4. WHAT I WILL DO DIFFERENTLY

For any future "what already exists" inventory, before a file is called prior art:

1. `git log --diff-filter=A --format=%cI -- <file>` — when it was **added**, not last touched;
2. the author and the commit subject — a seat's own module names its seat;
3. the **estimand**, read at the code, against the thing it is supposed to already provide;
4. and the same for anything I find in the working tree that is **untracked** — which
   `be_sigma_30m.py` was when I read it, and which I did not notice because I never asked.

**Step 4 is the one that matters here:** I had run `git status` in this repo many times this
session and would have seen the file listed as `??`. I did not look, because I was reading a
directory listing rather than a tree state.

## ROUTED

1. **BE — REVIEW 229's §5.2 is withdrawn**; `be_sigma_30m.py` is not duplicating anything and
   the Route-A corpus is a different estimand. Nothing to defer to.
2. **Coordinator — the rule is recorded** (§1, §4) and I would put it in SEAT_PROTOCOL in those
   terms: an inventory of existing assets must exclude work landed in the same session, and
   membership is decided by the estimand plus the add-date, never by a name or a signature.
3. **R-918 — the other four stand** (§3), with provenance checked for each rather than asserted.
