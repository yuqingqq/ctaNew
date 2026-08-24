# Does Binance lead PM flow? · protocol, frozen BEFORE the run

**Status: FROZEN per Ruling R-51, 2026-08-23. APPEND-ONLY from this point
(R-28); R-38 clause (d) applies — an amendment buys an obligation to re-measure,
never a verdict.** Re-pointed to the harder R-49 bar (500/1000 ms rungs) before
the freeze. **STILL BLOCKED ON DA's knowledge-time alignment (§1): the freeze
makes the bar binding, it does not discharge the precondition.** No measurement
has been run. Written by BE under **R-48**.

*Freeze applied late by BE — R-51 froze four protocols and BE left two files
reading DRAFT. Caught by the coordinator's R-36 check, not by BE's report. The
principle is worth recording: **a protocol's status is what its FILE says, not
what a ledger says about it**, so the freeze was not in force on either file
until this edit.* BE owns whether the lead exists and how
long it is; **DE owns the policy bound that consumes the answer.**

---

## 0. The bar is not "does Binance lead" — it is stated in ww_v1's own statistic

R-48 frames the comparison against ww_v1's median warning of ~0.16 s. **Verified
on disk** (`warning_window_v1_dayseries.json`, `join` arm, drift-bearing fills):

| day | coin | n | unwarned | p10 | p25 | **p50** | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 08-20 | btc | 10,387 | 11.9 % | 0.004 | 0.044 | **0.161** | 0.473 | 1.245 |
| 08-21 | btc | 10,994 | 11.0 % | 0.006 | 0.037 | **0.134** | 0.356 | 0.901 |
| 08-20 | eth | 2,003 | 12.3 % | 0.010 | 0.058 | **0.153** | 0.309 | 1.291 |
| 08-21 | eth | 2,003 | 11.2 % | 0.013 | 0.057 | **0.157** | 0.275 | 0.890 |

**Reading this correctly changes the question.** The book channels are not
uniformly useless — they are useless *at the median*, and the distribution is
violently right-skewed. `p50 ≈ 0.16 s` sits **below** the 250 ms knowledge lag,
so **more than half of fills are already unwarned before any cancel latency is
added** — that is why those channels failed. But `p75 ≈ 0.47 s` and
`p90 ≈ 1.25 s` already clear the lag comfortably.

**So a median comparison would be the wrong instrument.** The binding quantity is
the one `ww_v1` already uses:

```
R(tau) = share of DRIFT-BEARING fills whose warning W exceeds  lag + tau
         lag = 250 ms knowledge lag, frozen
```

measured for the book channels at **R(250 ms) = 0.153 (btc)**, **0.143 (eth)**.

### 0.1 RE-POINTED UNDER R-49 — the threshold moved from 160 ms to 750 ms

OPS's actuation bound fires: achievable `τ` is **351 ms at the most optimistic
reading**, typical **455 ms**, btc **p90 750 ms**, btc **p99 5,397 ms**. **The
250 ms knowledge-lag floor alone exceeds the 160 ms median warning before any
decide or ack term** — set ack to zero and it still fails.

So the operative rungs are **500 ms and 1000 ms, not 250 ms**, and the headline
statistic is the **share of adverse fills whose warning exceeds ~750 ms** (btc
p90 actuation). BE verified the consequence on the frozen receipt:

| coin | R(250ms) | R(500ms) | R(1000ms) | bar | shortfall @500 / @1000 |
|---|---:|---:|---:|---:|---|
| btc | 0.1527 | 0.1036 | 0.0570 | 0.309 | **3.0× / 5.4×** |
| eth | 0.1429 | 0.1090 | 0.0737 | 0.494 | **4.5× / 6.7×** |

**The hypothesis is now much harder and must be stated as such.** A bot reacting
to Binance clears in tens of milliseconds. So the question is no longer "does
Binance lead" — it is **whether a material share of PM adverse flow comes from
SLOW reactors**, since only slow reactors can leave us 750 ms of usable warning.

### 0.15 WHAT THIS MEASURES THAT WW_EBX DOES NOT — the instrument-free bound

Measured by BE 2026-08-23, because the two protocols read **different Binance
sources** and the difference is ~50,000× at the median:

| source | used by | cadence (btc) |
|---|---|---|
| deployed `crypto_prices` relay | DE's `WW_EBX` | **p50 999 ms** — exactly 1 Hz per symbol, all six coins |
| raw HF `bookTicker` | **this protocol** | **p50 0.02 ms**, p90 1.74 ms, 89 files across the era days |

*(BE's first pooled read of the relay suggested 37 ms and was a symbol-mixing
artefact; withdrawn. DE's "1 Hz" characterisation is correct.)*

**The consequence is arithmetic.** A 1 Hz sampler detects a move at a phase delay
uniform on `[0, 1 s]`, so it **understates warning by ~500 ms in expectation**.
At the R-49 rung of **τ = 500 ms the instrument's own sampling noise is the same
size as the rung**; at τ = 1000 ms it can consume the whole budget.

**Therefore the two results are not substitutes:**

- `WW_EBX` measures **the deployed instrument's** realisation. A negative there
  confounds *the market gives no warning* with *our 1 Hz relay cannot see the
  warning that exists*.
- **This protocol measures the market**, effectively instrument-free, and so
  bounds what **any** Binance channel could deliver.

**SCOPE CORRECTION, BE's own, 2026-08-23.** This protocol measures the **raw HF
tape**, which is still **our collector** — behind the same collection boundary as
the relay. It bounds what a **well-instrumented consumer** of Binance data could
see. It does **NOT** bound what a **direct exchange member** could see, and it
must not be cited as doing so. R-54's open item — *direct-feed non-relay
latency* — stays open either way, and is an infrastructure question rather than a
research one.

**Within that scope it is still the decisive direction:** if the raw tape shows under 750 ms of
warning on a material share of adverse fills, **no instrument improvement can
rescue the channel** — that closes the last open question more strongly than
`WW_EBX` alone can.

### 0.2 THE PLACEBO HAS ALREADY FIRED ON THE BOOK CHANNELS — and its direction is the signature

Surfaced by BE from the frozen `ww_v1` receipt; not in R-49. The receipt carries
`R_shuffled` beside `R`, and **the real tape gives LESS warning than a shuffled
one at every rung, on both verdict coins, on both days**:

| day | coin | τ=250 ms | τ=500 ms | τ=1000 ms |
|---|---|---|---|---|
| 08-20 | btc | 0.153 vs **0.209** | 0.104 vs **0.144** | 0.057 vs **0.080** |
| 08-21 | btc | 0.120 vs **0.151** | 0.072 vs **0.100** | 0.036 vs **0.050** |
| 08-20 | eth | 0.143 vs **0.174** | 0.109 vs **0.130** | 0.074 vs **0.093** |
| 08-21 | eth | 0.109 vs **0.152** | 0.067 vs **0.115** | 0.029 vs **0.067** |

Real `<` shuffled means the true triggers sit **CLOSER to fills than chance
would put them** — warnings are compressed toward zero relative to random. That
is the same finding as *"we do not arrive ahead of the adverse flow, we arrive
WITH it"*, already measured and sitting on disk.

**This sets the placebo expectation for Binance, and sharpens §2.** It is not
enough that the placebo be *indistinguishable* from baseline. **If the real
Binance tape gives less warning than its shuffled version, that is the
co-arrival signature and the channel is refuted**, however large its apparent
lead.

**THE BAR, therefore, is expressed in the same statistic so the two are
comparable by construction, not by narrative:**

```
does Binance raise R(tau) above the book channels' R(tau), at the same tau,
on the same fills, under the same lag?
```

A median that moves while `R(τ)` does not is **not** an improvement — it is mass
shifting inside the already-warned region, which buys nothing.

---

## 1. HARD PRECONDITION — this does not run until DA clears the alignment

**DA owns the alignment and has been told to stop if it cannot be made
knowledge-time honest. BE runs nothing until that clears, and records the block
rather than working around it.**

The reason is not procedural. **A look-ahead alignment MANUFACTURES the lead
being tested for, and it will look exactly like a discovery.** If PM events carry
`recv_ns` (our parse time) and Binance events carry an exchange timestamp, the
apparent lead is the *difference in collection latency* and has nothing to do
with markets. That failure mode produces a clean, tight, plausible number.

**Required from DA before any estimate:** both tapes on **knowledge time**, the
alignment's own knowledge-time error stated, and a declaration that the PM side
is not being compared against a Binance timestamp the collector could not have
known at that instant.

If the alignment cannot be made honest, the protocol's output is
**`VOID(ALIGNMENT)`** — not a weaker estimate.

### 1.1 ASSESSMENT, 2026-08-23 — §1 CLEARED ON THE CLOCK, with one named residual

BE assessed `da_hf_pm_alignment.py` against §1's three requirements and verified
the load-bearing claim in the source rather than in the docstring.

| §1 requirement | status | evidence |
|---|---|---|
| both tapes on knowledge time | **MET** | both collectors stamp `recv_ns = time.time_ns()` **at parse time on the same host** — `collect_pm.py:408` and `collect_hf.py`'s `self._on_msg(m, time.time_ns())`, verified by BE. One wall clock, so receipt times compare directly: no offset estimation, no exchange-clock arithmetic, no cross-venue skew model |
| knowledge-time error stated | **MET** | Binance `E` runs **~87 ms behind local receipt** and can arrive out of `E`-order; DA **refuses** `E`/`T` alignment for exactly that reason |
| no forward read | **MET** | `state_at(t)` returns the last row with `recv_ns <= t`, never the next, and returns `Unavailable` rather than reaching back past coverage start |

**DA made the refusal DETECTED rather than declared**, which is why BE accepts
it: a crafted tape where `E`-order and `recv`-order disagree returns a
*different* answer from an `E`-aligner, so a regression to exchange time **cannot
pass quietly**. 26 selftests pass. Same pattern as `assert_directional` — the
check makes the code reveal its alignment rather than asking it.

**THE NAMED RESIDUAL, and it is a coverage problem rather than a clock problem.**
Polymarket writes a structured gap ledger (188 `gap_closed` records with cause);
**Binance HF writes NO gap ledger at all**. HF gaps are therefore *inferred* from
tape silence and are labelled an inference.

**Direction of the resulting bias, stated before the run:** an unlogged HF gap
means a **stale** Binance state is read as current. Stale information cannot warn
about a move that has not yet reached it, so the error **suppresses** apparent
lead — it biases **against** the hypothesis. That is the safe direction for the
one lever everyone wants to be true, and it is why BE proceeds rather than
holding. But the **HF-gap-inferred share must be reported beside every `R(τ)`**,
because a coverage figure that is an inference may not be presented as a
measurement.

---

## 2. THE PLACEBO RUNS FIRST, AND IT GATES

R-48's second caution is that **everyone wants this to be true, including the
coordinator** — the one lever with a route back from the negative. Noting that is
not a control. This is:

**The placebo is computed and recorded BEFORE the real estimate is computed.**

- If the placebo is run afterwards, it checks your luck.
- If it is run first and gates, it is a falsifier.

```
placebo        circular-shift the Binance tape by a lag long enough to destroy
               contemporaneity while preserving its marginal distribution and
               autocorrelation  (the idiom this corpus already uses for A1
               independence and the B3 book-state control)
requirement    the placebo R(tau) must be INDISTINGUISHABLE FROM THE BOOK-ONLY
               baseline. If the shifted tape "predicts" PM flow, the estimator
               is measuring shared structure, not lead.
on failure     VOID(ESTIMATOR). The real estimate is not computed, and no
               number is reported -- because an estimator that passes noise
               will pass the real tape too.
```

---

## 3. Four ways this estimator manufactures a lead, and the control for each

| # | failure mode | control |
|---|---|---|
| 1 | **Alignment latency** read as lead | §1, plus the **reverse-direction test** below |
| 2 | **PM's own clustering.** Hawkes clustering runs **75–352 ms**, so "the next PM event after a Binance event" is partly PM's own inter-arrival distribution, not a response | condition on PM inter-arrival; compare against a PM-only baseline hazard rather than against zero |
| 3 | **Common driver.** Both tapes respond to the same macro impulse with no lead between them | the cross-correlation must be **ASYMMETRIC** about zero — a common driver is symmetric, and alignment error shifts the whole function without changing its shape |
| 4 | **Selection on the outcome.** Choosing "large" PM events and looking backward for Binance events conditions on the thing being predicted | events are selected on the **Binance** side and PM response measured forward; never the reverse |

### 3.1 The reverse-direction test — cheap and decisive

Run the identical estimator with the tapes **swapped**: does PM lead Binance?

- **Genuine lead** ⇒ asymmetric. Binance→PM shows a lead; PM→Binance does not.
- **Alignment error** ⇒ **symmetric in the wrong way**: each appears to lead the
  other, or the magnitudes mirror. That is the signature, and it is not subtle
  once looked for.

**Both directions are reported.** A one-directional report is not evidence of
asymmetry; it is the absence of the test.

---

## 4. What is reported — the DISTRIBUTION, never a point lead

R-48 is explicit and it is the right instruction: *"a mean lead of 400 ms with
most mass under 100 ms is a different world from a tight 400 ms, and only the
second is actionable."*

```
report   the FULL quantile ladder of the lead, matching ww_v1's grid:
         p10 p25 p50 p75 p90, plus the unwarned share
         AND R(tau) at the frozen ladder {250, 500, 750, 1000} ms -- OPERATIVE
             rungs are 500 and 1000 (R-49); 750 is the btc p90 actuation
         AND the incremental dR(tau) over the book-only baseline, paired on
             the SAME fills
per      coin (btc, eth = verdict coins; others descriptive), per day,
         never pooled across days; day-clustered where clusters permit
arms     R-DUAL: with and without the 0.02 micro class
```

**A point lead may not appear in the summary without its dispersion.** A
mean or median quoted alone is a reporting defect, not a shorthand.

---

## 5. The bar — FROZEN before the receipt exists

Let `dR(τ)` be the paired increment in `R(τ)` from adding the Binance channel to
the book channels, on the same drift-bearing fills, with a day-clustered
interval.

```
LEAD_MATERIAL      dR(750ms) CI lower bound > 0.10 on BOTH verdict coins
                   RE-POINTED under R-49. The book baseline at the operative
                   rungs is R(500ms)=0.104/0.109 and R(1000ms)=0.057/0.074,
                   against bars of 0.309/0.494 -- a 3.0x-6.7x shortfall. A
                   channel that does not move R at 750ms does not reach the
                   actuation window at all.
LEAD_MARGINAL      CI excludes zero but the lower bound is <= 0.10
NO_LEAD            CI includes zero
VOID(ALIGNMENT)    section 1 not cleared
VOID(ESTIMATOR)    section 2 placebo failed, or section 3.1 shows symmetry
```

**Both directions are named and the rule is symmetric**, so
`ev_gates.assert_directional` can be run against it — and will be, as a
registration check rather than a claim.

### 5.1 What it takes to FIRE — a-priori, before the run

Per R-14 Amendment 2, a gate declares what makes it fire, not only what makes it
fail. **The MDE on `dR(250 ms)` at the available fill count and day count is
computed and published BEFORE the estimate.** If the MDE exceeds 0.10, the
honest output is **`INSUFFICIENT_POWER`**, reported as such and never as
`NO_LEAD`.

### 5.2 The dispersion condition — the median may not do the work

`LEAD_MATERIAL` additionally requires that the gain is **not concentrated in the
already-warned tail**:

```
the increment dR must hold at tau = 500ms AND at tau = 1000ms
```

A channel that only helps fills which already had 0.5 s of warning has not
attacked the problem — the 11–12 % **unwarned** share and the sub-100 ms mass are
where the losses live.

---

### 5.3 WARNING AND ADVERSENESS ARE NOT INDEPENDENT — the selection trap

The bar is a share of **adverse** fills, and the two quantities are almost
certainly correlated in the unhelpful direction: **the most adverse fills are the
most likely to come from the FASTEST reactors**, who leave the least warning. So
conditioning on `warning > 750 ms` may select precisely the fills that were least
adverse — the ones that did not need cancelling.

**A channel that warns only about the least-adverse fills is worthless even if
`R(750 ms)` is high.** Mandatory, therefore:

- report the **joint** distribution of warning and post-fill drift, not the
  marginal of warning;
- report **drift-weighted** `R(τ)` beside the unweighted share — the fraction of
  *total adverse drift* that is warned, not the fraction of *fills*;
- `LEAD_MATERIAL` requires the **drift-weighted** increment to clear the bar too.
  If the unweighted share moves and the drift-weighted share does not, the
  channel is warning about the wrong fills and the verdict is `NO_LEAD`.

## 6. What this protocol will NOT do

- Not run before DA's alignment clears. `VOID(ALIGNMENT)` is the output, not a
  provisional number.
- Not report a point lead without its dispersion.
- Not compute the real estimate before the placebo has been recorded.
- Not select `τ`, coin or day after seeing results — grids frozen above.
- Not claim a cancellation policy. **BE reports whether the lead exists and how
  long it is; the policy bound is DE's.**
- Not pool across collector eras or sampling rules
  (`flow_intensity.assert_poolable`).

---

## 7. Deliverables

1. `BINANCE_LEAD_RESULTS.md` + a receipt carrying the placebo outcome **first**,
   both directions of §3.1, the full quantile ladder, `dR(τ)` per coin per day,
   the MDE, `sampling_rule`, and `days_sampled`.
2. Code with a selftest including the placebo gate, the reverse-direction test,
   and an `assert_directional` check on the §5 bar.
3. If blocked: a one-line record that the block is DA's alignment, with no
   provisional number attached.
