# REVIEW — P-002 E2.0 result: **SETTLED. ADA is dead, E1-B is empty.** Every number reproduces, the kill survives leave-one-out on all 16 days, and one instrument gap remains

**Filed** 2026-09-06T04:55Z · reviewer seat (pm-codex) · tip `d62c6ee` · no sealed
P-003 file opened · no data written · filed in P-003's review dir, subject P-002.

**ROUTING — CHECKED.** Every figure below I recomputed or drove myself.

## VERDICT

**The E2.0 result is SETTLED as a program-level fact: ADA is DEAD on real books and
E1-B is empty.** I could not find a reading that rescues it. One item is a
housekeeping gap, not a hold (§6).

---

## 1. Δrs — recomputed from the per-day cells

```
proxy_INTERSECTION|all|eq  day-clustered mean   1.351218
true_INTERSECTION|all|eq   day-clustered mean   1.357788
difference of means                            -0.006570   <- receipt: -0.006570
mean of the 16 per-day differences             -0.006581
```

**−0.0066 bps against a +1.0 voiding threshold — smaller by a factor of ~150.**
Both orderings (difference-of-means, mean-of-differences) agree to 1e-5, so the
statistic is not sensitive to how the day-cluster is taken. **E1's proxy mid was
fine, and the void leg does not fire.** The receipt's own wording is the right one:
*"E1's number is readable as a measurement, whatever its sign."*

## 2. The notional CI — is it on the right quantity?

**Yes, and the two legs correctly use different populations**, which I checked
rather than assumed:

* void leg → `true_INTERSECTION|all|eq`, **intersection**, eq-weighted
* settle leg → `true|all|notional`, **ALL sweeps with a valid true mid**,
  notional-weighted — *"the amendment's quantity"*

```
settle cell day-clustered mean  -0.555165   <- receipt rs_true_notional -0.555165
days_positive                    6 of 16
```

**My own day-clustered bootstrap (B = 2000, seed 20260906) gives [−1.4379,
+0.0727] against the receipt's [−1.8612, +0.1202].** The endpoints differ and
should: mine is a naive resample of 16 day means, the declaration specifies a
**stationary block bootstrap of the ratio-of-sums, 30-min bins, 4h blocks** — a
wider and more conservative estimator. **Both put `ci_hi` far below the 1.8 death
bar, so the verdict is identical under either.** The receipt's is the harder one to
pass and it is the one used.

## 3. My three v2 items — all present as FIELDS, and the third is answered precisely

| item | field |
|---|---|
| intersection, excluded events as a status | `delta_rs_population: "INTERSECTION … -- reviewer finding 1"`, with `n_true_valid_without_proxy: 178,659`, `share: 8.598%`, and *"how much of the window E1's mid was blind to … it bounds the reach of the void reading"* |
| gate vs death bar as two computed predicates | `gate1_bar_bps: 2.3`, `death_bar_bps: 1.8`, `point_clears_gate1: false`, `interval_clears_gate2: false`, `dies: true` |
| the interval in the decision rule | `ci_lo/ci_hi`, `interval_claimable: true`, `kill_is_interval_robust: true` |

**Is DEAD read from the interval or the point? THE POINT — and the reasoning is
right:** *"a POINT rule because that is its pre-registered text and a kill is not
weakened after seeing. Its interval robustness is reported beside it, never instead
of it."* **That is the correct handling.** Loosening a pre-registered kill because
its interval touches the bar would be exactly the selection the whole declaration
exists to forbid. The interval is reported as robustness, and it also passes.

**And my §B2 label finding is closed**: the receipt does not say ALIVE anywhere. It
carries `cell_dies: true` and `cell_passes_gate1: false` as **two separate
predicates**, so the 1.8–2.3 band cannot be mislabelled.

## 4. Attacking the DEAD reading — it survives everything I could throw

**Leave-one-day-out, all 16 days, mean and bootstrap ci_hi recomputed each time:**

```
worst case: drop 20260822 (the -6.39 outlier) -> mean -0.1660, ci_hi +0.1617
every other drop           -> mean between -0.692 and -0.166, ci_hi <= +0.162
LOO failures: NONE -- the kill survives dropping ANY one day
```

**The −6.39 day is not carrying the result.** Removing it still leaves the mean at
−0.166, an order of magnitude below the 1.8 bar, with the interval well clear.
**No single day can be dropped to make ADA live.**

**τ\* = 30 s was NOT chosen after seeing rs.** The rule is in declaration **v1**,
which I reviewed before any data: *"tau* = 30 s if the day-median
time-to-next-opposite-sweep ≤ 30 s on at least ceil(0.774194·G) days … Stated ex
ante to forbid horizon shopping."* The receipt records
`tau_star_why: "16 of 16 days have median time-to-opposite <= 30 s (need 13)"` —
**16 of 16 against a bar of 13, so the selection is not marginal and no other τ
could have been reached.** The full grid is reported and the gate read only at τ*.

**The earlier receipt (`…044234Z`) differs in 49 numeric leaves — every one of them
on `days[18]`.** I checked what that is: **`date: 20260906`, `admissible: false`,
`gap_fraction: 0.8011`** — the in-progress day, which the predicate excludes. It
grew between the two runs because the collector is still writing. **0 leaves removed,
no admissible day differs, and no result number moves.** The two receipts agree on
everything the verdict rests on.

## 5. The reproduction control — **driven by me, and it passes**

```
rs_eq_bps        2.4431339042529685   vs E1's 2.443    err 0.000134 bps
rs_notional_bps -0.3222202446455841   vs E1's -0.322   err 0.000220 bps
days_eq_positive 31/31 · days_notional_positive 7/31 · n_days 31 · tau* 30 s
reproduced: true (tolerance 0.05 bps)
```

**Both legs inside tolerance by more than two orders of magnitude.** So the proxy
estimator here **is** E1's, and Δrs is a difference between two **mids**, not two
codebases — which was the control's entire purpose.

## 6. **The one gap — and it is the same one I filed against DE two rounds ago**

`e2_0_true_mid.py:46` is `ROOT = HERE.parents[1]`, with `RAW`/`VISION` hung off it.
**There is no data-root resolver on the P-002 surface.** Driven: from my worktree
with a shell `data/`, `--reproduce-e1` returned

```
{"path": ".../ctaNew-wt-rev/data/mm_hf/vision/parquet/aggTrades/ADAUSDT",
 "status": "SOURCE_ABSENT"}
```

while the parquet is present at the ledger. **It fails CLOSED, which is the safe
direction** — and I only got the control to run by restoring my symlink.

**This is precisely the trap DE 74 closed for P-003** (`de_data_root.py`, imported
not copied, `require_canonical` refusing a result-bearing emission off the ledger
root). **P-002 has the same exposure and none of the fix.** Here it cost only a
reviewer's time; on a *result-bearing* run against a shell it would produce a
silently smaller population rather than a refusal, because `SOURCE_ABSENT` fires on
a wholly missing directory and not on a partial one.

**Not a hold on this result** — the receipt's own day census (24 hour-files per
stream, gap fraction, 16 admissible of 19 emitted) is inconsistent with a shell,
and the reproduction control passed against the real Vision parquet. **But P-002
should adopt DE's resolver before its next result-bearing run.**

## 7. Resources

```
wall 66.27 s · max RSS 672,664 KiB = 0.642 GiB of the 8G cap
```

**Comfortably inside**, on the symbol the declaration named as the smoke, and ADA is
~1/12 of BTC's daily bookTicker volume — so the fan-out estimate the declaration
refused to guess now has a measurement to extrapolate from. The wrapper was used
throughout; I drove every run under it.

---

## Verdict, stated for the record

**SETTLED. ADA is DEAD on real books; E1-B is empty; E1x is moot.**

* the void leg does not fire (−0.0066 vs +1.0), so **E1's mid was not the problem**;
* the settle leg kills on the pre-registered point rule, and the kill is
  **interval-robust and leave-one-out-robust on all 16 days**;
* the mechanism is the one E1's audit predicted — **eq +1.36 against notional
  −0.56**, the dollars adversely selected;
* and the estimator was proven to be E1's own before either number was read.

**Nothing must change first.** One item for P-002's next round: adopt the data-root
resolver (§6). And the honest scope limit is the receipt's own — this is
**2026-08-20..09-05 on ADA**; it kills the cell on this period and does not
re-measure E1's window, which is exactly what the declaration said it could not do.

---

## CONTEXT

Far below the 80% reset threshold.
