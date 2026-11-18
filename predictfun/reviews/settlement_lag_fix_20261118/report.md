# Settlement-lag correction — 2026-11-18

User authorization: **“Can we fix this issue”**, following the
[review of 084d5a19](../automatic/20261118_084d5a19/report.md).
Base commit: **52c45d57537fd152380112adef0393b43e258653**.
Development was isolated in `/tmp/predictfun-fix-k8-20261118`; other-session
working changes were preserved. This is an implementation correction, not
a review-only commit or a new model candidate.

**The common-offset inference bug is fixed.** A feasible common total delay
no longer becomes a maximum creation delay for every event. The corrected
known-order receipt has **no universal lag bound**, and the public-trade ST
window has no finite lower endpoint by default. Exact venue matching times
and simulation timing remain unvalidated.

## Corrected constraints

For leg i, on the bot clock, define match `m_i`, placement `p_i`, last possible
match `u_i`, and first receipt `n_i`. Its UUID creation stamp is `t_i` on the
venue clock. With common clock offset `delta` and separate creation lag `l_i`:

```
t_i = m_i + delta + l_i
l_i >= 0
p_i <= m_i <= u_i
t_i <= n_i + delta
```

The last inequality uses the actual first notice carrying this order/trade
identity: the UUID already existed when the notice arrived. This was verified
against the hash-pinned fill-events artifact on all 99 saved legs, matching
order ID, trade ID and exact receipt timestamp. See [identity evidence](notice_identity.json).

Intersecting these constraints with K7's clock interval gives:

```
delta_low  = max(K7.low,  max_i(t_i - n_i))
delta_high = min(K7.high, min_i(t_i - p_i))

known match_i ∈ [p_i, min(u_i, t_i - delta_low)]
known lag_i  ∈ [max(0, t_i - delta_high - u_i), t_i - delta_low - p_i]
```

These per-leg bounds apply only to their identified lifecycles. They are not
transferred to other orders or days. Creation at/after matching, shared
UUID/status clocks and the existing lifecycle/K7 premises remain explicit
assumptions. Equality of UUID and Matched **seconds** is reported descriptively;
different seconds do not invalidate a feasible nonnegative variable lag.
The old common-d intersection remains a labelled conditional diagnostic,
including the stronger zero-creation-lag clock scenario.

## Pipeline changes

- K8 uses the separate-lag constraints above. Missing or incompatible
  observations/clock bounds cannot yield a valid timing fit. Known match and
  creation-lag intervals are saved per leg.
- K5S and fill-notice checks use these wider valid intervals. They no longer
  infer slow cancellation from the unsupported common-delay match window.
- ST defaults to an upper-only UUID window, conditional on creation after
  matching and aligned clocks. Missing IDs retain the explicitly labelled S3
  fallback assumptions. A finite lag supplied directly to the diagnostic is
  an **explicit sensitivity assumption**, not a bound learned by K8.
- The known-case ST gate reads **UNIDENTIFIED-UPPER-BOUND-ONLY**, avoiding a
  vacuous acceptance from testing a fitted lower bound on the same sample.
- The default scenario bank omits **STlo** because no finite earliest-match
  endpoint is identified. SThi remains an upper-endpoint sensitivity scenario.
  No nonfinite timestamp may enter a replay. The bank now has 19 scenarios;
  this count change does not establish timing robustness.
- Notice, activation and slow-cancel mixture values inherited from v3 remain
  explicitly assumed, posthoc sensitivity settings. Their measured-distribution
  interpretation is withdrawn.
- Old K8 receipts and legacy scenario receipts are refused by the corrected
  consumers. Output versions advance to execution ledger **v5**, known ledger
  **v4**, and scenario replay **v4**; the driver uses these new paths. Earlier
  artifacts remain unchanged. [Supersession receipt](supersession.json)
  identifies the withdrawn claims and the exact prior artifact hashes.

## Existing known-order evidence after correction

Consumed **Nov 8**, output as of **2026-11-18T11:43:28.451761Z**; source case
receipt as of **2026-11-18T10:56:10.677848Z**. Population: **99 legs from 86
orders**, 93 anchored and six unanchored, with 131 candidate rows. These are
filled orders from one partial development day, not a representative latency
sample or untouched validation. Receipt: [known.json](known.json).

| Quantity | Corrected reading |
|---|---|
| Universal creation-lag bound | Unidentified; the 259.034 ms claim is withdrawn |
| Shared clock interval under the stated premises | -147.838 to +32.641 ms; includes zero |
| Cancellation within [0,100] ms, 77 orders with confirmations | 65 certified, 12 compatible-unidentified, zero contradicted |
| Remaining nine orders | No cancellation confirmation |
| Previously claimed 10 definite slow cancellations | Not established by the corrected bounds |
| Activation range [25,100] ms | One contradiction remains: acceptance at +21.165 ms; 85 compatible-unidentified |
| Notice-delay point of 100 ms, 99 legs | Excluded by bounds on 87; compatible-unidentified on 12 |
| Public ST match lower bound | Unidentified; not inferred from the sample |

The first-notice constraint legitimately narrows the earlier K7 clock interval
without forcing equal creation delays. S3's false-certified-avoidance check
therefore passes within this interval, but its other contradictions and the
replay-anchor failures remain in the report. This does not validate S3, the
book clock, queue behavior, or the execution engine.

The known per-leg creation-lag upper bounds extend to about 5.9 seconds on
these lifecycles. These are allowed limits, not observed delays or an estimate
of their distribution. The correction does not substitute another fitted
maximum for 259 ms.

## Verification

All execution used **PF_SLOT=codex** via the absolute `run_light.sh` path and
absolute scripts, with no full tape replay, training or untouched-day read.

- Both complete synthetic instrument suites pass, including the corrected
  scenario replay. [Synthetic run](synthetic_run.txt): 20.98 s, peak 703 MiB.
- The review's 10/700 ms variable-lag counterexample and the constant-lag
  positive control are accepted with valid intervals. Impossible creation
  times, inconsistent clocks and stale receipts are refused. The prior false
  slow-cancel inference becomes compatible-unidentified.
- **200 independently planted variable-lag worlds**, eight legs each, retain
  the true match, lag and common clock inside their inferred intervals.
- The full synthetic execution ledger also runs with **upper-only** ST
  windows; endpoint metadata records unknown lower bounds. A legacy scenario
  receipt refuses before data loading/output. All mutation replacement targets
  remain valid. [Synthetic receipt](synthetic.json).
- **10 targeted mutations** affecting K8, K5S, ST and receipt/gate checks are
  detected. The full historical mutation suites were not rerun.
  [Mutation evidence](mutants.json), [run](mutants_run.txt): 27.05 s.
- Rebuilt the existing known-order case ledger without re-extracting the tape:
  [run](known_run.txt), [cases](known_cases.csv), [orders](known_orders.csv).
  All input/output hashes are recorded; the corrected known receipt explicitly
  supersedes v3. Exact identity/receipt verification passed separately.
- Driver shell syntax and `git diff --check` pass. [Source hashes](sources.sha256).

No corrected public-day economics are claimed here. The old 20-scenario values
remain reproducible under their old assumptions, and are superseded as timing
evidence. Further narrowing requires an independent match/creation-delay or
clock constraint. Until then, economic results must retain this timing
uncertainty; model tuning is not a remedy for it.
