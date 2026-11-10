# Optimization loop: native A2 execution comparison — 2026-11-09

As-of **2026-11-09T17:07:39.812735+00:00**. Same **220 consumed Nov8 windows**, **279,059 decisions**, **13,077 neutral touch-pull actions per arm**; zero excluded windows. One partial UTC day, no confidence intervals or independent validation.

The quote-continuity benefit survives the current Rust A2 request mapper, venue throttle and retry backoff in this conditional no-fill test. The 50 ms policy still increases active exposure; it does not demonstrate lower inventory risk or improved P&L.

| Native A2 policy | Venue requests | Request change | Original aligned time change | Off-target time change | Active share-time change |
|---|---:|---:|---:|---:|---:|
| Reference | 62,788 | +0.00% | +0.00% | +0.00% | +0.00% |
| 100 ms timer first | 59,440 | -5.33% | +4.41% | +36.81% | +4.76% |
| 50 ms updates first | 59,444 | -5.33% | +4.17% | +28.36% | +4.10% |

Original-price coverage rises from **73.85% to 76.93%**. Updates-first50 retains **94.43%** of the 100 ms aligned-time gain and removes **22.96%** of its additional off-target time. It retains 4,773 pull actions and bridges 2,371 same-price reopens.

## What changed in the execution model

- The isolated native checkout is pinned to `f20fdb1`. The adapter calls the original A2 pair mapper and request methods, using an exact copy of the native intent-change loop. Removing the diagnostic accessors restores the original source. No live checkout or deployment was changed.
- A new quote is one whole order; size growth preserves the existing member and adds the remainder. Native request tokens, unanswered-order limits, and 250–4000 ms retry backoff are active.
- Original quote targets and guards remain factual and exogenous. Probes may cancel a held pair, but cannot create a fresh quote. No captured fill is applied to a hypothetical order. Native decision counters are zero because this harness invokes the request mapper directly; the source population still contains all 279,059 decisions.
- Venue feedback is assumed: 40 ms create/answer, 22 ms accepted cancel, at least 95 ms for pending-create cancellation, and 1 s external reservation grace. Capacity is 150 shares per side at hypothetical net zero, separate from the dollar loss budget. Recorded net only conditions exposure metrics.
- Orders are physically active until cancel completion and reserve external capacity through the additional grace. Locally refused creates count as native requests but never reach the modeled venue. No post-only rejection, shared kernel throttle or wallet I/O model is added.

| Native counters | Reference | 50 ms updates first |
|---|---:|---:|
| Native request outputs | 65,583 | 62,295 |
| Requests sent to venue | 62,788 | 59,444 |
| Native throttled attempts | 2,237 | 1,262 |
| Local capacity refusals | 2,795 | 2,851 |
| Retry timestamps executed | 3,832 | 3,090 |

The 50 ms arm changes reserved share-time by -1.22% but active inventory-range expansion by +4.16%. Fewer reservations do not imply lower active risk. Both captured modes have the same request/coverage direction: **True**; they are not independent days.

## Verification and provenance

- 51 existing native tests, seven adapter controls and four Python checks pass, covering paired partial fills/front rotation, request backoff and rejection reset, equal-clock versus late reopen, hard-probe cancellation, reservation release boundaries, malformed inputs and corrupt artifacts.
- Saved output verifies across 660 window-arm rows and 39,231 action-arm rows. Action identities match the original input events. Coverage and exposure are independently rebuilt from saved order intervals; command counts, capacity and lifecycle deadlines reconcile.
- The first preparation attempt refused because the adapter treated native `-1` off-quote ticks as prices. It produced no native experiment output. The corrected adapter and declaration were committed before outcomes, with a sentinel regression check. The failed declaration/input/log remain preserved and are superseded in-band by `native_a2_v2`.
- Source commit `4776d822`, declaration `8274fc63`; cumulative distinct policy settings remain 21. This retests three existing policies in a new execution model. It does not validate the full B4 engine or a deployed timer scheduler.
- Largest successful job: 758 MiB RSS; all numeric jobs complete. Protected Nov9–13, the frozen engine and loss-budget settings remain unchanged.

Earlier rounds and latency tests used the simpler research planner; retain their stated assumptions. The native comparison here uses only the base latency profile. Next: native faster/slower sensitivity with exact base parity, then full B4 integration and calibrated economic evidence.

Register: `reviews/optimization_loop_20261109/plan.json`. Combined superseding receipt: `findings_v3.json`. Native receipts: `native_a2_v2/`. Reproduction on the staged checkout: `pf_native_a2_continuity build`, `declare`, `prepare`, `run`, `verify`, then `pf_native_a2_report`, with numeric jobs under `run_light.sh`. Destinations are exclusive-create; preserved artifacts must not be overwritten.
