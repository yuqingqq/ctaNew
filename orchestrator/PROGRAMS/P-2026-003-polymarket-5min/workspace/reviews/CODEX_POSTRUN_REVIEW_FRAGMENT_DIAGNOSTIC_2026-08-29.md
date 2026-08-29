# Codex post-run review — R-293 fragment diagnostic — 2026-08-29

**Receipt:** `/home/yuqing/ctaNew/data/pm_5min/derived/be_fragment_diagnostic_v1.json`

**Receipt identity:** 14,790 bytes; SHA-256
`19286320e826d0407d1fbb068a211d28860a8cce5ab9dd98bf54b0479d3bf691`

**Run authorization:** `CODEX_FINAL_PRERUN_FRAGMENT_SCORE_2026-08-29.md`
at `1a4b914`; coordinator launch record R-324 at `96d6fe2`; first memory
interpretation R-325 at `81643c4`.

**Scope boundary:** I audited the one existing receipt and its producing logs,
source bytes, frozen artifacts, arithmetic, population, and interpretation. I
did **not** run `score_stage` again. R-293's one-run allowance is consumed.

## Verdict

**RECEIPT MECHANICS AND NUMBERS VERIFIED. POSITIVE MODEL DIAGNOSTIC; R-293'S
“WEAK COMFORT ONLY” READING UPHELD. NO MODEL, BUDGET, CANCELLATION-POLICY, OR
STRATEGY PROMOTION.**

The frozen candidate has informative conditional-value ranking on this
fragment: its net is positive, above the incumbent point estimate, and above
every one of the 200 matched-random candidate controls at all three frozen
budget labels. The effect is spread across the covered hours rather than being
one best-hour total.

That is the strongest supported statement. This remains an inadmissible,
censored, selected 253-window fragment with fewer than five complete untouched
UTC day clusters. It is not strategy performance and it does not test the full
queue-realistic `QR_CANCEL_HOLD_X_SKEW` policy against required comparator
`QR_SKEW_ONLY`.

Two statements in R-325/HANDOFF overreach the artifact and must be corrected:

1. the **candidate net**, not the candidate-minus-incumbent increment, is what
   was compared with matched random;
2. the emitted tie check is evaluated at the nominal top-k budget index, not at
   the actual frozen causal threshold, so it cannot prove that threshold ties
   were absent. Canonical sorting still makes this particular run deterministic.

Neither correction moves a number or licenses a rerun.

## What the receipt actually shows

All values below are cents at the action/first-crossing unit, not P&L:

| Frozen label | Candidate net | Incumbent net | Point increment | Candidate random max | Actual firing rate | Positive active hours | Best-hour share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% | +13,661.118 | +6,864.748 | +6,796.369 | +949.653 | 4.354% | 20/22 | 14.26% |
| 10% | +21,185.555 | +11,114.457 | +10,071.098 | +196.130 | 8.868% | 21/22 | 10.54% |
| 15% | +24,551.893 | +17,349.760 | +7,202.133 | −169.195 | 13.254% | 21/22 | 10.23% |

For every cell:

- candidate and incumbent run `CAUSAL_FROZEN_FROM_TRAIN`;
- `candidate_net = harm_avoided − sacrifice` exactly within floating tolerance;
- `increment = candidate_net − incumbent_net` exactly;
- `rho = harm_avoided / sacrifice` exactly;
- candidate net is above the maximum of 200 side×hour/action-count-matched
  random draws;
- net remains positive after removing the best hour.

The point increment is positive at all three labels. It has **no matched-random
null and no day-cluster interval** in this receipt. Therefore “every increment
beats the matched-random null” and “the 10% increment is 50× its null maximum”
are category errors: the quoted random maxima belong to candidate net, not to
the difference between candidate and incumbent. A paired delta null would be a
different statistic and was neither preregistered nor computed here.

The largest point increment occurs at 10%, but selecting 10% because of this
seen fragment would be tuning on an inadmissible read. The three frozen labels
remain descriptive; none is promoted.

## Population and valuation checks

The population reconciles without remainder:

```text
442,964 kept
+29,129 PRE_WINDOW
+   307 GAP_AT_CUTOFF
+    13 NO_LEVEL_HISTORY
=472,413 exposure OK rows
```

All other drop counters, including `state_join_failed`, are zero. The 442,964
kept rows form 224,648 unique actions. Every valuation row was validated before
the canonical `hm.keptrow` reconstruction; 64,343 gates are true and 378,621
false, summing exactly to the kept population. Candidate and incumbent vectors
both have 442,964 finite values and are non-constant.

The receipt is strict JSON, contains no non-finite constants, records real-data
provenance, the exact reviewed tape/exposure paths, the prewritten reason,
50 ms latency, the frozen model/scaler hashes, the v4/verdict/exposure/ledger
bindings, the absolute/window-relative clock contract, and the unconditional
inadmissibility statement.

## Tie-field correction

The receipt's field is named `gmax_tie_at_budget_boundary`, but the code computes
it as:

```text
k = int(n_actions * nominal_budget)
gmax[k - 1] == gmax[k]
```

The causal policies do not select those top-k counts. Their frozen thresholds
cancel 9,782 / 19,921 / 29,774 actions, versus nominal k values 11,232 /
22,464 / 33,697. The artifact does not carry each frozen `theta` or the number
of generation maxima equal to it. Thus `tie_at_boundary=false` answers a
retrospective nominal-top-k question, not whether the actual causal threshold
has ties.

This does **not** invalidate the recorded cells: `score_stage` canonically sorts
the unique decision rows before the evaluator, and the artifact honestly marks
the output deterministic but not the underlying evaluator order-independent.
It invalidates only the stronger memory claim that the tie diagnostic proved
order sensitivity “could not have bitten this population.” Future receipts
should report ties at the actual frozen theta (and retain the nominal field only
if it is labelled as such).

## External provenance binding and preservation finding

The current receipt omits three facts needed to reproduce its matched-random
headline from the artifact alone. I verified them in the exact committed
source used by the unchanged score path:

| Binding | Verified value |
|---|---|
| `be_fragment_diagnostic.py` SHA-256 | `398129ae6d571805e48add7b887c857b9ad070e121680c6205c6f03ce8ebef58` |
| `harmful_action_eval.py` SHA-256 | `2c4e21936e3fc1d2d91af5dad86afa113a7c3f5f8ca24a7a9efa1bd53f66db64` |
| `phase2_iter011_run.py` SHA-256 | `ca4dc7c624371ad114bb83e1fbc0d2ae60c4f5281763c73083e7e513403bbccc` |
| `phase2_arms.py` SHA-256 | `ab19f5c639333bdc6157fd11f46b32f641fc94c05a28694469f38cc67473d2ec` |
| measured fit identity | `e27cab9e5f6ce8e5` |
| matched-random count | 200 |
| matched-random seed | `20260825` |
| matched strata | side × absolute UTC hour, equal action count |

At review time the result file is ignored by the repository-wide `data/` rule
and is not tracked. A committed document naming its hash does not preserve the
bytes whose hash it names. Before this result is treated as durably recorded,
preserve the exact 14,790-byte file in git (it is far below the repository's
1 MB data limit) or write a committed, content-bound superseding receipt that
carries the complete result and the bindings above. **Do not rerun** to add
metadata; this is a non-numerical preservation/provenance repair.

Future result artifacts should self-carry scorer identity, `n_random`, seed,
matched-control design, each actual frozen theta, and tie count at that theta.

## Execution evidence

```text
systemd unit: be-fragscore-1788042712.service
result: success; exit status 0
runtime: 7m30.887s
CPU: 7m33.760s
memory peak: about 7.5 GiB; swap peak 0
R-148(3): research.slice + per-job MemoryMax=12G (allowed maximum 14G)
receipt SHA stable across repeated reads: 19286320e826d040...
strict JSON and arithmetic audit: PASS
```

## Research consequence

This result answers the narrow model question in the encouraging direction:
the conditional-value model is not merely detecting that flow arrives; on the
surviving fragment it ranks actions whose first-crossing cancellation value is
better after sacrifice is charged, and it does so more strongly than the
frozen linear incumbent at the point-estimate level.

It does **not** answer whether the integrated cancel/repost/skew strategy makes
robust queue-realistic returns. Keep the model and all three labels frozen;
make no feature, threshold, or budget choice from this fragment. Accumulate the
post-O1 independent complete-day window first. When the governed forward gate
opens, the integrated evaluation must still use `QR_CANCEL_HOLD_X_SKEW` as the
queue-realistic baseline and `QR_SKEW_ONLY` as the required comparator, with no
same-price zero-queue/front assumption and no promotion from seen days.
