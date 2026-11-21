# ENTRY-TARGET-AUDIT-v1 combined review — accounting accepted; test the initial order's exposure next

Date: 2026-11-21. Independent evidence checked through 07:59:40 UTC.

**Full reviewed target SHA: `37300cc93c47900e11eb1806eefb4c6febce2ccb`.**
Range: `d6d5d55e254c507db9799ef2fc1c6129d4657d06..37300cc93c47900e11eb1806eefb4c6febce2ccb`.
Pinned checkout `/tmp/pf-review-37300cc9`; development owns implementation. Review artifacts and review-owned state only; no push.

**The accounting results are reliable. There is no BLOCKING finding for using the saved populations, decompositions or descriptive tables.** All 24,416 source POST rows, including 160 excluded rows, reproduce; all 24,256 ok decompositions reconcile; the separate 760 reached rows and every summary column reproduce. The recommendation's strongest wording needs qualification: this is **rare initial execution under the frozen PI0 continuation**, not proof of rare market opportunity or proof that no entry-target change can help.

**ONE next priority: ENTRY-EXPOSURE-v1**, a bounded, declared comparison of the same initial post under PI0 versus keeping that initial generation without discretionary policy cancellation. Keep the structural rule, original END and all other decisions unchanged. This is a longer-exposure diagnostic on already consumed dates, not a new production minimum-hold timer or another fit. No additional USER approval is needed for the bounded contract below.

## Accepted readings and additional evidence

| Quantity | Oct24–25, in-sample dates | Oct26–27, adaptive development |
|---|---:|---:|
| All POST rows, statuses retained | 12,228 | 12,188 |
| Ok accounted rows | 12,168 | 12,088 |
| Weighted meaningful-Y share | 3.0263% | 2.8017% |
| Weighted initial-generation no-fill share | 97.1431% | 97.4100% |
| Initial-generation filled rows, unweighted | 344 (170 bid, 174 ask) | 297 (144 bid, 153 ask) |
| Weighted sum abs(Y) / weighted sum abs(V_initial) | 0.9449 | 1.1415 |
| Initial absolute-value mass offset within the same row by D_following | about 29.5% | about 6.1% |
| Policy PULL request within 100 ms of the initial POST, weighted row share | 88.7385% | 91.8716% |

The last two rows are independent review diagnostics, not new frozen acceptance gates. Sources: [diagnostics.json](diagnostics.json) and [data.json](data.json). The first-PULL count is 10,931 / 12,168 training rows and 11,161 / 12,088 evaluation rows; 10,692 and 10,896 of those respectively have no initial fill. These are logged **policy request times**, not reconstructed structural lifetimes or a counterfactual estimate of what keeping would earn. Repeated/related rows are not independent events.

The within-row offset calculation is
`sum(w * 1[V_initial * D_following < 0] * min(abs(V_initial), abs(D_following))) / sum(w * abs(V_initial))`.
It confirms that value-offsetting is not the dominant aggregate explanation on these evaluation rows, while avoiding the original predicate's ambiguity. It does not establish why fills are rare.

The original components and frozen-prediction cross-tabs also reproduce. A higher mean V_initial in a model's POST group than its WAIT group is not positive value or useful timing by itself: both means can be negative, and different fill rates contribute to that comparison. The R4E model/economic failures remain unchanged.

Everything here uses the existing gross, zero-fee five-second END-mark convention. Four consumed dates do not support validation or a day-cluster interval. The 760 reached cases are a separate world/side-balanced diagnostic; no pooling with the weighted neutral-source sample. No settlement P&L or profitable strategy is established.

## Consolidated ranked checklist

All findings below are **NONBLOCKING for the current accounting readings and the next bounded experiment**, with verified workarounds stated. Do not require a separate repair-only review. The unsupported stronger causal claims are not accepted.

1. **P1 — NONBLOCKING: “rare execution only” overstates what the predicate identifies.** `pf_entry_audit.py:210–214` tests a low meaningful-label share and a high initial no-fill share; it does not test order survival, structural removal, queue progress or market flow. `tools/fill_audit.py:13–18` then maps that flag to “relabeling ... will not solve the entry problem.” The descriptive sparsity is correct, but the implication is too strong. A different initial-order policy changes the outcome distribution. The independent logged-PULL diagnostic above makes that distinction material: PI0 requests very early removal in most rows. **Action in the next combined receipt:** say “changing Y to V_initial alone would retain the observed sparsity under PI0”; do not claim a market-wide scarcity or that sparsity alone proves the cause of model failure. Measure the changed exposure before another fit. The 50 ms is a decision grid, not a guarantee that the order rests for only 50 ms: for a POST at t followed by its first live PULL at t+100, modeled activation is t+58 and removal t+137, absent another termination (`pf_branch_engine.py:6–25,138–147`).

2. **P2 — NONBLOCKING: the named `cancellation` flag is a net absolute-mass screen, not a within-row offset measure.** `pf_entry_audit.py:213` and its header compare `sum(w abs(Y))` with `sum(w abs(V_initial))`. Large continuation effects elsewhere, or a continuation that overshoots through zero, can make the flag false even when it offsets initial value. The independent synthetic counterexample in [diagnostics.json](diagnostics.json) has all 100 c of initial value offset, but a separate 1,000 c continuation outcome makes the flag false. The actual-row overlap is about 6.1% of initial absolute mass on eval and 29.5% on train, so the corrected measurement supports the narrower present reading. **Action:** retain the frozen flag as historical provenance, name it as an aggregate attenuation screen, and show the rowwise offset measure when discussing mechanisms. Distinguish *value offsets* from *order cancellations*. No relabeling is required.

3. **P2 — NONBLOCKING: acceptance checks are weaker than source-to-output reconciliation, but this review supplies that reconciliation for the saved artifacts.** Check A verifies counts/status totals/uniqueness, not the exact source key and weight set; B checks an algebraic sum, not the component allocation from fills; D mainly checks 760 rows and the identity; E checks selected aggregate totals and the diagnosis string (`audit_checks.py:39–93`). A copy with `V_initial += 1`, `D_focal -= 1`, and `D_following -= 1` still passes B, recorded in [controls.json](controls.json). Also, `account`'s identity alone cannot establish that a plausible but wrong gid identifies the forced generation (`pf_entry_audit.py:78–104`). **Current mitigation:** independently stream the hash-bound original parts; locate the unique forced decision; verify gid/side membership and all component values from fills; reproduce every saved row/weight/category/prediction and every summary column. All pass. **Action:** reuse this reconciliation if the audit/checker is reused, including a compensating-component corruption and a wrong-but-in-range gid control. Do not rebuild good labels or expand into general hardening.

4. **P3 — NONBLOCKING: two reporting conventions need explicit denominators.** The receipt's 0.022 core-hours excludes nested mutation-job logs because `tools/fill_audit.py:38` uses `glob('*.log')`, despite saying every job. Recursive check F reported 0.03120 core-hours at its original cutoff; the final saved logs contain 0.03277 core-hours across 56 jobs, peak 456 MiB, still far below the one-hour ceiling. Separately, `group_summary` computes top-5 concentration from **unweighted row abs(Y)** (`pf_entry_audit.py:199`), whereas adjacent component totals are sampling-weighted. **Action:** use check F's resource aggregation and label the existing top-5 statistic “unweighted rows”; optionally add a distinctly named weighted/original-grouped concentration when that table is next touched. Neither issue changes the present sparsity or accounting conclusion.

## Previous items, amendment and bounded verification

The d6d5d55e target-audit population, full status retention, exhaustive accounting classes, timed economic comparison, exact/residue distinction, arm-local forced-generation identity, initial fill status, WAIT-next-entry measurement, frozen prediction joins, per-pick/stratum summaries and separate reached diagnostic are implemented and reproduced. The A / V_post / V_retain naming and mandatory-anchor versus all-sampled population clarifications are correct in this receipt. The previous episode-check F issue remains deferred with the independently checked R4E artifact workaround; that checker is not used here.

The writer amendment is acceptable: `6157b1a0983f8a0cb763682d3108d76786e41eb0` changes output naming/writing and the corresponding checker/test/formatter, not the accounting definitions or thresholds. The selftest now exercises the writer and rejects a restored filename collision. Original freeze `811fd7b3501602c3427b74dbc07032b611900559` is 07:47:50 UTC; amendment 07:50:18; accepted audit as-of 07:51:34; target result commit 07:52:30. The interrupted run's “no files written” assertion is developer provenance, not something the current artifacts can retrospectively prove; the accepted outputs and unchanged computation are verifiable.

Every executable review check used `PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh` and an absolute path to [verify.py](verify.py), importing the pinned checkout:

- [metadata.json](metadata.json): all nine committed/data output hashes; audit/model/prediction identities; 16 final-source mutation records with copied-record refusal; all synthetic accounting, predicate, join and writer tests.
- [data.json](data.json): streamed both accepted datasets and their part hashes; 24,416 POST rows including all statuses; 24,256 ok source ledgers and independent component allocation; every saved output column including weights, classes, in-sample/reused predictions and all summaries; 760 reached rows reproduced from source replay parts. No dropped status or mismatch.
- [diagnostics.json](diagnostics.json): rowwise value-offset measure and its synthetic counterexample; event counts by day/side. These are descriptive mechanism checks, not fitted parameters or hypothesis-test results.
- [controls.json](controls.json), [checks.json](checks.json), [checks_falsifier.json](checks_falsifier.json): A–G reproduced; all declared clean/known-bad controls hold; compensating-component checker limitation demonstrated on a temporary copy.

Four successful bounded jobs: **79.29 seconds total wall time, peak 456 MiB**. No fitting, label build, real-tape replay, new date or untouched-day access. Existing source/tape assumptions are inherited from the previous reviews; this was not a fresh simulator certification. No implementation files or other-session artifacts were changed.

## ONE next priority — ENTRY-EXPOSURE-v1

**Choose the longer-exposure experiment next, before fitting a conditional-value model.** It directly addresses whether the current target/controller combination removes the initial order before useful execution develops. It can also show the opposite: early PI0 removal may be preventing adverse fills. Current no-fill statistics cannot decide between those possibilities.

Declare one continuation-policy variant, count it in the experiment ledger, and submit **one combined code/synthetic/measurement receipt**. This is a diagnostic action-unit experiment, not a new fitted entry candidate or a profitable-policy claim. No interim approval/review after each repair.

### Population and deterministic manifest

- Dates **Oct24, Oct25, Oct26, Oct27 only**, from the existing accepted train/eval plans and producing inputs. All are consumed. Use original-time **neutral mandatory POST** forks (`pf_r2_label_sample.is_mandatory`), so the question is the first entry with the full original five-second horizon. This intentionally does not estimate later-generation performance.
- Select **up to four worlds per UTC hour per day**, at most **384 worlds / 768 side-specific forks**. Start from the accepted plans' world memberships, not successful fills or ok-outcome rows. Rank by SHA-256 of UTF-8 `entry_exposure_v1|` plus the existing canonical JSON of `{day,m,t_ms}`; ties by that JSON. Take the four lowest ranks in each hour, or all if fewer. Freeze the exact selected worlds, source rows/fingerprints, input/model/source hashes, per-hour population counts and inclusion fractions before simulation. No outcome, prediction or first-pull result enters selection.
- For each selected world and both sides, retrieve the saved mandatory row/context. Preserve a world's missing path, absent mandatory row, unavailable context or other exclusion as its own status; do not replace it with a successful world. No census regeneration. Reuse `fork_for`/validated contexts and the original builders/readers unchanged. Each reproduced baseline fork must match the saved fingerprint.
- Primary comparisons are **equal-world, per-day/per-side finite-sample means** on common evaluable forks; show denominators and all excluded statuses. Do not reuse the old per-pool row weight as if it were an original-entry weight. No sum over overlapping forks described as strategy P&L. An optional population-weighted table must separately declare the original-world and new hour-sample inclusion probabilities; it is not necessary for this bounded mechanism test.

### Three branches from each identical fork

| Branch | Focal action at the fork | Continuation |
|---|---|---|
| `WAIT_PI0` | WAIT this tick | Original PI0 on both sides |
| `POST_PI0` | POST now | Original PI0 on both sides; exact reproduction control |
| `POST_KEEP_INITIAL` | The identical POST now | Suppress discretionary policy PULL only while the forced initial gid is the live focal generation; all other decisions use the original PI0 |

In `POST_KEEP_INITIAL`, the initial generation is still subject to post-only refusal, partial/full fills, the unchanged **client-known structural pull rule**, existing pending/acknowledgement timing, inventory/reservation caps and original END. It closes normally. After it closes, PI0 again governs focal re-entry and all later generations; the opposite side always uses PI0. No new layer, order size, cancel retry, numerical score threshold, arbitrary 250 ms / 1 s hold, structural-rule removal, or endpoint extension. Keep size 30, cap 30 and the frozen 58/37/100 ms timing settings. This is permission to test one continued-resting alternative, not to force a new deployed minimum lifetime.

Build the wrapper **outside the frozen source closure**. Bind a fresh teacher to each actual branch; retain causal input/binding probes at the declared every-35-callback rate. Log the shadow PI0 recommendation and the actual overridden action separately; never label an override as an ordinary PI0 decision. The override selects its gid from the executed focal POST, not an assumed global gid or another arm. Engine structural requests run in their original order and cannot be undone by KEEP.

### Values and measurements

Use the same common END mark and gross costs. Declare before outputs:

- `Y_PI0 = R(POST_PI0) - R(WAIT_PI0)` — must reproduce the stored label/control arms.
- `Y_KEEP_INITIAL = R(POST_KEEP_INITIAL) - R(WAIT_PI0)` — the new, explicitly different committed-action contrast.
- `D_EXPOSURE = R(POST_KEEP_INITIAL) - R(POST_PI0)` — the primary paired exposure comparison. Verify `Y_KEEP_INITIAL = Y_PI0 + D_EXPOSURE`.

For each day/side report: statuses; initial fill probability and filled quantity; initial active time; time to first fill; request/effect times and termination reason; exact/meaningful value counts; initial-generation END value and subsequent/cross-side components; means of all three contrasts; positive/negative sums; worst values and abs(value)>30 c counts; open inventory and unfilled initial order at END. Preserve the actual queue-ahead trajectory/summary as a diagnostic if available without changing the engine, tagged as venue state rather than a policy input. More fills alone is not success.

Report the same branch fill-level execution edge and 50/100/1,000 ms markouts where their reference time is inside the **existing** episode and the mark is valid; retain missing/late-horizon statuses and quantity denominators. Do not extend END or read new dates to manufacture those marks. These conditional-on-fill tables explain adverse selection; the common-END, all-fork comparison remains primary. Neither a favorable markout nor an unfilled place near the front of the queue receives an arbitrary terminal bonus.

### Acceptance, budget and resulting decision

- Freeze adapter, exact manifest, identities, metrics and controls before any new branch result. Reproduce both stored PI0 arms and Y, including prefork fills, fill/decision hashes and accounting on **every evaluable selected fork**. No skipped failed reproduction. Preserve all three raw fill/decision/request logs, generation metadata, counters and common END/settings. Validate actual branch binding, source identities, fill-time bounds, cash/inventory/value and the full decision population.
- Synthetic controls must include a queue that needs additional resting time to fill; a harmful late fill avoided by PI0; structural removal/refusal despite KEEP_INITIAL; a partial fill then ordinary closure; later-generation and opposite-side PI0 actions untouched; wrong-gid/stale-branch refusals; and baseline equivalence when PI0 already keeps. A value increase is not a passing implementation test. Batch the present naming/diagnosis/accounting-check follow-ups with this measurement where cheap.
- Budget: **two aggregate core-hours**, within the already authorized development slots (up to four single-core slots, no extra cores); 1,536 MiB checkpoint / 2,048 MiB hard RSS and 120-second per-fork-package guard. Use the first fixed manifest partition as the cost pilot; project and stop with an explicit incomplete status if the ceiling cannot be met. Do not silently shorten the horizon, reduce the sample after results, relax the guard or accept held outputs. No fit and no full label expansion in this round.
- Readings are per day/side, descriptive, with no CI or significance claim. Do not select a favorable side/horizon. The result should say whether longer initial exposure produces additional fills **and what those fills cost**. If fill/value evidence remains thin or days disagree, say not identified. If improved exposure/value is supported, the next target must name this continuation and be paired with a controller that implements it; do not train on KEEP_INITIAL labels and silently deploy PI0's early cancellation. If extra fills are mostly harmful, retain that negative result and focus later work on adverse-selection discrimination, not larger posting size or unconditional posting.

## Decisions on the alternatives and USER boundary

- **Fill probability plus value conditional on a fill:** a legitimate future decomposition, but not the next experiment. Estimate probability on all decision-time opportunities; a conditional component may use filled outcomes for its declared conditional task, while evaluation and decision value remain on the full opportunity population. `P(fill) × E[V_initial | fill]` only estimates initial-order value; it does not equal the existing whole-branch Y without the continuation/opportunity-cost term. The present sample has only 170/174 filled training rows by side before clustering, and the existing PI0 J architecture already separates execution and conditional moments. Repeating that factorization on early-cancel labels will not create more execution evidence. No blanket claim that conditional models are forbidden, or that they cannot help later.
- **Longer entry commitment:** selected as the bounded KEEP_INITIAL contrast above. It tests an actual action/continuation change while preserving structural safety and common accounting, without selecting a fixed retry timer. It requires a new freeze/multiplicity entry, **not additional USER approval** under the standing consumed-day development loop and this budget.
- **Queue value / `>= 0`:** current labels include queue benefits realized through fills before END; remaining queue priority at END has no continuation value (`pf_branch_engine.finish`). This audit does not measure that omitted terminal value. KEEP_INITIAL tests part of the within-episode issue; it does not solve valuation beyond END. Changing `>0` to `>=0` only changes exact-zero predictions and does not repair the failed positive predictions. No undeclared queue credit, extended horizon or tie-rule change in this experiment.
- **More label dates, untouched validation, expanded resources, or a strategic move to a different research line:** needs an explicit USER scope/budget decision. None is necessary for the selected consumed-day probe, and an untouched test of RICH/SMOOTH remains unwarranted. A later model candidate needs its own declared full-opportunity evaluation; eventual validation needs a frozen pipeline and at least five complete untouched UTC days.
