# Optimization loop: paired engine and exact v3 validation

As-of **2026-11-09T17:52:27.717211+00:00**. Sixth engineering pass: **238 wholly synthetic traces**, zero exclusions; **51 Python checks** and one Rust test with positive and corrupt-request/state controls. No market or protected Nov9–13 data was read in this pass.

The generated research engine `research/pf_step4_ladder_pairs.py` adds opt-in `LZ`, matching native `incremental_v1` paired-order sizing at pinned Rust **f20fdb1**. A new quote starts as one order. Same-price size growth keeps its queue position and adds the remainder; notified fills reduce an unchanged logical target, and pending partners keep it active. The frozen engine remains unchanged.

## Verification scope

| Evidence | Population | Result |
|---|---:|---|
| Component lifecycle differential | 32 explicit + 200 seeded synthetic traces | Native request and pair-state match |
| Complete named v3 differential | 6 synthetic traces, both sides × no-trade / reducing-fill / cap-fill | Native request and pair-state match |
| No-token complete returned outputs | 6 component + 6 complete-v3 cases | Byte-identical to frozen outputs |
| Ordinary fixed-size complete returned outputs | 8 component + 4 complete-v3 no-trade cases | Paired and single outputs byte-identical |
| Resize positives | 2 component + 6 complete-v3 cases | Resize fires and requests differ from single |
| Fixed-size rejection-stress controls | 2 mirrored component cases | Intentionally differ: rejected-create bug corrected |

Across all238 traces, the native oracle matched **8,998 requests**, **14,961 decision pair states**, **2,177 fill notices**, **219 rejections** and **683 retries**. Counts and known-fill ledgers reconcile to saved events. Native replays reproduce their saved reports byte for byte; generated exact-arm output hashes and traces reproduce exactly. Corrupt identity, count, ledger, output hash and native request fixtures are refused.

The complete named arm is:

`MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95`

Its16 Python checks include CAP10 suppression after notified inventory, RO60 reducing-only behavior, paired partial fills without unnecessary replacement, and fixed-size/no-token controls over full300s synthetic inputs. The native oracle consumes already-computed Python desired quotes: these comparisons independently check mapping and lifecycle behavior, not native prediction or guard calculation.

## Bug corrected, and the explicit parity exception

The initial native comparison matched78 traces, then refused `random/46`. Frozen Python counted a rejected create as unanswered until its nominal44ms answer time. Rust releases it immediately when rejection arrives at40ms. At the three-unanswered limit, Python needlessly blocked a replacement and imposed a250ms retry delay.

Only `LZ` now releases the unanswered slot at rejection. `lz_rejection_answer_release` records the correction. This necessarily changes fixed-size rejection-stress paths, so **blanket fixed-size parity is not established**. Ordinary fixed-size controls pass, and no-token behavior stays exact. The original B4 requirement for both native parity and universally unchanged fixed-size behavior conflicts on this existing bug; B4 is recorded with that exception, not marked as passing every original gate. The frozen source SHA256 is `d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17`.

## Corrections and reproducibility

`findings_v6.json` supersedes v5 in-band. V5's wording “variable-size v3 cases” was too broad: those component tests used JOIN/S1 with AM sizing, PM90 and ladder rules. The exact full-arm supplement above closes that coverage gap; the component evidence itself is unchanged. Both reports remain available.

The first supplement artifact verifier refused tuple-versus-JSON-list container differences. Canonical JSON comparison corrected only the verifier; unchanged traces and output hashes then verified. Its failed log and correction are retained in the supplement receipt.

Research branch `research/predict-opt-loop-20261109`, checkout `/tmp/cta-opt-loop-20261109`; source changes are isolated there.

Implementation source805d467a and declarationd0cd457b precede corrected component outcomes. Exact-arm sourcef4194393 and declarationa638b203 precede its outcomes. Receipt checker13de677b records regenerated artifacts and corruption controls. The paired implementation adds no strategy parameter sweep; the cumulative count remains21.

Receipts and source pins:

- `reviews/optimization_loop_20261109/paired_engine_v2/`: component declaration, receipt and verification; supersedes the preserved failed first attempt.
- `reviews/optimization_loop_20261109/paired_v3_arm/`: exact-arm declaration, receipt and verification.
- `reviews/optimization_loop_20261109/findings_v6.json`: current combined index; prior quote-continuity/latency results remain in the linked v4/v5 chain.
- Generate with `python3 -m predictfun.research.pf_ladder_pairs_build`; validation stages live in `pf_pairs_validation`, `pf_pairs_v3_arm`, `pf_pairs_receipt` and `pf_pairs_v3_receipt`. Run serially with `run_light.sh`. Existing declarations/artifacts refuse overwrite; use new destinations for a new attempt.

## Remaining limits and next pass

Historical parquet/file-output parity has not been run. The fill/queue model is uncalibrated; kernel local refusals, wallet feedback, cross-market venue limits, late-fill reservation deployment integration and actual50ms timer scheduling remain outside this check. No P&L, inventory-variance reduction or deployment readiness follows from these synthetic results.

All jobs in this pass are complete (last successful verification17:52:26Z). Peak recorded RSS was228MiB for the native build; exact-arm verification127MiB. Research slot is free; recheck before C1 or another job. Live files, frozen engine and dollar loss budget were not changed.

Next: combine bounded50ms quote continuity with paired fills and inventory changes in synthetic lifecycle tests, including hard cap/reduce-only preemption and no timer extension. Calibrated fills or authorized forward evidence remain necessary before economic selection.
