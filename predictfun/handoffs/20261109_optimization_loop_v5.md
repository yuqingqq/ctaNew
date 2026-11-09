# Optimization loop: paired research engine — 2026-11-09

As-of **2026-11-09T17:42:57.198365+00:00**. **232 synthetic traces** (32 explicit,200 seeded), zero exclusions;35 Python checks. No historical/forward market data was read and no performance result is asserted.

The generated `pf_step4_ladder_pairs.py` adds opt-in `LZ` incremental paired orders to frozen v3. A new quote is one whole order; an increase preserves the existing member and adds the remainder. Notified member fills are deducted from the unchanged logical target, and a waiting partner keeps the logical order active. Physical order slots are fixed to the four policy lanes and their partners at +4.

## Native comparison

The pinned Rust f20fdb1 request mapper and feedback methods match all **8,936 requests**, **14,637 decision pair states**, **2,167 fill notices**, and **683 retry events** in the declared synthetic traces. Checks also cover 219 rejections, 70 throttled second-member attempts and 11 minimum/size withholds.

The native oracle consumes the research decisions and feedback. It verifies order mapping, notified-fill accounting and full-fill repost obligations; it does not independently re-estimate the model, calibrate market fills, or validate kernel/wallet scheduling. The saved native report is reproduced byte for byte. Corrupt request size, artifact counts and scenario identity are refused.

## A real frozen-handler mismatch found and corrected

The first oracle attempt matched78 traces, then refused random/46. Python kept a rejected create in the unanswered-request count until its nominal44ms answer deadline. Rust removes it immediately on rejection at40ms. With the three-unanswered limit already reached, Python unnecessarily withheld a replacement and scheduled a250ms retry. A later size-up therefore started from a different order.

`LZ` now releases that unanswered slot when the create is rejected. The frozen file is untouched. Two mirrored fixed-size rejection-stress controls demonstrate the intentional difference; a counter `lz_rejection_answer_release` makes it visible. The failed declaration, traces, partial native output and log remain preserved; `paired_engine_v2` supersedes them in-band.

**This prevents a blanket fixed-size parity claim.** The original B4 spec asks for unchanged fixed-size output and native parity; those requirements conflict on this pre-existing rejection bug. The corrected implementation prioritizes native lifecycle semantics only under `LZ`, records the exception, and retains exact frozen behavior without `LZ`. It is not a claim that every original B4 gate is satisfied.

## Parity evidence and limits

- Six no-token cases have byte-identical complete returned outputs against frozen v3. Eight ordinary fixed-size cases match paired/nonpaired outputs exactly. Two variable-size v3 cases produce resize events and differ, as required. Full output hashes are saved; two additional fixed-size rejection cases intentionally differ for the correction above.
- Tests include partial and whole pair fills, late cancelled-order fills, front-member rotation, pending creates, concurrent replacement, minimum notional after lot rounding, short residuals, reduce-only shrinkage, chronological paired fills and unsupported-mode refusals. Tracing does not change the returned results.
- The no-token and fixed-size comparisons are source-only synthetic function-output tests. Historical parquet/file-output parity was not run. The existing fill/queue model remains uncalibrated, so simulated profits are not presented as optimization evidence.
- Kernel local-refusal/capacity behavior, the deployment late-fill reservation window and actual50ms timer scheduling are outside this B4 check. Native and research guard/prediction integration is not independently validated by feeding already-computed desired quotes.
- Frozen v3 remains SHA256 `d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17`. No live deployment, dollar loss-budget change or protected Nov9–13 read.

Sources805d467a and declarationd0cd457b precede corrected outcomes. `LZ` is implementation work, not another strategy parameter sweep; the prior cumulative count stays21.

Build: `python3 -m predictfun.research.pf_ladder_pairs_build`. Validate with `pf_pairs_validation` stages and `pf_pairs_receipt`, under `run_light.sh`; declarations and destinations refuse overwrite. The research loop checkout is `/tmp/cta-opt-loop-20261109` on `research/predict-opt-loop-20261109`. Receipts: `reviews/optimization_loop_20261109/paired_engine_v2/`; combined index: `findings_v5.json`.

Next: combine the bounded quote-continuity policy with this paired engine in synthetic lifecycle/inventory tests. Economic claims remain gated on calibrated fills or authorized forward evidence.
