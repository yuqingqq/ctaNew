# MAKER ORDER-ESTIMATOR / LOGIC-CHAINS — independent review

Date: 2026-11-23 UTC. Full pinned target: **8d68808d4c5232f910d92808bef14871eb6ab1b5**.
Range: `6ddad3533..8d68808d4c5232f910d92808bef14871eb6ab1b5`.
Pinned source: `/tmp/pf-chains-review-8d68808d`. Three parallel static reviewers covered posting/estimation, pulls, and takes under the USER's explicit parallel-agent authorization. The root alone ran bounded checks in the codex slot. No implementation or frozen receipt was edited.

## Verdict

**Do not accept the headline decision logic as established.** Three material conclusions need correction: the unconditional short-rest proof, the direction of BTC movement relative to inventory at takes, and the interpretation of the primary pull control. The packet still contains useful, reproducible descriptive evidence; it does not need a full rebuild to correct these conclusions.

Most useful surviving clue: **adverse BTC movement is enriched around cancel-compatible terminal windows even under the declared fill-anchored comparison**, on its supported subset and on all four consumed days. This is evidence to investigate, not an identified cancel rule. Public observations still leave pull/exhaustion and repost/already-resting-ladder histories tied.

Population/as-of: the pinned 2026-11-23 receipt covers the ex-ante top three on 19 BOOK markets, Oct 24–27, 2026; 4,149 filled orders, 4,147 take records, 88 selected cases/62 connected episodes. Four consumed days; no independent-day interval, validation, fitted model or profit claim. This review used saved artifacts only, apart from synthetic controls.

## Consolidated findings — one development round

“BLOCKING” below means blocking the named behavioral conclusion or its use to design a rule. It does not mean that every saved count or every other finding must be rebuilt.

### 1. P1 — BLOCKING: a union of three timing scenarios is not a proof of short resting

**Evidence:** `tools/order_estimator.py:359-362,1042-1051`; `tools/chain_post_fill.py:549-560`; `pf_mt_lifecycle.py:13-18,478-513,627-640,737-738`; `v2/report_logic_chains.md:17-28,134` (tool paths are under `predictfun/reviews/maker_trace_20261122/`).

The estimator takes the earliest `post_lo` among S_FIFO, S_FIFO_LO and S_NOPRIO and derives `rest_ub` from that union. S_NOPRIO still fixes fills at the hi clock; S_FIFO_LO uses an evidence-refined lower clock which the lifecycle explicitly warns can be **later than the actual match under masked replenishment**. These scenarios do not exhaust all possible match times. The lifecycle separately computes a scenario-free hard bound and marks disagreement `assumption_sensitive`; the new classifier ignores that distinction and even upgrades any finite union to `B_constrained` in the detailed logic table.

Independent saved-row recount reproduces **576 late + 2,341 short = 2,917 scenario-short orders**, but only **957** of those have lifecycle `post_class=constrained`; **940** also have packet `status=ok` (940/4,149 = approximately 22.7%). Of the 2,917, **1,960** are assumption-sensitive and **413** carry capacity/clock/post-after-first-fill conflict flags. The 957 and 940 are conservative certification subsets, **not estimates that all remaining orders rested longer**. Even their bounds remain conditional on H1/H2/capacity premises and the declared maximum match lag.

A feasible countermechanism is an old order partially filling before the chosen scenario clocks, with replenishment masking its first drop. An intermediate frame can fit the true remaining quantity but not its total eventual fills. Scenario capacity then falsely forces a late post; the hard calculation considers only certainly-future fills and need not do so.

**Correction:** withdraw “at least 70% proven” and “not long-resting queue builders.” Preserve scenario widths as scenario estimates. Regrade the existing rows using the saved lifecycle class and clock/conflict statuses, with unknowns retained. The hard numeric bound is not separately saved for sensitive rows; do not invent it or spend on a broad lifecycle rerun just to force a narrower answer. Add a masked-fill/replenishment falsifier before this classifier is reused.

### 2. P1 — BLOCKING: the reported adverse-inventory TAKE association uses the opposite alignment

**Evidence:** `tools/chain_take.py:593-608,628-637,941,962,1121-1122`; `v2/report_logic_chains.md:96-105,137`.

The saved rank-1 **5.055% versus 2.686%** statistic is `tvc_r_5s_against`: BTC movement against the **take side**, not against inventory. For a sell that reduces a long position, rising BTC is against the sell but **with the long inventory**. The report reverses this interpretation. Pooling inventory-adding and inventory-reducing takes makes conversion by a simple global sign flip invalid.

**Correction:** retain the measured take-side statistic under its correct name and withdraw the inference that it demonstrates taking after adverse movement against inventory. A genuine inventory-aligned test would need the same alignment/status treatment in cases and controls, separated by reducing/adding/uncertain inventory. It is not necessary to launch that new measurement merely to correct this receipt. The existing own-fill-burst association is a separate result and is not invalidated by the sign error.

### 3. P1 — BLOCKING the claimed pull rule: the primary null admits fill-anchoring bias, and the declared countercheck was omitted

**Evidence:** `tools/chain_pull_selftest.py:290-299,508-515`; `tools/chain_pull.py:23,40-46,829-835,901-907`; `tools/chain_context.py:1253-1264`; `v2/report_logic_chains.md:67-81,136`.

The supplied synthetic test constructs **adverse fills followed by random pulls**. It explicitly requires the primary presence control to show an apparent association and the fill-anchored control to remove it. That test passes in this review. Terminal windows begin from the last fill's hard lower bound, and `_some` means the condition holds somewhere in the window; recent fills and inventory changes can therefore enter mechanically.

The report quotes only the primary comparison. The declared supplementary results already exist:

| S_FIFO, cancel-compatible, D=250 ms | Presence reference | Fill-anchored reference |
|---|---|---|
| Adverse BTC within 1 s | 34.62% vs 17.70%; +16.93 pp; n=2,715 | 28.34% vs 19.27%; **+9.06 pp; n=1,401** |
| Inventory built on order side within 5 s | 87.56% vs 71.02%; +16.55 pp; n=2,734 | 86.15% vs 78.80%; **+7.35 pp; n=1,408** |

These are different supported populations, not an exact same-row effect decomposition. Fill-anchored BTC differences by day are **+9.70, +9.38, +10.43, +4.17 pp** (n=501/312/407/181). Inventory differences are **+1.05, −7.77, +20.34, +22.24 pp** (n=501/319/407/181): a stable inventory rule is not established.

**Correction:** publish both declared control families, support and day breakdowns; describe the surviving BTC result as a terminal-window association. The reported p-values are conditional reference-draw tail fractions, not independent-day confidence or proof of informed cancellations. No full replay is required.

### 4. P2 — BLOCKING “fair order-level comparison”; NONBLOCKING when the actual reference populations are named

**PULL evidence:** `tools/chain_pull.py:557-604,821-823,855-907`; contract section 7.2 at `order_estimator_contract.md:323-326`.

“Matched moments when the order kept resting” is inaccurate. The primary mask is a **union of same-side orders' presence spans**, not a continuous span for the same order. A synthetic relay `[0,200] ∪ [200,400]` passes every sample in `[0,400]` although no one order covers it. The control uses the event order's price even when another-priced order supplies presence. Among 2,741 cancel-compatible events, saved controls contain **908 presence matches, 1,826 active-span fallbacks, 7 unavailable**. The reported primary means pool matches and fallbacks.

The exhausted reference is priority-conditional; its removal window is fixed by last-fill hard-low to hi, unlike the cancel-compatible window. Neither order age/remaining size nor remaining market time is matched. The reported 154 s versus 173 s remaining confirms a time mismatch. In the supplementary Oct 27 BTC comparison the exhausted enrichment is **+24.84 pp**, versus **+4.17 pp** for cancel-compatible endings; “exhaustion is weaker” is not a uniform discriminator.

**TAKE evidence:** `tools/chain_take.py:643-655,680-701,1127-1135`.

The implementation correctly uses equal window widths and the same feature function for cases/controls. However, only **3,977/4,147** take records use the full phase/sign/absolute-position-bin stratum; **125** drop sign and **45** drop inventory matching altogether. All enter reported estimates. Matching is pre/live phase, not remaining time. “Similar inventory” needs these qualifications.

**Correction:** expose reference membership and fallback counts; keep exhausted and cancel-compatible populations separate. A future order-level discriminator must bind the supporting order/cell/full span and comparable lifecycle time. Existing aggregated TAKE control tables do not preserve all per-take draws, so an exact-only or newly inventory-aligned null cannot generally be recovered by relabelling aggregate means. Do not require a rerun for this descriptive receipt; do not promote it as that unperformed test.

### 5. P2 — BLOCKING identified-command claims: the summaries drop the estimator's tied histories

**Evidence:** `tools/chain_pull.py:4-6,272-279,950-963`; `tools/chain_post_fill.py:697-710`; `tools/order_estimator.py:480-502,1068-1082`; `tools/chain_take.py:992-996,1023-1049`; `v2/report_logic_chains.md:55-61,79-81,107-145`.

- Every saved PULL row says `pull_identified=False`. The 2,741 primary cases are cancel-compatible endings, with exhaustion retained, not 2,741 known cancellations.
- A successor within 5 s is a later **first-filled same-side hash**, not a newly submitted replacement after cancellation.
- The 2,962 “reprice” and 577 “refill” order successors include **2,502 tied links** (2,223 and 279 respectively). The other 1,037 have sequential timing under declared scenarios; even those do not identify private intent or eliminate “unrelated.”
- Another same-hash fill observes continued execution, not a private KEEP callback; it can coexist with cancellation in flight.
- “Two-sided quoting” and ladder counts include feasible/timing-sensitive overlaps, not only identified simultaneous presence. First observed filled hashes do not enumerate unfilled posts.

**Correction:** carry evidence grades into every census and example: observed executions, constrained presence, scenario-dependent links, prior-driven estimates, unresolved alternatives. Show best-supported estimated histories as requested, but describe alternatives that change the logic. The report's final list of unidentified fields does not repair earlier assertions of those actions.

### 6. P2 — NONBLOCKING for correctly qualified inventory diagnostics: earlier excluded legs remain claim-eligible

**Evidence:** `tools/chain_take.py:139,410-425,925`; `tools/chain_context.py:593-622`.

Inventory is computed from status-ok legs. **248 takes** carry `excluded_before` yet remain claim-eligible: 107 reduce-partial, 93 add, 46 timing-sensitive and 2 no-change. The affected input cohort has **16 excluded top-three BOOK legs: 14 source `conflict`, 2 `off_grid`** (all have finite positive displayed quantities; that alone does not resolve source conflict).

**Correction:** qualify these classes as conditional on the retained ledger, or exclude them from identified inventory-mechanism claims while keeping status/counts. Do not silently treat all earlier exclusions as economically irrelevant or insert their quantities without resolving conflict. Agreement among recorded/certain/simultaneous variants is itself only agreement under those declared variants, not every feasible hidden inventory path or private notice state.

### 7. P3 — NONBLOCKING operational items, with bounded review workarounds

**Evidence:** `tools/chain_take.py:1330-1340`; `tools/chain_take_selftest.py:582-585`.

The TAKE receipt records output row counts but no output content hashes. This review recomputed its entire saved summary exactly from `takes.csv`, `take_controls.csv` and `actions.csv` and recorded their current hashes in `supplement.json`. This supports the inspected artifacts, not a retroactive freeze of every raw output byte. Bind the superseding output hashes in the next receipt.

The pinned TAKE selftest is **58/59 after commit**, not 59/59: its “untracked tool files” refusal relies on the actual checkout being unfrozen and therefore stops firing once the source is committed. In the detached checkout, a separate `relative_to(C.REPO)` metadata assumption first crashes the end-to-end section. A byte-verified main-path run gets past that metadata limitation and exposes only the environment-dependent refusal test. This is not evidence of bad market outcomes. Make the bad-state fixture explicit when next touching the suite; metadata portability can be deferred. No separate fix-only round.

## Accepted parts, prior closure, tests and limitations

- `f1e75e6f9` correctly supersedes the previous matched-gap inventory prescription and discloses simulated/recorded clocks, fills and fee conventions. That prior review's two interpretation findings are closed.
- Deterministic market-first case selection follows the frozen design and retains short populations. The estimator stores competing size/removal/link histories and priors separately; the principal problems arise in grading and interpretation, not a discovery that those alternatives were never saved.
- The own-order ground-truth result is an instrument check, not calibration of other makers' private clocks, original sizes, or all reconstructed fields. Its 85-covered/one-priority-miss description must keep that scope.
- Context uses venue/book/Binance clocks and explicit delay sensitivities. It is not identified maker-received information. Window context is sampled on the declared grid; retrospective pending-state reconstruction and opportunity masks are not live feature availability.
- Four pinned synthetic suites pass: context **173/173**, estimator **75/75**, post/fill **251/251**, pull **42/42**. TAKE behavior and writers pass in the byte-bound main-path check except the freeze fixture above (**58/59**). Tests do not demonstrate a profitable or inferred maker policy.
- POST and PULL output hashes match their committed receipts; all three committed summary CSVs match saved copies. Reviewed source hashes agree. TAKE summary recomputation is exact. [verify_saved.py](verify_saved.py), [evidence.json](evidence.json), [supplement.py](supplement.py), [supplement.json](supplement.json) contain row-level recounts, hashes, matching counts, reference tables, source-exclusion causes and the relay falsifier.
- Review-instrument failed attempts are preserved: a mixed numeric/`union` D column initially produced an empty filter; the relay helper needed its unused market argument; a temporary variable shadowed the regex module. There were also two shell invocation typos that ran no research code. These corrections changed only review tooling. No strategy/model/simulator/data-builder file was changed.
- No full context builder, label build, replay, raw tape experiment or fit was launched. All Python checks used the codex slot through the absolute `run_light.sh`; static subagents ran no computations. Review allowance **0.010 core-hours**, including failed attempts and commit verification. The task ledger before this review is **1.788861 h**, line **5.099561 h**, including the prior `6ddad3533` 0.010-h charge missing from the old helper. Packet development is **0.070861 h** before this review. Prior review allowances are not charged twice.
- Full-run evidence is adaptive development on consumed days. Real-data pilots were run before the final packet freeze and retained. The final source was frozen before the four full runs; this is not an untouched validation claim.

## ONE concrete next priority

**Produce one superseding saved-evidence logic-chain correction packet, within ≤0.025 core-hours all-in (including its review), bundling findings 1–7.** This is the completion of the current authorized packet, not a new model or strategy experiment.

1. Regrade the saved posting estimates using lifecycle hard/constrained versus scenario-sensitive/conflict statuses. Preserve unknowns; report the conservative constrained subset without asserting long rest for the remainder. Correct the detailed example grades and early-queue contradiction claims.
2. Publish both existing PULL reference families, their membership/support and per-day results. Keep the useful surviving adverse-BTC terminal-window association; withdraw identified pull timing, a stable inventory trigger, and an exhaustion discriminator that was not measured.
3. Correct TAKE side alignment and reference-match labels; carry excluded-before/uncertain inventory statuses. A new inventory-aligned null is optional future work, **not required** to make this receipt honest. Do not infer it from mixed aggregate controls.
4. Carry hypothesis grades into the action census and a few of the already predetermined 62 episodes. Present complete estimated post/partial-fill/terminal/successor/take paths with material alternatives and the already measured longer-horizon economics. No new sample, outcome-selected illustrative substitution or policy fit.
5. Bind the corrected artifacts and use targeted falsifiers for the two sign/grade errors, omitted-control presentation and observed-event versus inferred-command separation. Retain the original outputs. One combined acceptance request; defer cosmetic cleanup.

At the ledger snapshot, this review plus that entire follow-up gives approximately **0.105861/0.11 packet h**, **1.823861/1.85 task h**, **5.134561/5.25 line h**. Reconcile concurrent runs before launch. **No new USER approval is needed within this existing scope and allowance.** A lifecycle/full-context rebuild, new policy test, new days or a larger allowance is outside this priority and needs the USER if it exceeds the authorized scope/caps.

**Largest evidenced limitation:** the data can support useful historical context associations and bounded alternative histories, but the present summaries turn scenario choices, filled-order selection and tied terminal histories into stronger private-decision claims. Correcting that distinction is necessary before treating these chains as rules for our market maker.
