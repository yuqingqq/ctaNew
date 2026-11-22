# FEASIBLE-CANCEL-HEADROOM-v1 — combined review

2026-11-22. Full pinned target: **438ce3a889e8f2b27b5622315e0454ce949a4af9**.
Range: **60768c83cae4b9da439e943f45a32e4fed4c3551..438ce3a889e8f2b27b5622315e0454ce949a4af9**.
Reviewed freeze `2a2f78da` and results `438ce3a8` in detached checkout
`/tmp/pf-ch-review-438ce3a8`. Development owns implementation. This review
changes no strategy, engine, development test or frozen receipt; no push.

## Decision

**Accept the saved-path numerical diagnostic. No result-blocking defect or
required rerun.** Correct the interpretation below in band. The proposed
long-continuation pilot is useful, but its continuation and classification
contract needs the bundled amendments below before its freeze. Those are
requirements for the new experiment, not reasons to repeat this diagnostic.

This is hindsight deletion accounting on **19 complete markets on consumed
Oct 24–27**, as of the pinned target. The two arms remain separate. Original
coverage was 19 complete, 54 missing endpoint and 23 stale future book out
of 96 markets. Four consumed days and selected supported paths provide
neither untouched validation nor an estimate of full-day strategy returns.

| CURRENT_ASYNC_FAST10, c | Recorded path | Same +1 s oracle cut |
|---|---:|---:|
| Eligible +1 s value | -78,485.90 | +52,682.23 |
| END value | -68,319.95 | -55,252.46 |

The 1,254 cuts remove 131,384.63 c of gross negative +1 s value and 216.51 c
of positive +1 s value. Of them, 1,129 are at the first callback, 40 ms after
activation. The same cuts improve END by **13,067.49 c**, but worsen Oct 25
by **37,262.93 c**. The removed common cohort is **+19,972.36 c at END**.
These statements reconcile. They use different cohorts from the full END
comparison, and the cut was optimized for +1 s only.

The result establishes short-horizon hindsight headroom. It does not show
which orders can be identified in advance, how replacing them changes the
book, or whether early losses pay for later use of a retained queue.

## Consolidated findings and disposition

1. **P1 — BLOCKING for the proposed queue-value pilot as specified; no effect
   on the accepted headroom numbers. The two continuations differ in their
   holding policy.**

   `predictfun/reviews/ladder_long_continuation_contract_20261122/contract.md:44-48`
   exempts the retained generation from future pressure pulls, while nothing
   in the alternative arm is exempt. Thus a newly posted replacement can
   immediately be pulled by the old guard. The measured difference would
   combine retained priority with a different future holding policy. It
   cannot be presented as the common-continuation KEEP/PULL or POST/WAIT
   target required by `c590b083` and `60768c83`.

   **Bundled resolution:** use one declared retention rule for the focal
   slot's continuation in both arms, including its replacements. Define the
   focal lineage causally by side and birth rung, assigning new gids when
   they are submitted; never transfer priority or fills between gids. After
   the single forced action, pressure-withholding at admission remains the
   pinned rule in both arms, but pressure-pull exemptions apply identically
   to live orders in that focal lineage. Structural and inventory removals,
   caps, sizes, acknowledgements and price locks remain active. Every other
   slot follows the pinned pressure policy with its own branch's feedback.
   A replacement gets this same holding rule and its actual new queue.

   This is an explicit diagnostic continuation, not an approved trading
   strategy. It tests the marginal value of retaining versus rebuilding a
   queue under that continuation; it does not establish globally optimal
   two-sided quoting. Omit the optional all-orders CONT-REST arm from this
   small pilot. “Run if cost allows” is not a precise population or budget
   for a second experiment. No separate repair-only review is needed.

2. **P2 — BLOCKING for the proposed mechanism labels; define them before
   the pilot.**

   Contract `:83-86` calls `Y > tol` plus an early-loss/later-good sequence
   “compensated queue continuity.” Full-book Y can instead be driven by
   opposite-side fills or inventory feedback. The sign and sequence do not
   identify queue necessity. “Early,” “good,” “enough later value,” and the
   tolerance are not yet executable definitions. Also, a profitable retain
   arm with no early adverse fill has no category in that table.

   **Bundled resolution:** keep two separate axes. The exhaustive economic
   axis is retain-better / alternative-better / tie, plus invalid statuses,
   using full-book Y and a frozen accounting tolerance (1e-6 c is adequate).
   The sequence/mechanism axis records all cases, including no early loss,
   no later benefit and no fill. Define signed per-fill contributions at
   the common END for its economic sequence; retain fill-time and +1 s
   markouts as separately named adverse-selection diagnostics. Do not
   silently use short-horizon signs as long-horizon economic labels.

   Reconcile Y into focal-lineage and other-order contributions; record
   replacement delays, prices, queues and inventory constraints. Use
   “retention beneficial with an early loss” where warranted. Reserve a
   claim of queue-dependent compensation for evidence distinguishing queue
   loss from absence, changed prices, sizes and other-order feedback;
   otherwise record mechanism-unidentified while retaining the valid Y.
   An economically tied pair is not missing or censored data.

   The implementation must capture the actual pre-command state, not just
   the pre-grid state. Contract `:55-58` rolls to before tick k, whereas
   `pf_branch_ladder.py:223-241` advances trades/notices and executes sides
   in alternating order inside that tick. Replaying this common prefix is
   valid, but apply the intervention exactly once at the identified callback
   and fingerprint its actual context. Include a second-side/same-tick
   notice-and-command fixture. For absolute Q, account for the inventory
   already carried at the fork and only post-fork fills in the incremental
   decomposition (`contract.md:11-18,67,72`); the common starting wealth
   cancels in Y. Engine inventory is for accounting, client-known inventory
   for policy inputs. This is a specification clarification, not an observed
   simulator defect.

3. **P2 — NONBLOCKING for the numbers; correct the headroom interpretation
   before it is reused to choose an admission rule.**

   `predictfun/reviews/ladder_cancel_headroom_20261122/report.md:133-136`
   moves the chosen-cut classification to the first review. The **1,238**
   CURRENT pure suffixes are pure after the oracle's chosen callback;
   **1,114** are pure at the first callback. The other **124** are initially
   mixed: **53 compensated** and **71 uncompensated**. Their later cut
   preserves earlier good fills. FRONT has the corresponding distinction:
   1,560 pure at the chosen cut versus 1,241 at the first callback.
   This follows directly from the independently reproduced `a_cls`, `b_cls`
   and cut fields (`pf_ladder_cancel_headroom.py:265-268`; evidence.json).

   Replace “only bad fills after their first review” with “only bad fills
   after the chosen cut”; report first-review cases separately. The 90%
   first-cut statistic is correct. It supports examining admission and
   first review, but does not make all the economic decisions admission-only.

   Report `:90,141` and the commit headline also call the END result a
   smaller “bound.” Here **+13.1k c is the END gain of the +1 s-selected
   cuts**, not an END-optimized upper bound; the resulting CURRENT END is
   **-55.3k c**, not +13.1k c. The source correctly fixes the cut across
   horizons (`:166-177,211-217`). Use that precise wording. Do not optimize
   another oracle just to fix the description. The 629 c bad-then-good
   sequence total in report `:115` covers the 27 feasible sequence cuts,
   not all 29 flagged sequences; keep that denominator explicit. These
   corrections can accompany the next combined receipt, with no replay.

4. **P3 — NONBLOCKING; checker coverage remains narrower than the prose,
   with a verified workaround for this receipt.**

   `ch_checks.py:231-272` checks selected table fields, and `:291-298`
   checks case membership rather than its timeline. A 10,000 c corruption
   of summary `b_v30` passes `S_summary`; a changed case timeline passes
   `C_casebook`. These are named-family tests, not claims that an unmodified
   hash manifest accepts changed bytes.

   **Workaround completed:** independently recomputed primary economics,
   oracle choices, statuses, common-cohort values and panel aggregates; also
   reproduced every column of all 12,135 generation rows and the summary,
   classes, markets, sequences and casebook tables from the pinned builder
   and saved source records, including timelines and these ancillary fields.
   Raw output hashes and committed copies match. Actual values are correct.
   Extend named comparisons with the demonstrated corruptions when the fields
   next drive a consequential decision. Do not start a cleanup-only round.

## Resolved versus open prior items

- ASYNC review `3e84ca6d` P2 is resolved by
  `ladder_async_fast_20261122/report_interpretation_v2.md`. The correct
  eligible-markout / subsequent-movement / late-fill END identity agrees
  with the prior independent reconstruction on all 76 market-arm rows.
  Timing is appropriately described as a combined execution-profile change,
  and the unsupported claim about all decision-time information is withdrawn.
- The `6d200317` classification amendment is implemented: complete statuses,
  both anchors, positive/negative suffix costs, overlapping sequence flags,
  known versus venue queue fields, common/native horizons and deterministic
  examples. X matches `c72f0d37` per generation on all 12,135 rows.
- `90e82f8e` whole-life mechanism remains **unidentified**, correctly
  disclosed. Original cancellation truncates the observed continuation.
- `c590b083` / `60768c83` long-value and joint-inventory direction is adopted
  in the design, subject to findings 1–2. It has not yet been run.
- Inherited ancillary checker gaps remain deferred with the above verified
  workaround. The deleted first-draft mutant logs cannot be independently
  inspected; their declared 30 s allowance is counted. Final-source 38/38
  records are intact and re-judged here. No numerical rerun is justified.

## Verification and limits

Only bounded checks through `PF_SLOT=codex` and absolute script paths:

- Pinned eight-group synthetic selftest passes; 38 final mutant records
  re-judged against the pinned source/selftest/harness identities. No full
  mutation rerun or tape replay.
- Checker families A/B/E/O/Q/H/R/S/K/C/P/I/X pass on the saved artifacts;
  all 19 falsifier cases fire. Source closure is unchanged from the freeze
  for the 11 checked files. The pre-freeze synthetic rehearsal and its
  stand-in X are identified as synthetic, not empirical reproduction.
- 38 arm-markets reproduce accepted eligible +1 s and END values. Every
  order, fill and callback population reconciles. Fill/effect boundary is
  strict: fills at d+10 are removed, earlier/request-tick fills retained.
  KEEP wins ties, otherwise the earliest qualifying optimum is selected.
  Incomplete valuations remain statuses; their missing +1 s components do
  not become evidence of zero economic cost.
- All saved numeric marks are finite when present; 5/30 s marks past END
  are missing. Source queue fields and notice-delayed remaining quantities
  remain separated from hindsight venue quantities. Same-price re-entry
  keeps a new gid. All T1–T7 readouts and the ASYNC identity outputs reproduce
  byte-for-byte.
- `verify_v1.log` is retained: after the independent checks passed, the
  review's expanded comparison stopped on different row ordering. The
  corrected comparison sorts by exact keys; it passes. This was a review
  instrument error, not a source/result discrepancy. It costs 38.21 CPU s;
  the completing run costs 11.49 CPU s. Peak 1,239 MiB. Both count toward
  this review's 72 CPU s reserve. Commit verification is small and included
  in that reserve. Prior 3.01 s classification logs are **not counted again**.

No causal predictor, new day, label fit, strategy replay or profitability
claim. Even the local oracle fixes other orders and ignores inventory
feedback; it is not an upper bound on feasible market-making policies.

## ONE next priority — a 64-pair long-continuation pilot, after resource approval

**Implement, freeze and run LONG-CONTINUATION-PAIRED-VALUE-v1 with findings
1–2 folded in; bring its controls, accounting and results back together.**
Do not train an entry/cancel predictor or restart the stopped pressure arm.

Concrete preparation/acceptance contract:

1. Use CURRENT_ASYNC_FAST10 and the same eight consumed pilot markets.
   For REVIEW, sample the **first eligible recorded live KEEP of each
   generation**, two distinct gids per market × side by the frozen hash.
   For ADMISSION, sample two distinct recorded POST decisions per market ×
   side. This gives up to 64 pairs; enumerate short cells and statuses.
   Freeze the full eligible census, keys, draw probabilities and hashes
   before any paired outcome. Do not sample by oracle class or repeat
   long-lived generations through a tick-weighted pool. Keep the two kinds
   separate; any overlap of their generations is declared, not independence.

2. Apply the single forced action at the exact matching callback. Continue
   both arms under the common focal-lineage rule in finding 1 to their same
   supported absolute END, with own-branch fills, queue, reservations and
   inventory. Record the canonical starting state/mark and every status.
   Do not erase outstanding requests or unnotified fills at the fork.

3. Require no-op factual parity and same-action pair parity, an unused
   exemption control, an actual replacement-queue reset fixture, future-only
   perturbation, exact command application, per-event reservation checks,
   and full cash/inventory reconciliation. Add known-bad controls for an
   alternative-only pressure pull, inherited replacement priority, a reused
   arm state, wrong command cursor and double-counted pre-fork inventory.

4. Primary outcome is paired full-book END Y and both absolute Qs; explain
   it by focal-lineage versus other orders, execution edge versus inventory
   movement, matched turnover, lot holding times, time-weighted/peak/terminal
   inventory, queue reset, participation and churn. Report the same 5/30/60 s
   diagnostic pairs where supported, with common/native counts. Keep short
   adverse markouts and long economic signs separate. Publish exhaustive
   economic classes even when the mechanism remains unidentified.

5. Time an initial small, predeclared subset within the pilot allocation.
   Project the remaining pairs using measured roll-in and continuation cost,
   including tests and review; stop if it cannot fit. The contract's 0.5
   remaining-lifetime assumption is a projection, not a measurement of these
   forks. Do not reduce sample or change a rule after seeing economics.
   The pilot decides whether this target is reliable and economically varied
   enough to justify later discrimination work. It cannot demonstrate skill
   or justify model fitting automatically; a thin result starts no tuned
   variant.

**USER decision required:** approve this one pilot batch at **up to 0.40
additional core-hours including controls and a 0.03 h review reserve**
(approximately 0.36 h paired pilot + 0.01 h controls), and raise the ladder
line ceiling from **2.50 to 2.85 core-hours**. The present accepted line is
2.4189 h before this review and below 2.4389 h with its full reserve; this
pilot does not fit the existing ceiling. These numbers are a proposal, not
authorization. No new data, untouched validation, live use or full training
is included. This resource approval is required by the user's explicit
current ceiling and the submitted contract, not by a new review hurdle.

**Largest evidenced limitation:** the excellent +1 s hindsight deletion
result does not value a realizable longer-lived, two-sided continuation.
The same deleted fills can gain value later, and the existing policy
truncates both future fills and queue use. The amended paired pilot directly
addresses that limitation; another short-horizon cancel variant does not.
