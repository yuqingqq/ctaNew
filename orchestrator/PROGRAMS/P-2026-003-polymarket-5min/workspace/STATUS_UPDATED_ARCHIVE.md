# STATUS.yml `updated:` — archive

**Append-only.** Entries moved out of `STATUS.yml`'s `updated:` field when it
was ruled a rolling window of the newest three (coordinator ruling,
2026-08-28T10:02Z; MEM proposed, did not execute unilaterally, because other
seats read that field and the seat that accumulated it is the wrong seat to
decide alone that its own accumulation should be trimmed).

**Nothing here was deleted — only moved.** Each batch below records when it was
archived. Entries within a batch keep their original order (newest first),
separated by the `PRIOR:` markers they carried in the field. If an automated
reader turns out to need the full chain, it is here and in git history; that
recovery path is why moved-never-deleted was the design.

---

## Batch 1 — archived 2026-08-28T10:03Z (13 entries, from the field at `1bc65cc`)

The field had reached 255 lines / 18.7 KB with 15 PRIOR generations, 10 of them
written in a single day; a cold reader met `flags:` at line 261 of 3225.

**Exact reconstruction of the field as it stood at `1bc65cc`:** take the kept
entries from `STATUS.yml`'s `updated:` (with the `ARCHIVED:` pointer stripped),
join with `" PRIOR:"`, then append this block's body. Verified byte-for-byte at
the split before the move; the only normalisation was a single trailing space at
the split boundary, which a YAML folded scalar does not carry meaning in. **No
content differs.**

```yaml

  2026-08-28T09:48Z (MEM): R-246 + R-247. LANE-4 PARITY BATTERY delivered and
  verified in the coordinator's own run (14/14), with "bit-identical" now
  DEFINED over replay_traj_canon_v1 -- an unpinned signature rule would make
  every valid comparison fail indistinguishably from a real difference, so the
  canon removes representation noise ONLY and floats serialise by repr with no
  tolerance. The arm NAME is excluded from the canonical bytes and that
  exclusion is asserted as its own check, because including it would make the
  anchor pass nothing and fail nothing. SIGNED ZERO STAYS UNNORMALISED by
  ruling: normalising it would be a tolerance by another name in the one
  comparison whose whole value is admitting none, and a signed-zero difference
  between real arms is INFORMATIVE -- it betrays a different computational
  route to the same zero. DA's standard adopted: A REGISTER ENTRY CITING A
  PROPERTY OF MY CODE SHOULD HAVE A CHECK BEHIND IT -- rule 15 pushed up a
  level, the register being an instrument too. The idle-dispatch narrow reading
  is now REGISTER-BACKED at R-247(1), verified in the register before the
  upgrade; the superseded provenance line is kept as the record of how it was
  sourced. Tonight's runbook PRE-CHECKED ~14 h ahead: tree holds clob_v3_1,
  HEAD holds clob_v4, unit active. PRIOR:
  2026-08-28T09:44Z (MEM): R-243..R-245. DA's FAIR-PRICE IDENTITY ARTIFACT
  delivered and verified by the coordinator's own run (21 selftests), with the
  2B protocol draft RIDING THE BATCH so Codex reviews it BEFORE the user is
  asked to freeze it -- the 011 lesson applied forward, and the reasoning is
  the durable part: an amendment to a frozen document costs a USER ACT, a
  review comment on a draft costs nothing. The second 2B challenger is NAMED
  BEFORE the review (Binance USDM bookTicker mid; tape verified at the data;
  era floor = the hf_ws_v2 ledger boundary, pre-boundary instants INADMISSIBLE
  rather than merely noisier), and BOTH readings of a would-be win are
  PRE-DECLARED AND BOTH KEPT: DA's (closer to the settlement source, so a
  positive increment is venue lag, NOT alpha) and the coordinator's complement
  (lag is exactly what a fair-price successor exists to capture -- decision
  value). What is excluded is choosing the story after the sign; the
  admit-vs-one-challenger call is the USER's at freeze time. Two standing USER
  directives: the aggregate RESOURCE CAP is binding and verified live at the
  instrument (18.4G / 1200% research.slice), enforced AT CLEARANCE TIME rather
  than discovered at OOM; and IDLE SEATS TAKE THE NEXT ADMISSIBLE PLAN JOB --
  a COORDINATOR dispatch rule, not a licence for a seat to self-dispatch
  outside its surface. DA is now on the lane-4 parity stub battery, building
  the CHECKER while BE's arms stay the CHECKED. Gate model restated: fixes
  verified is not HOLD RELEASED. PRIOR:
  2026-08-28T09:31Z (MEM): R-242 -- THE USER FROZE AMENDMENT A1, and the
  sequencing is the point: the Q4 algebra flaw was DATA-INDEPENDENT, so it was
  corrected by a user-frozen amendment BEFORE any 011 number existed rather
  than argued about after one. §A1.1 OPTION 1 (separate p_positive/p_negative,
  exact under zero mass); the Holm DENOMINATOR IS FIXED AT 24 with unevaluable
  cells OCCUPYING THEIR SLOTS, which is what closes the shrinkable-family
  blocker; PERM_SEED 20260828 with SORTED-KEY consumption, so blocker 7's
  determinism lesson is carried into the new design instead of staying a
  one-off repair. MEM verified the frozen preregistration is BYTE-UNTOUCHED
  since 3b71d3e (empty diff): the amendment is a separate document, rule 13
  held. BE is cleared into A1.8 steps 2-5 red-first, but CLEARED TO BUILD IS
  NOT CLEARED TO FIT -- the fit/score hold lifts only on Codex's HOLD RELEASED.
  The USER also directed the hazard line to proceed, so DA is dispatched to the
  fair-price lane: typed Identity artifact plus a 2B challenger protocol as
  DRAFT-FOR-USER-FREEZE, with NO challenger scoring until frozen. The R-239
  reviews/ directory now holds its first filing, so the collision that hit my
  last sweep has a home. MEM's seat-handoff identity check is RATIFIED as a
  standing pattern binding future MEM sessions. PRIOR:
  2026-08-28T09:24Z (MEM): SWEPT R-236..R-241, and the shape of the gap matters
  more than its size. THE USER DESIGNATED CODEX AS SYSTEM REVIEWER (R-238); a
  standing protocol now governs -- build, commit+push, Codex reviews,
  coordinator verifies every claim BY EXECUTION, fixes land red-first,
  re-review, then proceed, ONE round per COMPLETED batch. Its first pass put TWO
  THINGS ON HOLD. Iteration 011: NO FIT, NO SCORE; five blockers, the first a
  Q4 ALGEBRA FLAW that is data-independent and is therefore being fixed by
  superseding amendment BEFORE any number exists (BE drafts, USER freezes,
  sequence step 1). Day-bar v2: MUST NOT judge 08-29 until repaired AND
  re-reviewed -- DA's five re-review blockers are CLOSED at f8581b6 and
  coordinator-verified by execution (suite 63), but THE HOLD REMAINS IN FORCE
  and releases only on Codex's explicit HOLD RELEASED. Fixes verified is not
  hold released. The prereg's P3 grounding column was corrected in-band
  (283.2/258.9) because Q-DA-115's implementation-vs-table match was AGREEMENT
  BETWEEN TWO RUNS OF THE SAME DEFECT, not validation; DA filed that against
  itself unprompted and adopted the rule that a filing may not say VALIDATED
  unless the entry point was exercised the way its launcher invokes it. Race
  accrual is now governed by the FREEZE-COMMIT epoch with day quality split from
  accrual, after DA caught a half-true ruling; 08-28 reports ACCRUES=False. O1's
  two deploy conditions were BOTH met ~13.5 h early, so the boundary deploy is
  ON for 00:00:00Z and 22:30Z is demoted to a confirmation check. MEM verified
  freeze receipt v2 at the artifact rather than from its commit message: the
  citation-correction block anchors race_clock_start_commit b3f7f9f and states
  outright that reading v2's date as a new freeze "would hand the candidate days
  it did not earn". The two reviewer-authored flags in this file are left
  VERBATIM; superseding state sits beside them, never over them. PRIOR:
  2026-08-28T08:46Z (independent pre-fit review): ITERATION 011 FIT HOLD; no
  011 result artifact exists. Pre-number blockers were found in the zero-mass
  estimand algebra, fail-open target construction, generator population build,
  outcome fence, metric alignment/domain checks, per-row Q4 prediction
  alignment, generation/action weighting, 24-cell null mapping, and standalone
  result provenance; the bac5469 output guard also makes --selftest print GREEN
  then exit nonzero for correctly producing no result. Day-bar v2 is also HOLD
  for the 08-29 judgment: P1/P2/P3 do not govern all_pass; an elapsed empty
  ledger passes; structural bad rows
  and gap_open_at_exit are ignored; CLI breadth rendering uses removed keys;
  and the default freeze epoch predates the btc freeze by 3.63 days. O1 stays
  staged at v3_1; its new paths need behavioral tests and gap_open_at_exit must
  feed the day bar. Freeze receipt v2 and the canonical determinism repair are
  internally consistent. PRIOR: 2026-08-28T06:56Z (MEM): BLOCKER 7 CLOSED, AND IT CLOSED BY FINDING SOMETHING
  (R-234 8da983e ruling / R-235 7ec5f4e close). The increment-null re-binding
  surfaced a real determinism defect: increments bit-identical, but 11 of 12
  p-values moved on the same seed and data, because sign order came from
  set iteration under an unpinned PYTHONHASHSEED -- PERM_SEED pinned the RNG,
  not the data order it was applied to, so every run was an independent MC
  draw wearing the appearance of exact reproducibility. Repaired canonically
  (sorted + pinned) with acceptance PRE-COMMITTED SIGHT-UNSEEN; canonical
  survivors unchanged (btc LGBM @5% Holm 0.00600, @10% 0.03298), which DA
  correctly insists was NOT knowable beforehand and is NOT why the run is
  accepted. Also explained after six unexplained appearances: the ~1e-11
  verifier delta is non-associative float addition (row order vs
  score-descending), and the orders deliberately STAY DIFFERENT to preserve
  cross-check independence. Blocker 6's mechanism is proven on real sidecar
  bytes with wiring scheduled to ride the first lattice-touching 011 commit;
  section 0.1 is 6 of 7. MEM flagged one residual for its owners: the freeze
  receipt quotes the superseded p-values and resolves its null to the
  superseded commit -- verdict-bearing statements all still hold, but the
  resolvable field points at e7caaeb, not the canonical 163bd36. PRIOR:
  2026-08-28T06:25Z (MEM): R-232 EXECUTION CLOSED at R-233 (166679c). DA's
  both-coin verification of v2.3 is closed (Q-DA-113, 846e1ca): 15/15 at the
  commit fd1e949, btc worst 1.273e-11 / eth 3.638e-12, and the FIRST
  verification self-attesting on runtime identity -- it names its own
  feature-code bytes and refuses wrong-tree modules, so the verifier proves
  which code it ran instead of hashing a repo copy beside it. The Q-DA-79
  caveat is back in the receipt (cd23ebd) with BINDING_STALE correctly marked;
  that is the THIRD hand application and the count is the argument for BE's
  merge, which is next. Everything for tonight is now staged: freeze live with
  its clock running, O1 held at v3_1 for the 00:00:00Z boundary, O2
  pre-registered and amended, 08-28 judged at 00:06Z under the old bar. PRIOR:
  2026-08-28T06:15Z (MEM): FIRST FROZEN CANDIDATE. The rule-12 freeze receipt
  landed at b3f7f9f -- LGBM_PINNED, btc-only, MARKED UNVALIDATED, multiplicity
  1, race clock running from the freeze commit against a bar of 5 later
  complete passing btc UTC days. Verified at the artifact: all three btc
  budgets carried (@15% is Holm 1.0000, indistinguishable from chance), eth
  negative at every budget, and the null's cluster unit disclosed as WEAKER
  than rule 8's ruled unit (window, not UTC day, because G=0 leaves the ruled
  unit with no replicates) -- optimistic p-values, evidence not a certificate.
  The programme now has something whose clock is running, which changes what a
  lost forward day COSTS. PRIOR:
  2026-08-28T06:09Z (MEM state sweep; clock read as a separate command per
  R-214): STATE FILES BROUGHT CURRENT THROUGH R-232 AND RECEIPT v2.3. They had
  stopped at R-228 / d506a06 -- R-229, R-230, R-231, R-232, the O1 collector
  package, the O2 day bar and iteration-011 appeared ZERO times in either
  STATUS.yml or HANDOFF.md before this write. LANDED SINCE: receipt v2.3
  (fd1e949) from fit7/score7 at e12e2c7 -- the SIXTH consecutive numerically
  identical generation (1,046 leaves compared, max abs delta 0.000e+00; sole
  differing leaf da_caveat_field, predicted before the run), and the
  population/reach disclosure is GENERATOR-OWNED for the first time. Read at
  the artifact for this entry, not from a report: population_and_reach =
  label da_development_topup, G_complete_utc_days 0, is_a_validation false,
  intervals_claimable false, dates_present [2026-08-25], 611,343 rows,
  span 14.41 h -- COMPUTED from the rows actually scored against rule 11's
  bar, which closes R-229's top debt by mechanism rather than by hand
  re-attachment. val_models.json {btc true, eth true} now sits INSIDE the hash
  lattice (14 file_hashes, up from 13), so score7 REQUIRED both val models
  instead of tolerating their absence. Six fits agree
  (ef9b775 / 19b0611 / 43f777d / 97b7183 / e12e2c7). UNCHANGED BY ANY OF IT:
  development population, G=0, NOT a validation -- R-225/R-228/R-230 hardened
  PROVENANCE, not reach. IN FLIGHT at this write: DA both-coin verification of
  v2.3, then the rule-12 LGBM freeze receipt (R-232(3): btc-only, MARKED
  UNVALIDATED, multiplicity = 1 recorded at freeze). TONIGHT: collector v4
  deploys at 2026-08-29T00:00:00Z under runbook cb85ebd; the 00:06Z per-coin
  verdict judges 08-28 under the OLD count bar; day-bar v2 (dfa0977, amended
  368345b) governs days >=2026-08-29 only. STATE-FILE OWNERSHIP moved to the
  MEM seat by coordinator standing division this session -- see the
  state_file_ownership flag for the CLAUDE.md conflict that is not yet
  reconciled. PRIOR:
  2026-08-28 (coordinator): harmful-fill programme plan updated for the next
  conditional-research cycle. Added separate conditional signed-value,
  timestamped fair-price successor, frozen-skew and common action-value replay
  lanes; seven-arm integration ablation; full lifecycle metrics; reliability
  seam blockers; and >=5 later complete UTC-day validation. Documentation only:
  no candidate frozen, fitted, scored or promoted. PRIOR:
  2026-08-28 (BE): R-228 chain CLOSED. Receipt v2.2 (c47eb83) from
  fit6/score6 at 97b7183; replication BIT-EXACT vs v2.1 (1,046 shared
  leaves, max abs delta 0.000e+00; sole differing leaf da_caveat_field,
  predicted). Fit manifest + parity committed (ff80ebd). Five fits agree.
  Battery 476 checks, 0 failing. Freeze remains WITH THE USER.
  PRIOR: 2026-08-28 (BE): R-225 enforcement chain CLOSED. Receipt v2.1 committed
  (2fbf233) from fit5/score5 at 43f777d under a guard-ENFORCED provenance
  chain; numbers IDENTICAL to the superseded v2 (980 leaves, max abs delta
  8.327e-17). v2 preserved by rename at
  phase2_four_arm_v2.SUPERSEDED_BY_v2_1.json (ecb8707, unedited, rule 13).
  Determinism across THREE fits (ef9b775/19b0611/43f777d). Battery 456 checks,
  0 failing. Freeze decision remains WITH THE USER.
  PRIOR: 2026-08-26 (coordinator): OB dynamics loop CLOSED at I5 — reduced fine
  spec CONFIRMED, five specs consumed, freeze decision with the USER.
  STATEFUL harmful-cancel phase dispatched per
  live/pm_research/plans/STATEFUL_HARMFUL_CANCEL_TODO.md and R-145
  (BE Phase 0 manifest/repro = blocking; DA Phase 1 state features +
  declared dev top-up; DE Phase 3 state machine + parity; OPS heavy-run
  hygiene + recv_ns measurement)
```

## Batch 2 — archived 2026-08-28T10:09Z (1 entry, rolling-window overflow)

Moved in the same commit as the sweep that pushed it out of the window, per
the ruling's point (3). Join rule as in batch 1.

```yaml

  2026-08-28T09:53Z (MEM): R-248 -- BATCH 1 COMPLETE, VERIFIED IN THE
  COORDINATOR'S OWN RUN, AND THE CODEX ROUND HAS FIRED at e72dd4c with the
  request filed under reviews/ (no state-file collision this time). FIT IS NOW
  DOUBLE-BLOCKED and both gates must clear: Codex HOLD RELEASED, AND the USER's
  ruling on Q2's cell statistic -- BE implemented min(AUC(p_pos), AUC(p_neg)),
  the WORSE side, so half a working head cannot carry a cell, but the choice
  fills a gap in a USER-FROZEN amendment and is therefore the user's, blocking
  fit and NOT a review matter. The A1.1 bias algebra was hand-verified
  independently, and the falsifier asserts BOTH directions -- with no zero mass
  the amended and superseded forms AGREE, which is the real check: an amendment
  that changed the answer everywhere would be a different estimand, not a
  correction. TWO BE SELF-CATCHES recorded as one lesson: an old assertion
  ("0, not a crash") was the DEFECT WRITTEN DOWN AS THE SPEC, now inverted into
  a refusal test; and the runner's row() helper had been manufacturing the exact
  malformed pair A1.3 bans, so every earlier test ran on impossible rows -- the
  strictness caught its own harness on first contact. Falsifier counts recorded
  FROM THE SCRIPT (81/38/15 = 134) against BE's messaged 126, over-delivery
  direction, queried not blocking. MEM's proposed house rule was adopted:
  SEAT_PROTOCOL rule 15, verified at the file. TONIGHT: the deploy is ON but
  NOT unconditional -- an adverse O1-relevant Codex finding before ~23:55Z
  arming postpones the boundary.
```

## Batch 3 — archived 2026-08-28T10:21Z (1 entry, rolling-window overflow)

Moved in the same commit as the sweep that pushed it out. Join rule as in batch 1.

```yaml

  2026-08-28T09:57Z (MEM): R-249 -- THE USER RULED Q2 = min (worse side), so
  the A1.4 gap is closed IN THE FROZEN FILE and BE's :231 is authorized. GATE
  ARITHMETIC, STATED PRECISELY because this is the update most easily misread:
  the fit was blocked by (a) HOLD RELEASED and (b) the user's ruling; (b) is
  now SATISFIED -- ruled, not dissolved and not found unnecessary -- so fit
  clearance blocks on (a) ALONE and a clean review DOES clear the fit. The
  one-side flag MEM routed into the round WAS REAL and BE closed it (b3f082e):
  report_arm had filtered None out and taken min() of what remained, so a p_pos
  of 0.92 beside p_neg None produced a cell of 0.92 -- one side sailing past
  the UNDERPOWERED machinery as though the pair had been measured. Q2 now
  requires BOTH sides, with the rule carried in the ARTIFACT rather than a
  commit message. MEM's four-instance consolidation is now SEAT_PROTOCOL RULE
  16, verified at the file: a control that cannot fail must never be mistaken
  for one that passed -- phrased control-side because the next instance will
  wear a shape none of the four had, plus the coordinator's completing clause
  that every control ships BOTH directions. And the count discrepancy held open
  rather than reconciled ALREADY PAID: chasing it, BE found its own
  falsifier_count.sh could fall back to running the runner BARE -- main(), the
  heavy data path, from the session shell, against a now-binding resource cap.
  Fallback removed; a helper for counting tests must not be able to start a
  research run.
```

## Batch 4 — archived 2026-08-28T14:20Z (1 entry, rolling-window overflow)

Moved in the same commit as the sweep that pushed it out. Join rule as in batch 1.

```yaml

  2026-08-28T10:00Z (MEM): R-250 CORRECTS A NUMBER THIS FILE TOLD READERS TO
  TRUST. The 134-vs-126 count was never an instrument disagreement: the
  pre-fix counting script requires a module argument, the coordinator's
  invocation passed none, the script died at the arg check with stderr
  suppressed by that command's own 2>/dev/null, and its || fallback grepped
  SOURCE TEXT to 81/38/15 -- which was then recorded under the instrument's
  name. BE's runtime 126 was correct at e72dd4c. CURRENT TRUTH 81/39/11 = 131,
  VERIFIED BY MEM'S OWN RUN of the fixed script at HEAD rather than taken from
  the entry. My earlier flag told future readers to prefer 134 and treat 126 as
  superseded; that instruction was WRONG, is superseded in-band, and the miss
  is named: I verified that the register SAID "by the counting script" without
  verifying the script PRODUCED it -- rule 16 says verify at the artifact THE
  CLAIM NAMES, and the claim named the script. When a state file is about to
  tell readers which of two conflicting numbers to trust, RUN THE INSTRUMENT.
  Two lessons adopted both seats: a count without a commit ref is not a
  measurement, and an instrument's name may only be attached to numbers the
  instrument actually produced. BE refuted both plausible explanations BY
  MEASUREMENT and refused to put a guessed cause in the register when the true
  one was invisible from its environment. Q2's both-sides fix is VERIFIED at
  b3f082e and composes with the min ruling: MIN ADJUDICATES ONLY WHEN BOTH
  SIDES ARE MEASURABLE, otherwise the cell is UNEVALUABLE and still occupies
  its Holm slot.
```

## Batch 5 — archived 2026-08-28T15:15Z (1 entry, rolling-window overflow)

Moved in the same commit as the sweep that pushed it out. Join rule as in batch 1.

```yaml

  2026-08-28T10:09Z (MEM): R-251 -- THE DEPLOY IS POSTPONED, AND IT POSTPONED
  ITSELF. Codex's batch-1 filing is in, verified in full by the coordinator with
  NO REVIEW ERRORS, and the pre-ruled adverse-finding condition FIRED on a
  verified DB2: no user ask was needed because R-240/R-245 had already written
  the condition before the finding existed. O1 MOVES TO 2026-08-30T00:00:00Z.
  TONIGHT: NO ARMING, NO DEPLOY, the v3_1 hold UNCHANGED -- and unchanged is
  correct, because the recommended fix is CONSUMER-SIDE, leaving committed v4
  untouched. Tonight's 00:06Z old-bar verdict runs unaffected. Consequences
  priced rather than discovered: 08-29 runs on the OLD collector so the post-O1
  P1 band moves to 08-30, 08-29's v2 verdict is inadmissible under the hold, and
  because the bar predates the day, re-verdicting after release is legitimate --
  ACCRUAL PAUSES, IT DOES NOT VOID. BOTH HOLDS MAINTAINED: DB1 publishes a
  coin-day PASS OVER A 4,000-SECOND OUTAGE (the recorded-not-enforced defect
  recurring at the PER-COIN level after the whole-day path was closed); DB2 is
  the seam NEITHER SIDE OWNED -- two green suites, and an integration that
  always refuses when O1d fires; I11-1 is still live at HEAD as the same blocker
  wearing a new face; I11-2 is a 24-cell evaluator that is unit-proven AND NEVER
  INVOKED, which is a control that cannot RUN rather than one that cannot FAIL;
  I11-3 lets a one-class head report OK. The 2B freeze is DEFERRED: a dollar mid
  is not a probability, and FREEZE-AFTER-REVIEW IS NOW TWICE VINDICATED -- FP2
  would have been frozen in.
```

## Batch 6 — archived 2026-08-28T15:19Z (1 entry, rolling-window overflow)

Moved in the same commit as the sweep that pushed it out. Join rule as in batch 1.

```yaml

  2026-08-28T10:21Z (MEM): R-252 + R-253. THE SETTLEMENT-SOURCE PREMISE WAS
  FALSE. Settlement is CHAINLINK TWAP-vs-OPEN, ties UP -- not Binance. Verified
  at markets.jsonl by my own run: 17,734 records, 17,734 Chainlink, ZERO
  Binance (the register's 17,727 was right as of its run one minute earlier;
  the file grew 7 records between -- rule 8's tape-grows-during-measurement,
  live). THE ESTIMAND CHANGES WORK, not just wording: the settlement event is
  P(TWAP over window >= the window's OWN OPEN) -- a PATH average, so any
  transformation built on terminal price would PRICE THE WRONG EVENT; tie->UP
  is pinned by the venue, not chosen by us. DA's R-244 venue-lag reading is
  VOID, and note the direction -- a positive bookTicker increment would now be
  a GENUINE CROSS-VENUE LEAD, a STRONGER claim than the voided reading allowed;
  the decision-value complement survives. THE PRE-DECLARATION IS WHAT MADE THIS
  CHEAP: both readings were on record before any number existed, so the false
  one died with no sign to build a story around, and NOTHING WAS EVER SCORED ON
  IT. Two facts kept separate: DA erred first AND corrected first; the
  coordinator ratified the false reading at R-244 and CO-OWNS it. Rule 9 stands
  through a corrected door -- the tautology was never Binance, it is that
  IDENTITY ALREADY PRICES THE EVENT. **CLAUDE.md line 130 carries the same false
  claim in the USER'S OWN FILE; no seat edits it, it is flagged to the user, and
  until their edit lands the claudemd_rule9_parenthetical_is_FALSE flag is the
  correction of record -- no seat may cite that parenthetical as fact.** Also:
  RULE 17 ADOPTED (suite-green is not pipeline-wired) after batch 2 DEMONSTRATED
  the both-halves closure, so it lands with a cite rather than an anecdote.
```

## Batch 7 — archived 2026-09-01T03:49:46Z (1 entry, rolling-window overflow)

Moved in the same documentation true-up that pushed it out. Join rule as in
batch 1.

```yaml

  2026-08-28T14:20Z (MEM): SWEPT R-254..R-276 after a 23-entry gap. THREE
  THINGS MOVED THAT A SUMMARY WOULD GET WRONG. (1) DAY-BAR V2 IS RELEASED and
  governs coin-days >=08-29; the R-256 inadmissibility interim lifts and 08-29
  RE-VERDICTS under released code -- but RELEASED IS NOT "THE DAY PASSES", and
  the filing explicitly does not pre-judge 08-29. (2) O1 IS CLEARED and the
  boundary RE-ARMS for 2026-08-30T00:00:00Z (22:30Z confirm, 23:56Z prep,
  00:00Z deploy + era stamp); the v3_1 hold stays until then. (3) THE
  PROVENANCE CHAIN ON THE COMMITTED RESULT IS OPEN: seam 47j fired because THE
  GATE THAT SIGNED FIT7'S TAPE VERDICT IS NOT THE GATE THAT EXISTS (two
  substantive gate-defect fixes landed since). BE's precision is the record --
  IT DOES NOT CLAIM THE VERDICT IS WRONG, THE CHAIN NO LONGER CLOSES -- and
  v2.3 and the freeze receipt MUST NOT be cited as gate-verified until DA's
  re-gate determination lands, both branches pre-declared. ALSO: THE ESTIMAND I
  RECORDED THIS MORNING IS SUPERSEDED -- the repo's own passed reconstruction
  refutes full-window TWAP (86.9%) in favour of S60 ENDPOINTS (99.8%), verified
  by me at EXP_RESULTS_2026-08-20.md:10-17; the description-vs-reconstruction
  tension is STATED, not resolved, and 2B is NOT FIT TO FREEZE. Second
  settlement correction in one day, both pre-freeze. THE USER RULED on
  CLAUDE.md (option (a): it defers to SEAT_PROTOCOL for this program, rule-9
  fix rides the same edit) -- text drafted, APPLIED ONLY BY THE USER'S HAND.
  011 stays DARK.
```

## Batch 8 — archived 2026-09-01T09:43:00Z (1 entry, rolling-window overflow)

Moved in the same criteria/live-status true-up that superseded it. Join rule as
in batch 1.

```yaml

  2026-09-01T03:49:46Z (USER-AUTHORIZED CODEX DOC TRUE-UP): THE GOVERNING TODO
  IS plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md §10, not the subordinate
  stateful-cancel worksheet. Current map: dataset/PRED_STATE_V1 complete;
  receipt/runtime seams substantially built with final model-path closure still
  owed; Iteration 011 preregistration and implementation built but review-held,
  nothing fitted or scored; typed fair-price Identity built but challenger
  protocol not freeze-ready; skew freeze is a draft; seven-arm work is
  stub/inert only; integrated replay/economics not complete. The separate BTC
  hazard seed remains FROZEN-UNVALIDATED and the integrated candidate is not
  frozen. Forward reach is G=0/5 qualifying complete UTC days. Collector v4.1
  is live since the 2026-08-31T22:00Z boundary (PID 1108125, NRestarts=0 at
  this as-of). 08-31 is mixed-era and BTC quality-failing; 09-01 is the first
  era-pure admissible v4.1 day but is incomplete. HEAD/origin were equal at
  1aaac18 before this documentation edit; that commit claims RR1/RR3 closure,
  but the latest independent Codex filing reviewed its parent and a fresh
  release has not yet been filed.
```

## Batch 9 — archived 2026-09-01T12:56Z (1 entry, rolling-window overflow)

Moved in the MEM sweep that recorded iteration 011's first real 24-cell family.
Join rule as in batch 1.

```yaml

  2026-08-28T15:15Z (MEM): R-277..R-288. THE PROVENANCE CHAIN RESTORES -- and
  the two states are different, so both are recorded: SCIENTIFICALLY CLOSED (the
  re-gate determination returned IDENTICAL on its pre-declared branch,
  independently confirmed) but NOT YET MECHANICALLY CLOSED (seam 47j stays
  DELIBERATELY RED until the fit-time re-stamp -- a correct instrument reporting
  an unfinished mechanism, not a live defect). v2.3, the freeze receipt and the
  increment-null stand ON THE RE-DERIVATION; the do-not-cite interim and the
  trajectory hold both LIFT. Method marks worth copying: subject identity
  checked FIRST (comparing verdicts about different tapes proves nothing) and
  THE COMPARATOR FALSIFIED BEFORE THE ANSWER ARRIVED. Honest scope kept: the
  gate fixes are NOT inert in general -- they were correctly invisible on THIS
  tape. A finding rode along: THE VERDICT THE WHOLE CHAIN RESTS ON WAS NEVER IN
  GIT, now committed BYTE-UNCHANGED (committing preserves a frozen artifact; it
  does not edit it). ON THE COMMITTED NULL: a two-sided/one-sided defect exists
  in the module that fed the canonical null, AND THE CONCLUSION STANDS -- for
  positive effects the two-sided p is ~double, so every surviving cell survives
  MORE easily under the correct test. Dependence checked three times
  independently, mine included (12 cells, three negative at p 0.23/0.28/0.42,
  ZERO negative under p<0.10) -- and my FIRST parse matched zero cells and would
  have passed VACUOUSLY, so the count falsifier is why the check counts. NO
  SUMMARY MAY SAY THE CANONICAL NULL WAS WRONG. Round 3 is FIRED at a63d717 with
  the O1-adverse condition LIVE AGAIN before tonight's arming.
```

## Batch 10 — archived 2026-09-01T13:53Z (1 entry, rolling-window overflow)

Moved in the MEM round-3 sweep of R-374..R-381. Join rule as in batch 1.

```yaml

  2026-08-28T15:19Z (MEM): R-289 -- MY DISCLOSURE CAUGHT A SECOND VACUUM, AND
  IT WAS IN THE REGISTER. The coordinator's R-288 "independent confirmation" of
  the committed-null dependence check was ITSELF vacuous: its parse matched all
  12 cells but guessed the observed field name, so every cell defaulted to 0 and
  the filter never fired -- a FIELD-LEVEL vacuum wearing the shape of a
  confirmation, recorded as independent evidence. Mine, an hour earlier, was a
  CELL-LEVEL vacuum (parse matched zero cells) caught by implausibility. SAME
  FOUR-LINE CHECK, TWO SEATS, ONE HOUR, TWO VACUUMS, EACH VACUOUS A DIFFERENT
  WAY -- and neither was visible from inside its own run. THE CONCLUSION WAS
  NEVER IN DANGER: BE's original check was real, and the ledger now reads BE +
  MEM's asserted check + the coordinator's CORRECTED check, agreeing exactly.
  What was wrong was the ACCOUNTING of the evidence, which my own flag had
  overstated as "checked three times independently" -- corrected in-band here,
  original line kept. THE RULE THE PAIR DEMONSTRATES: a verification claim
  entering the register must assert that its parse ACTUALLY READ the population
  AND the fields it filters on; "found nothing" from a reader that touched
  nothing is the empty-set trap in the checker's chair. Note what the matched
  pair proves that one instance could not: A COUNT ASSERTION ALONE WOULD HAVE
  CAUGHT MINE AND MISSED THE COORDINATOR'S.
```

## Batch 11 — archived 2026-09-01T14:16Z (1 entry, rolling-window overflow)

Moved in the MEM round-5 true-up of R-382..R-384. Join rule as in batch 1.

```yaml

  2026-09-01T09:43:00Z (CRITERIA/LIVE-STATUS TRUE-UP): THE GOVERNING TODO IS
  plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md §10. Dataset/PRED_STATE_V1 is
  complete; Iteration 011 is USER-RELEASED, its earlier queue-contamination
  halt is withdrawn. A memory-sliced BTC attempt started 09:34Z, was stopped at
  09:42Z after indexing, and produced no result artifact or recorded fit/score
  completion. QR_SKEW_ONLY semantics are USER-FROZEN; real
  seven-arm parity, lifecycle economics and the integrated candidate remain
  open. Forward reach remains G=0/5. On 09-01 at 09:29:39Z, all 113 elapsed
  BTC/ETH windows were present and both coins passed governing day_bar_v2:
  BTC 572.2s accumulated loss, pace-adjusted P1 about 60.3 vs 120, P2 zero
  material windows, P3 185.2 vs 900. Collector v4.1 is active at PID 1108125
  with NRestarts=0. The day is a PROVISIONAL QUALITY PASS but cannot accrue
  until closed. Era is an interlock, not a quality grade. Breadth is reported,
  not a gate: 52/113 elapsed BTC windows had some overlap, but replay clears,
  resets and resynchronizes after a gap; the claim that a gap poisons the rest
  of a window is withdrawn. The superseded v1 count predicate does not pass
  09-01 because two hours exceeded 15, despite the average being below 15;
  this has no effect on the governing v2 verdict.
```

## Batch 12 — archived 2026-09-01T17:24Z (1 entry, rolling-window overflow)

Moved in the MEM round-6 true-up of the released review cycle (R-385..R-393).
Join rule as in batch 1.

```yaml

  2026-09-01T12:56Z (MEM SWEEP -- ITERATION 011 HAS A REAL 24-CELL FAMILY;
  THIS SUPERSEDES THE 09:43Z "STOPPED AT 09:42Z, NO RESULT ARTIFACT" LINE,
  WHICH STAYS BELOW AS PROVENANCE). Verified at the artifacts (git show plus the
  JSON on disk), never from the dispatch that ordered the true-up. THE THREE
  COMMITS. 54f899d (10:20Z) fitted 011 inside the UNRAISED 12G cap by PACKING
  THE DESIGN MATRIX: compact_design packs PM+FN+ST into one float64 array and
  RELEASES the lists-of-lists (7.11 GB -> 0.45 GB for the same 578,917 rows), so
  the topup pass allocates into space already held instead of growing past the
  cap. Two of its own defects were caught by guards on the way -- a --coin slice
  applied to the TAPE INDEX starved eth to 0 of 520,033 rows and the absorption
  bound REFUSED it, and the source regression guard written to prevent its
  return MATCHED ITS OWN string literal. e326782 (10:48Z) got THE FIRST REAL FIT
  (12.0G peak, no oom-kill, both feature passes, score index and purge) and then
  hit a seam defect only ever reachable once 011 was actually fitted:
  phase2_arms._feature_pass projects kept rows to a FIXED field list that OMITS
  any_fill_ahead, which the FROZEN phase2_iter011.validate_row requires
  (MISSING_GATE) -- two frozen documents, each correct alone, that had never
  met. RESTORED IN THE RUNNER, which declares itself OUTSIDE the lattice,
  because phase2_arms.py is in CODE_IDENTITY_FILES and the frozen candidate
  binds its hash; the restoration CALLS the canonical
  harmful_exposure_rows.any_fill_ahead rather than reimplementing the predicate,
  and stored-vs-derived agree on 1,125,289 fragment and 638,917 topup rows with
  ZERO disagreements. 0b1f6bb (11:27Z) completed the science and fixed the
  mode-aware output declaration: the guard demanded the unsliced filename from a
  --coin run, so a guard that could not pass was refusing a run that had already
  written its artifact. THE ARTIFACT, on disk at 11:23:34Z:
  data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json, 96,707
  bytes. ALL 24 DECLARED CELLS PRESENT (OK 6 / NO_INCUMBENT_COUNTERPART 12 /
  AGGREGATION_UNDECLARED 6); the Holm denominator held at 24 and is DECLARED,
  NOT EVALUATED, with unevaluable cells occupying their slots. Q1_arrival
  SURVIVES Holm: auc 0.8303 lgbm / 0.7733 linear, p 0.001996, holm 0.0479.
  Q2_sign is NO_INCUMBENT_COUNTERPART BY DESIGN (comparable:false -- the
  incumbent has no sign head, so no incremental null exists; the p it carries is
  the MATCHED-RANDOM null and must never be read as the other one).
  Q3_magnitudes withholds p as AGGREGATION_UNDECLARED, AND THE RULING IT WAITS
  FOR ALREADY EXISTS: R-306 (USER, 2026-08-29T04:40Z) ruled CONJUNCTION + WORSE
  SIDE, recorded in the frozen plans/ITER011_PREREG_AMENDMENT_A1.md and never
  implemented in code; the per-coin evidence is preserved so the ruling can be
  applied without re-running (both matched_random_p 0.001996). Q4_combined_ev IS
  THE DECISION METRIC AND IT IS UNADJUDICATED: 6,962.4 to 14,477.0 net cents is
  the CANDIDATE'S OWN value, not an increment, while incumbent_null_applicability
  declares Q4 comparable:true -- so the incumbent SHOULD have applied and did
  not load. THERE IS NO ECONOMIC RESULT, and that is a defect to chase rather
  than a design limit. TWO LIMITS GOVERN HOW ANY OF IT MAY BE READ. (1)
  RESOLUTION: every surviving p is 0.001996 = 1/501, the FLOOR at 500
  permutations, and holm is 24 x 0.001996 = 0.0479 -- the family only just
  clears 0.05, and at 26 cells no cell could survive whatever the effect. (2)
  STATUS: the artifact COMPUTES is_a_validation=false itself -- eval is the
  da_development_topup population, 08-25 alone, 311,640 btc rows / 177,674
  actions, G=0 complete UTC days, intervals not claimable, clustered on WINDOW
  where the ruled unit is the UTC DAY, so the p-values are OPTIMISTIC. This is
  DEVELOPMENT EVIDENCE (prereg 4: it selects, it never validates). No full
  both-coin artifact exists: the slice records iter011_conditional_value_v1.json
  as NOT WRITTEN, and eth is reported-only under btc-only adjudication (R-306).
  Identity carried in the artifact: fit_code ad535550d366347d -- the post-09:00Z
  DECLARED drift, not the freeze's 3d0b6c8c6dfe9466 -- tape c7ab02ebcf27d2fc,
  fragment 19a50195c34d0af2, topup e75d0e210590e2a8, and
  standalone.is_in_identity_lattice=false. MY OWN CHECK SHIPPED A FALSIFIER
  (R-289, my own lesson): it REFUSES an emptied family and a
  status-field-stripped copy and ADMITS the real artifact -- both directions,
  because a count assertion alone would have caught only one of those two vacuum
  shapes. OPEN, and neither is MEM's to close: the Q4 incumbent-loading defect
  (no economics until it closes) and the unimplemented R-306 conjunction for Q3;
  more permutations are needed before this design can carry a verdict at all.
  EVERYTHING ELSE IN THE 09:43Z ENTRY BELOW STANDS UNCHANGED -- collector v4.1
  live, 09-01 the first era-pure admissible day but incomplete, G=0/5 forward
  reach, era as an interlock, breadth reported and not gated.
```

## Batch 13 — archived 2026-09-02T00:21Z (1 entry, rolling-window overflow)

Moved in the MEM round-7 true-up of the first accrued forward-race day.
Join rule as in batch 1.

```yaml

  2026-09-01T13:53Z (MEM ROUND 3 -- R-374..R-381 SWEPT INTO THE STATE FILES).
  Every claim below verified at its artifact by execution, never from the
  dispatch that ordered the sweep. THE REVIEWER SEAT CHANGED HANDS (R-375,
  8b47dff): Codex quota exhausted and the USER restarted pm-codex as a CLAUDE
  session. The seat's surface is unchanged -- filings under workspace/reviews/,
  holds and an explicit HOLD RELEASED, never fixes code, never touches state
  files -- but ITS INDEPENDENCE PROFILE DID NOT SURVIVE, and that is recorded
  rather than hidden: the reviewer is now the same model family as every seat it
  reviews, which is R-348's correlated-blind-spot finding applied to the review
  seat itself. The mitigation is GROUND, not prompt -- review only committed
  artifacts at the pinned tip, prefer EXECUTION over reading, and treat
  agreement with a seat's own claim as consistency and never confirmation. New
  filings are REVIEW_* carrying a reviewer line; the CODEX_* files are the
  Codex-era record and are never edited (the KIND of document must stay
  identifiable). The 1aaac18 RR1/RR3 closure claims remain UNRELEASED because
  the round-2 filing reviewed its parent. THE RESOURCE RULE IS NOW A MECHANISM,
  NOT A DISCIPLINE (R-376, 8b47dff): pm_research_guard runs on a 60-second timer
  OUTSIDE research.slice -- deliberately, because a memory-saturated slice must
  not stall its own guard -- and I verified it live, last run 19 s before this
  sweep. IN_SLICE is report-only (the kernel's job); COLLECTOR is NEVER touched
  and matches BOTH homes, pm-collector-* AND collectors.slice, an exemption
  CORRECTED BY MEASUREMENT after the first draft's match missed the P-2026-002
  collectors; outside the slice FLAG >= 2G and KILL >= 8G, where 8G is measured
  territory rather than taste (Q-BE-111 polled 8.8G one to two minutes before
  the box died). CPU is deliberately not killed on: weights already price
  contention, and on a swapless box memory is the failure that destroys.
  pm-measurement-pipeline was the SECOND unguarded 16G unit, found by looking
  for the CLASS instead of the instance; verified here that it and
  pm-evaluation-pipeline both read Slice=research.slice -- AND MY ROUND-2
  FINDING IS CLOSED: both repo mirrors are now byte-identical to the installed
  units, so the guard is reproducible from git and not merely live. COORDINATION
  IS BATCHED IN BOTH DIRECTIONS: R-377 (USER) makes the review cycle
  batch-complete -- one filing per round, all fixes landed and pushed TOGETHER,
  the reviewer notified ONCE at a pinned tip -- and R-378 (USER) applies the same
  law to the coordinator's own dispatch loop as SEAT_PROTOCOL rule 18: a seat
  receives its COMPLETE batch in one dispatch and nothing further while it is in
  flight, stop-the-line excepted. R-381 (USER) adds the clause that batching is
  about COMPLETENESS, NOT IDLENESS -- a closed round is followed promptly by the
  next complete batch, and a seat waiting between rounds is a coordination miss.
  THE DE SEAT IS STAFFED (R-379, d929031, USER act), executing R-165's parking
  clause: harmful_stateful_policy.py, de_actionspace.py and de_constraints.py
  transfer to DE, whose first batch is real-data seven-arm parity, the
  registry-closure draft and the Phase-4 grid protocol as DRAFT-FOR-USER-FREEZE.
  THE MODULE AUDIT, verified by me at live/pm_research/contracts/contracts.yaml
  (version 24, 28 modules): EV-Replay is THE gap -- exact-case "Replay" has ZERO
  hits, no module and no type, while the lowercase word appears only twice in
  prose bodies, so a grep for vocabulary must not be read as a reference (rule
  16). DE-ActionSpace is a registry inconsistency: the TYPE exists at :1201 and
  is referenced at :386 and the CODE exists, while the module list registers
  only DE-Constraints, DE-Actuator and DE-Allocator. OP-LatencyBudget has zero
  hits and is deferred-with-trigger, to be NAMED rather than silently absent.
  The OPS seat is consciously coordinator-absorbed and staffing it is the
  USER's call. ON THE 011 LANE THE ARTIFACT STATE IS NOT WHAT A READER WOULD
  ASSUME. DA's Q-DA-197 ran an INDEPENDENT reader
  (live/pm_research/da_iter011_contract_verify.py, no shared code with
  phase2_iter011*, R-235) and I REPRODUCED ITS VERDICT BY RUNNING IT: "14/23
  contract checks hold; 9 FAIL; reads: 24 cells; 296 typed field reads". The
  nine are disclosure and predicate defects, not a moved number: F1 the n a cell
  CARRIES is the arrival n in 12 of 24 cells, a 22x overstatement of the
  population behind the statistic; F2 the survivor predicate was HOLM ALONE, so
  NO_INCUMBENT_COUNTERPART cells were published as surviving; F3 the
  declaration's own handling does not reach Q3; F4 is rule 10's fourth instance
  (the Q3 string named the BETTER side while the code computed the worse),
  found independently by BE and DA in the same hours, which is what R-235
  exists for; F5 fit_code_ref null; F6 no as_of. BE cleared F3/F4 at 20d3c3a
  and, when F2 landed mid-fit, KILLED THE RE-RUN RATHER THAN LET IT EMIT
  (4438961): F2 changes a PUBLISHED VERDICT FIELD, so the artifact is to be BORN
  under the closed predicate instead of superseded afterwards. CURRENT ARTIFACT
  STATE, verified on disk: the declared path still holds 0b1f6bb's original
  (96,707 B, 11:23:34Z), preserved byte-identically beside it as
  __as_verified_by_Q-DA-197.json (same sha256 prefix 7d8437e6523ed32d); a
  __readjudicated_v2.json (101,789 B, 13:09Z) exists from the intermediate
  state; and iter011-fit-batch.service is ACTIVE/RUNNING, so NO artifact under
  the closed predicate exists yet and Q4 STILL HAS NO ECONOMIC RESULT.
  Q-DA-198 shipped the 0h breadth disclosure as REPORTED_NOT_GOVERNING carrying
  both denominators with a refusal behind it -- its own fixture-mirror mutant
  survived first and was killed -- and proved tonight's 00:06Z path by EXECUTION
  on closed 08-29, reproducing the HANDOFF row by a separate run. ROUND EDGES:
  BE open (fits running in-slice), DE open (first batch), DA re-opened on its
  round 2, the reviewer on a prep batch that files nothing, MEM closing here.
  UNCHANGED AND WORTH SAYING: nothing about the forward race moved today --
  09-01 is still the first possible forward day and reach is still G=0/5.
```

## Batch 14 — archived 2026-09-02T04:05Z (1 entry, rolling-window overflow)

Moved in the MEM round-8 true-up of the executed five decisions and the
fully-evaluated family. Join rule as in batch 1.

```yaml

  2026-09-01T14:16Z (MEM ROUND 5 -- R-382..R-384, AND THREE ITEMS NOW WAIT ON
  THE USER). Verified at the artifacts, including by running the instrument.
  THE ONE THAT MATTERED MOST IS A LAUNCHER-SEMANTICS DEFECT CAUGHT BEFORE IT
  RAN (b32e7e3, R-384): DA's own morning guard, assert_disclosure_carried(),
  raised a bare SystemExit -- which exits 1, and da_midnight_verify.sh reads
  rc 1 as "verified, and the day FAILS", a real result, while an instrument
  that refused to emit is rc 4, NOTHING WAS VERIFIED. So a guard refusal on
  TONIGHT'S 09-01 VERDICT -- the first day that can accrue -- would have been
  logged as a failing day. Fixed to a real Exception inside main()'s handler and
  PROVEN AT THE SUBPROCESS SEAM, because the exception type alone cannot show
  what the launcher sees; 2 mutants killed, 205 checks, 16 gates. DA records
  that it re-introduced the exact defect documented at the top of that same file
  (R-199 item 1): the class survived its own documentation, and the seam test is
  what holds it. Q-DA-199 -- THE CONTENT-LIVENESS RULE, DRAFTED FOR USER FREEZE
  (f1e3f53), closing R-370's open item: a feed that thins WITHOUT disconnecting
  writes no gap row, keeps full window coverage and passes P1/P2/P3, which is
  how 08-31 held 0.51% of normal rate for ~4.1 h with zero gap rows and how 668
  invisible windows sit across 7 of 13 days. I ran it: 30 checks pass, 2
  positive controls executed; the detector is pm_tape_density's, unchanged;
  thresholds are calibrated on consumed days <= 08-31 with CALIBRATION_MAX_DAY
  < EFFECTIVE_FROM_DAY enforced by a refusal rather than a comment. IT GOVERNS
  NOTHING TODAY BY CONSTRUCTION: governs() returns False unless BOTH
  FROZEN_BY_USER (currently False) AND day_token >= "20260902", one function
  holding both conditions so a consumer cannot satisfy one and forget the other;
  09-01 is deliberately NOT covered, because the rule was drafted while 09-01
  was in flight and applying it there would be choosing after seeing (rule 11).
  Q-DA-200 -- TWO BREADTH STATISTICS PULLED APART (f1e3f53): per_slug_affected
  lived inline in verify_day and could not be driven by any test, so the two
  could have drifted into one being derived from the other with nothing to
  notice. Both are callable now, each driven on a fixture built to make them
  DISAGREE IN BOTH DIRECTIONS, plus a signature check that neither can see the
  other's inputs; inert on real data (08-29 reproduces windows_gap_affected
  byte-identically, and only gap_series.ledger_lines moved, 9,682 -> 9,718,
  because the ledger grew between runs). docs/BREADTH_STATISTICS.md names which
  receipt carries which -- and the HANDOFF survey's own two columns are
  DIFFERENT statistics, one per-slug and one row-level. BE'S CODE HALF IS IN
  (4438961, 23/23 mutants) and the run was RELAUNCHED FROM THE COMMITTED TREE so
  fit_code_ref names a non-dirty commit; verified here that
  phase2_iter011_run.py is clean in the working tree and iter011-fit-batch has
  been active since 13:51:25Z. The batch closes on the run's artifact plus its
  Q-BE filing, which opens the review round. THREE ITEMS NOW WAIT ON THE USER
  and are gathered in HANDOFF's new PENDING USER DECISIONS table so the asks are
  findable in one place: (1) FREEZE THE CONTENT-LIVENESS RULE, the only one with
  a real clock -- a freeze after 09-02 opens costs its first governed day, so
  ~22:00Z tonight (R-383); (2) apply the CLAUDE.md amendment, whose two hunks
  are drafted at workspace/DRAFT_CLAUDE_MD_AMENDMENT.md; (3) freeze DE's Phase-4
  grid protocol when it lands. Per R-383 they go up as ONE composed ask, and if
  (2) and (3) miss the deadline then (1) escalates ALONE. ONE PRECISION NOTE:
  R-384 records the TODO sweep as "39->47"; both endpoints are right, and the
  path was 39 -> 42 (round 2, three ticks) -> 47 (round 4, five ticks), across
  two sweeps rather than one. UNCHANGED: 09-01 is still the first possible
  forward day and reach is still G=0/5, judged tonight at 00:06Z.
```

## Batch 15 — archived 2026-09-02T05:38Z (1 entry, rolling-window overflow)

Moved in the MEM round-9 true-up of the action-unit measurement.
Join rule as in batch 1.

```yaml

  2026-09-01T17:24Z (MEM ROUND 6 -- THE REVIEW CYCLE IS FULLY RELEASED AND THE
  011 RESULT OF RECORD IS ZERO SURVIVORS). Verified at the artifact, not from
  the register. THE ARC, R-385..R-393: the family went from SIX published
  survivors to ZERO, and that is the artifact becoming honest rather than the
  result getting worse. The six surviving cells' own declared_gate carried an
  incumbent conjunct THAT WAS NEVER EVALUATED -- apply_incumbent_hazard, built
  and falsifier-proven, ZERO production call sites: defect I11-2's shape for the
  third time in this programme, this time in the ONLY surviving head, found and
  escalated by BE itself (RR2-1). The fix makes the survivor predicate require
  every declared conjunct EVALUATED; failing cells become
  GATE_PARTIALLY_EVALUATED, reported and never dropped, denominator still 24.
  THE RESULT OF RECORD, measured at the file (142,609 B, as-of 16:57:01Z, and
  that as-of NAMES the population-read instant): surviving_cells = [], 0 of 24
  survive the joint reading; cells_by_status = 6 GATE_PARTIALLY_EVALUATED + 12
  NO_INCUMBENT_COUNTERPART + 6 OK. Q4'S INCREMENT IS POSITIVE IN ALL SIX CELLS
  (+278.6 to +3,867.1 net cents) AND CLEARS NO FAMILY-WISE BAR UNDER EITHER NULL
  FORM -- best one-sided p 0.01999 -> Holm 0.1199, with the two-sided form
  reported and never adjudicated. Q1's two AUCs (0.8303 lgbm / 0.7733 linear)
  are UNDECIDED pending the USER's Q1-leg ruling. NOTHING WAS DELETED TO REACH
  ZERO, and I checked rather than assumed: all six Q1 cells carry statistic,
  p_value and holm_p IDENTICAL TO THE DIGIT against the pre-fix artifact, and
  only status moved. Q4'S NUMBER ALSO CHANGED MEANING, which is easy to
  misread: this morning's +12,333.5c was the CANDIDATE'S OWN value, explicitly
  not an increment, because the incumbent never loaded; it now loads, so the
  cell REPORTS candidate +12,333.5 and incumbent +8,466.4 and ADJUDICATES the
  increment +3,867.1c over 166 windows against 2,000 sign-flip permutations --
  a reader comparing the two headline numbers across the day is comparing two
  different estimands. TWO REVIEWER RULINGS WORTH CARRYING: BE's refusal to
  raise the matched-random draw count was ENDORSED (A1.6's 2,000 pins the
  INCREMENT null; 5(1)'s matched-random declares >=200, so 500 satisfied the
  frozen design and raising it after seeing a one-draw margin would be rule 11),
  and the resolution for the NEXT run must be declared PROSPECTIVELY, a line
  that rides the A2 amendment -- BE refused a coordinator instruction and
  escalated instead, the protocol working against the coordinator, which is the
  correct direction. AND THE REVIEWER ATTACKED ITS OWN ACCEPTED FIX: it tried to
  defeat GATE_PARTIALLY_EVALUATED by dressing an unwired Q4 in it and the guard
  refused; its shrunk-coverage known-bad, admitted at 6 checks the round before,
  now REFUSES; removing either new rule kills the suite, and so does forcing
  every cell partial -- the admit direction it most expected to be missing. For
  a same-model reviewer (R-375) that is the ground the mitigation asks for: it
  RAN the code rather than reading it. Coordinator-side: RR2-3 at 9a53ea3, then
  RR3-2/RR3-3 at f72504d (a reversed-ledger-order fixture kills a loosened >= at
  check 173; a reused pid with no pin now REFUSES naming both candidate
  instants), 176 checks and 17 gates. BOTH OF THIS MORNING'S USER ASKS LANDED
  (R-386, "Yea proceed"): the content-liveness rule is FROZEN with
  FROZEN_BY_USER=True and EFFECTIVE_FROM_DAY unchanged at 20260902, so the first
  governed day is tomorrow and 09-01 is neither calibrated on nor judged; and
  the CLAUDE.md amendment LANDED with both hunks verbatim -- checked by
  re-running my own draft's anchor test IN REVERSE, so the claim rests on the
  file. Rule 9 no longer names Binance and asserts no settlement statistic, and
  the one-writer exception retires SEAT_PROTOCOL rule 6's standing caveat. FIVE
  ASKS NOW WAIT ON THE USER, gathered in HANDOFF's PENDING USER DECISIONS table:
  (1) wire Q1's incumbent leg or rule it out -- the sharpest, it decides whether
  the published survivor count returns 0 -> 6; (2) the Q3 gate ruling; (3)
  amendment A2 plus the prospective-resolution declaration; (4) the Phase-4
  protocol and registry freezes; (5) per-seat worktrees. ONE TIMESTAMP FLAG,
  recorded not adjudicated: R-393's header reads 17:35Z while the commit that
  created it (8b80d83) is stamped 17:22:09Z -- the entry runs ~13 minutes AHEAD
  of its own commit, rule 12's class in the forward direction. Nothing
  downstream depends on it and the register is the coordinator's surface.
  UNCHANGED: the forward race waits on tonight's 00:06Z first accrual-eligible
  verdict; G=0/5.
```

## Batch 16 — archived 2026-09-02T08:16Z (1 entry, rolling-window overflow)

Moved in the MEM round-10 true-up of the wiring, the blindness finding and the
established blackout cause. Join rule as in batch 1.

```yaml

  2026-09-02T00:21Z (MEM ROUND 7 -- THE FORWARD RACE HAS ITS FIRST DAY:
  2026-09-01 ACCRUED, G = 1/5). Verified at the artifact
  (da_dayverdict_20260901.json, written by the 00:06:00Z timer, as-of
  2026-09-02T00:06:01Z), with the conjunction RECOMPUTED rather than read back:
  FINISHED (day_closed true -- day_closed_calendar true, and the
  day_closed_selector false sub-field is a stated reason, not the conjunct) AND
  AFTER (post_freeze_pass) AND ADMISSIBLE (era_admissible, clob_v4_1,
  era_role INTERLOCK) AND HEALTHY (day_quality_pass, with BOTH adjudicated coins
  passing their governing day_bar_v2) -> race_accrual_eligible TRUE. btc P1 84.4
  s/hr against 120, P2 0 material windows, P3 185.2 against 900; eth P1 6.9,
  P3 107.9. I REPRODUCED DA'S INDEPENDENT CHECK BY RUNNING ITS INSTRUMENT:
  da_verdict_check --day 20260901 gives 8/8, accrues=True, four scopes, both
  denominators coinciding -- the check that catches an open-day elapsed count
  inside a closed-day report. FOUR MORE ACCRUING DAYS REACH THE >=5-DAY BAR;
  EARLIEST HONEST INTERVAL ~09-05, AND ONLY IF EVERY DAY ACCRUES.
  A CORRECTION THAT MATTERS BECAUSE A STATE FILE IS WHERE A NUMBER BECOMES THE
  RECORD: R-395 reports "the decision_note itself flags ~80% of btc windows
  touched at 28.0 gaps/hr", and the round-7 dispatch repeats it as this day's
  figure. IT IS NOT THIS DAY'S. That sentence lives inside decision_note as the
  instrument's STANDING ILLUSTRATION, phrased about "day one" of an earlier era,
  and it exists to argue that gaps/hour understates damage. 09-01's OWN btc
  numbers, at the artifact: breadth 160/288 = 55.6% COIN_LEVEL (the governing
  scope, R-191) and 159/288 = 55.2% PER_SLUG, at 14.38 gaps/hr with 345 gaps,
  2,025.5 lost seconds, 23 of 24 hours carrying a gap, and 8 hours over the
  SUPERSEDED v1 count bar (worst hour 31, governing nothing). The caution is
  sound and still applies -- read windows_gap_affected beside gaps/hour, never
  instead of it -- but the figures belong to a different day. Flagged for the
  coordinator; the register is their surface. THE HEAVY DISCLOSURE: ~115 MINUTES
  OF NEAR-TOTAL LOSS THAT NO GOVERNING BAR CAN SEE. DA's finding, two contiguous
  outages -- 00:00-01:05Z (65 min) and 22:45-23:35Z (50 min) -- at 0.01-2.2% of
  median window content, on ALL SEVEN COINS, with NO GAP ROWS. P1/P2/P3 pass
  straight through both, because the duration bars charge only for time the
  ledger knows about. TWO INDEPENDENT INSTRUMENTS AGREE TO ONE MINUTE: the
  collector log's msgs/s measure reads 116 intervals below a tenth of median
  (0.0806 of 1,439) and the raw gzip-trailer byte measure reads 115 -- different
  inputs, different code, no shared term, which is what makes it a measurement
  rather than one estimator's artifact. AND THE RULE WRITTEN FOR EXACTLY THIS
  CLASS BECAME EFFECTIVE TODAY WHILE REMAINING UNWIRED. The content-liveness
  rule is frozen (R-386) with EFFECTIVE_FROM_DAY = 20260902, so 09-02 is its
  first governed day -- but governs() returning True CHANGES NO VERDICT: I
  grepped it independently and the only reference outside its own file is
  v5_deploy_gates.py:54, which runs its SELFTEST. No consumer calls governs() or
  measure_day(). THAT IS RULE 17'S SHAPE TWICE IN TWELVE HOURS -- the Q1
  incumbent leg that cost the 011 family its six survivors, and now a frozen
  rule that governs nothing because nothing calls it. Both were built, both were
  falsifier-proven, neither was reached: a guard's EXISTENCE and a guard's
  WIRING are separate facts and only the second is load-bearing. DAY ONE'S
  ACCRUAL IS UNAFFECTED by any of it: the rule does not govern 09-01, reads it
  CONTENT_THIN on the margins (L1 0.07968 vs 0.08, L2 13 vs 12, three coins
  failing L2 by one window), and the breadth figures are reported, never
  governing. tape_density reads UNMEASURED for 09-01 -- correctly a status
  rather than a clean zero. DA has escalated for the USER that the bar now sits
  where the events are (65 min fails, 60 passes); the coordinator has not ruled
  whether that joins the numbered list, so the FIVE USER DECISIONS ARE
  UNCHANGED: Q1-leg wiring, the Q3 gate ruling, amendment A2 with its
  prospective resolution, the Phase-4 and registry freezes, and per-seat
  worktrees. The 011 result of record is unchanged and negative: 0 of 24 cells
  survive the joint reading.
```

## Batch 17 — archived 2026-09-02T09:38Z (1 entry, rolling-window overflow)

Moved in the MEM round-11 true-up of R-406..R-408. Join rule as in batch 1.

```yaml

  2026-09-02T04:05Z (MEM ROUND 8 -- ALL FIVE USER DECISIONS EXECUTED, AND THE
  FULLY-EVALUATED FAMILY IS THE RESULT OF RECORD: 12 SURVIVORS WITH THE DECISION
  METRIC STILL FAILING). Verified at the artifacts, recomputed not read back.
  THE FIVE (R-397, USER adopting each recommendation verbatim -- "we can proceed
  the five decisions according to the recommendation"): (1) Q1's incumbent leg
  WIRED; (2) the Q3 ruling -- each head is adjudicated against its OWN
  declared_gate, and a conjunct nobody computed reads null, never false; (3)
  amendment A2 FROZEN at Option 1 -- 5(2) amended to one-sided with R-286/R-288
  as the recorded cause, p_two_sided retained as a diagnostic, and the
  matched-random resolution DECLARED PROSPECTIVELY at 2,000 draws for the NEXT
  run while THIS family stays at 500 with its floor disclosure (verified: the A2
  file reads FROZEN -- IN FORCE); (4) the Phase-4 protocol FROZEN, declared
  before any cell is read and with no Phase-4 cell existing at freeze, and the
  registry APPLIED v24 -> v25 -> v26 (verified at contracts.yaml: version 26, 30
  modules, EV-Replay and DE-ActionSpace registered, ReplayWindowSpec added,
  config_supplied:ActionSet REMOVED under amendment E on BE's confirmation);
  (5) worktrees ADOPTED in execution form -- four exist under SEAT_PROTOCOL rule
  19, with the limitation STATED rather than papered over: git refuses one branch
  in two worktrees, so LANDING stays in the shared tree under pathspec
  discipline and the ledger keeps one writer path. THE RE-ADJUDICATED FAMILY
  (157,455 B, as-of 2026-09-02T03:46:59Z): 12 of 24 cells survive the joint
  reading -- recomputed from the cells here, not read off the summary --
  cells_by_status 18 OK + 6 NO_INCUMBENT_COUNTERPART, denominator 24. Q1_ARRIVAL
  EARNED ITS PASS: its gate has two conjuncts and BOTH are now computed --
  candidate AUC 0.8303 lgbm / 0.7733 linear against the incumbent hazard head's
  0.7139, increments +0.1164 / +0.0594, beats_incumbent_hazard_head true on both
  arms, 166/166 windows, zero exclusions. Eleven hours earlier that same head
  published six survivors on a gate half of which had never run; the number that
  came back is the one the COMPLETE gate produces. Q3_MAGNITUDES PASSES ITS OWN
  GATE (both slope conjuncts true) -- BUT THAT IS A WEAKER FACT THAN Q1'S, and
  the artifact is careful about it: Q3's frozen gate carries NO incumbent term
  (incumbent_counterpart_computed false), so it cleared a bar that never asked
  for a comparison. Reading the twelve as one uniform result would flatten
  exactly the distinction ruling 2 exists to preserve. Q2_SIGN correctly stays
  NO_INCUMBENT_COUNTERPART. AND Q4_COMBINED_EV, THE DECISION METRIC, STILL
  FAILS: all six increments positive (+278.6 to +3,867.1 net cents), all six
  survives=false, best one-sided p 0.01999 -> Holm 0.1199. TWELVE SURVIVORS AND
  NO ECONOMIC RESULT ARE THE SAME SENTENCE. Every surviving p also sits at the
  1/501 floor, which the cells disclose as A BOUND, NOT A MEASUREMENT -- one
  draw the other way moves Holm 0.0479 -> 0.0958. Development evidence only
  (prereg 4: it selects, it never validates); the lattice is UNMOVED at
  ad535550d366347d because neither 011 module is in CODE_IDENTITY_FILES.
  ONE QUESTION NOW WAITS ON THE USER, and it is the largest this programme has
  asked: DOES Q1'S FULL-GATE SURVIVAL CONSTITUTE THE PHASE-2 WINNER that, with
  the frozen Phase-4 protocol, unblocks DE's latency x queue-reset-cost x budget
  grids? Escalated by the coordinator (R-398), NOT decided. It became askable
  only when the gate was completed, and it weighs against a FAILING decision
  metric and development-only evidence. STILL OPEN AND NOT AMONG THE FIVE, so it
  did not get resolved with them: DA's escalation that the content-liveness bar
  now sits exactly where 09-01's events are (65 min fails, 60 passes), and that
  the rule -- effective since today -- IS STILL UNWIRED. Re-checked at 04:05Z:
  its only reference outside its own file remains v5_deploy_gates.py:54, which
  runs its selftest, so governs() returns True and no verdict consumes it.
  UNCHANGED: the forward race is at G=1/5 and 09-02 is accruing.
```

## Batch 18 — archived 2026-09-02T10:15Z (1 entry, rolling-window overflow)

Moved in the MEM round-12 true-up of R-409..R-412. Join rule as in batch 1.

```yaml

  2026-09-02T05:38Z (MEM ROUND 9 -- THE UNIT QUESTION IS ANSWERED AND THE ANSWER
  DOES NOT DEPEND ON THE UNIT; THE WINNER RULING IS UNBLOCKED). Recomputed from
  the artifact, not read off its summary. R-399's hold was RIGHT and closing it
  cost nothing: the reviewer released BE's batch and held only the WINNER
  INFERENCE, on RR4-3 -- Q1's AUC was computed over 311,640 ROWS while the
  cell's n read 177,674 ACTIONS (1.754 rows/action), CLAUDE.md rule 2's exact
  class sitting in the one surviving statistic. One deduplicated pass, no refit
  and no new estimand, was all the ruling needed. WHAT CAME BACK (188,119 B,
  as-of 2026-09-02T05:21:34Z): THE LEVEL IS A RANGE, NOT A REPLACEMENT NUMBER --
  lgbm 0.790 / 0.864 / 0.876 by collapse rule (first / mean / max) against
  row-level 0.830, and linear 0.735 / 0.798 / 0.814 against 0.773, with THE
  ROW-LEVEL FIGURE SITTING INSIDE THE RANGE ON BOTH ARMS, which is the finding:
  it was not an artefact of counting a generation more than once. AND THE
  COMPARISON IS INVARIANT TO EVERY CHOICE A SEAT MADE: the candidate beats the
  incumbent hazard head under EVERY unit and EVERY collapse rule on BOTH arms,
  4/4 each, agrees_with_row_level true -- that is the conjunct Q1's gate actually
  asks about, and it is why the unit debate does not reach the verdict. 3.44% of
  generations (6,108 of 177,674) carry DISAGREEING ROW LABELS, a counted
  population rather than an assumption. TWO THINGS A READER SHOULD NOT MISS,
  both disclosed in the artifact rather than dug out of it: the designated
  primary collapse rule is MAX, which is also the HIGHEST of the three, and
  under FIRST the action-unit AUC (0.790) is BELOW row level -- so "deduplication
  raises it" holds for the primary rule and for two of three, not universally,
  and the artifact states plainly that which rule adjudicates is a USER
  question. The honest headline is the INVARIANT COMPARISON, not the level.
  RR4-1 CLOSED AND IT MATTERS LATER: twelve cells had been asserting
  gate_conjuncts_evaluated true while carrying a NULL conjunct -- the RR2-1 shape
  again, harmless only while Q4 fails and a live defect the moment Q4 improves.
  It is now DERIVED from the conjuncts themselves and Q4's reads False. RR4-2
  (both one-draw numbers computed rather than multiplied) closed in the same
  batch; RR4-4 is the coordinator's. THE WINNER RULING IS UNBLOCKED PENDING ONLY
  THE REVIEWER'S RELEASE, round open at pinned tip c180061. NOTHING ABOUT THE
  TEMPER CHANGES: the family is still 12 of 24 surviving, Q4 -- THE DECISION
  METRIC -- STILL FAILS, every surviving p still sits at the 1/501 floor as a
  bound rather than a measurement, and this is development evidence (prereg 4:
  it selects, it never validates). UNCHANGED: forward race at G=1/5, 09-02
  accruing; DA's content-liveness bar escalation still unruled and the rule
  still unwired.
```

## Batch 19 — archived 2026-09-02T10:45Z (1 entry, rolling-window overflow)

Moved in the MEM round-13 true-up of R-413..R-416. Join rule as in batch 1.

```yaml

  2026-09-02T08:16Z (MEM ROUND 10 -- THE RULE WAS UNWIRED, THEN WIRED, THEN
  PROVED BLIND; AND THE BLACKOUTS ARE THE VENUE'S). R-402..R-405, verified at
  artifact and source, including by RUNNING the frozen rule. (1) THE FLAG I
  CARRIED FOR THREE ROUNDS WAS REAL AND CLOSED THE SAME DAY: the frozen
  content-liveness rule was NOT WIRED into the verdict path, found on its FIRST
  GOVERNED DAY ~16 h before the first governed verdict --
  da_forward_day_verify.py was still running a PRE-FREEZE INLINE COPY whose
  emitted why said "NO ratified band exists", text written before the rule was
  drafted. Rule 17's class on the GOVERNING instrument itself. Wired by DA the
  same day (R-402 -> R-403, 3298a1d, review-released) and verified here at
  source: CLR.governs(day_token) is called in the verdict path and the artifact
  REFUSES TO EXIST without consulting the frozen rule (rc 4, never rc 1); the
  veto is NOT adopted (content_thin_vetoes_HEALTHY false everywhere,
  guard-refused otherwise) because the freeze resolved NONE of section 8 --
  correctly escalated instead of chosen. (2) ITS FIRST GOVERNED DAY IS CARRYING
  THE EXACT DEFECT IT EXISTS FOR: a 3 h 20 m ALL-COIN blackout, 01:35-04:55Z, NO
  gap rows, which every legacy bar passes (btc P1 20.1 against 120). Measured by
  running the rule: governs('20260902') True and governs('20260901') False;
  09-02 reads CONTENT_THIN with btc L1 0.407 against the 0.08 bar and a
  40-WINDOW RUN -- 40 x 5 min is the blackout exactly. (3) AND THE RULE CANNOT
  SEE THE WORST VERSION OF WHAT IT WAS BUILT FOR. RR6-1, HIGH, against the
  FROZEN rule: thinness is measured against the day's OWN MEDIAN, so past ~60%
  dark the median crosses into the dark regime, every dark window stops being
  thin, and a mostly-dark day reads CONTENT_LIVE at L1 = 0.0000, L2 run = 0 --
  computed by the reviewer on real 09-02 bytes extended to 288 windows before
  filing. A DETECTOR CALIBRATED ON A RATIO TO ITSELF CANNOT SEE THE CASE WHERE
  THE DENOMINATOR MOVES WITH THE NUMERATOR, and it fails silently in the
  safe-looking direction. DA's "L2 cannot shrink" was right under benign
  continuations and wrong where it matters most. (4) THE CAUSE IS ESTABLISHED
  AND IT IS NOT OURS (Q-DA-203 at 4f892de, design committed at 9785e5e BEFORE
  any in-window rate was read): three events three-for-three -- E1 08-26
  04:35-07:55Z PM thin 1.000 (195/195) 633.7 -> 1.23 msg/s; E2 08-31
  06:40-10:40Z PM 1.000 (239/239) 475.0 -> 1.62; E3 09-02 01:35-04:55Z PM 1.000
  (200/200) 335.5 -> 1.76 -- while Binance and Hyperliquid, SAME host, path and
  seconds, did not thin in ONE interval of ~600 in-window minutes, with
  Binance's rate RISING in-window in E3 (1347 -> 1518) and its receive latency
  flat throughout (72/74/74 ms in-window vs 75 outside): POSITIVE evidence of a
  healthy path, not absence of evidence. Every coin's run ends on ONE instant
  per event while onsets stagger by up to 2h10m, and a per-coin cause cannot end
  on a single instant. SETTLED: our collector, host and network are exonerated
  for all three events. NOT SETTLED: whether the venue's markets traded normally
  during the silence -- a harvestability question, open. THREE METHOD MARKS
  WORTH COPYING, all DA's own disclosures: the host leg is PARTLY UNMEASURED and
  says so (the R-163 journal reaches back only to 09-02T02:46:26Z, so E1/E2 have
  no host record); a predicate that FIRED was reported as NOT-EVIDENCE because
  it tests an absolute level against a standing baseline; and the POSITIVE
  CONTROL FAILED FIRST -- Binance's predicate had never fired in 15 days, so its
  0.000 proved nothing until injection on the venue's real series showed it CAN
  fire on that data shape. A control that has never fired is not a control that
  passed. THREE ITEMS NOW WAIT ON THE USER, in HANDOFF's table: (1) DOES A
  CONTENT_THIN DAY ACCRUE, needed by ~00:06Z -- R-404's three closes supersede
  R-403's flat projection: (a) THIN at close, recommend EXCLUDE per frozen
  section 7's pre-declared mechanism, which predates seeing 09-02 and is the
  least choose-after-seeing path; (b) genuinely LIVE, accrues with the blackout
  disclosed; (c) LIVE-BY-MEDIAN-COLLAPSE, where the section-7 trigger never
  fires because the instrument cannot see it, recommend the coordinator-exclusion
  path with the reviewer's table as the stated reason. The KNOWN CAUSE
  STRENGTHENS EXCLUDE-IF-THIN: the darkness is venue-inflicted, so the tape
  genuinely lacks the venue's content. (2) THE PHASE-2 WINNER RULING, fully
  unblocked (R-401, no reviewer hold open anywhere), with the reviewer's framing
  verbatim: the COMPARISON is unit-invariant, the LEVEL is not (0.876 ranking /
  0.790 valuing / 0.830 row-level between), survival sits on a 500-draw floor,
  and Q4 STILL FAILS. (3) SOON, the rule-v2 freeze when DA's draft lands: an
  ABSOLUTE floor beside the relative one -- what closes RR6-1 -- calibrated on
  <=08-31 days only, anchored on the three measured events. NAMED RISK, AND ONE
  NUMBER I WOULD NOT REPEAT AS GIVEN: the venue silence is recorded as recurring
  "~weekly", but the three events are 08-26, 08-31 and 09-02 -- GAPS OF 5 AND 2
  DAYS, three events in a 7-day span, MEAN 3.5 DAYS. On the observed rate the
  forward race's 5-day set could take substantially longer than the calendar
  suggests, and "weekly" would under-plan it. Stated as an observed rate on n=3,
  never a forecast. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving with
  Q4 failing; development evidence.
```

## Batch 20 — archived 2026-09-02T10:55Z (1 entry, rolling-window overflow)

Moved in the MEM round-14 true-up of R-417..R-418. Join rule as in batch 1.

```yaml

  2026-09-02T09:38Z (MEM ROUND 11 -- THE THREE USER RULINGS GET A CITABLE
  ANCHOR, AND THE v2 DRAFT NAMES THE DAY IT GOES BLIND). R-406..R-408, each
  verified at its artifact. THE FINDING WORTH KEEPING FROM THIS ROUND IS A
  STATE-FILE ONE: the three pending rulings EXISTED ONLY IN THE COORDINATOR'S
  CHAT until R-408 (2026-09-02T09:35Z) -- a fresh session reading these files
  would not have known the programme was waiting on anything. They now have a
  citable anchor, and HANDOFF's PENDING USER DECISIONS carries options-as-filed
  plus the coordinator's recommendation on each, so the next reader sees the
  decision without the chat. NOTHING ELSE IN THE PROGRAMME IS BLOCKED BEHIND
  THEM and tonight's 00:06Z verdict runs unattended either way. THE THREE:
  (1) DOES 09-02 ACCRUE -- (a) CONTENT_THIN at close, frozen section-7 exclusion
  by coordinator act with a stated reason, recommend EXCLUDE; (b) genuinely
  LIVE, accrues with the blackout disclosed, recommend ACCRUE; (c)
  LIVE-by-median-collapse, where the instrument CANNOT FIRE so exclusion is a
  coordinator act resting on the reviewer's table, USER's call, recommend
  EXCLUDE. Carried with it so every future 00:06Z is mechanical: the v1 rule's
  own section 8 is still open and recommended PROSPECTIVELY -- (a) L1/L2 GOVERN,
  (b) per-coin-day granularity (R-211(3)), (c) a CONTENT_THIN day is
  INADMISSIBLE via section 7, not merely disclosed. (2) THE PHASE-2 WINNER
  (prereg 9.2/9.3): Q1's hazard head beats the incumbent's under EVERY unit,
  collapse rule and arm (12 of 24 survive Holm; level a range 0.79-0.88 lgbm,
  0.74-0.81 linear), Q3 survives at its OWN gate only, and Q4 -- THE DECISION
  METRIC -- FAILS ALL SIX CELLS (best Holm 0.12). Recommend DO NOT ADVANCE the
  composed candidate (9.2 names this case), record Q1 as the surviving COMPONENT
  of record, NO race admission for this family (9.3), next population under the
  frozen prospective 2000-draw declaration (A2); arm of record if any, LGBM.
  (3) FREEZE CONTENT-LIVENESS v2 (plans/DA_CONTENT_LIVENESS_RULE_V2_AMENDMENT.md
  section 9, released R-407): (e) adopt L3 as drafted / different constants /
  reject; (f) CONTENT_DARK governs or is reported beside; (g) re-state the 08-26
  hype coin-day under v2 or leave it as v1 recorded; (h) section 8's original
  (a)(b)(c) remain open. Recommend ADOPT AS DRAFTED, GOVERNING, EFFECTIVE
  2026-09-03 -- tonight runs v1 only and no day is re-judged -- 08-26 left as
  recorded, limit carried verbatim. WHAT v2 BUYS AND WHERE IT STILL STOPS,
  verified in the draft rather than taken from the brief: it turns "any total
  blackout is invisible" into "invisible past the FOURTH CONSECUTIVE dark day",
  because with K=7 and a median of priors THE REFERENCE ITSELF TURNS DARK once
  4 of 7 trailing days are dark; and a coin whose true volume steps down
  permanently READS DARK FOR UP TO 7 DAYS -- a declared false-positive mode and
  the stated price of a reference the day cannot move. The draft's own section-8
  heading is "Limitations, declared rather than guarded", and its reasoning is
  the one this programme keeps relearning: A GUARD THAT CANNOT FIRE IS NOT A
  GUARD, SO STATE IT RATHER THAN PATCH IT. THE v1 MODULE IS BYTE-UNTOUCHED by
  the draft (git diff 3298a1d..509859f on da_content_liveness_rule.py is EMPTY)
  -- wiring follows a freeze, never precedes one -- and the draft's checker
  passes 12/12 in my run. REVIEW STATE: NO HOLD IS OPEN ANYWHERE, in the
  filing's own words (REVIEW_DA_FORENSICS_AND_V2_DRAFT_2026-09-02.md, 98970c2,
  RELEASING DA rounds 4 and 5); RR7-1 (no per-venue regex check in the suite)
  and RR7-2 (status vocabularies extend rather than map) are FILED, NOT HOLDING,
  staged for DA's next round. Two marks from that filing worth copying: the
  reviewer RECOMPUTED E3 from raw logs with its OWN regexes, backward-walk
  dating and differencing and matched DA's artifact TO THE DIGIT -- which is
  what makes agreement evidence rather than an echo -- and it DISCLOSED THAT
  ONLY ONE MUTANT RAN this round, because the round's claims lived in
  recomputation rather than mutation. A reviewer stating where its effort did
  NOT go is rarer than one stating where it did. OPERATIONAL FACTS, checked:
  collector pid 1108125 alive, up 1d11h on collect_pm.py; the on-disk
  da_dayverdict_20260902.json is the 00:06:03Z OPEN-DAY snapshot carrying the
  LEGACY block only (no content_liveness_rule key, content_liveness.governs
  false), so TONIGHT'S CLOSING VERDICT IS THE FIRST TO CARRY THE FROZEN-RULE
  BLOCK (RR6-2); and the "greater-than" prompt line in every seat pane is Claude
  Code's DIMMED PROMPT SUGGESTION (verified ESC[2m), not an unsent dispatch --
  seats are on standby BY DESIGN and no batch is lost. REGISTER HOUSEKEEPING,
  verified as a PURE MOVE: R-396..R-408 had been inserted inside "## 7. Build
  order" and were relocated after R-395; a multiset comparison of every line
  before and after (fe27375 -> 0fc4445) shows 0 LOST and exactly 1 ADDED, the
  new R-408 header itself. FUTURE ENTRIES GO AFTER THE LAST R-ENTRY AND BEFORE
  "## 6. Build-readiness audit". CALENDAR: 00:06Z 2026-09-03 is the FIRST
  GOVERNED VERDICT (v1 governs 09-02); the coordinator verifies it and files
  either way; race G=1/5 per coin; and the venue-silence rate stays as measured
  -- 3 events in 7 days, n=3, an observed rate and never a forecast.
```
