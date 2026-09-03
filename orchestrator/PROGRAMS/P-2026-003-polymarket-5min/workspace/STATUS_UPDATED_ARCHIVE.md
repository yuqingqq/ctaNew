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

## Batch 21 — archived 2026-09-02T11:06Z (1 entry, rolling-window overflow)

Moved in the MEM round-15 true-up of R-419. Join rule as in batch 1.

```yaml

  2026-09-02T10:15Z (MEM ROUND 12 -- THE USER RULED THE ACCRUAL QUESTION, AND
  THE MASK BECAME A THREE-SEAT CONTRACT). R-409..R-412, verified at the
  artifacts. THE RULING WENT AGAINST THE COORDINATOR'S OWN RECOMMENDATION, and
  the register records both side by side: twice (R-403, R-408) the
  recommendation was EXCLUDE-IF-THIN; the USER ruled, verbatim at R-409
  (2026-09-02 ~09:48Z), "If the data quality is good over the non-blackout time,
  we should use that data." Recorded as a GENERAL disposition, not a one-day
  exception. So a blackout day is NOT thrown away: it ACCRUES on its
  non-blackout complement, with the dark windows MASKED as accounted loss --
  counted, reported, excluded from that day's forward score. IT RESOLVES v1
  SECTION 8: (a) L1/L2 GOVERN at WINDOW level, defining the mask; (b)
  granularity is PER COIN-DAY; (c) a CONTENT_THIN day is DISCLOSED AND MASKED,
  NOT INADMISSIBLE -- which SUPERSEDES the coordinator's R-403/R-408
  exclude-if-thin recommendation. Section 8(d), 08-31's status, is UNTOUCHED.
  content_thin_vetoes_HEALTHY: false is now the RULED state rather than an open
  question -- verified pinned at source with ruled_by "R-409" and the USER's
  sentence quoted in the file that consumes it -- and race_accrual_eligible
  KEEPS ITS FOUR-CONJUNCT DEFINITION unchanged. Rule 11 standing, stated in the
  entry itself: the ruling was made while 09-02 was still OPEN and BEFORE any
  forward score for any day exists. WHY THE RULE IS WORTH MORE THAN THE DAY IT
  SETTLED: excluding a day is cheap to implement and expensive in evidence --
  it throws away every good window to punish the bad ones -- while accruing on
  the complement keeps the evidence and pushes the cost onto the machinery,
  which must now identify WHICH windows were dark. That is why the v2 freeze
  stopped being housekeeping: v1 cannot see a mostly-dark day (RR6-1), so on
  such a day the complement cannot be identified at all. THE MASK IS NOW A
  CONTRACT BETWEEN THREE SEATS, R-410 amended in-band by R-411 and R-412 in nine
  minutes, each narrowing the same wiring question: PRESENCE CONSUMES (a mask,
  when present, is consumed for ANY day -- 09-01 has one and its 141 windows are
  masked at scoring) while GOVERNANCE REQUIRES (from EFFECTIVE_FROM_DAY 20260902
  a mask is REQUIRED, absent refuses, empty permitted); UNRESOLVED (not yet
  judged -- the rule block lands with the closing verdict) REFUSES AND RETRIES
  while UNJUDGEABLE (cannot be judged -- too few windows, zero median) REFUSES
  and emits routed_to "frozen rule section 7 -- coordinator exclusion with a
  stated reason" as TEXT, because the scorer never decides a disposition (rule
  14); THE PRODUCER'S COMMITTED ARTIFACT IS THE CONTRACT (RR8-1: BE's adapter
  REFUSED DA's real committed mask -- BE asserted protocol/per_coin, DA emits
  artifact/coins, substance identical; neither side wrong alone, each suite
  testing its own half, and the closure loads the REAL committed artifact);
  partial masks REFUSE via day_closed_calendar; and DA must emit an explicit
  mask for EVERY governed coin-day, empty permitted, BECAUSE ABSENCE MUST MEAN
  "THE PRODUCER DID NOT RUN", NEVER "NOTHING WAS THIN" -- without which R-409's
  accrue-on-the-complement could silently become do-not-accrue whenever a
  producer lagged. VERIFIED RATHER THAN ACCEPTED: da_blackout_mask_20260901.json
  declares artifact da_blackout_mask_v1, day_closed_calendar true, detector
  v1_FROZEN (thin_frac 0.05, module sha 7196676840304f30), and its seven
  per-coin counts -- btc 23, sol 23, eth 22, bnb 22, doge 22, xrp 20, hype 9 --
  SUM TO 141, matching its own stated total_masked_windows. TWO SEAT MARKS KEPT:
  the coordinator's dispatch asserted 09-01 had no thin windows and must emit an
  EMPTY mask when it has 141, recorded at Q-DA-201, and DA MEASURED INSTEAD OF
  COMPLYING, using a genuinely empty day (08-27) for the empty-mask control
  while proving 09-01's governing fields byte-identical -- the second time a
  seat has corrected a coordinator premise by measurement rather than obeying
  it, named against the seat in the register; and DA caught a defect of its OWN
  on real data before filing, the complement having been range(288) - masked,
  which credited the still-open 09-02 with 248 unmasked windows out of 119
  present. REVIEW STATE: NO HOLD IS OPEN ANYWHERE (R-407 stands). BE's mask-seam
  round is RELEASED at 3a1d475 with RR8-1 (HIGH), RR8-2 (MED) and RR8-3 (LOW)
  filed, and BE's fix batch is in flight; DA's producer round is IN REVIEW NOW
  (reviews/REQUEST_DA_MASK_PRODUCER_2026-09-02.md, tip 181b4fa). FOUR ITEMS WAIT
  ON THE USER, all unblocked, and the coordinator's positions below are
  RECOMMENDATIONS AND NOT RULINGS: R-408(2) the Phase-2 winner -- recommend DO
  NOT ADVANCE the composed candidate, record Q1 as the surviving COMPONENT of
  record, no race admission, arm of record if any LGBM; R-408(3) the v2 freeze
  -- recommend adopt as drafted, GOVERNING, effective 2026-09-03, 08-26 left as
  recorded; R-411(i) the minimum complement size for G-COUNTING -- recommend a
  coin-day counts toward the >=5 bar only if its unmasked complement covers >=
  50% of the calendar day (>=144/288), anchored on v1's ~60%-dark blindness so
  50% sits inside the instrument's validity rather than at its edge; and
  R-411(ii) which P1 denominator governs "quality is good" on the complement --
  recommend PER UNMASKED HOUR, since the calendar form dilutes loss by the very
  blackout it is meant to exclude (btc 09-02: 93.01 s per unmasked hour vs 25.51
  per calendar-24h, a 3.6x spread). WATCH: 00:06Z 2026-09-03 is the FIRST
  GOVERNED VERDICT, on 09-02, and the first closing verdict to carry a
  content_liveness_rule block (RR6-2); DA proved nothing governing moved, so it
  runs identically with or without this round's batches. Collector pid 1108125
  alive, up 1d12h, as of the 10:14Z clock read. NOTHING IN THE TODO TICKS THIS
  ROUND -- checked; no box covers masks, blackouts or content liveness -- stated
  rather than left as a silent empty sweep. UNCHANGED: G=1/5; the 011 family is
  12 of 24 surviving with Q4 failing; development evidence.
```

## Batch 22 — archived 2026-09-02T11:25Z (1 entry, rolling-window overflow)

Moved in the MEM round-16 true-up of R-421. Join rule as in batch 1.

```yaml

  2026-09-02T10:45Z (MEM ROUND 13 -- THE ONLY ACCRUED DAY GOT OVERWRITTEN, AND
  WAS RECOVERED). R-413..R-416 with Q-BE-225/226, Q-DA-205/206, Q-DE-22/23/24;
  verified at the artifacts, including by running both launchers. THE INCIDENT
  IS THE ITEM: at 10:16:14Z and 10:16:17Z a reviewer hand run using the WRONG
  ENVIRONMENT-VARIABLE NAMES (OUTDIR=/LOG= instead of the ones the launcher
  reads) overwrote the canonical da_dayverdict_20260901.json and _20260902.json
  with write_reason "UNATTRIBUTED hand run". Nothing governing moved, and that
  is nearly beside the point: the canonical record of the ONLY day this forward
  race has ever counted was replaced by a run nobody intended. IT WAS
  RECOVERABLE BECAUSE OF SOMETHING BUILT FOR ANOTHER REASON: DA's launcher
  echoes each verdict into its own log, so the originals came back BYTE-EXACT
  from a source that exists to make runs auditable, not to survive overwrites.
  VERIFIED AT THE ARTIFACTS RATHER THAN AT THE FILING: restored 09-01 sha
  c087d507fe433210 and 09-02 sha 09a14a7392abe224, both mtime 10:29:25Z;
  as_of 00:06:01.284484Z / 00:06:03.718835Z with write_reason "scheduled unit
  run"; restored.recovered_content_sha256 f18724e37d8f1e3f / b1d67fcd9b189489
  matching the coordinator's own 00:06Z captures; and 09-01 STILL READS ALL FOUR
  CONJUNCTS TRUE with race_accrual_eligible true, so THE R-395/R-396 ACCRUAL
  CHAIN IS INTACT AND G=1/5 STANDS. THE RESTORATION IS DELIBERATELY NOT
  BYTE-IDENTICAL (rule 13): each file carries the 00:06Z CONTENT plus a
  supersedes block naming what it replaced and a restored block naming where the
  bytes came from -- A RESTORING WRITE THAT CARRIED NO RECORD OF WHAT IT
  REPLACED WOULD BE THE SILENT VERSION OF THE SAME INCIDENT.
  prior_bytes_tracked_in_git: false is stated rather than glossed, and the
  overwritten bytes sit beside each file as
  .superseded_20260902T1016....json. THE HOLE IS SHUT IN THE WAY THAT PREVENTS
  RECURRENCE: both verdicts and both superseded copies are NOW GIT-TRACKED
  (84ec1a1); they were not when the overwrite happened, which is why "recovered
  from a log echo" was the best available option rather than "restored from
  version control". THE LAUNCH-INVARIANCE CLASS (CO-1/2/3) IS CLOSED across BE
  (1 module) and DE (5 modules, CO-2 closed AS A CLASS rather than per-module),
  with DA's modules passing both launchers at 235/19. CO-1 IS THE ONE WORTH
  REMEMBERING: the forward scorer imported the frozen rule bare, that import
  fails under python3 -m, and an "except Exception: EFFECTIVE_FROM_DAY = None"
  fallback made governed FALSE FOR EVERY DAY -- so a governed day with no mask
  would have scored WHOLE, SILENTLY. Green under the script-dir launch BE used,
  rc=1 under the package launch. A FALLBACK THAT CONVERTS A REQUIREMENT INTO
  PERMISSION IS WORSE THAN A CRASH, because the crash is visible. Confirmed
  closed by running both launches here: harmful_forward_scorer --selftest rc=0,
  60 checks, each way. REVIEW AND ROUND STATE: DA round 1 (mask producer)
  RELEASED with RR9-1/2/3 closed in round 7; BE round 2 (mask consumer) RELEASED
  with RR10-1 (LOW) open and its fix batch dispatched; the DE round IN REVIEW at
  21f4edf; DA round 2 QUEUED at 770e5ee. RR10-1 IS A CONTROL THAT CANNOT FAIL,
  this programme's recurring shape: BE's pre-governed control asserts 09-01 "now
  reads CONTENT_THIN", true when written and made FALSE by the 10:29Z
  restoration, and swapping the control's day leaves the suite 60/60 green; the
  closure anchors it to a FIXTURE verdict rather than a live day, which is the
  only version that stays true. ONE MISS RECORDED AGAINST THE COORDINATOR'S OWN
  SEAT (R-413): DE round 4 sat verified SIX HOURS LATE, named as a breach of
  R-381's no-idle clause by the coordinator rather than a DE problem -- the
  third coordinator self-correction in two days, after a premise corrected by
  DA's measurement and a breadth figure corrected by mine. TONIGHT: the 00:06Z
  09-03 run is the FIRST GOVERNED VERDICT (on 09-02) and the FIRST CANONICAL
  MASK; BE REFUSES 09-02 ON GOVERNANCE until that mask lands -- governed, thin,
  mask absent, refusing by name, which is the ruled behaviour and not a failure.
  09-02 carries the 01:35-04:55Z Polymarket-side blackout (R-405) and ACCRUES ON
  ITS COMPLEMENT per R-409, but THE ACCRUAL CALL IS THE USER'S. IN FLIGHT: BE's
  RR10-1 fixture batch, DA round 8 (governed-verdict preflight, predicates
  only), DE round 7 (supply-to-seam bridge with a parameter ratification ref),
  and the reviewer on DE. Collector pid 1108125 alive as of the 10:44Z clock
  read. USER DECISIONS PENDING, UNCHANGED AND FOUR: R-408(2) the Phase-2 winner,
  R-408(3) the v2 freeze, R-411(i) the minimum complement for G-counting, and
  R-411(ii) the P1 denominator on the complement. UNCHANGED: the 011 family is
  12 of 24 surviving with Q4 failing; development evidence.
```

## Batch 23 — archived 2026-09-02T11:36Z (1 entry, rolling-window overflow)

Moved in the MEM round-17 true-up of R-422. Join rule as in batch 1.

```yaml

  2026-09-02T10:55Z (MEM ROUND 14 -- A POPULATION RATIFIED, A DEVIATION
  ACCEPTED, AND A SCHEDULER I COULD NOT FIND). R-417/R-418 with Q-BE-227,
  Q-DA-207, Q-DE-25 and Q-MEM-1; verified at the artifacts, including by running
  the suites. R-418 RATIFIES A POPULATION BY REFUSING TO CHOOSE ONE: for a
  forward-race day the replay/scoring set is EVERY window
  de_admissible_windows.supply(D, present) emits -- present read from the day's
  own market ledger, minus the windows DA's committed mask masks -- WITH NO
  STRATIFIED OR CAPPED SELECTION (select_stratified stays a research-day
  instrument). That is the point: the set is a FUNCTION OF TWO COMMITTED
  ARTIFACTS, fixed by the supply's mask_identity_hash, so a receipt stamped
  ratification_ref R-418 reports WHICH WINDOWS IT RAN OVER rather than which it
  picked. It is R-409 applied, introduces no number, is the coordinator's
  R-ADMISS act under EV_REPLAY_PLAN section 2, and is USER-REVOCABLE -- and it
  explicitly does NOT ratify the G-counting minimum (R-411(i)), the P1
  denominator (R-411(ii)), any accrual call, or any Phase-2 admission
  (R-408(2)), all of which stay the USER's. DE's bridge was built BEFORE the ref
  existed and carries a fixture ref R-0 over 1,875 specs -- the same 1,875 =
  288 x 7 - 141 the supplier produced on the real 09-01 mask; receipts carrying
  R-418 re-stamp if the USER overrules any part of it. THE GAP NAMED IN R-417
  SECTION 2 IS NOW CARRIED: DA's TWO-LEG ADMISSION DEVIATION IS ACCEPTED
  (R-416 section 3(a)). DA was ordered to gate the nightly governed path on a
  SINGLE declared variable and implemented TWO LEGS instead -- cgroup identity
  OR DA_MIDNIGHT_MODE=production -- and the acceptance inverts the usual
  direction: it WEAKENS NOTHING (the hand-run path is exactly the single-leg
  form, and the identity leg is the same test write_reason already relies on)
  and it REMOVES AN OUTAGE MODE THE ORDER WOULD HAVE INTRODUCED, namely a
  nightly governed path that refuses when one new variable goes missing with
  nothing running to say so. Red-first evidence on the incident's own shape: a
  bare run gives rc 6; OUTDIR=/tmp/x LOG=/tmp/y -- the wrong names that caused
  the 10:16Z overwrite -- gives rc 6; one of the pair gives rc 5 under the older
  guard; AND THE LOG MTIME AND BOTH VERDICT SHAS WERE UNCHANGED AFTER ALL THREE,
  so the refusal precedes the log header and a rejected run cannot touch the
  artifacts. RR10-1 IS CLOSED (e56f70a) IN THE RIGHT SHAPE: the pre-governed
  control is now a FIXTURE PAIR derived from the frozen rule's own
  EFFECTIVE_FROM_DAY -- one day before passes, the same fixture on the governed
  day refuses by name -- plus a member that makes the fixture load-bearing, so
  it no longer depends on what any live day happens to read, which is exactly
  what made the old version unfailable. Verified here: 63 checks, rc 0, under
  both launchers. AND ONE THING I COULD NOT VERIFY, RECORDED RATHER THAN
  SMOOTHED OVER: R-417 states that the 00:14Z governed-verdict preflight is
  carried by "the cron one-shot". I looked. As of this clock read the user
  crontab holds THREE non-comment entries (a convexity monthly retrain and two
  okxSolver scans) and NONE mentions the preflight, pm_research, or a 00:14
  slot; there is no at binary; and the only systemd timer in this family is
  da-midnight-verify.timer at 00:06:00Z. THE VERDICT IS SCHEDULED; THE PREFLIGHT,
  AS FAR AS THIS BOX SHOWS, IS NOT. It may simply be owed by DA round 9, whose
  standby is described as carrying the 00:06Z timed task -- but "an instrument
  that runs first" and "an instrument nothing runs" are different states, and
  this programme has paid for that distinction three times (the unwired
  content-liveness rule, the unwired incumbent loader, six evaluator functions
  with no call sites). The preflight ITSELF is sound: da_governed_verdict_
  preflight --selftest is 30 checks rc 0 here, it is read-only and
  predicates-only, and it carries decides_nothing as a field. A DISAMBIGUATION
  MADE ON REQUEST: the older HANDOFF paragraph "the edit exists only in the
  installed unit" is provenance about pm-evaluation-pipeline.service and a SCOPE
  NOTE now sits above it so no reader takes it for the midnight verifier, which
  I re-checked independently -- da-midnight-verify.service installed is
  BYTE-FOR-BYTE IDENTICAL to live/pm_research/systemd/da-midnight-verify.service,
  stronger than the non-comment-lines claim, with
  Environment=DA_MIDNIGHT_MODE=production, DropInPaths empty, and the timer next
  elapsing 2026-09-03 00:06:00 UTC. REVIEW AND ROUND STATE: DA round 1 RELEASED;
  BE round 2 RELEASED with RR10-1 CLOSED, the review riding BE round 3; DE
  rounds 4-6 RELEASED (7a48333, no hold, all seven scopes by execution) with
  RR11-1 LOW open to DE round 8; DA round 2 IN REVIEW at 770e5ee. IN FLIGHT: BE
  round 3, the production run path with scores SEALED and counts and refusals
  only (Q-BE-228); DE round 8, RR11-1 plus the ratification checker and the
  proposed block format (Q-DE-26); DA round 9 on DELIBERATE STANDBY with the
  00:06Z timed task (Q-DA-208); the reviewer on DA round 2. USER DECISIONS NOW
  FIVE: R-408(2) the Phase-2 winner, R-408(3) the v2 freeze, R-411(i) the
  minimum complement for G-counting, R-411(ii) the P1 denominator, AND THE 09-02
  ACCRUAL CALL AFTER TONIGHT -- 09-02 carries the 01:35-04:55Z Polymarket-side
  blackout and accrues on its complement per R-409, but THE CALL IS THE USER'S
  and R-418 is explicit that ratifying the population does not make it.
  Collector pid 1108125 alive as of this clock read. UNCHANGED: G=1/5; the 011
  family is 12 of 24 surviving with Q4 failing; development evidence.
```

## Batch 24 — archived 2026-09-02T11:47Z (1 entry, rolling-window overflow)

Moved in the MEM round-18 true-up of R-423. Join rule as in batch 1.

```yaml

  2026-09-02T11:06Z (MEM ROUND 15 -- A SENTENCE ABOUT A RATIFICATION PASSES AS
  ONE, AND MY 00:14Z WATCH RESOLVES INTO SOMETHING MORE INTERESTING THAN THE
  WATCH). R-419 with DE round 8 (575f076, Q-DE-26) and the reviewer's DA round-2
  filing (1e6624a); verified at the artifacts and at the box. CO-4 IS THE
  FINDING, AND IT IS AT THE REGISTER RATHER THAN IN ANY MODULE: a fixture entry
  titled "MEM round 14 verified; recap of state", whose body is a single RECAP
  SENTENCE naming R-418's population and ending "Nothing here ratifies
  anything", returns VERIFIED with binding_source PROSE and all five decidable
  checks True. A CHECKER THAT BINDS FROM PROSE CANNOT TELL A RATIFICATION FROM A
  SENTENCE ABOUT ONE -- CLAUDE.md rule 16 at the register level, where grep hits
  on vocabulary are not references. What makes it sharp rather than cute is that
  THE EXPOSURE GROWS WITH EVERY SWEEP: coordinator entries, and MEM true-ups
  like this one, necessarily recite that vocabulary. Four smaller holes noted on
  the same read: day_in_scope evaluates scope_from ONLY (scope_to is parsed and
  ignored, so a block scoped to 09-01 reads True for 09-02); the block's ref is
  NOT checked against the entry heading; sampling != NONE LOWERS verified but
  does not refuse; and refusals is a dead list. DE ROUND 9 (Q-DE-27) IS IN
  FLIGHT to make prose binding admissible for the grandfathered R-418 only, with
  the R-9001 fixture as the live control. THE FORMAT IS ADOPTED (a format is the
  coordinator's and introduces no number): every R-ADMISS entry from R-419 on
  carries a fenced ratification block with ref, kind, population, sampling,
  present_source, scope_days, scope_from, scope_to, revocable_by and supersedes,
  with ref required to equal the heading's ref. R-419 SUPERSEDES R-418 IN-BAND
  (rule 13; R-418 stays as provenance, never edited) WITH THE CONTENT UNCHANGED
  -- the same population, the same no-sampling, the same list of what it does
  NOT ratify (R-411(i), R-411(ii), any accrual call, any Phase-2 admission, all
  the USER's) -- and the one field prose could not carry is now bound:
  scope_from 20260901, the first accrued race day, stated as a RESTATED FACT
  rather than a new number, with scope_to null, open until the USER closes or
  revokes it. REFS IN FLIGHT: BE round 3 was dispatched under R-418 and its
  SEALED SCRATCH RECEIPTS STAND AS PROVENANCE under that ref; from BE round 4
  the ref is R-419; DE's R-0 stays a fixture ref. MY ROUND-14 WATCH IS RESOLVED,
  AND THE RESOLUTION IS MORE INTERESTING THAN THE WATCH: the "cron one-shot"
  existed -- as a CLAUDE CODE SESSION-LOCAL SCHEDULER ENTRY inside the
  coordinator session (78375088), invisible to crontab -l and systemctl --user
  list-timers BY CONSTRUCTION, and DYING WITH ITS SESSION. DA's standby wait
  (bbp5f4bni) is the same kind of object, and DA's own filing records TWO PRIOR
  WAITS KILLED MID-FLIGHT. So both legs meant to carry the 00:14Z check lived
  inside processes no box-level tool can see -- a different failure from "nobody
  built it": IT IS SCHEDULING THAT CANNOT BE AUDITED FROM THE BOX IT RUNS ON,
  and the only reason it surfaced is that a state file asked where the scheduler
  was. The coordinator confirms MEM's reading was correct as stated: nothing at
  box level ran the preflight. A BOX-LEVEL LEG NOW EXISTS, verified by me at the
  box rather than taken: co-preflight-20260902.timer ->
  co-preflight-20260902.service, next elapse Thu 2026-09-03 00:14:00 UTC,
  running the read-only preflight on 09-02 with stdout to
  ~/.local/state/pm-co/preflight_20260902.json, and its POSITIVE CONTROL
  EXECUTED on 09-01 (rc 1, classification PRE_GOVERNED_ARTIFACT, output at
  preflight_probe_20260901.json, derived/ digest unchanged across the run).
  THREE LEGS NOW, ONE AT BOX LEVEL, and the verdict itself is written by the
  scheduled unit regardless of all three. ONE PROPERTY KEPT IN VIEW: the timer
  is TRANSIENT (systemd-run, Persistent=no), so it is gone once it fires and
  would not survive a restart of the user manager before 00:14Z -- correct for a
  one-shot, not a standing schedule. REVIEW TABLE: DA round 1 RELEASED; BE round
  2 RELEASED with RR10-1 closed, its review riding BE round 3; DE rounds 4-6
  RELEASED and DE round 8 VERIFIED with RR11-1 CLOSED; DA ROUND 2 IS FILED AT
  1e6624a AND IN VERIFICATION, NOT RELEASED -- the reviewer's filing states DA
  round 7 released, and the coordinator's verification of that filing comes in
  the next R-entry, so the table does not call it released yet. BE round 3's
  review request queues when Q-BE-228 lands; DE rounds 7-9's when Q-DE-27 lands.
  RR12-1 FROM THE DA FILING WILL BITE THE WORKTREES: da_blackout_mask.REPO is
  hardcoded to the SHARED tree while module_sha256_prefix comes from __file__,
  so a run from ~/ctaNew-wt-rev reports the shared tree's HEAD rather than the
  worktree's and a dirtied worktree module still reads
  tree_dirty_on_producing_files FALSE; the reviewer's own rehearsal therefore
  executed shared-tree code, where a mutation had no effect while the same one
  in-process fired immediately. Nothing shipped is wrong, but per-seat worktrees
  were adopted three rounds ago (rule 19), so a provenance pair that disagrees
  about which tree it is in is a LIVE hazard. AND A METHOD NOTE WORTH COPYING
  from the same filing: a same-size mutate/restore inside one second left STALE
  BYTECODE THAT READ EXACTLY LIKE A SURVIVING MUTANT -- recorded so the next
  mutation run does not file it as one. IN FLIGHT: BE round 3 (production run
  path, scores SEALED, counts and refusals only, Q-BE-228); DE round 9 (CO-4
  plus the block-format checker, Q-DE-27); DA on standby (Q-DA-208, after
  00:06Z); the reviewer IDLE pending the next request. USER DECISIONS UNCHANGED
  AND FIVE: R-408(2) the Phase-2 winner, R-408(3) the v2 freeze, R-411(i) the
  minimum complement for G-counting, R-411(ii) the P1 denominator, and the 09-02
  accrual call after tonight. UNCHANGED: G=1/5; the 011 family is 12 of 24
  surviving with Q4 failing; development evidence.
```

## Batch 25 — archived 2026-09-02T11:56Z (1 entry, rolling-window overflow)

Moved in the MEM round-19 true-up of R-424. Join rule as in batch 1.

```yaml

  2026-09-02T11:25Z (MEM ROUND 16 -- THE FREEZE IS A COMMIT, AND THE TREE WALKED
  AWAY FROM IT). R-421 swept; verified at the artifacts. BE ROUND 3 IS VERIFIED
  AND ITS REFUSAL IS THE FINDING: the frozen candidate
  harmful_reduced_fine_candidate_v1.json binds manifest sha eb8733da2c8e2126 and
  builder sha 0091fe75c38af79e, the manifest binds EIGHT reproducibility_anchor
  entries plus collector_runs.jsonl as state_at_build, and EVERY BOUND SHA
  EQUALS THE BLOB AT COMMIT 1b53929 (2026-08-26T10:49:55Z, "Authorised by the
  user's explicit yes in BE's pane", MULTIPLICITY 2) -- THAT COMMIT IS THE
  FREEZE (rule 12). THEN THE WORKING TREE MOVED THE ANCHORS IN NINE COMMITS:
  f30cf26 (08-26 15:45Z), f46f350, a410c07, 3f538a3, b6168b9 (08-27), 46ab455
  (08-28), 2e1204f (08-29), 851edaf (09-01 09:12Z), and the manifest's text at
  608d71a (08-26 14:47Z). So the code in the tree is NOT the code the freeze
  bound, and BE's gate is right to refuse; nothing is re-stamped and no frozen
  artifact is edited (rule 13). THE MANIFEST'S PROSE STILL READS "NOT FROZEN" /
  weights PENDING AND MUST BE LEFT ALONE: it was written BEFORE the freeze and
  deliberately never re-stamped, because A FREEZE IS A COMMIT, NOT A STATUS
  STRING (rule 12/13) -- the gate reads status from the CANDIDATE, which says
  FROZEN. Recorded in watch-out-for as "carry these exactly", because it is
  precisely the line a future sweep would tidy. The manifest's hashes block has
  been BYTE-IDENTICAL since. THE FROZEN BYTES STILL EXIST, retrievable from
  1b53929, which is why the disposition is a CHOICE rather than a loss -- AND IT
  IS THE SIXTH USER DECISION, ESCALATED AND DECIDED BY NO ONE (R-421 section 3).
  Plan section 10 step 9 says the frozen set is scored UNCHANGED and the frozen
  set is the commit's bytes, so either (a) the race runs on the frozen bytes:
  BE round 4 materialises them from 1b53929, verifies each sha BEFORE import,
  imports from the run dir and never from the tree, with the driver at HEAD as
  harness, the receipt recording frozen_commit, per-anchor shas, harness commit,
  the transitive import closure and BY NAME every module in that closure that is
  NOT an anchor and has moved since -- OUTPUT TO SCRATCH, SEALED, AN ESTIMATE
  AND NOT A RACE SCORE; or (b) the candidate is RE-FROZEN AT HEAD, which is a
  NEW CANDIDATE, MULTIPLICITY 3, and a NEW FREEZE COMMIT that only the USER
  authorises (rule 12; R-409's "any other things need my decision"). A seat
  choosing between those would be choosing the programme's own baseline. DE
  ROUND 9 VERIFIED (b98421d, Q-DE-27): 42 checks both launchers; R-419 binds
  from BLOCK with day_in_scope True and unverifiable []; R-418 refuses "FOR A NEW
  RUN ... SUPERSEDED by R-419" with refusal_scope keeping receipts as
  provenance; the R-9001 recap fixture refuses "no ratification block" -- CO-4
  CLOSED. CO-5 (LOW) OPEN: a block with NO scope_to line returns verified True
  with day_in_scope None and unverifiable ['day_in_scope'] -> DE round 10
  (Q-DE-28, in flight). THE REVIEWER'S REVIEW OF THE COORDINATOR'S OWN ACTS
  (1384ec5) IS VERIFIED, no hold, with four findings dispositioned. CO-R1
  (MEDIUM, live, reproduced): on the OPEN day 09-02 the ledger runs AHEAD of the
  tape -- 137 vs 135 windows per coin, the 14 ledger-only entries being the
  11:15Z and 11:20Z starts -- while on the CLOSED 09-01 the two agree 288/288.
  Already enforced (the driver refuses an open day at gate 1; the bridge refuses
  any supplied window with no archive), and NO RESTATEMENT TONIGHT because
  scope_days FORWARD_RACE_DAYS already binds FINISHED through the forward-race
  rule and a new block would supersede R-419 hours before the first run that
  stamps it; DE round 10 makes closure a DECIDED predicate. CO-R2 (MEDIUM)
  STATED AND CLOSED: the format was declared at 11:03Z and enforced from 11:09Z,
  and NO receipt stamped R-419 in that interval (round 3's stamp R-418 and are
  sealed scratch) -- a window that turned out to be empty, STATED rather than
  assumed empty. CO-R3 (MEDIUM) -> DE round 10: supersession evaluated against
  the RECEIPT'S OWN STAMP (as_of_utc and harness commit), so a receipt written
  before a superseding entry stays verified as provenance BY COMPUTATION, NOT BY
  A SENTENCE. CO-R4 (LOW, reproduced with a correction) LANDS ON THE VERY TIMER
  I VERIFIED LAST ROUND: on a day with no verdict the preflight raises
  PreflightRefused UNCAUGHT from main(), the traceback goes to STDERR and STDOUT
  IS EMPTY -- so preflight_20260902.json WOULD BE A ZERO-BYTE FILE if the
  verdict is absent at 00:14Z -- and its rc 1 COLLIDES with the ordinary
  n_failing > 0 return. A ZERO-BYTE JSON TOMORROW MORNING MEANS REFUSED, NOT
  CLEAN, and the reason would be in the journal rather than the file; I have
  attached that warning to the timer's own note, because the file is what a
  reader finds first and an empty file reads like nothing happened. This is the
  programme's standing shape -- absence reading as a pass -- arriving at the one
  instrument added to close an audit gap. DA round 10 (after tonight) gives it a
  JSON refusal object on stdout and a distinct rc. THE REVIEWER ALSO WITHDREW
  ONE OF ITS OWN LEGS (the V-from-$0 leg), accepted: the RR12-1 fix SPLITS
  provenance from execution -- provenance follows the bytes (REPO from __file__,
  the run records which tree it exercised), execution stays on the code that
  runs. REVIEW TABLE: DA rounds 1-2 RELEASED with RR12-1 to DA round 10; BE
  round 2 RELEASED; BE ROUNDS 3-4 REVIEWED TOGETHER, DELIBERATELY, because the
  run path is not finished until it executes the frozen bytes and reviewing the
  refusing half alone would review a frame; DE rounds 4-6 RELEASED and DE rounds
  7-9 REVIEW REQUEST FILED (REQUEST_DE_ROUNDS_7-9_2026-09-02.md, the reviewer on
  it at b98421d); the coordinator's acts REVIEWED. IN FLIGHT: BE round 4
  (Q-BE-229), DE round 10 (Q-DE-28), the reviewer on DE 7-9, DA on standby
  (Q-DA-208 after 00:06Z; round 10 after tonight = the RR12-1 split,
  identity-only admission log, and CO-R4). USER DECISIONS NOW SIX: R-408(2) the
  Phase-2 winner, R-408(3) the v2 freeze, R-411(i) the minimum complement for
  G-counting, R-411(ii) the P1 denominator, the 09-02 accrual call after 00:06Z,
  AND THE FREEZE DISPOSITION. R-419 remains revocable by the USER. UNCHANGED:
  G=1/5; the 011 family is 12 of 24 surviving with Q4 failing; development
  evidence.
```

## Batch 26 — archived 2026-09-02T12:02Z (1 entry, rolling-window overflow)

Moved in the MEM round-20 true-up of R-425. Join rule as in batch 1.

```yaml

  2026-09-02T11:36Z (MEM ROUND 17 -- A VOCABULARY MISS IS NOT AN ABSENCE).
  R-422 swept; verified at the artifacts and, on the central claim, at the
  source myself. DE ROUND 10 VERIFIED (2282e5c, Q-DE-28): CO-5, CO-R1's checker
  half and CO-R3 all CLOSED, and DE's own addition -- require_verified REFUSES a
  PROVENANCE result -- ACCEPTED. THE DE ROUNDS 7-9 REVIEW (b4da910) IS VERIFIED
  AND RELEASED, with DE-R1..R4 reproduced at 2282e5c and routed to DE round 11
  (Q-DE-29, in flight). AND ONE CLAIM IN THAT FILING DID NOT REPRODUCE, WHICH IS
  THE ITEM OF THE ROUND. The review stated that R-421 section 6's "the driver
  already refuses ... ledger-only windows" "does not hold at the artifact: no
  layer in the chain reads the tape". Executed on the working tree at 11:32Z,
  it does not reproduce: be_forward_day.selected_from_specs, given two real
  09-01 specs plus one for a ledger-only window with no archive yet, REFUSES --
  "1 supplied windows have no archive or no token map ... R-418 scores the
  complement WHOLE; dropping windows here would silently re-select" -- and the
  same for a slug that can never exist. I VERIFIED THE MECHANISM AT SOURCE
  INDEPENDENTLY: the gate reads the tape through fi._archive_paths() and
  fi.token_map() at be_forward_day.py:491-506, with the refusal asserted by its
  own control at :1055. WHY THE REVIEWER MISSED IT IS THE TRANSFERABLE PART:
  its search was for scan_day and raw/, and THE ARCHIVE INDEX ANSWERS TO NEITHER
  NAME. A GREP FOR VOCABULARY IS NOT A REFERENCE -- IN EITHER DIRECTION. This
  programme has recorded the forward version three times (a vocabulary HIT is
  not a reference); THIS IS THE MIRROR AND IT IS THE MORE DANGEROUS HALF, because
  a false positive from grep gets caught when someone opens the file while a
  false NEGATIVE produces a confident "no layer does this", which reads like a
  finding and travels as one -- this one reached the register before it was
  executed against. WHAT THE REVIEWER DID ESTABLISH IS TRUE AND NARROWER, and I
  confirmed both halves: DE's supply() and the seam bridge genuinely DO NOT read
  the tape (no _archive_paths, token_map or scan_day in either module), so a
  tape-less window IS supplied (1,876); and the driver's refusal sits at
  selected_from_specs, AFTER the frozen-contract gate, which on the current tree
  has never been reached in a real run. So THE PROTECTION IS REAL AND IT IS IN
  THE WRONG PLACE, which is a different statement from "there is no protection".
  DISPOSITION, accepting the reviewer's recommendation IN ITS REFUSE FORM ONLY:
  BE round 5 moves the ledger-vs-tape comparison into the POPULATION GATE
  (present_from_ledger, the receipt carrying ledger_minus_tape per coin BY NAME)
  and REFUSES ON ANY DIFFERENCE -- IT NEVER INTERSECTS, because intersecting
  would look like the helpful fix and would SILENTLY RE-SELECT THE RATIFIED
  POPULATION: R-418/R-419 fixed the complement as WHOLE, and a quiet
  intersection is exactly the kind of selection a ratification exists to forbid.
  R-419's TEXT IS UNCHANGED and the USER may restate or revoke it; the checker's
  day_closed (DE round 10) stays as the visible half. AND IT CORRECTS ONE OF MY
  OWN LINES: my round-16 entry attributed the refusal to "the bridge"; IT IS THE
  DRIVER. Corrected in place with the new section as its reason -- the
  attribution mattered here precisely because the residual finding turns on
  which layer reads what. REVIEW TABLE: DA rounds 1-2 RELEASED (RR12-1 to DA
  round 10); BE round 2 RELEASED and BE ROUNDS 3-4 REVIEWED TOGETHER when round
  4 lands, deliberately, because the run path is not finished until it executes
  the frozen bytes; DE rounds 4-6 RELEASED; DE ROUNDS 7-9 RELEASED; DE ROUND 10
  REVIEW REQUEST FILED (REQUEST_DE_ROUND_10_2026-09-02.md, the reviewer on it at
  2282e5c); the coordinator's acts REVIEWED. IN FLIGHT: BE round 4 (Q-BE-229),
  DE round 11 (Q-DE-29), the reviewer on DE round 10, DA on standby (Q-DA-208
  after 00:06Z); BE round 5 QUEUES the population-gate comparison and
  require_verified(). USER DECISIONS UNCHANGED AND SIX: R-408(2) the Phase-2
  winner, R-408(3) the v2 freeze, R-411(i) the minimum complement for
  G-counting, R-411(ii) the P1 denominator, the 09-02 accrual call after 00:06Z,
  and the freeze disposition. R-419 remains revocable by the USER. UNCHANGED:
  G=1/5; the 011 family is 12 of 24 surviving with Q4 failing; development
  evidence.
```

## Batch 27 — archived 2026-09-02T12:11Z (1 entry, rolling-window overflow)

Moved in the MEM round-21 true-up of R-426. Join rule as in batch 1.

```yaml

  2026-09-02T11:47Z (MEM ROUND 18 -- I CITED A DIRTY TREE, AND THE FREEZE HAS AN
  ANCHOR WITH NO COMMIT). R-423 swept. THE CORRECTION IS MINE: Q-MEM-5 cited
  be_forward_day.py:491-506 and :1055 as evidence that the driver reads the
  tape, but those line numbers come from BE's UNCOMMITTED round-4 WORKING TREE,
  not from 805fd39, the commit I named -- and :1055 cannot exist there at all,
  since that file is 810 LINES at the commit and 1,160 in the tree. THE IDENTITY
  CLAIM SURVIVES; THE CITATION DID NOT. Verified now at the commit itself:
  selected_from_specs reads fi._archive_paths() and fi.token_map() and carries
  the same refusal at 805fd39:252-275, so the correct citation is :252-275 at
  the commit while :487/:1055 describe a tree nobody else has. THIS IS RR12-1
  LANDING ON MY OWN FILING -- that finding is about provenance and execution
  disagreeing over which tree they are in, and I verified against whatever was
  in the tree and then reported it under a commit hash. A LINE NUMBER IS A CLAIM
  ABOUT A SPECIFIC ARTIFACT, AND MINE NAMED THE WRONG ONE: when a citation
  carries a commit, read the file FROM that commit. The register has also
  ADOPTED THE MIRROR RULE -- a vocabulary miss is not an absence -- into its own
  vocabulary. THE FACT THAT MOVES A PENDING DECISION: THE FREEZE HAS AN ANCHOR
  WITH NO COMMIT. BE round 4 reports -- REPORTED, NOT VERIFIED, AND NOT LANDED;
  every line of it becomes verified only when the commit lands -- that (i) the
  frozen code derives its DATA ROOT FROM __file__, so materialising anchors into
  the run dir SILENTLY REPOINTED flow_intensity.PM and EMPTIED THE ARCHIVE
  INDEX, fixed with a symlink (the freeze's code, today's data, both named in
  the receipt) plus a PROBE THAT REFUSES if the root does not resolve or the
  index is empty; (ii) materialising the DATA anchor SHADOWED that symlink, so
  ONLY CODE ANCHORS are materialised; and (iii) the data anchor
  harmful_exposure_rows_v3_eraB.json IS NOT IN THE FREEZE COMMIT AT ALL, because
  data/ is gitignored -- its bytes match the manifest ON DISK and are verified
  BY CONTENT with the source named. FACT (iii) IS AN ADDITION TO THE RECORD OF
  THE SIXTH USER DECISION, STATED FOR THE USER AND DECIDED BY NO ONE: the frozen
  set was described as "the commit's bytes", which is exact for the CODE
  anchors, but THE DATA ANCHOR HAS NO COMMIT TO BE FROZEN AT -- it is frozen BY
  MANIFEST SHA ONLY. So "race on the frozen bytes at 1b53929" resolves to code
  from the commit and data from a file whose only binding is its hash; that
  decides nothing and makes the option honest about what it is. THE DE ROUND 10
  REVIEW IS RELEASED (922bff6) WITH THE REVIEWER'S OWN IN-BAND CORRECTION (rule
  13; the released review untouched): it reproduced selected_from_specs refusing
  a tape-less window by name, NAMED ITS OWN ERROR AS A GREP ESTABLISHING AN
  ABSENCE, and WITHDREW the "intersect" half of its recommendation -- accepted,
  with the disposition staying refuse-on-ledger_minus_tape, never intersect. Its
  EXPIRY NOTE is recorded as method: A FIXTURE WHOSE ESSENTIAL PROPERTY IS "NOT
  WRITTEN YET" EXPIRES WITHIN MINUTES, so reproduce the mechanism on the current
  in-flight window AND a future one. DE10-R1 (MEDIUM) STANDS -> DE ROUND 12
  (Q-DE-30, in flight), and it is silent in BOTH directions: the checker
  compares timestamps LEXICOGRAPHICALLY AS STRINGS, so now_utc="zzzz" reads
  day_closed True and verified True while scope_to "not-a-date" reads
  day_in_scope True, verified True, unverifiable [] -- garbage sorts PERMISSIVE
  for now_utc/scope_to and RESTRICTIVE for scope_from, and NONE OF THE THREE
  SURFACES. Round 12 parses to datetimes and refuses an unparsable value BY NAME
  with a falsifier per field IN BOTH DIRECTIONS, because a fix tested only
  against permissive garbage would miss the restrictive half. DE ROUND 11
  VERIFIED (d07d901, Q-DE-29): DE-R1..R4 all CLOSED -- ratification 66,
  admissible 53, seam 69 checks under both launchers, checker audit 14 paths
  with survivors [] and all_load_bearing True -- and DE's TWO DELIBERATE
  SEPARATIONS ACCEPTED (kind kept outside the vocabulary loop; the
  sampling/population contradiction check kept apart from the vocabulary,
  because restricting sampling to NONE would have hardcoded that no sampled
  ratification can exist, contradicting KNOWN_POPULATIONS). Nothing under review
  moved: R-419 on 09-01 verified_for_new_run True and day_closed True; R-418
  stamped 10:30Z provenance True. ONE OPERATIONAL NOTE WORTH GENERALISING: BE's
  mutation harness was SIGKILLED mid-mutation by an outer timeout, its finally
  never ran, and a mutant (F17) STAYED APPLIED IN THE TREE until the next
  selftest caught it -- CLEANUP THAT LIVES ONLY IN finally DOES NOT SURVIVE
  SIGKILL; BE added a backup. AND A TIMESTAMP FOR THE RECORD: R-423's heading
  reads 11:47Z while the commit carrying it is 11:46:34Z, composed 26 s ahead --
  THE COMMIT TIME IS AUTHORITATIVE. Recorded, not adjudicated; the sixth
  instance of the stamp-vs-clock class, and the register now discloses them
  itself. REVIEW TABLE: DE round 10 RELEASED; DE round 11 REVIEW DISPATCHED at
  d07d901 (BE rounds 3-4 still queue behind it, to be reviewed together when
  round 4 lands); DA round 10 after tonight. OPEN FINDINGS: DE10-R1 -> DE round
  12; RR12-1 and CO-R4 -> DA round 10. IN FLIGHT: BE round 4 (Q-BE-229, not
  landed), DE round 12 (Q-DE-30), the reviewer on DE round 11, DA on standby
  (Q-DA-208 after 00:06Z); BE round 5 queues the population-gate comparison and
  require_verified(). USER DECISIONS UNCHANGED AND SIX, with section 4(iii)
  added to the sixth's record. R-419 remains revocable by the USER. UNCHANGED:
  G=1/5; the 011 family is 12 of 24 surviving with Q4 failing; development
  evidence.
```

## Batch 28 — archived 2026-09-02T12:20Z (1 entry, rolling-window overflow)

Moved in the MEM round-22 true-up of R-427. Join rule as in batch 1.

```yaml

  2026-09-02T11:56Z (MEM ROUND 19 -- THE RULING NAMED NO ITEM, AND THIS FILE WAS
  THE REFERENT). R-424 swept; the executed freeze verified at the artifact. THE
  USER RULED FOUR DECISIONS WITH SIX WORDS -- "Proceed according to your
  recommendation" (verbatim, ~11:49Z) -- and R-424 section 1 resolves the scope
  by NAMING THIS FILE: the recommendations on record at the moment of the ruling
  were HANDOFF's six-item table at 79f2db5, four rows each carrying an explicit
  "coordinator's recommendation", mirrored in R-408 and R-411. FOUR ADOPTED, ONE
  DELIBERATELY NOT REACHED. THE FREEZE DISPOSITION WAS NOT REACHED BECAUSE THE
  TABLE SAID IT HAD NO RECOMMENDATION, describing it as "decided by no one" with
  two options and nothing advised beside either -- and that is the whole value of
  marking a recommendation AS a recommendation: had that row carried a
  suggestion in the same voice as the other four, a six-word ruling would have
  silently adopted A NEW CANDIDATE, MULTIPLICITY 3, AND A NEW FREEZE COMMIT, the
  one decision the register insists only the USER can make. THE FOUR RULED:
  R-408(2) THE PHASE-2 WINNER -- the composed candidate DOES NOT ADVANCE (9.2,
  Q4 fails), Q1_arrival is the SURVIVING COMPONENT OF RECORD, NO RACE ADMISSION
  (9.3, multiplicity unchanged), the next population runs under A2 as frozen
  (2,000 draws one-sided; this family stays at 500 with its floor disclosure),
  arm of record if one is ever named composed_lgbm, and PHASE-4 GRIDS STAY
  GATED. Executed as a NEW document, plans/ITER011_PHASE2_ADJUDICATION_2026-09
  -02.md -- the frozen preregistration is NOT edited (rule 13; 9.2/9.3 call for
  exactly this in-band record), so a NEGATIVE result got the same ceremony a
  positive one would have. R-408(3) THE v2 FREEZE -- FROZEN, GOVERNING FROM
  2026-09-03: (e) adopted as drafted with no structural constant re-chosen, (f)
  CONTENT_DARK JOINS THE GOVERNING SET from the effective day, (g) the 08-26
  hype coin-day LEFT AS v1 RECORDED IT, and the section 8(1) limit carried
  verbatim (blind on the fourth consecutive dark day). VERIFIED BY ME AT THE
  ARTIFACT: FROZEN_BY_USER True, EFFECTIVE_FROM_DAY "20260903",
  CONTENT_DARK_GOVERNS True, RESTATE_20260826_HYPE False, 19 CHECKS rc 0, and --
  the part that matters tonight -- governs("20260902") is FALSE while
  governs("20260903") is TRUE. R-411(i) THE G-COUNTING FLOOR -- for G-COUNTING
  ONLY, a coin-day counts toward the >=5 bar only if its unmasked complement
  covers >= 144 OF 288 WINDOWS; EVERY GOOD WINDOW IS SCORED REGARDLESS.
  R-411(ii) THE P1 DENOMINATOR -- the P1 bar on a complement reads PER UNMASKED
  HOUR (loss per hour of usable feed), with the calendar-24h form KEPT BESIDE IT
  rather than replaced. Both are new constants named once each in
  da_blackout_mask.py with the ruling quoted and consumed by BE's scorer; the
  ESCALATION_no_minimum_complement_size block and the preflight's open_decisions
  entry become the ruled state naming R-424. TONIGHT IS UNTOUCHED AND THAT WAS
  CHECKED RATHER THAN ASSUMED: the v2 checker is imported by NO verdict path --
  not da_forward_day_verify.py, not da_midnight_verify.sh, not
  da_governed_verdict_preflight.py; only v5_deploy_gates.py runs its selftest --
  so the 09-02 closing verdict at 00:06Z RUNS v1 ONLY. A FREEZE THAT GOVERNS
  FROM TOMORROW CANNOT REACH TONIGHT. Wiring lands in DA round 10 AFTER that
  verdict is verified and BEFORE 2026-09-04 00:06Z, the first governed v2
  verdict. THE 09-02 ACCRUAL CALL IS NOT A FIFTH ADOPTION: R-409 already rules
  it as a principle, the coordinator applies it after the 00:06Z verdict as a
  section-7-style act with R-409 as the stated reason, and R-411(ii) now fixes
  WHICH DENOMINATOR that reading uses. THE SIXTH DECISION STAYS OPEN AND NOW HAS
  A RECOMMENDATION (R-424 section 6): race on the FROZEN BYTES AT 1b53929 -- the
  plan's reading, section 10 step 9, the frozen set scored UNCHANGED -- with code
  anchors materialised from the commit and SHA-VERIFIED BEFORE IMPORT, the data
  anchor harmful_exposure_rows_v3_eraB.json (which HAS NO COMMIT, data/ being
  gitignored; BE fact (iii), STILL UNVERIFIED until round 4 lands) bound by the
  sha the frozen manifest eb8733da records and VERIFIED BY CONTENT with the
  source named, and the driver at HEAD as harness with every non-anchor module
  in the closure that moved since 1b53929 NAMED IN THE RECEIPT. WHY NOT
  RE-FREEZE: a re-freeze is a NEW candidate (multiplicity 3) and a new freeze
  commit WITH NO NEW EVIDENCE BEHIND IT -- it would let the anchors' drift choose
  the candidate. Until ruled, BE round 4's output stays AN ESTIMATE IN SCRATCH,
  NOT A RACE SCORE. DA ROUND 10 IS ONE BATCH, BUILT NOW IN ~/ctaNew-wt-da AND
  LANDED ONLY AFTER THE 00:06Z VERDICT IS VERIFIED, because THE FIRST GOVERNED
  VERDICT RUNS THE TREE AS IT IS (R-402) and landing beforehand would change the
  instrument under the run it exists to read: (a) the RR12-1 split, REPO from
  __file__ with the record proving WHICH TREE RAN; (b) the identity-only
  admission log; (c) CO-R4, the preflight's refusal emitted as JSON ON STDOUT
  WITH A DISTINCT RC, since rc 1 means failing predicates and a refusal is not
  that; (d) the R-411(i)/(ii) constants with falsifiers ON THE BOUNDARY ITSELF
  (143 does not count, 144 does); (e) the v2 wiring, where CONTENT_DARK joins
  the governing set beside v1's statuses on a governed day, NO_REFERENCE IS
  NEVER A PASS, the composite is the more severe, v1 untouched, with a control
  that the 09-02 verdict path is BYTE-IDENTICAL with and without the wiring and
  a positive control on a synthetic governed dark day. BE round 5 (after the 3-4
  review) adds consuming counts_toward_G from the mask block and REFUSING IF
  ABSENT on a governed day, the population-gate ledger-vs-tape refusal, and
  require_verified(). STATE: FOUR USER DECISIONS RULED, ONE OPEN (the freeze
  disposition), the 09-02 accrual call mechanical after 00:06Z; R-419 remains
  revocable by the USER. Landed and awaiting the coordinator's next entry: MEM
  round 18 (79f2db5, Q-MEM-6 21da8fd), DE round 12 (9dbaa5a), the DE round 11
  review (1e494f9). UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving with
  Q4 failing; development evidence.
```

## Batch 29 — archived 2026-09-02T12:27Z (1 entry, rolling-window overflow)

Moved in the MEM round-23 true-up of R-428. Join rule as in batch 1.

```yaml

  2026-09-02T12:02Z (MEM ROUND 20 -- STATE THAT ONLY EXISTS IF THE PROCESS EXITS
  NORMALLY). R-425 swept; two of BE's in-flight facts checked LIVE at the box,
  the rest verified at the artifacts. THE CAP HELD AND THE FIX WAS THE CODE, NOT
  THE CEILING: BE's 09-01 run was OOM-KILLED AT 12.0 GiB AFTER 21 MINUTES and
  the answer was to RESTRUCTURE INTO A STREAMING PASS, not to raise the limit.
  Confirmed at the box at 12:02Z: be-fwd-0901c.service runs in research.slice
  with MemoryMax STILL 12 GiB (12,884,901,888) and MemoryCurrent about 2.75 GiB
  roughly seven minutes in -- far under the cap SO FAR, and the run is NOT
  FINISHED, so this is a mid-run observation and not a result. That is the same
  discipline that produced compact_design when iteration 011 OOM'd twice --
  PACK THE WORK, DON'T RAISE THE CAP -- and the second time this programme has
  taken the harder branch on memory. AND THE RECEIPT NOW EXISTS BEFORE THE RUN
  DOES: the killed run wrote NOTHING, so nobody could tell how far it got; I
  opened the current run's receipt WHILE IT WAS STILL RUNNING and
  be_forward_day_receipt_20260901.json (9,584 B) already carries a gates array
  with day_closed_and_attributed, population_supply_and_bridge and
  materialise_frozen_bytes each PASS, stamping ratification_ref R-419 -- correct
  for round 4 under R-419's supersession. A KILLED RUN NOW LEAVES A PARTIAL
  RECORD INSTEAD OF A HOLE. THOSE TWO ARE ONE LESSON IN TWO COSTUMES AND THIS
  SEAT HAS PAID FOR BOTH: earlier today BE's mutation harness was SIGKILLED and
  its finally never ran, so a mutant stayed applied in the tree; now a receipt
  written only at the end vanished with the process. STATE THAT EXISTS ONLY IF
  THE PROCESS EXITS NORMALLY IS NOT STATE, IT IS A WISH, and both fixes are the
  same instruction -- write as you go, and prove it survives a kill. The two
  falsifiers the BE 3-4 review must see are exactly right: the STREAMING PASS
  SCORES IDENTICALLY to the non-streamed one on a small population, and the
  PER-GATE FLUSH SURVIVES SIGKILL BETWEEN GATES. All five of BE round 4's
  in-pane facts remain REPORTED and NOT VERIFIED as results until the commit
  lands; what I add is only what I observed directly at the box. DE ROUND 12
  VERIFIED (9dbaa5a, Q-DE-30) and now UNDER REVIEW at that tip: 84 CHECKS rc 0
  under both launchers, reproduced here, with mutation_audit 19 PATHS and
  SURVIVORS []. DE10-R1 IS CLOSED IN BOTH DIRECTIONS -- permissive garbage
  (now_utc "zzzz", scope_to "not-a-date") AND restrictive garbage (scope_from
  "zzzz") both refuse by field and value; now_utc=123 refuses as a TYPE rather
  than crashing ("a TypeError from a comparison is not a refusal"); and the
  09-01 boundary reads 23:59:59Z not closed, 00:00:00Z closed. THE DE ROUND 11
  REVIEW IS VERIFIED AND RELEASED (1e494f9): DE-R1..R4 closed, TWO PAST THE ASK,
  and both deliberate separations accepted with the reviewer's own reason --
  STRATIFIED is a legal sampling value so the defect is in the PAIR, and folding
  the contradiction into the vocabulary loop would name one field for a
  two-field fault. DE11-R1 REPRODUCED: exec('import X'), eval("__import__('X')")
  and a REBOUND __import__ each parse to [] so reads_no_verdict answers TRUE --
  the controls behave (a literal is caught, a non-literal argument refuses) but
  dynamic forms slip past, and AN ANSWER ABOUT UNPARSED CODE IS NOT AN ANSWER.
  CO-6 IS THE ROUND'S QUIET ONE AND IT IS THE COORDINATOR FINDING ITS OWN
  DEFECT (LOW, at 9dbaa5a): stamped_at is PARSED ONLY ON THE SUPERSEDED BRANCH
  -- on R-418 (superseded) stamped_at "not-a-time" refuses by name, while on
  R-419 (not superseded) the identical garbage returned verified TRUE with the
  value carried VERBATIM into the emission, never parsed. A STAMP SUPPLIED IS A
  CLAIM ABOUT A RECEIPT WHETHER OR NOT A SUPERSEDER EXISTS TODAY, so a value
  that sorts nowhere until it matters is exactly DE10-R1's shape ONE BRANCH
  OVER, found hours after round 12 closed the other one. THE GENERALISATION
  WORTH KEEPING: WHEN A CLASS OF DEFECT IS FIXED ON ONE BRANCH, THE SAME CLASS
  ON THE SIBLING BRANCH IS NOT FIXED -- IT IS MERELY UNVISITED. Fix: parse at
  entry, refuse by field and value, keep None as "no receipt"; severity is the
  reviewer's to confirm in the round-12 review. Both DE11-R1 and CO-6 are routed
  to DE ROUND 13 (Q-DE-31, dispatched). REVIEW TABLE: DE round 11 RELEASED; DE
  round 12 VERIFIED and UNDER REVIEW at 9dbaa5a
  (REQUEST_DE_ROUND_12_2026-09-02.md), with BE ROUNDS 3-4 QUEUED BEHIND IT; DE
  round 13 IN FLIGHT; DA round 10 BUILDING in ~/ctaNew-wt-da and landing only
  after the 00:14Z read; BE round 4 in flight. OPEN FINDINGS: DE11-R1 and CO-6
  to DE round 13; RR12-1 and CO-R4 to DA round 10. USER DECISIONS UNCHANGED:
  FOUR RULED (R-408(2), R-408(3), R-411(i), R-411(ii)), ONE OPEN (the freeze
  disposition, with the coordinator's recommendation at R-424 section 6), and
  the 09-02 accrual call MECHANICAL after 00:06Z. R-419 remains revocable by the
  USER. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving with Q4 failing;
  development evidence.
```

## Batch 30 — archived 2026-09-02T12:38Z (1 entry, rolling-window overflow)

Moved in the MEM round-24 true-up of R-429. Join rule as in batch 1.

```yaml

  2026-09-02T12:11Z (MEM ROUND 21 -- A FIX WITHOUT ITS FALSIFIER, AND TWO
  CORRECTIONS TO MY OWN READING). R-426 swept; verified at the artifacts and
  live at the box. CO-7 (LOW, coordinator): THE CO-6 FIX IS CORRECT AND SHIPPED
  WITHOUT A FALSIFIER. stamped_at is now parsed at entry and refuses garbage on
  the branch that used to echo it -- but the diff added NO SELFTEST LINE, so the
  count went 84 -> 84 and NOTHING ASSERTS either the refusal on that branch or
  the echoed parsed value. Rule 15 says a checker ships its falsifier, and A
  COUNT THAT DOES NOT MOVE IS THE TELL -- sharper for the timing: the round-12
  review had just PROVED that assertion works by emptying a selftest loop and
  watching the count assertion fail (82 == 84). THE INSTRUMENT THAT WOULD HAVE
  CAUGHT CO-7 WAS DEMONSTRATED THREE COMMITS BEFORE CO-7 HAPPENED. Routed to DE
  round 14 (Q-DE-32) with the audit's unparsable_stamped_at case to be driven on
  the branch that used to be blind. TWO CORRECTIONS TO MY OWN ROUND-20 ENTRY,
  BOTH MINE AND BOTH THE SHAPE I KEEP RECORDING ABOUT OTHERS. (a) I reported the
  BE receipt as carrying THREE PASS gates; it carries SIX
  (day_closed_and_attributed, population_supply_and_bridge,
  materialise_frozen_bytes, import_closure_disclosure, import_anchors_from_run_
  dir, selection_from_specs) -- my print TRUNCATED AT 220 CHARACTERS and I
  described what it showed as though it were the whole array: A PARTIAL READ
  REPORTED AS COMPLETE, committed by the seat that has recorded that class four
  times this week. (b) I wrote that the receipt exists "while the run is still
  going" in a way that reads as PROGRESSIVE flushing. IT IS NOT: all six gates
  landed WITHIN ~4 s OF THE 11:55:06Z START and the file's mtime has not moved
  since 11:55:10.29 through fifteen minutes of streaming scoring -- so a kill
  during scoring would leave THE GATES AND NOTHING ABOUT SCORING PROGRESS. The
  hole is SMALLER, NOT CLOSED. Both corrected in place in HANDOFF. AND ONE
  THING I HAD NOT LOOKED AT CLOSELY ENOUGH TO GET WRONG YET: sealed: true in
  that receipt means METRICS GO TO THE SEALED FILE ONLY (rule 11; the receipt
  carries counts, identities and hashes and NO metric, and unsealing is the
  coordinator's or the USER's act), NOT that the run finished -- mid-run,
  "sealed" and "done" look identical to a careless reader and the sealing note
  is what separates them. BE's run re-checked live at 12:10Z: still active,
  MemoryMax 12 GiB UNRAISED, MemoryCurrent about 4.0 GiB (2.75 -> 4.0 over eight
  minutes), climbing, far under the cap, and still NOT A RESULT. THE DE ROUND 12
  REVIEW IS VERIFIED AND RELEASED (dcb7036): DE10-R1 closed AT THE ROOT (all
  five temporal comparison sites compare datetimes, _norm_ts parses); DE12-R1
  CONFIRMS CO-6 AND WIDENS IT -- on the non-superseded ref a NON-STRING stamp
  (123) was also accepted and echoed, not only garbage strings -- raised to
  MEDIUM-LOW with the reviewer's framing accepted: A STORED PROVENANCE FIELD THE
  CHECKER WILL LATER REFUSE TO READ, the failure DEFERRED onto the day a
  superseder appears; and DE12-R2 IS NEW -- SCOPE_OPEN_TOKENS = ('null','none',
  '') means a scope_to: with NOTHING AFTER THE COLON reads open-ended, verified
  True, unverifiable [], SILENTLY, so AN EDITING SLIP BECOMES AN UNBOUNDED
  RATIFICATION with no sign in the emission (a tilde refuses; an absent field
  reads MISSING; the hole is precisely the empty value, the one a human typo
  produces). DE ROUND 13 VERIFIED (f04c06a, Q-DE-31): admissible 62, ratification
  84, seam 69, rc 0 both launchers. DE11-R1 CLOSED -- exec('import X'),
  eval("__import__('X')"), bare compile(...) and a rebound __import__ all refuse
  by shape, seven controls hold, re.compile resolves to ['re'] and is NOT
  refused, and DECLARED_BLIND_SHAPES NAMES FIVE shapes the checker cannot see,
  which is the honest form of a limit. CO-6/DE12-R1 CLOSED AT ENTRY, non-strings
  included -- the reviewer's widening covered UNSEEN, since round 13 predates
  the review by three seconds. DE'S OWN FALSE POSITIVE IS RECORDED AS METHOD:
  matching on the attribute name made re.compile look like an opaque exec and
  THE SEAM REFUSED ITSELF -- caught by the dependent suite BEFORE IT SHIPPED.
  TWO METHOD MARKS FROM THE REVIEW: its structural closure of "the control that
  ran nothing" is the right kind of proof (emptying a loop FAILS the suite on the
  count assertion, so a loop that runs zero times cannot pass); and its
  audit-count note is accepted and forwarded -- 19 paths = 19 (input, refusal)
  CASES over THREE raise sites, which is CALL-SITE COVERAGE OF A SHARED PARSER,
  the right design, to be stated in the count's own emission rather than read as
  nineteen independent guards. REVIEW TABLE: DE round 12 RELEASED; DE round 13
  VERIFIED and UNDER REVIEW at f04c06a (REQUEST_DE_ROUND_13_2026-09-02.md), with
  BE ROUNDS 3-4 QUEUED BEHIND IT; DE round 14 IN FLIGHT; DA round 10 BUILDING
  and landing only after the 00:14Z read; BE round 4 in flight. OPEN FINDINGS:
  DE12-R2 and CO-7 to DE round 14; RR12-1 and CO-R4 to DA round 10. USER
  DECISIONS UNCHANGED: FOUR RULED, ONE OPEN (the freeze disposition, R-424
  section 6), the 09-02 accrual call MECHANICAL after 00:06Z. R-419 remains
  revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving
  with Q4 failing; development evidence.
```

## Batch 31 — archived 2026-09-02T12:54Z (1 entry, rolling-window overflow)

Moved in the MEM round-25 true-up of R-430. Join rule as in batch 1.

```yaml

  2026-09-02T12:20Z (MEM ROUND 22 -- THE WORKTREES PAID FOR THEMSELVES). R-427
  swept; verified at the artifacts, including one claim I reproduced as NOT
  reproducing. CO-7 CLOSED BY BEING MADE WORSE FIRST: the reviewer did not stop
  at "no check was added" -- it RESTORED THE EXACT PRE-FIX SHAPE (parse only
  inside the superseded branch, raw echo) and THE SUITE STAYED GREEN AT 84, so
  the defect was REINSTATABLE IN FULL, SILENTLY. Those two statements are the
  same fact but only the second prices it: A FIX WITHOUT A FALSIFIER IS NOT AN
  INCOMPLETE FIX, IT IS A FIX THAT CAN BE UNDONE WITHOUT ANYONE NOTICING. That
  is DE13-R1 (LOW-MEDIUM), now CLOSED at 194b5e9. A FILED CLAIM THAT DID NOT
  REPRODUCE, AND I CHECKED IT MYSELF: DE round 14 filed stamped_at_raw as
  DOCUMENTED; at that tip it appears FOUR TIMES -- one emission line (:672) and
  three selftest lines (:987, :990, :994) -- AND NOWHERE IN DOCUMENTATION. That
  is the reviewer's DE13-R2, STILL OPEN, with an IN-BAND CORRECTION REQUIRED in
  Q-DE-33. The direction matters: NO EXTERNAL CHECK WOULD FIND THIS, because the
  code is right and only the claim about it is wrong -- the same family as my own
  round-20 truncated read, one seat over. DE ROUND 14 VERIFIED (194b5e9,
  Q-DE-32): 102 CHECKS both launchers, reproduced here; DE12-R2 and CO-7 CLOSED;
  the empty-value refusal is GENERAL and DISTINCT from MISSING and from VALUE;
  "none" REMOVED as a decision because R-419 section 4 adopted "null" only; and
  the audit now reports n_cases 21 / n_raise_sites 16, COMPUTED rather than
  narrated. THE DE ROUND 13 REVIEW IS VERIFIED AND RELEASED (b7ce7bb): DE11-R1
  closed WIDER THAN FILED -- eight rebinding shapes of __import__ (chained alias,
  dict value, list element, default argument, keyword argument, tuple unpack,
  attribute assignment) all refuse, with a literal call, a string and a comment
  not swept up -- and the closure test that matters is the reviewer's:
  reads_no_verdict is TRUE on DE's own three files and FALSE on be_forward_day
  and da_blackout_mask because they import verdict producers, which is the
  predicate WORKING. The reviewer's judgement that A DECLARED LIMIT CAN BE
  TESTED FOR ITS CONSEQUENCE (expected-blind assertions, both directions) is
  ACCEPTED and routed to DE round 15. DA ROUND 10 IS BUILT AND HELD at worktree
  commit 3a89e6c -- I verified the HOLD rather than the intent: it is detached
  from b75c9fe, ON NO REMOTE BRANCH, nothing landed, nothing under data/, the
  installed unit unchanged, and the shared tree carries only BE's in-flight
  be_forward_day.py, with da-midnight-verify.timer armed for 2026-09-03
  00:06:00Z. Counts as reported, to be verified at landing: 235 -> 244 / mask 19
  -> 30 / preflight 30 -> 34. DA FOUND A CLASS, NOT A BUG: 32 FILES under
  live/pm_research/ derive a data/pm_5min path from __file__, which points at the
  CODE root, so inside a per-seat worktree they resolve to an EMPTY data/; the
  fix is a CODE_ROOT/DATA_ROOT SPLIT resolved in the lowest-level reader. DA
  FIXED THE SEVEN IT OWNS AND TOUCHED NO OTHER SEAT'S FILE (rule 18), which is
  correct and LEAVES TWENTY-FIVE INSTANCES STANDING. ONE OF THEM IS ON THE
  COORDINATOR'S OWN SURFACE -- CO-8: v41_boundary_preflight.py carries the same
  defect (REPO = P.REPO at :53, PROVENANCE_LEDGER at :177) and TWO OF ITS GATES
  FAIL IN A BARE WORKTREE; it is NOT on tonight's path (only v5_deploy_gates.py's
  selftest runs it, no timer does), so it is coordinator-owned and fixed after
  tonight, with the reviewer taking it in the DA round 10 review. WORTH STATING
  PLAINLY: PER-SEAT WORKTREES WERE ADOPTED THREE ROUNDS AGO FOR ISOLATION AND
  THEIR FIRST REAL YIELD IS A LATENT 32-FILE CLASS NOBODY HAD SEEN -- isolation
  did not cause this, it made a shared-tree assumption visible by removing the
  shared tree. AND DA FOUND THE CONTROL-THAT-RAN-NOTHING CLASS IN ITS OWN SUITE:
  SIX of its checks were SILENTLY SKIPPING in a worktree, 235 counted against
  229 RUN. The round-12 review had closed that class STRUCTURALLY FOR THE
  CHECKER by proving an emptied loop fails on the count; it reappeared ONE
  SURFACE OVER, which is CO-6's lesson in a different key -- FIXING A CLASS
  WHERE YOU FOUND IT DOES NOT FIX IT WHERE YOU DID NOT LOOK. The count now
  asserts over checks that RAN. A NUMBERING CORRECTION THE COORDINATOR TOOK
  AGAINST ITSELF: R-424 dispatched round 10 as Q-DA-208 when 208 was already
  assigned by round 9 to tonight's verdict filing; DA files as Q-DA-209 and 208
  STAYS WITH THE VERDICT -- FIRST-ASSIGNED KEEPS THE NUMBER, which is the right
  rule, since the alternative silently renames an artifact someone else has
  already cited. REVIEW TABLE: DE round 13 RELEASED; DE round 14 VERIFIED and
  UNDER REVIEW at 194b5e9; DE round 15 IN FLIGHT (Q-DE-33); DA round 10 HELD for
  the 00:14Z read; BE round 4 in flight, about 21 minutes in and under the cap.
  OPEN FINDINGS: DE13-R2 to DE round 15; RR12-1 and CO-R4 to DA round 10 (held);
  CO-8 to the coordinator after tonight. USER DECISIONS UNCHANGED: FOUR RULED,
  ONE OPEN (the freeze disposition, R-424 section 6), the 09-02 accrual call
  MECHANICAL after 00:06Z. R-419 remains revocable by the USER. UNCHANGED:
  G=1/5; the 011 family is 12 of 24 surviving with Q4 failing; development
  evidence.
```

## Batch 32 — archived 2026-09-02T13:10Z (1 entry, rolling-window overflow)

Moved in the MEM round-26 true-up of R-431/R-432. Join rule as in batch 1.

```yaml

  2026-09-02T12:27Z (MEM ROUND 23 -- A LIMITATION THE CODE DID NOT HAVE, AND A
  RECEIPT THAT DID NOT SURVIVE). R-428 swept; verified at the artifacts,
  including one check the dispatch asked for that is answerable NOW. THE
  DECLARED-BLIND LIST WAS WRONG IN THE RARE DIRECTION: IT CLAIMED A LIMITATION
  THE CODE DID NOT HAVE. builtins.__import__('x') was listed as invisible to the
  import checker and is in fact CAUGHT, because the matcher keys on the
  attribute name -- and THE EXPECTED-BLIND ASSERTIONS FOUND IT ON THEIR FIRST
  RUN, assertions that exist only because the round-13 reviewer argued that A
  DECLARED LIMIT CAN BE TESTED FOR ITS CONSEQUENCE. Verified here:
  DECLARED_BLIND_SHAPES is now FOUR entries (runpy, the attribute-form
  exec/eval/compile with the re.compile reason attached, getattr(importlib,
  "import_module"), and C extensions/import hooks) and builtins.__import__ is
  GONE. A FALSE STATED BLINDNESS IS NOT HARMLESS MODESTY: it invites a
  compensating control nobody needs while the genuine gaps sit beside it wearing
  the same label. The remaining four now EACH ASSERT EXPECTED-BLIND, and the
  consequence of a real blind shape -- that through the getattr form A VERDICT
  PRODUCER WOULD PASS -- is written as a CHECK rather than left as prose. THE
  REVIEWER'S OWN FILING CARRIED THE SAME FALSE CLAIM AND OWES AN IN-BAND
  CORRECTION: the round-13 review said "I verified all five declared shapes
  behave as declared", but its parenthetical ENUMERATED FIVE THINGS THAT ARE NOT
  THE LIST'S FIVE ENTRIES (collapsing three builtins forms into what is one
  entry), and builtins.__import__ APPEARS NOWHERE IN ITS EXECUTED EVIDENCE --
  and it was the one that was not blind. A COUNT THAT MATCHED THE LIST'S LENGTH
  STOOD IN FOR A CHECK OF THE LIST'S MEMBERS: R-289's family, in the reviewer's
  chair, the third instance of that shape this week. Rule 16 binds reviewer
  filings as it binds seats', and rule 13 puts the correction in the NEXT FILING
  rather than a sidecar -- REQUIRED in the DE round 15 review. Stated fairly:
  the review is RELEASED and the recommendation it made is exactly what found
  the error. DE ALSO CORRECTED ITS OWN FALSE "documented" CLAIM WITH THE CAUSE
  NAMED: a str.replace() on a NON-MATCHING ANCHOR is silently a NO-OP, and the
  edit was reported done without re-reading the file; DE NOW ASSERTS ITS
  ANCHORS. AN EDIT THAT CANNOT FAIL LOUDLY WILL EVENTUALLY REPORT SUCCESS FOR
  WORK IT DID NOT DO. DE ROUND 15 VERIFIED (0ca510e, Q-DE-33): admissible 69,
  ratification 104, seam 69, all reproduced here under both launchers, with
  DE13-R2 CLOSED (docstring plus a stamp_fields emission note plus two
  assertions). DE ROUND 16 IS STAGED behind DA round 10's landing and the r14
  review: the CODE_ROOT/DATA_ROOT split on the THREE DE-owned files that derive
  a data path from __file__ (de_admissible_windows :64/:77,
  de_ratification_check :43, de_lane4_results_doc), COUNTED AT THE TREE, with
  five other DE files NOT in the class, following DA's
  pm_tape_density._resolve_data_root convention so THE SPLIT IS WRITTEN ONCE.
  BE ROUND 4: THE 09-01 STREAMING PASS COMPLETED -- BE's report, to be verified
  at landing: exit 0, 26 minutes, PEAK 5.9 G AGAINST THE UNRAISED 12 G CAP, TEN
  GATES PASS, 1,875 == 1,875, 1,859 windows, 2,262,457 rows -> 1,847,824
  actions, reconciliation clean -- and BE is RE-RUNNING with two receipt
  disclosures. THE LANDING CHECK HAS AN ANSWER ALREADY AND IT IS NOT THE
  COMFORTABLE ONE: it asks whether the first pass's receipt SURVIVES the re-run
  (rule 13), and checked at the artifact at 12:27Z, IT DOES NOT -- the re-run
  (be-fwd-0901d.service) writes to THE SAME OUTDIR, and
  be_forward_day_receipt_20260901.json now carries as_of_utc 2026-09-02T12:23:29Z
  with SIX gates, the re-run's early flush, so THE COMPLETED TEN-GATE RECEIPT IS
  GONE, overwritten about two minutes after the run finished; the only other
  receipts on disk are an unrelated 11:02 pair in a different directory. SCOPE,
  STATED SO IT IS NOT READ AS LARGER THAN IT IS: this is SCRATCH, not derived/,
  nothing canonical was touched, and BE may hold a copy I did not find -- I
  checked the obvious places. NOT ADJUDICATED; recorded because the question was
  asked and is answerable now rather than at landing, and because it is the
  THIRD TIME IN ONE DAY that a record was lost to a SAME-PATH WRITE. The remedy
  is already known here: an outdir per run, or supersede rather than overwrite.
  TWO DISCLOSURES TO READ CAREFULLY RATHER THAN QUICKLY: n_masked 0 AT THE
  SCORING SEAM DOES NOT MEAN NOTHING WAS MASKED -- the mask was applied at
  SUPPLY, 141 windows gone BEFORE ANY ROW WAS BUILT, so a zero at the seam means
  "nothing left to mask here", not "no masking happened"; and THE FROZEN
  CANDIDATE FITS btc AND eth ONLY, with five coins supplied, replayed, counted
  and UNSCORED -- a FACT FOR THE RECEIPT, while what it means for G-COUNTING is
  a POLICY question (rule 14) that is the USER's and NOT a pending decision
  until someone puts it to them. REVIEW TABLE: DE round 14 UNDER REVIEW; DE
  round 15 VERIFIED with its review QUEUED and the reviewer's in-band correction
  REQUIRED there; DE round 16 STAGED; DA round 10 HELD for the 00:14Z read; BE
  round 4 IN FLIGHT on the re-run. OPEN: RR12-1 and CO-R4 to DA round 10; CO-8
  to the coordinator after tonight; the round-13 review's in-band correction to
  the reviewer. USER DECISIONS UNCHANGED: FOUR RULED, ONE OPEN (the freeze
  disposition, R-424 section 6), the 09-02 accrual call MECHANICAL after 00:06Z.
  R-419 remains revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of
  24 surviving with Q4 failing; development evidence.
```

## Batch 33 — archived 2026-09-02T13:23Z (1 entry, rolling-window overflow)

Moved in the MEM round-27 true-up of R-433. Join rule as in batch 1.

```yaml

  2026-09-02T12:38Z (MEM ROUND 24 -- THE RULE STOPPED ONE FIELD SHORT). R-429
  swept; verified at the artifacts, with the central mechanism read at SOURCE
  rather than taken. DE14-R1 (MEDIUM): DE12-R2 taught that AN EMPTY VALUE MUST
  REFUSE, and DE14-R1 finds THE ONE FIELD IT DID NOT REACH -- supersedes.
  superseded_by() compares str(blk.get("supersedes","")).strip() == ref
  (de_ratification_check.py:279-292), so an ABSENT OR EMPTY value becomes "" and
  simply FAILS TO MATCH; later entries' blocks are BOUND BUT NEVER VALIDATED, so
  r-902 (wrong case), R-9O2 (letter O for zero) and "R-902, R-901" (two refs in
  one field) are INVISIBLE the same way. THE FAILURE MODE IS THE QUIET ONE: not
  a wrong answer but "nothing supersedes this", IN THE FIELD THAT DRIVES THE
  CHECKER'S STRONGEST REFUSAL. BOUNDED, AND I CHECKED THE BOUND RATHER THAN
  REPEATING IT: the register holds EXACTLY ONE ratification block, R-419 with
  supersedes R-418, an exact match -- NOTHING SHIPPED IS WRONG and the exposure
  is FORWARD, at BE's check() call site. That distinction is worth keeping: a
  MEDIUM finding with no present instance is still worth fixing BEFORE the
  second block exists, which is precisely when it stops being checkable by eye.
  DE14-R2 (LOW-MEDIUM) IS THE WEEK'S MOST PERSISTENT SHAPE IN NEW CLOTHES: THE
  AUDIT REPORTS COVERAGE IT DOES NOT ASSERT -- "superseded" refuses at the
  HEADING-TIMESTAMP guard rather than the SUPERSEDED guard, and
  unknown_population_value at VALUE rather than the line it names, so DELETING
  ROUND 14'S OWN NEW CASE LEAVES THE SUITE GREEN. A coverage claim that names a
  guard the case never reaches is the same thing as a count standing in for a
  check, and this programme has now recorded that family IN THE CHECKER, IN THE
  REVIEWER'S CHAIR, IN DA'S SUITE, and here IN AN AUDIT'S OWN ATTRIBUTION.
  DE14-R3 (LOW): .lower() admits NULL/Null/nUlL while the module case-folds
  nowhere else -- DECIDED BY THE COORDINATOR AS RESTORATION (exact null, code
  matching R-419 section 4 as adopted), so NO SPEC CHANGE AND NO USER DECISION:
  housekeeping rather than a manufactured seventh item, the same restraint that
  kept the freeze disposition out of the four-item ruling. DE14-R4 (LOW):
  n_guards still carries the case count. All four go to DE ROUND 16 (Q-DE-34, in
  flight); DE ROUND 17 is the DATA_ROOT split, STAGED behind DA round 10. AND
  THE FALSIFIER ROUND 13 OWED WAS PAID: the reviewer's own pre-fix mutant now
  DIES BY NAME at check 46 under both launchers, and under it the audit surfaces
  survivors ['unparsable_stamped_at_not_superseded'] with attribution going
  NON-TOTAL (20 vs 21) -- a debt named two rounds ago, settled where it was
  incurred. The round-14 closures were driven rather than read: every one of the
  ten RATIFICATION_FIELDS present-and-empty refuses EMPTY, absent MISSING, wrong
  VALUE, on their own cases; two empties report together; scope_from null
  refuses "not a day"; the audit's numbers are COMPUTED from the physical raise
  line; 84 -> 102 accounted (8 + 8 + 2); emptying the garbage loop fires 99 ==
  102; 19 of 19 refusals interpolate, 0 constant. MY RECEIPT FINDING BECAME A
  BE LANDING CONDITION rather than a note (R-429 section 4), and the form is
  better than what I proposed: EITHER A COPY OF THE FIRST PASS'S RECEIPT EXISTS,
  OR THE RE-RUN'S TEN-GATE RECEIPT IS THE ONLY RECEIPT AND THE FIRST PASS'S PANE
  COUNTS MUST MATCH IT -- it does not pretend the bytes are recoverable, and it
  makes the surviving artifact carry the burden of agreeing with what was
  reported; it also rides the BE 3-4 review request. STILL LIVE AT 12:38Z: the
  re-run is active, the receipt at that path still reads as_of 2026-09-02T12:23:
  29Z with SIX gates, and NO TEN-GATE RECEIPT EXISTS YET, so the condition is
  not yet satisfiable either way. REVIEW TABLE: DE round 14 RELEASED; DE round
  15 VERIFIED and UNDER REVIEW at 0ca510e, with the round-13 section-3 IN-BAND
  CORRECTION REQUIRED AS ITS OWN SECTION of that review; DE round 16 IN FLIGHT;
  DE round 17 STAGED; DA round 10 HELD for the 00:14Z read; BE round 4 IN FLIGHT
  on the re-run (~12:50Z expected). OPEN: DE14-R1..R4 to DE round 16; RR12-1 and
  CO-R4 to DA round 10; CO-8 to the coordinator after tonight; the reviewer's
  round-13 section-3 correction to the DE round 15 review; the first-pass
  receipt to BE as a landing condition. USER DECISIONS UNCHANGED: FOUR RULED,
  ONE OPEN (the freeze disposition, R-424 section 6), the 09-02 accrual call
  MECHANICAL after 00:06Z. R-419 remains revocable by the USER. UNCHANGED:
  G=1/5; the 011 family is 12 of 24 surviving with Q4 failing; development
  evidence.
```

## Batch 34 — archived 2026-09-02T13:33Z (1 entry, rolling-window overflow)

Moved in the MEM round-28 true-up of R-434. Join rule as in batch 1.

```yaml

  2026-09-02T12:54Z (MEM ROUND 25 -- THE RUN FINISHED, AND ITS OWN NUMBERS SAY
  71.7% OF IT PRODUCES NO SCORE). R-430 swept; verified at the unit and the
  receipt, with the coverage arithmetic done here rather than repeated. BE'S
  RE-RUN COMPLETED 12:49:42Z: unit be-fwd-0901d.service Result=success,
  ExecMainStatus=0, TEN GATES PASS. THE LANDING CONDITION IS MET IN ITS SECOND
  FORM AND THAT FORM IS WEAKER ON PURPOSE: the counts MATCH THE FIRST PASS
  EXACTLY -- 1,875 supplied = 1,875 bridged, 1,859 windows with rows, 2,262,457
  rows -> 1,847,824 actions -- so the overwritten first-pass receipt is
  EVIDENCED ONLY BY THAT AGREEMENT unless BE holds a copy. That is what the
  condition was written to accept and it does not pretend to be the bytes. THE
  LANDING STILL OWES A COMMIT: the receipt reads working_tree_dirty TRUE beside
  carrying_commit 0ca510e, SO THE COMMIT IT NAMES IS NOT WHAT RAN -- RR12-1's
  family, and the same lesson my own round-18 citation earned: A HASH IN A
  RECEIPT IS A CLAIM ABOUT A SPECIFIC ARTIFACT, and a dirty tree quietly makes
  it a claim about something else. A reader should also note the receipt's as_of
  is the RUN'S START (12:23:29Z), twenty-six minutes before its own bytes were
  written at 12:49:42Z -- correct for a run receipt, misleading if read as a
  write time. AND THE COVERAGE ARITHMETIC, DONE HERE FROM THE RECEIPT'S OWN
  PER-COIN COUNTS: coin_coverage records seven coins supplied, btc and eth with
  a frozen fit, five without, and 1,344 windows supplied WITHOUT A FIT -- which
  against the receipt's own numbers is 1,344 OF 1,875, or 71.7% OF THE SUPPLIED
  POPULATION PRODUCING NO SCORE. The mask arithmetic closes exactly: 7 x 288 =
  2,016 present, minus 141 masked at supply = 1,875; 531 with a fit (btc 265 +
  eth 266) plus 1,344 without (bnb 266, doge 266, hype 279, sol 265, xrp 268) =
  1,875. SIXTEEN BRIDGED WINDOWS PRODUCED NO ROWS (1,875 bridged vs 1,859 with
  rows), which the receipt also carries. The receipt says it in its own voice --
  "the day is not scored whole and this says so" -- which is the right place for
  it. IT IS A FACT FOR THE RECEIPT; WHAT IT MEANS FOR G-COUNTING IS THE USER'S
  POLICY QUESTION (rule 14) AND STILL NOT A PENDING DECISION UNTIL SOMEONE PUTS
  IT TO THEM. But a reader meeting "ten gates PASS" and "counts match" should
  meet 71.7% in the same breath, which is why it sits beside them in HANDOFF
  rather than only in the artifact. DE ROUND 16 VERIFIED at 829910e (Q-DE-34)
  and UNDER REVIEW: 132 CHECKS, reproduced here; DE14-R1..R4 CLOSED; BOTH
  COORDINATOR MUTANTS DIE BY NAME; and the check that matters most is the
  NEGATIVE one -- THE R-419 AND R-418 VERDICTS ARE UNCHANGED FROM 0ca510e, since
  a validation round that MOVED a verdict would have been a different kind of
  change, and saying so is how the round proves it fixed plumbing rather than
  answers. THE DE ROUND 15 REVIEW IS RELEASED, AND A FIRST FOR THIS PROGRAMME:
  THE REVIEWER CORRECTED ITS OWN FILING IN BAND -- the round-13 section-3 claim
  about the five declared shapes, the one whose unexamined member turned out to
  be the wrong one, corrected in the round-15 review's OWN SECTION rather than a
  sidecar. RULE 13 APPLIED TO A REVIEWER'S FILING, BY THE REVIEWER. Its
  DE15-R1..R4 were reproduced by the coordinator and go to DE ROUND 17
  (dispatched, and also carrying the DATA_ROOT split staged behind DA round 10);
  one of them is the week's shape again -- THE SWAP-DOCSTRING MUTANT LEAVES 104
  GREEN. REVIEW TABLE: DE round 15 RELEASED; DE round 16 VERIFIED and UNDER
  REVIEW; DE round 17 DISPATCHED; DA round 10 HELD for the 00:14Z read; BE round
  4 COMPLETE and awaiting its landing commit. OPEN: DE15-R1..R4 to DE round 17;
  RR12-1 and CO-R4 to DA round 10; CO-8 to the coordinator after tonight; BE's
  landing commit, since the counts satisfy the receipt condition but the tree
  was dirty. USER DECISIONS UNCHANGED: FOUR RULED (R-424), ONE OPEN (the freeze
  disposition, R-424 section 6), the 09-02 accrual call MECHANICAL after 00:06Z.
  R-419 remains revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of
  24 surviving with Q4 failing; development evidence.
```

## Batch 35 — archived 2026-09-02T13:41Z (1 entry, rolling-window overflow)

Moved in the MEM round-29 true-up of R-435. Join rule as in batch 1.

```yaml

  2026-09-02T13:10Z (MEM ROUND 26 -- A RULE THAT BINDS MY OWN FILE, AND IT WAS
  IN BREACH). R-431 and R-432 swept; the round's finding reaches into the state
  files, so I checked it there before writing about it. DE16-R1 IS LIVE: A
  FENCED ratification BLOCK QUOTED IN A LATER, NON-RATIFYING ENTRY IS READ AS
  THAT ENTRY'S OWN, and the supersession is attributed to THE ENTRY'S HEADING
  REF, NOT THE BLOCK'S. Reproduced on the real register plus an appended sweep
  entry: a well-formed quoted block made R-419 read as "SUPERSEDED by R-999",
  and an EMPTY supersedes in that quoted block made R-419's check REFUSE; the
  plural form did the same. A SWEEP ENTRY THAT MERELY ILLUSTRATED A RATIFICATION
  WOULD HAVE SUPERSEDED ONE -- CO-4's family moved from the prose era into the
  block era, where the thing that looks like documentation is read as the thing
  itself. THE COORDINATOR FORMAT RULE, in force from R-432 until DE round 18
  lands (a format is the coordinator's; no number introduced): NO REGISTER ENTRY
  OTHER THAN AN R-ADMISS ENTRY'S OWN MAY CONTAIN A FENCED ratification BLOCK;
  spellings are quoted IN PROSE WITH INLINE BACKTICKS ONLY. THE RULE BINDS MEM'S
  SWEEPS AND THE STATE FILES, AND HANDOFF WAS IN BREACH: my round-15 entry
  carried a FENCED block to show what R-419 restated. REMOVED THIS ROUND, its
  fields re-quoted inline; STATUS.yml never carried one; verified ZERO fenced
  blocks in both state files. Recorded as a compliance check I FAILED AND FIXED
  rather than a rule merely relayed, because THE BLOCK I WROTE WAS EXACTLY THE
  SHAPE THE FINDING IS ABOUT. I ALSO CONFIRMED THE REGISTER IS CLEAN RATHER THAN
  TRUSTING IT: it holds EXACTLY ONE real fenced block, R-419's own at :18329,
  while the two other hits (:508 in a Q-DE-26 row, :18325 in R-419 section 4)
  are INLINE PROSE MENTIONS of the fence that the block finder correctly does
  not read -- which is the whole rule: SPELL THE FENCE, DON'T BUILD ONE. THE
  OTHER THREE FINDINGS, ALL REPRODUCED: DE16-R2 (LOW-MEDIUM) shape-only
  existence -- a supersedes naming a ref that EXISTS NOWHERE (R-9021, R-99999)
  leaves the base ref verifying True with unverifiable [] SILENTLY; DE16-R3
  (LOW-MEDIUM) TWO supersedes: LINES IN ONE BLOCK and bind_from_block takes
  LAST-WINS, so the first target is DROPPED WITHOUT A WORD -- fail-open; and
  DE16-R4 (LOW-MEDIUM) the three KNOWN-BAD comparisons after the coverage
  assertion cannot fire on the case that matters, WITH A NUANCE THE COORDINATOR
  MEASURED UNDER FOUR MAPS: they go red ONLY WHEN HARNESS AND MAP LEGITIMATELY
  CO-MOVE, i.e. ON MAINTENANCE, NOT ON THE DEFECT -- a more useful statement
  than "cannot fail", and one only running the four maps reveals. Also carried:
  MARKER-NAME UNIQUENESS IS UNASSERTED (24 raises / 24 tagged / 19 driven; a
  duplicated "# SITE:" name would merge two sites under one key). All four go to
  DE ROUND 18 (Q-DE-36, dispatched 13:09Z); DE ROUND 19 is the DATA_ROOT split,
  behind DA round 10. NEITHER OF THE TWO FAIL-OPEN FINDINGS PRODUCES A WRONG
  ANSWER LOUDLY -- both produce a confident nothing-to-see-here, which is this
  week's recurring signature. A CORRECTION THE COORDINATOR MADE AGAINST ITS OWN
  ENTRY (rule 13, R-432 section 0): R-431 said the DE round 16 review was "in
  flight" when it had LANDED at 81e050b, TWELVE SECONDS before R-431's own
  commit -- the entry was composed before the pre-commit pull and not re-read
  after it; nothing else in R-431 depends on the word. WORTH KEEPING AS A
  MECHANISM RATHER THAN A SCOLDING: A PULL BETWEEN COMPOSITION AND COMMIT CAN
  TURN A TRUE SENTENCE FALSE INSIDE THE INTERVAL, and only re-reading after the
  pull catches it. DE ROUND 16 REVIEW RELEASED (81e050b, scope 829910e) with
  SEQUENCING SATISFIED -- the reviewer found the one block, identical verdicts,
  and the checker's call site in BE's in-flight driver -- and its suggestion
  that BE's receipt carry the CHECKER's carrying commit goes to BE round 5. DE
  ROUND 17 VERIFIED at a8093a5 (DE15-R1..R4 closed, three mutants die by name),
  queued for review. THE REVIEWER'S QUEUE, IN ORDER: DA round 10 at 3a89e6c (in
  flight from 13:10Z), then DE round 17 at a8093a5, then BE rounds 3-4 when BE
  files. BE ROUND 4 IS STILL IN FLIGHT BY BE'S OWN AUDIT, and two things are
  already evidenced: THE COORDINATOR HOLDS A COPY OF THE RECEIPT (sha256
  68234320), so the landing condition's FIRST form is available after all, and
  the 09-02 OPEN-DAY REFUSAL RECEIPT was evidenced at 12:53Z. USER DECISIONS:
  FOUR RULED (R-424), ONE OPEN -- the freeze disposition (R-424 section 6); the
  09-02 accrual call is MECHANICAL after 00:06Z on 09-03. R-419 remains
  revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving
  with Q4 failing; development evidence; and BE's completed re-run still reports
  1,344 of 1,875 supplied windows (71.7%) producing no score.
```

## Batch 36 — archived 2026-09-02T13:56Z (1 entry, rolling-window overflow)

Moved in the MEM round-30 true-up of R-436. Join rule as in batch 1.

```yaml

  2026-09-02T13:23Z (MEM ROUND 27 -- FOUR CLOSED, AND THE FIFTH LEFT OPEN ON
  PURPOSE). R-433 swept; DE's counts reproduced and the fenced-block assertion
  re-checked after every edit. DE ROUND 18 VERIFIED at db039a3 (Q-DE-36 at
  cc497a1; de_ratification_check.py only, +407/-33, de_admissible_windows.py
  untouched): ratification 132 -> 150 and admissible 75, rc 0 under both
  launchers, with R-419 True/[] and R-418 REFUSED-FOR-A-NEW-RUN both UNCHANGED
  on the real register. DE16-R1 CLOSED: own_ratification_blocks() admits a block
  as the entry's own ONLY IF its ref equals the heading ref AND its kind is
  R-ADMISS, so the sweep-entry quotation that read as SUPERSEDED / REFUSED /
  REFUSED at a8093a5 now leaves R-419 True/[] in all three spellings, and two
  own blocks REFUSE by name. DE16-R2 CLOSED: supersedes R-9021, R-99999 and an
  absent R-418 each REFUSE by name at superseded_by#1. DE16-R3 CLOSED: two
  supersedes: lines REFUSE by name, _parse_block reporting and the callers
  refusing. DE16-R4 CLOSED BY HOOKS: mutation_audit gains _drop_case,
  _migrate_case and _add_case that mutate the HARNESS with coverage recomputed
  from REAL TRACEBACKS -- DE chose the hook over deletion under rule 15 ("a
  mutant that lives in a filing is one nobody re-runs"), accepted. MARKER
  UNIQUENESS ASSERTED: 28 markers, 28 names, 22 DRIVEN by the audit and SIX NOT,
  EACH NAMED rather than counted. FIVE COORDINATOR MUTANTS EACH KILLED BY NAME
  -- and ONE OF THEM NEEDED A TEMP TREE, which is a real property worth keeping:
  _site_names reads __file__, so an in-memory harness would have re-read the
  UNMUTATED file and the renamed-marker mutant only dies against a FILE COPY.
  Beside RR12-1 and my own round-18 citation error, the general form is A CHECK
  THAT READS ITS OWN SOURCE THROUGH __file__ IS CHECKING WHATEVER TREE IT
  HAPPENS TO BE IN. THE ROUND'S MOST INTERESTING MOVE IS THE ONE IT DID NOT
  MAKE: having closed the LATER-entry dangling-target case, DE hit the same
  question ONE STEP IN -- AN ENTRY WHOSE OWN BLOCK DECLARES supersedes: R-777,
  NO SUCH ENTRY, STILL VERIFIES True with unverifiable [] -- measured it, and
  DECLINED TO RULE. DE states it as SCOPE ("the target's existence becomes this
  question when someone checks that target"); the coordinator reads it as A
  WELL-SHAPED CLAIM TO SUPERSEDE NOTHING PASSING THE ENTRY MAKING IT; and rather
  than one overruling the other the disagreement WENT TO THE REVIEWER as item 2
  of the request. THAT RESTRAINT IS WORTH NAMING BECAUSE THE ALTERNATIVE WAS SO
  AVAILABLE: a seat four-for-four in a round can close a fifth by declaring it
  out of scope and nobody looks again; measuring it, recording BOTH readings and
  handing the call to a third party keeps "we fixed this" and "we decided this
  doesn't count" from blurring -- precisely the distinction the audit-coverage
  findings have been about all week. ALSO MEASURED: a quoted block placed BEFORE
  the entry's own block in the entry under check REFUSES at check#8 -- FAIL-
  CLOSED, the asymmetry DE states at :755-757, and the reason THE FORMAT RULE
  OUTLIVES ITS TRIGGER. THE RULE NOW RUNS UNTIL THE DE ROUND 18 REVIEW IS
  RELEASED (not until round 18 landed) AND BEYOND THAT FOR R-ADMISS ENTRIES
  THEMSELVES: the failure mode is asymmetric, fail-closed in one order and
  fail-open in the other, so a rule retired the moment the code improved would
  have retired against the wrong half. ASSERTED THIS ROUND AS REQUIRED: ZERO
  FENCED RATIFICATION BLOCKS IN BOTH STATE FILES (STATUS.yml 0, HANDOFF.md 0),
  checked after every edit rather than once at the end. MY ROUND 26 WAS VERIFIED
  AT THE COMMIT by the coordinator: HANDOFF's fenced block 1 -> 0, STATUS 0 -> 0
  at both commits, STATUS parses with 177 flags, window of three, archive batch
  32. REVIEWER QUEUE IN ORDER: DA round 10 at 3a89e6c (in flight from 13:10Z),
  then DE round 17 at a8093a5, then DE round 18 at db039a3
  (REQUEST_DE_ROUND_18_2026-09-02.md, seven items), then BE rounds 3-4 when BE
  files. DE IS ON DELIBERATE STANDBY (rounds 16, 17, 18 complete; round 19, the
  DATA_ROOT split, behind DA round 10's landing); DA IS HOLDING FOR 00:14Z; BE'S
  CONFIRMING AUDIT IS STILL RUNNING WITH NOTHING SENT (rule 18), and the shared
  tree still carries BE's dirty be_forward_day.py, untouched by any coordinator
  commit. OPEN FINDINGS: RR12-1 and CO-R4 to DA round 10 (under review); CO-8 to
  the coordinator after tonight; DE16-R1..R4 closed at db039a3 PENDING REVIEW;
  the own-supersedes-nonentry residual with the reviewer. USER DECISIONS: FOUR
  RULED (R-424), ONE OPEN -- the freeze disposition (R-424 section 6); the 09-02
  accrual call MECHANICAL after 00:06Z. R-419 remains revocable by the USER.
  UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving with Q4 failing;
  development evidence; BE's completed re-run still reports 1,344 of 1,875
  supplied windows (71.7%) producing no score.
```

## Batch 37 — archived 2026-09-02T14:05Z (1 entry, rolling-window overflow)

Moved in the MEM round-31 true-up of R-437. Join rule as in batch 1.

```yaml

  2026-09-02T13:33Z (MEM ROUND 28 -- THE BATCH MOVED THE PATH AND KEPT THE
  GATE). R-434 swept; the two non-reproducing claims and the new outdir checked
  at the tree myself. DA ROUND 10 IS HELD at 3a89e6c on five findings, each
  checked by the coordinator at the object, with the reviewer executing in
  ~/ctaNew-wt-rev and derived/ IDENTICAL BEFORE AND AFTER EVERY STEP (184
  entries), the log mtime unmoved, and the real launcher never run. NOTHING
  LANDS BEFORE 00:14Z AND TONIGHT IS UNCHANGED EITHER WAY; the re-review
  precedes the landing; DA round 11 is dispatched ON THE HELD COMMIT. DA10-R1
  (MEDIUM) IS THE ONE TO READ TWICE: round 10 was dispatched partly BECAUSE six
  of DA's checks silently skipped in a worktree, and THE BATCH MOVED THE PATH
  (__file__ -> DATA_ROOT) AND KEPT THE GATE -- "if _lg_p.exists():" still fences
  the six log-echo checks, there is NO EXPECTED_CHECKS anywhere in the module,
  and the run PRINTS THE COUNT AND RETURNS 0; measured 238 / 244 / 238 across
  three roots, rc 0 EVERY TIME. So the pane claim that "the count now asserts
  over checks that RAN" DOES NOT EXIST AT THE ARTIFACT. I CARRIED THAT CLAIM IN
  ROUND 22, SO PART OF THE CORRECTION IS MINE: I labelled it as DA's report
  rather than as verified, which is the right label, but this is the SECOND TIME
  THIS WEEK a pane fact reached these files ahead of its object, and the honest
  lesson is that LABELLING A CLAIM AS UNVERIFIED DOES NOT STOP IT FROM BEING
  READ. The root cause the reviewer names is the better keepsake: THE RESOLVER'S
  PREDICATE ASKS "carries data/pm_5min/raw" WHILE ITS CONSUMERS READ derived/
  AND data/mm_hf/ -- a resolver answering a different question from the one its
  callers ask, which is why moving the path fixed nothing. THE OTHER FOUR, EACH
  REPRODUCED AT THE OBJECT: R2, code_root and data_root are emitted BY THE MASK
  ONLY (da_blackout_mask.py:259-262, zero occurrences in the verifier and the
  preflight), so the governing artifact and the 00:14Z emission CANNOT SAY WHICH
  TREE PRODUCED THEM; R3, da_hf_pm_alignment.py:76 imports pm_tape_density bare
  with no sys.path.insert, so python3 -m raises ModuleNotFoundError while the
  path launch passes 53 checks -- CO-2's class, and the module is NOT in
  v5_deploy_gates.py; R4, _is_tracked() at :1834 asks git about
  /home/yuqing/ctaNew while building the path from DATA_ROOT, so a tracked file
  under any other worktree reports PROVENANCE ABSENT WHEN PRESENT; and R5, the
  RR12-1 control at :856-857 asserts the CHILD worktree's data_root equals the
  PARENT's DATA_ROOT, so the mask suite exits rc 1 from any non-canonical parent
  -- LOUD, BUT ENCODING THE ENVIRONMENT RATHER THAN THE PROPERTY, which is the
  failure mode that looks most like working correctly. AND TWO CLAIMS DID NOT
  REPRODUCE, BOTH OF WHICH WOULD HAVE CREATED WORK: CO-8, the coordinator's own
  worry that REPO = DATA_ROOT would propagate, IS DEAD -- v5_boundary_preflight
  defines its own REPO, v41_boundary_preflight.py:53 keeps a CODE root, and NO
  IMPORTER INHERITS THE REBOUND NAME (I confirmed at the tree that nothing
  imports that symbol at all); and THE REQUEST'S EXPECTED LAUNCHER REFUSAL WAS
  INVERTED AND THE CODE IS RIGHT -- under the full rehearsal pair a different
  binary is ADMITTED BY DESIGN and the substitution guard is reachable only in a
  named canonical run. A REVIEW THAT ONLY CONFIRMED FINDINGS WOULD HAVE SHIPPED
  TWO FIXES FOR DEFECTS THAT WERE NOT THERE. ONE PREMISE CORRECTION KEPT:
  ~/ctaNew-wt-rev carries a data/pm_5min/raw SYMLINK, the only seat worktree
  that does, so it resolves branch 2 -- the TAPE-PRESENT / ARTIFACTS-ABSENT
  layout the resolver's single test cannot see, and exactly where the six checks
  skip. CLOSED IN THE SAME REVIEW: CO-R4 (rc 3, JSON, classification REFUSED,
  distinct from rc 1, no collision in any single channel) and the R-411
  CONSTANTS VERBATIM to R-424 section 4 WITH NO NEW NUMBER (144, 288,
  per_unmasked_hour; counts_toward_G gates nothing yet); the v2 wiring reads
  governs False / True / True for 09-02 / 03 / 04 with V2_TRAILING_DAYS // 2 ==
  3 COMPUTED. AND THE OUTDIR LOOP IS CLOSED: BE's confirming pair now runs into
  a NEW OUTDIR (fwd5, unit be-fwd-final4.service) with Q-BE-229 to follow, and I
  verified fwd4's completed TEN-GATE receipt is still present and intact -- the
  remedy I proposed two rounds ago, AN OUTDIR PER RUN, is in use rather than
  merely recommended. A FORMAT RULING ON WHAT A READER MEETS AT 00:14Z: the
  preflight's open_decisions.ruled carried THREE of R-424's rulings and omitted
  R-408(2) by a stated scoping choice ("three of these were open questions of
  this instrument"); ruled to MIRROR ALL FOUR, because the emission is A
  STATEMENT OF THE REGISTER'S STATE, NOT OF THE INSTRUMENT'S OWN ESCALATION
  HISTORY -- a reader at 00:14Z has no way to know which questions this
  particular tool once asked. ASSERTED THIS ROUND: ZERO FENCED RATIFICATION
  BLOCKS IN BOTH STATE FILES, checked after every edit. REVIEWER QUEUE: DE round
  17 at a8093a5, then DE round 18 at db039a3, then DA round 11, then BE rounds
  3-4 when BE files. OPEN FINDINGS: DA10-R1..R5 to DA round 11 on the held
  commit; DE16-R1..R4 closed at db039a3 pending review; the own-supersedes-
  nonentry residual with the reviewer; BE's landing commit still owed
  (working_tree_dirty true), though the coordinator holds a receipt copy. USER
  DECISIONS: FOUR RULED (R-424), ONE OPEN -- the freeze disposition (R-424
  section 6); the 09-02 accrual call MECHANICAL after 00:06Z. R-419 remains
  revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving
  with Q4 failing; development evidence; 1,344 of 1,875 supplied windows (71.7%)
  produce no score.
```

## Batch 38 — archived 2026-09-02T14:10Z (1 entry, rolling-window overflow)

Moved in the MEM round-32 true-up of R-438. Join rule as in batch 1.

```yaml

  2026-09-02T13:41Z (MEM ROUND 29 -- EXISTENCE IS CHECKED; ASSOCIATION IS NOT).
  R-435 swept. DE ROUND 17 REVIEW VERIFIED AND RELEASED at a8093a5 (797ed9a, 225
  lines, executed 13:33-13:37Z with round 18 not fetched or read), no hold:
  DE15-R1..R4 close AT THE ARTIFACT WITH THE REVIEWER'S OWN MUTANTS -- a fifth
  entry with and without a key, the eval row deleted, eval starting to catch,
  compile starting to refuse, EACH RED AND NAMING THE ROW -- and the
  meanings-swap that sat GREEN AT 104 two rounds ago now dies at "BINDING
  PHRASES". ONE CHECK IN THAT REVIEW IS THE SCEPTICAL ONE WORTH MAKING EVERY
  TIME: the AST call-site census is IDENTICAL at 829910e and a8093a5 (67 / 39 /
  4, same loop lengths), so NO CHECK WAS REMOVED TO KEEP THE COUNT AT 132 -- a
  stable count can mean "nothing changed" or "something was deleted to make
  room", and only comparing the census tells them apart. After a week of counts
  standing in for checks, a reviewer checking what a STABLE count conceals is
  the right instinct. TWO FINDINGS, BOTH LOW, BOTH REPRODUCED BY THE COORDINATOR
  ON FILE COPIES AT A TEMP TREE, AND BOTH THE SAME SHAPE ONE LEVEL UP:
  MEMBERSHIP IS ASSERTED FOR EXISTENCE, NOT FOR ASSOCIATION. DE17-R1:
  BLIND_ENTRY_ASSERTIONS is keyed by LIST POSITION, so swapping entries 0 and 2
  with the map untouched gives "selftest OK -- 75 checks" at rc 0 while THE MAP
  NOW CLAIMS THE RUNPY ASSERTIONS COVER THE GETATTR ENTRY and nothing notices;
  only entry 3 is pinned by a token (:1109). Every closure this week has been
  about a MEMBER being present or a COUNT being right; DE17-R1 IS ABOUT THE
  MAPPING BETWEEN THEM, WHICH NO COUNT CAN SEE. DE17-R2: the OVER-CAUGHT
  paragraph (:172-181) DELETED leaves 75 GREEN -- it is the one statement in the
  block with NO CHECK BEHIND IT, and its own disposition asks for a "together"
  ON TRUST that the blind list enforces structurally. That is CO-7's family
  moved from a fix to a piece of documentation: THE BEHAVIOUR IS RIGHT, AND
  NOTHING WOULD NOTICE IF THE CLAIM ABOUT IT STOPPED BEING TRUE. THE DIRECTION
  CLAIM ITSELF HOLDS AND IS THE SAFE SIDE: the only outside consumer of
  reads_no_verdict is a SELFTEST (ev_replay_seam.py:1484), so a false catch
  REDDENS A SUITE AND NEVER ADMITS -- the finding is that a CORRECT claim is
  UNGUARDED, not that it is wrong. Both are TWO LOW FINDINGS ABOUT THE GAP
  BETWEEN BEING RIGHT AND BEING HELD RIGHT. CLOSURES DISPATCHED AS DE ROUND 19
  (Q-DE-37, de_admissible_windows.py only, one batch): a STABLE TOKEN PER ENTRY
  so a reorder goes red, and the OVER-CAUGHT binding phrase ASSERTED IN THE
  DOCSTRING TEXT as de_ratification_check.py:1151 already does for stamped_at --
  each with its own falsifier (the swap mutant dies by name; the deleted
  paragraph dies by name), 75 -> N stated per check, and nothing else moving
  (the seam's 1,875 specs, daw identity, R-419 True/[]). SEQUENCING NOTE: THE
  DATA_ROOT SPLIT SLIPS AGAIN, from round 19 to ROUND 20, because it stays
  behind DA round 11's landing -- three rounds of deferral, each time for the
  same reason and each time STATED rather than quietly dropped. ASSERTED THIS
  ROUND: ZERO FENCED RATIFICATION BLOCKS IN BOTH STATE FILES, checked after
  every edit. REVIEWER QUEUE: DE round 18 at db039a3, then DA round 11 when
  held, then DE round 19 when filed, then BE rounds 3-4 when BE files; DA round
  10's and DE round 17's reviews are BOTH DONE, the first HELD and the second
  RELEASED. DA ROUND 11 IS IN FLIGHT ON THE HELD COMMIT (3a89e6c, confirmed
  still the worktree tip and unpushed) and BE'S CONFIRMING PAIR IS STILL RUNNING
  into fwd5. OPEN FINDINGS: DE17-R1..R2 to DE round 19; DA10-R1..R5 to DA round
  11 on the held commit; DE16-R1..R4 closed at db039a3 pending review; the
  own-supersedes-nonentry residual with the reviewer; BE's landing commit still
  owed. USER DECISIONS: FOUR RULED (R-424), ONE OPEN -- the freeze disposition
  (R-424 section 6); the 09-02 accrual call MECHANICAL after 00:06Z. R-419
  remains revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of 24
  surviving with Q4 failing; development evidence; 1,344 of 1,875 supplied
  windows (71.7%) produce no score.
```

## Batch 39 — archived 2026-09-02T14:19Z (1 entry, rolling-window overflow)

Moved in the MEM round-33 true-up of R-439. Join rule as in batch 1.

```yaml
  2026-09-02T13:56Z (MEM ROUND 30 -- THE QUESTION DE REFUSED TO ANSWER CAME BACK
  AS A FINDING). R-436 swept; DE's count reproduced and DA's hold verified at the
  worktree. THE PROCESS PAID OUT: DE round 18 declined to close a fifth finding
  by fiat -- an entry whose OWN block claims to supersede a NON-EXISTENT entry
  still verifies -- and THE REVIEWER RULED IT A FINDING (DE18-R1, LOW-MEDIUM) on
  three grounds: THE DEFERRAL RESTS ON A CHECK NOBODY TRIGGERS (DE's "the
  target's existence becomes this question when someone checks that target"
  assumes a later check no one performs); check#1 ALREADY REFUSES THE SAME SHAPE
  ONE FIELD OVER, so the inconsistency is internal; and the predicate and pos are
  ALREADY IN HAND, the closure being one "named not in pos" at the entry under
  check. HAD DE CLOSED IT AS SCOPE, NOTHING WOULD HAVE LOOKED AGAIN -- which is
  what measuring a boundary question and handing it on buys. TWO MORE FINDINGS,
  BOTH LOW: DE18-R2, a quoted block placed FIRST with the own block second is
  REFUSED FOR THE RIGHT REASON BUT NAMES THE WRONG OWNER; DE18-R3, parse_day#1
  neutralised leaves 150 GREEN -- a guard REACHED BY NOTHING. Both to DE round
  20 (Q-DE-38 pending). A CORRECTION THAT LANDS ON SOMETHING I PRAISED LAST
  ROUND: I recorded the reviewer's "no check was removed to keep the count" as a
  sceptical check worth making every time; applied to ROUND 18 the same check
  came back POSITIVE -- the census is 110 -> 124 call sites and THREE CHECKS
  WERE REMOVED, the tautological KNOWN-BADs of DE16-R4, confirmed in the diff --
  so Q-DE-36's "none removed" was WRONG IN THE LETTER AND RIGHT IN SUBSTANCE.
  THAT IS THE BETTER VERSION OF THE LESSON AND IT CORRECTS MY FRAMING: I
  presented the check as one that CONFIRMS a stable count is honest, when its
  real value is that IT CAN COME BACK POSITIVE -- and when it does, the removal
  may still be legitimate, because "NOTHING LOAD-BEARING WAS REMOVED" and
  "NOTHING WAS REMOVED" ARE DIFFERENT CLAIMS AND ONLY ONE OF THEM WAS TRUE. DE
  ROUND 19 VERIFIED at 2f6da2c (Q-DE-37; de_admissible_windows.py only,
  +117/-29; de_ratification_check.py byte-identical to db039a3): admissible 75
  -> 79, reproduced here under both launchers, seam 69, n_supplied_total 1,875,
  R-419 True/[]/[] on the real register. DE17-R1 CLOSED -- the map is now keyed
  by A TOKEN THE ENTRY CONTAINS, with one-to-one and in-order asserted and the
  C-extension entry reached through its own key; DE17-R2 CLOSED --
  declared_limit_text() reads the "#:" block above the list, normalised, with the
  OVER-CAUGHT heading, its binding phrase and the subjects the two checks drive
  all asserted, and the reader driven on a cut copy. NINE COORDINATOR MUTANTS ON
  A FILE COPY, EACH RED BY NAME, including the reader returning the whole file
  NORMALISED, which dies at the known-bad DE itself named as a trap. TWO
  RESIDUALS MEASURED AND NOT RULED, to the reviewer: (A) the PROSE paragraphs of
  entries 0 and 2 swapped INSIDE the "#:" block, list and map untouched, leaves
  79 GREEN because the order check cites the prose order and nothing reads it;
  and (B) a blank non-"#:" line inserted above the OVER-CAUGHT heading. DE
  escalating twice in three rounds rather than self-closing is now the pattern,
  and round 18's outcome is the argument for it. THE FORMAT RULE R-432 SECTION 1
  NARROWS ON EVIDENCE RATHER THAN CAUTION: quotations in NON-RATIFYING entries
  proved harmless in EVERY SPELLING CONSTRUCTED, while a fenced block INSIDE a
  ratifying entry BEFORE its own is not -- so AN R-ADMISS ENTRY CARRIES EXACTLY
  ONE FENCED RATIFICATION BLOCK, ITS OWN, FIRST, while ANY OTHER ENTRY MAY QUOTE
  and the checker ignores it. The coordinator keeps quoting spellings in prose
  with inline backticks regardless, and THESE STATE FILES STILL CARRY ZERO
  FENCED BLOCKS, asserted again this round after every edit. DA ROUND 11 IS HELD
  at e292439 (DA10-R1..R5 plus R-434 section 2 on top of the round-10 batch),
  verified here as UNPUSHED AND ON NO REMOTE BRANCH; it is under review and
  LANDS AFTER THE 00:14Z READ AS Q-DA-209, with tonight running the shared
  tree's v1 unchanged and DA's round-9 00:06Z standby armed separately. Q-BE-229
  VERIFIED AT THE ARTIFACTS: the confirming 09-01 receipt shares 68 NUMERIC
  FIELDS with the superseded 12:49 one and DIFFERS IN TWO -- n_archive_slugs
  27,947 -> 28,031, the archive index having grown between runs and NOT a
  population count, and wall_seconds -- so EVERY POPULATION COUNT IS IDENTICAL
  (1,859 windows, 2,262,457 rows, 1,847,824 actions, btc 610,064 and eth 441,409
  scored, 1,344 supplied without a fit, 141 excluded at supply); 09-02 REFUSED
  AT GATE 1 BY NAME; derived/ untouched at 184 entries. AND A STALE LINE IN THAT
  FILING, RECORDED BECAUSE THESE FILES ARE THE REFERENT: Q-BE-229's disposition
  column says the four R-424 rulings are "USER-pending unchanged" when THEY ARE
  RULED and THE ONLY OPEN USER DECISION IS THE FREEZE DISPOSITION; BE supersedes
  the line in its next row (rule 13) with the old row untouched. R-424 SECTION 1
  RESOLVED A RULING'S SCOPE BY READING THIS TABLE, which is precisely why a
  stale pending-list anywhere else is worth catching rather than shrugging at.
  BE round 5 is dispatched. REVIEWER QUEUE, ALL THREE REQUESTS COMMITTED: DA
  round 11, then DE round 19, then BE rounds 3-4. OPEN FINDINGS: DE18-R1..R3 to
  DE round 20; DE round 19's two residuals with the reviewer, unruled;
  DA10-R1..R5 closed into the held DA round 11; BE's landing commit still owed.
  USER DECISIONS: FOUR RULED (R-424), ONE OPEN -- THE FREEZE DISPOSITION ONLY
  (R-424 section 6); the 09-02 accrual call MECHANICAL after 00:06Z. R-419
  remains revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of 24
  surviving with Q4 failing; development evidence; 1,344 of 1,875 supplied
  windows (71.7%) produce no score.
```

## Batch 40 — archived 2026-09-02T14:26Z (1 entry, rolling-window overflow)

Moved in the MEM round-34 true-up of R-440. Join rule as in batch 1.

```yaml
  2026-09-02T14:05Z (MEM ROUND 31 -- A POSITIVE CONTROL THAT PASSED BECAUSE ITS
  TARGET WAS ABSENT). R-437 swept; DE's count reproduced under both launchers.
  DE ROUND 20 VERIFIED at 0778918 (Q-DE-38 at 235e444; de_ratification_check.py
  only, +139/-17; de_admissible_windows.py byte-identical to 2f6da2c):
  ratification 150 -> 155, rc 0 each way, audit 29 cases / 23 sites with
  coverage True, EXPECTED_SITE 29, markers 29 UNIQUE; R-419 True/[]/[] on the
  real register, R-418 still REFUSED FOR A NEW RUN, seam 1,875. AND THE
  INTERESTING THING IS NOT THE CLOSURE BUT WHAT THE CLOSURE DISTURBED: A FIXTURE
  CHANGED SIDES AND DE READ THE CHANGE AS EVIDENCE.
  fixture_register(supersedes="R-418") had been a POSITIVE CONTROL, and it
  PASSED ONLY BECAUSE R-418 IS ABSENT FROM THAT FIXTURE REGISTER -- nothing to
  do with the property it was meant to demonstrate; under the new existence rule
  it becomes THE KNOWN-BAD (:1558), with the positive control REBUILT ON A
  TWO-ENTRY REGISTER WHERE THE TARGET EXISTS. THAT IS THIS WEEK'S CLASS SEEN
  FROM THE OTHER END: we have been finding controls that CANNOT FAIL, and this
  is a control that PASSED FOR THE WRONG REASON -- both are the same defect, THE
  CHECK IS NOT ATTACHED TO THE PROPERTY, and only one of them ever looks
  suspicious. WHEN A FIXTURE FLIPS ROLE UNDER A NEW RULE, THE FLIP IS EVIDENCE
  ABOUT THE OLD FIXTURE, and DE treated it that way rather than quietly
  re-labelling it. DE18-R1 CLOSED: check#16 (:832) refuses the entry under
  check's OWN supersedes R-777, naming "R-902's own block" and "NO ENTRY R-777",
  where the same string verified True/[] at db039a3 -- driven by an audit case
  (under_check_dangling_supersedes) and in-suite. DE18-R2 CLOSED: the shape rule
  moved from the first-fence branch into the OWN-BLOCK branch after check#8
  (:771), so the coordinator's quotation-first fixture is REFUSED AS A QUOTATION
  -- naming declares ref 'R-903' and WITHOUT "R-999's block" -- asserted ON THE
  MESSAGE TEXT IN BOTH DIRECTIONS (:1672), because a refusal that names the
  wrong thing is still wrong. DE18-R3 CLOSED BY DRIVING THE GUARD, NOT
  ANNOTATING IT (:1528), and DE's reason applies this week's biggest lesson
  PROSPECTIVELY: the guard defends an exported function's contract against a
  direct caller, and ANNOTATING IT UNREACHABLE WOULD DECLARE A LIMIT THE MODULE
  DOES NOT HAVE -- precisely the declared-blind failure, a list claiming a
  limitation the code lacked, REFUSED BEFORE IT COULD BE WRITTEN DOWN. A seat
  declining to create the exact defect the programme spent two rounds removing
  is worth the line. FOUR COORDINATOR MUTANTS ON A FILE COPY, EACH RED BY NAME:
  the existence rule neutralised (the DE18-R1 known-bad), parse_day coercing
  (DE's own "fair mutant", caught by the direct-call known-bad), check#16's
  marker renamed onto check#9 (the coverage assertion), and the EXPECTED_SITE
  row dropped (the coverage assertion again). RESIDUALS A AND B ARE UNTOUCHED BY
  DESIGN -- THE REVIEWER RULES FIRST -- which is the THIRD CONSECUTIVE ROUND in
  which DE has left a measured question open rather than closing it, and R-436
  is the argument for the discipline: the last question DE declined to close
  came back as a RULED FINDING. THE REVIEW REQUEST (REQUEST_DE_ROUND_20_2026-09
  -02.md, seven items) IS QUEUED FOURTH, and DE IS ON DELIBERATE STANDBY: round
  21 is the DATA_ROOT split behind DA's landing after 00:14Z, and residuals A/B
  await the round-19 ruling. REVIEWER QUEUE: DA round 11 IN FLIGHT, then DE
  round 19, then BE rounds 3-4, then DE round 20. BE ROUND 5 AND THE DA-11
  REVIEW ARE IN FLIGHT; DA IS HOLDING. ASSERTED THIS ROUND: ZERO FENCED
  RATIFICATION BLOCKS IN BOTH STATE FILES, checked after every edit. OPEN
  FINDINGS: DE18-R1..R3 CLOSED at 0778918 pending review; DE round 19's two
  residuals with the reviewer, unruled; DA10-R1..R5 closed into the held DA
  round 11 (e292439, unpushed, landing after the 00:14Z read as Q-DA-209); BE's
  landing commit still owed. USER DECISIONS: FOUR RULED (R-424), ONE OPEN -- THE
  FREEZE DISPOSITION ONLY (R-424 section 6); the 09-02 accrual call MECHANICAL
  after 00:06Z. R-419 remains revocable by the USER. UNCHANGED: G=1/5; the 011
  family is 12 of 24 surviving with Q4 failing; development evidence; 1,344 of
  1,875 supplied windows (71.7%) produce no score.
```

## Batch 41 — archived 2026-09-02T14:37Z (1 entry, rolling-window overflow)

Moved in the MEM round-35 true-up of R-441. Join rule as in batch 1.

```yaml
  2026-09-02T14:10Z (MEM ROUND 32 -- THE CODE SAYS "NOT A CLEAN PASS" AND THEN
  RECORDS A PASS). R-438 swept; DA11-R1 read at the held object rather than
  taken. DA ROUND 11 REVIEW VERIFIED AND RELEASED (a5e8b40, 289 lines) FOR
  e292439 AS THE CONTENT OF Q-DA-209: all five DA10 findings and R-434 section 2
  close at the object, and the closure is the strong kind -- ran + skipped == 247
  ASSERTED IN EVERY LAYOUT (worktree 241 + 6; complete scratch root 247 + 0; the
  root minus only the log 241 + 6, rc 0 each), with BOTH FALSIFIERS
  DISCRIMINATING: one check deleted gives rc 1 naming "246 ... expected 247",
  and the pre-fix silent "if _lg_p.exists():" gate restored goes GREEN WITH THE
  LOG AND RED WITHOUT IT, failing exactly where the old code was wrong. Also
  closed at the object: roots in the verdict, both preflight shapes including rc
  3 REFUSED, the mask, one branch per launch; da_hf_pm_alignment 53/53 both
  launchers; _is_tracked True/True/False; the mask suite 30 from the worktree
  and 30 under scratch PM_DATA_ROOT; "ruled" carrying ALL FOUR R-424 rulings
  each citing R-424 with still_open = freeze_disposition ALONE; constants
  unchanged (144 / 288 / per_unmasked_hour; governs F/T/T for 09-02/03/04); the
  shared tree's six files byte-identical to b75c9fe; derived/ 184 entries
  identical before and after; the unit untouched with next elapse 2026-09-03
  00:06:00 UTC. DA11-R1 (LOW-MED) IS THE RECURRING CLASS INSIDE THE BATCH THAT
  CLOSED IT: at pm_tape_density.py:443 the SKIP branch PRINTS "an EMPTY data
  root is a status, not a clean pass" AND THE VERY NEXT STATEMENT IS
  checks.append(True), so the closing line reports the same "N checks passed"
  for a complete root and an EMPTY one. THE CODE STATES THE RULE IN PROSE AND
  BREAKS IT ON THE FOLLOWING LINE -- rule 10's shape, a message beside a
  computation that contradicts it, fused with the control-that-ran-nothing
  class, ONE MODULE OVER FROM WHERE THE SAME DEFECT WAS JUST FIXED. The contrast
  is worth keeping: A ROUND CAN CLOSE ITS CLASS RIGOROUSLY IN ONE MODULE AND
  RE-COMMIT IT IN THE NEXT. DA11-R2 (LOW-MED) HAS A QUIET IRONY: the new
  da_hf_pm_alignment gate is spelled BY PATH, and the roster at e292439 is 21
  gates by path and EXACTLY ONE -m (tier1_pipeline), so THE -m BREAK THAT
  MOTIVATED DA10-R3 WOULD HAVE SAT UNINVOKED in the gate added to catch it. Both
  go to DA ROUND 12, DISPATCHED AND HELD ON TOP OF e292439, five items, with
  NOTHING MOVING FOR TONIGHT and Q-DA-209 LANDING AFTER THE 00:14Z READ WITH THE
  ROUND-12 TIP. ONE ARITHMETIC RECONCILIATION, A TRANSCRIPTION RATHER THAN A
  DEFECT: DA's pane read "238 ran + 6 SKIPs" when ran is 241 and the module's own
  printed line says 241 + 6 = 247 -- 238 WAS ROUND 10'S FIGURE carried forward
  into a round-11 sentence. THE 238 / 244 / 238 IN MY ROUND-28 ENTRY BELONGS TO
  ROUND 10 AND STANDS; this correction is about the pane's REUSE of that number,
  not about that record, and I have separated the two explicitly rather than
  reflexively retracting, because TWO ROUNDS' MEASUREMENTS SHARING A DIGIT IS
  PRECISELY HOW A STALE FIGURE SURVIVES A CORRECTION -- the safe move is to name
  which round a number belongs to. ALSO RECORDED: the RR12-1 branch conjunct is
  != and is satisfied by a missing key (an observation, not a finding); an opt-in
  --require-no-skips strict mode is recommended since rc stays 0 on a skip by
  default; AND A SEAT CLAIM THAT DID NOT REPRODUCE -- DA reported "21 of 22"
  gates and the reviewer MEASURED ALL 22 PASS at e292439, the third
  non-reproducing seat or coordinator claim this week, each time costing less to
  check than the fix would have. CO-8 IS CONFIRMED as a resolver question on the
  COORDINATOR'S OWN SURFACE, for after tonight. REVIEWER QUEUE: DE round 19
  (2f6da2c) IN FLIGHT, then BE rounds 3-4, then DE round 20; DA round 11's
  review is DONE AND RELEASED. DE remains on DELIBERATE STANDBY (round 21 = the
  DATA_ROOT split behind DA's landing; residuals A/B await the round-19 ruling);
  BE round 5 is in flight; DA is holding. ASSERTED THIS ROUND: ZERO FENCED
  RATIFICATION BLOCKS IN BOTH STATE FILES, checked after every edit. OPEN
  FINDINGS: DA11-R1..R2 to DA round 12 (held); DA10-R1..R5 CLOSED at e292439
  with the review released; DE18-R1..R3 closed at 0778918 pending review; DE
  round 19's two residuals with the reviewer, unruled; BE's landing commit still
  owed. USER DECISIONS: FOUR RULED (R-424), ONE OPEN -- THE FREEZE DISPOSITION
  ONLY (R-424 section 6); the 09-02 accrual call MECHANICAL after 00:06Z. R-419
  remains revocable by the USER. UNCHANGED: G=1/5; the 011 family is 12 of 24
  surviving with Q4 failing; development evidence; 1,344 of 1,875 supplied
  windows (71.7%) produce no score.
```

## Batch 42 — archived 2026-09-02T14:44Z (1 entry, rolling-window overflow)

Moved in the MEM round-36 true-up of R-442 and R-443. Join rule as in batch 1.

```yaml
  2026-09-02T14:19Z (MEM ROUND 33 -- THREE FOR THREE: EVERY QUESTION DE
  DECLINED TO CLOSE CAME BACK A FINDING). R-439 swept. The reviewer RELEASED
  DE round 19 at 2f6da2c (filing a558356) and RULED both escalated residuals
  as findings: DE19-R1 (LOW, the order check cites the prose and does not read
  it) and DE19-R2 (LOW-MED, a blank line inside the limit block above
  OVER-CAUGHT truncates the reader 3,754 -> 1,975 chars, 47% of the block
  unread including the heading and both upper sections, suite green), plus
  DE19-R3 (LOW, the declaration check's phrase conjunct has no in-suite
  driver). With DE18's escalation that is three for three on questions DE
  measured but declined to close -- escalating rather than self-closing has
  been right every time. The ruled closure for R-2 is a STRUCTURAL ANCHOR at
  the block's head, NOT a length pin: a pin would go green when the block
  legitimately grows and red for the wrong reason. The closures verified clean
  -- the map binds by content and goes red from either side, an entry that
  merely MENTIONS another's token fails LOUD rather than binding to the first
  match, and the census moved 70 -> 74 with "nothing removed" holding as a
  checked fact. DE round 21 was dispatched on de_admissible_windows.py only
  (DE19-R1..R3) and HAS SINCE FILED at 0255b60 (Q-DE-39, 79 -> 84), not yet
  coordinator-verified; the DATA_ROOT split slips to round 22 behind DA's landing
  after 00:14Z, its fourth deferral. The reviewer's queue is BE rounds 3-4 at
  248e99f, then DE round 20; item 1 of that request is the "frozen bytes" fact
  for the USER's open decision -- STATED, NOT RULED. USER-open: the freeze
  disposition only. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving
  with Q4 failing; development evidence.
```

## Batch 43 — archived 2026-09-02T15:06Z (1 entry, rolling-window overflow)

Moved in the MEM round-37 true-up of R-444 and R-445. Join rule as in batch 1.

```yaml
  2026-09-02T14:26Z (MEM ROUND 34 -- A KNOWN-BAD THAT MOVES THE WORLD AND ONE
  THAT MOVES THE ASSERTION ARE DIFFERENT PROOFS). R-440 swept. DE ROUND 21
  VERIFIED at 0255b60, reproduced here: EXPECTED_CHECKS = 84, rc 0 under both
  launchers from the repo root; DE19-R2 closed by THREE STRUCTURAL ANCHORS
  (THE DECLARED LIMIT / REFUSED (in the sets above) / the full DECLARED BLIND
  heading) with len(_limit) PRINTED AS A FACT AND ASSERTED NOWHERE -- the
  ruled shape, not a length pin; its known-bad reads 1975 of 3752 chars (47%
  unread), the block being 3,752 rather than round 33's 3,754 because a prose
  reword shortened it by two characters -- THE SAME MEASUREMENT, NOT A
  DISAGREEING ONE. DE19-R1's _key_order resolves to [64, 263, 621, 803] and
  its known-bad to [604, 246, 64, 803]. THE FIX SHIPS A GUARD AGAINST BECOMING
  THE DEFECT IT FIXES: len(_anchors) == 3 sits INSIDE the predicate because an
  emptied tuple would satisfy both the check and its known-bad vacuously -- the
  rule-16 failure mode of a NEW control anticipated in the same commit. ONE
  MUTANT STAYS GREEN AND I REPRODUCED IT RATHER THAN RELAYING IT: the phrase
  conjunct neutralised inside the declaration check's OWN ASSERTION leaves the
  suite green at 84; to know that green was the suite's property and not my
  harness's I ran _anchors = () in the same tree and got rc 1, RED BY NAME.
  It is a QUESTION, NOT A FINDING, and correctly routed: the _cut2 known-bad
  falsifies the SUBJECT, proving the check READS THE WORLD, which is a
  different proof from an ASSERTION-mutant proving the conjunct is
  LOAD-BEARING IN THE CHECK'S OWN CODE -- this suite ships only the first
  (item 3 of the DE-21 request). DA ROUND 12 VERIFIED STATICALLY at 636a455
  (held, unpushed, on e292439, four files): DA11-R1's ran + skipped == 9 is
  RAISED BEFORE the summary prints, so the constant in that line is
  unreachable unless the computed sum equals it -- CHECKED, and NOT an
  instance of a count standing in for a check; DA11-R2's -m twins are DERIVED,
  NOT TRANSCRIBED, and I recomputed the arithmetic at the held object rather
  than accepting it: roster 22, twins 14, total 36. The request's own open
  item -- which gates the derivation EXCLUDES -- has a factual answer now:
  EIGHT, seven being two-argument script gates with no --selftest at all (v5
  heartbeat behaviour, v5 deadline falsifier, chain equivalence, chain
  differential fuzz, preflight mutation audit, v4 behaviour (git-extracted),
  v4_1 mutation audit) and the eighth tier1 normalisation, ALREADY the -m
  form, so nothing with a module suite's shape is excluded; stated as MEM's
  computation, the reviewer still rules. TWO CITATION DETAILS FIXED IN PLACE,
  neither a defect: --require-no-skips lives in da_forward_day_verify.py
  (2400/4526/4543/4579), NOT in v5_deploy_gates.py where the roster is, and
  the membership conjunct is da_blackout_mask.py:874 -- filenames recorded
  because bare line numbers is the shape that sent me to a dirty tree in round
  18. DA caught a defect in its OWN INSTRUMENT (a pipe swallowed the exit code:
  rc 0 read for an rc-1 case) and re-measured without it. Reviewer queue: BE
  3-4 at 248e99f (in flight) -> DE 20 at 0778918 -> DE 21 at 0255b60 -> DA 12
  at 636a455 -> BE 5 once its row lands; BE round 5's file is committed at
  90036b7 (+464/-4, 468 lines touched) mid-batch with the row not yet filed,
  and nothing goes to BE in flight (rule 18). DE and DA both on DELIBERATE
  standby, DA holding for the 00:06Z/00:14Z reads; Q-DA-209 lands after the
  00:14Z read with 636a455 as its content. USER-open: the freeze disposition,
  still the only one. UNCHANGED: G=1/5; the 011 family is 12 of 24 surviving
  with Q4 failing; development evidence.
```

## Batch 44 — archived 2026-09-02T15:16Z (1 entry, rolling-window overflow)

Moved in the MEM round-38 true-up of R-446. Join rule as in batch 1.

```yaml
  2026-09-02T14:37Z (MEM ROUND 35 -- THE FACT THE USER'S DECISION TURNS ON,
  HASHED RATHER THAN QUOTED). R-441 swept. BE ROUNDS 3-4 REVIEW RELEASED
  (1d9c543): 248e99f accepted as THE CONFIRMING DRIVER OF RECORD for the 09-01
  score, 46 checks rc 0 both launchers, five findings BE34-R1..R5 (two MEDIUM).
  THE "FROZEN BYTES" FACT FOR THE USER'S OPEN FREEZE-DISPOSITION DECISION IS ON
  THE RECORD -- STATED BY THE REVIEWER, NOT RULED BY ANYONE, notified to the
  USER 14:30Z -- and I re-derived it from the objects rather than reading it:
  the candidate binds manifest_sha256 = eb8733da2c8e2126, and the manifest BLOB
  AT THE FREEZE COMMIT 1b53929 hashes to EXACTLY THAT; the manifest names 8
  reproducibility anchors (7 .py modules + 1 data file) plus one non-anchor hash
  (collector_runs.jsonl); warning_window is imported at module level by
  policy_bounds_v1:44 and an AST COMPARISON OF EVERY FUNCTION AT BOTH COMMITS
  returns exactly one difference, select_holdout, with nothing added or removed
  and select_by_day AST-IDENTICAL; tier1_pipeline's only import site is
  layer2_v1.py:167, inside load_winners (156-174), which nothing on the driver's
  path calls. ONE HALF OF THE FACT IS DOING MORE WORK THAN THE OTHER: the 7 code
  anchors are frozen-by-commit, but the data anchor
  harmful_exposure_rows_v3_eraB.json is UNTRACKED AND 1.24 GB, so NO COMMIT CAN
  EVER FREEZE IT -- the reason being CLAUDE.md's own rule against large data
  files in git, not an oversight. Rule 12's "a freeze is a commit" cannot bind
  it and the receipt is right to call it a DISCLOSURE, not a freeze; that
  distinction is the USER's to weigh and is not a defect. AND ONE FACT OF MY
  OWN, which sharpens what "the frozen bytes execute" means: the manifest HAS
  MOVED since the freeze (03762753 at 248e99f, at HEAD and in the working
  tree), and the driver does not paper over it -- I ran it and section 10(1)
  REFUSES, an independent re-reading agreeing a bound input moved, WITH A
  POSITIVE CONTROL proving a matching contract HOLDS so the gate discriminates
  rather than refusing universally; the code's own comment records that a
  mutant disabling this drift check ONCE SURVIVED and the falsifier was added
  afterwards. So the honest phrasing for a reader of R-424 section 6: THE
  FROZEN BYTES REACH THE RUN BY MATERIALISATION FROM 1b53929, NOT BY READING
  TODAY'S TREE, and the contract against today's tree is refused by name. THE
  FIVE FINDINGS REPRODUCE AT THEIR LINES: BE34-R1 (MEDIUM) build_and_score() at
  :622 has no falsifier and score_rows() at :707 has ZERO CALL SITES (grepped;
  the def line is the only hit), so streamed-vs-held cannot be compared and the
  68-field cross-pass agreement demonstrates DETERMINISM, NOT CORRECTNESS;
  BE34-R2 (MEDIUM) outdir.mkdir(parents=True, exist_ok=True) at :783 with
  fixed-name writes at :758/:773, the shape that destroyed the 12:49 receipt in
  fwd4/; BE34-R3 (LOW-MED) REPO = Path("/home/yuqing/ctaNew") at :35 spawned
  with cwd=str(REPO) at :1420; BE34-R4 (LOW) the usage branch returns 0 at
  :1438 (the print spans 1436-1437); BE34-R5 (LOW) the closure is computed
  statically and the receipt OVER-STATES HEAD EXPOSURE BY ONE MODULE, in BE's
  own disfavour. BE34-R3 IS VISIBLE IN THE TREE RIGHT NOW: the review measured
  46 checks at 248e99f and the driver at HEAD runs 78 (rc 0) because BE's
  round-5 file landed at 90036b7 underneath it. SEQUENCING RULE ADOPTED BY THE
  COORDINATOR, NOT A USER DECISION: no 09-02 scoring run until BE34-R1/R2 close
  AND are reviewed; be-fwd-final4.service is inactive and the driver runs by
  hand, so nothing enforces it but the rule -- a rule with no interlock is a
  promise, kept by a seat rather than by a unit. BE round 6 STAGED behind round
  5's row (rule 18; BE's fwd6 run and 50-mutant audit in flight); reviewer DE 20
  -> DE 21 -> DA 12 -> BE 5. Gate roster checked myself: 21 at 248e99f, 22 at
  e292439, the same file at two commits. USER-open: the freeze disposition, now
  with its fact on the record. UNCHANGED: G=1/5; the 011 family is 12 of 24
  surviving with Q4 failing; development evidence.
```

## Batch 45 — archived 2026-09-02T15:28Z (1 entry, rolling-window overflow)

Moved in the MEM round-39 true-up of R-447. Join rule as in batch 1.

```yaml
  2026-09-02T14:44Z (MEM ROUND 36 -- THE WORD LANDED, AND THE THING IT MAKES
  TRUE IS A SET OF HASHES). R-442 and R-443 swept. THE USER RULED THE FREEZE
  DISPOSITION -- "Yes proceed according to recommendation", verbatim, ~14:33Z,
  after the 14:30Z notification carrying the reviewer's section 1 fact: the race
  runs on the FROZEN BYTES at 1b53929, NO re-freeze, multiplicity stays 2, and
  the fwd5/ 09-01 receipt is THE 09-01 RACE SCORE OF RECORD, no longer an
  estimate in scratch. ALL SIX USER DECISIONS ARE RULED; NONE IS OPEN (four at
  R-424, the freeze disposition at R-442, and the 09-02 accrual call is not a
  separate decision but R-409's principle applied mechanically after 00:06Z).
  CHECKED AT THE BYTES, NOT AT THE ENTRY: all three record files re-hashed and
  EQUAL -- receipt 4000106752f816e4 (14,022 B), sealed file aca22317ab06adbf
  (54,213,086 B), 09-02 receipt 0907b0369e14d77b (1,123 B) -- and the safety
  copy's SHA256SUMS verifies OK on four files. The receipt carries the ruled
  shape field by field: frozen_commit 1b53929, manifest_sha256_bound
  eb8733da2c8e2126, the data anchor compared true with materialised_to null and
  its reason stated, n_not_frozen 2 naming tier1_pipeline and warning_window
  WITH A SHA AT BOTH COMMITS, carrying_commit 248e99f, outcome SCORED,
  coin_coverage.coins_with_a_frozen_fit = ['btc','eth']. THE SCORE STILL
  REPRODUCES TODAY and one input is why that is not automatic: the untracked
  1.24 GB data anchor hashes to 19a50195c34d0af2, exactly what the receipt
  binds, and it is the ONE INPUT NO COMMIT HOLDS AND NO COPY HOLDS -- the safety
  copy took the four small files, not the 1.24 GB anchor and not the
  materialised frozen/ dir (the code anchors are reconstructible from 1b53929,
  which is the point of freezing by commit). Its sha is its only binding,
  checked on every run rather than assumed -- now a property of the RACE SCORE
  OF RECORD, not a caveat about a scratch estimate. The 09-02 receipt is a
  REFUSAL record and its gate has a NAME worth using instead of its ordinal:
  refused_at = day_closed_and_attributed ("20260902 is not closed by calendar
  ... Scoring an OPEN day scores a population that is still growing"). ONE
  INSTRUCTION DID NOT SURVIVE CONTACT WITH THE ARTIFACT: R-442 section 5 asks
  for STATUS.yml still_open to be emptied, and THERE IS NO still_open FIELD IN
  STATUS.yml (searched the parsed document, not the text) -- the field lives in
  DA's da_governed_verdict_preflight.py open_decisions block, and the ":97" in
  the entry resolves to a line of my own prose QUOTING that artifact; the
  six-ruled state is therefore recorded in the forms these files actually use.
  AND THE REAL FIELD NEEDS A SEAT: at DA's held 636a455 the block reads
  esc["still_open"] = {"freeze_disposition": "... awaiting the USER's word."}
  AND A SELFTEST AT :537 ASSERTS "freeze_disposition" in
  r["open_decisions"]["still_open"] -- so the staleness R-442 creates is not
  silent, it is PINNED BY DA'S OWN SUITE, and when the held work lands after the
  00:14Z read the artifact will assert a decision is open that the USER settled
  at 14:33Z. DA's surface, not mine: recorded, not touched. TONIGHT IS
  UNAFFECTED AND I CHECKED WHY: DA's e292439 and 636a455 are NOT ON
  mm-research (ancestry tested; both subjects begin "HELD:"), the branch's last
  commit touching that file is fadc986 at 10:49Z, and the working tree is clean
  against it -- so the 00:14Z preflight runs fadc986, which has NO ruled /
  still_open block at all and cannot carry the stale claim. That is the hold
  working as designed (R-402), worth stating because "DA10-R1..R5 CLOSED at
  e292439" reads like the fixes are in the tree; they are not, deliberately.
  R-443: DE ROUND 20 RELEASED at 0778918 (819d225) -- DE18-R1/R2/R3 closed,
  census 124 -> 129 with "nothing removed" HOLDING this round; both findings
  reproduce, DE20-R1 being all_entries(register_text) at exactly three call
  sites (:342, :714, :831; definition :263) and DE20-R2 that existence still
  lacks DIRECTION so a self-supersession verifies; DE round 22 (Q-DE-40) in
  flight on de_ratification_check.py carrying two Q-DE-38 accounting corrections
  SUPERSEDED IN-BAND rather than edited; reviewer -> DE round 21. BE round 6 now
  also carries R-442 section 3(c): both receipts land under
  data/pm_5min/derived/ BYTE-IDENTICAL with shas asserted, no re-emission and no
  new field, AFTER the 00:14Z read on 09-03; the sealed file stays external at
  54 MB, identified by content. UNCHANGED: BE34-R1..R5 open; the sequencing rule
  (no second scoring day until R1/R2 close AND are reviewed); tonight's units;
  Phase-4 gated; R-419 revocable; G=1/5; the 011 family 12 of 24 with Q4
  failing.
```

## Batch 46 — archived 2026-09-02T15:33Z (1 entry, rolling-window overflow)

Moved in the MEM round-40 true-up of R-448. Join rule as in batch 1.

```yaml
  2026-09-02T15:06Z (MEM ROUND 37 -- THE SAME BYTES, FROM A DIFFERENT DRIVER).
  R-444 and R-445 swept. THE 09-01 SCORE OF RECORD REPRODUCES BYTE-FOR-BYTE and
  I ran cmp, not the report: BE's fwd6/ re-run -- driver sha 4c0425c578e36b2a
  (the 90036b7 file), a commit and a rebuilt file after the run that produced
  the record -- writes a sealed scores file sha aca22317ab06adbf that is
  cmp-EQUAL to fwd5/'s. The receipts DIFFER (a568346660a3b4db, 20,895 B against
  14,022 B) and that is the CORRECT shape: the receipt carries provenance,
  counts and identities, which grew with round 5's disclosures; the sealed file
  carries the values, which did not move. A reproduction that changed the
  receipt and not the scores is the one you want. 09-02 refusal receipt
  dd730f1aba7c67af; supersedes_receipt absent (first write of the run).
  TONIGHT'S PREFLIGHT LINE IS STALE AND THE USEFUL PART IS KNOWING HOW STALE:
  the shared tree's da_governed_verdict_preflight.py hashes to 6a15ed5dd25513b7,
  BYTE-IDENTICAL TO fadc986 (round-9 vintage, verified by hash not by reading),
  with NO ruled/still_open split at all -- it prints register_ids_transcribed
  (:340) with THREE entries each labelled "-- USER". So the 00:14Z run is stale
  about R-424's FOUR rulings as well as R-442's: a KNOWN-STALE PROVENANCE LINE
  of round-9 vintage, NOT three live open decisions. THE RULED STATE IS THE
  REGISTER'S: SIX RULED, NONE OPEN. R-402 working as designed, written down so
  nobody reading tonight's artifact tomorrow mistakes a vintage for a status.
  THE FINDING I RAISED LAST ROUND IS CLOSED, AND CLOSED GENERALLY: at DA's held
  e384792 freeze_disposition moves into `ruled` with R-442's words, still_open
  == {}, and _assert_decisions_coherent (:121, called from the production path
  at :428) makes the contradiction UNREPRESENTABLE -- any key in BOTH halves
  refuses naming it, any pre-ruling phrase surviving in the block refuses
  quoting it; its docstring names the instance ("which is how
  freeze_disposition read as 'awaiting the USER's word' for the whole of
  R-442's afternoon"). A one-key fix would have left the same trap for the next
  ruling. The guard's own first version was RULE 17's CLASS AGAIN -- driven by
  every check, called by nothing -- closed by poisoning the phrase list with a
  string the real block carries so only the production call can raise; THIRD
  TIME that class has been met in this one file. TWO REGISTER-DISCIPLINE FACTS
  VERIFIED RATHER THAN TAKEN: 768465a is EXACTLY one insertion and one deletion,
  a single -/+ pair on the Q-BE-230 row, so the row was REWRITTEN IN PLACE where
  the register's rows are append-only and rule 13's shape is a superseding row
  (nothing lost, git keeps both; recorded, not adjudicated); and on the row's own
  numbers the parenthetical reads 50 mutants -> 5 survived, 49 -> 3, 47 -> 47/47
  killed while the body says the first audit left 4 and the second 3, so THE
  SECOND AND THIRD PASSES AGREE AND THE DISAGREEMENT IS ISOLATED TO THE FIRST,
  5 against 4 -- stated precisely because it tells BE round 6 where to look.
  Q-BE-230's disposition column calls the freeze disposition "the ONLY open USER
  decision" TWICE with an as-of of 14:52Z, FIFTEEN MINUTES AFTER R-442 ruled it;
  BE had not read R-442; the register's state governs and BE round 6 supersedes
  the column in band; the decisions table stays SIX RULED, NONE OPEN. BE34-R2 IS
  GENUINELY CLOSED and I read the code, not the claim: an existing receipt is
  KEPT byte-identical and the run takes a NUMBERED SUCCESSOR carrying a
  supersedes_receipt block with the prior path, its sha and the reason, driven by
  a selftest asserting the successor exists and that the recorded sha equals the
  kept file's. COUNTS REPRODUCED HERE AT HEAD: ratification 160, admissible 87,
  driver 85, each rc 0. RR5-1/RR5-2 and RR7-1/RR7-2 are CLOSED (e8a9480) and I
  checked my own files for a line still staging them: THERE IS NONE -- the single
  mention is a historic entry recording that RR7-1/RR7-2 were filed NOT HOLDING,
  true when written and kept as provenance. DE'S METHOD FACT DESERVES TO OUTLIVE
  ITS ROUND: a flip mutant and its restore differed by ONE CHARACTER -- same
  size, same mtime second -- so __pycache__ kept executing the MUTANT's bytecode
  and the suite failed on a CORRECT file; every mutant since clears the cache on
  both sides. A mutation harness that does not invalidate bytecode is measuring
  the wrong file, and this failure mode is a false RED, the survivable direction;
  the same collision with the signs reversed would have been a false GREEN.
  SEATS: DA rounds 13 (e384792, 39) and 14 (DA12-R1) held; DE rounds 22 (92fc615,
  160) and 23 (a83083a, 87) landed and DE on DELIBERATE standby; BE round 6 in
  flight; reviewer on DE round 22 then BE round 5. DA's session-local 00:06Z
  standby wait was KILLED at ~14:43Z, the THIRD such kill this session, cause
  unknown, and DA is not re-arming: the audited legs are the BOX-LEVEL timers
  (da-midnight-verify.timer 00:06Z, co-preflight-20260902.timer 00:14Z), exactly
  as Q-MEM-3 found when both session-local legs proved invisible to crontab and
  systemctl. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 47 — archived 2026-09-02T15:36Z (1 entry, rolling-window overflow)

Moved in the MEM round-41 true-up of R-449. Join rule as in batch 1.

```yaml
  2026-09-02T15:16Z (MEM ROUND 38 -- A DUPLICATE THE REGISTER ALREADY CARRIES,
  AND A LINE NUMBER THAT POINTS ONE SHORT). R-446 swept. DE ROUND 22 RELEASED at
  92fc615 (8df60bf) with TWO RULINGS ADOPTED AS PROGRAMME STANDARDS, now
  recorded in standing_rules above and in HANDOFF, not only in the log: a
  CAUGHT-AND-NAMED REFUSAL inside a POSITIVE CONTROL is the right shape when the
  catch is NARROW (the module's own refusal type, verified at
  de_ratification_check.py :1076/:1665/:1752, each catching RatificationRefused
  only) and the sentinel is a CONJUNCT so it fails rather than degrades; and
  EVERY SEAT'S MUTANT LOOP CLEARS __pycache__ BEFORE EACH EXECUTION -- DE met
  the mechanism as a false RED, the survivable side, and the standard exists
  because the same collision with the signs reversed is a FALSE GREEN that
  nothing in a suite would report. DE22-R1 REPRODUCED ON THE REAL REGISTER WITH
  DE'S OWN PARSER: all_entries returns 437 entries / 436 distinct refs at my
  as-of (through R-446) against the coordinator's 436/435 through R-445 -- the
  delta is exactly the entry that landed between the two readings, which is what
  a growing tape looks like when both carry their as-of. R-6 heads TWO entries
  and entry_index resolves to the LATER one, nothing refusing it and nothing
  reporting it. BOUNDED, AND I CHECKED THE BOUND: no ratification block declares
  supersedes: R-6 -- the single text hit is INSIDE R-446's own prose saying so,
  the vocabulary-hit-is-not-a-reference shape again, separated by parsing the
  four fenced blocks rather than grepping; R-6 carries no block; latent, not
  live. ONE THING I FOUND THAT IS LIVE, BELONGING TO THE ROUND IN FLIGHT: DE's
  all_entries records `line` 0-INDEXED (:278, line: i from enumerate) while
  grep -n and every editor are 1-indexed -- R-6's entries are at FILE lines 1781
  and 9507 and the parser reports 1780 and 9506 -- and this already reaches
  prose, since check#18's refusal at :884-887 prints own_idx[ref]["line"] RAW
  under the words "register line", while the other extractor at :432 returns
  line_start = start + 1, 1-INDEXED. The ruled closure (a) requires the refusal
  to NAME BOTH LINES, so the convention decides whether a reader who follows the
  message lands on the entry or one line above it. Recorded for DE round 24 to
  rule; not fixed here. A SECOND MEASUREMENT CAVEAT: R-446 cites "217 of 436
  entry headings are stamped" as what makes "an unstamped heading is not an
  entry" unavailable as a rule; I recompute 217 of 437 under a strict
  "### R-N -- <ISO> -- " shape and 244 under a looser "heading contains an ISO
  stamp", so THE COUNT IS INSTRUMENT-DEPENDENT and the unstamped set is not
  purely the early era (R-226, R-227, R-228 and R-239 fall in it too). The
  ruling's premise survives both readings -- either way roughly half the
  register is unstamped -- but any future rule leaning on that count must state
  the shape with it. CLOSURE RULED at R-446 section 3 and dispatched as DE round
  24 (Q-DE-42, in flight): refuse where a duplicate can reach an answer, report
  by name where it cannot, FIRST occurrence kept BY RULE and stated, never
  chosen, and NO REGISTER EDIT (append-only, rule 13). NOTHING MOVES FOR
  TONIGHT: the 00:14Z preflight is round-9 vintage and does not import the
  checker; BE's require_verified() gate reads the real register, on which R-6
  falls in the REPORTED class. SEATS: reviewer on BE round 5 at baa986d, then DE
  round 23, then DE round 24; DE round 24, BE round 6 and DA round 14 (held) in
  flight. USER decisions: SIX RULED, NONE OPEN. UNCHANGED: G=1/5; the 011 family
  12 of 24 with Q4 failing; the sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 48 — archived 2026-09-02T15:46Z (1 entry, rolling-window overflow)

Moved in the MEM round-42 true-up of R-450. Join rule as in batch 1.

```yaml
  2026-09-02T15:28Z (MEM ROUND 39 -- THE NUMBERS THAT MOVE, AND THE ONE MESSAGE
  THAT STILL DOES NOT SAY WHICH KIND IT PRINTS). R-447 swept. DE ROUND 24
  VERIFIED at e0d1e9f: 168 checks reproduced here rc 0, the real register's R-6
  REPORTED at 0-based 1782/9508 (grep -n shows the headings at 1783/9509) with
  every live answer unchanged, require_verified() RETURNS on the reported
  duplication, R-418 refuses for a new run naming R-419, FIRST occurrence kept
  BY RULE and computed from the parse with no allowlist, and NO REGISTER EDIT.
  THE RIGHT LESSON FROM THOSE NUMBERS IS THAT THEY MOVE: the Q-filing table sits
  above line 1780, so EVERY ROW ANY SEAT FILES shifts both lines by one -- the
  reviewer's 1780/9506, the coordinator's 1782/9508 and mine are ONE FACT AT
  THREE AS-OFS, not three measurements disagreeing -- and DE's suite compares
  the index against an INDEPENDENT RECOUNT rather than a pinned literal, which
  is why it passed at 168 on an already-shifted register. A literal there would
  have been a time bomb with a filing cadence for a fuse. MY ROUND-38 NOTE WAS
  TAKEN UP AND I CAN SAY HOW FAR: every site DE added LABELS the convention in
  the message itself (:398, :411, :425 all say "0-based lines", and the suite's
  own assertion at :1811 says it too), which is the better fix than shifting the
  numbers because parse and prose now describe ONE system; THE RESIDUAL IS
  EXACTLY ONE MESSAGE -- check#18 at :967-970 still prints own_idx[ref]["line"]
  RAW under the bare words "register line", unlabelled, so a reader following
  that refusal still lands one line above the entry. Narrower than what I filed
  last round; belongs to the round-24 review or round 25. DA ROUND 14 HELD at
  801eb31 (chain 3a89e6c -> e292439 -> 636a455 -> e384792 -> 801eb31) and
  RECOMPUTED RATHER THAN ACCEPTED: in a PARITY TREE the roster is 23 declared +
  15 twins = 38, the excluded list holds 8 entries (seven behavioural gates with
  NO OTHER LAUNCHER TO DERIVE plus tier1 normalisation by name), and the
  synthetic-roster --selftest gives 6 checks rc 0. A CAVEAT ABOUT MY OWN
  INSTRUMENT: my first attempt read 2 twins because the derivation anchors on
  Path(argv[1]).parent == HERE and I ran a copy OUTSIDE live/pm_research/ -- the
  measurement is PATH-SENSITIVE, recorded because the wrong number I nearly
  reported was MINE, not DA's. The exclusion's reason reproduces at the source:
  python3 live/pm_research/tier1_pipeline.py --selftest -> rc 1
  ModuleNotFoundError: No module named 'live' (tier1_pipeline.py:55, a
  package-absolute import) against rc 0 under -m; NAMED, NOT REPAIRED, and named
  IN THE CODE (TWIN_EXCLUSIONS) with that reason so the next reader gets the fact
  and not just the exclusion. THE SCOPE DEVIATION IS RECORDED AS ACCEPTED with
  its open question attached: da_blackout_mask.py, ONE assertion in a selftest
  region -- the RR12-1 control asserted tree_dirty_on_producing_files is True,
  which held only while the fixture's copied files differed from the child's
  HEAD, so it went red the first time a commit touched none of them (e384792
  changed only the preflight); the expectation is now COMPUTED from the child
  tree's own git status --porcelain, and the code names it the THIRD INSTANCE OF
  THE DA10-R5 CLASS IN THE SAME CONTROL. A red control may not land, the hunk is
  one assertion, the property is the right one -- accepted in-batch by the
  coordinator; what it now discriminates (whether a constant-True and a
  constant-False mutant can both be red on one fixture arrangement) is REVIEW
  ITEM 8, a ROUND-15 CANDIDATE, NOT A HOLD. BE ROUND 6 IS CODE WITHOUT A ROW:
  faaabdc is in the tree, Q-BE-231 is ABSENT (checked, not assumed), so nothing
  about it is recorded as verified and nothing goes to BE while its batch is in
  flight (rule 18). SEQUENCING: reviewer BE round 5 (in flight) -> DE 23
  (a83083a) -> DE 24 (e0d1e9f) -> DA 13+14 (801eb31) -> BE 6 when its row lands;
  all three requests filed. DE and DA on DELIBERATE standby (R-381). BE round 7,
  the durable landing of the two receipts, is staged for AFTER the 00:14Z read,
  and Q-DA-209/210/211 land together after that read with 801eb31 as their
  content. Standing rules stay at 7; USER decisions SIX RULED, NONE OPEN.
  UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule
  (no second scoring day until BE34-R1/R2 close and are reviewed); Phase-4
  gated; R-419 revocable.
```

## Batch 49 — archived 2026-09-02T15:57Z (1 entry, rolling-window overflow)

Moved in the MEM round-43 true-up of R-451. Join rule as in batch 1.

```yaml
  2026-09-02T15:33Z (MEM ROUND 40 -- A COUNT THAT DEPENDS ON SOMEBODY ELSE'S
  UNCOMMITTED FILE). R-448 swept. BE ROUND 5 REVIEW RELEASED (d990162, 314
  lines) for baa986d, NO HOLD, and BE34-R2 CLOSED AT THE ARTIFACT rather than in
  the code alone: three runs into one outdir yield base + .1 + .2 with the base
  BYTE-IDENTICAL throughout, where rounds 3-4 left one file replaced in place;
  "same run" is rec["_receipt_path"], an IN-MEMORY key STRIPPED FROM EVERY
  WRITTEN RECEIPT (:1013, :1019, :1037, :1042 -- I read all four), not a
  timestamp, pid or file name, which is why three runs inside one second still
  produce three distinct files. ALL THREE FINDINGS REPRODUCE AT THE PINNED
  BYTES: BE5-R1, _flush (:996) computes the next free .N but records
  supersedes_receipt.path as p, THE CANONICAL BASE (:1029-1030), whatever N is
  -- so with base/.1/.2 both successors name the base, the supersession graph is
  a STAR, and "which is current" is answerable only by sorting filenames;
  BE5-R2, DECISION_ALLOWLIST (:954) has ZERO membership assertions (grepped)
  while the suite asserts excused_paths == ["gates[].gate"] (:1781), which
  reports what THIS EMISSION USED, so a second excused path leaves the suite
  green; BE5-R3, the module ships NO mutation audit -- the word "mutation"
  occurs ONCE, in a comment -- so "47/47 killed" is a report in a filing and the
  4-vs-5 I narrowed last round is UNSETTLEABLE BY ANY READER. BE5-R3 CARRIES THE
  STRUCTURAL MORAL: the two-way rebuild pins 90036b7's bytes (4c0425c5) while
  47/47 is reported at baa986d's (65da7ae0), so THE REBUILD EVIDENCE DOES NOT
  CARRY ACROSS THE COMMIT IT WAS MADE AT; closure is rule 15 at the HARNESS
  level -- ship the mutant table, assert survivors == [] in the suite, clear the
  cache before each execution (R-446) -- which is what the DE modules already do
  and why DE's counts can be re-derived by anyone. THE COUNT CORRECTION IS THE
  SHARPEST ITEM AND I CONFIRMED THE MECHANISM MYSELF: _selftest_launch spawns a
  child with BE_FORWARD_LAUNCH_CHECK=1 and cwd=REPO, the child skips the spawn
  and the parent adds the launch check, so 84 IS THE REPRODUCIBLE FIGURE AT
  baa986d AND THE 85TH IS THE SPAWN, whose child reads THE SHARED TREE'S FILE,
  not the pinned one (BE34-R3). Today that file is DIRTY with BE's round-6 WIP:
  committed HEAD 8a851eae, worktree e6cda52f, the pin 65da7ae0 -- and running
  the tree right now gives 94 full / 92 with the spawn skipped, a TWO-check
  launch contribution, not one. SO NO COUNT TAKEN FROM THAT FILE TODAY IS A
  COMMITTED FIGURE AT ALL, and Q-BE-230's "85" holds only while the shared tree
  equals the pin; that is BE34-R3 stated as a number rather than as a shape, and
  round 6 closing it is what makes any of these counts reproducible. TWO RULINGS
  ADOPTED, the first reframing what a gate is: require_verified() is the gate BY
  DATA DEPENDENCY -- its return value is consumed, so deleting the production
  call is NameError, RC 1, NOT A SILENT BYPASS, and faking the result is red at
  the PROVENANCE conjunct, the one BE's own pair logic cannot hold; the
  exception-type assertion is the SMALLER half of why. General form worth
  keeping: A CALL WHOSE RESULT IS CONSUMED CANNOT BE DELETED QUIETLY; A CALL MADE
  ONLY FOR ITS SIDE EFFECT CAN. Second ruling: the ONE excused path's SHAPE IS
  RIGHT (path-bound, string-typed, receipt-reported, vocabulary borrowed by
  value) and its weakness is GOVERNANCE -- growth invisible until used -- which
  is precisely BE5-R2. ROUTING AND WHY IT IS ORDERED THIS WAY: BE ROUND 7 =
  BE5-R1 + R2 + R3, ONE BATCH, dispatched when Q-BE-231 lands, BEFORE tonight's
  read (three one-edit closures plus the shipped audit; no run against a real
  day; nothing under derived/); THE DURABLE LANDING BECOMES BE ROUND 8, AFTER
  the 00:14Z read -- the landing is a FIRST WRITE INTO AN EMPTY TARGET so
  BE5-R1's successor naming never touches it, but the driver that lands it
  should already carry the audit and the pinned allowlist, which is why the
  order is this way round. MY check#18 RESIDUAL HAS A HOME: it sits inside the
  reviewer's round-24 ITEM 4 (consistency across every message that prints a
  line), so it reaches DE through that review or round 25 -- recorded there, not
  re-filed. ROSTER: BE34-R1/R3/R4/R5 -> BE r6 (row pending); BE5-R1/R2/R3 -> BE
  r7 (staged); DA12-R1 -> DA r14 (held, verified); DE22-R1 -> DE r24 (verified);
  CO-8 and the --require-no-skips shape -> after tonight. Reviewer: DE 23 (in
  flight) -> DE 24 -> DA 13+14 -> BE 6 -> BE 7. USER decisions SIX RULED, NONE
  OPEN. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 50 — archived 2026-09-02T16:16Z (1 entry, rolling-window overflow)

Moved in the MEM round-44 true-up of R-452. Join rule as in batch 1.

```yaml
  2026-09-02T15:36Z (MEM ROUND 41 -- A GUARD THAT ASKS THE WRONG QUESTION, AND A
  KNOWN-BAD THAT WILL FAIL WHEN THE CODE IS RIGHT). R-449 swept. DE ROUND 23
  REVIEW RELEASED (0b03618, 227 lines) for a83083a, NO HOLD: DE21-R1 closed for
  BOTH cut shapes through the reader, the predicate carrying ZERO anchor tokens
  in its body (prose-blind as claimed), both _declaration_holds conjunct drops
  red at the known-bad driving each half, declared_limit_boundary accepted as
  new module surface BY DESIGN. The shared tree's de_admissible_windows.py is
  BYTE-IDENTICAL to a83083a (checked). DE23-R1 REPRODUCED FROM SCRATCH: the
  predicate at :281 is `not above.startswith("#:")`, testing #:-ness where the
  property wanted is THE RUN WAS NOT CUT -- four different single lines inserted
  above the OVER-CAUGHT paragraph (a plain # comment, an INDENTED #:
  continuation, a bare #, and a code line X = 1) EACH read 1,975 of 3,752 chars,
  47% unread, with stopped_at_a_real_boundary TRUE; same numbers as the
  reviewer's and the coordinator's, arrived at independently. I ALSO DROVE THE
  PROPOSED CLOSURE IN BOTH DIRECTIONS: not above.lstrip().startswith("#")
  returns False -- correctly red -- for all three comment shapes and True for
  X = 1, so it covers three of four and THE FOURTH IS NOT A GAP BUT AN IDENTITY:
  the intact boundary IS a code line, so no predicate over that one line can
  separate the legitimate stop from the mutant. A LIMIT THAT CANNOT BE CLOSED
  SHOULD BE STATED, NOT APPROXIMATED -- the docstring, the module's own idiom.
  DE23-R2 REPRODUCED AND THEN BUILT PROSPECTIVELY: the extent known-bad
  (:1234-1248) asserts len(declared_limit_text(_above_head)) == len(_limit),
  true only because TODAY the head is the run's topmost line so cutting above it
  removes nothing; a CONTIGUOUS #: paragraph above the head -- the upward growth
  the round cites as its reason -- leaves the predicate CORRECT (boundary True,
  all three anchors present) while the block grows: 3,805 chars in my copy,
  3,791 in the coordinator's, THE DIFFERENCE BEING ONLY THE TEXT EACH OF US
  INSERTED (the effect is invariant to the text, the number is not -- said so
  nobody later "corrects" one to the other). I then GREW THE MODULE AND RAN THE
  GROWN MODULE'S OWN KNOWN-BAD AGAINST IT: the equality conjunct returns FALSE,
  because _limit is the in-memory block (3,805) while the known-bad reads its
  copy from Path(__file__).read_text() (3,752) -- THE SUITE GOES RED WITH
  NOTHING WRONG. That is DE21-R1's shape one artefact over: the first was a
  check that stayed GREEN when the world moved, this is a check that goes RED
  when the world IMPROVES, and both come from comparing against A NUMBER instead
  of against THE SAME SOURCE. TWO RULINGS ADOPTED: the shape is right and the
  predicate is ONE TOKEN SHORT (a design can be correct and its implementation
  still incomplete); and the round-21 ruling STANDS, REFINED -- lifting a
  predicate converts ASSERTION into SUBJECT for everything inside it, so the
  un-falsifiable surface shrinks to the ok(...) line alone, which is the general
  answer to the assertion-mutant question raised in round 34: not
  "assertion-mutants don't matter" but MAKE THE ASSERTION SMALLER UNTIL WHAT IT
  CONTAINS IS SUBJECT. DE ROUND 25 (Q-DE-43) DISPATCHED with both closures,
  de_admissible_windows.py only, de_ratification_check.py untouched while under
  review. THE REGISTER MOVED AGAIN AND THIS TIME I MOVED IT: R-6 parses at
  0-based 1784/9510 at my as-of against the coordinator's 1783/9509 at 1ba459c,
  the shift being MY OWN Q-MEM-28 ROW landing in between -- fourth as-of in the
  sequence (1780/9506 -> 1782/9508 -> 1783/9509 -> 1784/9510) and the cleanest
  illustration of RECOUNT, NEVER PIN: the seat recording the number is one of
  the things that moves it. BOTH ROUND-39 STATEMENTS RECORDED WHERE THEY BELONG,
  NOT RE-FILED: the check#18 residual (:967-970, the raw field under the bare
  words "register line") sits inside the ROUND-24 REVIEW'S ITEM 4 and reaches DE
  through that review or round 25; and my own instrument caveat -- a relocated
  copy of DA's runner derives fewer twins because _launch_twins anchors on
  Path(argv[1]).parent == HERE -- is noted FOR THE DA ROUNDS 13+14 REVIEW as a
  property of the runner, since the runner's own count assertion is what would
  refuse such a copy. SEATS: DE round 25 in flight; BE round 6's row pending
  (nothing to BE until Q-BE-231; round 7 staged with BE5-R1/R2/R3, round 8 the
  durable landing after the read); DA on deliberate standby; reviewer DE 24
  (e0d1e9f, in flight) -> DA 13+14 -> BE 6 -> BE 7. USER decisions SIX RULED,
  NONE OPEN. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 51 — archived 2026-09-02T16:28Z (1 entry, rolling-window overflow)

Moved in the MEM round-45 true-up of R-453. Join rule as in batch 1.

```yaml
  2026-09-02T15:46Z (MEM ROUND 42 -- A RULE THAT STOPPED LEANING ON A PROSE
  CONVENTION). R-450 swept. DE ROUND 25 VERIFIED at 50a9113 and RE-RUN HERE with
  the cache cleared first: EXPECTED_CHECKS = 91 at :839, 91 checks rc 0; on
  in-memory copies the intact walk is first_read_line 139 / above_line 137 /
  boundary True at 3,752 chars, and the three interruption shapes I drove last
  round (plain #, indented #:, bare #) now read 1,975 with boundary FALSE where
  they were TRUE at a83083a -- DE23-R1 closed exactly at the token I checked
  before it landed (:301, not above.lstrip().startswith("#")). THE FOURTH SHAPE
  IS NOW A DECLARATION RATHER THAN AN OMISSION: X = 1 still reads 1,975 with
  boundary True and :279 says why in the docstring -- a code line inserted into
  the run is indistinguishable from a real boundary because THE INTACT BOUNDARY
  IS A CODE LINE -- stated, not chased, with the anchors still naming which
  sections must be present so the residual is bounded rather than merely
  admitted. DE23-R2's length conjunct is GONE (zero occurrences; the block length
  printed, asserted nowhere), replaced by a POSITIVE CONTROL ON ONE SOURCE
  (:1297): a contiguous paragraph above the head keeps the boundary True and
  grows the block, the same copy with a blank between refuses. THREE INDEPENDENT
  INSERTIONS NOW EXIST FOR THAT CASE -- DE's 3,811, the coordinator's 3,791,
  mine 3,805 -- which is the invariance I recorded last round holding up in
  public: the effect is invariant to the inserted text, the number is not; three
  numbers, one behaviour, nothing to reconcile. THE RULING IS THE ROUND'S REAL
  CONTENT AND IT IS A REFINEMENT, NOT A REVERSAL: R-446 section 3(a)(ii), "named
  by any supersedes: in the register", now reads "named by the supersedes: of any
  entry's OWN ratification block" (own_ratification_blocks). RECORDED IN BAND IN
  THE STANDARDS SECTION OF HANDOFF WHERE R-446'S RULE LIVES, WITH R-446'S TEXT
  LEFT STANDING AS PROVENANCE (rule 13) -- never rewritten. It is right because
  the module already made this distinction: superseded_by reads own blocks only,
  and DE16-R1 settled in round 18 that a QUOTED block is not the quoting entry's
  ratification, so a quotation naming a duplicated ref CANNOT REACH AN ANSWER and
  by R-446 section 3's OWN CRITERION belongs on the reporting side. WHAT THAT
  BUYS IS WORTH NAMING: before the refinement the rule stayed sound only because
  R-432 section 1 -- A FORMAT CONVENTION ABOUT PROSE -- kept quoted fences rare,
  and A CORRECTNESS RULE RESTING ON A FORMATTING HABIT IS A RULE WITH AN
  UNDECLARED DEPENDENCY; the refinement removes it, every answer-reaching case
  still refuses, and none of it depends on how anyone writes an entry -- the same
  move as computing an expectation instead of asserting today's arrangement (the
  DA10-R5 shape), applied to a RULE instead of a control. Verified at the source:
  `named` is built from _fenced_blocks(e) at :404-405 (every fence, owned or
  quoted) while own_ratification_blocks exists separately at :580. DE24-R2 is the
  residual I measured in round 41, now CONFIRMED AT THE ARTIFACT and accepted as
  a finding: check#18 prints the 0-based field under the bare words "register
  line" (:967) and "(line ...)" (:970) while FOUR sites say "0-based lines";
  closure is those two words. It reached DE through the round-24 review's item 4
  exactly as R-448 routed it -- recorded, never re-filed, and it arrives with the
  reviewer's confirmation rather than mine alone. Both dispatched as DE ROUND 26
  (Q-DE-44), de_ratification_check.py only, with the reviewer's quoted-block
  fixture as a POSITIVE control (a quotation naming a duplicated ref must NOT
  refuse) and an own block naming one as the KNOWN-BAD. THE RECOUNT IS NOW STATED
  AS A RULE: R-6 sits at 0-based 1786/9512 at my as-of, matching the
  coordinator's reading at 304cd5f -- the fifth as-of in the chain (1780/9506 ->
  1781/9507 -> 1782/9508 -> 1783/9509 -> 1784/9510 -> 1786/9512) -- and THE
  DURABLE FORM IS THE SENTENCE, NOT THE PAIR: every filed Q-row moves both lines
  by one, the suite recounts; no number is pinned in these files. ONE COUNT
  STATED PRECISELY BECAUSE I CHECKED IT: the round-24 review filing is 253 lines
  by wc -l and 253 insertions by the diff where R-450 section 2 says 254 -- a
  one-line difference with nothing resting on it, recorded rather than silently
  normalised. Q-BE-231 IS STILL ABSENT (checked again, not assumed), so BE round
  6 remains unverified here and its row-pending line stands. SEATS: reviewer on
  DA rounds 13+14 at 801eb31 (held, in flight), then DE round 25 (REQUEST filed
  at dc83580), then BE round 6 when its row lands, then BE round 7; DE round 26
  in flight; DA on deliberate standby. USER decisions SIX RULED, NONE OPEN.
  UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule;
  Phase-4 gated; R-419 revocable.
```

## Batch 52 — archived 2026-09-02T16:38Z (1 entry, rolling-window overflow)

Moved in the MEM round-46 true-up of R-454. Join rule as in batch 1.

```yaml
  2026-09-02T15:57Z (MEM ROUND 43 -- A REFUSAL THAT DEPENDS ON AN UNRELATED
  ENTRY SOMEWHERE ELSE IN THE FILE). R-451 swept. DE ROUND 26 VERIFIED at
  89aef8c and RE-RUN HERE: EXPECTED_CHECKS = 171 (:1128), 171 rc 0; `named` is
  built from own_ratification_blocks(e) (:422-423); and DE24-R2 IS GONE --
  check#18 now reads "0-based register line" (:987) and "0-based line" (:990),
  so ALL SIX SITES THAT PRINT A LINE NOW SAY WHICH KIND IT IS. That residual
  took three rounds to travel from a measurement of mine to a labelled message
  and never needed a dispatch of its own. CO-9 REPRODUCED AT THE MECHANISM
  RATHER THAN READ, by driving entry_index directly on doctored copies of the
  real register: the real register returns OK with duplicate_refs {'R-6':
  [1788, 9514]}; fixture C -- an entry with TWO own blocks inserted EARLIER than
  R-419 -- REFUSES ("R-99900 carries 2 ratification blocks of its OWN"); fixture
  C2, the same entry with the duplicate R-6 heading renamed away, returns OK
  with duplicate_refs {}; fixture C3, the same malformed entry placed LATER,
  ALSO refuses, so THE SCAN IS POSITION-BLIND. C2 IS THE FINDING: whether a
  malformed entry refuses a check about a DIFFERENT entry depends on whether an
  UNRELATED duplicate exists elsewhere in the file -- R-6, a fact of this
  register since long before either -- and A REFUSAL WHOSE TRIGGER LIVES IN A
  THIRD ENTRY IS NOT A PROPERTY OF THE SUBJECT AT ALL. C3 shows why DE's
  ordering note could be true and still not cover this: the note is about ORDER
  and the scan does not consult order -- A CLAIM THAT NOTHING IS REFUSED EARLIER
  THAN BEFORE SAYS NOTHING ABOUT WHAT IS REFUSED AT ALL. THE (iii) REFINEMENT IS
  THE SAME MOVE AS (ii) AND I CHECKED BOTH FIXTURES: D, a duplicated heading
  whose second occurrence carries a QUOTED block, refuses at entry_index#3
  today; E, the same shape with an OWN block naming R-419, also refuses -- so
  the refinement's job is to SEPARATE them, D reaching no answer and belonging
  on the reporting side while E drops a real supersession under kept-first and
  must keep refusing. (ii) AND (iii) NOW SHARE ONE CRITERION -- OWNERSHIP AS THE
  MODULE DEFINES IT, NOT THE PRESENCE OF A FENCE -- recorded IN BAND beside the
  (ii) refinement in HANDOFF's standards section, WITH R-446 AND R-450 BOTH LEFT
  STANDING (rule 13). THE CLOSURE'S SHAPE IS WORTH AS MUCH AS THE RULE: the fix
  is a QUIET ownership filter for the two scans (the module's own predicate over
  _fenced_blocks, no adjudication) while own_ratification_blocks stays the
  ADJUDICATING reader ON THE PATH -- the distinction between READING TO DECIDE
  and READING TO SCAN, the round-26 fix having accidentally given the scan the
  decider's temperament. A READER THAT RAISES IS THE WRONG INSTRUMENT FOR A
  SURVEY. DA ROUNDS 13+14 REVIEW RELEASED (a2a1cf8, 278 lines) for BOTH e384792
  AND 801eb31 with identical dispositions and no hold, and its three findings
  reproduce at the held bytes: DA13-R1, the wiring poison is the LITERAL
  ("RULED at R-442",) at :628 asserted at :633 where every ruled entry carries
  the FORM "RULED at " (:399 and four more), so a legitimate re-ruling turns the
  control red -- DA10-R5 again, safe direction; DA14-R1, `_t2, _e2 =
  twins[:-1], excluded` then ok(len(_t2) + len(_e2) != len(roster)) is, given
  the invariant asserted two lines above, roster - 1 != roster, ARITHMETIC, true
  under every arrangement -- the DE16-R4 shape, a control that cannot fail;
  DA14-R2, the recomputed _exp_dirty is the right property but the fixture
  builds ONE arrangement so a producer hardcoding True still passes -- the
  round-15 candidate recorded at R-447 section 3, now a finding with the
  reviewer's ruling attached (the recomputed expectation is right AND NOT YET
  SUFFICIENT; drive both arrangements). MY FIGURE STANDS AND THE COORDINATOR'S
  IS WITHDRAWN: REVIEW_DE_ROUND_24 is 253 lines by wc -l and 253 insertions at
  304cd5f; R-450 section 2's 254 is withdrawn in band -- recorded because a
  withdrawn number that nobody records comes back. SEATS: DE round 27 (Q-DE-45)
  in flight with CO-9 and the (iii) refinement, C/C2/C3 and D/E as controls; DA
  ROUND 15 DISPATCHED AND HELD on the chain as its sixth commit, BEFORE tonight,
  selftest regions of three files only, no real-day run, shared tree and timers
  untouched; reviewer on DE round 25 (50a9113), then BE round 6 when its row
  lands, then BE round 7, then DA round 15 -- and DE ROUNDS 26+27 GO AS ONE
  FILING. USER decisions SIX RULED, NONE OPEN. UNCHANGED: G=1/5; the 011 family
  12 of 24 with Q4 failing; the sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 53 — archived 2026-09-02T17:12Z (1 entry, rolling-window overflow)

Moved in the MEM round-47 true-up of R-455. Join rule as in batch 1.

```yaml
  2026-09-02T16:16Z (MEM ROUND 44 -- THE SCAN STOPPED ADJUDICATING, AND THE
  READER THAT SHOULD STILL DOES). R-452 swept; five landings. CO-9 IS CLOSED at
  5e9dc8b and I RE-DROVE MY OWN FIXTURES at the tip rather than reading the
  result: the real register returns (duplicate_refs {'R-6': [1792, 9518]} at my
  as-of, the recount rule holding); C -- two own blocks EARLIER than R-419 --
  now RETURNS from entry_index (it refused at 89aef8c) AND superseded_by(R-419)
  returns []; C2 returns with {}; C3, the same entry placed LATER, RETURNS from
  entry_index but REFUSES in superseded_by; D (duplicate whose second occurrence
  carries a QUOTED block) RETURNS and is REPORTED; E (own block naming R-419)
  REFUSES. READ THE C3 ROW CAREFULLY BECAUSE IT IS THE WHOLE POINT: C3 refuses
  IN THE ADJUDICATING READER, NOT IN THE SCAN. own_blocks_quiet() (:631) is read
  by (ii) :423 and (iii) :446; own_ratification_blocks (:639) stays on the path
  -- SAME TEXT, TWO READERS, AND NOW THEY HAVE DIFFERENT JOBS. Ratification 177
  both launchers (EXPECTED_CHECKS = 177 at :1167); the #3 message says "carries
  an OWN ratification block" (:453). DE'S D2 IS THE ROUND'S BEST EVIDENCE AND IT
  IS NOT DE'S FIX BUT DE'S MUTANT: with only D, dropping the `kind` conjunct
  from the quiet filter PASSED, and D2 exists because the mutant was actually
  run -- A FIXTURE SET IS SIZED BY THE MUTANTS YOU RUN AGAINST IT, NOT BY THE
  CASES YOU THOUGHT OF. The reviewer has the reverse-direction question (the
  predicate now exists twice, :631 and :639) as item 1. DE ROUND 25 REVIEW
  RELEASED (a7860dc) for 50a9113 with three rulings adopted, and DE25-R1
  REPRODUCED TO THE DIGIT: a line at column 0 at the end of the module docstring
  -- an anchor collision -- makes declared_limit_text return 0 CHARS with
  stopped_at_a_real_boundary TRUE, first_read_line 53, above_line 52, and ALL
  THREE ANCHORS ABSENT; the suite is red because the anchors check fires, so it
  is a COMPLETENESS POINT ABOUT THE STATED LIMIT, NOT AN EXPOSURE -- and worse
  IN KIND than X = 1: A READ OF NOTHING THAT ANSWERS TRUE. DE round 28 (Q-DE-46)
  names the shape and the composition's own condition (the anchors cover code
  cuts only while they remain the block's topmost content). DA ROUND 15 VERIFIED
  at the chain's 8910701 (39 / 5 / 32 both launchers): DA13-R1 closed with the
  FORM at :635/:640, DA14-R1 closed BY DELETION with the reason in the comment
  :236-252, DA14-R2 closed with BOTH arrangements :849-898 -- and DA's own first
  clean arrangement deserves billing: NOT COPYING THE FILES RAN THE CHILD'S
  COMMITTED CODE, so a parent-side mutation never reached it and a hardcoded
  True SURVIVED; copy AND commit in the scratch child is the difference between a
  fixture that LOOKS isolated and one that IS. CO-10 IS THE DA10-R5 CLASS IN ITS
  MOST EXPENSIVE FORM AND I VERIFIED BOTH ENDS: at 801eb31 the carrying-commit
  control asserted IDENTITY (carrying_commit == _there, :847, the child's HEAD);
  the round-15 fixture commits in the child and moves that HEAD, so the check was
  rewritten (:902-903) as != _here and != _root_git.stdout.strip() -- AND _here
  IS _root_git.stdout.strip() (:802), two negatives of one value. The first three
  instances ENCODED an arrangement; this one encoded one, had it INVALIDATED by a
  fixture change, and was REPAIRED INTO A TAUTOLOGY. Recorded beside DA10-R5 in
  the generalising form: WHEN A FIXTURE CHANGE BREAKS A CONTROL, RE-DERIVE THE
  PROPERTY -- DO NOT WEAKEN THE ASSERTION UNTIL IT FITS. Dispatched as DA round
  16 (Q-DA-212), held on the chain, before tonight. BE ROUND 6 VERIFIED at
  5e9ed91: driver sha 957a9d3cc38b3dde, 95 both launchers, usage rc 2, all
  reproduced here; BE34-R1/R3/R4/R5 closed and BE34-R1's closure is the one to
  keep -- one fixture through both consumers, scores EQUAL per coin, the same
  featureless row dropped, and the scores DISTINCT AND SMALL so a 1e9-scale value
  cannot hide a 1e-9 perturbation. BE corrected three things of its own IN BAND,
  including naming the in-place row rewrite as its fault, and SETTLED THE 5-VS-4
  I NARROWED TWO ROUNDS AGO: two passes, mr5 13/4 and mall 50 with SIX survivor
  lines but FIVE distinct -- H14 listed twice by the harness. The number was
  never wrong about the code; the harness printed one mutant twice. ONE
  DISCLOSURE OF THE COORDINATOR'S KEPT VISIBLE because it is the kind of thing
  that normally goes unrecorded: a compound command carried its cd into DA's
  worktree and ran git pull --ff-only there; it ABORTED and nothing moved,
  verified from outside -- THE ABORTED FORM IS THE ONLY REASON NOTHING MOVED.
  SEATS: DE round 28, DA round 16 (held) and BE round 7 (Q-BE-232, BE5-R1/R2/R3,
  no real-day run) in flight; reviewer takes DE ROUNDS 26+27 AS ONE FILING, then
  DA round 15, then BE round 6; three requests filed. Tonight's timers unchanged.
  USER decisions SIX RULED, NONE OPEN. UNCHANGED: G=1/5; the 011 family 12 of 24
  with Q4 failing; the sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 54 — archived 2026-09-02T17:18Z (1 entry, rolling-window overflow)

Moved in the MEM round-48 true-up of R-456 and R-457. Join rule as in batch 1.

```yaml
  2026-09-02T16:28Z (MEM ROUND 45 -- THE SUITE'S LAST WORD WAS "PASS" AND THE
  PROCESS EXITED 1). R-453 swept; four landings. DE ROUND 28 VERIFIED at 1480ab6
  and RE-RUN HERE on the same collision fixture I built two rounds ago: 92
  checks rc 0; intact 3,752 chars / first_read_line 139 / above_line 137 /
  read_nothing False / boundary True; THE COLLISION now reads 0 CHARS with
  read_nothing TRUE and boundary FALSE where the same fixture answered TRUE at
  50a9113 -- DE25-R1 CLOSED, and the closure is the honest one: read_nothing =
  first == i (:319) folded into the predicate so A READ OF NOTHING CAN NO LONGER
  ANSWER TRUE. Growth control boundary True at 3,805 chars in my copy against
  3,798 in the coordinator's -- the THIRD independent number for that case and
  the third time the effect is invariant while the number is not. DE'S OWN
  LAYERED MUTANT IS WORTH MORE THAN ITS FIX: it found a ZeroDivisionError in the
  anchors known-bad's MESSAGE when _limit is empty -- A MESSAGE THAT CRASHES
  BEFORE IT CAN NAME THE FAILURE, DE23-R2's family, the assertion right and the
  reporting not, guarded at the right size ('?', not a rewrite). DE ROUNDS 26+27
  REVIEW RELEASED (723271e, 218 lines) for BOTH 5e9dc8b and 89aef8c, no hold,
  the reviewer's eight-fixture matrix reproducing the register's table exactly
  and D2's discovery reproduced GREEN at 176. DE27-R1 REPRODUCED AT HEAD IN A
  PARITY TREE, AND THE REPRODUCTION IS THE FINDING: the ownership predicate is
  written twice (own_blocks_quiet :631, own_ratification_blocks :639); three of
  the four conjunct drops die at a control AIMED AT THAT CONJUNCT; the fourth,
  the adjudicating `kind` drop, exits RC 1 WITH AN UNCAUGHT RatificationRefused
  ("REFUSED FOR A NEW RUN: R-419 is SUPERSEDED by R-999") -- a traceback on
  stderr from INSIDE A POSITIVE CONTROL. AND THE PART THAT MAKES IT WORTH A
  ROUND: with that mutant in place THE LAST LINE ON STDOUT IS "PASS", the
  verdict living only in the exit code and on stderr, so A SEAT TAILING STDOUT
  SEES A SUITE THAT PASSED. The finding is not that the mutant survives -- it
  dies loudly -- it is that RED IS NOT THE SAME AS CAUGHT: the failure has no
  name, no site and no line in the transcript a reader will look at. A MUTANT
  THAT KILLS THE PROCESS IS NOT EVIDENCE THAT A CONTROL EXISTS. DE round 29
  (Q-DE-47) is aimed at exactly that gap: ONE TEXT (the quiet filter returns the
  (blk, dups) pairs and the adjudicating reader consumes it, adding only its two
  raises), a NAMED control for the `kind` conjunct, the AST-census one-place
  assertion if DE takes it, mutants re-driven, 177 -> N. THE CAPTURE NOTE
  BELONGS IN EVERY SEAT'S HABITS AND THIS ROUND SHOWS WHY: da_blackout_mask's
  FAIL line goes to STDOUT while de_admissible_windows' refusal is a SystemExit
  on STDERR -- capture both streams TO SEPARATE FILES, since a single merged
  capture would have shown a PASS-terminated log for a failing run. CO-10 CLOSED
  ON THE CHAIN at 3b7e10a pending the reviewer, and I read the closure rather
  than the claim: _child_head re-read from the child AFTER the fixture commit
  (:885-887), a precondition asserting it is a THIRD value distinct from both
  (:888), the control asserting carrying_commit == _child_head AND != _here
  (:927-928) -- THE IDENTITY CONJUNCT IS BACK -- and != _there standing
  SEPARATELY (:944), so dropping the identity alone leaves != _here satisfied by
  an intact producer and BOTH LINES MUST GO for the hole to reopen; the HEAD~1
  producer mutant is RED BY NAME at the CO-10 CONTROL; mask 32 -> 34, gates 5,
  preflight 39, the redundant third run gone with one execution feeding the
  assertions. THE ADDENDUM'S QUESTION IS THE RIGHT ONE TO LEAVE OPEN: identity
  plus the precondition already implies both negatives, so is the separate !=
  _there line a control WITH ITS OWN FALSIFIER or belt-and-braces carrying a
  stale count in its message -- items 7-10, and the reviewer takes ROUNDS 15+16
  AS ONE FILING at 3b7e10a. TONIGHT UNCHANGED: the 00:06Z verdict timer, the
  00:14Z preflight timer, the coordinator's wake after it, R-409 with the
  R-411(ii) denominator, then DA lands Q-DA-209..212 with the chain's tip, BE's
  durable landing, then CO-8, --require-no-skips and the DATA_ROOT split. SEATS:
  DE round 29 and BE round 7 in flight; DA on DELIBERATE standby (R-381), a
  further finding at the tip re-opening as round 17. USER decisions SIX RULED,
  NONE OPEN. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 55 — archived 2026-09-02T18:00Z (1 entry, rolling-window overflow)

Moved in the MEM round-49 true-up of R-458, R-459 and R-460. Join rule as in batch 1.

```yaml
  2026-09-02T16:38Z (MEM ROUND 46 -- THE STANDARD EARNED ITS KEEP IN ONE ROUND).
  R-454 swept; three landings. DE ROUND 29 VERIFIED at ea3b525 and RE-RUN HERE:
  180 both launchers rc 0, census ok 93 -> 96, refuses 54. The closure is real --
  own_blocks_quiet (:631-648) returns (block, duplicated keys) PAIRS and is the
  ONE TEXT of the two conjuncts, own_ratification_blocks (:650-676) CONSUMES it
  at :658 and adds only its two raises -- and the thing DE27-R1 was actually
  about is fixed: a NAMED CONTROL (:2074-2107) now sits on the `kind` conjunct so
  the drop is CAUGHT AND REPORTED rather than escaping as a traceback; four
  mutants red by name, ZERO TRACEBACKS. CO-11 REPRODUCED HERE IN A PARITY TREE
  AND IT REPRODUCES THE EXACT SHAPE I RECORDED LAST ROUND: _ownership_sites keys
  the census on the VARIABLE NAME (getattr(n.func.value, "id", "") == "blk",
  :2130), so pasting the filter back into the adjudicating reader with the loop
  variable RENAMED (blk -> b) -- semantically the second text DE27-R1 removed --
  leaves ONE OWNERSHIP TEXT saying PASS at stdout line 124, WHICH WAS ALSO THE
  LAST LINE OF STDOUT. THE MESSAGE CLAIMS THE PREDICATE; THE CHECK ASSERTS THE
  IDIOM. The secondary half fell out of the same run and I verified its cause
  statically: the text `own = own_blocks_quiet(entry)` occurs TWICE in the module
  -- the code at :658 and a STRING LITERAL at :2146 inside the census known-bad's
  own .replace -- so with the code line renamed away the replace hits the
  literal, the copy is left with an unterminated string, and the run dies
  SyntaxError: unterminated string literal (detected at line 2148) ON STDERR: A
  TRACEBACK WHERE A REFUSAL BY NAME BELONGS. SO: LAST STDOUT LINE "PASS", EXIT 1,
  ONE ROUND AFTER THAT OBSERVATION WAS ADOPTED AS THE STANDARD AND INSIDE THE FIX
  FOR THE FINDING IT WAS ADOPTED FOR. That is the argument for the standard being
  a HABIT rather than a rule you remember when relevant: EVERY GUARD ADDED TO
  CATCH A CLASS IS ITSELF A CANDIDATE FOR THAT CLASS, and the only routine that
  catches it is capture both streams separately, read the exit code, and never
  take the last line of stdout for the verdict. THE GENERAL LESSON BENEATH CO-11
  IS NARROWER AND MORE USEFUL THAN "the census is weak": a drift guard's
  known-bad exercised EXACTLY THE IDIOM THE GUARD KEYS ON, so the falsifier could
  not fail by any other spelling -- RULE 15'S KNOWN-BAD NARROWER THAN THE CLAIM
  IT IS DEFENDING. Round 30 (Q-DE-48) is the right shape: key on the CONSTANT AND
  THE SHAPE, drive the known-bad under BOTH the same idiom and a renamed copy,
  and assert the anchor so an absent line REFUSES BY NAME instead of crashing the
  parser. DA ROUNDS 15+16 REVIEW RELEASED (5d9bfb8, 219 lines) for 3b7e10a, no
  hold, CO-10 CONFIRMED CLOSED by the 2x2+1, and three carried facts of mine
  RULED: the separate != _there line IS a control with its own falsifier (cell
  4); "32 checks at 8910701" is HISTORY CORRECTLY SCOPED, NOT A STALE COUNT; and
  DA14-R1's deletion-over-a-hook is the honest call with "a tripwire on a future
  edit" the honest label. DA16-R1 IS THE SAME SPECIES AS CO-10 ONE LAYER UP AND
  THE LOGIC IS CHECKABLE BY READING: with the identity conjunct dropped what
  remains is carrying_commit != _here (:927-928) and != _there (:944), so a
  producer answering HEAD~2 yields a FOURTH value -- neither -- BOTH NEGATIVES
  HOLD AND THE SUITE GOES GREEN AT 34; the pair is jointly sufficient for THE
  VALUE CO-10 WAS FOUND ON, not for THE PROPERTY THE CONTROL CLAIMS. I DID NOT
  RUN THIS ONE: it needs a worktree on DA's chain and git worktree list reads 33
  here as it should -- verified by reading, and said so. THE ROUTING IS THE PART
  WORTH KEEPING: the closure is DRIVEN, NOT A FOURTH != -- another negative just
  extends an enumerated set by one value and leaves the identity drop invisible
  -- so state the control's predicate ONCE AS A FUNCTION OF THE EMISSION, run a
  wrong-commit producer copy through _measure in the child and assert it REFUSED
  by that predicate, so dropping the identity turns THAT FALSIFIER red; != _there
  kept as ruled; a fourth-distinct-value precondition beside the third. THE
  DA14-R2 SHAPE APPLIED TO A CONJUNCT: GIVE IT A DRIVER, NOT A COMPANION.
  TONIGHT'S LANDING RULE AS RULED: the landing takes THE CHAIN'S TIP OF THE
  MOMENT -- 3b7e10a is RELEASED, and a round-17 commit above it is SELFTEST-ONLY
  and lands ONLY IF coordinator-verified at the artifact before the landing
  dispatch, else the landing is at 3b7e10a; timers unchanged. The reviewer's pane
  showed context at 100% after its filing; nothing was done about it and nothing
  needs to be -- THE REQUESTS OF RECORD CARRY THE STATE, which is why they are
  files rather than messages. SEATS: DE round 30, DA round 17 (held) and BE round
  7 in flight; reviewer BE round 6 (5e9ed91) now, then DE round 29 (ea3b525),
  then BE round 7, then DA round 17. USER decisions SIX RULED, NONE OPEN.
  UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule;
  Phase-4 gated; R-419 revocable.
```

## Batch 56 — archived 2026-09-02T18:28Z (1 entry, rolling-window overflow)

Moved in the MEM round-50 true-up of R-461, R-462 and R-463. Join rule as in batch 1.

```yaml
  2026-09-02T17:12Z (MEM ROUND 47 -- I MEASURED THE WRONG FILE, AND THE ROUND
  BEFORE LAST IS WHY I NOTICED). R-455 swept; four landings, two new findings at
  the BE tip, and one failure of my own worth more than the checks that went
  right. THE FAILURE FIRST: I set out to verify CO-13 by counting PASS lines
  against the printed total; HEAD's COMMITTED be_forward_day.py hashes to
  857819a76ca5c3a9, exactly fcafe9f -- I checked that -- but the file I RAN was
  the WORKING TREE'S, and by the time the run started the tree was DIRTY WITH
  BE'S UNCOMMITTED ROUND-8 WIP (ab65b026e3093cad). THE TREE MOVED BETWEEN TWO OF
  MY OWN READS INSIDE ONE ROUND, so my partial transcript (103 PASS lines) is a
  count of a file nobody has committed: it corroborates nothing AND I WITHDRAW
  IT. I stopped the run; derived/ reads 173 before and after. THAT IS ROUND 40'S
  FINDING TURNED ON ITS AUTHOR -- I wrote then that no count taken from that file
  is a committed figure, and then took one -- and the fix is mechanical: HASH THE
  FILE IMMEDIATELY BEFORE THE RUN AND AGAIN AFTER, since a check at the top of
  the round is a check of the wrong moment. CO-13 IS THEREFORE CARRIED AS THE
  COORDINATOR'S MEASUREMENT, WITH ITS STATIC HALF VERIFIED BY ME AT fcafe9f: the
  BE5-R3 block calls ok(...), which increments checks, and is IMMEDIATELY
  FOLLOWED BY A BARE checks += 1 -- ONE ASSERTION, TWO COUNTS -- so the printed
  total is one ahead of the assertions that ran, which is exactly 101 versus
  "102 checks OK". Read, not run, and said so. CO-12 IS THE SHARPER OF THE TWO
  AND IT IS THE NEW STANDARD'S SECOND INSTANCE: the audit's attribution is
  at_named = want in out over stdout + stderr (:1566) while ok prints
  "  PASS  {label}" for every check that passes (:1591) and raises
  AssertionError(label) when one fails (:1589), so for any case whose `want` is a
  prefix of its own check's label -- 7 OF THE 10 -- the predicate is satisfied by
  THE GREEN BASELINE TRANSCRIPT: it tests that the named check RAN, not that the
  mutant DIED THERE. The ten mutants do die; THE ATTRIBUTION THE ROW RESTS ON IS
  WHAT HAS NO FALSIFIER. Closure right: attribute on the AssertionError line on
  STDERR, and ship a control BOTH DIRECTIONS (a mis-named case asserted a
  SURVIVOR, the same edit correctly named asserted KILLED). AND THE STANDARD IS
  NOW THREE-FOR-THREE: every guard added to catch a class is itself a candidate
  for that class -- CO-11 lived in the census guarding DE27-R1's fix, CO-12 lives
  in the audit shipped to close BE5-R3 -- so I have ADOPTED IT INTO
  standing_rules (7 -> 8) and into HANDOFF's standards section in the
  operational form: WHEN A FIX ADDS A CONTROL, MUTATE THE CONTROL. DE ROUND 30
  RE-RUN HERE: 183 both launchers, EXPECTED_CHECKS = 183, census ok 97, CO-11
  CLOSED; the observation the coordinator did NOT file (a name-bound
  comprehension spelling still passes) is right as an ITEM rather than a finding
  -- a drift guard whose message names its key owes no dataflow census, and that
  message now names the key exactly. DA ROUND 17 at e353119 READ AT THE ARTIFACT:
  da_blackout_mask.py only, +57/-2, and the shape is precisely what was routed --
  _names_the_executing_tree defined ONCE (:892) with TWO call sites, the CO-10
  CONTROL (:951) and at :1023 the same predicate under `not` as the DA16-R1
  FALSIFIER: THE CONJUNCT GOT A DRIVER, NOT A COMPANION; mask 38; and UNDER
  R-454 SECTION 4 THIS IS THE TIP THAT LANDS TONIGHT, a HOLD falling back to
  3b7e10a. ONE THING OF MINE TO CORRECT RATHER THAN EXPLAIN AWAY: Q-BE-232 landed
  inside MY commit 1b874f9 -- I checked, and that commit added TWO rows, mine and
  BE's -- because A PATHSPEC PROTECTS AGAINST UNRELATED FILES, NOT AGAINST
  ANOTHER SEAT'S UNCOMMITTED EDITS IN THE SAME FILE; content intact, nothing to
  correct, and the discipline is mine to fix: BEFORE COMMITTING A SHARED
  APPEND-ONLY FILE, VERIFY THE STAGED DIFF CONTAINS ONLY MY OWN ROW (done this
  round). BE round 8 (Q-BE-233) dispatched with CO-12 and CO-13; the durable
  landing is BE ROUND 9, after the 00:14Z read. Reviewer queue: BE r6 (in flight)
  -> DE 29+30 as one filing at 27d0d37 -> BE 7(+8) -> DA 17 at e353119; three
  requests filed. DE and DA on DELIBERATE standby (R-381). TONIGHT UNCHANGED:
  00:06Z verdict, 00:14Z preflight, the coordinator's wake after it, then DA's
  landing at the chain's tip of the moment, BE round 9, CO-8,
  --require-no-skips, the DATA_ROOT split. USER decisions SIX RULED, NONE OPEN.
  UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule;
  Phase-4 gated; R-419 revocable.
```

## Batch 57 — archived 2026-09-02T18:43Z (1 entry, rolling-window overflow)

Moved in the MEM round-51 true-up of R-464. Join rule as in batch 1.

```yaml
  2026-09-02T17:18Z (MEM ROUND 48 -- SEVEN FINDINGS, AND A RULE MADE OUT OF MY
  OWN MISTAKE). R-456 and R-457 swept in one batch. NOTHING WAS RUN FROM THE MAIN
  TREE THIS ROUND: BE's round-8 WIP is sitting in live/pm_research/, so
  everything was read from COMMITTED BLOBS (git show <tip>:<path>) or from the
  safety copy of the race record -- the new rule applied to the round that
  adopted it. MY CALL AS ASKED: the shared-tree class is a NINTH STANDING RULE,
  NOT A CLAUSE UNDER THE EIGHTH -- the eighth is about CONTROLS (a guard that
  needs mutating), this is about WHERE YOU ARE STANDING WHEN YOU MEASURE, and its
  natural neighbour is R-446's pre-run cache clear; both are measurement hygiene
  and both were bought with a wrong reading. It is in standing_rules (8 -> 9) and
  in HANDOFF's standards section with the non-obvious part kept: HASHING
  IMMEDIATELY BEFORE AND AFTER THE RUN IS NECESSARY BUT NOT SUFFICIENT, BECAUSE
  THE WIP CAN BE PRESENT AT BOTH HASHES. THE REVIEWER'S BE ROUND 6 FILING
  (03b5dca, 273 lines, committed 17:07:24Z) VERIFIED: RELEASE for 5e9ed91, seven
  findings BE6-R1..R7. TWO OF THEM I VERIFIED AT THE BLOBS: BE6-R1 --
  rec["refused_at"] = rec["gates"][-1]["gate"] if rec["gates"] else None is
  present at 5e9ed91:1220 and UNCHANGED at fcafe9f:1246, and since a passing gate
  appends {"gate": ..., "result": "PASS"}, a BARE RAISE leaves gates[-1] naming
  THE LAST GATE THAT PASSED, so the receipt attributes a refusal to a check that
  SUCCEEDED; BE6-R2 -- _launch_parity is `return rc == 0 and child == expect`, A
  COUNT, so a byte-different tree with the same number of checks passes it. ONE
  CITATION NEEDS FIXING AND I SAY SO RATHER THAN QUIETLY USING THE RIGHT ONE:
  BE6-R2's second citation fcafe9f:2588-2590 DOES NOT RESOLVE -- the file at
  fcafe9f is 2,580 lines -- the code being at :2481 (definition) and :2483 (the
  rc-and-count line) with the paired ok at :2532-2534 where at_entry is compared,
  the shape the ruling says stands. SAME CLASS AS THE ROUND-18 DIRTY-TREE LINE
  NUMBERS: A CITATION THAT CARRIES A COMMIT MUST RESOLVE AT THAT COMMIT. BE6-R7
  CORROBORATED FROM THE RECEIPT OF RECORD, WHICH MAKES THE FINDING SHARPER:
  coin_coverage carries coins_supplied = 7 and coins_supplied_without_a_fit = 5
  (bnb, doge, hype, sol, xrp), so the no-fit class is THE MAJORITY OF THE DAY and
  THE DRIVER ALREADY NAMES IT IN ITS OWN RECEIPT while the
  one-fixture-two-consumers check omits it -- a class the artifact reports and
  the fixture does not exercise is a gap with its own evidence attached. THE
  SECOND ADOPTED RULING IS THE ONE WORTH CARRYING FORWARD: the shipped audit must
  COMPUTE verdict counts from verdict-initial lines, NEVER GREP VOCABULARY (rule
  10 at the harness level), folded into CO-12's closure -- and it is the same
  defect CO-12 names from the other side, CO-12 being attribution by substring
  and this counting by substring, both a text search standing in for a
  computation. SEQUENCING STATED SO NOTHING SLIDES: BE6-R1..R7 are THE BE ROUND
  AFTER 8 with BE6-R2 AND BE6-R1 FIRST; THE DURABLE LANDING IS UNAFFECTED BY THE
  SEVEN and stays the round after the 00:14Z read; the reviewer is on DE rounds
  29+30 at 27d0d37. AND A CORRECTION OF RECORD CARRIED BECAUSE CORRECTIONS NOBODY
  CARRIES ARE HOW A STALE NUMBER SURVIVES: R-456's "17:10Z" and "17:12Z" were
  composed ahead of the clock; THE TIMES OF RECORD ARE 17:08-17:09Z FOR BOTH
  EVENTS (R-457 section 3, in band), R-456 standing as provenance -- and the
  reviewer's filing commit reads 17:07:24Z, which I checked, consistent with the
  corrected window and not with the original. USER decisions SIX RULED, NONE
  OPEN. Tonight unchanged. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4
  failing; the sequencing rule; Phase-4 gated; R-419 revocable.
```

## Batch 58 — archived 2026-09-02T18:50Z (1 entry, rolling-window overflow)

Moved in the MEM round-52 true-up of R-465. Join rule as in batch 1.

```yaml
  2026-09-02T18:00Z (MEM ROUND 49 -- A SEVENTH RULING, AND A CONSTANT THAT WAS
  FIXED AT ONE CALL SITE). R-458, R-459 and R-460 swept in one batch. NOTHING WAS
  RUN: BE round 9 is in flight so the main tree's live/pm_research/ is BE's
  working surface again, and every check was a COMMITTED BLOB or a PLAN FILE --
  the ninth standing rule's second round of use. THE RULING IS THE ROUND: "Yes
  schedule this test", the SEVENTH USER decision. I checked what it lifts at the
  artifact rather than from the entry: DE_PHASE4_PROTOCOL_DRAFT.md:3 reads FROZEN
  -- IN FORCE and :13-17 carries the gate in its own words ("it may not be run
  until the hold is lifted by the USER") -- THAT IS THE SENTENCE THIS RULING
  ANSWERS, AND IT ANSWERS IT ONCE. The protocol's sha is ab07fd71c9fc2bff and
  DE's addendum (DE_PHASE4_DIAGNOSTIC_ADDENDUM_2026-09-02.md, 35e8aba1381cfa4e)
  BINDS IT BY THAT SHA and names R-459, so THE THING BEING RUN IS IDENTIFIED BY
  CONTENT, NOT BY TITLE. WHAT THE RULING DOES NOT DO IS THE PART WORTH KEEPING
  LEGIBLE: the population is the protocol's own section 3 fragment and :80-84
  already says what it is -- CONSUMED, is_a_validation = false, G = 0, no
  interval claimable, no forward verdict -- so the run CANNOT BECOME EVIDENCE BY
  ACCIDENT, the frozen document refusing that on its own terms before any receipt
  field does; latency is swept because :101 says it is NOT a selection axis, and
  budget is reported in all three rungs with NONE SELECTED because it IS one.
  ONE ITEM 4 CHANGE, RECORDED BESIDE THE ITEM AND NOT INSTEAD OF IT: the
  Immediate-order item STANDS and its EXECUTION HOLD IS LIFTED FOR THAT EXECUTION
  ONLY; no PnL, capacity, promotion or forward verdict is claimable, INCLUDING
  FOR THAT RUN, every output carrying DIAGNOSTIC_NEVER_EVIDENCE. AND THE
  DECISIONS SECTION STOPS SAYING "NOTHING IS AWAITING THE USER", which it has
  said since R-442: per R-460 TWO decisions are OPEN AND UNBLOCKED -- the Phase-2
  winner ruling and the content-liveness v2 freeze -- NEITHER NEEDED TONIGHT;
  superseded IN BAND rather than edited away, because a table that once said
  "none open" is exactly what a later reader trusts without re-checking. A
  CITATION TO FIX, THE THIRD IN THREE ROUNDS: R-459 cites the budget axis at :105
  but :105 is the PROTECTION MODE row -- THE BUDGET ROW IS :103 ("| budget b |
  5%, 10%, 15% | YES -- someone chooses a budget |"); the substance is exactly
  right and the pointer is two lines off, and three drifted citations in three
  rounds (fcafe9f:2588-2590 out of range, :405-406 vs :404-405, now :105 vs :103)
  is not luck but WHAT HAPPENS WHEN A LINE NUMBER IS TYPED FROM A READING RATHER
  THAN RE-DERIVED AT COMPOSE TIME. R-459 stands as provenance. BE7-R4 VERIFIED AT
  THE BLOB, AND IT HAS A ROOT I CAN NAME: _provenance (fcafe9f:70) runs git
  rev-parse HEAD and git status --porcelain with cwd=str(REPO), and REPO is STILL
  Path("/home/yuqing/ctaNew") HARDCODED at c54e48e:37, so a driver executing in
  ANY worktree records THE MAIN TREE'S commit and dirtiness -- THE SAME CONSTANT
  BE34-R3 WAS ABOUT, round 6 having fixed it AT THE SPAWN SITE ONLY (:2611 now
  uses Path(__file__).resolve().parents[2]) and left the constant and its other
  readers in place, with an audit case named "spawn root REPO vs parents[2]"
  (:1401) so THE FIX AND ITS COVERAGE ARE BOTH SCOPED TO THE ONE CALL SITE.
  FIXING A USE OF A BAD CONSTANT IS NOT FIXING THE CONSTANT. AND ONE OF THE TWO
  REMAINING READERS IS BENIGN, WHICH MATTERS: the frozen-blob reader at :103 also
  uses cwd=REPO but runs git show <sha>:<path>, and worktrees SHARE THE OBJECT
  DATABASE, so those bytes are identical from anywhere -- :79 reads PER-WORKTREE
  STATE and is the finding, :103 reads OBJECT-STORE CONTENT and is not; two
  identical spellings, one defect. CARRIED WITHOUT INDEPENDENT MEASUREMENT AND
  MARKED AS SUCH: BE round 8's 106/106 and DE round 31's instrument counts (21 /
  24 / 20, ratification 184, phase-4 check 15) are THE COORDINATOR'S COUNTS --
  running them would mean the main tree (forbidden while BE round 9 is open) or a
  worktree of my own, and git worktree list stays 33; structure verified, numbers
  carried. ONE LAYOUT FACT FROM DE ROUND 31 THAT WILL BITE SOMEONE AT 2 A.M.: a
  BARE DETACHED WORKTREE HAS NO data/, so the runner must mirror data/pm_5min per
  entry (derived/ 173) or THE DRIVER REFUSES AT CHECK 24 BY DESIGN -- that
  refusal is right and the note is what stops it being read as a break.
  SEQUENCING FOR THE DIAGNOSTIC, DELIBERATELY SLOW: DE declares (done, r31) ->
  reviewer reads the declaration -> DE builds the runner (r32, in flight) ->
  reviewer reads -> run (NO DATE; see below) -> coordinator entry -> USER: FOUR GATES
  BETWEEN A RULING AND A NUMBER, which is the point of freezing the protocol
  first. Also swept: R-458's DE 29+30 RELEASE for 27d0d37 with CO-11 CLOSED at
  both tips and DE30-R1 since closed in r31, and its IN-BAND CORRECTION of
  R-456 section 2's citation -- fcafe9f:2588-2590 does not resolve, the citation
  of record being :2481/:2483 with the paired ok at :2532-2534, R-456 standing as
  provenance. Filings of record: Q-BE-233, Q-DE-49, REVIEW_BE_ROUND_7 (0f34aad).
  IN FLIGHT: BE r9 (Q-BE-234: BE7-R4 first, then BE6-R1..R7 and BE7-R1..R3),
  reviewer on DA 17 at e353119 then DE 31, DE r32 (Q-DE-50, the runner, no run),
  DA standby (R-381) until the 00:14Z read. TONIGHT UNCHANGED:
  da-midnight-verify.timer 00:06Z (09-03), co-preflight-20260902.timer 00:14Z,
  coordinator wake after; R-409 accrual with the R-411(ii) denominator; DA lands
  Q-DA-209..213 at e353119 (HOLD -> 3b7e10a); BE durable landing the round after
  the read; CO-8; --require-no-skips; DATA_ROOT split after DA's landing.
  UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule;
  R-419 revocable.
```

## Batch 59 — archived 2026-09-02T19:05Z (1 entry, rolling-window overflow)

Moved in the MEM round-53 true-up of R-466, R-467 and R-468. Join rule as in batch 1.

```yaml
  2026-09-02T18:28Z (MEM ROUND 50 -- THE CAP IS RECORDED, ASSERTED AGAINST
  ITSELF, AND NEVER ENFORCED). R-461, R-462 and R-463 swept in one batch. Nothing
  run and nothing read from a forbidden surface: BE round 9 is open so
  be_forward_day.py went untouched, DE round 33 is in flight so ~/ctaNew-wt-de
  went unread and only its CODE TIP 6d04833 (18:20:40Z) is recorded; everything
  else is a committed blob or a plan file. ITEM 8'S RULING IS THE ROUND AND BOTH
  HALVES VERIFY: the one-second horizon is not an addendum choice but THE FROZEN
  PROTOCOL'S OWN CAP 2 (DE_PHASE4_PROTOCOL_DRAFT.md:41-45) -- FILL_HORIZON_S =
  1.0 s, every cell meaning "value preventable WITHIN ONE SECOND of the decision
  row", and phase4_generation_tables.tranche_table REFUSING TO EMIT WITHOUT
  declare_cap=True -- while THE ADDENDUM DOES NOT CARRY IT (grepped in five
  spellings, ZERO hits). So the document that BINDS THE PROTOCOL BY SHA omits the
  protocol's own semantic cap: BINDING BY CONTENT PROVES WHICH DOCUMENT, IT DOES
  NOT CARRY THAT DOCUMENT'S OBLIGATIONS INTO YOURS. AND THE RUNNER'S TREATMENT OF
  THE CAP IS THE SHARPER HALF: at e52d183 the cap is IMPORTED (:61), RECORDED in
  the receipt (:205), EXPLAINED in prose (:207) and CHECKED at :438-440 by
  asserting rec["fill_horizon_s"] == FILL_HORIZON_S and that "WITHIN ONE SECOND"
  appears in the note THE SAME CODE WROTE -- BOTH SIDES OF THAT CHECK COME FROM
  ONE SOURCE -- while tranche_table occurs EXACTLY ONCE IN THE FILE, ON LINE 26,
  INSIDE THE DOCSTRING, AND IS NEVER CALLED. The protocol's actual enforcement,
  the refusal, is NAMED IN PROSE AND NEVER INVOKED, and what runs is a field
  compared to the constant it was copied from: A CHECK WHOSE TWO SIDES SHARE AN
  ORIGIN IS A SPELLING TEST. That is why "verified by count and FOUND SHORT" is
  the right verdict on round 32 and why the review's separation matters: e52d183
  is released as DECLARATION + THREE INSTRUMENTS and THE RUNNER IS NOT RELEASED
  AS A PRODUCER; Q-DE-50 BUILT THE RUNNER'S SHELL, not "the runner", and the
  review table says it that way because the other phrasing would read as a
  producer existing. FINDINGS WITH ROUTING KEPT HONEST ABOUT WHAT IS AND IS NOT
  DISPATCHED: DE32-R1 and DE32-R2 (MEDIUM) stay OPEN pending the round-33 landing
  check, remainder to round 34; DE31-R1 and DE32-R3 are NOT IN THE ROUND-33
  DISPATCH -- round 34 unless the tip carries them, which the LANDING CHECK
  decides, not the dispatch; DE32-R4 / DE31-R2 / DE32-R5 (LOW) likewise; DA17-R1
  (LOW) sits behind DA's landing. RECORDING THE ABSENCE FROM A DISPATCH IS THE
  PART THAT USUALLY GOES UNWRITTEN AND IS EXACTLY HOW A FINDING QUIETLY
  DISAPPEARS. ONE TRUE-UP BEYOND THE DISPATCH: it says Q-DE-51 is PENDING; the
  row LANDED AT 2b72d02 while this batch was composed, and I record it as FILED,
  NOT COORDINATOR-VERIFIED -- same treatment as DE round 21 in round 33, because
  a row landing is not a verification. THE RUN'S PRECONDITIONS ARE NOW WRITTEN
  BESIDE THE RULING where a reader meets them: the round AFTER DE round 33 lands
  AND the reviewer reads it, EARLIEST 09-03, after tonight's landings, by
  coordinator dispatch under systemd-run --user --scope --slice=research.slice -p
  MemoryMax=12G, INTO THE DECLARED OUTDIR ONLY -- four gates, with the protocol
  frozen before any of them, which is the whole reason a diagnostic can be
  scheduled without becoming evidence. R-461'S CORRECTION IS CARRIED AND IT WAS
  MINE: R-459 section 2's budget-axis citation :105 -> :103, adopted in band with
  R-459 standing as provenance; and BE7-R4's root -- REPO hardcoded at
  c54e48e:37 -- is carried into BE ROUND 9'S LANDING CHECK, the right home for
  it, since the finding was about one reader, the root is about the constant, and
  a landing check is where a root gets tested rather than a symptom. Reviewer's
  DA 17 filing f4c7734 (145 lines) RELEASE for e353119; DE 31+32 filing 82126ca
  (251 lines); reviewer now on BE ROUND 8 at c54e48e per
  REQUEST_BE_ROUND_8_2026-09-02.md. TONIGHT UNCHANGED: 00:06Z verdict timer,
  00:14Z preflight, coordinator wake after, R-409 accrual with the R-411(ii)
  denominator, DA landing at e353119 (HOLD -> 3b7e10a), BE durable landing the
  round after the read, CO-8, --require-no-skips, DATA_ROOT split behind DA's
  landing. USER: SEVEN RULED; TWO OPEN -- the Phase-2 winner ruling and the
  content-liveness v2 freeze -- NEITHER NEEDED TONIGHT. UNCHANGED: G=1/5; the 011
  family 12 of 24 with Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 60 — archived 2026-09-02T19:25Z (1 entry, rolling-window overflow)

Moved in the MEM round-54 true-up of R-469, R-470, R-471 and Q-DE-53. Join rule as in batch 1.

```yaml
  2026-09-02T18:43Z (MEM ROUND 51 -- "NEVER CALLED" WAS FIXED BY CALLING IT).
  R-464 swept. Nothing run; ~/ctaNew-wt-de unread (DE round 34 in flight);
  everything a committed blob or a plan/manifest file. THE CAP ARC IS NOW THREE
  ROUNDS LONG AND IS THE CLEAREST LESSON OF THE WEEK: at e52d183 I found
  tranche_table NAMED IN A DOCSTRING AND NEVER CALLED; at 6d04833 it is called
  TWICE and I read both -- :979 INSIDE THE SELFTEST on a four-line hand-built
  _rows fixture, its own message reading "DE32-C2 CLOSED: `tranche_table` is
  CALLED (not merely named in a docstring)", which asserts THE FACT OF THE CALL,
  precisely what the previous finding's wording made salient; and :1068 ON THE
  PRODUCTION PATH inside the loop over coins x budgets x latency rungs, cap =
  tranche_table(rows, L, declare_cap=True), where THE NAME `cap` NEVER APPEARS
  AGAIN IN THE FILE -- assigned, dropped. SO THE DEFECT SURVIVED ITS OWN FIX BY
  MATCHING THE WORDS OF THE REPORT: a call whose result is discarded is not
  enforcement, it is a call. This is the round-46 standard with a new edge -- A
  FINDING PHRASED AS "X IS NEVER CALLED" INVITES A FIX THAT CALLS X -- and the
  durable phrasing is DE33-C7's: DECLARED BY AN UNCONSUMED CALL. From now on I
  name THE CONSUMPTION, not the call. NINE FINDINGS, AND THE THREE HIGH ONES ARE
  ABOUT THE OBJECT UNDER TEST, NOT THE PLUMBING: C1 the heads are not scored on
  their own features so IR-R4 IS NOT CLOSED; C2 the incumbent's thresholds are
  read at the WRONG KEY; C3 the acting control DOES NOT ACT on the drawn
  generation -- together meaning the runner would produce numbers that LOOK like
  the estimand and are not it. C4-C6 are the same family one layer down (a
  fixture forced null again, two of five arms never replayed with a default
  theta, rho's denominator a DECLARED CONSTANT); C8/C9 are the reporting layer
  (tracebacks as refusals, one key with two meanings, a silent tranche drop).
  Q-DE-51'S STATUS IS NOW VERIFIED-SHORT AND THE DISTINCTION IS WORTH KEEPING:
  ITS COUNTS, TIMING AND NO-ECONOMICS STATEMENTS STAND -- found short by
  execution is NOT found wrong, and a round can be honest in everything it claims
  and still not have built what the next step needs. THE RUN HAS NO DATE:
  "earliest 09-03" is WITHDRAWN (R-464 section 6); it is the round AFTER DE round
  34 lands, the reviewer reads it, AND section 5 is settled -- and I REPLACED the
  dated line beside the ruling rather than annotating it, because a withdrawn
  date left in place is exactly what gets quoted back as a commitment. A THIRD
  USER DECISION IS OPEN AND I CHECKED ITS PREMISE MYSELF: the runner chooses
  theta_repost = theta_cancel / 2 (:188) and HALF_SPREAD_CENTS = 0.5 (:101), and
  grepping the FROZEN PROTOCOL, the ADDENDUM and the MANIFEST for both names
  returns ZERO HITS IN ALL THREE (the fits I did not check, and say so) -- POLICY
  CONSTANTS CHOSEN AT THE BOTTOM OF THE STACK, in a file released as a shell, and
  harmful_stateful_policy REFUSES to default the first one precisely because it
  encodes a policy choice. The coordinator's recommendation is RECORDED, NOT
  RULED: make (ii) a MEASUREMENT by carrying the mid at fill, and put (i) to the
  USER as DE's proposal in a DATED ADDENDUM v2 BEFORE ANY RUN, with sensitivity
  at x1 and x0.5; nothing runs until both are settled. SEVEN RULED, THREE OPEN
  (the Phase-2 winner ruling, the content-liveness v2 freeze, and these two
  numbers), NONE NEEDED TONIGHT. SEQUENCING: DE round 34 (Q-DE-52) dispatched
  18:42Z with DE33-C1..C9 PLUS the five reviewer findings still open (DE31-R1,
  DE31-R2, DE32-R2, DE32-R3, DE32-R4; DE32-R1 closed FOR THE LGBM HEAD ONLY,
  DE32-R5 closed in tense), THE FEATURE TABLE AS THE OBJECT, no section 3
  economics read, ONE timed feature build under the 12G scope; the reviewer takes
  DE 33 + 34 AS ONE FILING at the round-34 tip, after BE round 8 (in flight at
  c54e48e) and BE round 9 (90638c3, row Q-BE-234 07681d2) which I record as
  FILED, COORDINATOR VERIFICATION PENDING, with the main tree CLEAN AT THE TIP
  (checked). TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator
  wake after, R-409 accrual with the R-411(ii) denominator, DA landing at
  e353119 (HOLD -> 3b7e10a), BE durable landing the round after the read, CO-8,
  --require-no-skips, DATA_ROOT split after DA's landing. UNCHANGED: G=1/5; the
  011 family 12 of 24 with Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 61 — archived 2026-09-02T19:39Z (1 entry, rolling-window overflow)

Moved in the MEM round-55 true-up of R-472, R-473 and the DE-35 filing. Join rule as in batch 1.

```yaml
  2026-09-02T18:50Z (MEM ROUND 52 -- I REPLACED ONE OCCURRENCE AND CALLED IT
  REPLACED). R-465 swept; nothing run; committed blobs only. MY RESIDUAL FIRST,
  BECAUSE IT IS THE SAME FAILURE I NAMED A ROUND AGO: I wrote "replaced, not
  annotated" about the withdrawn Phase-4 date and then replaced ONE occurrence;
  the coordinator found two more, and grepping the phrase across both files
  found THREE live statements of it, not two -- STATUS.yml:3806 (the
  phase4_run_when FLAG, current state), the round-49 window narrative ("reviewer
  reads -> run, earliest 09-03"), and HANDOFF.md:1830, WHICH THE DISPATCH DID
  NOT NAME. All three replaced; the only surviving occurrences NAME THE
  WITHDRAWAL. THE LESSON IS NOT "BE CAREFUL": A WITHDRAWAL IS A GREP, NOT AN
  EDIT -- a date lives in as many places as it was useful, and the one you
  remember is the one you wrote last. I EDITED A DATED WINDOW ENTRY TO DO IT AND
  SAY SO: the round-49 narrative now reads "run (NO DATE; see below)"; that entry
  has not yet rotated to the archive so nothing is lost, GIT HOLDS THE ORIGINAL
  WORDING, and when it rotates it carries the corrected text. I would not do this
  to a frozen artifact; the rolling window is CURRENT CONTEXT A READER CONSUMES
  and a withdrawn date sitting in it is exactly the quotation hazard. BE ROUND 9
  WAS EXECUTED, NOT JUST READ, AND THAT IS WHY IT IS SHORT: 93 PASS then RC 1
  under both launchers in a detached scratch worktree. The failure is the BE7-R4
  FLIP CHECK (90638c3:2565-2588) and I read it -- it takes _main_head and _prev =
  HEAD~1 with cwd=str(REPO), THE MAIN TREE, adds a worktree detached at that
  HEAD~1, copies the running file in and asserts it is DIRTY there -- A PREMISE
  THAT HOLDS ONLY UNTIL THE NEXT COMMIT LANDS ON THE BRANCH. From the first
  commit after BE's own, HEAD~1 no longer holds BE's version and the check's
  verdict changes WITHOUT THE DRIVER CHANGING AT ALL: A CHECK WHOSE ANSWER
  DEPENDS ON THE BRANCH'S HISTORY IS MEASURING THE REPOSITORY, NOT THE CODE. AND
  IT IS THE FOURTH SYMPTOM OF A SINGLE CONSTANT: REPO = Path("/home/yuqing/
  ctaNew") produced BE34-R3 (the spawned child, closed in round 6 AT ONE CALL
  SITE), BE7-R4 (the provenance block, round 49), BE9-C1 (this flip check) and
  BE9-C2 (anchors, data and the audit tree still rooted there, so "the tree that
  executed" is only ever the receipt's) -- ROUND 6 FIXED A USE; THE CONSTANT IS
  STILL THE DEFECT, exactly as recorded three rounds ago, and it has now cost
  four findings across four rounds. THE CONSEQUENCE FOR THE COUNT IS WHAT MAKES
  THIS HIGH RATHER THAN ANNOYING: CHECKS 95-117 ARE UNVERIFIABLE AT THAT TIP IN
  ANY TREE, twenty-three assertions past the failure never executing, so "the
  closures are present" is a claim about LINES, not about BEHAVIOUR -- which is
  why Q-BE-234 is VERIFIED-SHORT: closures present at the line, nothing past
  check 94 ran. THE LANDING MOVE IS RIGHT AND WORTH STATING AS A PRINCIPLE: the
  R-442 section 3(c) durable landing becomes BE ROUND 11, after the 00:14Z read
  AND after round 10 lands, because A DRIVER WHOSE OWN SELFTEST FAILS AT ITS TIP
  CANNOT PRODUCE THE ARTIFACT OF RECORD -- the artifact would be reproducible
  only by a driver that cannot demonstrate itself. BE ROUND 8'S REVIEW IS
  RELEASED (f804f33): CO-12 and CO-13 BOTH CONFIRMED CLOSED, two LOW findings --
  so the attribution defect I verified statically in round 47 is now closed and
  confirmed at the artifact by someone other than its author. SEQUENCING: BE
  round 10 (Q-BE-235) dispatched 18:48Z with BE9-C1..C3 and NO RUN AGAINST A REAL
  DAY; the reviewer takes BE 9 + 10 AS ONE FILING at the round-10 tip, then DE 33
  + 34 as one filing at the round-34 tip. TONIGHT'S TIMERS UNCHANGED: 00:06Z
  verdict, 00:14Z preflight, coordinator wake after, R-409 accrual with the
  R-411(ii) denominator, DA landing at e353119 (HOLD -> 3b7e10a), CO-8,
  --require-no-skips, DATA_ROOT split after DA's landing. USER: SEVEN RULED,
  THREE OPEN, none needed tonight. UNCHANGED: G=1/5; the 011 family 12 of 24 with
  Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 62 — archived 2026-09-02T20:09Z (1 entry, rolling-window overflow)

Moved in the MEM round-56 true-up of R-475, R-476 and the DE-36 filing. Join rule as in batch 1.

```yaml
  2026-09-02T19:05Z (MEM ROUND 53 -- THE FROZEN TEXT ALREADY ASKED FOR TWO OF THE
  THINGS THE RUNNER WOULD HAVE HAD TO INVENT). R-466, R-467 and R-468 swept;
  nothing run; frozen plan files and committed blobs only. I CHECKED THE ESTIMAND
  FILING'S CITATIONS AT THE FROZEN LINES AND THE READING THEY SUPPORT IS STRONGER
  THAN "FIVE FINDINGS": DRAFT:212-213 defines rho as the RETAINED-BOOK
  ADVERSE-COST / SPREAD-CAPTURE RATIO and :68 row 5 fixes the feed as
  GENERATION-LEVEL TRANCHE TABLES, NEVER PER-ROW LATENCY LABELS -- so EST-R1 is
  not a preference about denominators: THE FROZEN PROTOCOL ALREADY SAYS THE
  DENOMINATOR IS MEASURED, which means a constant HALF_SPREAD_CENTS does not
  merely approximate it, IT MAKES THE READING THRESHOLD THE CONSTANT (0.7
  c/share closes the route at H = 0.5 and does not at H = 1.0). A CONSTANT
  STANDING IN FOR A MEASUREMENT MOVES THE VERDICT, NOT THE PRECISION. AND EST-R2
  LANDS ON THE RECEIPT, NOT THE NUMBER: the over-the-hold value IS the frozen
  feed's (:68), so what is wrong is the receipt BINDING fill_horizon_s and an
  estimand note that declare a 1-second cap over it -- and I checked the other
  half myself: THE ADDENDUM MENTIONS "horizon" ZERO TIMES, so the governing
  document declares NO horizon, the receipt declares ONE, and they are about
  different quantities, which is why the closure is "declare the horizon the
  number has, in addendum v2, before the run" rather than "fix the constant".
  This is the round-50 finding grown a layer: the addendum omitted Cap 2, and it
  also omits ANY horizon at all. TWO OF THE SIX ADDENDUM ITEMS ARE NOT NEW ASKS
  -- THE FROZEN TEXT ALREADY DEMANDS THEM: DRAFT:71 row 8 says
  max_cancels_per_minute is DECLARED PER CELL with requested / effective(passed)
  / suppressed counts REPORTED, so EST-R4's identity is THE PROTOCOL'S OWN DUTY,
  unmet, and the runner's stated reason for skipping it is false on both halves;
  likewise STATEFUL_HARMFUL_CANCEL_TODO.md:381-382 REQUIRES theta_repost <
  theta_cancel FOR A DECLARED DWELL, so REPOST_DWELL_S is a number THE
  PROGRAMME'S OWN TODO DEMANDED and nobody has proposed (2.0 s with no proposal
  on record). THE GAP BETWEEN "THE DOCUMENT REQUIRES IT" AND "SOMEBODY CHOSE IT"
  IS WHERE ALL THREE NUMBERS LIVE. SO THE USER ITEM WIDENED WITHOUT MULTIPLYING:
  it is still ONE decision -- ONE DATED ADDENDUM v2 THE USER FREEZES -- now
  carrying THREE NUMBERS (theta_repost with sensitivity at 1.0x-eps and 0.5x and
  NEITHER SELECTED; REPOST_DWELL_S; HALF_SPREAD_CENTS ONLY IF KEPT, both DE and
  the reviewer recommending it be MEASURED AWAY) and THREE DECLARATIONS (the
  horizon the number has; repost parity in the control, required by the
  estimand's logic and SILENT in the frozen text; the rate-limit declaration with
  its identity). BUNDLING THEM IS THE RIGHT SHAPE: six separate asks would arrive
  as six chances to answer partially. EST-R5 IS THE ONE TO FLAG TO A READER IN A
  HURRY: the cancel set MUST be the drawn generations (:147-156) and the control
  at :601-604 DISCARDS _gen AND COLLAPSES same-(slug, side) draws, so THE ACTION
  COUNT IS NOT PRESERVED -- rule 2 of this programme's own reliability rules,
  ROWS ARE ACTIONS, appearing INSIDE THE MATCHED CONTROL where it silently
  changes what the control is matched on. DE ROUND 34 EXECUTED AT 47a2ba6 AND
  COUNTS AS FILED: four findings closed, C1 HALF-CLOSED, five named open,
  DE34-C1..C4 raised -- "counts as filed" being the honest status, the round
  having done what it said and named what it did not do. DE ROUND 35 IS ROUTED
  with the feed change the five open findings share, the code halves of
  EST-R1/R2/R4/R5, and THE ADDENDUM v2 DRAFT FOR THE USER -- the right ordering,
  because the draft is written by the seat that must implement it and frozen by
  the USER who must own the numbers. TWO IN-BAND CORRECTIONS CARRIED WITHOUT
  FLATTENING THEM: R-467 corrected R-466's reading times, and R-468 section 0
  corrects R-467 section 1's own "verified 18:53Z" -- A CORRECTION OF A
  CORRECTION IS NOT NOISE, it is the only way a time written ahead of the clock
  stops propagating. BE8-R1/R2 go to BE ROUND 12; the reviewer takes DE 33 + 34
  AS ONE FILING at 47a2ba6 (request c70e8e2), then BE 9 + 10 at the round-10 tip.
  THE PHASE-4 RUN HAS NO DATE. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z
  preflight, coordinator wake after, R-409 accrual with the R-411(ii)
  denominator, DA landing at e353119 (HOLD -> 3b7e10a), THE DURABLE LANDING IS BE
  ROUND 11 AFTER ROUND 10 LANDS, CO-8, --require-no-skips, DATA_ROOT split. USER:
  SEVEN RULED, THREE OPEN, none needed tonight. UNCHANGED: G=1/5; the 011 family
  12 of 24 with Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 63 — archived 2026-09-02T21:07Z (1 entry, rolling-window overflow)

Moved in the MEM round-57 true-up of R-477 and R-478. Join rule as in batch 1.

```yaml
  2026-09-02T19:25Z (MEM ROUND 54 -- A CHECK WHOSE JOB IS TO PROVE A DOCUMENT IS
  NOT BEING OBEYED). R-469, R-470, R-471 and Q-DE-53 swept; nothing run; no seat
  worktree read; be_forward_day.py untouched (BE round 10's row in flight,
  standing rule 9). THE DISPATCH SAYS THE ADDENDUM V2 DRAFT IS "CITED BY
  NOTHING", AND THE PRECISE VERSION IS BETTER THAN THE SUMMARY: I grepped, and
  the draft IS referenced exactly once in code, at
  de_phase4_protocol_check.py:287 -- AND THAT REFERENCE EXISTS TO PROVE THE
  ABSENCE. The check asserts the file exists AS A PROPOSAL THAT SAYS SO IN ITS
  OWN FIRST 400 CHARACTERS, and a second assertion at :296 builds the runner and
  head-scoring sources and requires "ADDENDUM_V2" not in _srcs, its message
  spelling out why: "a proposal cited by running code would be a seat deciding
  what the USER has not ruled (rule 14)". THAT IS THE INVERSE OF EVERY DEFECT
  CATALOGUED THIS WEEK -- the usual shape is a claim with no check behind it,
  and this is A CHECK WHOSE ENTIRE JOB IS TO KEEP A DOCUMENT NON-LOAD-BEARING, a
  negative control on AUTHORITY that names the rule it enforces. So "cited by
  nothing" is EXACT WHERE IT COUNTS (no number of the draft's is consumed) and
  LITERALLY FALSE (one reference exists and it is the guard); both halves belong
  in the record, because a later reader grepping the name will find a hit and
  needs to know it is the proof, not the breach. DE35-C1 IS WHY THE PACKAGE IS
  HELD, AND THE REASON IS SYMMETRY, NOT CAUTION: the control's REPOST EVENT HAS
  NO COUNTERPART IN THE TREATED STREAM, so a comparison meant to differ only in
  the policy differs also in WHAT EVENTS EXIST AT ALL, and forwarding section 5
  with that open would ask the USER to FREEZE A NUMBER WHOSE CONTROL IS NOT YET
  SYMMETRIC -- worse than waiting one round. "The USER package is ONE REVIEWER
  ROUND FROM READY" is the right status line: it names the DISTANCE, not a date.
  ADDENDUM v2 DRAFT sha16 6edefdfda909a897, LANDED and NOT FORWARDED. DE ROUND
  35 LANDED AND EXECUTED at 27c1ccd (Q-DE-53 19ddb43): counts
  21/67/26/21/24/21/184/92 reproduce and THE PREFLIGHT REFUSES BEFORE THE FEED,
  which matters more than it reads because DE34-C1 was precisely a refusal
  arriving AFTER the expensive step. DE34-C1..C4 ALL CONFIRMED by the reviewer
  (20bd233, 219 lines), and 47a2ba6 released AS ROUND 35'S BASE ONLY -- a release
  scoped to what it can support, not to the whole artifact -- with seven findings
  DE34-R1..R7. FIVE COORDINATOR FINDINGS DE35-C1..C5 sit on the same object.
  ROUND 36 IS QUEUED BEHIND THE REVIEWER'S DE-35 FILING (section 5 restated,
  DE35-C2..C5, DE34-R2/R3/R5/R6) -- QUEUED, NOT DISPATCHED, recorded that way
  because a queued round and a dispatched one differ in exactly the thing that
  goes wrong later. BE ROUND 10'S CODE LANDED at ff60d0a 19:13:34Z with a commit
  title that is itself the finding it closes ("the check was a function of the
  branch, not of the code"), and ITS ROW IS IN FLIGHT: recorded as
  LANDED-CODE / ROW-PENDING / VERIFICATION-PENDING and NOT READ HERE, the file
  being the surface of an open BE round -- the rule I bought in round 47 by
  measuring the wrong file. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight,
  coordinator wake after, R-409 accrual with the R-411(ii) denominator, DA
  landing at e353119 (HOLD -> 3b7e10a), THE DURABLE LANDING IS BE ROUND 11 AFTER
  ROUND 10 LANDS, CO-8, --require-no-skips, DATA_ROOT split. THE PHASE-4 RUN HAS
  NO DATE. USER: SEVEN RULED, THREE OPEN, none needed tonight. UNCHANGED: G=1/5;
  the 011 family 12 of 24 with Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 64 — archived 2026-09-02T21:15Z (1 entry, rolling-window overflow)

Moved in the MEM round-58 true-up of R-479. Join rule as in batch 1.

```yaml
  2026-09-02T19:39Z (MEM ROUND 55 -- THE FINDING SURVIVED; ITS EXPLANATION DID
  NOT). R-472, R-473 and the reviewer's DE-35 filing swept; nothing run; state
  files only. R-473 SECTION 0 IS THE SHAPE WORTH LEARNING FROM AND IT IS NOT A
  RETRACTION: R-471 section 2(a) explained DE35-C1 by saying the treated arm "is
  HELD with no event to anchor a repost", and the reviewer's fixture shows that
  is FALSE -- the treated arm DOES repost through a later generation's own
  below-theta_repost score, measured TREATED 1 cancel / 1 repost / value 4.000
  against CONTROL 2 / 2 / 0.000 on the same fixture. THE FINDING STANDS AND ITS
  MECHANISM IS REPLACED: the defect is the control's INVENTED LITERAL-0.0 EVENT
  at t0 + REPOST_DWELL_S (:802-804) whose PRESENCE AND TIMING DEPEND ON THE DRAW,
  so the null's repost economics are AN ARTEFACT OF THE CONTROL'S CONSTRUCTION,
  not of the scored stream. A FINDING CAN BE RIGHT ABOUT THE ASYMMETRY AND WRONG
  ABOUT WHY, and the corrected version is the MORE damaging one: an artefact that
  moves with the draw is worse than an absence. BOTH STRING CORRECTIONS MADE, AND
  THE SECOND IS MINE TO OWN: the 19:2xZ in STATUS.yml:3801 and HANDOFF.md:923 was
  THE COORDINATOR'S PLACEHOLDER WHICH I COPIED FAITHFULLY -- and faithfully is
  the problem: A PLACEHOLDER COPIED IS A PLACEHOLDER PUBLISHED. The register's
  stamps put the dispatch at 19:23:09-19:23:21Z so both now read 19:23Z, and the
  lesson is that AN "x" IN A TIMESTAMP IS NOT A TRANSCRIPTION, IT IS AN
  UNFINISHED FIELD -- resolve it or don't carry it. THE FIRST CORRECTION I
  CHECKED AT THE BLOB RATHER THAN TAKING IT: HALF_SPREAD_CENTS IS GONE AT
  27c1ccd, one mention surviving at :139 and it is THE COMMENT RECORDING THE
  DELETION ("EST-R1: HALF_SPREAD_CENTS IS GONE") with DRAFT:212-213 cited as the
  reason -- same shape as last round's addendum guard, THE SURVIVING STRING IS
  THE RECORD OF THE ABSENCE. So the USER package is TWO NUMBERS, and
  REPOST_DWELL_S = 2.0 now stands as a declared module constant with its own
  reason beside it ("an undeclared default in a policy runner is a policy choice
  nobody made"). A THIRD STALE OCCURRENCE EXISTS AND I AM DELIBERATELY NOT
  CHANGING IT: HANDOFF.md:1834, inside my round-53 dated entry, still says
  "HALF_SPREAD_CENTS only if kept" -- I checked the clocks, that entry was
  committed 19:06:05Z and 27c1ccd landed 19:12:02Z, SIX MINUTES LATER, so the
  sentence WAS TRUE WHEN WRITTEN. That is the line separating it from the
  "earliest 09-03" case I mishandled two rounds ago: THAT was a forward
  commitment that outlived its withdrawal, THIS is a dated observation accurate
  at its stamp. CORRECT CURRENT-STATE STATEMENTS; LEAVE DATED STATEMENTS THAT
  WERE TRUE WHEN STAMPED -- AND READ THE STAMP BEFORE DECIDING WHICH YOU HAVE.
  THE REVIEWER SPLIT THE BUNDLE RATHER THAN BLOCKING IT, the more useful verdict:
  SECTIONS 1 AND 4 MAY GO AHEAD OF 5; 2 AND 3 MAY NOT; 5 is restated by round 36
  and the package then goes to the USER WHOLE, IN ONE NOTIFICATION (R-473 section
  2). DE35-R2 TRAVELS WITH IT: each null draw is FOUR REPLAYS, ~800 PER CELL, so
  v1 section d's "of order 6 hours" is UNDERSTATED BY ABOUT 4x -- a cost estimate
  belongs in the package the USER freezes, not in the round that discovers it was
  wrong. TWO RULINGS RECORDED AS RULINGS: section 5 becomes THE TREATED ARM'S OWN
  STREAM PERMUTED WITHIN (side, hour) STRATA, so the control stops being invented
  and starts being a permutation of the thing it controls; and DE34-R7 is run
  AGAINST THE TIP with the called set COMPUTED and the residue CARRIED AS
  STATUSES, phase2_arms.py NOT_CALLED -- statuses rather than silence, this
  programme's rule 4 applied to arms. DE ROUND 35 REVIEW RELEASED (df123f2, 276
  lines): DE35-C1..C5 ALL CONFIRMED with C4 CONTESTED AS A BLOCKER, NOT AS A
  FINDING; five findings DE35-R1..R5; 27c1ccd released as ROUND 36'S BASE. SEATS:
  DE round 36 DISPATCHED at this tip (Q-DE-54 in flight, its row NOT landed when
  I filed -- checked; the scoring wiring is a LATER round with DE35-R3); the
  REVIEWER on RECORDED STANDBY until BE round 10's row lands, then BE 9 + 10 as
  ONE filing; BE round 10's row IN FLIGHT (code ff60d0a); DA on standby until
  tonight's 00:06Z / 00:14Z reads. THE PHASE-4 RUN HAS NO DATE. TONIGHT
  UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator wake after, R-409
  accrual with the R-411(ii) denominator, DA landing at e353119 (HOLD ->
  3b7e10a), the durable landing BE ROUND 11 after round 10 lands, CO-8,
  --require-no-skips, DATA_ROOT split. USER: SEVEN RULED, THREE OPEN, none needed
  tonight. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; R-419 revocable.
```

## Batch 65 — archived 2026-09-02T22:00Z (1 entry, rolling-window overflow)

Moved in the MEM round-59 true-up of R-480 and Q-DE-56. Join rule as in batch 1.

```yaml
  2026-09-02T20:09Z (MEM ROUND 56 -- THE GUARD KEEPS THE PROPOSAL FROM BEING
  OBEYED; IT DOES NOT KEEP IT FROM BEING WRONG). R-475, R-476, Q-DE-54, Q-BE-235
  and the reviewer's DE-36 filing swept; nothing run; committed blobs and plan
  files only. DE36-R1 IS THE SHARPEST ITEM THIS WEEK AND IT LANDS ONE LAYER ABOVE
  EVERY FINDING BEFORE IT: the addendum v2 DRAFT -- THE DOCUMENT THE USER IS
  BEING ASKED TO FREEZE -- says at :144 "This is a property, not a number, and
  the runner already implements it", THE REVIEWER RULES THAT FALSE, and IT MUST
  NOT REACH THE USER; I read the line at the file. SET THAT BESIDE LAST ROUND'S
  NEGATIVE CONTROL AND THE GAP IS EXACT: de_phase4_protocol_check.py proves
  NOTHING CITES THE DRAFT AS AUTHORITY -- an excellent guard that says nothing
  about whether THE DRAFT'S OWN PROSE IS TRUE. THE GUARD KEEPS A PROPOSAL FROM
  BEING OBEYED; IT DOES NOT KEEP IT FROM BEING WRONG. A false sentence about the
  code inside the document the USER freezes is worse than one in code: THE CODE
  HAS A SUITE, THE PROSE HAS A READER. SECTION 5 IS NOW RULED IN FULL and the
  (gamma) wording is a stronger object than what it replaces: TOTAL permutation
  of ALL above-threshold values within (side, hour) strata with THE DRAW NAMING
  WHICH GENERATIONS RECEIVE THEM; matched on the REALISED ACTION COUNT AFTER THE
  REPLAY with failed draws REJECTED AND REDRAWN under a bound;
  n_draws_attempted / n_draws_accepted / n_rejected_by_stratum IN THE RECEIPT;
  control#2 WITHDRAWN; P1-P4 predicates REPLACING THE SUBSTRING CHECK. THE MATCH
  MOVED FROM WHAT WAS INTENDED TO WHAT ACTUALLY HAPPENED -- a realised count
  cannot be satisfied by a draw that failed, which is precisely how the previous
  control flattered itself. AND C1 WAS CONFIRMED THE HARD WAY: MEASURED ON A
  FIXTURE, WITH A TRUE SWAP SHOWN NOT TO FIX IT -- the obvious repair was tested
  and rejected before the real one was ruled. DE36-R4 IS THE WEEK'S RECURRING
  GENUS, THIRD INSTANCE: three checks assert SOURCE STRINGS -- ok("res =
  arm_result(" in _null_src) at :1372, ok("preflight()" in _runsrc) at :1673, and
  ok(... "_above = [e for e in treated_scores" in _ctrl_src) at :1733 -- which is
  CO-11 (keyed on a spelling) and CO-12 (attribution by substring) in a third
  costume: A CHECK THAT READS SOURCE TEXT INSTEAD OF RUNNING IT PASSES FOR A
  RENAME AND FAILS FOR A REFORMATTING. Round 37 replaces them with predicates, in
  the right order -- SECTION 5 FIRST, THE DRAFT'S TWO SENTENCES SECOND, the two
  things that can reach the USER. THE COMPUTE FIGURE NOW TRAVELS HONESTLY SPLIT
  and the reason matters more than the number: the FEED ~28.6 MIN IS MEASURED
  (round 33) and travels; THE REPLAY IS UNMEASURED and its synthetic figure is A
  FLOOR, the fixture being 20 SLUGS x ONE GENERATION x ONE TRANCHE x ONE SIDE,
  NOT the "471 windows" a reader would assume; DE35-R2's 4x STANDS and the
  "~1000x overstated in total" half is DE'S OWN AND NOT ESTABLISHED. A COST
  ESTIMATE THAT MIXES ONE MEASURED HALF WITH ONE SYNTHETIC HALF IS NOT A RANGE,
  IT IS TWO DIFFERENT CLAIMS WEARING ONE NUMBER. ONE TRANSCRIPTION SLIP CAUGHT AT
  THE SOURCE: Q-DE-54 reports the runner "68 -> 71" while EXPECTED_CHECKS reads
  67 at 27c1ccd and 71 at 92c7da4, so it is 67 -> 71 -- and the correction
  matters because THE DELTA IS WHAT A READER USES: +4, NOT +3 (R-471 and Q-DE-53
  both recorded 67). Q-BE-235 IS LANDED AND NOT VERIFIED, and I keep those two
  words apart: BE9-C1..C3 closed, executed IN TWO TREES, a 26-CASE mutation
  audit, with COORDINATOR VERIFICATION IN FLIGHT SINCE 20:01:48Z; RUN B'S TREE IS
  A QUESTION FOR THE REVIEWER'S BE 9+10 ROUND, not a settled fact. The pin's
  three rulings and called#1's falsifier are recorded, and THE THREE DECLARED
  REASONS ARE TRUE -- checked by the reviewer, carried by me. The reviewer also
  corrected IN BAND its own round-35 "asserted from the parse" label. RELEASE:
  92c7da4 as ROUND 37'S BASE. SEATS: DE round 37 DISPATCHED at e791f4f (Q-DE-55
  in flight); the REVIEWER on RECORDED STANDBY until R-477 (BE round 10
  verified), then BE 9 + 10 AS ONE ROUND; BE on RECORDED STANDBY, round 11 being
  the durable landing after the 00:14Z read; DA standby until 00:06Z / 00:14Z.
  THE PHASE-4 RUN HAS NO DATE. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z
  preflight, coordinator wake after, R-409 accrual with the R-411(ii)
  denominator, DA landing at e353119 (HOLD -> 3b7e10a), CO-8,
  --require-no-skips, DATA_ROOT split. USER: SEVEN RULED, THREE OPEN, none needed
  tonight. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; R-419 revocable.
```

## Batch 66 — archived 2026-09-02T22:11Z (1 entry, rolling-window overflow)

Moved in the MEM round-60 true-up of R-481 and the DE-38 filing. Join rule as in batch 1.

```yaml
  2026-09-02T21:07Z (MEM ROUND 57 -- A VARIABLE ASSIGNED ONCE AND READ ZERO
  TIMES). R-477 and R-478 swept; nothing run; committed blobs only. DE37-C1 IS
  THE THIRD ACT OF THE STORY THIS FILE HAS BEEN TELLING ALL EVENING: round 36
  ruled section 5's (gamma) wording, round 37 WROTE IT INTO THE DRAFT VERBATIM
  and DID NOT BUILD IT ON THE RUN PATH -- the demand is still ACTIONS
  (:1093-1094), permuted_stream returns ok=False and a TRUNCATED-ZIP stream,
  _perm_ok is ASSIGNED ONCE AND READ ZERO TIMES, stream_predicates is
  SELFTEST-ONLY, and measured, TWO OF THREE DRAWS FAIL P2 with the third failing
  P3. So the document now says the right thing and the code does not do it, WHICH
  IS EXACTLY DE36-R1'S FAILURE MODE IN THE OPPOSITE DIRECTION: in one round the
  prose was FALSE about the code, in the next the prose is TRUE and the code is
  ABSENT. A VARIABLE ASSIGNED ONCE AND READ ZERO TIMES IS THE CHEAPEST POSSIBLE
  TELL, and it is the one a substring check would never find. DE37-C2 IS THE SAME
  DISEASE IN THE SEAL: DECLARED_ADDITIVE_SHAS = {} is FILLED FROM THE CURRENT
  FILE so the seal certifies whatever it is shown, and the coordinator DROVE it
  -- an edited select_v2_era body still reads ADDITIVE_DECLARED with the seal
  moving 3b34bdc86b1056ca -> 9a1158dd13713ad0. A DECLARATION THAT COMPUTES ITS
  OWN EXPECTED VALUE IS NOT A DECLARATION, IT IS A MIRROR. BE ROUND 10 VERIFIED
  AT 121/121 and the review released it AS ROUND 11'S BASE with the sentence that
  matters: NOTHING PRECEDES THE LANDING. BE9-C1..C3 CONFIRMED CLOSED AT THE PASS
  LINES, the 26-case audit green, nothing leaking; RUN B'S TREE ANSWERED FROM THE
  REFLOG -- it was THE SHARED MAIN TREE, no checkout in the window -- and the
  reviewer REPRODUCED RUN B'S CONDITION IN ITS OWN SCRATCH WORKTREE AT 874a041,
  also 121/121. So Q-BE-234'S NUMBERS STAND AS HISTORY, NOT AS PROPERTIES OF THE
  CODE: the count-versus-property distinction, settled by someone reproducing the
  condition somewhere else. STANDING RULE 10 ADOPTED (9 -> 10) AND IT IS NOT RULE
  9 RESTATED: RULE 9 SAYS WHERE YOU MAY READ, RULE 10 SAYS WHERE YOU MAY RUN and
  what you owe if you must run elsewhere -- the run DECLARED IN THE ROW BEFORE IT
  IS MADE (tree, HEAD, condition, and why no other tree produces it), NO WRITE
  outside its own git-admin entries, VERIFIED AFTERWARDS FROM A THIRD TREE by git
  worktree list + git status --short; and the first clause does the work: LOOK
  FIRST FOR A COMMIT THAT REPRODUCES THE CONDITION IN YOUR OWN TREE. A LOW
  AGAINST ME, AND THE FIX IS NOT THE NUMBER: I carried the DRAFT's sha as a BARE
  NAME and it has moved twice (6edefdfda909a897 at 27c1ccd -> ec1538f1545999d1 at
  218509e). A LIVING DOCUMENT'S SHA IS A FACT WITH AN AS-OF. I learned exactly
  this for the register's line numbers -- RECOUNT, NEVER PIN -- and did not carry
  it across to shas: HAVING A RULE AND APPLYING IT TO ONE DATATYPE IS HOW IT GETS
  RELEARNED. The field now reads "sha ... AS OF 218509e". A PATTERN, NOT A SLIP,
  AND I SAY SO BECAUSE IT IS THE SECOND: Q-DE-55 reports the runner "74 -> 85"
  while EXPECTED_CHECKS reads 71 at 92c7da4 and 85 at 218509e, so 71 -> 85, +14;
  last round it was "68 -> 71" for 67 -> 71. TWO CONSECUTIVE ROUNDS WHERE THE
  PRIOR COUNT IS WRONG AND THE NEW ONE IS RIGHT -- the new count comes from the
  run, the prior one from memory -- flagged as a pattern, not corrected twice in
  silence. TWO ITEMS JOIN THE USER PACKAGE, both honest about their status: THE
  SPLIT QUESTION (the section 3 population 08-24/08-25 SPANS BOTH FIT SPLITS and
  the DRAFT DOES NOT CHOOSE -- raised rather than settled, the right instinct)
  and A THIRD COST, UNMEASURED (tape 3,170,987,711 B + fragment 1,241,115,096 B,
  BYTE COUNTS VERIFIED against the files, the row and split counts DE'S) --
  recorded with that seam visible, because the last compute figure had to be
  split for exactly this reason. SEATS: the reviewer is on the DE ROUND-37 FILING
  (request REQUEST_DE_ROUND_37_2026-09-02.md at 2ca1c81, in flight); DE on
  RECORDED STANDBY (round 38 = C1..C5 as ruled plus DE37-Rn); BE on RECORDED
  STANDBY (round 11 = THE DURABLE LANDING after the 00:14Z read; round 12 =
  BE10-R1..R4 with BE8-R1/R2, R2 FIRST if the file is opened before the landing);
  DA standby until 00:06Z / 00:14Z. THE PHASE-4 RUN HAS NO DATE; THE PACKAGE IS
  NOT FORWARDED. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator
  wake after, R-409 accrual with the R-411(ii) denominator, DA landing at
  e353119 (HOLD -> 3b7e10a), CO-8, --require-no-skips, DATA_ROOT split. USER:
  SEVEN RULED, THREE OPEN, none needed tonight. UNCHANGED: G=1/5; the 011 family
  12 of 24 with Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 67 — archived 2026-09-02T22:52Z (1 entry, rolling-window overflow)

Moved in the MEM round-61 true-up of R-482 and Q-DE-57. Join rule as in batch 1.

```yaml
  2026-09-02T21:15Z (MEM ROUND 58 -- THE COMMENT PROMISES THE PROPERTY THE CODE
  COMPUTES AWAY). R-479 swept; nothing run; committed blobs only. DE37-C1..C5 ALL
  CONFIRMED, NONE CONTESTED -- the first round this evening where nothing the
  coordinator raised was argued down. DE37-R1 IS C2 WITH THE INTENT MADE VISIBLE
  AND I READ BOTH ENDS AT THE BLOB: at :155-161 the comment states the property
  in as many words ("A later edit to either side RE-OPENS THE QUESTION instead of
  inheriting this pass -- rule 12's shape applied to a declaration"), while at
  :380-386 _seal_declarations() computes those shas AT IMPORT FROM THE SOURCES
  THE COMPARISON READS -- so THE MODULE DOCUMENTS THE GUARANTEE IT COMPUTES AWAY,
  an edit moving the compared value and its expected value together. Driven both
  ways by the reviewer: an UNDECLARED edit to join_fills BLOCKS, a DECLARED edit
  to select_v2_era PROCEEDS with the seal simply moving -- so THE THREE DECLARED
  FUNCTIONS ARE A PERMANENT EXEMPTION, not a declaration that can expire. AND
  THAT NAMES SOMETHING SEEN THREE TIMES IN ONE EVENING: DE36-R1 was PROSE IN A
  DRAFT false about the code; DE37-C1 was PROSE IN A DRAFT true while the code
  was absent; this is PROSE IN THE MODULE promising what the implementation
  removes. THE THREE ARE ONE CLASS: A SENTENCE THAT DESCRIBES A PROPERTY NOBODY
  COMPUTES. The ruled fix is the inverse move -- the seal becomes LITERALS IN THE
  SOURCE (reason, sha_at_fit, sha_at_declaring_tip) with an edited function body
  as the falsifier: STOP COMPUTING THE EXPECTED VALUE. DE37-R2 IS THE SHARPEST
  SINGLE SENTENCE IN THE FILING: the (gamma) fixture at :2049-2051 calls
  permuted_stream directly on a hand-built draw, so it SATISFIES THE DEMAND BY
  CONSTRUCTION and is THE ONE STATE THE RUN PATH CANNOT PRODUCE -- a green suite
  certifying a state that cannot occur. That is why ruling (b) reads as it does:
  SECTION 5'S TEXT SURVIVES as the text the USER rules on, THE CODE IS WHAT
  FAILS, and THE PACKAGE MUST NOT TRAVEL WHILE THE SUITE PRESENTS (gamma) AS
  ACHIEVED. The document is not wrong; THE EVIDENCE FOR IT IS. R3 is small and
  worth keeping for its shape: P3 filters the draw to the stream's keys so AN
  EMPTY INTERSECTION IS VACUOUSLY TRUE -- a predicate that passes hardest exactly
  when there is nothing to check; ruling (f) fixes the order, assert want subset
  of keys(stream) FIRST. R4 IS MY OWN CATCH, CORROBORATED: the reviewer
  independently reports Q-DE-55's prior count of 74 as 71, the same figure I
  verified at EXPECTED_CHECKS last round and the SECOND CONSECUTIVE round of it,
  filed as ROW HYGIENE -- two seats saying so rather than one. RELEASE 218509e
  WITH ITS REASON ATTACHED BECAUSE THE REASON IS THE INTERESTING PART: nothing
  can run, THE PREFLIGHT REFUSES AT THE SCORER, so NO FINDING REACHES AN
  ARTIFACT -- a release granted BECAUSE THE CODE CANNOT PRODUCE ANYTHING is a
  very different object from one granted because the code is right, and the state
  files say which this is. FOUR CONDITIONS NOW STAND BETWEEN THE PACKAGE AND THE
  USER: the declared-vs-built sentence (or (gamma) built first); section 5 saying
  what happens to BELOW-threshold values with section 2 re-read; the seal's form
  settled; and THE TWO NUMBERS TRAVELLING WITH THE SPLIT QUESTION -- a judgement
  I would have got wrong, since the split question LOOKS like context and the
  ruling makes it A DECISION THE USER IS GIVEN WITH THE SECTION 5 NUMBERS, NOT A
  FOOTNOTE. MY ROUND 57 VERIFIED WITH NOTHING FOUND, and the round-56 LOW is
  recorded CLOSED AT THE SOURCE -- the bundle field now carries its AS-OF rather
  than a bare sha, which was the point of the correction rather than the sha
  itself. SEATS: DE round 38 IN FLIGHT (Q-DE-56, dispatched 21:14Z, the
  reviewer's six-step order); the REVIEWER on RECORDED STANDBY until Q-DE-56
  lands, then DE 38 as ONE round; BE on RECORDED STANDBY (round 11 = THE DURABLE
  LANDING after the 00:14Z read; round 12 = BE8-R1/R2 + BE10-R1..R4); DA standby
  until 00:06Z / 00:14Z. THE PHASE-4 RUN HAS NO DATE; THE PACKAGE IS NOT
  FORWARDED. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator
  wake after, R-409 accrual with the R-411(ii) denominator, DA landing at
  e353119 (HOLD -> 3b7e10a), CO-8, --require-no-skips, DATA_ROOT split. USER:
  SEVEN RULED, THREE OPEN, none needed tonight. UNCHANGED: G=1/5; the 011 family
  12 of 24 with Q4 failing; the sequencing rule; R-419 revocable.
```

## Batch 68 — archived 2026-09-02T22:59Z (1 entry, rolling-window overflow)

Moved in the MEM round-62 true-up of R-483 and the DE-39 filing. Join rule as in batch 1.

```yaml
  2026-09-02T22:00Z (MEM ROUND 59 -- THE CONTROL WAS FINALLY BUILT, AND IT
  ACCEPTS ONLY THE DRAW THAT CHANGES NOTHING). R-480 and Q-DE-56 swept; nothing
  run; committed blobs only. FIRST THE GOOD NEWS, BECAUSE IT IS REAL AND
  COMPLETE: at dfd4c00 EVERY DE37 ITEM CLOSES -- C1's three parts, C2, C3, C4,
  C5 and R1..R4. THE SEAL IS NOW SIX LITERALS and it was DRIVEN CLOSED BY THE
  COORDINATOR'S OWN EDIT, the falsifier the ruling asked for rather than the
  assertion that it works; (gamma) IS BUILT ON THE RUN PATH and `gen` is REQUIRED
  AT THE ADAPTER; counts 31/101/26/21/25/21/184/92 reproduce; I VERIFIED ALL
  THREE SHAS MYSELF -- runner a49458a04253175d, score-stream 4ccdadeafe982b87, v2
  DRAFT a45b87624f72b567. AND THE MOMENT (gamma) ACTUALLY RAN IT PRODUCED
  DE38-C1: on DE'S OWN C1 FIXTURE EVERY P4-ACCEPTED DRAW IS THE IDENTITY DRAW --
  the control stream EQUALS the treated stream, the null value EQUALS the treated
  value (40.0), and net_diff_vs_null_median_cents is 0.0. A CONTROL THAT IS
  FINALLY CORRECT BY CONSTRUCTION CAN STILL BE EMPTY BY SELECTION: the
  permutation is real, and the acceptance rule keeps only the permutation that
  permutes nothing. THE GUARD THAT EXISTS FOR EXACTLY THIS CANNOT FIRE: handed
  THE ACTIONS, under (gamma) with a held above event, the identity guard fires
  0 OF 200; handed THE DEMAND it fires 65 OF 200 -- so the check is not weak, IT
  IS LOOKING AT THE WRONG OBJECT, and that difference is the whole finding. AND
  THE TWO DIAGNOSTICS THAT WOULD HAVE SHOWN THE COLLAPSE ARE MEASURED ON THE
  WRONG POPULATION: n_distinct_draws and point_mass are computed over the
  ATTEMPTED draws, not the ACCEPTED ones -- A DIAGNOSTIC COMPUTED ON THE
  ATTEMPTED SET CANNOT REPORT A COLLAPSE IN THE ACCEPTED SET; it will show
  healthy variety in draws that were all thrown away. That is this evening's
  recurring shape at the OUTCOME layer: the number is real, THE POPULATION UNDER
  IT IS THE WRONG ONE. READ THE THREE ROUNDS TOGETHER: round 37 DECLARED (gamma)
  and did not build it; round 38 BUILT it and the built version ACCEPTS ONLY THE
  DRAW THAT CHANGES NOTHING -- each round's fix correct, each exposing the next
  layer, which is what a review loop should look like; the thing to resist is
  reading "all closed" as "done". THE DRAFT'S SHA MOVED A THIRD TIME AND THE
  FRAMING HELD: 6edefdfda909a897 (27c1ccd) -> ec1538f1545999d1 (218509e) ->
  a45b87624f72b567 (dfd4c00). Last round I stopped carrying it as a bare name;
  this round it moved again AND NEEDED NO CORRECTION, ONLY AN AS-OF -- what a
  good framing buys is that THE VALUE CHANGES AND THE RECORD DOES NOT BECOME
  WRONG. THE PACKAGE IS NOT FORWARDED AND THE REASONS NOW STACK THREE DEEP: the
  reviewer's four conditions (i)-(iv), UNVERIFIED BY THE REVIEWER, AND DE38-C1
  OPEN -- all three written into the bundle status rather than the newest alone,
  because A SINGLE STATED BLOCKER INVITES THE READING THAT CLEARING IT CLEARS THE
  WAY. DE38-C2 (LOW) and DE38-C3 (a QUESTION) are also open. UNCHANGED AND
  STATED: the PHASE-4 RUN HAS NO DATE and preflight() STILL REFUSES AT THE SCORER
  (PA.tape_index / PA._feature_pass NOT WIRED). SEATS: the reviewer has DE round
  38 (dispatched 21:58Z, request REQUEST_DE_ROUND_38_2026-09-02.md, ONE filing
  expected); DE on RECORDED STANDBY with round 39 waiting on that filing; BE and
  DA on RECORDED STANDBY. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight,
  coordinator wake after, R-409 accrual with the R-411(ii) denominator, DA
  landing at e353119 (HOLD -> 3b7e10a), BE round 11 = the durable landing, CO-8,
  --require-no-skips, DATA_ROOT split. USER: SEVEN RULED, THREE OPEN, none needed
  tonight. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; R-419 revocable.
```

## Batch 69 — archived 2026-09-02T23:34Z (1 entry, rolling-window overflow)

Moved in the MEM round-63 true-up of R-484 and Q-DE-58. Join rule as in batch 1.

```yaml
  2026-09-02T22:11Z (MEM ROUND 60 -- THE IDENTITY DRAW IS NOT A BUG TO EXCLUDE,
  IT IS A STATE TO COUNT). R-481 and the reviewer's DE-38 filing swept; nothing
  run; committed blobs only. THE RULING ON DE38-C1 IS BETTER THAN THE FIX I WOULD
  HAVE EXPECTED AND THE DIFFERENCE IS WORTH STATING: the obvious response to
  "every accepted draw is the identity draw" is to STOP ACCEPTING IT, and the
  ruling does the OPPOSITE -- the identity draw is ADMITTED AND COUNTED
  (n_accepted_identity per stratum), the identity guard is RETIRED for (gamma)
  with its parse assertion deleted, n_distinct and point_mass move onto the
  ACCEPTED set, and an accepted set of ONE DISTINCT DRAW becomes null:
  DEGENERATE -- no interval, a point estimate, LABELLED AS SUCH. EXCLUDING THE
  IDENTITY DRAW WOULD HAVE MANUFACTURED A NULL THAT DIFFERS; COUNTING IT REPORTS
  THE TRUTH THAT THIS NULL DOES NOT. AND THE BOUNDARY IS DRAWN EXACTLY WHERE RULE
  13 REQUIRES: the collapse RE-OPENS REPORTING ONLY, the frozen matching rule
  (DRAFT:147-156) UNTOUCHED -- a finding does not get to reach back into a frozen
  document because it is inconvenient; it changes what the artifact SAYS ABOUT
  ITSELF. The DRIVEN check is specified so it cannot pass on the degenerate case:
  it must assert AN ACCEPTED DRAW WHOSE CONTROL DIFFERS and n_distinct_accepted
  >= 2 -- the shape this programme keeps arriving at, A CONTROL MUST DEMONSTRATE
  THE STATE IT CLAIMS TO DISTINGUISH, not merely run. DE38-R1 I VERIFIED AT THE
  BLOB AND THE MISMATCH IS A POPULATION MISMATCH AGAIN: pool is built over THE
  REFERENCE'S GENERATIONS (:1139) while the draw is over THE STREAM'S
  ABOVE-THRESHOLD EVENTS, and _room (:1192, :1200) and strata_with_room (:1331)
  are computed on that pool, so the receipt reports FREEDOM THE DRAW CANNOT USE
  -- the THIRD distinct instance tonight of THE NUMBER IS REAL, THE POPULATION
  UNDER IT IS THE WRONG ONE (after n_distinct/point_mass on the attempted set,
  and the identity guard handed the actions instead of the demand). THE REVIEWER
  CONFIRMED DE37'S CLOSURES INDEPENDENTLY, including by RE-DRIVING THE SEAL WITH
  ITS OWN EDIT: two seats have now driven that falsifier from different trees, and
  A SEAL THAT ONLY ITS AUTHOR CAN BREAK IS NOT SEALED -- this one has been broken
  twice, on purpose, and closed both times. THE CONDITION-(i) VERDICT IS A SPLIT I
  WOULD HAVE FLATTENED AND SHOULD NOT HAVE: it IS met as to THE STREAM AND THE
  REJECTION ACCOUNTING and is NOT met as to THE NULL'S SECTION 5 PROMISES, with
  the reviewer's sentence carried verbatim -- "a USER reading 5 today would be
  adopting the words while the artifact behind them produces a null that cannot
  differ". A CONDITION CAN BE MET IN ITS MECHANICS AND UNMET IN ITS MEANING, AND
  ONLY THE SECOND ONE PROTECTS THE USER. THE STACK SHRANK AND I REPLACED RATHER
  THAN CARRIED: "unverified by the reviewer" is CLOSED by this filing, so it is
  GONE from the bundle status, not annotated as satisfied; what remains is
  DE38-C1 OPEN and CONDITION (i) UNMET AS TO SECTION 5, with (ii)-(iv) MET at
  dfd4c00. A REASON LIST THAT ONLY EVER GROWS STOPS BEING READ; this one now says
  exactly what is left. DE38-C2 CONFIRMED (LOW); DE38-C3 CONFIRMED; DE38-R2/R3/R4
  LOW (a docstring-asserted limit at de_score_stream:342, two sources for the
  event contract at :155 and :172-174, and a falsifier-flag receipt that must say
  so). UNCHANGED AND STATED: the PHASE-4 RUN HAS NO DATE; preflight() STILL
  REFUSES AT THE SCORER; the BUNDLE SHA STAYS a45b87624f72b567 AS OF dfd4c00
  until round 39 moves the DRAFT. RELEASE dfd4c00 AS ROUND 39'S BASE, dispatched
  with the six-step order and THE SECTION 5 REPORTING SENTENCE LAST -- the right
  place for it, THE WORDS GOING IN AFTER THE ARTIFACT BEHIND THEM IS TRUE. SEATS:
  DE round 39 DISPATCHED (Q-DE-57 expected, ONE commit); the REVIEWER on RECORDED
  STANDBY until it lands; BE and DA on RECORDED STANDBY. MY ROUND 59 VERIFIED
  WITH NOTHING FOUND and the archive batch-65 move recorded verbatim. TONIGHT
  UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator wake after, R-409
  accrual with the R-411(ii) denominator, DA landing at e353119 (HOLD ->
  3b7e10a), BE round 11 = the durable landing, CO-8, --require-no-skips,
  DATA_ROOT split. USER: SEVEN RULED, THREE OPEN, none needed tonight. UNCHANGED:
  G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule; R-419
  revocable.
```

## Batch 70 — archived 2026-09-02T23:41Z (1 entry, rolling-window overflow)

Moved in the MEM round-64 true-up of R-485 and the DE-40 filing. Join rule as in batch 1.

```yaml
  2026-09-02T22:52Z (MEM ROUND 61 -- A DEGENERATE NULL THAT REFUSES ITS OWN
  INTERVAL). R-482 and Q-DE-57 swept; nothing run; committed blobs only. I
  VERIFIED ALL THREE SHAS AND BOTH LINE COUNTS AT cd93663 -- runner
  2976b46e1eb67a22 (3,201), score-stream f85be3354610e2ce (420), DRAFT
  6a62569f536e460f (290) -- and EXPECTED_CHECKS = 115. THE BUILD IS THE ANSWER TO
  LAST ROUND'S RULING AND THE BEST PART OF IT IS A REFUSAL: a null whose accepted
  set holds ONE DISTINCT DRAW now emits null: DEGENERATE (n_distinct_accepted =
  1) with NO QUANTILES, NO net_diff, and the predicate interval:
  POINT_ESTIMATE_NO_INTERVAL. THE ARTIFACT DECLINES TO PRODUCE THE STATISTIC IT
  CANNOT SUPPORT -- rarer and more valuable than producing it with a caveat,
  because A CAVEAT TRAVELS SEPARATELY FROM THE NUMBER AND A MISSING FIELD DOES
  NOT. And accepted_by_stratum comes BEFORE any section 3 number: ORDERING IS AN
  ARGUMENT -- the reader meets the population before the estimate, so a
  degenerate accepted set cannot be discovered after the figure has been read.
  THE GUARD'S RETIREMENT WAS DONE PROPERLY -- ITS PARSE CERTIFICATE WAS INVERTED,
  NOT DELETED: a retired control that simply disappears leaves a suite that once
  proved something and now proves nothing with no record of the change, while
  inverting the certificate keeps the fact of the retirement INSIDE the thing
  that used to assert it. DE39-C1 IS THE ONE CANDIDATE AND I READ IT AT THE BLOB
  BECAUSE "DECISION-INERT" IS EXACTLY THE PHRASE THAT BURIES FINDINGS: the
  computation is SET IDENTITY (_e["distinct"].add(frozenset(_keys)) and if _keys
  == _above_by_st.get(_st, set()), :1382-1386) while the comment two lines below
  promises STREAM IDENTITY ("the control's stream is then the treated arm's,
  exactly") -- THE CODE COMPARES SETS OF KEYS, THE PROSE ASSERTS THE STREAMS ARE
  EQUAL. They coincide today at one theta with enable_reduce False, and that is
  the point: A DEFINITION MISMATCH THAT IS DECISION-INERT TODAY IS A DORMANT
  FINDING, NOT A RESOLVED ONE -- it starts costing the moment a second theta or a
  live enable_reduce makes the definitions come apart. THE REVIEWER'S TO RULE;
  recorded OPEN, with the measurement, NOT closed. THAT IS THIS EVENING'S CLASS
  IN ITS MOST EASILY-LOST POSITION: the gap between prose and computation has
  appeared FOUR times today -- false about the code, true while the code was
  absent, promising what the implementation removed, and now DESCRIBING A
  STRONGER PROPERTY THAN THE COMPUTATION DELIVERS IN A PLACE WHERE NOTHING
  CURRENTLY DISAGREES. THE FIRST THREE WERE CAUGHT BECAUSE SOMETHING FAILED; THIS
  ONE CAN ONLY BE CAUGHT BY READING. THE BUNDLE'S REMAINING REASONS NOW EACH NAME
  A SEAT AND A TIP: DE39-C1 -> the reviewer's round; condition (i) -- MEASURED as
  built by the coordinator, THE REVIEWER'S CONFIRMATION BEING THE CLOSURE, not
  the measurement; (ii)-(iv) -- MET at dfd4c00, TO BE RE-STATED at cd93663. A
  BLOCKER WITH AN OWNER IS A BLOCKER; ONE WITHOUT IS A BACKLOG. THE DRAFT'S SHA
  ADVANCED A FOURTH TIME (6edefdfda909a897 -> ec1538f1545999d1 ->
  a45b87624f72b567 -> 6a62569f536e460f), FOUR SUPERSESSIONS IN FOUR ROUNDS, NONE
  OF THEM A CORRECTION -- the AS-OF framing has now been tested more thoroughly
  than most of the code. UNCHANGED AND STATED: the PHASE-4 RUN HAS NO DATE;
  preflight() STILL REFUSES AT THE SCORER. SEATS: the reviewer on DE 39 AS ONE
  ROUND (REQUEST_DE_ROUND_39_2026-09-02.md); DE, BE and DA on RECORDED STANDBY.
  MY ROUND 60 VERIFIED WITH NOTHING FOUND, archive batch 66 recorded verbatim.
  TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator wake after,
  R-409 accrual with the R-411(ii) denominator, DA landing at e353119 (HOLD ->
  3b7e10a), BE round 11 = the durable landing, CO-8, --require-no-skips,
  DATA_ROOT split. USER: SEVEN RULED, THREE OPEN, none needed tonight. UNCHANGED:
  G=1/5; the 011 family 12 of 24 with Q4 failing; the sequencing rule; R-419
  revocable.
```

## Batch 71 — archived 2026-09-03T00:36Z (1 entry, rolling-window overflow)

Moved in the MEM round-65 true-up of R-486, R-487 and the three landings. Join rule as in batch 1.

```yaml
  2026-09-02T22:59Z (MEM ROUND 62 -- THE REFUSAL DOES NOT SAY WHICH REFUSAL IT
  IS). R-483 and the reviewer's DE-39 filing swept; nothing run; committed blobs
  only. THE MILESTONE FIRST, STATED SO THE TWO HALVES CANNOT MERGE: the reviewer
  has RELEASED the package -- conditions (i)-(iv) ALL MET at cd93663, with its
  own question answered on the record, "May the package travel whole? YES --
  DE39-C1 does not hold it", sent WITH section 5's enable_reduce clause -- AND IT
  IS NOT FORWARDED. It travels AT DE'S ROUND-40 TIP, ONCE THE COORDINATOR
  VERIFIES THE DRAFT CLAUSE AT THE BLOB, because A PACKAGE FORWARDED BEFORE THE
  CLAUSE LANDS WOULD BE SUPERSEDED IN-BAND THE SAME NIGHT (rule 13) -- the USER
  would be reading a document a correction was already chasing. After many rounds
  of "not forwarded" the word that changed is RELEASED, and I have KEPT THE TWO
  APART in the bundle status rather than letting the good news blur the gate. The
  two reasons that stood there last round are CLOSED AND REPLACED, NOT ANNOTATED
  (DE39-C1 and condition (i)): a status field that accumulates struck-through
  reasons stops being a status field. DE39-R1 IS THE FINDING I WOULD MOST WANT A
  READER TO SEE AND I READ IT AT THE BLOB: the predicate row computes "interval":
  ("NULL_QUANTILES" if c.get("null_quantiles") else "POINT_ESTIMATE_NO_INTERVAL"),
  so POINT_ESTIMATE_NO_INTERVAL IS EMITTED WHENEVER null_quantiles IS FALSY --
  covering A NULL THAT COLLAPSED (degenerate, one distinct accepted draw) AND A
  NULL THAT NEVER RAN -- with NO null FIELD IN THE ROW TO TELL THEM APART, while
  the comment directly above says "an interval only where the draws ran;
  everywhere else THE LABEL SAYS WHAT IT IS". THE LABEL IS THE SAME IN BOTH CASES,
  SO IT DOES NOT. READ THAT AGAINST LAST ROUND AND THE LESSON IS SHARPER THAN
  EITHER FINDING ALONE: round 39 taught the artifact to REFUSE A STATISTIC IT
  CANNOT SUPPORT and I recorded that as its best property; THIS ROUND SHOWS THE
  REFUSAL DOES NOT SAY WHICH REFUSAL IT IS. A REFUSAL IS ONLY AS INFORMATIVE AS
  ITS REASON, and "no interval" answers a question nobody asked -- the reader
  wants to know whether the null was EMPTY or ABSENT; the prose promising the
  discrimination the code does not make is the same class AGAIN, now INSIDE THE
  FIX I PRAISED. DE39-R2 CAME IN THE OTHER DIRECTION AND THAT IS WORTH ITS OWN
  LINE: the coordinator had len(_ca) == 1 (:2492) as "noted, not a finding" and
  THE REVIEWER'S READING WAS ADOPTED -- the check ENCODES TODAY'S SINGLE
  changed_at and should group by it. Most weeks the traffic runs the other way (a
  claim that does not reproduce, a contested blocker); A NOTE PROMOTED TO A
  FINDING BY A SECOND READER is the same machinery working in the direction
  nobody celebrates. EVERY DE38 CLOSURE IS CONFIRMED, EACH DRIVEN, and the six
  literals were verified at BOTH SIDES of 851edaf by the reviewer's own _fn_asts
  -- the THIRD independent instrument on that seal in three rounds: A SEAL
  CHECKED ONLY BY THE TOOL THAT WROTE IT IS A MIRROR, and this one has now been
  read by three different parsers. SEQUENCING WITH THE ONE RESTRAINT THAT
  MATTERS: cd93663 is round 40's base, four-step order (ruling (i); rulings
  (ii)+(iii); DE39-R1; DE39-R2) with THE DRAFT CLAUSE LAST, and THE REVIEWER'S
  STEP 5 -- the wiring's expensive half -- IS USER-GATED AND NOT DISPATCHED: a
  reviewer may recommend work the coordinator may not start, and recording the
  gap is how the gate survives the enthusiasm. UNCHANGED AND STATED: the PHASE-4
  RUN HAS NO DATE; preflight() STILL REFUSES AT THE SCORER (1.33 s); the bundle
  sha stays 6a62569f536e460f AS OF cd93663 until round 40 moves the DRAFT. MY
  ROUND 61 VERIFIED WITH NOTHING FOUND, archive batch 67 verbatim. SEATS: DE
  round 40 dispatched (Q-DE-58 expected); the reviewer, BE and DA on RECORDED
  STANDBY. TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator wake
  after, R-409 accrual with the R-411(ii) denominator, DA landing at e353119
  (HOLD -> 3b7e10a), BE round 11 = the durable landing, CO-8,
  --require-no-skips, DATA_ROOT split. USER: SEVEN RULED, THREE OPEN, none needed
  tonight. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; R-419 revocable.
```

## Batch 72 — archived 2026-09-03T00:54Z (1 entry, rolling-window overflow)

Moved in the MEM round-66 true-up of R-488, R-489 and the landings filing. Join rule as in batch 1.

```yaml
  2026-09-02T23:34Z (MEM ROUND 63 -- FORWARDED, AND THE DELTA IS +17 -0). R-484
  and Q-DE-58 swept; nothing run; committed blobs only. THE PACKAGE HAS GONE TO
  THE USER: released by the reviewer at 650569c, FORWARDED AT 35452c0 PER R-484,
  and the ORDERING is the whole reason it is worth a line -- the release came
  first, the clause was written second, and the forwarding waited for the clause
  to EXIST TO BE READ. A RELEASE IS A JUDGEMENT ABOUT A TEXT; A FORWARDING IS A
  CLAIM THAT THE TEXT IS THERE. This programme kept those apart for four rounds
  and then did them in order. I VERIFIED EVERY FIGURE AT THE BLOB: runner
  3f4bf21da2dfa188, 3,329 lines, EXPECTED_CHECKS = 119; DRAFT cb693000880c3d94,
  307 lines. AND THE DRAFT DELTA IS +17 -0, WHICH DESERVES ITS OWN SENTENCE: the
  round that carried the clause ADDED SEVENTEEN LINES AND REMOVED NONE, so THE
  TEXT THE REVIEWER RELEASED IS STILL, LINE FOR LINE, INSIDE THE TEXT THE USER
  NOW READS -- nothing re-worded on the way out. For a document about to be
  frozen that is the strongest cheap statement available, and it is trivially
  checkable now and unprovable in a month. FIVE ASKS ARE WITH THE USER AND ONE OF
  THEM IS A SHAPE RATHER THAN A NUMBER: 1 the horizon; 2 theta_repost and 3
  REPOST_DWELL_S -- PAIRS OR FIXED; 4 inf PLUS THE IDENTITY; 5 repost parity WITH
  the enable_reduce clause; and 1a TRAIN/SCORE RULED WITH 2 AND 4, NOT
  SEPARATELY -- the same instinct that made the split question travel WITH the
  numbers rather than beneath them: BUNDLING IS NOT TIDINESS, IT STOPS A DECISION
  BEING ANSWERED IN A FORM THAT PRESUPPOSES THE OTHERS. USER-PENDING IS NOW FOUR
  and I REPLACED the three-item line rather than appending: (1) the 09-02 accrual
  after tonight's 00:06Z / 00:14Z reads; (2) the Phase-2 winner; (3) the
  content-liveness v2 freeze; (4) the addendum v2 package. ITEM (1) IS NEW TO
  THIS LIST -- it has lived in these files as R-409'S PRINCIPLE APPLIED
  MECHANICALLY AFTER THE VERDICT, and it now appears as an item awaiting
  tonight's reads; BOTH READINGS ARE IN THE RECORD AND I HAVE NOT SILENTLY MERGED
  THEM. DE ROUND 40 EXECUTED WITH EVERY RULING DRIVEN -- (i), (ii), (iii),
  DE39-R1 and DE39-R2 -- PLUS AN IN-SUITE REORDERING INVARIANCE, the right
  companion to a set-identity fix: having named the identity as SET identity, the
  suite now DEMONSTRATES that order does not change the answer rather than
  asserting it in prose, so THE CLAIM AND THE DEMONSTRATION ARRIVE TOGETHER.
  Nothing from DE is open pending the reviewer's DE-40 filing, and THE REVIEWER'S
  STEP 5 -- the wiring's expensive half -- REMAINS USER-GATED AND NOT DISPATCHED:
  the gate outlived the round that wanted through it. UNCHANGED AND STATED: the
  PHASE-4 RUN HAS NO DATE; preflight() STILL REFUSES AT THE SCORER. MY ROUND 62
  VERIFIED. SEATS: the reviewer on DE round 40
  (REQUEST_DE_ROUND_40_2026-09-02.md); DE, BE and DA on RECORDED STANDBY. TONIGHT
  UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator wake after, R-409
  accrual with the R-411(ii) denominator, DA landing at e353119 (HOLD ->
  3b7e10a), BE round 11 = the durable landing, CO-8, --require-no-skips,
  DATA_ROOT split. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; R-419 revocable.
```

## Batch 73 — archived 2026-09-03T01:04Z (1 entry, rolling-window overflow)

Moved in the MEM round-67 true-up of R-490 and the DE-41 filing. Join rule as in batch 1.

```yaml
  2026-09-02T23:41Z (MEM ROUND 64 -- A COST ESTIMATE THAT INCLUDES "AND THEN WE
  ASK YOU AGAIN"). R-485 and the reviewer's DE-40 filing swept; nothing run;
  committed blobs only; no plan, review or request file touched. THE ITEM WORTH
  THE ROUND IS THE REVIEWER PRICING ASK (5) PER ANSWER, NOT PER TASK: answering
  MECHANICS costs ONE FIELD, ONE REFUSAL, ONE FALSIFIER; answering the
  score-split costs A ROUND -- AND A RE-DECLARATION THAT TRAVELS BACK TO THE
  USER. So ONE BRANCH OF THIS DECISION ENDS IN ANOTHER QUESTION FOR THE SAME
  PERSON and the other does not. THAT IS A DIFFERENT KIND OF ESTIMATE FROM THE
  ONES THIS FILE HAS CARRIED ALL EVENING, AND BETTER: the compute figures priced
  THE WORK, this prices EACH ANSWER INCLUDING THE ANSWER'S OWN FOLLOW-ON, and A
  DECISION-MAKER TOLD ONLY THE TOTAL COST CANNOT SEE THAT ONE OPTION BUYS A
  SECOND INTERRUPTION OF THEIR OWN TIME. It is the reviewer's ESTIMATE and
  NOTHING IS DECIDED (rule 14) -- written into the bundle block AS A QUOTED
  ESTIMATE rather than as a recommendation, because A COST ATTACHED TO ONE BRANCH
  READS AS ADVICE UNLESS IT IS LABELLED. THE CLOSURES ARE CLEAN AND ONE IS
  MEASURED RATHER THAN ASSERTED: ruling (i) is YES; (ii)+(iii) are CONFIRMED
  CLOSED on a 21-FIELD MEASUREMENT IN WHICH EXACTLY ONE FIELD MOVES -- THE
  DISCRIMINATING COUNT, NOT THE REASSURING ONE, since twenty fields staying put is
  what makes the one that moves meaningful; DE39-R1 is CONFIRMED CLOSED and
  DE39-R2 is CLOSED IN FORM, a phrase kept EXACTLY AS FILED because "closed in
  form" and "closed" are different claims and the difference is the whole reason
  to write it down. THE THREE NEW FINDINGS ARE ALL SMALL AND ALL SPECIES THIS
  PROGRAMME KEEPS MEETING: DE40-R1, _by_ca has NO FALSIFIER -- a check with
  nothing that makes it fail; DE40-R2, null_status derived FROM ABSENCE RATHER
  THAN FROM REQUEST -- a status inferred from a missing thing rather than from
  what was asked for; DE40-R3, a SIX-FIELD ENUMERATION WHERE THE BLOCK IS WIDER
  -- a list that will be right until the block grows. Three LOWs, three shapes
  this file already has names for, and NONE OF THEM HOLDS THE BASE: 35452c0 is
  round 41's base, three-step order, and DE round 41 is IN FLIGHT. NOTHING ELSE
  MOVED: the package stays FORWARDED at 35452c0 with DRAFT cb693000880c3d94;
  USER-PENDING REMAINS FOUR; the PHASE-4 RUN HAS NO DATE; preflight() still
  refuses at the scorer; STEP 5 STAYS USER-GATED. MY ROUND 63 VERIFIED. SEATS:
  DE round 41 in flight; the reviewer on RECORDED STANDBY; BE and DA standby.
  TONIGHT UNCHANGED: 00:06Z verdict, 00:14Z preflight, coordinator wake after,
  R-409 accrual with the R-411(ii) denominator, DA landing at e353119 (HOLD ->
  3b7e10a), BE round 11 = the durable landing, CO-8, --require-no-skips,
  DATA_ROOT split. UNCHANGED: G=1/5; the 011 family 12 of 24 with Q4 failing; the
  sequencing rule; R-419 revocable.
```

## Batch 74 — archived 2026-09-03T01:14Z (1 entry, rolling-window overflow)

Moved in the MEM round-68 true-up of R-491 and DE round 42. Join rule as in batch 1.

```yaml
  2026-09-03T00:36Z (MEM ROUND 65 -- THE NIGHT THE CALENDAR STOPPED BEING A
  PLAN). R-486, R-487 and the three landings swept; nothing run; every figure
  re-derived at the blob or the file. THE FIRST GOVERNED VERDICT EXISTS: the
  00:06Z unit fired with STATUS 0, and I read the artifacts rather than the
  report -- the 09-02 verdict hashes 6f283262df463957, as_of
  2026-09-03T00:06:01.399260Z, with race_accrual_eligible TRUE for btc and eth;
  the mask hashes 0bac652c44fba8f2 and carries 251 MASKED WINDOWS ACROSS 7 COINS,
  EVERY ONE CONTENT_THIN (btc 40, eth 40, xrp 40, sol 39, bnb 38, doge 38, hype
  16); the 00:14Z preflight read 10/10 GOVERNED_VERDICT_COMPLETE. After weeks of
  building the instrument, THE INSTRUMENT RAN ON A REAL DAY AND PRODUCED A
  VERDICT. AND THE DECISION IT PRODUCES IS NOT OURS: the 09-02 ACCRUAL on its
  complement is THE USER'S CALL (R-409), and R-486 RECOMMENDS ACCRUE AND IT IS
  UNRULED -- written in that order, because A RECOMMENDATION RECORDED NEXT TO AN
  UNRULED ITEM IS ONE CARELESS READ AWAY FROM BECOMING THE DECISION. THE RECEIPTS
  OF RECORD ARE IN GIT AND BYTE-IDENTICAL: BE round 11 landed 4000106752f816e4
  (14,022 B) and 0907b0369e14d77b (1,123 B), both RE-HASHED FROM THE COMMIT --
  the 09-01 race score has moved FROM A SCRATCHPAD UNDER /tmp TO A TRACKED
  ARTIFACT, which finishes the whole arc of R-442 section 3(c); DA round 18
  landed the chain REBASED WITH NO CONTENT MOVED plus the tracked verdict and the
  force-added mask. THE NEW FACT IS A WIRING FACT AND DESERVES ITS ESCALATION:
  the 09-04 00:06Z run WILL EXECUTE THE LANDED CHAIN (rounds 10-12 are production
  wiring) and THE INSTALLED UNIT IS UNPINNED -- I DIFFED THEM: the repo's unit
  file carries Environment=DA_MIDNIGHT_VERIFY_BIN at :51 and THE INSTALLED UNIT
  HAS NO SUCH LINE, only ExecStart at :47, so tomorrow's run RESOLVES ITS BINARY
  AT RUN TIME. The coordinator recommends NO PIN AND NO INSTALL and the
  reviewer's landings round verifies the launcher path first. AN UNPINNED UNIT IS
  NOT A DEFECT TONIGHT; IT IS A FACT THAT MUST BE KNOWN BEFORE THE NIGHT IT
  MATTERS. ONE MISMATCH THAT IS EXPECTED, WITH A DETAIL INSIDE IT THAT IS NOT
  OBVIOUS: the 09-02 mask's producer.module_sha256_prefix is d191695dcff0546e
  while the working da_blackout_mask.py is 15ea6dcb8c97c72d -- expected, because
  THE BINDING IS carrying_commit -- and the carrying_commit it records is
  3eabeeb, MY OWN ROUND-64 STATE-FILE COMMIT. The artifact of record for 09-02 is
  bound to a bookkeeping commit simply because that was the branch tip at 00:06Z:
  carrying_commit NAMES THE TREE, NOT THE AUTHOR, worth writing down before
  someone reads a MEM commit as provenance for a DA artifact. A GATE IS RED OFF
  THE UNIT PATH AND THE RESPONSE IS THE RIGHT ONE: v5_deploy_gates "host-load
  join" reads 36/38 because SA25 WAS RECYCLED BY SYSSTAT -- an input that AGED
  OUT, not a code change -- and RULE 15 RULES OUT A SKIP, so DA round 19 is A
  PROPOSAL ROW rather than a quiet exclusion: AN ABSENT INPUT IS A STATUS, and
  the rule that forbids the easy fix is doing exactly what it was written for.
  AND THE CALENDAR CAUGHT UP WITH A PHRASE: the withdrawn "earliest 09-03" is now
  DATED HISTORY -- 09-03 ARRIVED AND THE RUN DID NOT. I re-read all five
  occurrences and REWROTE THE THREE THAT SPOKE IN THE FUTURE TENSE (both
  phase4_run_* flags and the preconditions block beside the ruling), because
  their conditions have ALL HAPPENED: DE is at round 41, the reviewer has read
  through 40, section 5 is settled, the package is FORWARDED, and WHAT GATES THE
  RUN NOW IS THE USER'S ANSWER. The TWO occurrences inside dated entries I LEFT
  UNTOUCHED -- true when stamped, the rule I set in round 55 and the second time
  it has decided a case cleanly. DE ROUND 41 VERIFIED at 8479b67: 124 checks,
  DE40-R1/R2/R3 closed and driven, four mutants red by name; its review QUEUED
  behind the landings round. USER ITEMS OPEN: FIVE -- the 09-02 accrual, the
  Phase-2 winner, the content-liveness v2 freeze, the addendum v2 package, and
  the 09-04 run on the landed chain (A FACT WITH A RECOMMENDATION, not a request
  for a number). NOTHING RUNS. UNCHANGED: G is now the race's to count on the
  09-02 complement once the USER rules; the 011 family 12 of 24 with Q4 failing;
  the sequencing rule; R-419 revocable.
```
