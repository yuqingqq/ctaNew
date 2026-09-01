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
