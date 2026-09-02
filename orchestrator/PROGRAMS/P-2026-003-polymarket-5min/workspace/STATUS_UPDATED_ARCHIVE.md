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
