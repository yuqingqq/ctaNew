"""EXECUTION-WINDOW LEDGER AND WINDOW ORDERING CLASSIFIER FOR THE KEEP / PULL SELECTION (2026-11-18, session aea5bfb9) - v2 of the
fill-versus-cancel ordering audit. USER: "New review commit there, check it, we have some issues for simulator, need to correct these issues
first". The review of d1de7504 (reviews/automatic/20261118_d1de7504/report.md: P2 the removed-value headline mixes populations; P2 the
linked timing evidence is not retained) and its user-directed addendum (execution_windows_addendum.md), whose one next priority is: "Build
and preserve an evidence-backed execution-window ledger, then validate the ordering classifier on known-order lifecycle cases ... accept
separated windows, mark overlap / unknowns explicitly, and refuse wrong identities, contradictory windows or a delayed status timestamp
substituted for a match. Include cancellation / clock uncertainty and preserve original observations." Declared before any data is read.
The v1 audit (pf_keep_pull_timing_audit.py, d1de7504) and its receipt stay unchanged as provenance; this ledger supersedes its headline.
REVIEW OF c23e6f29 (reviews/automatic/20261118_c23e6f29/report.md; USER: "make sure you clear all issues"), corrected before any data is
read: (P2) contradictions and evidence-cap violations are aggregated per fill over ALL its candidate events - all / some / none /
unlinked - and reconcile to every fill and its gross (control f); (P2) the value bounds are EXACT joint extremes - one instant per event and
one identity per ambiguous fill, solved per component by enumerating identity assignments - and only a component with more than JCAP
assignments gets the outer relaxation, labelled in the receipt and the report with its fills and gross; (limit) K0-K3 are instrument checks,
not a venue-engine acceptance: K4 is now a computed reading and the known verdict carries it. The re-simulation of queues / fills /
eligibility under these windows is pf_keep_pull_scenarios.py.
REVIEW OF 1fd77004 (reviews/automatic/20261118_1fd77004/report.md; USER: "it still has time disagreements, and the issues are still there,
resolve these issues, all timestamp related issues, correct these ambiguity"), corrected before any further data is read - v3 of the ledger,
v2 of the known-order check (the earlier outputs stay as provenance):
 (P2 K5) the order path is an INTERVAL per UNIQUE order, both endpoints checked: activation in [signed, accepted] - place, the cancel effect in
   [0, confirmation - request]; each CONTRADICTED / COMPATIBLE-UNIDENTIFIED / CERTIFIED against its declared range. A late confirmation is
   an upper bound, never a measured slow effect; its source (HTTP answer / user stream), the core's CancelDone and throttles are kept apart.
 (P2 contradictions) "contradictory timing assumptions", split by cause at event and fill level: evidence_cap (a window beyond the REST cap)
   and leg_intersection (legs that cannot share one instant; mixed when a fill's candidates differ); the leg disagreements are decomposed
   by the sources that bind (WS leg / cell link, E2, REST second), different REST seconds and the WS spread.
 (next priority) a versioned known-order CASE LEDGER: every one of the 99 own legs (93 with an engine anchor, 6 without) traced to its
   order / trade / transaction identity, status stamps and their receipts, lifecycle, candidate rows and events on Nov 8's own tables, the
   engine anchor and its origin, and the ledger's OWN windows (K4 tested a proxy: anchor +- width); each miss carries its cause
   (event_proxy / own link by source / evidence cap / leg intersection / unresolved linkage) and the acceptance gate - no known fill certified
   avoided by a ledger window at any admissible clock offset - is computed per scenario (K6); the bot-to-venue clock offset is bounded from
   the own notices (K7).
 (timestamp semantics) the 11-09 audit established that REST executedAt is the CONFIRMED second, 0 / 1 / 10 s after Matched; the WS
   linker (pf_step1_timing: +-1.5 s), the E2 book-drop search (pf_step1_booktime: +-600 ms) and S1's imputed window (+-500 ms) are all
   centred on it as if it were the match second. S3 - the semantics windows - drops that premise: every leg's REST range is [S * 1000 -
   10 s, (S + 1) * 1000]; a WS leg's window is the hull over EVERY value-consistent pending increment in that range (the link's identity is
   by value, not signed); E2 and REST-only legs keep the REST range (the drop search could not reach the match). S1 / S2 stay as the
   program's declared ranges and measured envelopes - their contradictions are findings, not repaired.
SETTLEMENT-LAG CORRECTION (v5 / known v4; review 52c45d57, USER: "Can we fix this issue"):
UUIDv7 records ID creation, whose delay after matching can differ per event. K8 keeps one shared clock offset and a nonnegative lag for
EACH leg. The identity-linked first notice bounds creation from above; the known lifecycle bounds matching. A feasible constant total
delay is retained only as a conditional diagnostic. No sample-derived maximum is transferred to public trades. ST is upper-only by
default (creation after match and aligned clocks are still premises); an explicit finite lag is a labelled sensitivity assumption.
K5S and notice bounds use the full admissible known-lifecycle intervals. An upper-only ST gate is UNIDENTIFIED, not evidence of acceptance.
Earlier v4/v3 receipts remain provenance and must not be read as corrected timing evidence.
K6 BASIS (known v5; check of 9a114cc1): K6's declared gate stays at K7's offsets (its own premises). The K8 shared clock is narrower -
the first notice carrying each UUID and the placements bound it - but holds only if the UUID clock is the windows' clock (REST status /
WS book); its gate is a separate, conditional reading (_clk), like the zero-lag settlement offsets (_set). v4's "K6 GATE S3: PASS" was
read at the K8 clock under the K7 key; at K7's offsets S3 calls 3 known fills avoided (FAIL), as in v3.
K6 PER CLOCK DOMAIN (known v6; USER "Go" after the timing case audit, reviews/timing_case_audit_20261118/report.md): K7 bounds the REST
STATUS clock (floor seconds, notices); S3's three false avoidances bind on WS BOOK stamps of their own settlement, a clock K7 does not bound.
The gate of record is now 'K6 GATE {k} (per clock domain)': each window's start is split by the clock of the legs that set it (case_linkage:
WS / E2 legs on V_book, imputed legs and REST ranges on V_status; M50 by its anchor's origin) and each end is mapped with the lowest admissible
offset of its own clock - book ends with the leg's book-clock bound B from the bot's own frame receipts (pf_known_clock_census.py; an
unidentified bound fails closed), status ends with K7. 'K6 GATE {k}' keeps its v5 meaning (K7 on every end) as provenance, beside the K8
shared-clock and settlement-offset readings. Known v6 (OUT_K); v5 stays as provenance.

WHAT CHANGES FROM v1 (the review / addendum item each answers)
 1 Population (P2-1, item 4): v1 split ALL fills of the score's pulled orders and called it removed value. Now a cross-tab: replay ordering
   (REMOVED = the replay's fill at or after the pull's landing, KEPT = before it) x window class; fills / net / gross c per cell; shares on
   each row's own denominator.
 2 Retention (P2-2, item 3): a ledger of every candidate leg (day + its legs_timed2 row; its taker order = pf_txmap tk with tk_status; the raw
   observations executedAt second, first receipt, WS pending source time and how, E2 estimate - never overwritten), every event, every
   fill's candidate events and every order of the neutral population with its status; input, dependency and output hashes.
 3 Shared events (main review, item 3): one taker order is one matching operation, so its legs share ONE instant: their windows INTERSECT (an
   empty intersection is CONTRADICTORY - kept, never repaired). A fill whose anchor carries several events has candidate identities
   (multi_event): classified only where every candidate agrees. The value bounds move each event's instant once for all the orders it fills
   (v1 moved each order's copy independently); multi_event fills move independently over their candidates' windows.
 4 Status timestamps (addendum): REST executedAt is the Confirmed status second (11-09 fill-clock semantics audit: equal to the Confirmed
   stamp on 99 / 99 own legs; 0 / 1 / 10 s after Matched). It bounds the match from ABOVE only (evidence_bound (S + 1) * 1000); window_from
   refuses it as a point; no public source has established match-instant semantics (MATCH_SOURCES is empty).
 5 Cancellation (item 1): the classifier takes a cancel-effect window: CP = the replay's point (decision + 37 ms), CR = [decision, decision
   + 100 ms] (assumption range). Value bounds at cancel latency 0 / 37 / 100 ms. Activation (+58 ms) and the own-fill notice (+100 ms) stay
   the replay's configured points - recorded as assumed, not varied (the queue-aware scenario replay that re-simulates fills varies them).
 6 Ties (addendum): separated except at a touching endpoint = TIE; no venue tie-order rule is established and the replay's at-or-after
   convention is a modelling choice, not evidence.
 7 Known lifecycles (addendum): mode `known` checks the classifier and the windows on live's own 99 confirmed maker legs (K1-K4).

WINDOWS (venue clock; the replay axis is venue time; the client's feed delay is not modelled - an assumption)
 per leg            EV (evidence)                      S1 (the program's declared ranges)    S2 (measured envelopes, per day)
 WS (how > 0)       (-inf, (S + 1) * 1000]             [t_ms - 250, t_ms]                    [t_ms - h, t_ms + h]
 E2 (t_e2 finite)   (-inf, (S + 1) * 1000]             [t_e2 - 250, t_e2 + 250]              [t_e2 - h, t_e2 + h]
 imputed (S only)   (-inf, (S + 1) * 1000]             [1000 S - 500, 1000 S + 500] *        [1000 S + q01, 1000 S + q99] *
 S = executedAt second; t_ms = settlementsPending source time, linked to the leg by value (no signed identity); h = the day's p90 |E2 - WS|;
 q01 / q99 = the day's percentiles of WS - executedAt. * centred on the Confirmed status second: an assumption the known semantics contradict
 (the match can precede it by up to 10 s). Bound types: EV lo unknown, hi evidence_bound; S1 / S2 / S3 lo / hi assumption_range.
 S3 (the semantics windows, sem_windows): REST range R = [1000 S - REST_LAG, 1000 (S + 1)], REST_LAG = 10 s (the largest Confirmed - Matched
 lag on live's 99 own legs, one partial day). WS (how 1 / 2): the hull of [t - 250, t] over every pending increment at the leg's market and
 price whose amount matches its value (how 1 shares x p, how 2 its cell's - the linker's own rule and tolerance) with t in [R lo, R hi + 250]
 (the frame is at most 250 ms after the match, S1's premise), clipped to R; none -> R. E2 / imputed: R. Premises (stated, tested on the
 known legs by K6, never fitted): the match lies within REST_LAG of its Confirmed second; a WS leg's true increment, when visible, carries
 its value; book and REST stamps share one clock.
 per event: S1 / S2 = the intersection over its legs, capped by EV (admissible = assumption AND evidence): ok / truncated_by_evidence /
 contradictory_evidence / contradictory_legs. A contradictory event's instant is unknown: in the value bounds it moves over the whole axis.
 Every event row carries, per scenario and endpoint, its BOUND TYPE (evidence_bound / assumption_range / unknown / contradictory) and its
 BASIS (the observation it comes from), and a representative time with the rule that chose it (M50: anchor - 50 ms, an assumption).
 ledger_schema.json documents every column of every ledger table with its clock domain (venue clocks whose mutual alignment is not
 established; the collector's local receipt clock; the replay axis); an undocumented column refuses the output.
ORDERING classify(match window, cancel window): CANCEL_FIRST (cancel_hi < match_lo: the pull avoids the fill), MATCH_FIRST (match_hi <
 cancel_lo: too late), TIE (separated except at a touching endpoint), UNKNOWN (overlap with an open endpoint), AMBIGUOUS (bounded overlap),
 CONTRADICTORY (an empty match window).
VALUE: E = score improvement - exact STATES-null expectation (pf_keep_pull_clock_audit), quantities held at M50 (a stated limit here: no fill
 absent at M50 is discovered, no queue re-simulated - pf_keep_pull_scenarios.py re-simulates them). E_min / E_max = exact joint extremes:
 one instant per event in its admissible window and one identity per ambiguous fill (at the extremes this also covers any split of its
 shares between its candidates, the value being linear in them). Events tied by an ambiguous fill form a component solved by enumerating
 its identity assignments (each decouples into one-instant sweeps); above JCAP assignments the component is bounded by the outer relaxation
 (pairs moved independently) and the bounds are labelled OUTER with that component's fills and gross.

READINGS (computed; declared before any data is read)
 V0 CONTROLS, else no output: (a) collapsed to the replay's own times E = +234,279 c and the score improvement 1,169,671 c (to 1 c); (b) v1's
    own functions on v1's hull windows reproduce v1's committed S1 / S2 bounds (to 1 c) and ordering counts (exactly); (c) every M50 fill
    has a candidate event (v1 and v2 links agree); (d) every event leg's loader fields equal its legs_timed2 row and its event id the
    taker-order rule (tk, else -1 - row); (e) every cross-tab row's cells sum to the replay's removed / kept totals.
 V1 SIGN (event-consistent, admissible windows, cancel latency 37 ms) per scenario S1 / S2 / S3 / EV: EXCESS-POSITIVE-IDENTIFIED (E_min > 0),
    EXCESS-NEGATIVE-IDENTIFIED (E_max < 0) or SIGN-NOT-IDENTIFIED; the same predicate at 0 and 100 ms printed beside it.
 V2 EVIDENCE COVERAGE (EV x CP): the share of the replay's removed / kept value (gross) and of the pulled orders whose ordering the evidence
    identifies - printed. Under EV only MATCH_FIRST is identifiable (no evidence bounds the match from below).
 V3 CONTRADICTORY TIMING ASSUMPTIONS: per fill over ALL its candidate events (all / some / none / unlinked), of every fill and its gross,
    by link, day and pulled, per scenario S1 / S2 / S3: any contradiction, split by cause - evidence_cap (the scenario's window beyond the
    REST cap) and leg_intersection (its legs' windows cannot share one instant); mixed = all candidates contradictory with different causes;
    replay times after the evidence cap; events by status and class; the leg disagreements' binding source pairs - printed. Control (f):
    each split sums to every fill and its gross.
 KNOWN (mode known: the 11-09 fill-clock audit's 99 confirmed own maker legs, 86 orders, consumed Nov 8, hash-pinned; lifecycle on the
    bot's clock, REST / Matched / the research anchor Z on the venue clock, offset 0 where they meet - as the 11-09 audits did):
  K0 the 11-09 counts reproduce (REST floor-second non-overlap, Matched floor-second non-overlap), else no output.
  K1 CLASSIFIER-CONSISTENCY (a consistency check, not an acceptance: its match upper bound already includes the cancel ack): match
     [place, min(first fill notice, cancel ack)] vs cancel effect [cancel request, cancel ack]: YES iff no known fill is CANCEL_FIRST and no
     window is empty.
  K2 DELAYED-STATUS-REFUSED: YES iff REST executedAt used as the match second contradicts >= 1 known lifecycle while its evidence use
     (-inf, (S + 1) * 1000] contradicts none, and window_from refuses it as a point.
  K3 IDENTITY-CHECK-FIRES: the Matched status second against the order's lifecycle: YES iff the true identity contradicts none and >= 95% of
     200 within-run permutations contradict >= 1.
  K4 REPLAY-ANCHOR-VS-LIFECYCLE (v1's proxy, kept for continuity): the engine anchor Z with the S1 / S2 widths (+-250 / +-600 ms) on the
     93 legs WITH an anchor (the 6 without - ambiguous leg clocks - are named, not tested here): CONSISTENT iff no lifecycle is missed and no
     known fill is called CANCEL_FIRST under either width, else CONTRADICTED (counts, and whether one constant offset per run reconciles).
  K5 ORDER-PATH INTERVALS vs DECLARED RANGES (v2, per unique order): activation in [signed - place, accepted - place] (premise: live after
     signing and before its Accepted answer is received) against [25, 100] ms; cancel effect in [0, confirmation - request] (premise: before
     its first confirmation is received) against [0, 100] ms; each CONTRADICTED / COMPATIBLE-UNIDENTIFIED / CERTIFIED. CONTRADICTED iff some
     order's interval misses its range, else NOT-CONTRADICTED (the counts printed). A late confirmation never reads as a slow effect.
  K6 THE LEDGER'S WINDOWS vs KNOWN LIFECYCLES (all 99 legs via their candidate rows on Nov 8's own tables; S1 / S2 / S3 / EV as events_of
     builds them, and the replay's own point M50 = anchor - 50 ms): per leg compatible / missed before / missed after / contradictory (cap,
     legs) / ambiguous (candidates disagree), at offset 0 and at every admissible offset (K7); each miss with its cause. THE GATE per ledger
     scenario: PASS iff no known fill is classified CANCEL_FIRST against its actual cancel window at any admissible offset, else FAIL.
  K7 CLOCK OFFSET (venue status clock - bot clock), identity-safe from own notices: every notice is received after its status second starts
     (d >= stamp - receipt, Matched and Confirmed) and every match follows its placement and precedes its Matched second's end (d <= (M + 1)
     * 1000 - place); the interval, whether it contains 0, or EMPTY.
  KNOWN VERDICT: K0-K3 are instrument checks (the classifier and the refusals behave); K4 the replay anchor; K6 the gate per scenario; K5 the
     order-path ranges. The verdict names each; no reading here validates the venue engine.
SELFTEST (synthetic, no data): every class incl. TIE / UNKNOWN / CONTRADICTORY; window_from refusals; event intersection, evidence cap and
 statuses; link statuses; the event sweep = a dense grid of the shared instant, and on singletons v1's bounds; the collapsed control =
 pf_keep_pull_clock_audit.per_fill; K0-K4 on planted cases; an end-to-end rehearsal of run() on a synthetic source into a temporary dir;
 the S3 windows on planted increments (single, several, late link, cell value, out-of-range, E2 / REST-only); the cause split and the
 binding pairs; K5 on the review's positive and known-bad intervals and per order; K6 / K7 on planted lifecycles (a false CANCEL_FIRST at
 offset 0 and at one admissible offset only, an event proxy, an own E2 miss, a contradictory event, two disagreeing candidates); the case
 linkage on a planted day (candidate rows, the anchor rule and its origin). Mutations: pf_keep_pull_exec_windows_mutants.py.
LIMITS: consumed days; one declared rule; STATES is local selection; M50 quantities held (pf_keep_pull_scenarios.py re-simulates fills); S1 /
 S2 / S3 / CR are assumption ranges (S3's premises stated above); the REST-to-book clock alignment is not established (K7 bounds the bot's
 clock against the venue's status clock only); one partial day of known lifecycles, filled orders only (outcome-selected); only same-side
 legs at the fill's market join an event.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_exec_windows.py selftest | run [day] | known  (one day: its own output dir; V0 (a) / (b) against the
 committed per-day values - the clock audit's E and score, v1's per-day bounds; v1 kept its ordering counts pooled only)
"""
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402
import pf_keep_pull_timing_audit as V1                    # noqa: E402  v1 (d1de7504): its own functions run the regression control
import pf_settlement_times as SETT                        # noqa: E402  the settlementId's UUIDv7 time per leg (11-18)

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'keep_pull_exec_windows_v5_20261118'  # v5: event-specific creation lag; earlier receipts stay as provenance
OUT_K = Path(I.K.IO) / 'keep_pull_exec_windows_known_v6_20261118'   # v6: K6's gate of record per clock domain (v5, v4 stay as provenance)
OUT_CENSUS = Path(I.K.IO) / 'known_clock_census_v3_20261118'           # the book-clock bound per known leg (pf_known_clock_census.py v3)
SETTLEMENT_MODEL = 'shared_clock_event_specific_nonnegative_lag_v2'
FC = Path(I.K.IO) / 'fill_clock_20261109'                 # the 11-09 fill-clock audit (own legs, lifecycles, status stamps)
LIVE = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/live_vs_replay')
NOV8 = LIVE.parent / 'research' / 'step1' / '20261108'     # Nov 8's own research tables (legs_timed2 / legs_tk pinned by the 11-09 declaration)
REST_LAG = 10000.                                         # S3: the largest Confirmed - Matched lag on live's 99 own legs (0 / 1 / 10 s on 49 / 49 / 1)
WS_DELAY = 250.                                           # S3: the first frame showing a pending increment is at most 250 ms after the match (S1's premise)
VAL_TOL = (0.011, 0.01)                                   # the WS linker's value rule (pf_step1_timing): |increment - value| <= max(1.1% x value, $0.01)
SEM_CODES = ('ws_unique', 'ws_multi', 'rest_range_no_ws_candidate', 'rest_range_e2_unsupported', 'rest_range_rest_only')
SRC4 = ('ws_leg_link', 'ws_cell_link', 'e2_drop', 'rest_second')    # a leg's own timing source (how 1 / how 2 / E2 / the REST second)
CKEY = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']   # the 11-09 audit's REST leg content key
VERD = ('compatible', 'missed_before', 'missed_after', 'contradictory_evidence', 'contradictory_legs', 'ambiguous', 'unlinked')
CAUSES = ('event_proxy', 'own_ws_leg_link', 'own_ws_cell_link', 'own_e2_drop', 'own_rest_second', 'contradictory_evidence_cap', 'contradictory_leg_intersection',
          'unresolved_linkage')
KSC = ('S1', 'S2', 'S3', 'ST', 'EV', 'M50')               # K6: the ledger's windows and the replay's own point (anchor - 50 ms)
GATE = ('S1', 'S2', 'S3', 'ST', 'EV')                     # the acceptance gate: the ledger's windows
TS_SEM = {'exec_s': 'REST executedAt: venue status clock, whole second (floor) of the CONFIRMED status, 0 / 1 / 10 s after Matched on own legs; the match ends before (S + 1) * 1000 (evidence); S3 assumes it is at most REST_LAG earlier',
          't_ms': 'WS settlementsPending: the venue book clock (updateTimestampMs) of the first frame whose aggregated pending amount at the price rose by the leg\'s (how 1) or its cell\'s (how 2) value; linked by value within +-1.5 s of the Confirmed second (no signed identity)',
          't_e2': 'E2: the venue book clock of the displayed-size drop closest to the cell volume within [S * 1000 - 600, S * 1000 + 600) - a search centred on the Confirmed second; S3 does not use it',
          'first_recv_ns': 'the collector\'s local receipt clock (ns) of the first REST page carrying the leg; not aligned to the venue clocks',
          'place / signed / accepted': 'the bot (kernel) clock: Control Place, Exec Signed, the Accepted answer received',
          'cancel request': 'the bot (core) clock: Request::Cancel t',
          'cancel confirmation': 'the bot (kernel) clock: CancelPendingAudit - the first HTTP Canceled answer or user-stream cancel received (an upper bound on the effect)',
          'fill notice': 'the bot (kernel) clock: the first Matched / Confirmed fill notice received',
          'Matched / Confirmed stamps': 'the venue status clock, whole seconds (the notice\'s match_time per status)',
          'replay axis': 'venue time in ms after the order posting; the replay treats the book, WS and REST stamps as one clock (not established)'}
CC_SHA = '2f1a580bb6bb3689bf39b52ddbf5a7a443ea004e0d281274510f1193eb108a19'   # clock_cases.parquet (as pinned by pf_b1a_live_orders)
SHIFT = 50.                                               # the primary replay's trade shift (M50)
CANCEL_RANGE = (0., 100.)                                 # CR: the cancel takes effect between the decision and +100 ms (assumption)
ACT_RANGE = (25., 100.)                                   # the scenario replay's activation range (+58 ms the replay's point; assumption)
LATS = (0., KP.CANCEL, 100.)
MATCH_SOURCES = ()                                        # sources with established match-instant semantics: none
SCEN = ('S1', 'S2', 'S3', 'ST', 'EV')
CWIN = ('CP', 'CR')
ORD = ('CANCEL_FIRST', 'MATCH_FIRST', 'AMBIGUOUS', 'TIE', 'UNKNOWN', 'CONTRADICTORY')
EVC = ('WS', 'E2', 'imputed', 'mixed')
WST = ('ok', 'truncated_by_evidence', 'contradictory_evidence', 'contradictory_legs')
LINK = ('unlinked', 'unique_event', 'multi_event')
CST = ('unlinked', 'none', 'some', 'all')                 # of a fill's candidate events: none / some / all flagged
BOUND_TYPES = ('evidence_bound', 'assumption_range', 'unknown', 'contradictory')
BASIS = {('S1', 0): ('latest WS pending source time over the legs - 250 ms', 'earliest WS pending source time over the legs'),
         ('S1', 1): ('latest E2 book-drop estimate over the legs - 250 ms', 'earliest E2 book-drop estimate over the legs + 250 ms'),
         ('S1', 2): ('latest REST second over the legs - 500 ms (status-centred assumption)', 'earliest REST second over the legs + 500 ms (status-centred assumption)'),
         ('S1', 3): ('the latest start of its WS and E2 legs\' windows', 'the earliest end of its WS and E2 legs\' windows'),
         ('S2', 0): ('latest WS pending source time over the legs - h', 'earliest WS pending source time over the legs + h'),
         ('S2', 1): ('latest E2 book-drop estimate over the legs - h', 'earliest E2 book-drop estimate over the legs + h'),
         ('S2', 2): ('latest REST second over the legs + q01 (status-centred assumption)', 'earliest REST second over the legs + q99 (status-centred assumption)'),
         ('S2', 3): ('the latest start of its WS and E2 legs\' windows', 'the earliest end of its WS and E2 legs\' windows'),
         **{('ST', c): ('latest leg lower bound: mapped UUID has none unless an explicit lag scenario is supplied; unmapped legs retain S3 assumptions',
                        'the earliest end over its legs\' ST windows (t_set, else the S3 end)') for c in range(4)},
         **{('S3', c): ('the latest start over its legs\' S3 windows (a WS leg: its earliest value-consistent pending increment - 250 ms; otherwise its REST '
                        'Confirmed second - REST_LAG)', 'the earliest end over its legs\' S3 windows (a WS leg: its latest value-consistent increment; otherwise its '
                        'REST second end)') for c in range(4)}}
CAP_BASIS = 'REST executedAt (the Confirmed status) second end, earliest over the legs'
REP_RULE = 'M50: the replay anchor - 50 ms (the replay assumption, not a match time)'
CLOCKS = {'venue_ms': 'REST executedAt, WS settlementsPending source and book frame times: venue clocks, ms; their mutual alignment is NOT established',
          'local_ns': 'first_recv_ns: the collector local receipt clock, ns; not aligned to the venue clocks',
          'replay_ms': 'ms after the order posting time t0 on the venue axis (the replay axis)'}
SCHEMA = {
    'ledger_events': dict(day='UTC day', group='event id: the taker order (pf_txmap tk), else -1 - its legs_timed2 row (a leg without a taker identity)', m='market',
                          sides='number of sides its legs are on', side='its lowest side (0 bid / 1 ask of the Up book)', anc='the replay loader anchor, venue_ms',
                          legs='its legs in the candidate markets', cls='evidence class 0 WS / 1 E2 / 2 imputed / 3 mixed', ws_span='spread of its legs\' WS pending times, ms',
                          e='event index (ledger_fill_events.e)', rep_time='representative execution time, venue_ms (see rep_rule)', rep_rule='the rule that chose rep_time'),
    'ledger_legs': dict(mid='market', side='side (0 bid / 1 ask)', anc='the replay loader anchor, venue_ms', ptick='price tick', shares='shares', how='WS link: 0 none / 1 leg / 2 cell',
                        t_ms='WS settlementsPending source time linked by value (no signed identity), venue_ms', t_e2='E2 book-drop estimate (inferred), venue_ms',
                        exec_s='REST executedAt second = the Confirmed status, venue clock s (an upper bound on the match)', row_id='its legs_timed2 row (stable identity with the file hash)',
                        group='its event (ledger_events.group)', first_recv_ns='first receipt of the REST leg, local_ns', price_executed='REST execution price',
                        signer='maker signer address', taker_index='taker index in the match', taker_quote='taker quote flag', tk='taker-order id (pf_txmap), -1 none',
                        tk_status='pf_txmap status 0 mapped / 1 absent / 2 count mismatch / 3 ambiguous / 4 no taker hash / 5 refined by signer', day='UTC day',
                        S3_llo='the leg\'s S3 window start, venue_ms', S3_lhi='the leg\'s S3 window end, venue_ms', S3_code='its S3 basis (SEM_CODES)',
                        S3_ncand='value-consistent pending increments in its REST range (a WS leg)',
                        link_late='its linked WS increment is later than its REST cap + 250 ms (the link contradicts S3\'s premises)',
                        t_set_ms='its settlement time (the settlementId\'s UUIDv7 ms, pf_settlement_times), venue settlement clock', set_status='its settlement status (pf_settlement_times.STATUS)',
                        ST_llo='ST lower endpoint: -inf (unidentified) for mapped UUID by default; finite only for an explicit lag scenario or S3 fallback, venue_ms', ST_lhi='its ST window end (t_set, else its S3 end)',
                        ST_code='0 its settlement time, 1 its S3 window (no settlement time)'),
    'ledger_fill_events': dict(fid='fill (ledger_fills.fid)', day='UTC day', group='candidate event (ledger_events.group)', e='candidate event index'),
    'ledger_fills': dict(fid='fill id', day='UTC day', oid='order (ledger_orders.oid)', m='market', side='side', p='order price', t0='order posting time, venue_ms',
                         t_replay='the replay M50 fill time, replay_ms', anchor='its trade anchor, venue_ms', q='filled shares (M50)', value='fill value at END, c',
                         src='the replay source code 0 WS / 1 E2 / 2 imputed / 3 mixed', pulled='the declared score executed a pull on its order',
                         T='the score decision step (99 never)', landing37='the score pull landing at 37 ms, replay_ms', replay_removed='removed by the pull in the replay',
                         link='unlinked / unique_event / multi_event', n_events='candidate events', event='its event index if unique, else -1',
                         activation_assumed='the replay activation (configured point), ms', notice_assumed='the replay own-fill notice delay (configured point), ms'),
    'ledger_orders': dict(day='UTC day', oid='order', m='market', side='side', t='posting time, venue_ms', status='ledger status (exclusions are statuses)',
                          T='the score decision step', pulled='the score executed a pull', endj='the client last decision step + 1', m50_fills='its M50 fills',
                          m50_value='its M50 fill value at END, c')}
SCHEMA_PREFIX = {'ledger_events': {'S1_': 'S1 window (the program declared range AND the REST cap), venue_ms: lo / hi (capped) / rhi (before the cap) / st (WST) / lo_type, hi_type '
                                          '(BOUND_TYPES) / lo_basis, hi_basis (the observation each endpoint comes from)',
                                   'S2_': 'S2 window (measured envelopes AND the REST cap): the same fields as S1_',
                                   'S3_': 'S3 window (the semantics windows: REST range, value-consistent WS increments, AND the REST cap): the same fields as S1_',
                                   'ST_': 'ST window (upper-only UUID by default, or explicit lag assumption; S3 fallback; AND the REST cap): same fields as S1_',
                                   'EV_': 'EV window (evidence only), venue_ms: lo (-inf: unknown) / hi (the REST cap) / st / types / bases'},
                 'ledger_fills': {'cls_': 'ordering class (ORD) per scenario and cancel window (CP / CR), -1 not pulled',
                                  'contra_': 'per-fill status over ALL its candidate events (CST): S1 / S2 / S3 contradictory timing assumptions (any; _cap the '
                                             'evidence cap; _legs the leg intersection), replay time after the evidence cap'},
                 'ledger_orders': {'decision_': 'decision evaluability per scenario: not_pulled / no_m50_fill / evaluable / not_evaluable / not_eligible'}}


def schema_gaps(tables):
    """Columns of the ledger tables ({table: columns}) without a description in SCHEMA / SCHEMA_PREFIX."""
    return [f'{t}.{c}' for t, cols in tables.items() for c in cols if c not in SCHEMA.get(t, {}) and not any(c.startswith(x) for x in SCHEMA_PREFIX.get(t, {}))]


def schema_doc(tables):
    """ledger_schema.json: every column described, the bound-type and status vocabularies, the endpoint bases, the clocks, the rule of rep_time."""
    def d(t, c):
        return SCHEMA.get(t, {}).get(c) or next(v for x, v in SCHEMA_PREFIX.get(t, {}).items() if c.startswith(x))
    return dict(tables={t: {c: d(t, c) for c in cols} for t, cols in tables.items()}, bound_types=list(BOUND_TYPES), window_status=list(WST), contradiction_status=list(CST),
                ordering=list(ORD), link=list(LINK), evidence_class=list(EVC), basis={f'{k}|{EVC[c]}': list(v) for (k, c), v in BASIS.items()}, cap_basis=CAP_BASIS,
                rep_rule=REP_RULE, clocks=CLOCKS, timestamp_semantics=TS_SEM, settlement_model=SETTLEMENT_MODEL,
                settlement_premises=['UUID created at/after matching', 'UUID and book/status clocks aligned'], s3=dict(codes=list(SEM_CODES), rest_lag_ms=REST_LAG, ws_delay_ms=WS_DELAY,
                                                                                       value_tolerance=list(VAL_TOL)))
JCAP = 256                                                # identity assignments enumerated per component; above it the labelled outer relaxation
LEG_COLS = ['mid', 'side', 'anc', 'ptick', 'shares', 'how', 't_ms', 't_e2', 'exec_s', 'row_id', 'group']
V1_COLS = ['mid', 'side', 'anc', 'ptick', 'how', 't_ms', 't_e2', 'exec_s']
RAW_COLS = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'signer', 'taker_index', 'taker_quote', 't_ms', 'how', 't_e2']
RECON = ('mid', 'side', 'ptick', 'shares', 'exec_s', 'how', 't_ms', 't_e2')


def classify(m_lo, m_hi, c_lo, c_hi):
    """Ordering class (index into ORD) of match windows [m_lo, m_hi] against cancel-effect windows [c_lo, c_hi] on one clock: closed
    intervals, +-inf an open endpoint, m_lo > m_hi an empty (contradictory) window."""
    m_lo, m_hi, c_lo, c_hi = np.broadcast_arrays(*(np.asarray(x, float) for x in (m_lo, m_hi, c_lo, c_hi)))
    C.require(bool(np.all(c_lo <= c_hi)), 'an empty cancel window')
    cf = c_hi < m_lo; mf = m_hi < c_lo; tie = ~cf & ~mf & ((m_hi == c_lo) | (c_hi == m_lo))
    opn = ~(np.isfinite(m_lo) & np.isfinite(m_hi) & np.isfinite(c_lo) & np.isfinite(c_hi))
    return np.where(m_lo > m_hi, 5, np.where(cf, 0, np.where(mf, 1, np.where(tie, 3, np.where(opn, 4, 2)))))


def window_from(kind, x, point=False):
    """A match window (lo, hi) from one typed observation, vectorized over x. 'rest_confirmed_s': the REST executedAt second (the Confirmed
    status; the match precedes the end of that second); 'matched_status_s': own orders' Matched status second (floor reading - an
    assumption); 'lifecycle': x = (place, upper), own orders' causal interval; 'match_instant': needs a source in MATCH_SOURCES. A status
    asked for a point is refused: a delayed status time is never a match time."""
    if kind in ('rest_confirmed_s', 'matched_status_s') and point:
        raise ValueError(f'REFUSED: {kind} is a status timestamp, not the matching instant')
    if kind == 'rest_confirmed_s':
        x = np.asarray(x, float); return np.full(x.shape, -np.inf), (x + 1.) * 1000.
    if kind == 'matched_status_s':
        x = np.asarray(x, float); return x * 1000., x * 1000. + 999.999
    if kind == 'lifecycle':
        lo, hi = (np.asarray(v, float) for v in x); C.require(bool(np.all(lo <= hi)), 'reversed causal bounds'); return lo, hi
    if kind == 'match_instant':
        raise ValueError(f'REFUSED: no source has established match-instant semantics (MATCH_SOURCES = {MATCH_SOURCES})')
    raise ValueError(f'unknown observation kind {kind}')


def leg_windows(L):
    """Per leg (columns how, t_ms, t_e2, exec_s, its day's h, q01, q99 and its S3 window S3_llo / S3_lhi from sem_windows): class (0 WS, 1 E2,
    2 imputed) and {scenario: (lo, hi)}."""
    C.require('S3_llo' in L and 'S3_lhi' in L and 'ST_llo' in L and 'ST_lhi' in L, 'legs without their S3 / ST windows (sem_windows / st_windows)')
    ws = (L.how.values > 0) & np.isfinite(L.t_ms.values); e2 = ~ws & np.isfinite(L.t_e2.values); S = L.exec_s.values * 1000.
    cls = np.where(ws, 0, np.where(e2, 1, 2)); x = np.where(ws, L.t_ms.values, np.where(e2, L.t_e2.values, S))
    h, q01, q99 = L.h.values, L.q01.values, L.q99.values
    return cls, dict(S1=(np.where(cls < 2, x - 250., x - 500.), np.where(ws, x, np.where(e2, x + 250., x + 500.))),
                     S2=(np.where(cls < 2, x - h, x + q01), np.where(cls < 2, x + h, x + q99)), S3=(L.S3_llo.values.astype(float), L.S3_lhi.values.astype(float)),
                     ST=(L.ST_llo.values.astype(float), L.ST_lhi.values.astype(float)),
                     EV=window_from('rest_confirmed_s', L.exec_s.values))


def rest_range(exec_s):
    """S3's REST range of a leg: [S * 1000 - REST_LAG, (S + 1) * 1000] (the Confirmed second, at most REST_LAG after the match)."""
    S = np.asarray(exec_s, float) * 1000.; return S - REST_LAG, S + 1000.


def sem_windows(R, P, chunk=4_000_000):
    """S3 per leg. R: the day's legs (mid, exec_s, side, ptick, shares, how, t_ms, t_e2), P: its pending increments (mid, t_ms, ptick, inc). A
    WS leg (how 1 / 2) gets the hull of [t - WS_DELAY, t] over EVERY increment at its market and price (either side, as the linker) whose
    amount matches its value - how 1 shares x p, how 2 its cell's (the linker's cell pass: its how-2 legs at one market, second, side and
    price) - within the linker's tolerance and whose time is in [R lo, R hi + WS_DELAY], clipped to its REST range R; a WS leg without one,
    an E2 leg and a REST-only leg get R; candidate pairs are expanded `chunk` at a time. -> DataFrame in R's order: S3_llo, S3_lhi, S3_code
    (SEM_CODES), S3_ncand, link_late."""
    n = len(R); how = R.how.values.astype(int); S = R.exec_s.values.astype(float); pt = R.ptick.values.astype(float)
    lo0, hi0 = rest_range(S); v = R.shares.values.astype(float) * pt / 100.; c2 = how == 2
    if c2.any():
        g = pd.DataFrame(dict(m=R.mid.values[c2], s=S[c2], d=R.side.values[c2], p=pt[c2], q=R.shares.values[c2].astype(float)))
        v[c2] = g.groupby(['m', 's', 'd', 'p']).q.transform('sum').values * pt[c2] / 100.
    lo, hi = lo0.copy(), hi0.copy(); nc = np.zeros(n, np.int64)
    code = np.where(how > 0, 2, np.where(np.isfinite(R.t_e2.values.astype(float)), 3, 4)); q = np.flatnonzero(how > 0)
    if len(q) and len(P):
        kp = pd.MultiIndex.from_arrays([P.mid.values.astype(np.int64), P.ptick.values.astype(np.int64)]); u = kp.unique()
        ip = u.get_indexer(kp); iq = u.get_indexer(pd.MultiIndex.from_arrays([R.mid.values[q].astype(np.int64), pt[q].astype(np.int64)]))
        t = P.t_ms.values.astype(float); base = min(float(t.min()), float(lo0[q].min())) - 1.; span = 2. ** 30
        C.require(max(float(t.max()), float(hi0[q].max()) + WS_DELAY) - base < span and len(u) * span < 2. ** 52, 'pending times or keys exceed the exact key range')
        kk = ip * span + (t - base); o = np.argsort(kk, kind='stable'); kk, tp, inc = kk[o], t[o], P.inc.values.astype(float)[o]; has = iq >= 0
        i0 = np.searchsorted(kk, np.where(has, iq * span + (lo0[q] - base), np.inf), side='left')
        i1 = np.searchsorted(kk, np.where(has, iq * span + (hi0[q] + WS_DELAY - base), -np.inf), side='right'); cnt = np.maximum(i1 - i0, 0)
        nq = np.zeros(len(q), np.int64); mn = np.full(len(q), np.inf); mx = np.full(len(q), -np.inf); cs = np.cumsum(cnt); a_ = 0
        while a_ < len(q) and cnt.sum():                                       # candidate pairs in chunks of <= `chunk` (the memory guard)
            b_ = max(int(np.searchsorted(cs, (cs[a_ - 1] if a_ else 0) + chunk, side='right')), a_ + 1); c_ = cnt[a_:b_]
            qi = a_ + np.repeat(np.arange(b_ - a_), c_); pi = np.arange(int(c_.sum())) - np.repeat(np.cumsum(c_) - c_, c_) + np.repeat(i0[a_:b_], c_)
            vq = v[q][qi]; ok = np.abs(inc[pi] - vq) <= np.maximum(VAL_TOL[0] * vq, VAL_TOL[1]); qi, tt = qi[ok], tp[pi][ok]
            nq += np.bincount(qi, minlength=len(q)); np.minimum.at(mn, qi, tt); np.maximum.at(mx, qi, tt); a_ = b_; del pi, vq, ok
            C.guard('S3 candidate increments')
        g_ = nq > 0; j = q[g_]; nc[q] = nq
        lo[j] = np.maximum(mn[g_] - WS_DELAY, lo0[j]); hi[j] = np.minimum(mx[g_], hi0[j]); code[j] = np.where(nq[g_] == 1, 0, 1)
    late = (how > 0) & (R.t_ms.values.astype(float) > hi0 + WS_DELAY)
    return pd.DataFrame(dict(S3_llo=lo, S3_lhi=hi, S3_code=code, S3_ncand=nc, link_late=late))


def day_sem(day, rows):
    """The day's S3 leg windows (rows: day_rows, in legs_timed2 order) from its pending increments -> (frame, input hash)."""
    p = Path(f'{I.IO}/step1/{day}') / 'pend_inc.parquet'
    return sem_windows(rows, pd.read_parquet(p, columns=['mid', 't_ms', 'ptick', 'inc'])), {str(p): C.sha(p)}


def st_windows(t_set, s3lo, s3hi, lag):
    """UUID creation gives an upper bound only, conditional on creation after matching and aligned venue clocks.
    A finite lag is an explicitly supplied scenario assumption, never inferred from K8. Missing IDs retain S3's labelled assumptions.
    -> (lo, hi, code 0 UUID / 1 S3). No mapped leg has a finite lower bound by default."""
    C.require(lag is None or (np.isfinite(lag) and lag >= 0.), 'ST lag must be an explicit finite nonnegative scenario assumption or None')
    t = np.asarray(t_set, float); f = np.isfinite(t)
    lower = np.full(t.shape, -np.inf) if lag is None else t - lag
    return np.where(f, lower, s3lo), np.where(f, t, s3hi), np.where(f, 0, 1).astype(np.int8)


def set_of(day):
    """The day's settlement times (pf_settlement_times, row-aligned with legs_timed2), its controls passed and its hash as its receipt states."""
    d = SETT.OUT / day; r = json.loads((d / 'settlement.json').read_text()); C.require(C.sha(d / 'legs_set.parquet') == r['outputs']['legs_set.parquet'], 'settlement times changed since their receipt')
    C.require(all(r['R']['controls'].values()), 'settlement times without passed controls')
    return pd.read_parquet(d / 'legs_set.parquet', columns=['t_set_ms', 'set_status']), {str(d / 'legs_set.parquet'): r['outputs']['legs_set.parquet']}


def known_lag(out=None):
    """Read the corrected known receipt; no sampled per-leg bound is transferable to public trades.
    Refuse the old common-offset receipt instead of silently reusing its 259 ms bound. -> (None, receipt hash)."""
    p = Path(out or OUT_K) / 'known.json'; j = json.loads(p.read_text()); k8 = j['R'].get('K8') or {}
    C.require(k8.get('model') == SETTLEMENT_MODEL and bool(k8.get('consistent')) and k8.get('lag_bound_ms') is None,
              'legacy or inconsistent K8 receipt: no transferable ST lag bound; rebuild the known ledger')
    return None, {str(p): C.sha(p)}


def settlement_diag(R_, t, lag, env):
    """On the day's legs (R_: day_rows + their S3 / ST windows) with a settlement time t: the REST cap and the Confirmed lag; the WS pending delay
    t_ms - t_set per link type and E2-minus-creation differences; window overlap with ST. These differences mix clock offset and
    event delays; they do not independently establish clock alignment, match-time error, or link correctness."""
    m = np.isfinite(t); how = R_.how.values; tms = R_.t_ms.values.astype(float); te2 = R_.t_e2.values.astype(float); S = R_.exec_s.values.astype(float) * 1000.
    pct = lambda x: {f'p{q}': float(np.percentile(x, q)) for q in (1, 5, 50, 95, 99)} if len(x) else {}
    out = dict(legs=int(len(R_)), with_t_set=int(m.sum()), rest_cap_violations=int((t[m] > S[m] + 1000.).sum()), rest_second_minus_t_set_ms=pct((S - t)[m]))
    for nm, mk in (('ws_leg_link', how == 1), ('ws_cell_link', how == 2)):
        x = (tms - t)[m & mk & np.isfinite(tms)]
        out[f'{nm}_minus_t_set_ms'] = dict(n=int(len(x)), **pct(x), below_minus_lag=float((x < -lag).mean()) if len(x) and lag is not None else None, over_250=float((x > 250.).mean()) if len(x) else None,
                                           over_1000=float((x > 1000.).mean()) if len(x) else None)
    x = (te2 - t)[m & (how == 0) & np.isfinite(te2)]; out['e2_minus_t_set_ms'] = dict(n=int(len(x)), **pct(x), abs_over_600=float((np.abs(x) > 600.).mean()) if len(x) else None)
    _, W = leg_windows(R_.assign(h=env['h'], q01=env['q01'], q99=env['q99'])); hold = {}
    for k in ('S1', 'S2', 'S3'):
        lo, hi = W[k]; hold[k] = dict(holds_t_set=float(((lo <= t) & (t <= hi))[m].mean()), meets_st=float(((lo <= t) & (hi >= R_.ST_llo.values))[m].mean()))
    out['windows_vs_settlement'] = hold
    return out


def events_of(L):
    """Per event (day, group): market, side(s), the loader's anchor, legs, class, WS span and per scenario the intersection of its legs'
    windows capped by the evidence, with its status (WST)."""
    cls, W = leg_windows(L)
    g = pd.DataFrame(dict(day=L.day.values, group=L.group.values, m=L.mid.values, side=L.side.values, anc=L.anc.values, cls=cls,
                          wst=np.where(cls == 0, L.t_ms.values, np.nan), EV_hi=W['EV'][1], S1_lo=W['S1'][0], S1_hi=W['S1'][1],
                          S2_lo=W['S2'][0], S2_hi=W['S2'][1], S3_lo=W['S3'][0], S3_hi=W['S3'][1], ST_lo=W['ST'][0], ST_hi=W['ST'][1]))
    E = g.groupby(['day', 'group'], sort=True).agg(m=('m', 'first'), sides=('side', 'nunique'), side=('side', 'min'), anc=('anc', 'min'),
                                                   anc_max=('anc', 'max'), legs=('cls', 'size'), cmin=('cls', 'min'), cmax=('cls', 'max'),
                                                   ws_lo=('wst', 'min'), ws_hi=('wst', 'max'), EV_hi=('EV_hi', 'min'), S1_lo=('S1_lo', 'max'),
                                                   S1_hi=('S1_hi', 'min'), S2_lo=('S2_lo', 'max'), S2_hi=('S2_hi', 'min'), S3_lo=('S3_lo', 'max'),
                                                   S3_hi=('S3_hi', 'min'), ST_lo=('ST_lo', 'max'), ST_hi=('ST_hi', 'min')).reset_index()
    C.require(bool((E.anc == E.anc_max).all()), 'an event with two loader anchors')
    E['cls'] = np.where(E.cmin == E.cmax, E.cmin, 3); E['ws_span'] = E.ws_hi - E.ws_lo; E['EV_lo'] = -np.inf; E['EV_st'] = 0
    for k in ('S1', 'S2', 'S3', 'ST'):
        legs_empty = (E[f'{k}_lo'] > E[f'{k}_hi']).values; cap = E.EV_hi.values < E[f'{k}_hi'].values
        E[f'{k}_rhi'] = E[f'{k}_hi']; E[f'{k}_hi'] = np.minimum(E[f'{k}_hi'], E.EV_hi)
        E[f'{k}_st'] = np.where(legs_empty, 3, np.where(E[f'{k}_lo'].values > E[f'{k}_hi'].values, 2, np.where(cap, 1, 0)))
    c_ = E.cls.values.astype(int)
    for k in ('S1', 'S2', 'S3', 'ST'):
        st = E[f'{k}_st'].values
        E[f'{k}_lo_type'] = np.where(st >= 2, 'contradictory', np.where(np.isneginf(E[f'{k}_lo']), 'unknown', 'assumption_range'))
        E[f'{k}_hi_type'] = np.where(st >= 2, 'contradictory', np.where(st == 1, 'evidence_bound', 'assumption_range'))
        E[f'{k}_lo_basis'] = np.array([BASIS[k, x][0] for x in c_], dtype=object)
        E[f'{k}_hi_basis'] = np.where(st == 1, CAP_BASIS, np.array([BASIS[k, x][1] for x in c_], dtype=object))
    E['EV_lo_type'], E['EV_hi_type'] = 'unknown', 'evidence_bound'
    E['EV_lo_basis'], E['EV_hi_basis'] = 'none: no public evidence bounds the match from below', CAP_BASIS
    E['rep_time'], E['rep_rule'] = E.anc - SHIFT, REP_RULE
    return E.drop(columns=['anc_max', 'cmin', 'cmax', 'ws_lo', 'ws_hi'])


def link(Fd, LT):
    """Candidate events of each fill: the legs of its market and side whose anchor equals the fill's (anchor = replay time + 50 ms + posting
    time, to 0.01 ms, as v1) and whose price can fill the order (at or through it) name the events (taker orders). -> pairs (fid, group) and
    every leg of those events in LT (all prices: one instant)."""
    U = LT[['mid', 'side', 'anc']].drop_duplicates().sort_values('anc'); U = U.assign(mid=U.mid.astype(np.int64), side=U.side.astype(np.int64))
    x = Fd.assign(m=Fd.m.astype(np.int64), side=Fd.side.astype(np.int64)).sort_values('a')
    x = pd.merge_asof(x, U.rename(columns={'mid': 'm'}), left_on='a', right_on='anc', by=['m', 'side'], direction='nearest', tolerance=0.01)
    y = x.dropna(subset=['anc']).merge(LT.assign(m=LT.mid.astype(np.int64), side=LT.side.astype(np.int64))[['m', 'side', 'anc', 'ptick', 'group']],
                                       on=['m', 'side', 'anc'])
    y = y[np.where(y.side.values == 0, y.ptick.values <= y.p.values, y.ptick.values >= y.p.values)]
    pairs = y[['fid', 'group']].drop_duplicates().reset_index(drop=True)
    return pairs, LT[LT.group.isin(pairs.group.unique())]


def day_env(day):
    """The day's measured envelopes (v1's) and the hashes of every file this audit reads for the day."""
    S1d = Path(f'{I.IO}/step1/{day}'); b = json.loads((S1d / 'booktime.json').read_text()); t = json.loads((S1d / 'timing.json').read_text())
    files = (S1d / 'legs_timed2.parquet', S1d / 'legs_tk.parquet', S1d / 'booktime.json', S1d / 'timing.json', Path(C.V.E.OUTD) / f'day_{day}_imp.json')
    return dict(h=float(b['E2']['p90']), q01=float(t['offset_ms_pct']['p01']), q99=float(t['offset_ms_pct']['p99']), inputs={str(p): C.sha(p) for p in files})


def day_rows(day):
    """The day's legs_timed2 rows (the loader's row_id is the row position) with each leg's taker order (pf_txmap: tk, tk_status)."""
    S1d = Path(f'{I.IO}/step1/{day}'); L = pd.read_parquet(S1d / 'legs_timed2.parquet', columns=RAW_COLS)
    tk = pd.read_parquet(S1d / 'legs_tk.parquet', columns=['tk', 'tk_status']); C.require(len(tk) == len(L), f'{day}: legs_tk not row-aligned')
    return pd.concat([L.reset_index(drop=True), tk.reset_index(drop=True)], axis=1)


def link_day(day, Fd, src, env, rows):
    """One day's fills, market batch by market batch through the replay's own loader -> (v1's hull per fill, candidate pairs, the events'
    legs with their legs_timed2 identity and raw observations, identity mismatches)."""
    offset = src.offset(day); H1, P, LG = [], [], []; markets = sorted(set(Fd.m.astype(int)))
    for k in range(0, len(markets), 24):
        batch = markets[k:k + 24]; LT = src.load(day, set(batch), offset)['LT'][LEG_COLS]; f = Fd[Fd.m.isin(batch)]
        H1.append(V1.link(f, LT[V1_COLS], env['h'], env['q01'], env['q99'])); p, lg = link(f, LT); P.append(p); LG.append(lg)
        del LT; C.guard('exec-windows link')
    H1, P = pd.concat(H1), pd.concat(P, ignore_index=True); LG = pd.concat(LG, ignore_index=True).drop_duplicates('row_id').reset_index(drop=True)
    R = rows.iloc[LG.row_id.values.astype(np.int64)].reset_index(drop=True); bad = 0
    for c in RECON:
        a, b = LG[c].values.astype(float), R[c].values.astype(float); bad += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    tk = R.tk.values.astype(np.int64); bad += int((LG.group.values.astype(np.int64) != np.where(tk >= 0, tk, -1 - LG.row_id.values.astype(np.int64))).sum())
    LG = pd.concat([LG, R[[c for c in RAW_COLS if c not in LEG_COLS] + ['tk', 'tk_status']]], axis=1)
    return H1, P, LG, bad


def contrib(tau, F, CW, lat=KP.CANCEL):
    """A fill's contribution to E if its match time were tau (ms after posting) and every cancel landed lat ms after its decision. F: per
    fill val, pulled, T (the score's step), s (stratum), endj, lend (structural landing, inf if none). As v1 at lat = 37 ms."""
    La = KP.landing(np.arange(1, KP.NSTEP + 1), lat); top = np.minimum(np.searchsorted(La, tau, side='right'), F['endj'] - 1)
    W = np.where(top >= KP.J0, CW[F['s'], np.maximum(top, 0)] - CW[F['s'], KP.J0 - 1], 0.)
    c = -F['val'] * (np.where(F['pulled'] & (tau >= KP.landing(np.clip(F['T'], 1, KP.NSTEP), lat)), 1., 0.) - W)
    return np.where((tau < KP.PLACE) | (tau > KP.H) | (tau >= F['lend']), 0., c)


def sweep(ev, wlo, whi, d, F, CW, lat, chunk=4000):
    """Exact min / max over ONE shared instant per event of the sum of its fills' contributions. ev: event index per fill; wlo / whi: the
    event's window on its own axis (+-inf allowed); d: the fill's posting time on that axis (tau_fill = tau_event - d). The sum is a
    right-continuous step function: evaluated at the window's start and after every breakpoint in (wlo, whi], equal instants aggregated."""
    nE = len(wlo); bp = np.r_[KP.landing(np.arange(1, KP.NSTEP + 1), lat), KP.PLACE, KP.H, np.nextafter(KP.H, np.inf)]
    s0 = wlo[ev] - d; fin = np.isfinite(s0); g0 = np.zeros(nE)
    np.add.at(g0, ev, np.where(fin, contrib(np.where(fin, s0, 0.), F, CW, lat), 0.)); mn, mx = g0.copy(), g0.copy()
    o = np.argsort(ev, kind='stable'); cuts = np.r_[0, np.flatnonzero(np.diff(ev[o])) + 1, len(o)]; i = 0
    while i < len(cuts) - 1:
        j = i + 1
        while j < len(cuts) - 1 and cuts[j + 1] - cuts[i] <= chunk:
            j += 1
        idx = o[cuts[i]:cuts[j]]; i = j; e_ = ev[idx]
        B = np.sort(np.c_[np.broadcast_to(bp, (len(idx), len(bp))), F['lend'][idx]], axis=1); Bev = d[idx, None] + B
        dup = np.zeros(B.shape, bool); dup[:, 1:] = B[:, 1:] == B[:, :-1]                     # one jump per breakpoint of a fill (lend can equal a landing)
        fi, bi = np.nonzero(np.isfinite(B) & ~dup & (Bev > wlo[e_, None]) & (Bev <= whi[e_, None]))
        if not len(fi):
            continue
        rel = B[fi, bi]; sub = {k: v[idx][fi] for k, v in F.items()}
        dl = contrib(rel, sub, CW, lat) - contrib(np.nextafter(rel, -np.inf), sub, CW, lat)
        ee, bb = e_[fi], Bev[fi, bi]; q = np.lexsort((bb, ee)); ee, bb, dl = ee[q], bb[q], dl[q]
        st = np.flatnonzero(np.r_[True, (np.diff(ee) != 0) | (np.diff(bb) != 0)]); du = np.add.reduceat(dl, st); eu = ee[st]
        cs = np.cumsum(du); s_ = np.flatnonzero(np.r_[True, np.diff(eu) != 0]); base = np.r_[0., cs[s_[1:] - 1]]
        v = g0[eu] + cs - np.repeat(base, np.diff(np.r_[s_, len(eu)])); E_ = eu[s_]
        mn[E_] = np.minimum(mn[E_], np.minimum.reduceat(v, s_)); mx[E_] = np.maximum(mx[E_], np.maximum.reduceat(v, s_))
        C.guard('exec-windows sweep')
    return mn, mx


def event_axis_windows(EVT, k):
    """Per event, scenario k's window on the event's own axis (its anchor = 0); a contradictory event's instant is unknown: (-inf, inf)."""
    bad = EVT[f'{k}_st'].values >= 2; anc = EVT.anc.values
    return np.where(bad, -np.inf, EVT[f'{k}_lo'].values - anc), np.where(bad, np.inf, EVT[f'{k}_hi'].values - anc)


def struct_landing(g, lat):
    """A structural cancel's effect on fills at cancel latency lat - the kernel's order of a grid instant (its trades first, then its decisions:
    at zero latency it acts just after g) and the pending rule (decided before the +58 ms activation: lands at activation + lat); inf where
    none. Equals the committed landing at 37 ms (checked) and pf_keep_pull_scenarios.eff_landing."""
    g = np.asarray(g, float); x = np.where(g < KP.PLACE, np.maximum(g + lat, KP.PLACE + lat), g + lat)
    return np.where(np.isnan(g), np.inf, np.where(x > g, x, np.nextafter(g, np.inf)))


def at_latency(F, gs, lat):
    """The per-fill inputs at cancel latency lat: every cancel moves - the structural landing too (fills held at M50)."""
    return dict(F, lend=struct_landing(gs, lat))


def candidate_status(fid, flag, n):
    """Per fill, over its candidate events (pairs fid / flag): 0 unlinked (no candidate), 1 none, 2 some, 3 all of them flagged (index into CST)."""
    k = np.bincount(fid, minlength=n); h = np.bincount(fid, weights=np.asarray(flag, float), minlength=n)
    return np.where(k == 0, 0, np.where(h == 0, 1, np.where(h < k, 2, 3)))


def contradiction_accounting(pf_, pe, EVT, n, gross, pulled, link_st, fday, days, after):
    """V3 per fill over ALL its candidate events (pairs pf_ / pe), every split over every fill and its gross: per scenario S1 / S2 / S3 a
    contradictory timing assumption of any cause (event status >= 2), the evidence cap (2) and the leg intersection (3) - mixed: all candidates
    contradictory, not all of one cause - and `after` (per pair: the replay time after the evidence cap). -> (V3, per-fill CST codes)."""
    flags = {}
    for k in ('S1', 'S2', 'S3', 'ST'):
        st = EVT[f'{k}_st'].values[pe]; flags[k] = st >= 2; flags[k + '_cap'] = st == 2; flags[k + '_legs'] = st == 3
    flags['after_evidence'] = after; V3, cst = {}, {}
    for k, flag in flags.items():
        cst[k] = cs = candidate_status(pf_, flag, n)
        V3[k] = dict(fills={CST[c]: int((cs == c).sum()) for c in range(4)}, gross={CST[c]: float(gross[cs == c].sum()) for c in range(4)},
                     pulled={CST[c]: int(((cs == c) & pulled).sum()) for c in range(4)}, by_link={LINK[l]: {CST[c]: int(((cs == c) & (link_st == l)).sum()) for c in range(4)}
                                                                                              for l in range(3)},
                     by_day={d: {CST[c]: int(((cs == c) & (fday == d)).sum()) for c in range(4)} for d in days}, of_fills=n, of_gross=float(gross.sum()))
    for k in ('S1', 'S2', 'S3', 'ST'):
        mixed = (cst[k] == 3) & (cst[k + '_cap'] != 3) & (cst[k + '_legs'] != 3); st = EVT[f'{k}_st'].values
        V3[k].update(mixed_all=dict(fills=int(mixed.sum()), gross=float(gross[mixed].sum())), events={WST[s_]: int((st == s_).sum()) for s_ in range(4)},
                     events_by_cls={EVC[c]: {WST[s_]: int(((st == s_) & (EVT.cls.values == c)).sum()) for s_ in range(4)} for c in range(4)})
    return V3, cst


def causes_reconcile(V3):
    """Every fill contradictory on all its candidates is all-evidence-cap, all-leg-intersection or mixed - exactly once (fills and gross)."""
    return all(V3[k]['fills']['all'] == V3[k + '_cap']['fills']['all'] + V3[k + '_legs']['fills']['all'] + V3[k]['mixed_all']['fills']
               and abs(V3[k]['gross']['all'] - V3[k + '_cap']['gross']['all'] - V3[k + '_legs']['gross']['all'] - V3[k]['mixed_all']['gross'])
               <= 1e-6 * max(1., V3[k]['gross']['all']) for k in ('S1', 'S2', 'S3', 'ST'))


def leg_source(how, t_e2):
    """A leg's own timing source (index into SRC4): 0 WS leg link, 1 WS cell link, 2 E2 book drop, 3 the REST second."""
    how = np.asarray(how); return np.where(how == 1, 0, np.where(how == 2, 1, np.where(np.isfinite(np.asarray(t_e2, float)), 2, 3)))


def disagreement(L, EVT, k):
    """Of scenario k's events whose legs cannot share one instant (status contradictory_legs): the binding pair - the sources (SRC4) of the
    leg with the latest window start and of the leg with the earliest window end - the events whose legs carry different REST seconds, the
    spread of their WS pending times, and how many have an admissible S3 window (status < 2)."""
    _, W = leg_windows(L); lo, hi = W[k]
    e = pd.MultiIndex.from_frame(EVT[['day', 'group']]).get_indexer(pd.MultiIndex.from_arrays([L.day.values, L.group.values])); C.require(bool((e >= 0).all()), 'a leg without an event')
    bad = (EVT[f'{k}_st'].values == 3)[e]
    D = pd.DataFrame(dict(e=e[bad], lo=lo[bad], hi=hi[bad], src=leg_source(L.how.values, L.t_e2.values)[bad], S=L.exec_s.values[bad].astype(float),
                          ws=np.where(L.how.values > 0, L.t_ms.values.astype(float), np.nan)[bad]))
    if not len(D):
        return dict(events=0, binding_pair={}, different_rest_seconds=0, ws_spread_ms=dict(events=0), admissible_under_S3=0)
    g = D.groupby('e'); a = D.src.values[g.lo.idxmax().values]; b = D.src.values[g.hi.idxmin().values]       # D has a RangeIndex: labels are positions
    pair = pd.Series(['|'.join(sorted((SRC4[x], SRC4[y]))) for x, y in zip(a, b)]).value_counts(); sp = (g.ws.max() - g.ws.min())[g.ws.count() >= 2].values
    ev = np.array(sorted(g.groups)); pct = {f'p{q}': float(np.percentile(sp, q)) for q in (50, 90)} if len(sp) else {}
    return dict(events=int(len(ev)), binding_pair={str(x): int(y) for x, y in pair.items()}, different_rest_seconds=int((g.S.nunique() > 1).sum()),
                ws_spread_ms=dict(events=int(len(sp)), **pct, **({'max': float(sp.max())} if len(sp) else {})), admissible_under_S3=int((EVT.S3_st.values[ev] < 2).sum()))


def joint_bounds(pf_, pe, n, wl, wh, anc, t0, F, CW, lat, cap=JCAP):
    """Extremes of E with ONE instant per event (window wl / wh on the event's own axis, anchor anc) and ONE identity per ambiguous fill (a
    fill with several candidate events takes one of them). Events tied by an ambiguous fill form a component: exact by enumerating its
    identity assignments (each one decouples the events into one-instant sweeps; min / max over assignments); above `cap` assignments the
    component gets the outer relaxation (its unique events swept, each ambiguous fill / candidate pair moved independently). -> one row per
    unit: kind event (an untied event, exact) / joint (a component, exact) / relaxed (a component, outer), e (an event of it), fills, gross, mn, mx."""
    nE = len(wl); nc = np.bincount(pf_, minlength=n); amb = nc[pf_] > 1; par = np.arange(nE); aval = np.abs(F['val'])

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    cand = {}
    for f, e in zip(pf_[amb].tolist(), pe[amb].tolist()):
        cand.setdefault(f, []).append(e)
    for es in cand.values():
        for e in es[1:]:
            a, b = find(es[0]), find(e)
            if a != b:
                par[b] = a
    root = np.array([find(e) for e in range(nE)], np.int64); croot = np.array(sorted({int(root[es[0]]) for es in cand.values()}), np.int64)
    uf, ue = pf_[~amb], pe[~amb]; iso = ~np.isin(root[ue], croot); out = []
    if iso.any():
        ids, ix = np.unique(ue[iso], return_inverse=True)
        mn, mx = sweep(ix, wl[ids], wh[ids], t0[uf[iso]] - anc[ue[iso]], {k: v[uf[iso]] for k, v in F.items()}, CW, lat)
        out.append(pd.DataFrame(dict(kind='event', e=ids, fills=np.bincount(ix), gross=np.bincount(ix, weights=aval[uf[iso]]), mn=mn, mx=mx)))
    comp = {}
    for f, es in cand.items():
        comp.setdefault(int(root[es[0]]), dict(multi=[], unique={}))['multi'].append((f, es))
    for f, e in zip(uf[~iso].tolist(), ue[~iso].tolist()):
        comp[int(root[e])]['unique'].setdefault(e, []).append(f)
    key, vf, ve, vid, relax = [], [], [], [], []
    for r, c in comp.items():
        evs = sorted({e for _, es in c['multi'] for e in es} | set(c['unique'])); na = float(np.prod([float(len(es)) for _, es in c['multi']]))
        if na > cap:
            relax.append(r); continue
        for a, asg in enumerate(itertools.product(*[es for _, es in c['multi']])):
            for e in evs:
                fl = c['unique'].get(e, []) + [f for (f, _), x in zip(c['multi'], asg) if x == e]
                v = len(key); key.append((r, a, e)); vf += fl; ve += [e] * len(fl); vid += [v] * len(fl)
    rows = []
    if key:
        K_ = pd.DataFrame(key, columns=['r', 'a', 'e']); vf, ve, vid = (np.array(x, np.int64) for x in (vf, ve, vid))
        K_['mn'], K_['mx'] = sweep(vid, wl[K_.e.values], wh[K_.e.values], t0[vf] - anc[ve], {k: v[vf] for k, v in F.items()}, CW, lat)
        S = K_.groupby(['r', 'a']).agg(mn=('mn', 'sum'), mx=('mx', 'sum')).groupby(level=0).agg(mn=('mn', 'min'), mx=('mx', 'max'))
        for r, x in S.iterrows():
            c = comp[r]; fl = sorted({f for f, _ in c['multi']} | {f for v in c['unique'].values() for f in v})
            rows.append(('joint', min(e for _, es in c['multi'] for e in es), len(fl), float(aval[fl].sum()), float(x.mn), float(x.mx)))
    for r in relax:
        c = comp[r]; ue_ = [e for e, fs in c['unique'].items() for _ in fs]; uf_ = [f for fs in c['unique'].values() for f in fs]
        mf_ = [f for f, es in c['multi'] for _ in es]; me_ = [e for _, es in c['multi'] for e in es]; lo_ = hi_ = 0.
        if uf_:
            ids, ix = np.unique(np.array(ue_, np.int64), return_inverse=True); uf_ = np.array(uf_, np.int64)
            a1, b1 = sweep(ix, wl[ids], wh[ids], t0[uf_] - anc[np.array(ue_, np.int64)], {k: v[uf_] for k, v in F.items()}, CW, lat); lo_, hi_ = float(a1.sum()), float(b1.sum())
        mf_, me_ = np.array(mf_, np.int64), np.array(me_, np.int64)
        a2, b2 = sweep(np.arange(len(mf_)), wl[me_], wh[me_], t0[mf_] - anc[me_], {k: v[mf_] for k, v in F.items()}, CW, lat)
        fm = pd.DataFrame(dict(f=mf_, mn=a2, mx=b2)).groupby('f').agg(mn=('mn', 'min'), mx=('mx', 'max'))
        fl = sorted(set(mf_.tolist()) | set(np.asarray(uf_, np.int64).tolist()))
        rows.append(('relaxed', int(min(me_)), len(fl), float(aval[fl].sum()), lo_ + float(fm.mn.sum()), hi_ + float(fm.mx.sum())))
    if rows:
        out.append(pd.DataFrame(rows, columns=['kind', 'e', 'fills', 'gross', 'mn', 'mx']))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['kind', 'e', 'fills', 'gross', 'mn', 'mx'])


def combine(fid, cls, n):
    """Per fill: the class every candidate event agrees on, else AMBIGUOUS (identity unknown among candidates that disagree); -1 if none."""
    lo = np.full(n, 99); hi = np.full(n, -1); np.minimum.at(lo, fid, cls); np.maximum.at(hi, fid, cls)
    return np.where(hi < 0, -1, np.where(lo == hi, lo, 2))


def xtab(cls, removed, val, mask):
    """Rows REMOVED / KEPT x ORD: fills, net c (what removing them avoids, -value) and gross c; each row with its own total."""
    out = {}
    for rn, rm in (('REMOVED', removed), ('KEPT', ~removed)):
        m = mask & rm; out[rn] = {'_total': dict(fills=int(m.sum()), net=float(-val[m].sum()), gross=float(np.abs(val[m]).sum()))}
        for c, nm in enumerate(ORD):
            x = m & (cls == c); out[rn][nm] = dict(fills=int(x.sum()), net=float(-val[x].sum()), gross=float(np.abs(val[x]).sum()))
    return out


def conserved(X):
    """Every row's cells sum to its total (fills exactly, c to 1e-6 relative)."""
    for r in X.values():
        t = r['_total']; f = sum(r[nm]['fills'] for nm in ORD); g = sum(r[nm]['gross'] for nm in ORD); nt = sum(r[nm]['net'] for nm in ORD)
        if f != t['fills'] or abs(g - t['gross']) > 1e-6 * max(1., t['gross']) or abs(nt - t['net']) > 1e-6 * max(1., abs(t['gross'])):
            return False
    return True


def sign(mn, mx):
    return 'EXCESS-POSITIVE-IDENTIFIED' if mn > 0 else 'EXCESS-NEGATIVE-IDENTIFIED' if mx < 0 else 'SIGN-NOT-IDENTIFIED'


def v1_reference(days=KP.DAYS):
    """v1's committed numbers (its output JSON, hash-equal to the copy committed in d1de7504): on the full run its pooled S1 / S2 bounds and
    ordering counts; on a subset of days its per-day bounds (v1 kept no per-day ordering counts: order None, not compared)."""
    p = V1.OUT / 'timing_audit.json'; q = HERE.parent / 'reviews' / 'keep_pull_timing_audit_20261118' / 'timing_audit.json'
    C.require(C.sha(p) == C.sha(q), 'v1 output differs from its committed copy'); R = json.loads(p.read_text())['R']
    if list(days) == list(KP.DAYS):
        return {k: dict(E_min=R[k]['E_min'], E_max=R[k]['E_max'], order={nm: R[k]['order'][nm]['fills'] for nm in ('AFTER', 'BEFORE', 'AMBIGUOUS')}) for k in ('S1', 'S2')}
    return {k: dict(E_min=float(sum(R[k]['by_day'][d]['E_min'] for d in days)), E_max=float(sum(R[k]['by_day'][d]['E_max'] for d in days)), order=None) for k in ('S1', 'S2')}


def collapsed_reference(days=KP.DAYS):
    """The committed collapsed E and score improvement: pooled +234,279 / 1,169,671 c; on a subset of days the clock audit's per-day values for
    M50 fills at M50 eligibility (its JSON, hash-equal to the copy committed in 1a1abbaa)."""
    if list(days) == list(KP.DAYS):
        return 234279., 1169671.
    p = Path(I.K.IO) / 'keep_pull_clock_audit_20261118' / 'clock_audit.json'; q = HERE.parent / 'reviews' / 'keep_pull_clock_audit_20261118' / 'clock_audit.json'
    C.require(C.sha(p) == C.sha(q), 'clock audit output differs from its committed copy'); D = json.loads(p.read_text())['R']['M50_M50']['day']
    return float(sum(D[d]['excess'] for d in days)), float(sum(D[d]['score'] for d in days))


def accounted(V3, n, total):
    """Control (f): every contradiction split covers every fill once and the whole gross."""
    return all(sum(v['fills'].values()) == n and abs(sum(v['gross'].values()) - total) <= 1e-6 * max(1., total) for v in V3.values())


def run(out=OUT, src=None, tables=None, rows_of=None, env_of=None, v1=None, days=None, expect=None, tol=1., jcap=JCAP, sem_of=None, set_of_=None, st_lag=None):
    """The ledger on the committed keep / pull run (or, in the selftest's rehearsal, on synthetic tables, source and output dir)."""
    start = time.time(); out = Path(out); C.require(not out.exists(), 'ledger output exists')
    if tables is None:
        selftest()
    days = list(days or KP.DAYS); C.require(tables is not None or set(days) <= set(KP.DAYS), 'undeclared day')
    O, Fl = tables if tables is not None else KP.load(KP.OUT, days=days); src = src or KP.Tape(); expect = expect or collapsed_reference(days)
    rows_of = rows_of or day_rows; env_of = env_of or day_env; sem_of = sem_of or day_sem; v1 = v1 or v1_reference(days); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    set_of_ = set_of_ or set_of; lag_in = {}
    if st_lag is None:
        st_lag, lag_in = known_lag()
    E = O[O.status == 'eligible'].reset_index(drop=True); cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
    endj = A.endj_of(E.t_full.values, E.g_struct.values); act, CW = V1.null_weights(T, cell, endj); ri, t, val, srcc = A.fills_of(E, Fl, SHIFT)
    n = len(ri); lend = np.where(np.isnan(E.land_struct.values[ri]), np.inf, E.land_struct.values[ri]); t0 = E.t.values[ri]
    F = dict(val=val, pulled=act[ri], T=T[ri], s=cell[ri], endj=endj[ri], lend=lend); r37 = La[np.clip(T[ri], 1, KP.NSTEP) - 1]
    F1 = dict(val=val, pulled=act[ri], r=r37, s=cell[ri], endj=endj[ri], lend=lend); gs = E.g_struct.values[ri]
    C.require(np.array_equal(struct_landing(gs, KP.CANCEL), lend), 'the structural landing rule differs from the committed landing at 37 ms')
    pf = A.per_fill(T, cell, endj, (ri, t, val, srcc)); e0 = float(contrib(t, F, CW).sum()); s0 = float(pf['score'].sum())
    ok_a = abs(e0 - float(pf['score'].sum() - pf['null'].sum())) < 1e-6 and abs(e0 - expect[0]) <= tol and abs(s0 - expect[1]) <= tol
    if not ok_a:
        sys.exit(f'EXEC-WINDOW LEDGER FAILED -> no output: collapsed E {e0:,.1f} / score {s0:,.1f}, not {expect[0]:+,.0f} / {expect[1]:,.0f}')
    Fd = pd.DataFrame(dict(fid=np.arange(n), day=E.day.values[ri], m=E.m.values[ri], side=E.side.values[ri], a=t + SHIFT + t0, p=E.p.values[ri]))
    H1, P, LG, env, recon, semst, sdiag = [], [], [], {}, {}, {}, {}
    for d in days:
        env[d] = dict(env_of(d)); rows = rows_of(d); sem, sem_in = sem_of(d, rows); ts, ts_in = set_of_(d)
        C.require(len(sem) == len(rows) == len(ts), f'{d}: S3 windows or settlement times not row-aligned')
        lo_, hi_, cd_ = st_windows(ts.t_set_ms.values, sem.S3_llo.values, sem.S3_lhi.values, st_lag)
        stw = pd.DataFrame(dict(t_set_ms=ts.t_set_ms.values.astype(float), set_status=ts.set_status.values, ST_llo=lo_, ST_lhi=hi_, ST_code=cd_)); del ts
        env[d]['inputs'] = dict(env[d].get('inputs', {}), **sem_in, **ts_in, **lag_in)
        sdiag[d] = settlement_diag(pd.concat([rows.reset_index(drop=True), sem[['S3_llo', 'S3_lhi']], stw[['ST_llo', 'ST_lhi']]], axis=1), stw.t_set_ms.values, st_lag, env[d])
        h_, p_, lg, recon[d] = link_day(d, Fd[Fd.day == d], src, env[d], rows); del rows
        lg = pd.concat([lg, sem.iloc[lg.row_id.values.astype(np.int64)].reset_index(drop=True), stw.iloc[lg.row_id.values.astype(np.int64)].reset_index(drop=True)], axis=1)
        semst[d] = dict(day_legs={SEM_CODES[c]: int((sem.S3_code.values == c).sum()) for c in range(len(SEM_CODES))}, day_link_late=int(sem.link_late.sum()),
                        event_legs={SEM_CODES[c]: int((lg.S3_code.values == c).sum()) for c in range(len(SEM_CODES))}, event_link_late=int(lg.link_late.sum())); del sem, stw
        H1.append(h_); P.append(p_.assign(day=d)); LG.append(lg.assign(day=d, h=env[d]['h'], q01=env[d]['q01'], q99=env[d]['q99']))
        print(json.dumps(dict(stage='link', day=d, fills=int((Fd.day == d).sum()), linked=int(p_.fid.nunique()), legs=len(lg), identity_mismatches=recon[d],
                              seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    H1, P, LG = pd.concat(H1), pd.concat(P, ignore_index=True), pd.concat(LG, ignore_index=True)
    # V0 (b) v1's own functions on v1's hull windows; (c) both links cover every fill; (d) identities
    C.require(H1.index.is_unique, 'a fill linked twice by v1'); H1 = H1.reindex(np.arange(n)); unl = int(H1.legs.isna().sum())
    ok_c = unl == 0 and set(P.fid.unique()) == set(range(n)); pl = F['pulled']
    reg = {}
    for k, (a_, b_) in (('S1', ('lo1', 'hi1')), ('S2', ('lo2', 'hi2'))):
        lo, hi = H1[a_].values - t0, H1[b_].values - t0; mn, mx = V1.bounds(lo, hi, F1, CW); o = np.where(pl, V1.order_of(lo, hi, r37), -1)
        reg[k] = dict(E_min=float(mn.sum()), E_max=float(mx.sum()), order={nm: int(((o == c) & pl).sum()) for c, nm in enumerate(('AFTER', 'BEFORE', 'AMBIGUOUS'))})
    ok_b = all(abs(reg[k][x] - v1[k][x]) <= tol for k in reg for x in ('E_min', 'E_max')) and all(v1[k]['order'] is None or reg[k]['order'] == v1[k]['order'] for k in reg)
    ok_d = sum(recon.values()) == 0
    # events, candidate windows, classes
    EVT = events_of(LG); EVT['e'] = np.arange(len(EVT)); P = P.merge(EVT[['day', 'group', 'e']], on=['day', 'group'], how='left', validate='many_to_one')
    C.require(bool(P.e.notna().all()), 'a candidate event without legs'); pe = P.e.values.astype(np.int64); pf_ = P.fid.values.astype(np.int64)
    ncand = np.bincount(pf_, minlength=n); link_st = np.where(ncand == 0, 0, np.where(ncand == 1, 1, 2))
    cw = dict(CP=(r37, r37), CR=(KP.STEP * np.clip(T[ri], 1, KP.NSTEP) + CANCEL_RANGE[0], KP.STEP * np.clip(T[ri], 1, KP.NSTEP) + CANCEL_RANGE[1]))
    cls = {}
    for k in SCEN:
        lo = EVT[f'{k}_lo'].values[pe] - t0[pf_]; hi = EVT[f'{k}_hi'].values[pe] - t0[pf_]
        for c in CWIN:
            cls[k, c] = np.where(pl, combine(pf_, classify(lo, hi, cw[c][0][pf_], cw[c][1][pf_]), n), -1)
    removed = t >= r37; X = {f'{k}|{c}': xtab(cls[k, c], removed, val, pl) for k in SCEN for c in CWIN}; ok_e = all(conserved(x) for x in X.values())
    gross = np.abs(val)                                       # per fill over ALL its candidate events, every fill and link status in the split, by cause
    V3, cst = contradiction_accounting(pf_, pe, EVT, n, gross, pl, link_st, Fd.day.values, days, (t + t0)[pf_] > EVT.EV_hi.values[pe])
    ok_f = accounted(V3, n, float(gross.sum())) and causes_reconcile(V3)
    DIS = {k: disagreement(LG, EVT, k) for k in ('S1', 'S2', 'S3', 'ST')}
    V0 = dict(a_collapsed=ok_a, b_v1_regression=ok_b, c_linked=ok_c, d_identity=ok_d, e_conservation=ok_e, f_contradiction_accounting=ok_f)
    if not all(V0.values()):
        sys.exit(f'EXEC-WINDOW LEDGER FAILED -> no output: V0 {V0} (v1 regression {reg} vs {v1}; unlinked {unl}; identity mismatches {recon})')
    # value bounds: ONE instant per event and ONE identity per ambiguous fill - exact by component where enumerable, else labelled OUTER
    e_u = np.zeros(n, np.int64); e_u[pf_[ncand[pf_] == 1]] = pe[ncand[pf_] == 1]; anc = EVT.anc.values; B = {}
    ev_key = dict(side=np.where(EVT.side.values == 0, 'bid', 'ask'), day=EVT.day.values.astype(str), cls=np.array(EVC)[EVT.cls.values])
    for k in SCEN:
        wl, wh = event_axis_windows(EVT, k)
        for lat in LATS:
            U = joint_bounds(pf_, pe, n, wl, wh, anc, t0, at_latency(F, gs, lat), CW, lat, cap=jcap); kd = U.kind.values.astype(str); ev_ = U.e.values.astype(np.int64)
            C.require(int(U.fills.sum()) == int((ncand > 0).sum()), 'a linked fill missing from the bounds'); br = {}
            for nm, ke in ev_key.items():
                g = np.where((kd == 'event') | (nm != 'cls'), ke[ev_], kd)
                br[nm] = {str(x): [float(U.mn.values[g == x].sum()), float(U.mx.values[g == x].sum())] for x in sorted(set(g.tolist()))}
            rel = kd == 'relaxed'
            B[k, lat] = dict(E_min=float(U.mn.sum()), E_max=float(U.mx.sum()), by=br, bounds='outer' if rel.any() else 'exact',
                             units={x: int((kd == x).sum()) for x in ('event', 'joint', 'relaxed')}, fills={x: int(U.fills.values[kd == x].sum()) for x in ('event', 'joint', 'relaxed')},
                             relaxed_gross=float(U.gross.values[rel].sum()), of_fills=int((ncand > 0).sum()))
    # coverage, decisions, contradictions
    cls_name = np.array(EVC + ('multi', 'unlinked'))[np.where(link_st == 1, EVT.cls.values[e_u], np.where(link_st == 2, 4, 5))]
    side_nm = np.where(Fd.side.values == 0, 'bid', 'ask'); cov = {}
    for k in SCEN:
        ident = np.isin(cls[k, 'CP'], (0, 1)) & pl; cov[k] = {nm: {str(g): float(gross[ident & (key == g)].sum() / max(gross[pl & (key == g)].sum(), 1e-12))
                                                                  for g in sorted(set(key[pl]))} for nm, key in (('day', Fd.day.values), ('side', side_nm), ('cls', cls_name),
                                                                                                                 ('link', np.array(LINK)[link_st]))}
    po = np.flatnonzero(act); nf = np.bincount(ri, minlength=len(E)); dec = {}
    for k in SCEN:
        unev = np.bincount(ri[pl & ~np.isin(cls[k, 'CP'], (0, 1))], minlength=len(E))
        dec[k] = dict(pulled_orders=int(len(po)), no_m50_fill=int((nf[po] == 0).sum()), evaluable=int(((nf[po] > 0) & (unev[po] == 0)).sum()),
                      not_evaluable=int((unev[po] > 0).sum()))
    read = {'V0 CONTROLS': 'YES'}
    for k in SCEN:
        read[f'V1 SIGN {k}'] = sign(B[k, KP.CANCEL]['E_min'], B[k, KP.CANCEL]['E_max']) + ('' if B[k, KP.CANCEL]['bounds'] == 'exact' else ' (OUTER BOUNDS)')
    R = dict(fills=n, eligible=len(E), counted=len(O), status={str(s): int(v) for s, v in O.status.value_counts().items()}, link={LINK[s]: int((link_st == s).sum()) for s in range(3)},
             events=len(EVT), events_by_cls={EVC[c]: int((EVT.cls == c).sum()) for c in range(4)}, events_multi_side=int((EVT.sides > 1).sum()),
             legs=len(LG), legs_tk_status={str(s): int(v) for s, v in LG.tk_status.value_counts().items()}, env=env, collapsed_E=e0, score=s0, v1_regression=reg,
             xtab=X, bounds={f'{k}|{lat:g}': v for (k, lat), v in B.items()}, coverage=cov, decisions=dec, V3=V3, sim_removed=float(-val[pl & removed].sum()),
             sim_kept=float(-val[pl & ~removed].sum()), disagreement=DIS, s3=dict(semst, rest_lag_ms=REST_LAG, ws_delay_ms=WS_DELAY), st_lag_ms=st_lag, settlement_model=SETTLEMENT_MODEL, st_lag_role='UNIDENTIFIED' if st_lag is None else 'EXPLICIT_SCENARIO_ASSUMPTION', settlement=sdiag)
    mt = np.isfinite(LG.t_set_ms.values); x = (LG.anc.values - SHIFT - LG.t_set_ms.values)[mt]                   # the replay's own point against the settlement time
    R['replay_point_minus_t_set_ms'] = dict(legs=int(mt.sum()), **({f'p{q}': float(np.percentile(x, q)) for q in (1, 5, 50, 95, 99)} if len(x) else {}),
                                            inside_st=float(((x >= (-st_lag if st_lag is not None else -np.inf)) & (x <= 0.)).mean()) if len(x) else None, beyond_1s=float((np.abs(x) > 1000.).mean()) if len(x) else None)
    # the ledger tables (all built, documented and checked before anything is written)
    ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values])); f50 = Fl[Fl.clock == SHIFT]
    q50 = f50.q.values[~np.isnan(ix.reindex(pd.MultiIndex.from_arrays([f50.day.values, f50.oid.values])).values)]; C.require(len(q50) == n, 'fill quantities misaligned')
    FL = pd.DataFrame(dict(fid=np.arange(n), day=Fd.day.values, oid=E.oid.values[ri], m=Fd.m.values, side=Fd.side.values, p=Fd.p.values, t0=t0, t_replay=t, anchor=Fd.a.values,
                           q=q50, value=val, src=srcc, pulled=pl, T=T[ri], landing37=r37, replay_removed=removed & pl, link=np.array(LINK)[link_st], n_events=ncand,
                           event=np.where(link_st == 1, e_u, -1), activation_assumed=KP.PLACE, notice_assumed=KP.NOTICE,
                           **{f'contra_{k}': np.array(CST)[cst[k]] for k in cst},
                           **{f'cls_{k}_{c}': cls[k, c] for k in SCEN for c in CWIN}))
    OL = O[['day', 'oid', 'm', 'side', 't', 'status']].copy(); ix = pd.MultiIndex.from_arrays([E.day.values, E.oid.values])
    for nm, v in (('T', T), ('pulled', act), ('endj', endj), ('m50_fills', nf), ('m50_value', np.bincount(ri, weights=val, minlength=len(E)))):
        OL[nm] = pd.Series(v, index=ix).reindex(pd.MultiIndex.from_arrays([OL.day.values, OL.oid.values])).values
    for k in SCEN:
        unev = np.bincount(ri[pl & ~np.isin(cls[k, 'CP'], (0, 1))], minlength=len(E))
        ev_ = np.where(~act, 'not_pulled', np.where(nf == 0, 'no_m50_fill', np.where(unev == 0, 'evaluable', 'not_evaluable')))
        OL[f'decision_{k}'] = pd.Series(ev_, index=ix).reindex(pd.MultiIndex.from_arrays([OL.day.values, OL.oid.values])).fillna('not_eligible').values
    TB = dict(ledger_events=EVT, ledger_legs=LG.drop(columns=['h', 'q01', 'q99']), ledger_fill_events=P[['fid', 'day', 'group', 'e']], ledger_fills=FL, ledger_orders=OL)
    cols = {k: list(v.columns) for k, v in TB.items()}; gaps = schema_gaps(cols)
    if gaps:
        sys.exit(f'EXEC-WINDOW LEDGER FAILED -> no output: undocumented ledger columns {gaps}')
    out.mkdir(parents=True); files = {}
    for k, v in TB.items():
        v.to_parquet(out / f'{k}.parquet', index=False); files[f'{k}.parquet'] = C.sha(out / f'{k}.parquet')
    C.dump(out / 'ledger_schema.json', schema_doc(cols)); files['ledger_schema.json'] = C.sha(out / 'ledger_schema.json')
    L = report(R, read, days)
    (out / 'exec_windows.txt').write_text('\n'.join(L) + '\n')
    deps = [Path(__file__).resolve(), Path(KP.__file__).resolve(), Path(A.__file__).resolve(), Path(V1.__file__).resolve(), Path(I.__file__).resolve()]
    C.dump(out / 'exec_windows.json', dict(asof=C.stamp(), readings=read, V0=V0, R=R, outputs=files, source_receipts={d: C.sha(KP.OUT / d / 'receipt.json') for d in days}
                                         if tables is None else {}, v1_reference=v1, dependencies={p.name: C.sha(p) for p in deps}, provenance=KP.provenance(),
                                         source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return R, read


def report(R, read, days):
    def cell(x, tot):
        return f"{x['fills']:,} / {x['net']:+,.0f} c ({100 * x['gross'] / max(tot, 1e-12):.1f}% of the row's gross)"
    L = ['EXECUTION-WINDOW LEDGER of the keep / pull selection, declared score A+0 (pf_keep_pull_exec_windows.py; the committed keep / pull run, M50 fills, '
         f'days {", ".join(days)} (consumed); c summed; v3: the review of 1fd77004 - contradictions by cause, the S3 semantics windows; supersedes v2\'s receipt)',
         f"  {R['counted']:,} counted states ({', '.join(f'{k} {v:,}' for k, v in R['status'].items())}); {R['fills']:,} M50 fills; links "
         + ', '.join(f'{k} {v:,}' for k, v in R['link'].items()),
         f"  {R['events']:,} events (taker orders; {', '.join(f'{k} {v:,}' for k, v in R['events_by_cls'].items())}; {R['events_multi_side']:,} with legs on both sides); "
         f"{R['legs']:,} legs (tk_status {R['legs_tk_status']})",
         f"  V0 controls: collapsed E {R['collapsed_E']:+,.0f} / score {R['score']:,.0f}; v1 regression " + '; '.join(
             f"{k} [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}] {v['order']}" for k, v in R['v1_regression'].items()), '',
         'PART 1 - the score\'s pulled orders\' fills: replay ordering (row) x window class (cell: fills / net c = what removing them avoids)',
         f"  replay: REMOVED {R['sim_removed']:+,.0f} c, KEPT {R['sim_kept']:+,.0f} c"]
    for key, X in R['xtab'].items():
        L.append(f'  {key}:')
        for rn in ('REMOVED', 'KEPT'):
            tot = X[rn]['_total']['gross']; L.append(f"    {rn:<7} ({X[rn]['_total']['fills']:,} fills, {X[rn]['_total']['net']:+,.0f} c): "
                                                    + ' | '.join(f'{nm} {cell(X[rn][nm], tot)}' for nm in ORD if X[rn][nm]['fills']))
    L += ['', 'PART 2 - coverage: share of the pulled orders\' fill value (gross) whose ordering is identified (CANCEL_FIRST or MATCH_FIRST) against the replay\'s cancel point']
    for k, v in R['coverage'].items():
        L.append(f'  {k}: ' + ' | '.join(f"{nm} " + ', '.join(f'{g} {100 * x:.1f}%' for g, x in d.items()) for nm, d in v.items()))
    L.append('  decisions (pulled orders): ' + ' | '.join(f"{k} evaluable {v['evaluable']:,} / not {v['not_evaluable']:,} / no M50 fill {v['no_m50_fill']:,} of {v['pulled_orders']:,}"
                                                         for k, v in R['decisions'].items()))
    L += ['', 'PART 3 - bounds on E = score improvement - exact STATES-null expectation (one shared instant per event in its admissible window; M50 quantities held: '
          'exact for this fixed-fill / identity problem, NOT a bound over the queue replay - pf_keep_pull_scenarios.py re-simulates fills)']
    for key, v in R['bounds'].items():
        L.append(f"  {key} ms: [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}] {v['bounds'].upper()} (of {v['of_fills']:,} linked fills: untied events {v['fills']['event']:,}, "
                 f"joint components {v['units']['joint']:,} / {v['fills']['joint']:,} fills exact, relaxed {v['units']['relaxed']:,} / {v['fills']['relaxed']:,} fills / "
                 f"{v['relaxed_gross']:,.0f} c gross) | " + ' | '.join(f'{nm} ' + ', '.join(f'{g} [{a:+,.0f}, {b:+,.0f}]' for g, (a, b) in d.items()) for nm, d in v['by'].items()))
    L += ['', 'PART 4 - CONTRADICTORY TIMING ASSUMPTIONS per fill over ALL its candidate events: the scenario\'s own windows cannot all hold - evidence_cap '
          '(its window lies beyond the REST cap) or leg_intersection (its legs\' windows cannot share one instant); not a REST refutation of a match time; kept, '
          'never repaired; denominators: every fill and its gross']
    nm_ = dict(after_evidence='replay time after the evidence cap', **{k + x: f'{k} {y}' for k in ('S1', 'S2', 'S3', 'ST') for x, y in
                                                                       (('', 'any cause'), ('_cap', 'evidence_cap'), ('_legs', 'leg_intersection'))})
    for k, nm in nm_.items():
        v = R['V3'][k]; L.append(f"  {nm}: of {v['of_fills']:,} fills / {v['of_gross']:,.0f} c gross - " + ', '.join(f"{c} candidates {v['fills'][c]:,} / {v['gross'][c]:,.0f} c"
                                                                                                                for c in ('all', 'some', 'none', 'unlinked'))
                 + f"; on pulled orders {v['pulled']}; by link {v['by_link']}" + (f"; mixed causes on all candidates {v['mixed_all']['fills']:,} / {v['mixed_all']['gross']:,.0f} c; "
                                                                                   f"events {v['events']}; by class {v['events_by_cls']}" if 'events' in v else ''))
    L.append('  leg disagreements (events with status contradictory_legs): ' + ' | '.join(
        f"{k} {v['events']:,} events, binding source pairs {v['binding_pair']}, legs on different REST seconds {v['different_rest_seconds']:,}, WS spread {v['ws_spread_ms']}, "
        f"admissible under S3 {v['admissible_under_S3']:,}" for k, v in R['disagreement'].items()))
    L.append('  S3 leg bases ' + ' | '.join(f"{d}: day {v['day_legs']} (links later than the REST cap + 250 ms {v['day_link_late']:,}); event legs {v['event_legs']}"
                                           for d, v in R['s3'].items() if isinstance(v, dict)))
    L += ['', f"PART 5 - SETTLEMENT TIMES (the settlementId's UUIDv7 ms; ST lag {R['st_lag_ms']} ms, role {R['st_lag_role']}): the timing sources and the "
          'declared windows against the venue\'s own timestamp']
    for d, v in R['settlement'].items():
        L.append(f'  {d}: ' + json.dumps(v))
    L.append(f"  the replay's own point (anchor - 50 ms) minus the settlement time, on the fills' event legs: {R['replay_point_minus_t_set_ms']}")
    L += ['', 'READINGS (declared before any data was read)'] + [f'  {k}: {v}' for k, v in read.items()] + [
          '', 'EVIDENCE: the settlementId\'s UUIDv7 time records ID creation, not a proven match instant. ST has no lower bound unless an explicit scenario lag is supplied. '
          'Creation at/after matching and clock alignment remain premises. The REST executedAt is the Confirmed status (an upper bound only); the WS '
          'pending link is by value; cancel effect and activation are configured points.', 'LIMITS: consumed days; one declared rule; STATES is local selection; M50 quantities held; S1 / S2 / S3 / CR are assumption '
          f'ranges (S3: the match within {REST_LAG / 1000:g} s of its Confirmed second, a WS leg\'s increment carries its value, book and REST stamps on one clock); '
          'the REST-to-book clock alignment is not established.']
    return L


CASE_SCHEMA = {
    'known_cases': dict(run='live run (v6 / basis)', cid='the bot order id', place='Control Place, bot clock ms', accepted='the Accepted answer received, bot clock ms',
                        upper='min(first positive fill notice, cancel confirmation), bot clock ms: the match precedes it', notice='first Matched / Confirmed notice received, bot clock ms',
                        ack='cancel confirmation (CancelPendingAudit: the first HTTP Canceled answer or user-stream cancel received), bot clock ms',
                        req='cancel request (core Request::Cancel t), bot clock ms', S='REST executedAt second (the Confirmed status), venue status clock s',
                        M='the Matched status stamp, venue status clock s', Z='the engine anchor of the leg (11-09 audit; NaN: ambiguous leg clock), venue book / REST clock ms',
                        signed='Exec Signed, bot clock ms', ucancel='user-stream cancel received, bot clock ms', cdone='core CancelDone known, bot clock ms',
                        thr='core Throttled known, bot clock ms', rM='first Matched notice received, bot clock ms', rC='first Confirmed notice received, bot clock ms',
                        Cs='the Confirmed status stamp, venue status clock s', lag='REST second - Matched stamp, ms (11-09 semantics audit)', order_hash='signed order hash',
                        trade_id='trade (settlement) id', tx='transaction hash', maker='maker signer', rest_raw='REST executedAt as received (text)',
                        leg_clock_status='the 11-09 audit\'s leg clock status (unique_leg / identical_clock_candidates / ambiguous_leg_clock)',
                        candidate_legs='the 11-09 audit\'s candidate rows (content key)', n_candidates='candidate rows found on Nov 8\'s legs_timed2 (must equal candidate_legs)',
                        anchor_origin='the engine anchor\'s origin over the leg\'s candidate rows (pending_increment / book_drop / REST_second; ambiguous: they differ)',
                        anchor_own_row='the anchor comes from the leg\'s own row (yes / no; ambiguous: its candidate rows differ)',
                        mid='market', exec_s='REST executedAt second (as S)', first_recv_ns='first REST receipt, collector local clock ns', side='side (0 bid / 1 ask of the Up book)',
                        ptick='price tick', shares='shares', price_executed='REST execution price', taker_index='taker index', taker_quote='taker quote flag',
                        t_set='the settlement time: the trade id\'s (= settlementId) UUIDv7 ms, venue settlement clock'),
    'known_case_rows': dict(i='the leg (known_cases row)', row_id='its candidate row in Nov 8\'s legs_timed2', group='the row\'s event: taker order tk, else -1 - row',
                            e='event index in this ledger', src='the row\'s own timing source (SRC4)', Zr='the pinned engine anchor of the row, venue ms',
                            origin='the anchor\'s origin by the engine rule (pending_increment / book_drop / REST_second)', anchor_own='the anchor comes from this row',
                            anchor_rule_ok='the rule reproduces the pinned anchor', how='WS link 0 none / 1 leg / 2 cell', t_ms='WS pending time, venue book clock ms',
                            t_e2='E2 book-drop estimate, venue book clock ms', S3_code='the row\'s S3 basis (SEM_CODES)', S3_ncand='its value-consistent increments',
                            link_late='its linked increment is later than its REST cap + 250 ms', event_legs='legs of its event', event_ws_span='spread of its event\'s WS times, ms'),
    'known_orders': dict(run='live run', cid='the bot order id', act_lo='activation - place, lower bound (signed - place), ms', act_hi='activation - place, upper bound (accepted - place; premise: live before the Accepted answer is received), ms',
                         act_status='against the declared activation range', cancel_lo='cancel effect - request, lower bound (0), ms',
                         cancel_hi='cancel effect - request, upper bound (confirmation - request; premise: the effect precedes it), ms', cancel_status='against the declared cancel range',
                         confirm_source='the first confirmation: http_answer / user_stream', user_stream_after_req='user-stream cancel received - request, ms',
                         core_done_after_req='core CancelDone known - request, ms', throttled='a core Throttled event on the order',
                         act_hi_set='activation - place, upper bound with the settlement times (<= its first match), ms', act_status_set='against the declared activation range',
                         cancel_lo_set='cancel effect - request, lower bound with the settlement times (after its latest match), ms', cancel_status_set='against the declared cancel range')}
CASE_PREFIX = {'known_cases': {**{f'{k}_': f'{k}: the leg\'s verdict (VERD) at offset 0 and at every admissible offset, its cause (CAUSES), false CANCEL_FIRST at offset 0 / '
                                            'at some admissible offset (no suffix: K7\'s offsets, the declared gate; _clk: the K8 shared clock, conditional on the UUID clock '
                                            'being the windows\' clock; _set: the zero-lag settlement premise; _dom: per clock domain - the gate of record, v6)' for k in KSC},
                               'K4_': 'K4 (v1\'s proxy: the engine anchor Z +- 250 / 600 ms): missed / CANCEL_FIRST per width; Z - place and Z - upper, ms (venue vs bot clock at '
                                      'offset 0; NaN: no anchor)',
                               'K8_': 'K8 under event-specific nonnegative creation lag: match_lo/hi_bot from lifecycle plus shared clock bounds; creation_lag_lo/hi for this identified leg only, ms (not transferable)',
                               'K6D_': 'K6 per clock domain (v6): the leg\'s book-clock lower bound B, venue book - bot ms, from pf_known_clock_census (NaN: unidentified - its book ends fail closed)'},
               'known_case_rows': {**{f'{k}_': f'{k}: the event window (lo / hi, venue ms) and status (st); lo_book / lo_status: its start split by the clock domain of the legs '
                                             'that set it (v6; -inf: none on that clock)' for k in KSC},
                                   'own_': 'the row\'s own window per scenario (its own source alone, capped by its own REST second), venue ms'}}


def case_gaps(tables):
    return [f'{t}.{c}' for t, cols in tables.items() for c in cols if c not in CASE_SCHEMA.get(t, {}) and not any(c.startswith(x) for x in CASE_PREFIX.get(t, {}))]


def known_frame():
    """The 11-09 fill-clock audit's 99 confirmed own maker legs, hash-pinned: identities, lifecycles and order paths (bot clock), status stamps
    (venue) and their first receipts (bot), and the research linkage on Nov 8's own tables (case_linkage). -> (K per leg, KE per leg x
    candidate row, the 11-09 counts, input hashes and Nov 8's envelopes)."""
    sem = json.loads((FC / 'semantics_declaration.json').read_text()); rep = json.loads((FC / 'semantics_report.json').read_text())
    dec = json.loads((FC / 'declaration_v3.json').read_text())['input_sha']; art = json.loads((FC / 'report.json').read_text())['artifacts']
    C.require(C.sha(FC / 'clock_cases.parquet') == CC_SHA == sem['input_sha']['clock_cases.parquet'], 'clock_cases changed since the fill-clock audit')
    C.require(C.sha(FC / 'status_clock_cases.parquet') == rep['artifact_sha'], 'status_clock_cases changed since the semantics audit')
    for f in ('fill_events.parquet', 'orders.parquet', 'engine_anchors.parquet', 'own_rest_legs.parquet'):
        C.require(C.sha(FC / f) == art[f], f'{f} changed since the fill-clock audit')
    pinned = [LIVE / f'core_orders_{r}.parquet' for r in ('v6', 'basis')] + [NOV8 / 'legs_timed2.parquet', NOV8 / 'legs_tk.parquet']
    for p in pinned:
        C.require(C.sha(p) == dec[str(p)], f'{p.name} changed since the fill-clock audit')
    inputs = {str(p): C.sha(p) for p in [FC / f for f in ('clock_cases.parquet', 'status_clock_cases.parquet', 'fill_events.parquet', 'orders.parquet', 'engine_anchors.parquet')]
              + pinned + [NOV8 / f for f in ('pend_inc.parquet', 'booktime.json', 'timing.json')]}
    CC = pd.read_parquet(FC / 'clock_cases.parquet'); SC = pd.read_parquet(FC / 'status_clock_cases.parquet'); FE = pd.read_parquet(FC / 'fill_events.parquet')
    OR = pd.read_parquet(FC / 'orders.parquet', columns=['run', 'cid', 't_signed', 't_UserCancel'])
    core = pd.concat([pd.read_parquet(LIVE / f'core_orders_{r}.parquet', columns=['cid', 't_cancel', 't_CancelDone', 't_Throttled']) for r in ('v6', 'basis')])
    C.require(core.cid.is_unique and not OR.duplicated(['run', 'cid']).any(), 'an order id twice')
    k = CC.merge(SC[['run', 'cid', 'order_hash', 'trade_id', 'matched_reported_ms', 'rest_raw', 'rest_minus_matched_ms']], on=['run', 'cid', 'order_hash', 'trade_id'],
                 how='left', validate='one_to_one').merge(core, on='cid', how='left', validate='many_to_one').merge(OR, on=['run', 'cid'], how='left', validate='many_to_one')
    FE = FE.merge(CC[['order_hash', 'trade_id']].drop_duplicates(), on=['order_hash', 'trade_id'], how='inner')     # the 99 legs' own trades only
    rc = FE.groupby(['order_hash', 'trade_id', 'status'])['at'].min().unstack(); cf = FE[FE.status == 'Confirmed'].groupby(['order_hash', 'trade_id']).reported_match_ms.agg(['min', 'max'])
    C.require(bool((cf['min'] == cf['max']).all()), 'a trade with two Confirmed stamps'); ix = pd.MultiIndex.from_arrays([k.order_hash.values, k.trade_id.values])
    K = pd.DataFrame(dict(run=k.run.values, cid=k.cid.values, place=k.place.values, accepted=k.accepted.values, upper=k.upper.values, notice=k.first_positive_notice.values,
                          ack=k.cancel.values, req=k.t_cancel.values, S=k.exec_s.values.astype(float), M=k.matched_reported_ms.values / 1000., Z=k.Z.values,
                          signed=k.t_signed.values, ucancel=k.t_UserCancel.values, cdone=k.t_CancelDone.values, thr=k.t_Throttled.values,
                          rM=rc['Matched'].reindex(ix).values, rC=rc['Confirmed'].reindex(ix).values, Cs=cf['min'].reindex(ix).values / 1000., lag=k.rest_minus_matched_ms.values,
                          t_set=np.array([SETT.uuid7_ms(x) for x in k.trade_id.values]),
                          order_hash=k.order_hash.values, trade_id=k.trade_id.values, tx=k.tx.values, maker=k.maker.values, rest_raw=k.rest_raw.values,
                          leg_clock_status=k.leg_clock_status.values, candidate_legs=k.candidate_legs.values, **{c: k[c].values for c in CKEY}))
    L8 = pd.read_parquet(NOV8 / 'legs_timed2.parquet', columns=CKEY + ['how', 't_ms', 't_e2']); tk = pd.read_parquet(NOV8 / 'legs_tk.parquet', columns=['tk'])
    ZA = pd.read_parquet(FC / 'engine_anchors.parquet', columns=['Z']); C.require(len(L8) == len(tk) == len(ZA), 'Nov 8 tables not row-aligned')
    L8 = L8.assign(tk=tk.tk.values, Zr=ZA.Z.values); del tk, ZA
    b = json.loads((NOV8 / 'booktime.json').read_text()); t = json.loads((NOV8 / 'timing.json').read_text())
    env = dict(h=float(b['E2']['p90']), q01=float(t['offset_ms_pct']['p01']), q99=float(t['offset_ms_pct']['p99']))
    KE = case_linkage(K, L8, pd.read_parquet(NOV8 / 'pend_inc.parquet', columns=['mid', 't_ms', 'ptick', 'inc']), env)
    ref = dict(rest_floor_nonoverlap=rep['timestamp_checks']['rest_floor_nonoverlap'], matched_floor_nonoverlap=rep['timestamp_checks']['matched_floor_nonoverlap'])
    return K, KE, ref, dict(inputs=inputs, env=env)


def case_linkage(K, L8, P8, env, day='20261108'):
    """Per known leg (K: CKEY, candidate_legs): its candidate rows in the day's legs (L8: CKEY, how, t_ms, t_e2, tk, Zr - the pinned engine
    anchor per row) by the 11-09 audit's content key; per candidate row its event (taker order tk, else -1 - row) with the ledger's S1 / S2 /
    S3 / EV windows (events_of over the event's legs, S3 from sem_windows on the whole day), the row's own windows (its source alone, capped
    by its REST second), its source, the replay's point (anchor - 50 ms) and the anchor's origin by the engine's rule - the event's earliest
    WS pending time, else its earliest E2 estimate, else the row's REST second - with whether that rule reproduces the pinned anchor. -> KE."""
    L8 = L8.reset_index(drop=True); L8 = L8.assign(row_id=np.arange(len(L8))); L8['group'] = np.where(L8.tk.values >= 0, L8.tk.values, -1 - L8.row_id.values)
    M_ = K.reset_index(drop=True).assign(i=np.arange(len(K)))[['i'] + CKEY].merge(L8[CKEY + ['row_id']], on=CKEY, how='inner').sort_values(['i', 'row_id']).reset_index(drop=True)
    C.require(np.array_equal(np.bincount(M_.i.values, minlength=len(K)), K.candidate_legs.values.astype(int)), 'candidate rows differ from the 11-09 audit\'s counts')
    A_ = pd.concat([L8, sem_windows(L8, P8)], axis=1).assign(day=day, h=env['h'], q01=env['q01'], q99=env['q99']); grp = A_.group.values; r = M_.row_id.values
    A_['ST_llo'], A_['ST_lhi'] = A_.S3_llo, A_.S3_lhi                                  # (unused here: K8 gives each known leg its own settlement window)
    LE = A_[np.isin(grp, np.unique(grp[r]))].copy(); LE['anc'] = LE.groupby('group').Zr.transform('min').fillna(-1.)
    EVT = events_of(LE); own = A_.iloc[r]; _, W = leg_windows(own); cap = (own.exec_s.values.astype(float) + 1.) * 1000.
    e = pd.MultiIndex.from_frame(EVT[['day', 'group']]).get_indexer(pd.MultiIndex.from_arrays([np.full(len(r), day), grp[r]])); C.require(bool((e >= 0).all()), 'a candidate row without its event')
    ws = pd.Series(np.where(LE.how.values > 0, LE.t_ms.values.astype(float), np.nan)).groupby(LE.group.values).min()
    e2 = pd.Series(np.where((LE.how.values == 0) & np.isfinite(LE.t_e2.values.astype(float)), LE.t_e2.values.astype(float), np.nan)).groupby(LE.group.values).min()
    wmin, emin = ws.reindex(grp[r]).values, e2.reindex(grp[r]).values; how = own.how.values.astype(int); t_ms = own.t_ms.values.astype(float); t_e2 = own.t_e2.values.astype(float)
    Srow = own.exec_s.values.astype(float) * 1000.; rule = np.where(np.isfinite(wmin), wmin, np.where(np.isfinite(emin), emin, Srow))
    origin = np.where(np.isfinite(wmin), 'pending_increment', np.where(np.isfinite(emin), 'book_drop', 'REST_second'))
    a_own = np.where(origin == 'pending_increment', (how > 0) & (t_ms == wmin), np.where(origin == 'book_drop', (how == 0) & (t_e2 == emin), True))
    ownt = np.where(how > 0, t_ms, np.where(np.isfinite(t_e2), t_e2, Srow)); Zr = own.Zr.values.astype(float)
    KE = pd.DataFrame(dict(i=M_.i.values, row_id=r, group=grp[r], e=e, src=leg_source(how, t_e2), Zr=Zr, origin=origin, anchor_own=a_own, anchor_rule_ok=rule == Zr,
                           how=how, t_ms=t_ms, t_e2=t_e2, S3_code=own.S3_code.values, S3_ncand=own.S3_ncand.values, link_late=own.link_late.values,
                           event_legs=EVT.legs.values[e], event_ws_span=EVT.ws_span.values[e]))
    for k in ('S1', 'S2', 'S3'):
        KE[f'{k}_lo'], KE[f'{k}_hi'], KE[f'{k}_st'] = EVT[f'{k}_lo'].values[e], EVT[f'{k}_hi'].values[e], EVT[f'{k}_st'].values[e]
        KE[f'own_{k}_lo'], KE[f'own_{k}_hi'] = W[k][0], np.minimum(W[k][1], cap)
    KE['EV_lo'], KE['EV_hi'], KE['EV_st'], KE['own_EV_lo'], KE['own_EV_hi'] = -np.inf, EVT.EV_hi.values[e], 0, -np.inf, cap
    KE['M50_lo'] = KE['M50_hi'] = Zr - SHIFT; KE['M50_st'] = 0; KE['own_M50_lo'] = KE['own_M50_hi'] = ownt - SHIFT
    # v6: the event's lower end split by the clock domain of the legs that set it (True: V_book - WS pending / E2 book-drop times; else V_status -
    # REST seconds); its max reproduces the event's start exactly
    cls_le, W_le = leg_windows(LE); lo0_le = rest_range(LE.exec_s.values)[0]
    dom = {'S1': cls_le < 2, 'S2': cls_le < 2, 'S3': LE.S3_code.isin([0, 1]).values & (LE.S3_llo.values > lo0_le)}
    ie = pd.MultiIndex.from_frame(EVT[['day', 'group']]).get_indexer(pd.MultiIndex.from_frame(LE[['day', 'group']])); C.require(bool((ie >= 0).all()), 'a leg without its event')
    for k, bk in dom.items():
        lo_ = np.asarray(W_le[k][0], float); lb = np.full(len(EVT), -np.inf); ls = np.full(len(EVT), -np.inf)
        np.maximum.at(lb, ie[bk], lo_[bk]); np.maximum.at(ls, ie[~bk], lo_[~bk])
        C.require(bool(np.array_equal(np.maximum(lb, ls), EVT[f'{k}_lo'].values.astype(float))), f'{k}: the domain split does not reproduce the event start')
        KE[f'{k}_lo_book'], KE[f'{k}_lo_status'] = lb[e], ls[e]
    return KE


def miss(lo, hi, K):
    """Windows [lo, hi] that miss the leg's lifecycle [place, upper] entirely."""
    return (hi < K.place.values) | (lo > K.upper.values)


def interval_status(lo, hi, R):
    """An interval [lo, hi] against a declared range R: CONTRADICTED (disjoint), CERTIFIED (inside it), COMPATIBLE-UNIDENTIFIED (overlapping, not inside)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.where((hi < R[0]) | (lo > R[1]), 'CONTRADICTED', np.where((lo >= R[0]) & (hi <= R[1]), 'CERTIFIED', 'COMPATIBLE-UNIDENTIFIED'))


def order_path(K, mlo=None, mhi=None):
    """K5 per UNIQUE order (run, cid; a leg is not the unit): activation - place in [signed - place, accepted - place] (live after signing;
    premise: live before its Accepted answer is received) against ACT_RANGE, the cancel effect - request in [0, confirmation - request]
    (premise: the effect precedes its first confirmation - the HTTP Canceled answer or the user-stream cancel - being received) against
    CANCEL_RANGE. A confirmation bounds the effect from above only: a late one is not a slow effect. Its source, the user-stream receipt, the
    core's CancelDone and a throttle are kept apart; a confirmation before its request (core and kernel clocks disagree) is a status. With the
    fills' match bounds on the bot clock (mlo / mhi per leg, K8's lifecycle plus shared-clock constraints): activation <= its first match, and the cancel takes effect after every match
    of the order (a cancelled order cannot fill) - the _set columns and counts."""
    U = K.drop_duplicates(['run', 'cid']).reset_index(drop=True); pl = U.place.values.astype(float)
    a_lo = np.where(np.isfinite(U.signed.values.astype(float)), np.maximum(U.signed.values.astype(float) - pl, 0.), 0.); a_hi = U.accepted.values.astype(float) - pl
    act = np.where(np.isfinite(a_hi), interval_status(a_lo, a_hi, ACT_RANGE), 'NO-ACCEPTED-ANSWER')
    req, ack, uc = U.req.values.astype(float), U.ack.values.astype(float), U.ucancel.values.astype(float); c_hi = ack - req; has = np.isfinite(c_hi)
    can = np.where(~np.isfinite(ack), 'NO-CANCEL-CONFIRMATION', np.where(~np.isfinite(req), 'NO-CANCEL-REQUEST', np.where(c_hi < 0, 'INCONSISTENT-CLOCKS',
                                                                                                                       interval_status(np.zeros(len(U)), c_hi, CANCEL_RANGE))))
    src = np.where(~np.isfinite(ack), '', np.where(~np.isfinite(uc) | (ack < uc - 1e-6), 'http_answer', 'user_stream'))
    P = pd.DataFrame(dict(run=U.run.values, cid=U.cid.values, act_lo=a_lo, act_hi=a_hi, act_status=act, cancel_lo=np.where(has, 0., np.nan), cancel_hi=c_hi, cancel_status=can,
                          confirm_source=src, user_stream_after_req=uc - req, core_done_after_req=U.cdone.values.astype(float) - req, throttled=np.isfinite(U.thr.values.astype(float))))
    cnt = lambda x: {str(s): int((x == s).sum()) for s in sorted(set(x.tolist()))}
    if mlo is not None:                                       # the settlement bounds: per order the earliest match's upper end, the latest match's lower end
        g = pd.DataFrame(dict(run=K.run.values, cid=K.cid.values, mlo=np.asarray(mlo, float), mhi=np.asarray(mhi, float))).groupby(['run', 'cid'])
        key = pd.MultiIndex.from_arrays([U.run.values, U.cid.values]); first_hi = g.mhi.min().reindex(key).values; last_lo = g.mlo.max().reindex(key).values
        a_hi_s = np.fmin(a_hi, first_hi - pl); c_lo_s = np.where(has, np.maximum(0., last_lo - req), np.nan)
        P['act_hi_set'] = a_hi_s; P['act_status_set'] = np.where(np.isfinite(a_hi_s), interval_status(a_lo, a_hi_s, ACT_RANGE), 'NO-BOUND')
        P['cancel_lo_set'] = c_lo_s; P['cancel_status_set'] = np.where(has & (c_hi >= 0), interval_status(np.nan_to_num(c_lo_s), c_hi, CANCEL_RANGE), can)
    ch = c_hi[has & (c_hi >= 0)]; pc = {f'p{q}': float(np.percentile(ch, q)) for q in (50, 95)} if len(ch) else {}
    k5 = dict(orders=len(P), legs=len(K), activation=cnt(act), cancel=cnt(can), declared_activation=list(ACT_RANGE), declared_cancel=list(CANCEL_RANGE),
              confirmation_after_request_ms=dict(n=int(len(ch)), **pc, **({'max': float(ch.max())} if len(ch) else {})), confirmation_source=cnt(src[has]),
              throttled=int(P.throttled.sum()), smallest_activation_upper_ms=float(np.nanmin(a_hi)) if np.isfinite(a_hi).any() else None)
    if mlo is not None:
        cl = P.cancel_lo_set.values[np.isfinite(P.cancel_lo_set.values)]
        k5.update(activation_with_settlement=cnt(P.act_status_set.values), cancel_with_settlement=cnt(P.cancel_status_set.values),
                  orders_with_a_match_after_the_request=int((cl > 0).sum()), largest_cancel_lower_bound_ms=float(cl.max()) if len(cl) else None,
                  smallest_activation_upper_with_settlement_ms=float(np.nanmin(P.act_hi_set.values)) if np.isfinite(P.act_hi_set.values).any() else None)
    return P, k5


def offset_interval(K):
    """K7: the offset d = venue status clock - bot clock consistent with every own leg (identity-safe: the notices are our own): a notice is
    received after its status second starts (d >= stamp - receipt, Matched and Confirmed) and the match follows the placement and precedes
    the end of its Matched second (d <= (M + 1) * 1000 - place). -> dict(lo, hi, exists, includes_zero, counts)."""
    a = np.r_[K.M.values.astype(float) * 1000. - K.rM.values.astype(float), K.Cs.values.astype(float) * 1000. - K.rC.values.astype(float)]
    b = (K.M.values.astype(float) + 1.) * 1000. - K.place.values.astype(float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    lo = float(a.max()) if len(a) else -np.inf; hi = float(b.min()) if len(b) else np.inf
    return dict(lo=lo, hi=hi, exists=bool(lo <= hi), includes_zero=bool(lo <= 0. <= hi), receipt_bounds=int(len(a)), placement_bounds=int(len(b)))


def settlement_checks(K, off):
    """K8: t_set_i = match_i_bot + delta + lag_i, with one clock delta and independent lag_i >= 0.
    Premises: UUID created at/after matching; UUID/status clock aligned; K7 bounds delta. A feasible common total delay is only a
    diagnostic/conditional scenario. It cannot bound individual lags. Known matches retain their lifecycle lower bounds; public matches
    acquire no lower bound. Return (R, known match lows, known match highs, conditional zero-lag clock interval).
    """
    t = K.t_set.values.astype(float); place = K.place.values.astype(float); upper = K.upper.values.astype(float)
    v7 = np.isfinite(t); fl = v7 & (np.floor(t / 1000.) == K.M.values.astype(float))
    lo_, hi_ = t - upper, t - place
    per_run = {}
    for r in sorted(K.run.unique()):
        m = (K.run.values == r) & v7
        per_run[str(r)] = dict(lo=float(lo_[m].max()), hi=float(hi_[m].min()), exists=bool(lo_[m].max() <= hi_[m].min()), legs=int(m.sum())) if m.any() else dict(exists=False, legs=0)
    d_lo, d_hi = (float(np.nanmax(lo_)), float(np.nanmin(hi_))) if v7.any() else (np.inf, -np.inf)
    notice = K.notice.values.astype(float)
    complete = bool(v7.all() and len(K) and np.isfinite(place).all() and np.isfinite(upper).all() and np.isfinite(notice).all())
    # The identity-linked first notice already contains this UUID: creation precedes receipt.
    clock_lo = max(float(off['lo']), float(np.nanmax(t - notice))) if off['exists'] and complete else np.nan
    clock_hi = min(float(off['hi']), d_hi) if off['exists'] else np.nan
    ok = bool(complete and off['exists'] and np.isfinite(clock_lo) and np.isfinite(clock_hi)
              and clock_lo <= clock_hi and np.all(place <= upper))
    clock = dict(lo=clock_lo, hi=clock_hi, exists=ok, includes_zero=bool(ok and clock_lo <= 0. <= clock_hi))
    R = dict(model=SETTLEMENT_MODEL, legs=len(K), v7=int(v7.sum()), floor_is_matched_second=int(fl.sum()),
             after_place_at_0=int((t > place).sum()), before_notice_at_0=int((t < K.notice.values).sum()),
             inside_lifecycle_at_0=int(((t >= place) & (t <= upper)).sum()),
             d=dict(lo=d_lo, hi=d_hi, exists=bool(d_lo <= d_hi)), d_per_run=per_run,
             common_d_role='conditional constant-total-delay fit, not the event-lag model or an acceptance gate',
             consistent=ok, status='COMPATIBLE' if ok else ('CONTRADICTED' if complete and off['exists'] else 'UNIDENTIFIED'),
             clock_offset=clock, lag_bound_ms=None, universal_lag_status='UNIDENTIFIED',
             assumptions=['UUID creation at or after matching', 'UUID and status stamps share a clock',
                          'the identity-linked first notice already contains this UUID', 'K7 clock and lifecycle premises'],
             evidence_gate='NOT_ESTABLISHED_BY_SAME_SAMPLE_COMPATIBILITY')
    off8 = dict(lo=np.nan, hi=np.nan, exists=False, includes_zero=False)
    if not ok:
        return R, None, None, off8
    mlo, mhi = place.copy(), np.minimum(upper, t - clock_lo)
    lag_lo = np.maximum(0., t - clock_hi - upper); lag_hi = t - clock_lo - place
    nd_lo, nd_hi = K.notice.values.astype(float) - mhi, K.notice.values.astype(float) - mlo
    ns = interval_status(nd_lo, nd_hi, (KP.NOTICE, KP.NOTICE))
    zlo, zhi = max(d_lo, clock_lo), min(d_hi, clock_hi)
    off8 = dict(lo=zlo, hi=zhi, exists=bool(zlo <= zhi), includes_zero=bool(zlo <= 0. <= zhi))
    R.update(known_leg_lag_ms=dict(lo=lag_lo.tolist(), hi=lag_hi.tolist(), scope='these identified lifecycles only; not a public-trade bound'),
             offset_settlement_premise=dict(off8, premise='CONDITIONAL: zero UUID creation lag on every leg'),
             notice_delay_ms=dict(lo_min=float(nd_lo.min()), lo_p50=float(np.median(nd_lo)), hi_p50=float(np.median(nd_hi)), hi_max=float(nd_hi.max()),
                                  vs_replay_point={str(x): int((ns == x).sum()) for x in sorted(set(ns.tolist()))}, replay_point_ms=KP.NOTICE))
    return R, mlo, mhi, off8


def window_checks(K, KE, off, scen=KSC, tag=''):
    """K6 on every candidate row of every known leg: the ledger's windows (S1 / S2 / S3 / EV) and the replay's own point (M50) against the
    leg's lifecycle [place, upper] and its actual cancel window (confirmation c_hi; bot clock), at offset 0 and over the admissible offsets
    off (venue - bot, K7). Per leg over its candidates: the verdict they agree on, else ambiguous; CANCEL_FIRST only if every candidate is
    (a false certified avoidance: the fill happened); a miss's cause - the row's own window fits the lifecycle -> event_proxy (the event's
    other legs or its anchor decide), else the row's own source; a contradictory event its status; disagreeing causes -> unresolved_linkage.
    -> (per scenario counts, per-leg columns)."""
    n = len(K); i = KE.i.values.astype(np.int64); nk = np.bincount(i, minlength=n); has = nk > 0
    place, upper, c_hi = K.place.values[i].astype(float), K.upper.values[i].astype(float), K.ack.values[i].astype(float); can = np.isfinite(c_hi)
    dl, dh = (off['lo'], off['hi']) if off['exists'] else (0., 0.)

    def agree(x, fill):                                       # the value every candidate of a leg agrees on, else fill; -9 without a candidate
        lo_ = np.full(n, 10 ** 6); hi_ = np.full(n, -10 ** 6); np.minimum.at(lo_, i, x); np.maximum.at(hi_, i, x)
        return np.where(~has, -9, np.where(lo_ == hi_, lo_, fill))

    def every(b):
        return has & (np.bincount(i, weights=np.asarray(b, float), minlength=n) == nk)
    out, cols = {}, {}
    for k in scen:
        lo, hi, st = KE[f'{k}_lo'].values.astype(float), KE[f'{k}_hi'].values.astype(float), KE[f'{k}_st'].values.astype(int)
        v0 = np.where(st == 2, 3, np.where(st == 3, 4, np.where(hi < place, 1, np.where(lo > upper, 2, 0))))
        vr = np.where(st >= 2, v0, np.where(hi - dl < place, 1, np.where(lo - dh > upper, 2, 0)))           # missed at EVERY admissible offset
        ok = can & (st < 2); cf0, cfw, cfr = ok & (c_hi < lo), ok & (c_hi < lo - dl), ok & (c_hi < lo - dh)
        own = (KE[f'own_{k}_hi'].values.astype(float) >= place) & (KE[f'own_{k}_lo'].values.astype(float) <= upper)
        cz = np.where(v0 == 0, -1, np.where(v0 == 3, 5, np.where(v0 == 4, 6, np.where(own, 0, 1 + KE.src.values.astype(int)))))
        v0l, vrl = (np.where(x == -9, 6, x) for x in (agree(v0, 5), agree(vr, 5))); czl = agree(cz, 7)
        czl = np.where((v0l == 0) | (v0l == 6), -1, np.where(v0l == 5, 7, czl)); f0, fw, fr = every(cf0), every(cfw), every(cfr)
        out[k] = dict(legs=n, verdict_at_offset_0={VERD[x]: int((v0l == x).sum()) for x in range(len(VERD))},
                      verdict_every_offset={VERD[x]: int((vrl == x).sum()) for x in range(len(VERD))},
                      false_cancel_first=dict(at_offset_0=int(f0.sum()), some_offset=int(fw.sum()), every_offset=int(fr.sum())),
                      causes={CAUSES[c]: int((czl == c).sum()) for c in range(len(CAUSES))}, cancelled_legs=int(np.isfinite(K.ack.values.astype(float)).sum()))
        cols.update({f'{k}_verdict{tag}': np.array(VERD)[v0l], f'{k}_verdict_every_offset{tag}': np.array(VERD)[vrl],
                     f'{k}_cause{tag}': np.where(czl >= 0, np.array(CAUSES)[np.clip(czl, 0, None)], ''), f'{k}_false_cf_0{tag}': f0, f'{k}_false_cf_some_offset{tag}': fw})
    return out, cols


def window_checks_dom(K, KE, dlb, dls, scen=KSC, tag='_dom'):
    """K6 PER CLOCK DOMAIN (known v6, the gate of record; USER "Go" after the timing case audit): a known fill is called CANCEL_FIRST at some
    admissible offset iff its cancel confirmation c_hi (bot clock) precedes max(lo_book - dlb, lo_status - dls) on every candidate row - each
    lower end mapped with the lowest admissible offset OF ITS OWN CLOCK: dlb the leg's book-clock bound B (venue book - bot, from the bot's own
    frame receipts; NaN: unidentified, so a finite book end reads +inf - fail-closed), dls K7's status-clock lower end. A row's lower ends by
    domain: S1 / S2 / S3 from case_linkage (WS and E2 legs on V_book; imputed legs and REST ranges on V_status); M50 by its anchor's origin;
    ST / EV have none. -> (per scenario counts, per-leg columns)."""
    n = len(K); i = KE.i.values.astype(np.int64); nk = np.bincount(i, minlength=n); has = nk > 0
    c_hi = K.ack.values[i].astype(float); can = np.isfinite(c_hi); db = np.asarray(dlb, float)[i]

    def every(b):                                             # every candidate row of the leg (a leg without one: False)
        cnt = np.bincount(i, weights=np.asarray(b, float), minlength=n); return has & (cnt == nk)

    def any_(b):
        return np.bincount(i, weights=np.asarray(b, float), minlength=n) > 0
    out, cols = {}, {}
    for k in scen:
        lo, st = KE[f'{k}_lo'].values.astype(float), KE[f'{k}_st'].values.astype(int)
        if f'{k}_lo_book' in KE:
            lb, ls = KE[f'{k}_lo_book'].values.astype(float), KE[f'{k}_lo_status'].values.astype(float)
        elif k == 'M50':
            bk = KE.origin.values != 'REST_second'; lb, ls = np.where(bk, lo, -np.inf), np.where(bk, -np.inf, lo)
        else:
            C.require(bool(np.isneginf(lo).all()), f'{k}: a lower end without a clock domain'); lb = ls = lo
        C.require(bool(np.array_equal(np.maximum(lb, ls), lo)), f'{k}: the domain split does not reproduce the window start')
        mb = np.where(np.isneginf(lb), -np.inf, np.where(np.isfinite(db), lb - np.nan_to_num(db), np.inf))
        ms = np.where(np.isneginf(ls), -np.inf, ls - dls)
        ok = can & (st < 2); f0 = every(ok & (c_hi < lo)); fd = every(ok & (c_hi < np.maximum(mb, ms)))
        bo = any_(ok & np.isfinite(lb))
        out[k] = dict(legs=n, false_cancel_first=dict(at_offset_0=int(f0.sum()), some_offset_per_domain=int(fd.sum())),
                      legs_with_a_book_lower_end=int(bo.sum()), of_them_book_bound_unidentified=int((bo & ~np.isfinite(np.asarray(dlb, float))).sum()))
        cols[f'{k}_false_cf{tag}'] = fd
    return out, cols


def known_checks(K, ref, KE=None, draws=200, seed=20261118, book=None):
    """K0-K7 on known own legs (K: run, cid, place, accepted, upper, notice, ack, req, S, M, Z, signed, ucancel, cdone, thr, rM, rC, Cs; KE: the
    case rows of case_linkage, or None - K6 not run; ref: the 11-09 counts) -> (R, readings); R carries the per-order and per-leg frames."""
    lo, hi = window_from('lifecycle', (K.place.values, K.upper.values)); can = np.isfinite(K.ack.values)
    c_lo = np.where(np.isfinite(K.req.values), K.req.values, K.place.values); c_hi = K.ack.values
    rest_pt = (K.S.values * 1000., K.S.values * 1000. + 999.999); rest_ev = window_from('rest_confirmed_s', K.S.values); mt = window_from('matched_status_s', K.M.values)
    k0 = dict(rest_floor_nonoverlap=int(miss(*rest_pt, K).sum()), matched_floor_nonoverlap=int(miss(*mt, K).sum()))
    c1 = classify(lo[can], hi[can], c_lo[can], c_hi[can]); k1 = {ORD[c]: int((c1 == c).sum()) for c in range(len(ORD))}
    try:
        window_from('rest_confirmed_s', K.S.values, point=True); refused = False
    except ValueError:
        refused = True
    c2 = classify(rest_pt[0][can], rest_pt[1][can], c_lo[can], c_hi[can])
    k2 = dict(point_contradictions=int(miss(*rest_pt, K).sum()), evidence_contradictions=int(miss(*rest_ev, K).sum()), point_refused=refused,
              point_cancel_first=int((c2 == 0).sum()), evidence_classes={ORD[c]: int((v == c).sum()) for v in [classify(rest_ev[0][can], rest_ev[1][can], c_lo[can], c_hi[can])]
                                                                         for c in range(len(ORD))})
    rng = np.random.default_rng(seed); hits = []
    for _ in range(draws):
        Kp = K.copy()
        for r in Kp.run.unique():
            ix = np.flatnonzero(Kp.run.values == r); pm = rng.permutation(ix); Kp.loc[Kp.index[ix], ['place', 'upper']] = K[['place', 'upper']].values[pm]
        hits.append(int(miss(*mt, Kp).sum()))
    hits = np.array(hits); k3 = dict(true_contradictions=k0['matched_floor_nonoverlap'], draws=draws, share_draws_firing=float((hits >= 1).mean()), median_hits=float(np.median(hits)))
    k4, k4c = {}, {}
    ok = np.isfinite(K.Z.values)                                                  # legs without an engine anchor are a status, not a test
    for nm, w in (('S1', 250.), ('S2', 600.)):
        zl, zh = K.Z.values - w, K.Z.values + w; c4 = classify(zl[can & ok], zh[can & ok], c_lo[can & ok], c_hi[can & ok])
        feas = {}
        for r in sorted(K.run.unique()):
            x = (K.run.values == r) & ok; dlo = float((K.place.values[x] - zh[x]).max()); dhi = float((K.upper.values[x] - zl[x]).min())
            feas[str(r)] = dict(lo=dlo, hi=dhi, exists=bool(dlo <= dhi), legs=int(x.sum()))
        mi = miss(zl, zh, K) & ok; cf_ = np.zeros(len(K), bool); cf_[np.flatnonzero(can & ok)] = c4 == 0
        k4c.update({f'K4_{nm}_missed': mi, f'K4_{nm}_cancel_first': cf_})
        k4[nm] = dict(anchored=int(ok.sum()), missed=int(miss(zl, zh, K)[ok].sum()), cancel_first=int((c4 == 0).sum()), no_anchor=int((~ok).sum()), constant_offset=feas)
    k4c.update(K4_anchor_minus_place_ms=K.Z.values - K.place.values, K4_anchor_minus_upper_ms=K.Z.values - K.upper.values)
    read = {'K0 REPRODUCES 11-09': 'YES' if k0 == ref else 'NO',
            'K1 CLASSIFIER-CONSISTENCY': 'YES' if k1['CANCEL_FIRST'] == 0 and k1['CONTRADICTORY'] == 0 and bool(np.all(lo <= hi)) else 'NO',
            'K2 DELAYED-STATUS-REFUSED': 'YES' if k2['point_contradictions'] >= 1 and k2['evidence_contradictions'] == 0 and refused else 'NO',
            'K3 IDENTITY-CHECK-FIRES': 'YES' if k3['true_contradictions'] == 0 and k3['share_draws_firing'] >= .95 else 'NO',
            'K4 REPLAY-ANCHOR-VS-LIFECYCLE': 'CONSISTENT' if all(v['missed'] == 0 and v['cancel_first'] == 0 for v in k4.values()) else 'CONTRADICTED'}
    off = offset_interval(K); read['K7 CLOCK OFFSET (venue status - bot)'] = (f"[{off['lo']:+.1f}, {off['hi']:+.1f}] ms, " + ('CONTAINS 0' if off['includes_zero'] else 'EXCLUDES 0')
                                                                             if off['exists'] else 'EMPTY')
    k8, mlo, mhi, off8 = settlement_checks(K, off) if 't_set' in K else (None, None, None, dict(exists=False))
    PO, k5 = order_path(K, mlo, mhi)
    read['K5 ORDER-PATH INTERVALS vs DECLARED RANGES'] = ('CONTRADICTED' if k5['activation'].get('CONTRADICTED', 0) + k5['cancel'].get('CONTRADICTED', 0) > 0
                                                          else 'NOT-CONTRADICTED')
    if k8 is not None:
        read['K8 SETTLEMENT TIME'] = k8['status']
        if k8['consistent']:
            o8 = k8['offset_settlement_premise']; nd = k8['notice_delay_ms']['vs_replay_point']
            read['K8 CLOCK OFFSET (settlement premise: the id stamped at the match)'] = ('CONDITIONAL ' + str(o8))
            read['K8 UNIVERSAL CREATION-LAG BOUND'] = k8['universal_lag_status']
            read['K8 FILL-NOTICE DELAY vs the replay 100 ms'] = (f"CONTRADICTED on {nd.get('CONTRADICTED', 0)} of {k8['legs']} fills" if nd.get('CONTRADICTED', 0) else 'NOT-CONTRADICTED')
            read['K5S ORDER PATH WITH SETTLEMENT TIMES'] = ('CONTRADICTED' if k5['activation_with_settlement'].get('CONTRADICTED', 0) + k5['cancel_with_settlement'].get('CONTRADICTED', 0) > 0
                                                         else 'NOT-CONTRADICTED')
    st_ok = KE is not None and k8 is not None and k8['consistent']
    if st_ok:                                                 # Upper-only UUID windows; do not fit a public lower bound to these lifecycles.
        tl = K.t_set.values.astype(float)[KE.i.values.astype(np.int64)]
        hi_ = np.minimum(tl, KE.EV_hi.values.astype(float))
        KE = KE.assign(ST_lo=-np.inf, ST_hi=hi_, ST_st=0, own_ST_lo=-np.inf, own_ST_hi=hi_)
    scen_ = tuple(k for k in KSC if k != 'ST' or st_ok)
    # The declared gate stays at K7's offsets (its own premises). K8's shared clock is narrower (the first notice carrying each UUID, the
    # placements) but holds only if the UUID clock is the windows' clock (REST status / WS book): a separate, conditional reading.
    clk = k8['clock_offset'] if k8 is not None and k8['consistent'] else dict(exists=False)
    k6, cols = window_checks(K, KE, off, scen_) if KE is not None else (None, {})
    k6c, cols_c = window_checks(K, KE, clk, scen_, tag='_clk') if KE is not None and clk['exists'] else (None, {})
    k6s, cols_s = window_checks(K, KE, off8, scen_, tag='_set') if KE is not None and off8['exists'] else (None, {})
    # v6: the gate of record per clock domain - book endpoints at the leg's own book-clock bound (the census), status endpoints at K7
    k6d, cols_d = (window_checks_dom(K, KE, book, off['lo'] if off['exists'] else 0., scen_) if KE is not None and book is not None else (None, {}))
    if k6 is not None:
        for k in GATE:
            read[f'K6 GATE {k}'] = ('PASS' if k6[k]['false_cancel_first']['some_offset'] == 0 else 'FAIL') if k in k6 else 'NOT-RUN'
            if k6c is not None:
                read[f'K6 GATE {k} (K8 shared clock)'] = ('PASS' if k6c[k]['false_cancel_first']['some_offset'] == 0 else 'FAIL') if k in k6c else 'NOT-RUN'
            if k6s is not None:
                read[f'K6 GATE {k} (settlement offsets)'] = ('PASS' if k6s[k]['false_cancel_first']['some_offset'] == 0 else 'FAIL') if k in k6s else 'NOT-RUN'
        if k6d is not None:
            for k in GATE:
                read[f'K6 GATE {k} (per clock domain)'] = ('UNIDENTIFIED-UPPER-BOUND-ONLY' if k == 'ST' and st_ok else 'NOT-RUN' if k not in k6d
                                                           else 'PASS' if k6d[k]['false_cancel_first']['some_offset_per_domain'] == 0 else 'FAIL')
        if st_ok:
            read['K6 GATE ST'] = 'UNIDENTIFIED-UPPER-BOUND-ONLY'
            if k6c is not None:
                read['K6 GATE ST (K8 shared clock)'] = 'UNIDENTIFIED-UPPER-BOUND-ONLY'
            if k6s is not None:
                read['K6 GATE ST (settlement offsets)'] = 'UNIDENTIFIED-UPPER-BOUND-ONLY'
        m = k6['M50']; read['K6 REPLAY POINT M50'] = ('CONSISTENT' if m['false_cancel_first']['at_offset_0'] == 0 and m['verdict_at_offset_0']['missed_before']
                                                      + m['verdict_at_offset_0']['missed_after'] == 0 else 'CONTRADICTED')
    inst = all(read[x] == 'YES' for x in ('K0 REPRODUCES 11-09', 'K1 CLASSIFIER-CONSISTENCY', 'K2 DELAYED-STATUS-REFUSED', 'K3 IDENTITY-CHECK-FIRES'))
    gate = ', '.join(f"{k} {read[f'K6 GATE {k}']}" for k in GATE) if k6 is not None else 'NOT-RUN'
    gate_c = ', '.join(f"{k} {read[f'K6 GATE {k} (K8 shared clock)']}" for k in GATE) if k6c is not None else 'NOT-RUN'
    gate_s = ', '.join(f"{k} {read[f'K6 GATE {k} (settlement offsets)']}" for k in GATE) if k6s is not None else 'NOT-RUN'
    if k6d is not None:                                       # v6: the gate of record per clock domain; K7 on every endpoint is v5's reading
        gate_d = ', '.join(f"{k} {read[f'K6 GATE {k} (per clock domain)']}" for k in GATE)
        win = ('; LEDGER-WINDOW GATE OF RECORD (per clock domain: book ends at the leg\'s own book-clock bound, status ends at K7) ' + gate_d
               + '; v5 basis - K7 offsets on every end: ' + gate + '; conditional - K8 shared clock: ' + gate_c + '; settlement offsets: ' + gate_s)
    else:
        win = '; LEDGER-WINDOW GATE ' + gate + ' (K7 offsets; conditional - K8 shared clock: ' + gate_c + '; settlement offsets: ' + gate_s + ')'
    read['KNOWN VERDICT'] = (('INSTRUMENT-CHECKS-PASS' if inst else 'INSTRUMENT-CHECK-FAILED') + '; REPLAY-ANCHOR-' + read['K4 REPLAY-ANCHOR-VS-LIFECYCLE']
                             + '; SETTLEMENT-TIME ' + read.get('K8 SETTLEMENT TIME', 'NOT-RUN') + win
                             + '; ORDER-PATH RANGES ' + read['K5 ORDER-PATH INTERVALS vs DECLARED RANGES'] + '; NOT A VENUE-ENGINE ACCEPTANCE')
    k8c = dict(K8_match_lo_bot=mlo, K8_match_hi_bot=mhi, K8_creation_lag_lo=k8['known_leg_lag_ms']['lo'], K8_creation_lag_hi=k8['known_leg_lag_ms']['hi']) if mlo is not None else {}
    R = dict(legs=len(K), orders=int(K.drop_duplicates(['run', 'cid']).shape[0]), cancelled=int(can.sum()), K0=k0, K0_ref=ref, K1=k1, K2=k2, K3=k3, K4=k4, K5=k5, K6=k6,
             K6_settlement_offsets=k6s, K6_shared_clock=k6c, K6_per_domain=k6d, K7=off, K8=k8, _orders=PO,
             _legs=dict(k4c, **cols, **cols_c, **cols_s, **cols_d, **({'K6D_book_bound_ms': np.asarray(book, float)} if book is not None else {}), **k8c), _KE=KE)
    if KE is not None:
        R['linkage'] = dict(candidate_rows=int(len(KE)), legs_with_several=int((np.bincount(KE.i.values, minlength=len(K)) > 1).sum()),
                            anchor_rule_reproduced=int(KE.anchor_rule_ok.sum()), origin={str(x): int(y) for x, y in KE.origin.value_counts().items()},
                            anchor_from_own_row=int(KE.anchor_own.sum()), s3_codes={SEM_CODES[c]: int((KE.S3_code.values == c).sum()) for c in range(len(SEM_CODES))},
                            link_late=int(KE.link_late.sum()))
    return R, read


def known_report(R, read, KC=None):
    L = [('KNOWN-ORDER CASE LEDGER v6 (pf_keep_pull_exec_windows.py known; the 11-09 fill-clock audit\'s confirmed own maker legs, consumed Nov 8; the gate of record per clock '
          'domain after the timing case audit)' if R.get('K6_per_domain') is not None else 'KNOWN-ORDER CASE LEDGER v5 (pf_keep_pull_exec_windows.py known; the 11-09 '
          'fill-clock audit\'s confirmed own maker legs, consumed Nov 8; review of 1fd77004 + settlement times)'),
         f"  {R['legs']} legs of {R['orders']} orders: {R['K4']['S1']['anchored']} with an engine anchor, {R['K4']['S1']['no_anchor']} without (ambiguous leg clocks); "
         f"{R['cancelled']} legs on orders with a cancel confirmation. Bot clock: lifecycle, cancel request / confirmation, notice receipts; venue clocks: REST / Matched / "
         'Confirmed stamps, WS / E2 times and the windows',
         f"  K0 the 11-09 counts: {R['K0']} (11-09: {R['K0_ref']})",
         f"  K1 lifecycle match window vs the actual cancel window: {R['K1']}",
         f"  K2 REST executedAt as the match second: {R['K2']['point_contradictions']} lifecycles contradicted, {R['K2']['point_cancel_first']} known fills it would call "
         f"CANCEL_FIRST; as the evidence bound: {R['K2']['evidence_contradictions']} contradicted, classes {R['K2']['evidence_classes']}; refused as a point: {R['K2']['point_refused']}",
         f"  K3 Matched second vs lifecycle: true identity {R['K3']['true_contradictions']} contradicted; {100 * R['K3']['share_draws_firing']:.1f}% of {R['K3']['draws']} "
         f"within-run permutations contradict >= 1 (median {R['K3']['median_hits']:.0f})"]
    for nm, v in R['K4'].items():
        L.append(f"  K4 (v1's proxy) engine anchor Z +-{250 if nm == 'S1' else 600} ms on the {v['anchored']} anchored legs ({v['no_anchor']} without an anchor, not tested here): "
                 f"misses {v['missed']} lifecycles; would call {v['cancel_first']} known fills CANCEL_FIRST; constant offset per run {v['constant_offset']}")
    k5 = R['K5']
    L.append(f"  K5 order path per UNIQUE order ({k5['orders']} orders of {k5['legs']} legs): activation in [signed, accepted] - place vs {k5['declared_activation']} ms: "
             f"{k5['activation']} (smallest upper bound {k5['smallest_activation_upper_ms']} ms); cancel effect in [0, confirmation - request] vs {k5['declared_cancel']} ms: "
             f"{k5['cancel']}; confirmation after the request {k5['confirmation_after_request_ms']} (an upper bound on the effect, not its latency); first confirmation "
             f"{k5['confirmation_source']}; throttled orders {k5['throttled']}")
    k5 = R['K5']; o = R['K7']; L.append(f"  K7 clock offset venue status - bot (own notices and placements): [{o['lo']:+.1f}, {o['hi']:+.1f}] ms, exists {o['exists']}, contains 0 {o['includes_zero']}")
    k8 = R.get('K8')
    if k8 is not None:
        L.append(f"  K8 settlement time (the trade id's UUIDv7 ms) on {k8['legs']} legs: version 7 {k8['v7']}; floor = the Matched second {k8['floor_is_matched_second']}; "
                 f"after placement {k8['after_place_at_0']}, before the first notice {k8['before_notice_at_0']}, inside the lifecycle {k8['inside_lifecycle_at_0']} (offset 0); "
                 f"conditional constant-d fit {k8['d']} (per run {k8['d_per_run']}); independent-lag model {k8['status']}")
        if k8['consistent']:
            L.append(f"    -> universal lag bound UNIDENTIFIED; shared clock {k8['clock_offset']}; conditional zero-lag clock {k8['offset_settlement_premise']}; "
                     f"fill-notice delay {k8['notice_delay_ms']}")
            L.append(f"    -> K5 with lifecycle/clock/independent-lag bounds: activation {k5.get('activation_with_settlement')} (smallest upper bound {k5.get('smallest_activation_upper_with_settlement_ms')} ms); "
                     f"cancel {k5.get('cancel_with_settlement')} ({k5.get('orders_with_a_match_after_the_request')} orders matched after their request, largest lower bound "
                     f"{k5.get('largest_cancel_lower_bound_ms')} ms)")
    if R.get('K6') is not None:
        L.append(f"  case linkage: {R['linkage']}")
        for k, v in R['K6'].items():
            L.append(f"  K6 {k}: at offset 0 {v['verdict_at_offset_0']}; at every admissible offset {v['verdict_every_offset']}; known fills called CANCEL_FIRST {v['false_cancel_first']}; "
                     f"causes {v['causes']}")
        for k, v in (R.get('K6_shared_clock') or {}).items():
            L.append(f"  K6 {k} at the K8 shared-clock offsets (conditional: the UUID clock is the windows' clock): every admissible offset {v['verdict_every_offset']}; "
                     f"known fills called CANCEL_FIRST {v['false_cancel_first']}")
        for k, v in (R.get('K6_settlement_offsets') or {}).items():
            L.append(f"  K6 {k} at the settlement-premise offsets: every admissible offset {v['verdict_every_offset']}; known fills called CANCEL_FIRST {v['false_cancel_first']}")
        for k, v in (R.get('K6_per_domain') or {}).items():
            L.append(f"  K6 {k} PER CLOCK DOMAIN (the gate of record, v6: book ends at the leg's book-clock bound, status ends at K7): known fills called CANCEL_FIRST "
                     f"{v['false_cancel_first']}; legs with a book lower end {v['legs_with_a_book_lower_end']}, of them without an identified bound (fail-closed) "
                     f"{v['of_them_book_bound_unidentified']}")
    if KC is not None and 'K4_S1_missed' in KC:
        T_ = KC[KC.K4_S1_missed.values.astype(bool)].to_dict('records'); f_ = lambda r, c, fmt='{}': fmt.format(r[c]) if c in r and r[c] == r[c] else '-'
        L.append(f'  K4 S1 failures traced ({len(T_)}; identity, the anchor against the lifecycle, its origin, the ledger windows\' verdicts and causes):')
        for r in T_:
            L.append(f"    {f_(r, 'run')} cid {f_(r, 'cid')} trade {f_(r, 'trade_id')} tx {str(r.get('tx', '-'))[:12]}: Z - place {f_(r, 'K4_anchor_minus_place_ms', '{:+.0f}')} ms, "
                     f"Z - upper {f_(r, 'K4_anchor_minus_upper_ms', '{:+.0f}')} ms; Confirmed - Matched {f_(r, 'lag', '{:.0f}')} ms; anchor {f_(r, 'anchor_origin')} (own row "
                     f"{f_(r, 'anchor_own_row')}); " + '; '.join(f"{k} {f_(r, k + '_verdict')} {f_(r, k + '_cause')}" for k in KSC if k + '_verdict' in r))
    L += ['', 'READINGS (declared before the data was read)'] + [f'  {k}: {v}' for k, v in read.items()] + [
          '', 'LIMITS: one consumed partial day; filled orders only (an outcome-selected sample); the premises of K5 (activation before the Accepted answer; the cancel '
          'effect before its confirmation) and of K7 (notices after their status; floor-second stamps) are stated, not proven; book-to-status clock alignment not '
          'established; UUID creation after matching and a shared UUID/status clock are premises; K6\'s declared gate is at K7\'s offsets, its K8 shared-clock and settlement-offset readings hold only under those premises; upper-only ST cannot certify match-vs-cancel ordering; development evidence - Nov 8 is not untouched validation.']
    return L


def known_write(out, K, KE, R, read, meta):
    """The case ledger: known_cases (per leg), known_case_rows (per candidate row), known_orders (per unique order), every column documented
    (known_schema.json; an undocumented column refuses the output), known.txt / known.json."""
    PO = R.pop('_orders'); cols = R.pop('_legs'); KE = R.pop('_KE', KE); extra = {}
    if KE is not None:
        g = KE.groupby('i'); one = lambda x, f: f(x.iloc[0]) if x.nunique() == 1 else 'ambiguous'
        extra = dict(n_candidates=np.bincount(KE.i.values, minlength=len(K)), anchor_origin=g.origin.agg(lambda x: one(x, str)).reindex(range(len(K))).fillna('unlinked').values,
                     anchor_own_row=g.anchor_own.agg(lambda x: one(x, lambda y: 'yes' if y else 'no')).reindex(range(len(K))).fillna('unlinked').values)
    KC = K.reset_index(drop=True).assign(**cols, **extra)
    TB = dict(known_cases=KC, known_orders=PO, **({'known_case_rows': KE} if KE is not None else {})); gaps = case_gaps({t: list(v.columns) for t, v in TB.items()})
    if gaps:
        sys.exit(f'KNOWN-LIFECYCLE CHECK FAILED -> no output: undocumented case-ledger columns {gaps}')
    L = known_report(R, read, KC); out.mkdir(parents=True); files = {}
    for t, v in TB.items():
        v.to_parquet(out / f'{t}.parquet', index=False); files[f'{t}.parquet'] = C.sha(out / f'{t}.parquet')
    C.dump(out / 'known_schema.json', dict(tables={t: {c: CASE_SCHEMA.get(t, {}).get(c) or next((d for x, d in CASE_PREFIX.get(t, {}).items() if c.startswith(x)), 'UNDOCUMENTED')
                                                       for c in v.columns}
                                                   for t, v in TB.items()}, verdicts=list(VERD), causes=list(CAUSES), sources=list(SRC4), s3_codes=list(SEM_CODES),
                                            timestamp_semantics=TS_SEM)); files['known_schema.json'] = C.sha(out / 'known_schema.json')
    (out / 'known.txt').write_text('\n'.join(L) + '\n')
    C.dump(out / 'known.json', dict(asof=C.stamp(), readings=read, R=R, outputs=files, **meta))
    return L


def book_bounds(K, path=None):
    """v6: the book-clock lower bound per known leg from the census receipt (pf_known_clock_census.py; outputs hash-checked, its N0 reproduction
    YES, exactly K's legs) -> (array aligned with K: B where IDENTIFIED, NaN otherwise; the receipt's hash)."""
    d = Path(path or OUT_CENSUS); j = json.loads((d / 'census.json').read_text())
    C.require(C.sha(d / 'legs.parquet') == j['outputs']['legs.parquet'], 'census legs differ from their receipt')
    C.require(j['readings'].get('N0 REPRODUCES-THE-CASE-AUDIT') == 'YES', 'the census does not reproduce the timing case audit: no book-clock bound')
    L = pd.read_parquet(d / 'legs.parquet'); key = ['run', 'cid', 'trade_id']
    C.require(not L.duplicated(key).any() and len(L) == len(K), 'census legs are not one per known leg')
    m = K[key].reset_index(drop=True).merge(L[key + ['B', 'bound_status']], on=key, how='left', validate='one_to_one')
    C.require(bool(m.bound_status.notna().all()), 'a known leg without its census row')
    return np.where(m.bound_status.values == 'IDENTIFIED', m.B.values.astype(float), np.nan), {str(d / 'census.json'): C.sha(d / 'census.json')}


def known(out=OUT_K):
    start = time.time(); out = Path(out); C.require(not out.exists(), 'known-lifecycle output exists'); selftest()
    K, KE, ref, meta = known_frame(); book, hb = book_bounds(K); meta['inputs'].update(hb); R, read = known_checks(K, ref, KE, book=book)
    if read['K0 REPRODUCES 11-09'] != 'YES':
        sys.exit(f"KNOWN-LIFECYCLE CHECK FAILED -> no output: K0 {R['K0']} != the 11-09 counts {ref}")
    L = known_write(out, K, KE, R, read, dict(meta, provenance=KP.provenance(), source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)


def _brute(ev, wlo, whi, d, F, CW, lat):
    """Dense-grid min / max of each event's summed contribution (integer breakpoints in the selftest: a 0.5 ms grid hits every piece)."""
    nE = len(wlo); mn = np.full(nE, np.inf); mx = np.full(nE, -np.inf)
    for e in range(nE):
        f = np.flatnonzero(ev == e)
        if not len(f):
            mn[e] = mx[e] = 0.; continue
        a = wlo[e] if np.isfinite(wlo[e]) else float(d[f].min()) - 10.; b = whi[e] if np.isfinite(whi[e]) else float(d[f].max()) + KP.H + 10.
        g = np.r_[np.arange(np.ceil(a * 2) / 2, b + 1e-9, .5), a, b]; g = g[(g >= a) & (g <= b)]
        tot = sum(contrib(g - d[i], {k: np.full(len(g), v[i]) for k, v in F.items()}, CW, lat) for i in f)
        mn[e], mx[e] = float(tot.min()), float(tot.max())
    return mn, mx


def _synthetic(rng, day='20990101'):
    """A synthetic keep / pull day: orders (incl. two refusals), M50 / M250 fills, and the legs the loader would return (unique, multi and
    singleton events; a leg of a candidate event at a price that cannot fill)."""
    n = 40; side = rng.integers(0, 2, n); p = rng.integers(30, 70, n).astype(float); t = 1.0e6 + 1000. * np.arange(n) + rng.integers(0, 50, n)
    f = p[:, None] + rng.normal(0., 2., (n, KP.NSTEP + 1)); st = np.where(np.arange(n) < 38, 'eligible', 'refused_post_only')
    keep = np.arange(n) % 3 == 0; f[keep] = p[keep, None] + np.where(side[keep] == 0, 1., -1.)[:, None] * (1. + np.abs(rng.normal(0., 2., (int(keep.sum()), KP.NSTEP + 1))))
    gst = np.where(rng.random(n) < .1, 3000., np.nan); rng.random(n)                                 # (the second draw keeps the stream as before)
    O = pd.DataFrame(dict(day=day, oid=np.arange(n), m=np.where(np.arange(n) % 2, 11, 12), side=side, t=t, ws=0., phase=0, p=p, status=st, mark=p + rng.normal(0., 3., n),
                          g_struct=gst, land_struct=np.where(np.isnan(gst), np.nan, gst + KP.CANCEL),                 # the landing of its own decision
                          t_full=np.where(rng.random(n) < .2, 2500., np.nan), **{f'f{j:02d}': f[:, j] for j in range(KP.NSTEP + 1)}))
    fl, legs, g = [], [], 100
    for o in range(n):
        for _ in range(rng.integers(0, 4)):
            ls_ = O.land_struct[o]; tt = float(rng.integers(60, 4990 if np.isnan(ls_) else int(ls_)))   # the replay never fills after the structural landing
            fl.append((day, o, 50., tt, float(rng.integers(1, 30)), int(rng.integers(0, 4)))); fl.append((day, o, 250., tt + 7., 5., 0))
            a = tt + SHIFT + t[o]; S = np.floor(a / 1000.) + rng.integers(0, 2); kind = rng.integers(0, 3); ngr = 2 if rng.random() < .2 else 1
            for _k in range(ngr):
                g += 1; tk = g if rng.random() < .8 else -1
                for leg in range(rng.integers(1, 3)):
                    ws = kind == 0; tms = a + (0. if leg == 0 else float(rng.integers(0, 300))) if ws else np.nan
                    legs.append(dict(mid=int(O.m[o]), side=int(side[o]), anc=a, ptick=int(p[o]) if leg == 0 else int(p[o]) + (1 if side[o] == 0 else -1),
                                     shares=5., how=1 if ws else 0, t_ms=tms, t_e2=a if kind == 1 else np.nan, exec_s=float(S), tk=tk, tk_status=0 if tk >= 0 else 1,
                                     first_recv_ns=0, price_executed=p[o] / 100., signer='x', taker_index=0, taker_quote=0))
    Fl = pd.DataFrame(fl, columns=['day', 'oid', 'clock', 't', 'q', 'src']); L = pd.DataFrame(legs); L['row_id'] = np.arange(len(L))
    L['group'] = np.where(L.tk.values >= 0, L.tk.values, -1 - L.row_id.values); w = L[L.how == 1]                 # pending increments: each WS leg's own, a second for every third
    v = w.shares.values * w.ptick.values / 100.; x = w.row_id.values % 3 == 0
    P = pd.DataFrame(dict(mid=np.r_[w.mid.values, w.mid.values[x]], t_ms=np.r_[w.t_ms.values, w.t_ms.values[x] - 400.], ptick=np.r_[w.ptick.values, w.ptick.values[x]], inc=np.r_[v, v[x]]))
    return O, Fl, L, P


def _v1_expect(O, Fl, L, h=600., q01=-900., q99=1050.):
    """v1's own functions on a synthetic day (the rehearsal's reference): S1 / S2 bounds and ordering counts, the collapsed E and score."""
    La = KP.landing(np.arange(1, KP.NSTEP + 1)); E = O[O.status == 'eligible'].reset_index(drop=True); cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
    endj = A.endj_of(E.t_full.values, E.g_struct.values); act, CW = V1.null_weights(T, cell, endj); ri, t, val, src = A.fills_of(E, Fl, SHIFT)
    pf = A.per_fill(T, cell, endj, (ri, t, val, src)); lend = np.where(np.isnan(E.land_struct.values[ri]), np.inf, E.land_struct.values[ri])
    Fd = pd.DataFrame(dict(fid=np.arange(len(ri)), m=E.m.values[ri], side=E.side.values[ri], a=t + SHIFT + E.t.values[ri], p=E.p.values[ri]))
    H = V1.link(Fd, L[V1_COLS], h, q01, q99).reindex(np.arange(len(ri))); r = La[np.clip(T[ri], 1, KP.NSTEP) - 1]
    F1 = dict(val=val, pulled=act[ri], r=r, s=cell[ri], endj=endj[ri], lend=lend); v1 = {}
    for k, (a_, b_) in (('S1', ('lo1', 'hi1')), ('S2', ('lo2', 'hi2'))):
        lo, hi = H[a_].values - E.t.values[ri], H[b_].values - E.t.values[ri]; mn, mx = V1.bounds(lo, hi, F1, CW); o = np.where(act[ri], V1.order_of(lo, hi, r), -1)
        v1[k] = dict(E_min=float(mn.sum()), E_max=float(mx.sum()), order={nm: int(((o == c) & act[ri]).sum()) for c, nm in enumerate(('AFTER', 'BEFORE', 'AMBIGUOUS'))})
    return v1, float(pf['score'].sum() - pf['null'].sum()), float(pf['score'].sum()), dict(ri=ri, t=t, val=val, act=act, r=r)


class _Src:
    def __init__(self, L):
        self.L = L

    def offset(self, day):
        return 0.

    def load(self, day, batch, offset):
        return dict(LT=self.L[self.L.mid.isin(batch)].copy())


def _refused_v(fn):
    """True if fn refuses (ValueError from C.require)."""
    try:
        fn()
    except ValueError:
        return True
    return False


def _s3_selftest():
    """sem_windows on planted increments (two candidates and a near miss in value, one late but within + 250 ms, a late link, a cell value, out
    of range, E2 / REST-only legs, no increments); the contradiction accounting by cause and its falsifier; the leg disagreements' binding pairs."""
    bad = []
    R = pd.DataFrame(dict(mid=[7, 7, 7, 7, 7, 7, 7, 9], exec_s=100., side=0, ptick=[50, 60, 70, 40, 40, 30, 30, 50], shares=[10., 10., 10., 3., 2., 1., 1., 10.],
                          how=[1, 1, 1, 2, 2, 0, 0, 1], t_ms=[100500., 101100., 101400., 100200., 100200., np.nan, np.nan, 100000.], t_e2=[np.nan] * 5 + [100300., np.nan, np.nan]))
    P = pd.DataFrame(dict(mid=7, t_ms=[100500., 99000., 95000., 89000., 101300., 99500., 101100., 101400., 100200., 100300.], ptick=[50, 50, 50, 50, 50, 50, 60, 70, 40, 40],
                          inc=[5., 5.03, 6., 5., 5., 5.06, 6., 7., 2., 1.2]))
    want = [(98750., 100500., 1, 2, False), (100850., 101000., 0, 1, False), (90000., 101000., 2, 0, True), (99950., 100200., 0, 1, False), (99950., 100200., 0, 1, False),
            (90000., 101000., 3, 0, False), (90000., 101000., 4, 0, False), (90000., 101000., 2, 0, False)]
    got = [(float(a), float(b), int(c), int(d), bool(e)) for a, b, c, d, e in sem_windows(R, P)[['S3_llo', 'S3_lhi', 'S3_code', 'S3_ncand', 'link_late']].itertuples(index=False, name=None)]
    if got != want:
        bad.append(f'S3 windows: {got} != {want}')
    if not sem_windows(R, P, chunk=1).equals(sem_windows(R, P)):
        bad.append('S3 windows differ when the candidate pairs are chunked')
    EVc = pd.DataFrame(dict(S1_st=[2, 3, 0], S2_st=0, S3_st=0, ST_st=[0, 0, 2], cls=0)); gr = np.array([1., 2., 4., 8., 16.])
    V3c, _ = contradiction_accounting(np.array([0, 1, 2, 2, 3, 3]), np.array([0, 1, 0, 1, 0, 2]), EVc, 5, gr, np.ones(5, bool), np.array([1, 1, 2, 2, 0]), np.full(5, 'd'), ['d'],
                                      np.zeros(6, bool))
    if not ((V3c['S1']['fills']['all'], V3c['S1']['fills']['some'], V3c['S1']['fills']['unlinked'], V3c['S1_cap']['fills']['all'], V3c['S1_cap']['fills']['some'],
             V3c['S1_legs']['fills']['all'], V3c['S1_legs']['fills']['some'], V3c['S1']['mixed_all']['fills'], V3c['S1']['mixed_all']['gross'], V3c['S1']['gross']['all'])
            == (3, 1, 1, 1, 2, 1, 1, 1, 4., 7.) and causes_reconcile(V3c) and accounted(V3c, 5, float(gr.sum()))):
        bad.append(f"contradiction accounting by cause: {V3c['S1']} {V3c['S1_cap']['fills']} {V3c['S1_legs']['fills']}")
    V3x = json.loads(json.dumps(V3c)); V3x['S1_cap']['fills']['all'] += 1
    if causes_reconcile(V3x):
        bad.append('the cause reconciliation does not fire on a fill counted under two causes')
    Ld = pd.DataFrame(dict(day='d', group=[1, 1, 2, 2, 3], mid=7, side=0, anc=0., how=[1, 0, 1, 2, 1], t_ms=[1000., np.nan, 2000., 2600., 5000.], t_e2=np.nan,
                           exec_s=[1., 3., 2., 2., 5.], h=600., q01=-900., q99=1050., S3_llo=[750., -7000., 1750., 2350., 4750.], S3_lhi=[1000., 4000., 2000., 2600., 5000.],
                           ST_llo=[700., 700., 1800., 1800., 4800.], ST_hi=0.))
    Ld['ST_lhi'] = [950., 950., 2000., 2000., 5000.]; Ld = Ld.drop(columns='ST_hi')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:                                   # the settlement file and the lag bound: receipts and refusals
        old = SETT.OUT; SETT.OUT = Path(tmp); d0 = Path(tmp) / '20990101'; d0.mkdir()
        try:
            pd.DataFrame(dict(t_set_ms=[1., np.nan], set_status=[0, 1])).to_parquet(d0 / 'legs_set.parquet', index=False)
            rec = dict(outputs={'legs_set.parquet': C.sha(d0 / 'legs_set.parquet')}, R=dict(controls=dict(C1=True, C2=True, C3=True, C4=True)))
            (d0 / 'settlement.json').write_text(json.dumps(rec)); ok1 = len(set_of('20990101')[0]) == 2
            (d0 / 'settlement.json').write_text(json.dumps(dict(rec, R=dict(controls=dict(C1=False))))); r1 = _refused_v(lambda: set_of('20990101'))
            (d0 / 'settlement.json').write_text(json.dumps(rec)); pd.DataFrame(dict(t_set_ms=[2., np.nan], set_status=[0, 1])).to_parquet(d0 / 'legs_set.parquet', index=False)
            r2 = _refused_v(lambda: set_of('20990101'))
            (Path(tmp) / 'known.json').write_text(json.dumps(dict(R=dict(K8=dict(model=SETTLEMENT_MODEL, consistent=True, lag_bound_ms=None))))); ok2 = known_lag(tmp)[0] is None
            (Path(tmp) / 'known.json').write_text(json.dumps(dict(R=dict(K8=dict(consistent=True, lag_bound_ms=100.))))); r3 = _refused_v(lambda: known_lag(tmp))
            (Path(tmp) / 'known.json').write_text(json.dumps(dict(R=dict(K8=dict(model=SETTLEMENT_MODEL, consistent=False, lag_bound_ms=None))))); r4 = _refused_v(lambda: known_lag(tmp))
        finally:
            SETT.OUT = old
        if not (ok1 and r1 and r2 and ok2 and r3 and r4):
            bad.append(f'settlement file / lag bound readers: {ok1} {r1} {r2} {ok2} {r3}')
    Dd = disagreement(Ld, events_of(Ld), 'S1')
    if Dd != dict(events=2, binding_pair={'rest_second|ws_leg_link': 1, 'ws_cell_link|ws_leg_link': 1}, different_rest_seconds=1,
                  ws_spread_ms={'events': 1, 'p50': 600., 'p90': 600., 'max': 600.}, admissible_under_S3=1):
        bad.append(f'leg disagreements: {Dd}')
    return bad


def _k6_selftest():
    """K6 / K7 on planted lifecycles (a false CANCEL_FIRST at offset 0, one only at an admissible offset, an event proxy, an own E2 miss, a
    contradictory event, two disagreeing candidates); the case linkage on a planted day (candidate rows, events, the anchor rule and its
    origin, a wrong pinned anchor, a wrong candidate count); the case ledger written, an undocumented column refused (synthetic only)."""
    bad = []; B_ = 1.79e12 + 7.2e6; pl = B_ + np.array([0., 10000., 10000., 20000., 30940., 40000.])
    Kw = pd.DataFrame(dict(run='w', cid=[0, 1, 1, 2, 3, 4], place=pl, accepted=pl + 30., upper=B_ + np.array([300., 10260., 10260., 20300., 31240., 40300.]),
                           notice=B_ + np.array([300., 10400., 10400., 20500., 31240., 40300.]), ack=B_ + np.array([np.nan, 10260., 10260., 20260., 31240., np.nan]),
                           req=B_ + np.array([np.nan, 10200., 10200., 20200., 31100., np.nan]), M=np.floor(pl / 1000.), Z=pl + 100., signed=pl + 5., ucancel=np.nan, cdone=np.nan,
                           thr=np.nan, rM=pl + 150., rC=pl + 1100.))
    Kw['S'] = Kw.M + 1.; Kw['Cs'] = Kw.M + 1.; Kw.loc[5, 'Z'] = pl[5] + 5000.                    # the one K4 failure, to be traced
    rows = np.array([(0, 0, 100., 200., 0, 100., 200.), (1, 0, 10400., 10450., 0, 10150., 10250.), (2, 0, 10100., 10150., 0, 10100., 10150.), (2, 1, 10500., 10600., 0, 10500., 10600.),
                     (3, 0, 20230., 20240., 0, 20230., 20240.), (4, 2, 31500., 31400., 3, 30950., 31000.), (5, 0, 40320., 40330., 0, 40320., 40330.)])
    i_ = rows[:, 0].astype(int); up, plc = Kw.upper.values[i_], Kw.place.values[i_]
    KE = pd.DataFrame(dict(i=i_, src=rows[:, 1].astype(int), origin='pending_increment', anchor_own=True, anchor_rule_ok=True, S3_code=0, link_late=False))
    for k in ('S1', 'S2'):
        KE[f'{k}_lo'], KE[f'{k}_hi'], KE[f'{k}_st'], KE[f'own_{k}_lo'], KE[f'own_{k}_hi'] = B_ + rows[:, 2], B_ + rows[:, 3], rows[:, 4].astype(int), B_ + rows[:, 5], B_ + rows[:, 6]
    KE['S3_lo'], KE['S3_hi'], KE['S3_st'], KE['own_S3_lo'], KE['own_S3_hi'] = plc - 50., up, 0, plc - 50., up
    KE['EV_lo'], KE['EV_hi'], KE['EV_st'], KE['own_EV_lo'], KE['own_EV_hi'] = -np.inf, up + 500., 0, -np.inf, up + 500.
    KE['M50_lo'] = KE['M50_hi'] = B_ + np.array([150., 10420., 10120., 10520., 20235., 29000., 40150.]); KE['M50_st'] = 0
    KE['own_M50_lo'] = KE['own_M50_hi'] = B_ + np.array([150., 10200., 10120., 10520., 20235., 29000., 40150.])
    ref_w = dict(rest_floor_nonoverlap=int(miss(Kw.S.values * 1000., Kw.S.values * 1000. + 999.999, Kw).sum()),
                 matched_floor_nonoverlap=int(miss(*window_from('matched_status_s', Kw.M.values), Kw).sum()))
    Rw, rw = known_checks(Kw, ref_w, KE, draws=10); s1 = Rw['K6']['S1']; m5 = Rw['K6']['M50']
    if not (Rw['K7']['lo'] == -100. and Rw['K7']['hi'] == 60. and Rw['K7']['includes_zero']
            and s1['verdict_at_offset_0'] == dict(compatible=2, missed_before=0, missed_after=2, contradictory_evidence=0, contradictory_legs=1, ambiguous=1, unlinked=0)
            and s1['verdict_every_offset'] == dict(compatible=3, missed_before=0, missed_after=1, contradictory_evidence=0, contradictory_legs=1, ambiguous=1, unlinked=0)
            and s1['false_cancel_first'] == dict(at_offset_0=1, some_offset=2, every_offset=1)
            and {k: v for k, v in s1['causes'].items() if v} == dict(event_proxy=1, own_ws_leg_link=1, contradictory_leg_intersection=1, unresolved_linkage=1)
            and [rw[f'K6 GATE {k}'] for k in GATE] == ['FAIL', 'FAIL', 'PASS', 'NOT-RUN', 'PASS'] and rw['K6 REPLAY POINT M50'] == 'CONTRADICTED'
            and {k: v for k, v in m5['causes'].items() if v} == dict(event_proxy=1, own_e2_drop=1, unresolved_linkage=1) and m5['false_cancel_first']['at_offset_0'] == 1
            and 'LEDGER-WINDOW GATE S1 FAIL, S2 FAIL, S3 PASS, ST NOT-RUN, EV PASS' in rw['KNOWN VERDICT'] and list(Rw['_legs']['S1_verdict']) == ['compatible', 'missed_after', 'ambiguous',
                                                                                                                                      'compatible', 'contradictory_legs', 'missed_after']):
        bad.append(f"K6 / K7 on planted lifecycles: {Rw['K7']} {Rw['K6']} {rw}")
    Kz = Kw.copy(); Kz['rM'] = Kz.rM - 2000.                                                       # inconsistent receipts: no admissible offset -> offset 0
    if known_checks(Kz, ref_w, KE, draws=10)[0]['K7']['exists'] or known_checks(Kz, ref_w, KE, draws=10)[0]['K6']['S1']['false_cancel_first']['some_offset'] != 1:
        bad.append('K6 without an admissible offset must fall back to offset 0')
    L8 = pd.DataFrame(dict(mid=5, exec_s=[100, 100, 101, 102, 102], first_recv_ns=[1, 2, 3, 4, 4], side=[0, 0, 1, 0, 0], ptick=[50, 51, 48, 47, 47], shares=[10., 4., 2., 3., 3.],
                           price_executed=[.5, .51, .48, .47, .47], taker_index=1, taker_quote=0, how=[1, 1, 0, 0, 1], t_ms=[100200., 99100., np.nan, np.nan, 102400.],
                           t_e2=[np.nan, np.nan, 101300., np.nan, np.nan], tk=[7, 7, -1, 9, 11], Zr=[99100., 99100., 101300., 102000., 102400.]))
    P8 = pd.DataFrame(dict(mid=5, t_ms=[100200., 99100.], ptick=[50, 51], inc=[5., 2.04])); env8 = dict(h=600., q01=-900., q99=1050.)
    Kl = L8.iloc[[0, 2, 3]][CKEY].reset_index(drop=True).assign(candidate_legs=[1, 1, 2])
    try:
        E8 = case_linkage(Kl, L8, P8, env8, day='20990101')
        if not (E8[['i', 'row_id']].values.tolist() == [[0, 0], [1, 2], [2, 3], [2, 4]] and E8.origin.tolist() == ['pending_increment', 'book_drop', 'REST_second', 'pending_increment']
                and E8.anchor_own.tolist() == [False, True, True, True] and bool(E8.anchor_rule_ok.all()) and E8.src.tolist() == [0, 2, 3, 0]
                and (E8.S1_st[0], E8.own_S1_lo[0], E8.own_S1_hi[0], E8.M50_lo[0], E8.own_M50_lo[0], E8.S3_st[0]) == (3, 99950., 100200., 99050., 100150., 3)
                and (E8.S1_lo[1], E8.S1_hi[1], E8.S1_st[1]) == (101050., 101550., 0) and (E8.S1_lo[2], E8.S1_hi[2]) == (101500., 102500.)
                and (E8.S3_code[3], E8.S3_lo[3], E8.S3_hi[3]) == (2, 92000., 103000.) and E8.event_legs.tolist() == [2, 1, 1, 1] and E8.own_S2_hi[2] == 103000.):
            bad.append(f"case linkage: {E8.drop(columns=[c for c in E8.columns if c.startswith(('S2', 'EV', 'own_S2', 'own_S3', 'own_EV'))]).to_dict('records')}")
        if case_linkage(Kl, L8.assign(Zr=[99100., 99100., 101000., 102000., 102400.]), P8, env8, day='20990101').anchor_rule_ok.tolist() != [True, False, True, True]:
            bad.append('a pinned anchor the engine rule does not reproduce is not marked')
    except (SystemExit, ValueError) as e:
        bad.append(f'case linkage refused: {e}')
    try:
        case_linkage(Kl.assign(candidate_legs=[1, 1, 1]), L8, P8, env8, day='20990101'); bad.append('case linkage accepted a candidate count that differs from the audit\'s')
    except ValueError:
        pass
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        try:
            known_write(Path(tmp) / 'k', Kw, KE, Rw, rw, dict(inputs={}, env={}))
            outs = sorted(x.name for x in (Path(tmp) / 'k').iterdir()); KC = pd.read_parquet(Path(tmp) / 'k' / 'known_cases.parquet')
            sch = json.loads((Path(tmp) / 'k' / 'known_schema.json').read_text())['tables']
            if not (outs == ['known.json', 'known.txt', 'known_case_rows.parquet', 'known_cases.parquet', 'known_orders.parquet', 'known_schema.json']
                    and KC.n_candidates.tolist() == [1, 1, 2, 1, 1, 1] and set(KC.columns) == set(sch['known_cases']) and len(pd.read_parquet(Path(tmp) / 'k' / 'known_orders.parquet')) == 5
                    and KC.anchor_origin.tolist() == ['pending_increment'] * 6 and KC.K4_S1_missed.tolist() == [False] * 5 + [True]
                    and 'K4 S1 failures traced (1;' in (Path(tmp) / 'k' / 'known.txt').read_text() and 'w cid 4 trade - tx -: Z - place +5000 ms' in (Path(tmp) / 'k' / 'known.txt').read_text()):
                bad.append(f'case ledger outputs: {outs}')
        except (SystemExit, ValueError) as e:
            bad.append(f'case ledger refused: {e}')
        R2, r2 = known_checks(Kw.assign(zzz=1.), ref_w, KE, draws=10)
        try:
            known_write(Path(tmp) / 'z', Kw.assign(zzz=1.), KE, R2, r2, dict(inputs={}, env={})); bad.append('the case ledger accepted an undocumented column')
        except SystemExit:
            pass
        if (Path(tmp) / 'z').exists():
            bad.append('a refused case ledger wrote output')
    ts_ = B_ + np.array([100., 10150., 10250., 20150., 31000., 40000.])                       # K8: settlement times inside their lifecycles
    Kt = Kw.assign(t_set=ts_, M=np.floor(ts_ / 1000.))
    Kt['S'] = Kt.M + 1.; Kt['Cs'] = Kt.M + 1.; Kt['rM'] = Kt.M * 1000. + 150.; Kt['rC'] = Kt.Cs * 1000. + 100.
    ref_t = dict(rest_floor_nonoverlap=int(miss(Kt.S.values * 1000., Kt.S.values * 1000. + 999.999, Kt).sum()),
                 matched_floor_nonoverlap=int(miss(*window_from('matched_status_s', Kt.M.values), Kt).sum()))
    Rt, rt = known_checks(Kt, ref_t, KE, draws=10); k8 = Rt['K8']; k5 = Rt['K5']
    if not (k8['consistent'] and k8['d'] == dict(lo=-10., hi=0., exists=True) and rt['K8 SETTLEMENT TIME'] == 'COMPATIBLE'
            and k8['lag_bound_ms'] is None and k8['universal_lag_status'] == 'UNIDENTIFIED' and k8['notice_delay_ms']['lo_min'] == 100.
            and k5['orders_with_a_match_after_the_request'] == 0 and k5['largest_cancel_lower_bound_ms'] == 0.
            and k5['smallest_activation_upper_with_settlement_ms'] == 30. and rt['K6 GATE ST'] == 'UNIDENTIFIED-UPPER-BOUND-ONLY'
            and Rt['K6']['ST']['false_cancel_first']['at_offset_0'] == 0 and 'K6 GATE S1 (settlement offsets)' in rt
            and np.allclose(Rt['_legs']['K8_match_hi_bot'] - B_, [200., 10250., 10260., 20250., 31100., 40100.])
            and np.array_equal(Rt['_legs']['K8_match_lo_bot'], Kt.place.values)):
        bad.append(f"K8 on planted settlement times: {k8} {k5.get('cancel_with_settlement')} {rt}")
    # K6 basis (check of 9a114cc1): leg 0's first notice 20 ms after its UUID lifts the K8 shared clock's lower end above K7's (-20 vs -100 ms);
    # leg 3's S3 window starts 30 ms before its cancel confirmation - called avoided only below -30 ms. The declared gate (K7) must FAIL on it,
    # the K8 shared-clock and settlement-offset readings (conditional) PASS, each under its own key.
    Kc = Kt.assign(notice=Kt.notice.values - np.r_[180., np.zeros(5)], upper=Kt.upper.values - np.r_[180., np.zeros(5)])
    KEc = KE.copy(); KEc.loc[4, ['S3_lo', 'own_S3_lo']] = B_ + 20230.
    Rc, rc = known_checks(Kc, ref_t, KEc, draws=10)
    if not (rc.get('K6 GATE S3') == 'FAIL' and rc.get('K6 GATE S3 (K8 shared clock)') == 'PASS' and rc.get('K6 GATE S3 (settlement offsets)') == 'PASS'
            and Rc['K7']['lo'] == -100. and Rc['K8']['clock_offset']['lo'] == -20. and Rc['K6']['S3']['false_cancel_first']['some_offset'] == 1
            and Rc['K6_shared_clock']['S3']['false_cancel_first']['some_offset'] == 0 and rc.get('K6 GATE ST (K8 shared clock)') == 'UNIDENTIFIED-UPPER-BOUND-ONLY'
            and 'LEDGER-WINDOW GATE S1 FAIL, S2 FAIL, S3 FAIL, ST UNIDENTIFIED-UPPER-BOUND-ONLY, EV PASS (K7 offsets; conditional - K8 shared clock: S1 FAIL, S2 FAIL, S3 PASS'
            in rc['KNOWN VERDICT'] and list(Rc['_legs']['S3_false_cf_some_offset']) == [False, False, False, True, False, False]
            and not any(Rc['_legs']['S3_false_cf_some_offset_clk'])):
        bad.append(f"K6 basis (the declared K7 gate vs the conditional K8 shared clock): {rc}")
    for nm_, kt in (('creation after its containing notice at every admissible clock', Kt.assign(t_set=ts_ + np.r_[5000., np.zeros(5)])),
                    ('creation before placement at every admissible clock', Kt.assign(t_set=ts_ - np.r_[5000., np.zeros(5)]))):
        if known_checks(kt, ref_t, KE, draws=10)[1].get('K8 SETTLEMENT TIME') != 'CONTRADICTED':
            bad.append(f'K8 accepted {nm_}')
    return bad


def _k6_domain_selftest():
    """v6, the gate of record per clock domain (synthetic): leg 3's S3 window starts 30 ms before its cancel confirmation. K7 (-100 ms) calls it
    avoided (the v5 basis FAILs); its book end at the leg's own book bound (-20 ms) does not (PASS); an unidentified book bound fails closed; the
    same end on the status clock is read at K7 (FAIL); M50 by its anchor's origin; the case linkage's split reproduces the event start; the
    census reader refuses a changed table, an unreproduced audit and a missing leg."""
    bad = []; B_ = 1.79e12 + 7.2e6; pl = B_ + np.array([0., 10000., 10000., 20000., 30940., 40000.])
    K = pd.DataFrame(dict(run='w', cid=[0, 1, 1, 2, 3, 4], trade_id=[f't{x}' for x in range(6)], place=pl, accepted=pl + 30.,
                          upper=B_ + np.array([300., 10260., 10260., 20300., 31240., 40300.]), notice=B_ + np.array([300., 10400., 10400., 20500., 31240., 40300.]),
                          ack=B_ + np.array([np.nan, 10260., 10260., 20260., 31240., np.nan]), req=B_ + np.array([np.nan, 10200., 10200., 20200., 31100., np.nan]),
                          M=np.floor(pl / 1000.), Z=pl + 100., signed=pl + 5., ucancel=np.nan, cdone=np.nan, thr=np.nan, rM=pl + 150., rC=pl + 1100.))
    K['S'] = K.M + 1.; K['Cs'] = K.M + 1.
    i_ = np.array([0, 1, 2, 3, 4, 5]); up = K.upper.values
    KE = pd.DataFrame(dict(i=i_, src=0, origin=['pending_increment'] * 5 + ['REST_second'], anchor_own=True, anchor_rule_ok=True, S3_code=0, link_late=False))
    for k in ('S1', 'S2', 'S3'):
        KE[f'{k}_lo'], KE[f'{k}_hi'], KE[f'{k}_st'] = pl - 50., up, 0
        KE[f'own_{k}_lo'], KE[f'own_{k}_hi'] = pl - 50., up
        KE[f'{k}_lo_book'], KE[f'{k}_lo_status'] = pl - 50., -np.inf
    KE.loc[3, ['S3_lo', 'own_S3_lo', 'S3_lo_book']] = B_ + 20230.                               # c_hi 20260: avoided only below -30 ms
    KE['EV_lo'], KE['EV_hi'], KE['EV_st'], KE['own_EV_lo'], KE['own_EV_hi'] = -np.inf, up + 500., 0, -np.inf, up + 500.
    KE['M50_lo'] = KE['M50_hi'] = K.place.values + 60.; KE['M50_st'] = 0; KE['own_M50_lo'] = KE['own_M50_hi'] = KE.M50_lo
    KE.loc[3, ['M50_lo', 'M50_hi', 'own_M50_lo', 'own_M50_hi']] = B_ + 20280.                  # the replay point after c_hi: avoided at offset 0 already
    ref = dict(rest_floor_nonoverlap=int(miss(K.S.values * 1000., K.S.values * 1000. + 999.999, K).sum()),
               matched_floor_nonoverlap=int(miss(*window_from('matched_status_s', K.M.values), K).sum()))
    book = np.full(6, -20.)
    try:
        R, r = known_checks(K, ref, KE, draws=10, book=book); d = R['K6_per_domain']
        if not (r.get('K6 GATE S3') == 'FAIL' and r.get('K6 GATE S3 (per clock domain)') == 'PASS' and R['K7']['lo'] == -100.
                and d['S3']['false_cancel_first'] == dict(at_offset_0=0, some_offset_per_domain=0) and d['S3']['legs_with_a_book_lower_end'] == 4
                and d['M50']['false_cancel_first']['at_offset_0'] == 1 and r.get('K6 GATE EV (per clock domain)') == 'PASS'
                and 'LEDGER-WINDOW GATE OF RECORD (per clock domain' in r['KNOWN VERDICT'] and 'S3 PASS' in r['KNOWN VERDICT'].split('; v5 basis')[0]
                and 'v5 basis - K7 offsets on every end: S1 PASS, S2 PASS, S3 FAIL' in r['KNOWN VERDICT'] and list(R['_legs']['K6D_book_bound_ms']) == [-20.] * 6):
            bad.append(f"K6 per clock domain on planted legs: {d} {r}")
        R2, r2 = known_checks(K, ref, KE, draws=10, book=np.where(np.arange(6) == 3, np.nan, -20.))
        if not (r2.get('K6 GATE S3 (per clock domain)') == 'FAIL' and R2['K6_per_domain']['S3']['of_them_book_bound_unidentified'] == 1):
            bad.append(f"K6 per clock domain: an unidentified book bound must fail closed: {R2['K6_per_domain']['S3']}")
        KEs = KE.copy(); KEs.loc[3, 'S3_lo_book'], KEs.loc[3, 'S3_lo_status'] = -np.inf, B_ + 20230.
        R3, r3 = known_checks(K, ref, KEs, draws=10, book=book)
        if r3.get('K6 GATE S3 (per clock domain)') != 'FAIL':
            bad.append('K6 per clock domain: a status-clock end must be read at K7')
        KEm = KE.copy(); KEm.loc[3, ['M50_lo', 'M50_hi', 'own_M50_lo', 'own_M50_hi']] = B_ + 20240.; KEm.loc[3, 'origin'] = 'REST_second'
        KEb = KEm.copy(); KEb.loc[3, 'origin'] = 'book_drop'
        m_s = known_checks(K, ref, KEm, draws=10, book=book)[0]['K6_per_domain']['M50']['false_cancel_first']['some_offset_per_domain']
        m_b = known_checks(K, ref, KEb, draws=10, book=book)[0]['K6_per_domain']['M50']['false_cancel_first']['some_offset_per_domain']
        if (m_s, m_b) != (1, 0):
            bad.append(f'K6 per clock domain: M50 by its origin (status {m_s}, book {m_b})')
        KEx = KE.copy(); KEx.loc[3, 'S3_lo_book'] = B_ + 20200.
        try:
            known_checks(K, ref, KEx, draws=10, book=book); bad.append('K6 per clock domain accepted a split that does not reproduce the window start')
        except ValueError:
            pass
    except (SystemExit, ValueError, KeyError) as e:
        bad.append(f'K6 per clock domain raised {type(e).__name__}: {e}')
    # the case linkage's split on a planted day (WS, E2 and REST-second legs)
    L8 = pd.DataFrame(dict(mid=5, exec_s=[100, 100, 101, 102, 102], first_recv_ns=[1, 2, 3, 4, 4], side=[0, 0, 1, 0, 0], ptick=[50, 51, 48, 47, 47], shares=[10., 4., 2., 3., 3.],
                           price_executed=[.5, .51, .48, .47, .47], taker_index=1, taker_quote=0, how=[1, 1, 0, 0, 1], t_ms=[100200., 99100., np.nan, np.nan, 102400.],
                           t_e2=[np.nan, np.nan, 101300., np.nan, np.nan], tk=[7, 7, -1, 9, 11], Zr=[99100., 99100., 101300., 102000., 102400.]))
    P8 = pd.DataFrame(dict(mid=5, t_ms=[100200., 99100.], ptick=[50, 51], inc=[5., 2.04]))
    Kl = L8.iloc[[0, 2, 3]][CKEY].reset_index(drop=True).assign(candidate_legs=[1, 1, 2])
    try:
        E8 = case_linkage(Kl, L8, P8, dict(h=600., q01=-900., q99=1050.), day='20990101')
        if not (list(np.isneginf(E8.S1_lo_status.values)) == [True, True, False, True] and list(np.isneginf(E8.S1_lo_book.values)) == [False, False, True, False]
                and np.array_equal(np.maximum(E8.S3_lo_book.values, E8.S3_lo_status.values), E8.S3_lo.values.astype(float))
                and np.isneginf(E8.S3_lo_book.values[2]) and E8.S3_lo_status.values[2] == E8.S3_lo.values[2]):
            bad.append(f"case linkage split: {E8[['S1_lo', 'S1_lo_book', 'S1_lo_status', 'S3_lo', 'S3_lo_book', 'S3_lo_status']].to_dict('records')}")
    except (SystemExit, ValueError) as e:
        bad.append(f'case linkage split refused: {e}')
    # the census reader
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        d = Path(tmp); Lc = pd.DataFrame(dict(run='w', cid=K.cid.values, trade_id=K.trade_id.values, B=-20., bound_status=['IDENTIFIED'] * 5 + ['UNIDENTIFIED']))
        Lc.to_parquet(d / 'legs.parquet', index=False)
        put = lambda n0='YES', h=None: (d / 'census.json').write_text(json.dumps(dict(readings={'N0 REPRODUCES-THE-CASE-AUDIT': n0}, outputs={'legs.parquet': h or C.sha(d / 'legs.parquet')})))
        put()
        try:
            b_, hb = book_bounds(K, d)
            if not (np.array_equal(b_[:5], np.full(5, -20.)) and np.isnan(b_[5]) and list(hb) == [str(d / 'census.json')]):
                bad.append(f'census reader: {b_}')
        except ValueError as e:
            bad.append(f'census reader refused a valid receipt: {e}')
        for nm_, setup in (('a changed table', lambda: put(h='0' * 64)), ('an unreproduced audit', lambda: put(n0='NO')),
                           ('a missing leg', lambda: (Lc.iloc[:5].to_parquet(d / 'legs.parquet', index=False), put()))):
            setup()
            try:
                book_bounds(K, d); bad.append(f'census reader accepted {nm_}')
            except ValueError:
                pass
            Lc.to_parquet(d / 'legs.parquet', index=False); put()
    return bad


def _settlement_selftest():
    """Independent variable-lag worlds, invalid clocks/receipts, non-transferability, and order-path propagation."""
    bad = []; b = 1_800_000_000_000.
    K = pd.DataFrame(dict(run='r', cid=['a', 'b'], place=b, upper=b + np.array([40., 900.]),
                          t_set=b + np.array([30., 800.]), M=b / 1000., notice=b + 1000.,
                          signed=b + 1., accepted=b + 30., req=[np.nan, b + 200.], ack=[np.nan, b + 900.],
                          ucancel=np.nan, cdone=np.nan, thr=np.nan))
    off = dict(lo=0., hi=0., exists=True)
    R, lo, hi, z = settlement_checks(K, off)
    if not (R['consistent'] and R['lag_bound_ms'] is None and R['known_leg_lag_ms']['hi'] == [30., 800.]
            and np.array_equal(lo - b, [0., 0.]) and np.array_equal(hi - b, [30., 800.])):
        bad.append(f'variable-lag bounds: {R}')
    else:
        for matches in ([10., 780.], [20., 100.]):  # constant 20 ms lag, or independent 10/700 ms lags
            m = b + np.array(matches)
            if not np.all((lo <= m) & (m <= hi)):
                bad.append(f'compatible true matches excluded: {matches}')
        P, _ = order_path(K, lo, hi)
        if not (P.cancel_lo_set.iloc[1] == 0. and P.cancel_hi.iloc[1] == 700.
                and P.cancel_status_set.iloc[1] == 'COMPATIBLE-UNIDENTIFIED'
                and R['notice_delay_ms']['hi_max'] == 1000.):
            bad.append('variable creation lag falsely certifies slow cancellation or bounds notice to 230 ms')
    # No common total-delay fit, yet both observations allow independent nonnegative delays.
    delayed = K.copy(); delayed.loc[1, 'upper'] = b + 110.
    rr, ll, hh, zz = settlement_checks(delayed, off)
    if not (rr['consistent'] and not rr['d']['exists'] and not zz['exists'] and ll[1] <= b + 100. <= hh[1]):
        bad.append('common-d nonexistence rejected a valid variable-lag world')
    # Matched-second equality is descriptive, not a premise of nonnegative variable lag.
    later = K.copy(); later.loc[1, 't_set'] = b + 1800.; later.loc[1, 'notice'] = b + 2000.
    rr, _, _, _ = settlement_checks(later, off)
    if not (rr['consistent'] and rr['floor_is_matched_second'] == 1):
        bad.append('later-second creation discarded or floor disagreement hidden')
    for case in (K.assign(t_set=[b - 1., b + 800.]), K.assign(notice=[b + 10., b + 1000.]), K.assign(upper=[b - 1., b + 900.])):
        if settlement_checks(case, off)[0]['consistent']:
            bad.append('impossible creation/lifecycle/receipt accepted')
    for clock in (dict(lo=1., hi=0., exists=True), dict(lo=0., hi=0., exists=False)):
        if settlement_checks(K, clock)[0]['consistent']:
            bad.append('inconsistent or unidentified clock accepted')
    # Distinguish the order of two tranches when provided independently justified match intervals.
    tranches = pd.concat([K.iloc[[1]], K.iloc[[1]]], ignore_index=True)
    P, _ = order_path(tranches, b + np.array([230., 350.]), b + np.array([240., 360.]))
    if P.cancel_lo_set.iloc[0] != 150.:
        bad.append('per-order cancellation must follow its latest tranche')
    P, _ = order_path(K, b + np.array([10., 100.]), b + np.array([15., 110.]))
    if P.act_hi_set.iloc[0] != 15.:
        bad.append('independently justified first match must bound activation')
    low, high, _ = st_windows([800., np.nan], [50., 60.], [900., 1000.], None)
    if not (np.isneginf(low[0]) and high[0] == 800. and low[1] == 60. and high[1] == 1000.):
        bad.append('public UUID window invented a lower bound or lost labelled fallback')
    for lag in (-1., np.nan, np.inf):
        if not _refused_v(lambda: st_windows([800.], [0.], [900.], lag)):
            bad.append('invalid explicit lag assumption accepted')
    return bad


def selftest(rehearse=True):
    bad = []; rng = np.random.default_rng(KP.SEED)
    I_ = np.inf; c = classify(np.array([140., 50., 100., 137., 50., -I_, -I_, 300., 80., 150., 90., 100., 100., 210., 200.]),
                              np.array([300., 130., 200., 300., 137., 130., 200., 200., 120., 190., 180., 150., 110., 250., 250.]),
                              np.r_[np.full(11, 137.), 120., 120., 120., 120.], np.r_[np.full(11, 137.), 200., 200., 200., 200.])
    if c.tolist() != [0, 1, 2, 3, 3, 1, 4, 5, 1, 0, 2, 2, 1, 0, 3]:
        bad.append(f'classify: {c.tolist()}')
    for kind, kw in (('rest_confirmed_s', dict(point=True)), ('matched_status_s', dict(point=True)), ('match_instant', {})):
        try:
            window_from(kind, np.array([5.]), **kw); bad.append(f'window_from {kind} {kw} not refused')
        except ValueError:
            pass
    lo, hi = window_from('rest_confirmed_s', np.array([5.]))
    if not (np.isneginf(lo[0]) and hi[0] == 6000.):
        bad.append('REST executedAt evidence window')
    try:
        window_from('lifecycle', (np.array([10.]), np.array([5.]))); bad.append('a reversed lifecycle accepted')
    except ValueError:
        pass
    L = pd.DataFrame(dict(day='d', group=[1, 1, 2, 3, 4, 4, 5, 6], mid=7, side=0, anc=1000., how=[1, 1, 0, 1, 1, 1, 0, 0], t_ms=[1000., 1100., np.nan, 1000., 1000., 1300., np.nan, np.nan],
                          t_e2=[np.nan, np.nan, 1010., np.nan, np.nan, np.nan, np.nan, 1200.], exec_s=[2., 2., 0., 0., 2., 2., 3., 2.], h=600., q01=-900., q99=1050.))
    L['S3_llo'], L['S3_lhi'] = rest_range(L.exec_s.values); L.loc[L.group == 1, ['S3_llo', 'S3_lhi']] = [[750., 1000.], [850., 1100.]]
    L['ST_llo'], L['ST_lhi'] = L.S3_llo - 0., L.S3_lhi - 0.; L.loc[L.group == 2, ['ST_llo', 'ST_lhi']] = [[1500., 1700.]]   # group 2: a settlement after its cap
    try:
        leg_windows(L.drop(columns=['S3_llo'])); bad.append('legs without their S3 windows accepted')
    except ValueError:
        pass
    EV = events_of(L).set_index('group')
    if not ((EV.S3_lo[1], EV.S3_hi[1], EV.S3_st[1]) == (850., 1000., 0) and (EV.S3_lo[5], EV.S3_hi[5], EV.S3_st[5]) == (-7000., 4000., 0)
            and EV.S3_lo_basis[5].startswith('the latest start over its legs\' S3') and EV.S3_st[2] == 0 and EV.S3_hi[2] == 1000.
            and EV.ST_st[2] == 2 and EV.ST_lo[1] == 850. and EV.ST_hi[1] == 1000.):
        bad.append(f'events_of S3: {EV[["S3_lo", "S3_hi", "S3_st"]].to_dict("index")}')
    want = {1: (0, 850., 1000., 0), 2: (1, 760., 1000., 1), 3: (0, 750., 1000., 0), 4: (0, 1050., 1000., 3), 5: (2, 2500., 3500., 0), 6: (1, 950., 1450., 0)}
    got = {g: (int(r.cls), float(r.S1_lo), float(r.S1_hi), int(r.S1_st)) for g, r in EV.iterrows()}
    if got != want:
        bad.append(f'events_of: {got} != {want}')
    lab = {g: (r.S1_lo_type, r.S1_hi_type, r.EV_lo_type, r.EV_hi_type) for g, r in EV.iterrows()}
    if not (lab[1] == ('assumption_range', 'assumption_range', 'unknown', 'evidence_bound') and lab[2][1] == 'evidence_bound' and EV.S1_hi_basis[2] == CAP_BASIS
            and lab[4][:2] == ('contradictory', 'contradictory') and EV.S1_lo_basis[5].startswith('latest REST second') and EV.S1_lo_basis[1].startswith('latest WS pending')
            and float(EV.rep_time[1]) == 950. and EV.rep_rule[1] == REP_RULE):
        bad.append(f'explicit bound types / bases / representative time: {lab} {EV[["S1_lo_basis", "S1_hi_basis", "rep_time"]].to_dict("index")}')
    if schema_gaps({'ledger_fills': ['fid', 'cls_S1_CP', 'zzz']}) != ['ledger_fills.zzz'] or schema_gaps({'ledger_events': list(EV.reset_index().columns) + ['e']}):
        bad.append('the column dictionary check: an undocumented column must be named, every events column documented')
    L2 = L.copy(); L2.loc[L2.group == 3, 'exec_s'] = -1.                                      # evidence cap at 0 ms < the window's start
    if int(events_of(L2).set_index('group').S1_st[3]) != 2:
        bad.append('an evidence-contradicted event not marked contradictory_evidence')
    LT = pd.DataFrame(dict(mid=[7, 7, 7, 7, 7, 8, 7], side=[0, 0, 0, 0, 0, 1, 0], anc=[1000., 1000., 1000., 2000., 1000., 3000., 1000.], ptick=[50, 49, 51, 50, 50, 60, 52],
                           group=[9, 9, 9, 3, 4, 5, 8]))                                        # group 8: at the anchor, but it cannot fill a bid at 50
    Fd = pd.DataFrame(dict(fid=[0, 1, 2], m=[7, 7, 8], side=[0, 0, 1], a=[1000.004, 2000., 2500.], p=[50, 50, 60]))
    pr, lg = link(Fd, LT)
    if sorted(map(tuple, pr[['fid', 'group']].values.tolist())) != [(0, 4), (0, 9), (1, 3)] or sorted(lg.group.unique().tolist()) != [3, 4, 9] or len(lg) != 5:
        bad.append(f'link: {pr.values.tolist()} / {sorted(lg.group.unique().tolist())}')
    if combine(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3).tolist() != [2, 1, -1]:
        bad.append('combine: disagreeing candidates must be AMBIGUOUS, none = -1')
    CWp = np.zeros((1, KP.NSTEP + 2)); CWp[0, 5:] = .5; Fp = dict(val=np.array([-10.]), pulled=np.array([True]), T=np.array([5]), s=np.array([0]), endj=np.array([99]),
                                                               lend=np.array([np.inf]))
    got = [float(contrib(np.array([300.]), Fp, CWp, lat)[0]) for lat in LATS]
    if got != [5., 5., 0.]:
        bad.append(f'contrib at cancel latencies 0 / 37 / 100: {got} != [5, 5, 0] (score and null landings move with the latency)')
    wl_, wh_ = event_axis_windows(EV.reset_index(), 'S1'); gi = {g: i for i, g in enumerate(EV.index)}
    if not (np.isneginf(wl_[gi[4]]) and np.isposinf(wh_[gi[4]]) and (wl_[gi[1]], wh_[gi[1]]) == (-150., 0.)):
        bad.append('event-axis windows: a contradictory event must move over the whole axis')
    n = 240; cell = np.repeat(np.arange(20), 12); endj = rng.integers(KP.J0, KP.NSTEP + 2, n); T = np.where(rng.random(n) < .5, rng.integers(KP.J0, 40, n), KP.NSTEP + 1)
    m = 600; ri = rng.integers(0, n, m); t = np.r_[rng.integers(60, 4990, m - 30).astype(float), np.full(30, KP.H)]; val = rng.normal(0., 10., m)
    lend = np.where(rng.random(n) < .2, rng.integers(500, 4000, n).astype(float), np.inf); keep = t < lend[ri]; ri, t, val = ri[keep], t[keep], val[keep]; m = len(ri)
    act, CW = V1.null_weights(T, cell, endj); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    F = dict(val=val, pulled=act[ri], T=T[ri], s=cell[ri], endj=endj[ri], lend=lend[ri]); F1 = dict(val=val, pulled=act[ri], r=La[np.clip(T[ri], 1, KP.NSTEP) - 1], s=cell[ri], endj=endj[ri], lend=lend[ri])
    pf = A.per_fill(T, cell, endj, (ri, t, val, np.zeros(m, int)))
    if abs(contrib(t, F, CW).sum() - (pf['score'].sum() - pf['null'].sum())) > 1e-6:
        bad.append('collapsed contributions differ from the clock audit per fill')
    lo = t - rng.integers(0, 600, m); hi = t + rng.integers(0, 600, m); mn1, mx1 = V1.bounds(lo, hi, F1, CW)
    mn, mx = sweep(np.arange(m), lo, hi, np.zeros(m), F, CW, KP.CANCEL)
    if not (np.allclose(mn, mn1, atol=1e-9) and np.allclose(mx, mx1, atol=1e-9)):
        bad.append('singleton sweep differs from v1 bounds')
    nE = 30; ev = rng.integers(0, nE, m); d = rng.integers(-300, 300, m).astype(float); ctr = rng.integers(0, 5000, nE).astype(float)
    wl = ctr - rng.integers(0, 700, nE); wh = ctr + rng.integers(0, 700, nE); wl[:4] = -np.inf; wh[2:6] = np.inf
    for lat in LATS:
        a1, b1 = sweep(ev, wl, wh, d, F, CW, lat, chunk=50); a2, b2 = _brute(ev, wl, wh, d, F, CW, lat)
        if not (np.allclose(a1, a2, atol=1e-9) and np.allclose(b1, b2, atol=1e-9)):
            bad.append(f'event sweep differs from the dense grid at lat {lat:g}: {int((~np.isclose(a1, a2)).sum())} minima, {int((~np.isclose(b1, b2)).sum())} maxima')
    Fz = dict(val=np.array([-10., 10.]), pulled=np.array([True, True]), T=np.array([5, 5]), s=np.array([0, 0]), endj=np.array([99, 99]), lend=np.array([np.inf, np.inf]))
    mz, xz = sweep(np.array([0, 0]), np.array([0.]), np.array([1000.]), np.zeros(2), Fz, np.zeros((1, KP.NSTEP + 2)), KP.CANCEL)
    if not (mz[0] == 0. and xz[0] == 0.):
        bad.append('two jumps of one event at one instant must be summed before the state is read (true sum 0 everywhere)')
    if candidate_status(np.array([0, 0, 1, 1, 2]), np.array([True, False, True, True, False]), 4).tolist() != [2, 3, 1, 0]:
        bad.append('candidate status: some / all / none / unlinked')
    Fj = dict(val=np.array([-10., -10., 10.]), pulled=np.ones(3, bool), T=np.full(3, 5), s=np.zeros(3, int), endj=np.full(3, 99), lend=np.full(3, np.inf))
    args_j = (np.array([0, 1, 2, 2]), np.array([0, 1, 0, 1]), 3, np.full(2, 100.), np.full(2, 400.), np.zeros(2), np.zeros(3), Fj, np.zeros((1, KP.NSTEP + 2)), KP.CANCEL)
    Uj, Ur = joint_bounds(*args_j), joint_bounds(*args_j, cap=1)                               # review of c23e6f29: coherent [0, 10]; relaxed [-10, 20]
    if not ([float(Uj.mn.sum()), float(Uj.mx.sum())] == [0., 10.] and Uj.kind.tolist() == ['joint'] and int(Uj.fills.sum()) == 3
            and [float(Ur.mn.sum()), float(Ur.mx.sum())] == [-10., 20.] and Ur.kind.tolist() == ['relaxed']):
        bad.append(f'joint bounds: exact {Uj.to_dict("records")}, relaxed {Ur.to_dict("records")} (want [0, 10] joint, [-10, 20] relaxed)')
    Uc = joint_bounds(np.array([0, 1]), np.array([0, 0]), 2, np.array([100.]), np.array([400.]), np.zeros(1), np.zeros(2),
                      {k: v[[0, 2]] for k, v in Fj.items()}, np.zeros((1, KP.NSTEP + 2)), KP.CANCEL)
    if not ([float(Uc.mn.sum()), float(Uc.mx.sum())] == [0., 0.] and Uc.kind.tolist() == ['event']):
        bad.append('joint bounds: two opposite fills of one event must give [0, 0]')
    for trial in range(4):                                                                     # the joint solver against an independent brute force
        nf_ = 6; Fr = dict(val=rng.normal(0., 10., nf_), pulled=rng.random(nf_) < .7, T=rng.integers(KP.J0, 8, nf_), s=np.zeros(nf_, int), endj=np.full(nf_, 99),
                           lend=np.where(rng.random(nf_) < .3, rng.integers(150, 450, nf_).astype(float), np.inf))
        CWr = np.zeros((1, KP.NSTEP + 2)); CWr[0, 3:] = .3; CWr[0, 6:] = .6; wl_r = rng.integers(0, 200, 2).astype(float); wh_r = wl_r + rng.integers(0, 160, 2)
        pf_r = np.array([0, 1, 2, 3, 3, 4, 4, 5]); pe_r = np.array([0, 1, 1, 0, 1, 0, 1, 0]); d_r = rng.integers(-60, 60, nf_).astype(float); an_r = np.zeros(2)
        Ub = joint_bounds(pf_r, pe_r, nf_, wl_r, wh_r, an_r, d_r, Fr, CWr, KP.CANCEL)
        g0_, g1_ = (np.arange(wl_r[e], wh_r[e] + 1e-9, .5) for e in (0, 1)); T0, T1 = np.meshgrid(g0_, g1_, indexing='ij'); best = [np.inf, -np.inf]
        cand_ = {f: pe_r[pf_r == f].tolist() for f in range(nf_)}
        for asg in itertools.product(*[cand_[f] for f in range(nf_)]):
            tot = sum(contrib(np.where(e == 0, T0, T1).ravel() - d_r[f], {k: np.full(T0.size, v[f]) for k, v in Fr.items()}, CWr) for f, e in enumerate(asg))
            best = [min(best[0], float(tot.min())), max(best[1], float(tot.max()))]
        if not (abs(float(Ub.mn.sum()) - best[0]) < 1e-9 and abs(float(Ub.mx.sum()) - best[1]) < 1e-9 and set(Ub.kind) == {'joint'}):
            bad.append(f'joint bounds trial {trial}: {float(Ub.mn.sum()):.6f} / {float(Ub.mx.sum()):.6f} != brute force {best}')
        Uo = joint_bounds(pf_r, pe_r, nf_, wl_r, wh_r, an_r, d_r, Fr, CWr, KP.CANCEL, cap=0)
        if not (float(Uo.mn.sum()) <= best[0] + 1e-9 and float(Uo.mx.sum()) >= best[1] - 1e-9 and set(Uo.kind) == {'relaxed'}):
            bad.append(f'outer relaxation trial {trial} does not contain the exact bounds')
    X = xtab(np.array([0, 1, 2, 2, 4, -1]), np.array([True, True, False, False, True, True]), np.array([-5., 3., 2., -1., 4., 9.]), np.array([True] * 5 + [False]))
    if not (conserved(X) and X['REMOVED']['CANCEL_FIRST']['net'] == 5. and X['KEPT']['AMBIGUOUS']['fills'] == 2 and X['REMOVED']['_total']['fills'] == 3):
        bad.append('cross-tab cells / conservation')
    X['KEPT']['AMBIGUOUS']['fills'] = 1
    if conserved(X):
        bad.append('conservation check does not fire on a lost fill')
    base = 1.79e12; pl_ = base + 1000. * np.arange(8) * 60.
    Kk = pd.DataFrame(dict(run=['a'] * 4 + ['b'] * 4, cid=np.arange(8), place=pl_, accepted=pl_ + 30., upper=pl_ + 160., notice=pl_ + 600., ack=np.r_[pl_[:7] + 160., np.nan],
                           req=pl_ + 100., S=np.floor((pl_ + 80.) / 1000.), M=np.floor((pl_ + 80.) / 1000.), Z=pl_ + 70., signed=pl_ + 5., ucancel=np.nan, cdone=pl_ + 170.,
                           thr=np.nan, rM=pl_ + 150., rC=pl_ + 1200., Cs=np.floor((pl_ + 80.) / 1000.) + 1.))
    Kk.loc[1, 'notice'] = pl_[1] + 90.; Kk.loc[1, 'upper'] = pl_[1] + 90.                      # notice before the cancel request: MATCH_FIRST
    Kk.loc[2, 'S'] = np.floor((pl_[2] + 80.) / 1000.) + 10.                                     # Confirmed 10 s late: the substitution must fire
    Kk.loc[3, 'Z'] = pl_[3] + 10070.; Kk.loc[6, 'Z'] = pl_[6] - 5000.                           # research anchors outside their lifecycles
    ref = dict(rest_floor_nonoverlap=int(miss(Kk.S.values * 1000., Kk.S.values * 1000. + 999.999, Kk).sum()), matched_floor_nonoverlap=0)
    Rk, rk = known_checks(Kk, ref, draws=50)
    inst_ = ('K0 REPRODUCES 11-09', 'K1 CLASSIFIER-CONSISTENCY', 'K2 DELAYED-STATUS-REFUSED', 'K3 IDENTITY-CHECK-FIRES')
    if not (all(rk[x] == 'YES' for x in inst_) and rk['K4 REPLAY-ANCHOR-VS-LIFECYCLE'] == 'CONTRADICTED'
            and rk['KNOWN VERDICT'] == ('INSTRUMENT-CHECKS-PASS; REPLAY-ANCHOR-CONTRADICTED; SETTLEMENT-TIME NOT-RUN; LEDGER-WINDOW GATE NOT-RUN (K7 offsets; conditional - '
                                        'K8 shared clock: NOT-RUN; settlement offsets: NOT-RUN); ORDER-PATH RANGES NOT-CONTRADICTED; NOT A VENUE-ENGINE ACCEPTANCE')
            and Rk['K1']['MATCH_FIRST'] == 1 and Rk['K1']['AMBIGUOUS'] == 6 and Rk['K1']['CANCEL_FIRST'] == 0 and Rk['K2']['point_contradictions'] == 1
            and Rk['K4']['S1']['missed'] == 2 and Rk['K4']['S1']['cancel_first'] == 1 and not Rk['K4']['S1']['constant_offset']['a']['exists'] and Rk['K4']['S1']['anchored'] == 8):
        bad.append(f'known checks on planted cases: {rk} {Rk}')
    if not (np.flatnonzero(Rk['_legs']['K4_S1_missed']).tolist() == [3, 6] and np.flatnonzero(Rk['_legs']['K4_S1_cancel_first']).tolist() == [3]
            and Rk['_legs']['K4_anchor_minus_place_ms'][3] == 10070.):
        bad.append(f"K4 per leg: {np.flatnonzero(Rk['_legs']['K4_S1_missed'])} {np.flatnonzero(Rk['_legs']['K4_S1_cancel_first'])}")
    if not (Rk['K5']['orders'] == 8 and Rk['K5']['activation'] == {'COMPATIBLE-UNIDENTIFIED': 8} and Rk['K5']['cancel'] == {'CERTIFIED': 7, 'NO-CANCEL-CONFIRMATION': 1}
            and Rk['K5']['confirmation_after_request_ms']['max'] == 60. and rk['K7 CLOCK OFFSET (venue status - bot)'] == '[-150.0, +1000.0] ms, CONTAINS 0'):
        bad.append(f"K5 / K7 on planted cases: {Rk['K5']} {rk['K7 CLOCK OFFSET (venue status - bot)']}")
    Ks = Kk.copy(); Ks['ack'] = Ks.req + 5000.; Ks['upper'] = np.minimum(Ks.notice, Ks.ack)                # confirmations 5 s after the request: an upper bound only
    Rs, rs = known_checks(Ks, dict(ref, rest_floor_nonoverlap=int(miss(Ks.S.values * 1000., Ks.S.values * 1000. + 999.999, Ks).sum())), draws=50)
    if not (rs['K5 ORDER-PATH INTERVALS vs DECLARED RANGES'] == 'NOT-CONTRADICTED' and Rs['K5']['cancel'] == {'COMPATIBLE-UNIDENTIFIED': 8}):
        bad.append(f"a late cancel confirmation read as a contradiction (or certified) by K5: {Rs['K5']['cancel']}")
    for acc, sig, want in ((40., 5., 'COMPATIBLE-UNIDENTIFIED'), (1., .5, 'CONTRADICTED'), (30., 26., 'CERTIFIED')):   # the review's positive / known-bad cases
        Ka = Kk.copy(); Ka['accepted'] = Ka.place + acc; Ka['signed'] = Ka.place + sig; Pa, k5a = order_path(Ka)
        if k5a['activation'] != {want: 8} or (want == 'CONTRADICTED') != (known_checks(Ka, ref, draws=10)[1]['K5 ORDER-PATH INTERVALS vs DECLARED RANGES'] == 'CONTRADICTED'):
            bad.append(f"K5 activation accepted +{acc:g} / signed +{sig:g} ms: {k5a['activation']} (want {want})")
    Kd = pd.concat([Kk, Kk.iloc[[0]]], ignore_index=True); Kd.loc[3, 'ack'] = Kd.req[3] - 10.; Kd.loc[4, 'ucancel'] = Kd.ack[4]; Kd.loc[5, 'ucancel'] = Kd.ack[5] + 50.
    Pd_, k5d = order_path(Kd)                                                                     # two legs of one order; a confirmation before its request; sources
    if not (k5d['orders'] == 8 and k5d['legs'] == 9 and k5d['cancel'].get('INCONSISTENT-CLOCKS') == 1 and list(Pd_.confirm_source[[4, 5, 6]]) == ['user_stream', 'http_answer', 'http_answer']):
        bad.append(f"K5 per order / statuses / sources: {k5d} {Pd_.confirm_source.tolist()}")
    if offset_interval(Kk.assign(rM=Kk.rM - np.r_[0., 0., 3000., np.zeros(5)]))['exists']:                # a notice received 2.85 s before its status second
        bad.append('K7: inconsistent status receipts must leave no admissible offset')
    Kn = Kk.copy(); Kn.loc[2, 'Z'] = np.nan; R_n = known_checks(Kn, ref, draws=50)[0]['K4']['S1']              # a leg without a research anchor
    if not (R_n['no_anchor'] == 1 and R_n['anchored'] == 7 and all(np.isfinite(v['lo']) and np.isfinite(v['hi']) for v in R_n['constant_offset'].values()) and R_n['missed'] == 2):
        bad.append(f'K4 with a leg without an anchor: {R_n}')
    if not (np.array_equal(struct_landing(np.array([1050., np.nan, 50.]), 0.), np.array([np.nextafter(1050., np.inf), np.inf, 58.]))
            and np.array_equal(struct_landing(np.array([1050., np.nan, 50.]), KP.CANCEL), np.array([1087., np.inf, 95.]))
            and np.array_equal(at_latency(dict(val=np.zeros(3)), np.array([1050., np.nan, 50.]), 100.)['lend'], np.array([1150., np.inf, 158.]))):
        bad.append('the structural landing at latency 0 / 37 / 100 ms')
    Kg = Kk.copy(); Kg.loc[3, 'Z'] = pl_[3] + 70.; Kg.loc[6, 'Z'] = pl_[6] + 70.                 # every research anchor inside its lifecycle
    if known_checks(Kg, ref, draws=50)[1]['K4 REPLAY-ANCHOR-VS-LIFECYCLE'] != 'CONSISTENT':
        bad.append('consistent research anchors not read CONSISTENT by K4')
    Kc = Kk.copy(); Kc.loc[5, 'S'] = np.floor(Kc.place[5] / 1000.) - 5.                       # a REST second ending before the order existed
    if known_checks(Kc, dict(ref, rest_floor_nonoverlap=ref['rest_floor_nonoverlap'] + 1), draws=50)[1]['K2 DELAYED-STATUS-REFUSED'] != 'NO':
        bad.append('an evidence bound that contradicts its lifecycle passes K2')
    Kb = Kk.copy(); Kb.loc[0, 'M'] += 5.                                                       # a wrong identity: Matched second 5 s off its order
    if known_checks(Kb, dict(ref, matched_floor_nonoverlap=1), draws=50)[1]['K3 IDENTITY-CHECK-FIRES'] != 'NO':
        bad.append('a wrong identity passes K3')
    if case_gaps({'known_cases': ['run', 'S3_verdict', 'zzz'], 'known_orders': ['cid', 'act_lo']}) != ['known_cases.zzz']:
        bad.append('the case-ledger column check: an undocumented column must be named, the documented ones accepted')
    bad += _k6_selftest() + _s3_selftest() + _settlement_selftest() + _k6_domain_selftest()
    if rehearse and not bad:
        O, Fl, Ls, Ps = _synthetic(rng); day = '20990101'; Ls_rows = Ls[RAW_COLS + ['tk', 'tk_status']].reset_index(drop=True)
        env = dict(h=600., q01=-900., q99=1050., inputs={}); v1, e_, s_, aux = _v1_expect(O, Fl, Ls); ri_, t_, val_, act_, r_ = (aux[k] for k in ('ri', 't', 'val', 'act', 'r'))
        tset = pd.DataFrame(dict(t_set_ms=np.where(Ls.tk.values >= 0, Ls.anc.values - 20., np.nan), set_status=np.where(Ls.tk.values >= 0, 0, 1)))   # synthetic settlement times
        base = dict(src=_Src(Ls[LEG_COLS]), tables=(O, Fl), rows_of=lambda d_: Ls_rows, env_of=lambda d_: env, v1=v1, days=[day], expect=(e_, s_), tol=1e-6,
                    sem_of=lambda d_, rows: (sem_windows(rows, Ps), {}), set_of_=lambda d_: (tset, {}), st_lag=150.)
        with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
            try:
                R, read = run(out=Path(tmp) / 'x', **base)
                outs = sorted(p.name for p in (Path(tmp) / 'x').iterdir()); FLx = pd.read_parquet(Path(tmp) / 'x' / 'ledger_fills.parquet')
                OLx = pd.read_parquet(Path(tmp) / 'x' / 'ledger_orders.parquet')
                if not (read['V0 CONTROLS'] == 'YES' and len(FLx) == len(ri_) and len(OLx) == len(O) and R['link']['unlinked'] == 0 and R['link']['multi_event'] > 0
                        and 'ledger_legs.parquet' in outs and 'exec_windows.json' in outs and (FLx.link != 'unlinked').all() and 'ledger_schema.json' in outs
                        and all(set(pd.read_parquet(Path(tmp) / 'x' / f'{t}.parquet').columns) == set(json.loads((Path(tmp) / 'x' / 'ledger_schema.json').read_text())['tables'][t])
                                for t in SCHEMA)):
                    bad.append(f'rehearsal outputs: {outs} {R["link"]}')
                Xs = R['xtab']['S1|CP']; npl = int(act_[ri_].sum())
                if not (Xs['REMOVED']['_total']['fills'] + Xs['KEPT']['_total']['fills'] == npl < len(ri_)
                        and abs(R['sim_removed'] - float(-val_[act_[ri_] & (t_ >= r_)].sum())) < 1e-9):
                    bad.append(f'rehearsal cross-tab population: {npl} pulled-order fills of {len(ri_)}; removed {R["sim_removed"]}')
                g_ = FLx[FLx.pulled].groupby('oid').cls_S1_CP.apply(lambda x: bool(np.isin(x.values, (0, 1)).all()))
                P_ = OLx[OLx.decision_S1.isin(['evaluable', 'not_evaluable', 'no_m50_fill'])]
                want_ = np.where(P_.m50_fills.values == 0, 'no_m50_fill', np.where(g_.reindex(P_.oid.values).fillna(False).values.astype(bool), 'evaluable', 'not_evaluable'))
                if not ((P_.decision_S1.values == want_).all() and R['decisions']['S1']['evaluable'] == int((OLx.decision_S1 == 'evaluable').sum()) > 0):
                    bad.append('rehearsal decisions differ from the ledger fills\' classes')
                b37 = R['bounds'][f'S1|{KP.CANCEL:g}']
                if not (b37['bounds'] in ('exact', 'outer') and sum(b37['fills'].values()) == b37['of_fills'] == len(ri_) and b37['units']['joint'] > 0):
                    bad.append(f'rehearsal bounds metadata: {b37["bounds"]} {b37["units"]} {b37["fills"]}')
                s3d = R['s3'][day]; LGx = pd.read_parquet(Path(tmp) / 'x' / 'ledger_legs.parquet')
                if not (set(R['V3']) == {'S1', 'S1_cap', 'S1_legs', 'S2', 'S2_cap', 'S2_legs', 'S3', 'S3_cap', 'S3_legs', 'ST', 'ST_cap', 'ST_legs', 'after_evidence'}
                        and causes_reconcile(R['V3']) and set(R['disagreement']) == {'S1', 'S2', 'S3', 'ST'} and s3d['day_legs']['ws_multi'] > 0 and s3d['day_legs']['ws_unique'] > 0
                        and R['st_lag_ms'] == 150. and R['settlement'][day]['with_t_set'] == int((Ls.tk >= 0).sum()) and f'ST|{KP.CANCEL:g}' in R['bounds'] and 'V1 SIGN ST' in read
                        and R['replay_point_minus_t_set_ms']['p50'] == -30. and R['V3']['ST']['fills']['all'] == 0
                        and R['settlement'][day]['ws_leg_link_minus_t_set_ms']['p1'] >= 20. and R['settlement'][day]['ws_leg_link_minus_t_set_ms']['below_minus_lag'] == 0.
                        and np.array_equal(LGx.ST_llo.values, np.where(np.isfinite(LGx.t_set_ms.values), LGx.t_set_ms.values - 150., LGx.S3_llo.values))
                        and s3d['day_legs']['rest_range_e2_unsupported'] > 0 and f'S3|{KP.CANCEL:g}' in R['bounds'] and 'V1 SIGN S3' in read
                        and np.array_equal(LGx.S3_llo.values, sem_windows(Ls_rows, Ps).S3_llo.values[LGx.row_id.values.astype(int)])):
                    bad.append(f"rehearsal S3 / causes: {sorted(R['V3'])} {s3d} {R['disagreement']}")
            except (SystemExit, ValueError) as e:
                bad.append(f'rehearsal refused: {e}')
            # check of 9a114cc1: the PRODUCTION default - no lag passed, the corrected known receipt read, ST upper-only - rehearsed end to end
            global OUT_K
            kk = Path(tmp) / 'k5'; kk.mkdir(); (kk / 'known.json').write_text(json.dumps(dict(R=dict(K8=dict(model=SETTLEMENT_MODEL, consistent=True, lag_bound_ms=None)))))
            old_k = OUT_K; OUT_K = kk
            try:
                Ru, ru = run(out=Path(tmp) / 'u', **dict(base, st_lag=None))
                LGu = pd.read_parquet(Path(tmp) / 'u' / 'ledger_legs.parquet'); EVu = pd.read_parquet(Path(tmp) / 'u' / 'ledger_events.parquet'); mu = np.isfinite(LGu.t_set_ms.values)
                if not (ru['V0 CONTROLS'] == 'YES' and Ru['st_lag_ms'] is None and Ru['st_lag_role'] == 'UNIDENTIFIED' and mu.any() and (~mu).any()
                        and np.isneginf(LGu.ST_llo.values[mu]).all() and np.array_equal(LGu.ST_llo.values[~mu], LGu.S3_llo.values[~mu])
                        and (EVu.ST_lo_type == 'unknown').any() and f'ST|{KP.CANCEL:g}' in Ru['bounds'] and 'V1 SIGN ST' in ru
                        and Ru['V3']['ST']['fills']['all'] == 0 and causes_reconcile(Ru['V3'])):
                    bad.append(f"upper-only production rehearsal: lag {Ru['st_lag_ms']} role {Ru['st_lag_role']} mapped legs {int(mu.sum())} of {len(mu)}")
            except (SystemExit, ValueError) as e:
                bad.append(f'upper-only production rehearsal refused: {e}')
            finally:
                OUT_K = old_k
            # review of c23e6f29: every candidate window contradicted and every replay time after its cap - ALL fills count, multi-event ones too
            Lc = Ls.copy(); Lc['how'] = 1; Lc['t_ms'] = Lc.anc; Lc['t_e2'] = np.nan; Lc['exec_s'] = np.floor(Lc.anc / 1000.) - 10.
            Lc_rows = Lc[RAW_COLS + ['tk', 'tk_status']].reset_index(drop=True); v1c, ec_, sc_, _ = _v1_expect(O, Fl, Lc)
            try:
                Rc, _ = run(out=Path(tmp) / 'c', **dict(base, src=_Src(Lc[LEG_COLS]), rows_of=lambda d_: Lc_rows, v1=v1c, expect=(ec_, sc_)))
                if not (Rc['link']['multi_event'] > 0 and all(Rc['V3'][k]['fills']['all'] == len(ri_) and abs(Rc['V3'][k]['gross']['all'] - Rc['V3'][k]['of_gross']) < 1e-9
                                                              for k in ('S1', 'S2', 'after_evidence', 'S1_cap', 'S2_cap'))
                        and Rc['V3']['S1_legs']['fills']['all'] == 0 and Rc['V3']['S3']['fills']['all'] == 0 and Rc['s3'][day]['day_link_late'] == len(Lc)):
                    bad.append(f"all-contradictory case: {[(k, Rc['V3'][k]['fills']) for k in ('S1', 'S2', 'after_evidence')]} of {len(ri_)} fills")
                if accounted({'x': dict(Rc['V3']['S1'], fills=dict(Rc['V3']['S1']['fills'], all=Rc['V3']['S1']['fills']['all'] - 1))}, len(ri_), Rc['V3']['S1']['of_gross']):
                    bad.append('the contradiction accounting check does not fire on a lost fill')
                Ro, ro = run(out=Path(tmp) / 'o', **dict(base, jcap=0))                       # every component forced to the outer relaxation
                if not (Ro['bounds'][f'S1|{KP.CANCEL:g}']['bounds'] == 'outer' and Ro['bounds'][f'S1|{KP.CANCEL:g}']['units']['relaxed'] > 0
                        and ro['V1 SIGN S1'].endswith('(OUTER BOUNDS)') and Ro['bounds'][f'S1|{KP.CANCEL:g}']['E_min'] <= R['bounds'][f'S1|{KP.CANCEL:g}']['E_min'] + 1e-9):
                    bad.append('outer-relaxed bounds not labelled OUTER (or not containing the exact ones)')
            except (SystemExit, ValueError) as e:
                bad.append(f'all-contradictory / outer case refused: {e}')
            rb1 = Ls_rows.copy(); rb1.loc[0, 'shares'] += 1.; rb2 = Ls_rows.copy(); rb2.loc[0, 'tk'] = rb2.tk[0] + 1000 if rb2.tk[0] >= 0 else 7
            v1b = {k: dict(v, E_min=v['E_min'] - 1.) for k, v in v1.items()}
            for nm_, kw in (('a wrong collapsed expectation', dict(expect=(e_ + 1., s_))), ('a wrong v1 reference', dict(v1=v1b)),
                            ('a leg field that differs from its legs_timed2 row', dict(rows_of=lambda d_: rb1)), ('a leg whose event id breaks the taker-order rule', dict(rows_of=lambda d_: rb2))):
                o_ = Path(tmp) / ('y' + str(abs(hash(nm_)) % 1000))
                try:
                    run(out=o_, **dict(base, **kw)); bad.append(f'run accepted {nm_}')
                except SystemExit:
                    pass
                if o_.exists():
                    bad.append(f'a refused run wrote output ({nm_})')
    if bad:
        sys.exit('EXEC-WINDOW SELFTEST FAILED: ' + ' | '.join(bad))
    print('exec-window selftests passed (synthetic only): classes incl. TIE / UNKNOWN / CONTRADICTORY; status timestamps refused as points; event intersection, '
          'evidence cap and statuses; link candidates; the event sweep = a dense grid (3 latencies) and v1 on singletons; the joint bounds = the review\'s '
          'coherent [0, 10] (outer relaxation [-10, 20] labelled) and [0, 0]; per-fill contradiction status; collapsed = the clock audit; cross-tab '
          'conservation (and its falsifier); K0-K3 instrument checks and K4 CONTRADICTED / CONSISTENT on planted cases (a wrong identity refused); '
          'run() rehearsed end to end, the all-contradictory case counting every fill, four bad inputs refused; S3 windows on planted increments; the '
          'contradiction causes reconciled (and their falsifier); binding pairs; K5 per order on the review\'s cases; K6 / K7 on planted lifecycles; the '
          'case linkage and the case ledger (an undocumented column refused); K6 per clock domain (fail-closed, status ends at K7, M50 by origin, the '
          'split reproduced) and the census reader\'s refusals', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        run()
    elif len(a) == 2 and a[0] == 'run':                                     # one consumed day first (its own output dir, the per-day committed controls)
        C.require(a[1] in KP.DAYS, 'undeclared day'); run(out=OUT.parent / f'{OUT.name}_{a[1]}', days=[a[1]])        # keep_pull_exec_windows_v3_20261118_<day>
    elif a == ['known']:
        known()
    else:
        sys.exit('usage: pf_keep_pull_exec_windows.py selftest | run [day] | known')
