"""TIMING CASE AUDIT OF THE S3 FALSE-AVOIDANCE LEGS (2026-11-18, session aea5bfb9) - the review of 3bf8f3a9
(reviews/automatic/20261118_3bf8f3a9/report.md; USER: "Check review and next priority, proceed next"). Its one next priority: "Resolve the
timing gate with an independent clock/identity case audit of the three S3 false-avoidance legs. For each, retain the exact raw
identity-bearing placement, match/status/UUID, first-notice and cancel-confirmation records; document clock domains and derive only
independently supported bounds. A conclusive result must either justify the clock constraint that removes the contradiction, or retain it and
identify the missing observation needed to resolve it. Passing under a narrower fitted clock is insufficient. Use the same audit to
distinguish evidence-backed match windows from assumed fallback windows before using their economics as model targets."
Declared before any data is read. The one look taken before: a feasibility count of the frame identity rule on the three legs' markets
(bot frames with one / several / no content twin; counts only, no time difference computed or printed; its basis scan passed Nov 9 lines
through a byte-prefix filter without parsing any - this module stops at the cutoff instead).

WHY THESE LEGS FAIL (known v5, computed there): K6 calls a known fill CANCEL_FIRST at offset d (venue - bot) iff its cancel confirmation c_hi
(bot clock) precedes the event window's lower end lo - d. For the three legs lo is the event's S3 lower end; the leg is called avoided only for
d < lo - c_hi (the CRITICAL offset, about -210 / -219 / -225 ms), which K7's status-clock interval [-226.4, +49.3] ms admits. K7 bounds the
REST STATUS clock (floor-second stamps, notices); the ledger applies it to every venue window. This audit asks which clock lo is on and
whether that clock is bounded independently of the gate's outcome.

POPULATION  the legs known ledger v5 (keep_pull_exec_windows_known_v5_20261118, outputs hash-checked against known.json) marks
            S3_false_cf_some_offset (the count is read, not assumed). Consumed Nov 8 only; no record stamped at or after CUTOFF_MS (Nov 8 23:02Z,
            the fill-clock audit's cutoff; Nov 9 is protected) is parsed - every reader stops there.
SOURCES (hashes in the receipt)  pf-exec kernel journals (live-samewindow-20261108-kernel/{run}.kernel.jsonl: Control / Exec step frames and
            their outboxes), pfmm input journals (tokyo-live-20261109/{run}/run-*/inputs.jsonl.gz: the engine's book frames and order events),
            core_orders_{run} (cid -> engine id), the research collector's Nov 8 WS (predictOrderbook) and REST (/v1/orders/matches) hour files,
            Nov 8's research tables (legs_timed2, legs_tk, pend_inc), and the Oct 23 ledger v5 tables (A6).
CLOCK DOMAINS (CLOCKS below): B the bot host wall clock (kernel `at`, input `known_ms`); B_mono its monotonic clock; V_book the venue book
            stamp updateTimestampMs (S3's WS windows and pend_inc t_ms are on it); V_status the status seconds (pf-exec Fill match_time rounded to
            the second, REST executedAt = the Confirmed second); V_uuid the settlementId UUIDv7 ms; R the research collector's receipt clock (file
            selection only). No two venue domains are assumed aligned.
RAW RECORDS per leg, verbatim (never re-serialized; kernel / input / WS lines as read, REST match elements with their page line's hash):
            placement (Control Place + outbox, Signed, Accepted, the user-stream Placement), match / status / UUID (every Fill notice; the REST
            match elements of its transaction), first notice (the first positive Fill carrying the trade id), cancel (Control Cancel + outbox,
            CancelAnswer, UserCancel, the cancel audit records), and the book frame whose pending increment sets the event's lower end.
BOUNDS per leg, each typed: EVIDENCE (a causal relation between identity-linked records on one clock), CONDITIONAL (holds under a stated
            venue premise), ASSUMPTION (a declared range, e.g. S3's WS_DELAY). They are listed, never fitted.
INDEPENDENT BOOK-CLOCK BOUND  d_book = V_book - B >= u - known_ms for every book frame the bot received (premise: a frame's updateTimestampMs is
            stamped before the bot receives it, and the research tape carries the same stamp for the same book update). Identity of a bot frame
            with a research frame: the same market, the same five best levels per side (tick, size to DP decimals) and the same pending amounts
            per (side, tick), with exactly ONE research twin within +-W_ID of the bot receipt and exactly one bot twin within +-W_ID of that
            research frame; anything else is a status (none / several). Local window: bot receipts within +-W_LOCAL of the leg's cancel
            confirmation. Declared statistic B = the ceil(n/2)-th largest u - known_ms: d_book >= B holds if at least one of those ceil(n/2) pairs
            is a correct identity (a wrong pairing cannot raise B unless more than half the upper pairs are wrong). B_max (needs every pair
            right) and the binding frame's own pair are printed beside it, not used.
READINGS (declared; evaluated in code):
  A0 RECORDS-REPRODUCE per leg: the kernel's place / signed / accepted / first notice / first Matched and Confirmed receipts and stamps, trade
     id and order hash equal the known case table's (0.001 ms; ids exact), and the order's Accepted carries the same instant in both bot
     journals (one bot clock). YES / NO (each field named).
  A1 CANCEL-ANSWER per leg (the gate's premise, not a condition of A5): the venue's first answer to the cancel, verbatim, classed
     MATCHED-OR-GONE (a CancelAnswer NotCanceled "can't be found - already canceled or matched": the venue found the order gone, so the match
     preceded its processing - match <= c_hi is EVIDENCE), CANCELED (pf-exec journals the first confirmation, the venue's Canceled answer or a
     user-stream cancel, as CancelPendingAudit; its own semantics: "terminal residency is known; whether it filled before cancelling is not
     known until the trade audit" - match <= c_hi is NOT ORDERED by the answer), NO-EFFECT (Http / transport / deadline / another refusal) or
     NO-ANSWER (both: no bound). The post-cancel terminal order snapshot (CancelQuantityReconciled size_matched) and the order's filled
     size are printed beside it.
  A2 EVENT-IDENTITY per leg: the leg(s) whose window binds the event's lower end are found among the REST makers of the same transaction and
     settlement id as the leg: SAME-SETTLEMENT / DIFFERENT / AMBIGUOUS / ABSENT.
  A3 BINDING-ENDPOINT per leg: the event's S3 lower end recomputed from its legs (sem_windows on Nov 8's tables) equals the case table's; its
     clock domain (V_book: a WS pending increment's updateTimestampMs - WS_DELAY; V_status: a REST range); the binding stamp u*; the binding
     increment recomputed from the research frames at u* (raw line kept).
  A4 BOOK-CLOCK-BOUND per leg: pairs n, B (declared), B_max, B_min, statuses; the bot clock's step check (kernel at - mono spread within the
     window <= CLOCK_STEP_MS); the binding frame's own pair; the bot's receipt of the binding frame minus c_hi (one clock).
  A5 VERDICT per leg: CONTRADICTION-REMOVED-BY-INDEPENDENT-BOOK-CLOCK-BOUND iff A3 reproduces with domain V_book, A0 YES, A2
     SAME-SETTLEMENT, the bot clock did not step, n >= N_MIN and B >= the critical offset (then no admissible book-clock offset makes the
     window start after c_hi, so the known fill is not called avoided); else CONTRADICTION-RETAINED with the missing observation for each
     failed condition. A1 does not enter: a known fill was not avoided whatever the order of its match and the cancel. AUDIT VERDICT:
     REMOVED for every leg, else RETAINED (legs named). Either way S3's lower ends stay ASSUMPTIONS (WS_DELAY, REST_LAG): removing this
     contradiction makes S3 compatible with the known legs, not identified.
  A6 EVIDENCE-TIERS on the Oct 23 ledger v5 (saved tables; no tape read): per scenario (S1 / S2 / S3 / ST) and cancel window (CP / CR), the
     REMOVED fills and gross by class x tier: EVIDENCE (the evidence-only window EV certifies the same class), BEYOND-EVIDENCE (certified only
     under the scenario's ranges; the deciding endpoint's basis: for CANCEL_FIRST the binding lower end - S3 / ST fallback split into WS
     pending - WS_DELAY (V_book) and REST second - REST_LAG (V_status), mapped ST legs have none; for MATCH_FIRST the binding upper end -
     t_set (V_uuid, conditional), WS pending (V_book), REST second end (V_status)), UNCERTIFIED. EV never certifies CANCEL_FIRST (refused
     otherwise). REPRODUCES iff every class total equals the ledger receipt's cross-tab.
SELFTEST (synthetic, no data): the content key and the identity rule (unique, several either side, none), B and its order statistics, the
 verdict on planted legs (removed; B below critical; too few pairs; stepped clock; status-clock endpoint; wrong settlement; no-effect cancel),
 the binding endpoint on planted legs through sem_windows, the readers on planted files (kernel needles + outbox by seq, cutoff stops before a
 corrupt line, clock regression refused, open gzip stream, an hour outside Nov 8 refused), the cancel-answer classes, and A6 on a planted
 ledger (tiers, bases, the EV refusal, the receipt reproduction). Mutations: pf_timing_case_audit_mutants.py.
LIMITS: three legs of one consumed partial day; the book-clock premise (stamp before receipt, one stamp per update across subscribers) is stated,
 not proven; content identity can pair a repeated book state (bounded by the uniqueness rule and the declared order statistic); the bound is
 one-sided (no upper book-clock bound is derived - the gate's false CANCEL_FIRST binds at the lowest offset); WS_DELAY and REST_LAG remain
 assumptions; development evidence - Nov 8 is not untouched validation.
V2 (2026-11-18, after v1's output was seen; v1's receipt timing_case_audit_20261118 stays as provenance). v1 returned RETAINED for all three
 legs on ONE failed A0 field, one_bot_clock: the pfmm input journal's order id is unique only within a market (id 146 recurs in many markets)
 and v1 keyed its records by the id alone, so the leg's Accepted was never compared alone. Fixed: records are keyed by (market, id), with a
 planted same-id record in another market as the falsifier and a mutant. Also, without any reading depending on them: pf-exec's outbox
 alerts naming the order by its hex hash are retained as raw records, and two descriptive A1 fields are printed (the order filled after a
 Canceled answer; a 0-matched terminal snapshot received after the fill notice). Every declared reading, statistic, threshold and window is
 unchanged.
Usage (run_light, PF_SLOT=claude): pf_timing_case_audit.py selftest | run
"""
import bisect
import collections
import gzip
import hashlib
import json
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_exec_windows as X                     # noqa: E402  known v5, sem_windows, WS_DELAY / REST_LAG, the ledger's classes
import pf_settlement_times as SETT                        # noqa: E402  uuid7_ms
from predictfun.research import pf_io as io               # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'timing_case_audit_v2_20261118'      # v2: the input-journal key fix (v1, timing_case_audit_20261118, stays as provenance)
KX = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108-kernel')
TL = Path('/home/yuqing/pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109')
INPUTS = {'v6': TL / 'v6/run-1794108465172/inputs.jsonl.gz', 'basis': TL / 'basis/run-1794147344737/inputs.jsonl.gz'}
LEDGER = Path(I.K.IO) / 'keep_pull_exec_windows_v5_20261118_20261023'
DAY = '20261108'
CUTOFF_MS = 1794178920000.                                # Nov 8 23:02Z: no record at or after it is parsed (Nov 9 is protected)
W_LOCAL, W_ID, N_MIN, DP, CLOCK_STEP_MS = 10_000., 5_000., 20, 4, 1.
BOT_CLOCK_TOL_MS = 0.1                                    # one bot clock: the two journals' stamps of one event are separate reads of the host clock (v3)
CLOCKS = {'B': "bot host wall clock (Tokyo): kernel step `at` (ns) and the pfmm input journal `known_ms` - one clock (A0: the order's Accepted in both)",
          'B_mono': 'bot host monotonic clock: kernel `mono` (ns); used only to check that B did not step (at - mono spread)',
          'V_book': 'venue book stamp: predictOrderbook `updateTimestampMs` (ms); S3 WS windows and pend_inc t_ms are on it',
          'V_status': 'venue status seconds: pf-exec Fill `match_time` (rounded to the second by pf-exec), REST `executedAt` (the Confirmed second)',
          'V_uuid': 'venue settlement id: settlementId UUIDv7 creation ms',
          'R': 'research collector host clock: `recv_ns` of the research WS / REST rows; file selection only'}
ANSWER = ('MATCHED-OR-GONE', 'CANCELED', 'NO-EFFECT', 'NO-ANSWER')
PREMISE = {'MATCHED-OR-GONE': 'EVIDENCE', 'CANCELED': 'NOT-ORDERED-BY-THE-ANSWER', 'NO-EFFECT': 'NONE', 'NO-ANSWER': 'NONE'}
VERDICT_OK = 'CONTRADICTION-REMOVED-BY-INDEPENDENT-BOOK-CLOCK-BOUND'
STEP_RE = re.compile(rb'^\{"frame":\{"at":([0-9]+),')
MONO_RE = re.compile(rb'"mono":([0-9]+)')
SEQ_RE = re.compile(rb'"seq":([0-9]+)')
NOW_RE = re.compile(r'^\{"Advance":\{"now_ms":([0-9.]+)')
TOPIC_RE = re.compile(r'predictOrderbook/([0-9]+)')
MISSING = {'A0': "the kernel / input journal records reproducing the known case table's lifecycle (bot clock)",
           'A2': "the binding leg's REST maker record in the leg's own transaction / settlement id",
           'A3': 'a binding lower end on the venue book clock (a status-clock endpoint needs a ms status stamp or a status-clock offset bound)',
           'clock': 'a bot wall clock without a step across the window (kernel at - mono)',
           'pairs': f'>= {N_MIN} identity-matched bot receipts of venue book frames within +-{W_LOCAL / 1000:g} s of the cancel confirmation',
           'bound': "a tighter book-clock bound: the venue's updateTimestampMs recorded with the bot's own receipt of each frame"}


def hour_of(ms):
    """UTC hour file name of a ms instant (YYYYMMDD_HH)."""
    return datetime.fromtimestamp(ms / 1000., tz=timezone.utc).strftime('%Y%m%d_%H')


def hours_between(lo_ms, hi_ms, day=DAY):
    """Hour files covering [lo, hi] (ms); every one must be on `day` (no padding into another day)."""
    h = sorted({hour_of(t) for t in np.arange(lo_ms, hi_ms, 3_600_000.)} | {hour_of(hi_ms)})
    C.require(all(x.startswith(day + '_') for x in h), f'an hour outside {day}: {h}')
    return h


def gz_lines(p):
    """Complete lines of a gzip file; an open stream (no end marker) ends at its last complete line."""
    with gzip.open(p, 'rt') as f:
        try:
            for line in f:
                if line.endswith('\n'):
                    yield line
        except EOFError:
            return


def book_key(bids, asks, pend):
    """Content of a book frame: five best levels per side (tick, size to DP) and the pending amounts per (side, tick) (to DP)."""
    return (tuple((int(p), round(float(s), DP)) for p, s in bids[:5]), tuple((int(p), round(float(s), DP)) for p, s in asks[:5]),
            tuple(sorted((int(a), int(b), round(float(v), DP)) for (a, b), v in pend.items() if round(float(v), DP) != 0.)))


def key_bot(fr):
    """A pfmm input Frame (ticks, top five, pending [side, tick, amount]) -> book_key."""
    pend = collections.defaultdict(float)
    for a, b, v in fr.get('pending') or []:
        pend[(int(a), int(b))] += float(v)
    return book_key(fr['bids'], fr['asks'], pend)


def pending_of(d):
    """settlementsPending of a predictOrderbook message -> {(side 0 bids / 1 asks, tick): amount} (pf_step1's aggregation)."""
    sp = d.get('settlementsPending') or {}; cur = collections.defaultdict(float)
    for si, k in ((0, 'bids'), (1, 'asks')):
        for p, v in sp.get(k) or []:
            cur[(si, int(round(float(p) * 100)))] += float(v)
    return dict(cur)


def key_ws(d):
    """A predictOrderbook message (prices, full depth) -> book_key."""
    b = sorted(d.get('bids') or [], key=lambda z: -z[0])[:5]; a = sorted(d.get('asks') or [], key=lambda z: z[0])[:5]
    return book_key([(round(p * 100), s) for p, s in b], [(round(p * 100), s) for p, s in a], pending_of(d))


def pairs(tb, kb, tu, ku):
    """Identity pairs of bot frames (receipt tb, key kb) with research frames (stamp tu, key ku): one research twin within +-W_ID of the bot
    receipt AND one bot twin within +-W_ID of that research frame. -> (DataFrame ib, iu, t_bot, u, x = u - t_bot; status counts per bot frame)."""
    gb, gu = collections.defaultdict(list), collections.defaultdict(list)
    for i, k in enumerate(kb):
        gb[k].append(i)
    for j, k in enumerate(ku):
        gu[k].append(j)
    out, st = [], collections.Counter()
    for k, ib in gb.items():
        ju = gu.get(k, [])
        for i in ib:
            c = [j for j in ju if abs(tu[j] - tb[i]) <= W_ID]
            if not c:
                st['none'] += 1; continue
            if len(c) > 1:
                st['several_research'] += 1; continue
            j = c[0]; back = [i2 for i2 in ib if abs(tu[j] - tb[i2]) <= W_ID]
            if len(back) != 1:
                st['several_bot'] += 1; continue
            st['paired'] += 1; out.append((i, j, tb[i], tu[j], tu[j] - tb[i]))
    return pd.DataFrame(out, columns=['ib', 'iu', 't_bot', 'u', 'x']), dict(st)


def bound(x):
    """The declared book-clock lower bound: B = the ceil(n/2)-th largest x (valid if one of those pairs is a correct identity); with n, B_max,
    B_min. Empty -> B None."""
    x = np.sort(np.asarray(x, float))[::-1]; n = len(x)
    if n == 0:
        return dict(n=0, B=None, B_max=None, B_min=None)
    return dict(n=int(n), B=float(x[int(np.ceil(n / 2.)) - 1]), B_max=float(x[0]), B_min=float(x[-1]))


def answer_class(a, pending_audit=False):
    """The venue's first answer to a cancel: the kernel's CancelAnswer `answer` (None if there was none) and whether the first confirmation was
    journaled as CancelPendingAudit (pf-exec's form of a Canceled answer or a user-stream cancel) -> ANSWER class."""
    if a is None:
        return 'CANCELED' if pending_audit else 'NO-ANSWER'
    if isinstance(a, str):
        return 'CANCELED' if a == 'Canceled' else 'NO-EFFECT'
    if isinstance(a, dict) and 'NotCanceled' in a:
        t = str(a['NotCanceled']).lower()
        return 'MATCHED-OR-GONE' if ("can't be found" in t and 'matched' in t) else 'NO-EFFECT'
    return 'NO-EFFECT'


def verdict(a0, a2, a3_ok, domain, step_ms, b, crit):
    """A5 for one leg -> (verdict, missing observations)."""
    miss = []
    if not a0:
        miss.append(MISSING['A0'])
    if a2 != 'SAME-SETTLEMENT':
        miss.append(MISSING['A2'])
    if not a3_ok or domain != 'V_book':
        miss.append(MISSING['A3'])
    if step_ms is None or not np.isfinite(step_ms) or step_ms > CLOCK_STEP_MS:
        miss.append(MISSING['clock'])
    if b['n'] < N_MIN:
        miss.append(MISSING['pairs'])
    elif not b['B'] >= crit:
        miss.append(MISSING['bound'])
    return (VERDICT_OK if not miss else 'CONTRADICTION-RETAINED'), miss


# ------------------------------------------------------------------ readers (bounded at CUTOFF_MS)
def kernel_records(path, legs, wins):
    """One run's kernel journal up to CUTOFF_MS. legs: {key: dict(cid, oid_json, trade_id)}; wins: {key: (lo_ms, hi_ms)} bot-clock windows for
    the step check. -> ({key: [(at_ms, kind, verbatim line)]}, {key: [(at_ns, mono_ns)]}, stats). A step frame is kept when it carries the
    leg's cid, order id bytes or trade id; an outbox frame when it carries them or follows a kept step (same seq)."""
    need = {k: [b'"cid":%d,' % v['cid'], b'"cid":%d}' % v['cid'], v['oid_json'].encode(), v['trade_id'].encode()] + ([v['hex'].encode()] if v.get('hex') else [])
            for k, v in legs.items()}
    rx = re.compile(b'|'.join(re.escape(x) for nd in need.values() for x in nd)) if need else None   # a gate only: hits are re-checked per leg below
    wl = sorted((lo, hi, k) for k, (lo, hi) in wins.items()); wlo = [w[0] for w in wl]; span = max((hi - lo for lo, hi, _ in wl), default=0.)
    rec, clk = collections.defaultdict(list), collections.defaultdict(list); last = 0; kept_seq = {}; n = 0; stopped = False
    with open(path, 'rb') as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                at = int(m[1])
                if at >= CUTOFF_MS * 1e6:
                    stopped = True; break
                C.require(at >= last, 'kernel UTC clock regressed'); last = at; n += 1
                t_ = at / 1e6
                for j in range(bisect.bisect_left(wlo, t_ - span), bisect.bisect_right(wlo, t_)):
                    lo, hi, k = wl[j]
                    if lo <= t_ <= hi:
                        mo = MONO_RE.search(line); clk[k].append((at, int(mo[1]) if mo else -1))
                for k, nd in (need.items() if rx is not None and rx.search(line) else ()):
                    if any(x in line for x in nd):
                        fr = json.loads(line)['frame']; inp = fr.get('input')
                        kind = next(iter(inp)) + '/' + next(iter(inp[next(iter(inp))])) if isinstance(inp, dict) and isinstance(inp[next(iter(inp))], dict) else str(inp)
                        rec[k].append((at / 1e6, 'step:' + kind, line.decode())); kept_seq[fr['seq']] = (k, at / 1e6)
            elif b'"type":"outbox"' in line[-60:]:
                s = SEQ_RE.search(line); s = int(s[1]) if s else -1
                hit = [k for k, nd in need.items() if any(x in line for x in nd)] if rx is not None and rx.search(line) else []
                if s in kept_seq and kept_seq[s][0] not in hit:
                    hit.append(kept_seq[s][0])
                for k in hit:
                    rec[k].append((kept_seq.get(s, (k, np.nan))[1], 'outbox', line.decode()))
    return dict(rec), dict(clk), dict(steps=n, stopped_at_cutoff=stopped, last_at_ms=last / 1e6)


def input_frames(path, wins, ids):
    """One run's pfmm input journal up to CUTOFF_MS. wins: {mid: [(lo, hi)]} bot-clock ranges for book frames; ids: (market, engine order id)
    pairs - the engine's id is unique only within a market (v2) - whose Accepted / Answered / CancelDone / Filled records are kept verbatim.
    -> (frames {mid: [(known_ms, key)]}, records {(market, id): [line]}, stats)."""
    fr, recs = collections.defaultdict(list), collections.defaultdict(list); pre = {m: '{"Frame":{"market":%d,' % m for m in wins}; stopped = False; n = 0
    idn = {(m, i): ('"market":%d,' % m, '"id":%d}' % i) for m, i in ids}
    for line in gz_lines(path):
        n += 1
        if line.startswith('{"Advance"'):
            m = NOW_RE.match(line)
            if m and float(m[1]) >= CUTOFF_MS:
                stopped = True; break
            continue
        if line.startswith('{"Frame":'):
            j = line.find(',', 19) if line.startswith('{"Frame":{"market":') else -1
            mid = int(line[19:j]) if j > 19 and line[19:j].isdigit() else None
            if mid in pre and line.startswith(pre[mid]):
                f = json.loads(line)['Frame']; t = float(f['known_ms'])
                if t >= CUTOFF_MS:
                    stopped = True; break
                if any(lo <= t <= hi for lo, hi in wins[mid]):
                    fr[mid].append((t, key_bot(f)))
            continue
        if line.startswith(('{"Accepted"', '{"Answered"', '{"CancelDone"', '{"Filled"')):
            for i, (mk, s) in idn.items():
                if mk in line and (line.rstrip().endswith(s + '}') or (s[:-1] + ',') in line):
                    recs[i].append(line.rstrip('\n'))
    return dict(fr), dict(recs), dict(lines=n, stopped_at_cutoff=stopped)


def research_frames(hours, wins, keep_u=None, root=None):
    """The research collector's predictOrderbook frames in Nov 8 `hours` for wins {mid: [(lo, hi)]} (venue ms). keep_u {mid: set of stamps}:
    those frames' raw lines are kept verbatim. -> ({mid: DataFrame u, recv_ns, key, pend (file order)}, {mid: [(u, line)]}, hashes)."""
    root = Path(root or io.PF_RUN) / 'predict_ws'; keep_u = keep_u or {}; out, raw, hs = collections.defaultdict(list), collections.defaultdict(list), {}
    for h in hours:
        C.require(h.startswith(DAY + '_'), f'an hour outside {DAY}: {h}')
        p = root / f'{h}.jsonl.gz'; hs[str(p)] = C.sha(p)
        for line in gz_lines(p):
            if 'predictOrderbook/' not in line:
                continue
            m = TOPIC_RE.search(line)
            if not m or int(m[1]) not in wins:
                continue
            r = json.loads(line); msg = json.loads(r['raw'])
            d = msg.get('data')
            if msg.get('type') != 'M' or not isinstance(d, dict) or 'updateTimestampMs' not in d:
                continue
            mid, u = int(d['marketId']), float(d['updateTimestampMs'])
            if mid in wins and any(lo <= u <= hi for lo, hi in wins[mid]):
                out[mid].append((u, int(r['recv_ns']), key_ws(d), pending_of(d)))
                if u in keep_u.get(mid, ()):
                    raw[mid].append((u, line.rstrip('\n')))
    return {m: pd.DataFrame(v, columns=['u', 'recv_ns', 'key', 'pend']) for m, v in out.items()}, dict(raw), hs


def rest_elements(hours, keep, root=None):
    """REST /v1/orders/matches elements in Nov 8 `hours` whose market and executedAt second are in keep {(mid, exec_s)}: (recv_ns, page line
    sha256, element) - the raw identity of every match in those seconds."""
    out, hs = [], {}
    for h in hours:
        C.require(h.startswith(DAY + '_'), f'an hour outside {DAY}: {h}')
        p = Path(root or io.PF_RUN) / 'rest' / f'{h}.jsonl.gz'; hs[str(p)] = C.sha(p)
        for line in gz_lines(p):
            j = line.find('"path":"', 0, 400)
            if j < 0 or not line.startswith('/v1/orders/matches', j + 8):
                continue
            r = json.loads(line)
            if r.get('status') != 200:
                continue
            try:
                payload = json.loads(r['raw'])
            except (TypeError, ValueError):
                continue
            lh = hashlib.sha256(line.encode()).hexdigest()
            for x in payload.get('data') or []:
                mid = (x.get('market') or {}).get('id')
                if (mid, int(io.iso_s(x['executedAt']))) in keep:
                    out.append((int(r['recv_ns']), lh, x))
    return out, hs


def maker_rows(elems):
    """Every maker of the REST elements as a row: content (mid, exec_s, side, ptick, shares, price_executed, taker_index, taker_quote), its tx,
    settlement id and order hash (pf_fill_clock's mapping)."""
    rows = []
    for recv, lh, x in elems:
        t = x['taker']
        for mk in x['makers']:
            s, py = io.yes_side(mk)
            rows.append(dict(mid=x['market']['id'], exec_s=int(io.iso_s(x['executedAt'])), side=0 if s == 'bid' else 1, ptick=int(round(py * 100)),
                             shares=int(mk['amount']) / io.E18, price_executed=int(x['priceExecuted']) / io.E18, taker_index=t['outcome']['indexSet'],
                             taker_quote=0 if t['quoteType'] == 'Bid' else 1, tx=x['transactionHash'].lower(), sid=x.get('settlementId') or '',
                             order_hash=mk['hash'].lower(), recv_ns=recv, line_sha=lh))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the case audit
def known_legs():
    """The legs known v5 marks S3_false_cf_some_offset, with their candidate rows' event windows and critical offsets (hash-checked)."""
    j = json.loads((X.OUT_K / 'known.json').read_text())
    for f in ('known_cases.parquet', 'known_case_rows.parquet'):
        C.require(C.sha(X.OUT_K / f) == j['outputs'][f], f'{f} differs from its known v5 receipt')
    K = pd.read_parquet(X.OUT_K / 'known_cases.parquet'); KE = pd.read_parquet(X.OUT_K / 'known_case_rows.parquet')
    sel = np.flatnonzero(K.S3_false_cf_some_offset.values.astype(bool))
    rows = []
    for i in sel:
        r = K.iloc[i]; e = KE[KE.i == i]
        lo = float(e.S3_lo.min()); rows.append(dict(i=int(i), **{c: r[c] for c in ('run', 'cid', 'mid', 'order_hash', 'trade_id', 'tx', 'place', 'signed', 'accepted', 'upper',
                                                                                  'notice', 'ack', 'req', 'cdone', 'ucancel', 'rM', 'rC', 'M', 'Cs', 'S', 't_set', 'exec_s', 'first_recv_ns',
                                                                                  'side', 'ptick', 'shares')},
                                                group=int(e.group.iloc[0]), groups=sorted(set(e.group.astype(int))), S3_lo=lo, S3_hi=float(e.S3_hi.min()),
                                                critical=lo - float(r['ack'])))
    return pd.DataFrame(rows), j['R']['K7'], {str(X.OUT_K / 'known.json'): C.sha(X.OUT_K / 'known.json')}


def event_binding(group, L8, P8):
    """A3: the event's legs (legs_timed2 rows of taker order `group`) with their S3 windows (sem_windows on Nov 8's increments), the event's
    lower end, the binding legs, their clock domain and the binding stamp u*."""
    ev = L8.tk.values == group; C.require(ev.any(), f'event {group} without legs')
    cell = np.isin(L8.mid.values, np.unique(L8.mid.values[ev])) & np.isin(L8.exec_s.values, np.unique(L8.exec_s.values[ev]))   # a cell link sums its whole cell
    A_ = L8[cell].reset_index(drop=True); A_ = pd.concat([A_, X.sem_windows(A_, P8)], axis=1); R = A_[A_.tk.values == group].reset_index(drop=True)
    lo0 = X.rest_range(R.exec_s.values)[0]; lo = float(R.S3_llo.max()); b = R[R.S3_llo.values == lo]
    ws = b.S3_code.isin([0, 1]).values & (b.S3_llo.values > lo0[R.S3_llo.values == lo])
    dom = 'V_book' if ws.any() else 'V_status'
    return R, lo, b[ws] if ws.any() else b, dom, (lo + X.WS_DELAY) if dom == 'V_book' else np.nan


def binding_increment(F, row, u):
    """The binding leg's increment at stamp u recomputed from the research frames F of its market (file order, pf_step1's rule): the positive
    pending change at the leg's (side either, tick) in the frame(s) stamped u. -> list of (side, tick, increment)."""
    prev, inc = {}, []
    for uu, pend in zip(F.u.values, F.pend.values):
        if uu == u:
            for k, v in pend.items():
                d = v - prev.get(k, 0.)
                if d > 1e-9 and k[1] == int(row.ptick):
                    inc.append((int(k[0]), int(k[1]), float(d)))
        prev = pend
    return inc


def a6(F, FE, E, LG, xtab):
    """A6 on a ledger (F fills, FE fill -> event, E events, LG legs, xtab the receipt's cross-tab) -> (rows, reproduces)."""
    key = pd.MultiIndex.from_frame(E[['day', 'group']]); fe = FE.assign(k=key.get_indexer(pd.MultiIndex.from_frame(FE[['day', 'group']])))
    C.require(bool((fe.k >= 0).all()), 'a fill event without its event row')
    lg = LG.assign(k=key.get_indexer(pd.MultiIndex.from_frame(LG[['day', 'group']]))); lg = lg[lg.k >= 0]
    lo0 = X.rest_range(lg.exec_s.values)[0]
    s3dom = np.where(lg.S3_code.isin([0, 1]).values & (lg.S3_llo.values > lo0), 'WS pending - WS_DELAY (V_book)', 'REST second - REST_LAG (V_status)')
    lg = lg.assign(s3dom=s3dom)
    top = lambda col: lg.assign(v=lg[col].values).sort_values(['k', 'v'], ascending=[True, False]).drop_duplicates('k')
    b_s3 = top('S3_llo').set_index('k').s3dom
    st = lg.assign(v=np.where(np.isfinite(lg.ST_llo.values), lg.ST_llo.values, -np.inf)).sort_values(['k', 'v'], ascending=[True, False]).drop_duplicates('k').set_index('k')
    b_st = pd.Series(np.where(np.isneginf(st.v.values), 'none (mapped UUID: no lower end)', 'S3 fallback: ' + st.s3dom.values), index=st.index)
    hi = lambda col, code_col: lg.assign(v=lg[col].values).sort_values(['k', 'v']).drop_duplicates('k').set_index('k')
    hs3 = hi('S3_lhi', 'S3_code'); h_s3 = pd.Series(np.where(hs3.S3_code.isin([0, 1]).values, 'WS pending (V_book)', 'REST second end (V_status)'), index=hs3.index)
    hst = hi('ST_lhi', 'ST_code'); h_st = pd.Series(np.where(hst.ST_code.values == 0, 't_set (V_uuid, conditional)', 'S3 fallback end'), index=hst.index)
    rows = []; ok = True; rem = F.replay_removed.values.astype(bool); gross = np.abs(F.value.values.astype(float))
    for k in ('S1', 'S2', 'S3', 'ST'):
        for cw in ('CP', 'CR'):
            c = F[f'cls_{k}_{cw}'].values.astype(int); ev = F[f'cls_EV_{cw}'].values.astype(int)
            C.require(not (ev == 0).any(), 'EV (evidence only) certified a CANCEL_FIRST')
            for o in range(len(X.ORD)):
                m = rem & (c == o); x = xtab.get(f'{k}|{cw}', {}).get('REMOVED', {}).get(X.ORD[o], {})
                ok &= int(m.sum()) == int(x.get('fills', 0)) and abs(float(gross[m].sum()) - float(x.get('gross', 0.))) < 1e-6 * max(1., float(x.get('gross', 0.)))
                if not m.any():
                    continue
                n0 = len(rows)
                if o > 1:
                    rows.append(dict(scenario=k, cancel=cw, cls=X.ORD[o], tier='UNCERTIFIED', basis='', fills=int(m.sum()), gross=float(gross[m].sum()))); continue
                evd = m & (ev == o)
                if evd.any():
                    rows.append(dict(scenario=k, cancel=cw, cls=X.ORD[o], tier='EVIDENCE', basis='EV: REST second end (V_status)', fills=int(evd.sum()), gross=float(gross[evd].sum())))
                bey = np.flatnonzero(m & (ev != o))
                if not len(bey):
                    C.require(sum(x['fills'] for x in rows[n0:]) == int(m.sum()), f'{k}|{cw} {X.ORD[o]}: the tiers do not partition the class'); continue
                fid = F.fid.values[bey]; ks = fe[fe.fid.isin(fid)]
                if o == 0:
                    src = (b_s3 if k == 'S3' else b_st if k == 'ST' else E[f'{k}_lo_basis'].reindex(range(len(E))))
                else:
                    src = (h_s3 if k == 'S3' else h_st if k == 'ST' else E[f'{k}_hi_basis'].reindex(range(len(E))))
                bs = ks.assign(b=src.reindex(ks.k.values).values).groupby('fid').b.agg(lambda s: s.iloc[0] if s.nunique() == 1 else 'mixed over candidates')
                g = pd.DataFrame(dict(fid=fid, gross=gross[bey])).assign(b=lambda d: bs.reindex(d.fid.values).values)
                for b_, gg in g.groupby('b', dropna=False):
                    rows.append(dict(scenario=k, cancel=cw, cls=X.ORD[o], tier='BEYOND-EVIDENCE', basis=str(b_), fills=int(len(gg)), gross=float(gg.gross.sum())))
                C.require(sum(x['fills'] for x in rows[n0:]) == int(m.sum()), f'{k}|{cw} {X.ORD[o]}: the tiers do not partition the class')
    return pd.DataFrame(rows), bool(ok)


def run(out=OUT):
    """The audit on consumed Nov 8 (and the saved Oct 23 ledger for A6)."""
    t0 = time.time(); out = Path(out); C.require(not (out / 'audit.json').exists(), f'{out} exists: a new version needs a new directory')
    out.mkdir(parents=True, exist_ok=True)
    L, K7, hs = known_legs(); C.require(len(L) > 0, 'no leg to audit'); inputs = dict(hs)
    # Nov 8's tables (pinned by the known ledger's inputs)
    kj = json.loads((X.OUT_K / 'known.json').read_text())['inputs']
    for f in ('legs_timed2.parquet', 'legs_tk.parquet', 'pend_inc.parquet'):
        p = X.NOV8 / f; C.require(kj.get(str(p)) == C.sha(p), f'{f} differs from the known ledger inputs'); inputs[str(p)] = kj[str(p)]
    L8 = pd.read_parquet(X.NOV8 / 'legs_timed2.parquet', columns=X.CKEY + ['how', 't_ms', 't_e2']).assign(tk=pd.read_parquet(X.NOV8 / 'legs_tk.parquet', columns=['tk']).tk.values)
    P8 = pd.read_parquet(X.NOV8 / 'pend_inc.parquet', columns=['mid', 't_ms', 'ptick', 'inc'])
    core = {r: pd.read_parquet(X.LIVE / f'core_orders_{r}.parquet', columns=['cid', 'id', 'size', 'filled']) for r in ('v6', 'basis')}
    for r in core:
        inputs[str(X.LIVE / f'core_orders_{r}.parquet')] = C.sha(X.LIVE / f'core_orders_{r}.parquet')
    legs = {}
    for _, r in L.iterrows():
        R_, lo, B_, dom, us = event_binding(r.group, L8, P8)
        eng = core[r.run]; j_ = np.flatnonzero(eng.cid.values == r.cid); C.require(len(j_) == 1, f'cid {r.cid} not once in core_orders_{r.run}'); j_ = int(j_[0])
        legs[int(r.i)] = dict(row=r, ev=R_, lo=lo, bind=B_, dom=dom, u=us, eid=int(eng.id.values[j_]), size=float(eng['size'].values[j_]), filled=float(eng.filled.values[j_]),
                              oid_json='"order_id":[' + ','.join(str(b) for b in bytes.fromhex(r.order_hash[2:])) + ']')
    # kernel journals
    K_rec, K_clk, kst = {}, {}, {}
    for run_ in sorted(set(L.run)):
        ks = {i: dict(cid=int(v['row'].cid), oid_json=v['oid_json'], trade_id=v['row'].trade_id, hex=v['row'].order_hash.lower()) for i, v in legs.items() if v['row'].run == run_}
        ws = {i: (float(v['row'].ack) - W_LOCAL, float(v['row'].ack) + W_LOCAL) for i, v in legs.items() if v['row'].run == run_}
        p = KX / f'{run_}.kernel.jsonl'; inputs[str(p)] = C.sha(p)
        a, b, c = kernel_records(p, ks, ws); K_rec.update(a); K_clk.update(b); kst[run_] = c
    # pfmm input journals: book frames around each leg's cancel confirmation, and the order's own records
    BF, BR, ist = {}, {}, {}
    for run_ in sorted(set(L.run)):
        wins = collections.defaultdict(list)
        for i, v in legs.items():
            if v['row'].run == run_:
                a = float(v['row'].ack); wins[int(v['row'].mid)].append((a - W_LOCAL - 2 * W_ID, a + W_LOCAL + 2 * W_ID))
        p = INPUTS[run_]; inputs[str(p)] = C.sha(p)
        a, b, c = input_frames(p, dict(wins), [(int(v['row'].mid), v['eid']) for v in legs.values() if v['row'].run == run_]); BF[run_] = a; BR[run_] = b; ist[run_] = c
    # research WS frames (venue ms) and REST match elements
    wins, keep_u, hrs = collections.defaultdict(list), collections.defaultdict(set), set()
    for i, v in legs.items():
        a = float(v['row'].ack); lo_, hi_ = a - W_LOCAL - 3 * W_ID, a + W_LOCAL + 3 * W_ID; wins[int(v['row'].mid)].append((lo_, hi_))
        hrs |= set(hours_between(lo_ - 60_000., hi_ + 60_000.))
        if np.isfinite(v['u']):
            keep_u[int(v['row'].mid)].add(float(v['u']))
    RF, RAW, h1 = research_frames(sorted(hrs), dict(wins), dict(keep_u)); inputs.update(h1)
    want = {(int(v['row'].mid), int(v['row'].exec_s)) for v in legs.values()} | {(int(m), int(s)) for v in legs.values() for m, s in zip(v['bind'].mid, v['bind'].exec_s)}
    rh = sorted({hour_of(float(v['row'].first_recv_ns) / 1e6) for v in legs.values()} | {hour_of(float(x) / 1e6) for v in legs.values() for x in v['bind'].first_recv_ns})
    EL, h2 = rest_elements(rh, want); inputs.update(h2); MR = maker_rows(EL)
    # per leg
    res, raw_out = [], []
    for i, v in legs.items():
        r = v['row']; rec = K_rec.get(i, []); br = BR[r.run].get((int(r.mid), v['eid']), []); Fm = RF.get(int(r.mid), pd.DataFrame(columns=['u', 'recv_ns', 'key', 'pend']))
        x_, ids = leg_eval(i, v, rec, br, Fm, BF[r.run].get(int(r.mid), []), MR, np.array(K_clk.get(i, []), float), K7); res.append(x_)
        for t, k, line in sorted(rec, key=lambda z: (z[0], z[1] != 'outbox')):
            raw_out.append(dict(leg=i, source='kernel', kind=k, at_ms=t, clock='B', line=line))
        for s_ in br:
            raw_out.append(dict(leg=i, source='pfmm_inputs', kind=next(iter(json.loads(s_))), at_ms=next(iter(json.loads(s_).values()))['known_ms'], clock='B', line=s_))
        for u, line in RAW.get(int(r.mid), []):
            if u == v['u']:
                raw_out.append(dict(leg=i, source='research_ws', kind='predictOrderbook', at_ms=u, clock='V_book', line=line))
        for recv, lh, x in EL:
            if x['transactionHash'].lower() == r.tx.lower() or any(t_ == x['transactionHash'].lower() for t_, _ in ids):
                raw_out.append(dict(leg=i, source='research_rest', kind='/v1/orders/matches element', at_ms=float(io.iso_s(x['executedAt'])) * 1000., clock='V_status',
                                    page_line_sha256=lh, recv_ns=recv, element=x, uuid7_ms=SETT.uuid7_ms(x.get('settlementId') or '')))
    A6, a6_ok, a6_in = a6_ledger()
    inputs.update(a6_in)
    allok = all(x['A5'] == VERDICT_OK for x in res)
    read = {f"A5 LEG {x['leg']} ({x['run']} cid {x['cid']})": x['A5'] for x in res}
    read['AUDIT VERDICT'] = ('REMOVED: every audited S3 false avoidance needs a book-clock offset below the independent bound' if allok else
                             'RETAINED for legs ' + ', '.join(str(x['leg']) for x in res if x['A5'] != VERDICT_OK))
    read['A6 REPRODUCES THE LEDGER CROSS-TAB'] = 'YES' if a6_ok else 'NO'
    R = dict(asof=C.stamp(), legs=res, clocks=CLOCKS, readers=dict(kernel=kst, inputs=ist), cutoff_ms=CUTOFF_MS, params=dict(W_LOCAL=W_LOCAL, W_ID=W_ID, N_MIN=N_MIN, DP=DP, CLOCK_STEP_MS=CLOCK_STEP_MS,
             WS_DELAY=X.WS_DELAY, REST_LAG=X.REST_LAG), A6=A6.to_dict('records'), readings=read, inputs=inputs, source_sha256=C.sha(Path(__file__)),
             seconds=time.time() - t0)
    with open(out / 'raw_records.jsonl', 'w') as f:
        for x in raw_out:
            f.write(json.dumps(x, allow_nan=False) + '\n')
    A6.to_csv(out / 'evidence_tiers.csv', index=False); C.dump(out / 'audit.json', R); (out / 'audit.txt').write_text(report(R))
    print(report(R)); return R


def leg_eval(i, v, rec, br, Fm, bf, MR, ck, K7):
    """A0-A5 for one leg from its loaded evidence: v (row = the known case row, lo / bind / dom / u = event_binding, eid / size / filled = core),
    rec its kernel records [(at_ms, kind, line)], br its input-journal records, Fm its market's research frames, bf its market's bot frames
    [(known_ms, key)], MR the REST maker rows, ck the kernel (at_ns, mono_ns) samples around c_hi, K7. -> (result, binding settlement ids)."""
    r = v['row']; L_ = lifecycle(r, rec, br); rec = L_['rec']; a0, a0f, a1, a1_ans, a1_at, a1_src = (L_[k] for k in ('a0', 'a0f', 'a1', 'a1_ans', 'a1_at', 'a1_src'))
    uc, cq, pos, k_cancel = L_['uc'], L_['cq'], L_['pos'], L_['k_cancel']
    # A2: the binding legs' REST makers, by content, against the leg's own transaction / settlement
    own = MR[MR.order_hash == r.order_hash.lower()] if len(MR) else MR
    own_ids = sorted(set(zip(own.tx, own.sid))) if len(own) else []
    bk = ['mid', 'exec_s', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']; ids = set()
    for _, b in v['bind'].iterrows():
        m = MR[np.logical_and.reduce([np.isclose(MR[c].values.astype(float), float(b[c])) for c in bk])] if len(MR) else MR
        ids |= set(zip(m.tx, m.sid)) if len(m) else {('', '')}
    a2 = ('ABSENT' if not own_ids or not ids or ('', '') in ids else 'AMBIGUOUS' if len(ids) > 1 or len(own_ids) > 1
          else 'SAME-SETTLEMENT' if ids == set(own_ids) else 'DIFFERENT')
    # A3
    a3_ok = bool(np.isclose(v['lo'], r.S3_lo)) and len(r.groups) == 1
    inc = [binding_increment(Fm, b, v['u']) for _, b in v['bind'].iterrows()] if v['dom'] == 'V_book' and len(Fm) else []
    # A4
    P_, st, b, step = book_pairs(float(r.ack), Fm, bf, ck)
    bp = P_[P_.u.values == v['u']] if len(P_) and np.isfinite(v['u']) else P_.iloc[:0]
    vd, miss = verdict(a0, a2, a3_ok, v['dom'], step, b, float(r.critical))
    return dict(leg=i, run=r.run, cid=int(r.cid), mid=int(r.mid), order_hash=r.order_hash, trade_id=r.trade_id, tx=r.tx, group=int(r.group),
                critical_ms=float(r.critical), S3_lo=float(r.S3_lo), c_hi=float(r.ack), cancel_decision_minus_req=float(k_cancel - r.req), A0=a0, A0_fields=a0f,
                A1=a1, A1_answer=a1_ans, A1_answer_at=a1_at, A1_source=a1_src, A1_premise=PREMISE[a1], A1_user_cancel_at=[t for t, _ in uc],
                A1_terminal_snapshot=[dict(at=t, size_matched=x['size_matched']) for t, x in cq], order_size=v['size'], order_filled=v['filled'],
                A1_filled_after_canceled_answer=bool(a1 == 'CANCELED' and v['filled'] > 0),
                A1_zero_snapshot_after_fill_notice=bool(pos) and any(x['size_matched'] == 0 and t >= min(pos) for t, x in cq),
                A1_alerts=[json.loads(line)['frame']['alerts'] for t, k, line in rec if k == 'outbox' and r.order_hash.lower() in line],
                A2=a2, A2_binding_ids=sorted(ids), A2_own_ids=own_ids, A3_reproduces=a3_ok, A3_domain=v['dom'], A3_u=v['u'], A3_binding_legs=int(len(v['bind'])),
                A3_event_legs=int(len(v['ev'])), A3_increments=inc,
                A4=dict(**b, statuses=st, local_window_ms=W_LOCAL, clock_step_ms=step, clock_samples=int(len(ck)),
                        binding_pair_x=[float(x) for x in bp.x.values], binding_receipt_minus_c_hi=[float(t - float(r.ack)) for t in bp.t_bot.values]),
                A5=vd, A5_missing=miss, K7=K7, bounds=leg_bounds(r, a1, a2, v, bp)), ids


def lifecycle(r, rec, br):
    """A0 / A1 of one leg (r: its known case row) from its kernel records rec [(at_ms, kind, line)] and its input-journal records br -> dict:
    the sorted records and steps, a0 / a0f (the case table reproduced, one bot clock), a1 (the first cancel confirmation's class, answer, time,
    source), uc / cq (user-stream cancels, post-cancel terminal snapshots), pos (positive fill notices), k_cancel (the cancel decision)."""
    rec = sorted(rec, key=lambda z: (z[0], z[1] != 'outbox'))
    steps = [(t, k, json.loads(line)['frame']['input']) for t, k, line in rec if k != 'outbox']

    def first(kind, pred=lambda x: True):
        c = [t for t, k, x in steps if k == kind and pred(x)]; return min(c) if c else np.nan
    oid = list(bytes.fromhex(r.order_hash[2:]))             # a settlement can fill two own orders: its fills are the order's own (v3)
    fills = [(t, x['Exec']['Fill']) for t, k, x in steps if k == 'step:Exec/Fill' and x['Exec']['Fill']['trade_id'] == r.trade_id and x['Exec']['Fill']['order_id'] == oid]
    pos = [t for t, f in fills if f['status'] in ('Matched', 'Confirmed')]
    mt = {s: [f['match_time'] / 1e9 for t, f in fills if f['status'] == s] for s in ('Matched', 'Confirmed')}
    k_place = first('step:Control/Place', lambda x: x['Control']['Place']['cid'] == r.cid); k_signed = first('step:Exec/Signed', lambda x: x['Exec']['Signed']['cid'] == r.cid)
    k_acc = first('step:Exec/Accepted', lambda x: x['Exec']['Accepted']['cid'] == r.cid); k_cancel = first('step:Control/Cancel', lambda x: x['Control']['Cancel']['cid'] == r.cid)
    ans = [(t, x['Exec']['CancelAnswer']['answer']) for t, k, x in steps if k == 'step:Exec/CancelAnswer' and x['Exec']['CancelAnswer']['cid'] == r.cid]
    of = lambda k_: [(t, x['Exec'][k_]) for t, k, x in steps if k == f'step:Exec/{k_}' and x['Exec'][k_]['order_id'] == oid]
    pa, uc, cq = of('CancelPendingAudit'), of('UserCancel'), of('CancelQuantityReconciled')
    conf = sorted([(t, 'answer', a) for t, a in ans] + [(t, 'pending_audit', None) for t, _ in pa], key=lambda z: z[0])
    sig_hash = [('0x' + bytes(x['Exec']['Signed']['order_id']).hex()) for t, k, x in steps if k == 'step:Exec/Signed' and x['Exec']['Signed']['cid'] == r.cid]
    acc_in = [json.loads(s)['Accepted']['known_ms'] for s in br if s.startswith('{"Accepted"')]
    near = lambda a, b: bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-3)
    a1_ans = conf[0][2] if conf else None; a1 = answer_class(a1_ans, bool(conf) and conf[0][1] == 'pending_audit'); a1_at = conf[0][0] if conf else np.nan
    a1_src = ('' if not conf else 'cancel answer' if conf[0][1] == 'answer' else 'CancelPendingAudit: the HTTP Canceled answer (a user-stream cancel follows)'
              if any(t > a1_at for t, _ in uc) else 'CancelPendingAudit: a user-stream cancel or the HTTP answer')
    a0f = dict(place=near(k_place, r.place), signed=near(k_signed, r.signed), accepted=near(k_acc, r.accepted), notice=near(min(pos) if pos else np.nan, r.notice),
               rM=near(min([t for t, f in fills if f['status'] == 'Matched'], default=np.nan), r.rM),
               rC=near(min([t for t, f in fills if f['status'] == 'Confirmed'], default=np.nan), r.rC),
               M=bool(mt['Matched']) and min(mt['Matched']) == r.M, Cs=bool(mt['Confirmed']) and min(mt['Confirmed']) == r.Cs,
               order_hash=sig_hash == [r.order_hash], trade_id=bool(fills),
               one_bot_clock=len(acc_in) == 1 and np.isfinite(k_acc) and abs(acc_in[0] - k_acc) <= BOT_CLOCK_TOL_MS,
               cancel_confirmation=near(min(t for t, _ in pa), r.ack) if pa else not np.isfinite(float(r.ack)))   # the case table's ack: the first CancelPendingAudit
    return dict(rec=rec, steps=steps, a0=all(a0f.values()), a0f=a0f, a1=a1, a1_ans=a1_ans, a1_at=a1_at, a1_src=a1_src, uc=uc, cq=cq, pos=pos, k_cancel=k_cancel, pa=pa)


def book_pairs(center, Fm, bf, ck):
    """A4 of one leg: identity pairs of its market's bot frames bf [(known_ms, key)] with the research frames Fm; the declared bound B over the
    pairs whose bot receipt is within +-W_LOCAL of `center` (the cancel confirmation); the bot clock's step over the kernel samples ck. ->
    (all pairs, statuses, bound, step ms or None)."""
    tb = np.array([t for t, _ in bf], float); kb = [k for _, k in bf]
    P_, st = pairs(tb, kb, Fm.u.values.astype(float), list(Fm.key.values)) if len(bf) and len(Fm) else (pd.DataFrame(columns=['ib', 'iu', 't_bot', 'u', 'x']), {})
    loc = P_[np.abs(P_.t_bot.values.astype(float) - center) <= W_LOCAL] if len(P_) else P_
    step = float((ck[:, 0] - ck[:, 1]).max() - (ck[:, 0] - ck[:, 1]).min()) / 1e6 if len(ck) and (ck[:, 1] >= 0).all() else None
    return P_, st, bound(loc.x.values if len(loc) else []), step


def leg_bounds(r, a1, a2, v, bp):
    """The typed bounds of one leg (name, value, clock, type, premise)."""
    t = lambda n, x, c, k, p: dict(name=n, value=None if x is None or not np.isfinite(x) else float(x), clock=c, type=k, premise=p)
    out = [t('match >= placement decision', r.place, 'B', 'EVIDENCE', 'an order cannot trade before the engine decided to place it'),
           t('match <= cancel answer receipt (c_hi)', r.ack, 'B', 'EVIDENCE' if a1 not in ('NO-EFFECT', 'NO-ANSWER') else 'NONE',
             f'the venue answered the cancel ({a1}) after processing it, and the leg had matched by then'),
           t('match <= first fill notice', r.notice, 'B', 'EVIDENCE', 'a notice follows its match'),
           t('match within its Matched second', r.M * 1000., 'V_status', 'ASSUMPTION', 'pf-exec rounds match_time to the second (floor premise)'),
           t('match <= Confirmed second end', (r.S + 1.) * 1000., 'V_status', 'EVIDENCE', 'executedAt is the Confirmed second, floored'),
           t('match <= settlement id creation', r.t_set, 'V_uuid', 'CONDITIONAL', 'the UUID is created at or after the match')]
    if v['dom'] == 'V_book':
        out += [t('match <= binding frame stamp u*', v['u'], 'V_book', 'EVIDENCE' if a2 == 'SAME-SETTLEMENT' else 'CONDITIONAL',
                  'the frame showing the pending increment follows the match; the increment is a leg of the same settlement (A2)'),
                t('match >= u* - WS_DELAY', v['u'] - X.WS_DELAY, 'V_book', 'ASSUMPTION', 'S3: the first frame showing a pending increment is at most 250 ms after the match')]
        for tb in bp.t_bot.values:
            out.append(t("match <= the bot's receipt of the binding frame", tb, 'B', 'EVIDENCE' if a2 == 'SAME-SETTLEMENT' else 'CONDITIONAL',
                         'the bot received the frame showing the increment after the match (content identity)'))
    else:
        out.append(t('match >= REST second - REST_LAG', (r.S * 1000. - X.REST_LAG), 'V_status', 'ASSUMPTION', 'S3: the largest Confirmed - Matched lag on 99 own legs'))
    return out


def a6_ledger():
    """A6 on the saved Oct 23 ledger v5 (hash-checked against its receipt)."""
    j = json.loads((LEDGER / 'exec_windows.json').read_text()); ins = {}
    for f in ('ledger_fills.parquet', 'ledger_fill_events.parquet', 'ledger_events.parquet', 'ledger_legs.parquet'):
        C.require(C.sha(LEDGER / f) == j['outputs'][f], f'{f} differs from its ledger receipt'); ins[str(LEDGER / f)] = j['outputs'][f]
    F = pd.read_parquet(LEDGER / 'ledger_fills.parquet'); FE = pd.read_parquet(LEDGER / 'ledger_fill_events.parquet'); E = pd.read_parquet(LEDGER / 'ledger_events.parquet')
    LG = pd.read_parquet(LEDGER / 'ledger_legs.parquet', columns=['day', 'group', 'exec_s', 'S3_llo', 'S3_lhi', 'S3_code', 'ST_llo', 'ST_lhi', 'ST_code'])
    A, ok = a6(F, FE, E, LG, j['R']['xtab'])
    return A, ok, ins


def report(R):
    L = ['TIMING CASE AUDIT (pf_timing_case_audit.py run; consumed Nov 8 to 23:02Z; A6 on the saved Oct 23 ledger v5)',
         '  clock domains: ' + '; '.join(f'{k}: {v}' for k, v in R['clocks'].items())]
    for x in R['legs']:
        a4 = x['A4']
        L += [f"  LEG {x['leg']} {x['run']} cid {x['cid']} market {x['mid']} trade {x['trade_id']} tx {x['tx'][:12]}: S3 lower end {x['S3_lo']:.0f} (venue), "
              f"c_hi {x['c_hi']:.3f} (bot), critical offset {x['critical_ms']:+.1f} ms (K7 [{x['K7']['lo']:+.1f}, {x['K7']['hi']:+.1f}])",
              f"    A0 records reproduce: {x['A0']} {x['A0_fields']}",
              f"    A1 cancel answer: {x['A1']} ({x['A1_source']}; {json.dumps(x['A1_answer'])}) at {x['A1_answer_at']}; match <= c_hi: {x['A1_premise']}; "
              f"user-stream cancel at {x['A1_user_cancel_at']}; post-cancel terminal snapshot {x['A1_terminal_snapshot']}; order {x['order_filled']:g} of {x['order_size']:g} filled; "
              f"filled after a Canceled answer {x['A1_filled_after_canceled_answer']}; a 0-matched snapshot received after the fill notice {x['A1_zero_snapshot_after_fill_notice']}; "
              f"pf-exec alerts naming the order {x['A1_alerts']}",
              f"    A2 event identity: {x['A2']} (binding {x['A2_binding_ids']}, own {x['A2_own_ids']})",
              f"    A3 binding endpoint: reproduces {x['A3_reproduces']}, domain {x['A3_domain']}, u* {x['A3_u']}, binding legs {x['A3_binding_legs']} of {x['A3_event_legs']}, increments {x['A3_increments']}",
              f"    A4 book-clock bound (+-{a4['local_window_ms'] / 1000:g} s): n {a4['n']}, B {a4['B']}, B_max {a4['B_max']}, B_min {a4['B_min']}, statuses {a4['statuses']}; "
              f"clock step {a4['clock_step_ms']} ms over {a4['clock_samples']} steps; binding frame pair x {a4['binding_pair_x']}, its bot receipt - c_hi {a4['binding_receipt_minus_c_hi']}",
              f"    A5 {x['A5']}" + (f"; missing: {'; '.join(x['A5_missing'])}" if x['A5_missing'] else ''),
              '    bounds: ' + ' | '.join(f"{b['name']} = {b['value']} [{b['clock']}, {b['type']}]" for b in x['bounds'])]
    L.append('  A6 evidence tiers on the Oct 23 ledger v5 (REMOVED fills; gross c):')
    for r in R['A6']:
        L.append(f"    {r['scenario']}|{r['cancel']} {r['cls']:<13} {r['tier']:<16} {r['fills']:>6,} fills {r['gross']:>12,.0f} c  {r['basis']}")
    L.append('READINGS (declared before the data was read)')
    L += [f'  {k}: {v}' for k, v in R['readings'].items()]
    L.append('LIMITS: three legs of one consumed partial day; the book-clock premise (stamp before receipt; one stamp per update across subscribers) is stated, '
             'not proven; content identity can pair a repeated book state (the uniqueness rule and the declared order statistic bound it); one-sided bound; '
             'WS_DELAY and REST_LAG remain assumptions; development evidence - Nov 8 is not untouched validation.')
    return '\n'.join(L) + '\n'


def _planted_leg(x_bot=-3., bind_tx='0xabc', acc_shift=0., step_ms=0., answer=None, far=False, snap=(5000000, 15.)):
    """A synthetic leg for leg_eval: the first S3 false avoidance's shape (critical -210.3 ms), its kernel / input records, 31 identity pairs
    (x = x_bot ms, one the binding frame), the REST makers and the clock samples. far: 10 extra pairs 20 s away with x = +500 ms."""
    base = 1794124542000.; cid = 1794108465179877; oid = list(range(32)); tid = '01a1ba27-f105-72b0-83dc-9e674f56ae84'
    r = pd.Series(dict(i=0, run='v6', cid=cid, mid=7, order_hash='0x' + bytes(oid).hex(), trade_id=tid, tx='0xabc', place=base + 150.8, signed=base + 151.1,
                       accepted=base + 183.1, upper=base + 214.3, notice=base + 702.7, ack=base + 214.3, req=base + 200., cdone=base + 214.3, ucancel=base + 292.,
                       rM=base + 702.7, rC=base + 2432.7, M=1794124542., Cs=1794124542., S=1794124542., t_set=base + 213., exec_s=1794124542, first_recv_ns=0, side=1,
                       ptick=23, shares=5., group=5, groups=[5], S3_lo=base + 4., S3_hi=base + 254., critical=4. - 214.3))
    bind = pd.DataFrame([dict(mid=7, exec_s=1794124542, side=1, ptick=60, shares=3., price_executed=.6, taker_index=1, taker_quote=0, first_recv_ns=0)])
    v = dict(row=r, ev=pd.concat([bind, bind]), lo=base + 4., bind=bind, dom='V_book', u=base + 254., eid=146, size=5., filled=5.)
    ns = lambda ms: int(round(ms * 1e6))
    st_ = lambda ms, inp, q: (ms, 'step:' + next(iter(inp)) + '/' + next(iter(inp[next(iter(inp))])),
                              json.dumps({"frame": {"at": ns(ms), "input": inp, "mono": ns(ms) - 10 ** 12, "seq": q}, "i": q, "type": "step", "version": 1}, separators=(',', ':')))
    rec = [st_(r.place, {"Control": {"Place": {"cid": cid, "market": 0, "price": 230000, "side": "BuyDown", "size": 5000000}}}, 1),
           (r.place, 'outbox', '{"frame":{"alerts":[],"cancels":[],"control":[],"places":[],"requests":[],"seq":1,"timers":[]},"i":2,"type":"outbox","version":1}'),
           st_(r.signed, {"Exec": {"Signed": {"cid": cid, "order_id": oid}}}, 2), st_(r.accepted, {"Exec": {"Accepted": {"cid": cid, "status": "Live"}}}, 3),
           st_(r.req + 0.4, {"Control": {"Cancel": {"cid": cid}}}, 4), st_(r.ucancel, {"Exec": {"UserCancel": {"order_id": oid}}}, 6),
           st_(r.rM, {"Exec": {"Fill": {"match_time": int(r.M) * 10 ** 9, "order_id": oid, "price": 1, "role": "Maker", "side": "Buy", "size": 5000000, "status": "Matched", "token": [1], "trade_id": tid}}}, 7),
           st_(r.rC, {"Exec": {"Fill": {"match_time": int(r.Cs) * 10 ** 9, "order_id": oid, "price": 1, "role": "Maker", "side": "Buy", "size": 5000000, "status": "Confirmed", "token": [1], "trade_id": tid}}}, 8),
           st_(r.ack + snap[1], {"Exec": {"CancelQuantityReconciled": {"order_id": oid, "size_matched": snap[0]}}}, 9)]
    rec.append(st_(r.ack, {"Exec": {"CancelPendingAudit": {"order_id": oid}}}, 5))
    if answer is not None:                                    # a first answer before the confirmed cancel (the case table's ack: the pending audit)
        rec.append(st_(r.ack - 1., {"Exec": {"CancelAnswer": {"answer": answer, "cid": cid}}}, 10))
    br = ['{"Accepted":{"market":7,"known_ms":%.4f,"id":146}}' % (r.accepted + acc_shift)]
    u = [r.ack - 9000. + 600. * j for j in range(30)] + [base + 253., base + 254.]; keys = [('k', j) for j in range(30)] + [('kp',), ('kb',)]
    pend = [{} for _ in range(31)] + [{(1, 60): 1.8}]
    if far:
        u += [r.ack + 20_000. + 100. * j for j in range(10)]; keys += [('f', j) for j in range(10)]; pend += [{} for _ in range(10)]
    Fm = pd.DataFrame(dict(u=u, recv_ns=0, key=keys, pend=pend))
    bf = [(uu - x_bot if not (far and k[0] == 'f') else uu - 500., k) for uu, k in zip(u, keys) if k != ('kp',)]
    MR = pd.DataFrame([dict(mid=7, exec_s=1794124542, side=1, ptick=23, shares=5., price_executed=.23, taker_index=1, taker_quote=0, tx='0xabc', sid=tid, order_hash=r.order_hash),
                       dict(mid=7, exec_s=1794124542, side=1, ptick=60, shares=3., price_executed=.6, taker_index=1, taker_quote=0, tx=bind_tx, sid=tid if bind_tx == '0xabc' else 'x',
                            order_hash='0x' + '11' * 32)])
    t_ = np.arange(r.ack - 9000., r.ack + 9000., 500.) * 1e6; ck = np.c_[t_, t_ - 1e12 - np.where(np.arange(len(t_)) >= len(t_) // 2, step_ms * 1e6, 0.)]
    return v, rec, br, Fm, bf, MR, ck, dict(lo=-226.4, hi=49.3)


# ------------------------------------------------------------------ selftest (synthetic only)
def selftest():
    bad = []
    # content key and identity rule
    fr = dict(bids=[[49, 10.0], [48, 5.123456]], asks=[[51, 3.0]], pending=[[0, 49, 1.5], [0, 49, 0.5]])
    d = dict(bids=[[0.48, 5.12345649], [0.49, 10.0]], asks=[[0.51, 3.0]], settlementsPending=dict(bids=[[0.49, 2.0]], asks=[]))
    if key_bot(fr) != key_ws(d):
        bad.append(f'content key: bot {key_bot(fr)} vs ws {key_ws(d)}')
    if key_bot(fr) == key_ws(dict(d, asks=[[0.51, 3.1]])) or key_bot(fr) == key_ws(dict(d, settlementsPending=dict(bids=[[0.49, 2.5]], asks=[]))):
        bad.append('content key: a different size or pending amount gave the same key')
    k1, k2, k3 = ('a',), ('b',), ('c',)
    tb = np.array([1000., 2000., 3000., 20000.]); kb = [k1, k2, k3, k1]
    tu = np.array([1010., 1990., 2500., 3010., 3020.]); ku = [k1, k2, k2, k3, k3]
    P_, st = pairs(tb, kb, tu, ku)
    if not (len(P_) == 1 and P_.x.iloc[0] == 10. and st == {'paired': 1, 'several_research': 2, 'none': 1}):
        bad.append(f'identity rule (unique / several research / none): {P_.to_dict("records")} {st}')
    P2, st2 = pairs(np.array([1000., 1500.]), [k1, k1], np.array([1200.]), [k1])
    if len(P2) or st2 != {'several_bot': 2}:
        bad.append(f'identity rule (two bot twins of one research frame): {st2}')
    b = bound([5., -3., 10., -50., 1.])
    if b != dict(n=5, B=1., B_max=10., B_min=-50.) or bound([])['B'] is not None or bound([2., 1.])['B'] != 2.:
        bad.append(f'bound order statistic: {b}')
    # answers
    if [answer_class(a, p) for a, p in (({'NotCanceled': "order can't be found - already canceled or matched (predict noop)"}, False),
                                       ({'NotCanceled': 'absent from the cancel answer'}, False), ({'Http': 429}, False), (None, False), ('Canceled', False),
                                       (None, True), ('Transport', False))] != ['MATCHED-OR-GONE', 'NO-EFFECT', 'NO-EFFECT', 'NO-ANSWER', 'CANCELED', 'CANCELED', 'NO-EFFECT']:
        bad.append('cancel answer classes')
    # verdict on planted legs
    good = dict(n=40, B=-5., B_max=3., B_min=-80.)
    cases = [((True, 'SAME-SETTLEMENT', True, 'V_book', 0.1, good, -210.), True),
             ((True, 'SAME-SETTLEMENT', True, 'V_book', 0.1, dict(good, B=-300.), -210.), False),
             ((True, 'SAME-SETTLEMENT', True, 'V_book', 0.1, dict(good, n=19), -210.), False),
             ((True, 'SAME-SETTLEMENT', True, 'V_book', 5., good, -210.), False),
             ((True, 'SAME-SETTLEMENT', True, 'V_book', None, good, -210.), False),
             ((True, 'SAME-SETTLEMENT', True, 'V_status', 0.1, good, -210.), False),
             ((True, 'DIFFERENT', True, 'V_book', 0.1, good, -210.), False),
             ((True, 'AMBIGUOUS', True, 'V_book', 0.1, good, -210.), False),
             ((False, 'SAME-SETTLEMENT', True, 'V_book', 0.1, good, -210.), False),
             ((True, 'SAME-SETTLEMENT', False, 'V_book', 0.1, good, -210.), False),
             ((True, 'SAME-SETTLEMENT', True, 'V_book', 0.1, dict(good, B=-210.), -210.), True),
             ((True, 'SAME-SETTLEMENT', True, 'V_book', 0.1, dict(good, B=-210.001), -210.), False)]
    for a, want in cases:
        v, miss = verdict(*a)
        if (v == VERDICT_OK) != want or (not want and not miss):
            bad.append(f'verdict {a[:5]} B {a[5]["B"]} n {a[5]["n"]}: {v} {miss}')
    # the per-leg evaluation on planted legs (leg_eval: A0-A5)
    def ev_(**kw):
        try:
            return leg_eval(0, *_planted_leg(**kw))[0]
        except Exception as e:                            # a crash is a failed check, never a pass
            return dict(A5=f'raised {type(e).__name__}: {e}', A5_missing=['raised'], A0=False, A1='', A2='', A4=dict(n=-1, B=None))
    g0 = ev_()
    if not (g0['A5'] == VERDICT_OK and g0['A0'] and g0['A1'] == 'CANCELED' and g0['A1_premise'] == 'NOT-ORDERED-BY-THE-ANSWER' and 'user-stream cancel follows' in g0['A1_source']
            and g0['A2'] == 'SAME-SETTLEMENT' and g0['A3_reproduces'] and g0['A4']['n'] == 31 and g0['A4']['B'] == -3. and g0['A4']['clock_step_ms'] == 0.
            and g0['A4']['binding_pair_x'] == [-3.] and abs(g0['A4']['binding_receipt_minus_c_hi'][0] - 42.7) < 1e-3 and g0['A3_increments'] == [[(1, 60, 1.8)]]
            and g0['A1_terminal_snapshot'][0]['size_matched'] == 5000000 and abs(g0['cancel_decision_minus_req'] - 0.4) < 1e-3
            and g0['A1_filled_after_canceled_answer'] is True and g0['A1_zero_snapshot_after_fill_notice'] is False):
        bad.append(f"leg_eval on the planted leg: {g0.get('A5')} A0 {g0.get('A0_fields')} A1 {g0.get('A1')} A2 {g0.get('A2')} A4 {g0.get('A4')} inc {g0.get('A3_increments')}")
    for kw, want in ((dict(x_bot=-300.), MISSING['bound']), (dict(bind_tx='0xdef'), MISSING['A2']), (dict(acc_shift=0.5), MISSING['A0']), (dict(step_ms=5.), MISSING['clock'])):
        g = ev_(**kw)
        if g['A5'] == VERDICT_OK or want not in g['A5_missing']:
            bad.append(f'leg_eval {kw}: {g["A5"]} {g["A5_missing"]}')
    g = ev_(far=True)
    if not (g['A5'] == VERDICT_OK and g['A4']['n'] == 31 and g['A4']['B'] == -3.):
        bad.append(f'leg_eval: pairs outside the local window counted: {g["A4"]}')
    v_, rec_, br_ = _planted_leg()[:3]; r_ = v_['row'].copy(); r_['ack'] = np.nan
    L0 = lifecycle(r_, [x for x in rec_ if 'CancelPendingAudit' not in x[2] and 'CancelQuantityReconciled' not in x[2]], br_)
    L1 = lifecycle(r_, rec_, br_)
    if not (L0['a0'] and L0['a1'] == 'NO-ANSWER' and not L1['a0f']['cancel_confirmation']):
        bad.append(f"lifecycle without a cancel confirmation: {L0['a0f']} {L0['a1']} / with an unexpected pending audit {L1['a0f']['cancel_confirmation']}")
    # v3 falsifiers (the census): a sibling own order filled in the same settlement; a refused first answer, then the confirmed cancel; the two
    # journals' stamps 5 us apart (one clock) and 0.5 ms apart (refused by the A0 test above)
    v_, rec_, br_ = _planted_leg()[:3]; r_ = v_['row']
    sib = json.dumps({"frame": {"at": int(round((r_.rM - 300.) * 1e6)), "input": {"Exec": {"Fill": {"match_time": int(r_.M) * 10 ** 9, "order_id": list(range(1, 33)), "price": 1,
                      "role": "Maker", "side": "Buy", "size": 5000000, "status": "Matched", "token": [1], "trade_id": r_.trade_id}}}, "mono": 1, "seq": 50}, "i": 50,
                      "type": "step", "version": 1}, separators=(',', ':'))
    Ls = lifecycle(r_, rec_ + [(r_.rM - 300., 'step:Exec/Fill', sib)], br_)
    if not (Ls['a0f']['notice'] and Ls['a0f']['rM'] and Ls['a0']):
        bad.append(f"lifecycle: a sibling order's fill in the same settlement was counted: {Ls['a0f']}")
    ans_ = json.dumps({"frame": {"at": int(round((r_.ack - 5000.) * 1e6)), "input": {"Exec": {"CancelAnswer": {"answer": {"Http": 429}, "cid": int(r_.cid)}}}, "mono": 1, "seq": 51},
                       "i": 51, "type": "step", "version": 1}, separators=(',', ':'))
    La = lifecycle(r_, rec_ + [(r_.ack - 5000., 'step:Exec/CancelAnswer', ans_)], br_)
    if not (La['a1'] == 'NO-EFFECT' and La['a0f']['cancel_confirmation'] and La['a0']):
        bad.append(f"lifecycle: a refused first answer before the confirmed cancel: {La['a1']} {La['a0f']}")
    Lb = lifecycle(r_, rec_, ['{"Accepted":{"market":7,"known_ms":%.4f,"id":146}}' % (r_.accepted + 0.005)])
    if not Lb['a0f']['one_bot_clock']:
        bad.append('lifecycle: two reads 5 us apart of the one bot clock were refused')
    g = ev_(snap=(0, 600.))
    if not (g.get('A1_zero_snapshot_after_fill_notice') is True and g['A5'] == VERDICT_OK):
        bad.append(f'leg_eval: a 0-matched snapshot after the fill notice not flagged: {g.get("A1_terminal_snapshot")}')
    g = ev_(answer={'NotCanceled': "order can't be found - already canceled or matched (predict noop)"})
    if not (g['A1'] == 'MATCHED-OR-GONE' and g['A1_premise'] == 'EVIDENCE' and g['A5'] == VERDICT_OK):
        bad.append(f'leg_eval with a matched-or-gone answer: {g["A1"]} {g.get("A1_premise")} {g["A5"]}')
    # binding endpoint through sem_windows: a WS leg with one value-consistent increment and a REST-range leg in one event
    L8 = pd.DataFrame(dict(mid=[7, 7], exec_s=[1000, 1000], first_recv_ns=[0, 0], side=[0, 1], ptick=[40, 60], shares=[10., 5.], price_executed=[.4, .6],
                           taker_index=[1, 1], taker_quote=[0, 0], how=[1, 0], t_ms=[1000200., np.nan], t_e2=[np.nan, 999900.], tk=[5, 5]))
    P8 = pd.DataFrame(dict(mid=[7, 7], t_ms=[1000200., 1003000.], ptick=[40, 40], inc=[4.0, 9.9]))
    R_, lo, B_, dom, us = event_binding(5, L8, P8)
    if not (dom == 'V_book' and lo == 1000200. - X.WS_DELAY and us == 1000200. and len(B_) == 1 and int(B_.ptick.iloc[0]) == 40):
        bad.append(f'binding endpoint (WS leg binds): {dom} {lo} {us}')
    R_, lo, B_, dom, us = event_binding(5, L8.assign(how=[0, 0], t_ms=[np.nan, np.nan]), P8)
    if not (dom == 'V_status' and lo == 1000 * 1000. - X.REST_LAG and np.isnan(us)):
        bad.append(f'binding endpoint (REST ranges only): {dom} {lo} {us}')
    F_ = pd.DataFrame(dict(u=[1000100., 1000200., 1000200.], pend=[{(0, 40): 1.0}, {(0, 40): 5.0, (1, 60): 2.}, {(0, 40): 5.0}]))
    if binding_increment(F_, pd.Series(dict(ptick=40)), 1000200.) != [(0, 40, 4.0)]:
        bad.append('binding increment recomputation')
    with tempfile.TemporaryDirectory(dir=I.K.IO) as tmp:
        tmp = Path(tmp)
        # kernel reader: needles, outbox by seq, a step of another order, the cutoff before a corrupt line, the clock check
        cid, oid = 1794108465179877, list(range(32)); oj = '"order_id":[' + ','.join(map(str, oid)) + ']'
        t0 = int((CUTOFF_MS - 10_000.) * 1e6)
        L_ = [{"frame": {"at": t0, "input": {"Control": {"Place": {"cid": cid, "market": 0, "price": 1, "side": "BuyUp", "size": 1}}}, "mono": 5, "seq": 1}, "i": 1, "type": "step", "version": 1},
              {"frame": {"alerts": [], "cancels": [], "control": [], "places": [], "requests": [], "seq": 1, "timers": []}, "i": 2, "type": "outbox", "version": 1},
              {"frame": {"at": t0 + 1_000_000, "input": {"Exec": {"Signed": {"cid": cid, "order_id": oid}}}, "mono": 1_000_005, "seq": 2}, "i": 3, "type": "step", "version": 1},
              {"frame": {"at": t0 + 2_000_000, "input": {"Exec": {"Accepted": {"cid": cid + 1, "status": "Live"}}}, "mono": 2_000_005, "seq": 3}, "i": 4, "type": "step", "version": 1},
              {"frame": {"at": t0 + 3_000_000, "input": {"Exec": {"Fill": {"match_time": 0, "order_id": oid, "price": 1, "role": "Maker", "side": "Buy", "size": 1, "status": "Matched",
                                                                         "token": [1], "trade_id": "01a1ba27-f105-72b0-83dc-9e674f56ae84"}}}, "mono": 3_000_005, "seq": 4}, "i": 5, "type": "step", "version": 1}]
        kp = tmp / 'k.jsonl'
        kp.write_text(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in L_) + '{"frame":{"at":%d,"input":' % int(CUTOFF_MS * 1e6) + ' corrupt\n')
        try:
            rec, clk, ks = kernel_records(kp, {0: dict(cid=cid, oid_json=oj, trade_id='01a1ba27-f105-72b0-83dc-9e674f56ae84')}, {0: (CUTOFF_MS - 20_000., CUTOFF_MS)})
        except Exception as e:
            rec, clk, ks = {}, {}, dict(stopped_at_cutoff=f'raised {e}')
        kinds = [k for _, k, _ in rec.get(0, [])]
        if not (kinds == ['step:Control/Place', 'outbox', 'step:Exec/Signed', 'step:Exec/Fill'] and ks['stopped_at_cutoff'] and len(clk.get(0, [])) == 4):
            bad.append(f'kernel reader: {kinds} {ks} {len(clk.get(0, []))}')
        kp.write_text(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in [L_[2], L_[0]]))
        try:
            kernel_records(kp, {0: dict(cid=cid, oid_json=oj, trade_id='x')}, {}); bad.append('kernel reader: a clock regression was accepted')
        except ValueError:
            pass
        # input reader: frames in / out of the window, the order's records, the cutoff, an open gzip stream
        ip = tmp / 'in.jsonl.gz'; lines = ['{"Frame":{"market":7,"known_ms":%.3f,"bids":[[49,1.0]],"asks":[[51,2.0]],"pending":[]}}' % (CUTOFF_MS - 5000.),
                                          '{"Frame":{"market":7,"known_ms":%.3f,"bids":[[49,1.0]],"asks":[[51,2.0]],"pending":[]}}' % (CUTOFF_MS - 50_000.),
                                          '{"Frame":{"market":8,"known_ms":%.3f,"bids":[[49,1.0]],"asks":[[51,2.0]],"pending":[]}}' % (CUTOFF_MS - 5000.),
                                          '{"Accepted":{"market":7,"known_ms":%.3f,"id":146}}' % (CUTOFF_MS - 4000.), '{"Accepted":{"market":7,"known_ms":1.0,"id":1460}}',
                                          '{"Accepted":{"market":8,"known_ms":2.0,"id":146}}',
                                          '{"Advance":{"now_ms":%.3f}}' % CUTOFF_MS, '{"Frame":{"market":7, corrupt']
        raw = gzip.compress(('\n'.join(lines) + '\n').encode()); ip.write_bytes(raw[:-8])          # no end marker: an open stream
        try:
            f_, r_, s_ = input_frames(ip, {7: [(CUTOFF_MS - 10_000., CUTOFF_MS)]}, [(7, 146)])
        except Exception as e:
            f_, r_, s_ = {}, {}, dict(stopped_at_cutoff=f'raised {e}')
        if not (len(f_.get(7, [])) == 1 and 8 not in f_ and len(r_.get((7, 146), [])) == 1 and s_['stopped_at_cutoff'] is True):
            bad.append(f'input reader: {f_} {r_} {s_}')
        try:
            hours_between(CUTOFF_MS - 60_000., CUTOFF_MS + 3_600_000.); bad.append('hours: a Nov 9 hour was accepted')
        except ValueError:
            pass
        # A6 on a planted ledger
        E = pd.DataFrame(dict(day=['d'] * 4, group=[1, 2, 3, 4], S1_lo_basis=['s1lo'] * 4, S1_hi_basis=['s1hi'] * 4, S2_lo_basis=['s2lo'] * 4, S2_hi_basis=['s2hi'] * 4))
        LG = pd.DataFrame(dict(day=['d'] * 5, group=[1, 2, 2, 3, 4], exec_s=[100] * 5, S3_llo=[99_800., 90_000., 99_700., 90_000., 90_000.],
                               S3_lhi=[100_000., 101_000., 100_100., 101_000., 100_500.], S3_code=[0, 3, 0, 4, 0], ST_llo=[-np.inf, 90_000., -np.inf, 90_000., -np.inf],
                               ST_lhi=[99_990., 101_000., 100_050., 101_000., 100_400.], ST_code=[0, 1, 0, 1, 0]))
        Fl = pd.DataFrame(dict(fid=[0, 1, 2, 3, 4], replay_removed=[True, True, True, False, True], value=[-10., 20., -5., 7., 3.],
                               **{f'cls_{k}_{c}': v for k, v0 in (('S1', [0, 1, 4, 0, 4]), ('S2', [2, 1, 4, 0, 4]), ('S3', [0, 1, 0, 0, 0]), ('ST', [4, 1, 0, 0, 4]))
                                  for c, v in (('CP', v0), ('CR', v0))}, cls_EV_CP=[4, 1, 4, 4, 4], cls_EV_CR=[4, 1, 4, 4, 4]))
        FE = pd.DataFrame(dict(fid=[0, 1, 2, 3, 4], day=['d'] * 5, group=[1, 2, 3, 1, 4]))
        def xt(F):
            o = {}
            for k in ('S1', 'S2', 'S3', 'ST'):
                for c in ('CP', 'CR'):
                    m = F.replay_removed.values; cl = F[f'cls_{k}_{c}'].values
                    o[f'{k}|{c}'] = dict(REMOVED={X.ORD[q]: dict(fills=int((m & (cl == q)).sum()), gross=float(np.abs(F.value.values[m & (cl == q)]).sum())) for q in range(6)})
            return o
        try:
            A, ok = a6(Fl, FE, E, LG, xt(Fl))
        except Exception as e:                            # a refusal on a valid planted ledger is a failed check
            A, ok = pd.DataFrame(columns=['scenario', 'cancel', 'cls', 'tier', 'basis', 'fills', 'gross']), f'raised {e}'
        g = {(r['scenario'], r['cancel'], r['cls'], r['tier'], r['basis']): (r['fills'], r['gross']) for r in A.to_dict('records')}
        want = {('S3', 'CP', 'CANCEL_FIRST', 'BEYOND-EVIDENCE', 'WS pending - WS_DELAY (V_book)'): (1, 10.),
                ('S3', 'CP', 'CANCEL_FIRST', 'BEYOND-EVIDENCE', 'REST second - REST_LAG (V_status)'): (2, 8.),
                ('S3', 'CP', 'MATCH_FIRST', 'EVIDENCE', 'EV: REST second end (V_status)'): (1, 20.),
                ('ST', 'CP', 'CANCEL_FIRST', 'BEYOND-EVIDENCE', 'S3 fallback: REST second - REST_LAG (V_status)'): (1, 5.),
                ('S1', 'CP', 'CANCEL_FIRST', 'BEYOND-EVIDENCE', 's1lo'): (1, 10.), ('S2', 'CP', 'AMBIGUOUS', 'UNCERTIFIED', ''): (1, 10.)}
        s3cp = {k[2:]: v for k, v in g.items() if k[:2] == ('S3', 'CP')}
        if ok is not True or any(g.get(k) != v for k, v in want.items()) or s3cp != {('CANCEL_FIRST', 'BEYOND-EVIDENCE', 'WS pending - WS_DELAY (V_book)'): (1, 10.),
                                                                            ('CANCEL_FIRST', 'BEYOND-EVIDENCE', 'REST second - REST_LAG (V_status)'): (2, 8.),
                                                                            ('MATCH_FIRST', 'EVIDENCE', 'EV: REST second end (V_status)'): (1, 20.)}:
            bad.append(f'A6 tiers on a planted ledger: ok {ok} {g}')
        bx = xt(Fl); bx['S3|CP']['REMOVED']['CANCEL_FIRST']['fills'] += 1
        try:
            if a6(Fl, FE, E, LG, bx)[1]:
                bad.append('A6: a cross-tab that does not reproduce was accepted')
        except Exception as e:
            bad.append(f'A6 on a non-reproducing cross-tab raised {e}')
        try:
            a6(Fl.assign(cls_EV_CP=[0, 1, 4, 4, 4]), FE, E, LG, xt(Fl)); bad.append('A6: an EV CANCEL_FIRST was accepted')
        except ValueError:
            pass
    print('SELFTEST', 'PASS' if not bad else 'FAIL')
    for b_ in bad:
        print('  -', b_)
    return not bad


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    if cmd == 'run':
        C.require(selftest(), 'selftest failed: no run'); run()
    else:
        sys.exit(f'unknown command {cmd}')
