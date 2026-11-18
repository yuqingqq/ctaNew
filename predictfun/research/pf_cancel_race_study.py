"""PER-CANCEL RACE STUDY (2026-11-18, session aea5bfb9; the simulator readiness loop's item 4, review 8e63f178's next priority: "useful if it
separates request, response, actual match bounds and notice times, includes no-fill orders and retries, and preserves censoring / unknown
outcomes. It must not train a cancel failure model from 'notice after answer'"). Declared before any data is read.

WHY  The census (pf_known_clock_census.py) counts cancel answers among orders that FILLED - an outcome-selected population (rule 1). The
simulator's cancel rule (a pull lands at activation + cancel latency; nothing fills after the landing) is a statement about EVERY cancel.
This study takes every cancelled order, no-fill orders and retries included, and keeps the request, the venue's response, the match bounds
and the notices apart. Nothing here is a cancel-failure probability or a training target.
POPULATION  every order (cid placed by Control/Place) with at least one cancel request (Control/Cancel) in the two Nov 8 kernel journals
            (runs v6 and basis; pf_timing_case_audit.KX), each read to Nov 8 23:02Z (pf_timing_case_audit.CUTOFF_MS: no record at or after it
            is parsed). The order's venue id is its Exec/Signed order_id; its fills are the Exec/Fill records carrying that id (never a trade
            id alone: a settlement can fill two own orders).
PER ORDER (bot clock B - the kernel step `at` - unless named):
  request   t_req = the first Control/Cancel, n_req = every Control/Cancel of the cid (retries), t_req_last;
  response  the first confirmation as pf_timing_case_audit.lifecycle defines it (CancelAnswer of the cid or CancelPendingAudit of the order,
            the earlier; its class by pf_timing_case_audit.answer_class), its receipt t_ans and t_ans - t_req; every CancelAnswer's class in
            order (retries); the first UserCancel and CancelQuantityReconciled receipts;
  fills     per trade (the order's own): the first positive notice receipt (Matched / Confirmed), the size, the Matched second M (the earliest
            Matched notice's match_time, venue status clock); non-positive statuses counted by status;
  bounds    per fill: ub_n = the first positive notice receipt (EVIDENCE: a notice follows its match); lb_s = M * 1000 - K7.hi and
            ub_s = (M + 1) * 1000 - K7.lo (CONDITIONAL on K7 - known v7's status-clock offset interval, venue status - bot, and its premises);
  horizon   the run's last step before the cutoff; h = horizon - t_req; censored iff h < H_MIN.
FILL CLASS (computed per fill against the order's t_req and t_ans; t_ans counts only when t_ans >= t_req):
  BEFORE-REQUEST   min(ub_n, ub_s) < t_req (bound EVIDENCE when ub_n < t_req, else CONDITIONAL)
  AFTER-ANSWER     lb_s > t_ans (CONDITIONAL): the venue had answered the cancel and the order matched later
  AFTER-REQUEST    lb_s > t_req, not provably after the answer (CONDITIONAL)
  UNORDERED        none of the above
ORDER CLASS: the first of AFTER-ANSWER, AFTER-REQUEST, UNORDERED, BEFORE-REQUEST among its fills (a post-request match is never hidden by an
  earlier partial fill); without a fill NO-FILL, or CENSORED when h < H_MIN. BOOK EVIDENCE (separate column, never mixed into the class): on
  known v7's legs, K9_match_before_request (CONDITIONAL on the value link and the book-clock premise); a fill K9 places before the request while
  K7 places it after is CONTRADICTORY (a premise fails on that leg).
READINGS (declared; evaluated in code):
  C0 POPULATION: per run - steps, the last step and whether the read stopped at the cutoff; orders placed / signed; cancelled orders,
     requests, orders with retries; statuses: requests of a cid never placed, cancelled orders never signed, a first confirmation before the
     first request.
  C1 RESPONSE: per first-confirmation class - orders, and t_ans - t_req quantiles (p10 / p50 / p90 / p99 / max) and shares above 100 ms, 1 s
     and 4 s. A response is the venue's answer, not the cancel's effect (review 8e63f178 P2-3): descriptive, never a landing time.
  C2 RACE: order classes among cancelled orders (uncensored and censored apart) and by first-confirmation class; fills per fill class with
     their bound types.
  C3 MATCH-AFTER-ANSWER (CONDITIONAL on K7): YES iff some cancelled order has an AFTER-ANSWER fill - the venue matched an order after it had
     answered the order's cancel; else NO. With the count, the share of uncensored cancelled orders and of those whose first confirmation is
     CANCELED. A YES contradicts the simulator's landing rule on those orders under K7's premises; it is not a failure probability.
  C4 BOOK-EVIDENCE: on known v7's legs, fills by fill class x K9 flag; CONTRADICTORY count. CONSISTENT iff 0 CONTRADICTORY, else
     CONTRADICTORY.
  C5 REPRODUCES-THE-CENSUS: on census v4's 86 orders, the first confirmation's class and receipt, the first positive notice and the
     filled-in-full flag equal census v4's (receipts within 1e-6 ms). YES / NO with every mismatch named (an instrument check: the census
     read the same journals through pf_timing_case_audit.lifecycle).
Exclusions are statuses (C0); nothing is dropped silently.
SELFTEST (synthetic, no data): planted journals in the export's line format through the same reader - every fill and order class incl. the
  priority rule, retries, a confirmation by CancelPendingAudit, no answer, censoring, a sibling order's fill in the same trade, a record at the
  cutoff, a request for an unplaced cid; K9 consistent and contradictory; the readings on planted counts; refusals (a regressing clock, an
  empty K7, inputs changed since their receipts). Mutations: pf_cancel_race_study_mutants.py.
LIMITS: one consumed partial day, two runs of the live strategy (not the keep/pull rule); K7 and K9 premises as stated; the cancel's effect
  time is not observed (no class here dates it); fills after the horizon are unseen (censoring).
V2 (2026-11-18, after v1's output was read; v1's receipt cancel_race_study_20261118 stays as provenance). Three definitional defects:
 (1) retries: the bot never re-requests a cancel (one Control/Cancel per order on both runs); pf-exec's HTTP cancel SENDS are the kernel's
 outbox entries {"One": {"cid", "market"}} - a refused send (an HTTP 429 is NO-EFFECT) is re-sent from there. v2 counts every outbox send
 per order (n_send; retries = orders with n_send >= 2) and every CancelAnswer record (n_ans), and the mass cancels ("All") as a status;
 (2) C5 compared census v4's 86 orders, one of which was never cancelled (no request in the case table): v2 compares the census orders
 the case table records a cancel request for (known v7 req finite) - each must be here, equal on every field - and requires every other
 census order to be absent here; (3) C3 dated a match against the FIRST confirmation, which may be a refusal (the order then stays live). v2
 adds C3b MATCH-AFTER-A-CANCELED-CONFIRMATION (CONDITIONAL on K7): YES iff some fill's lb_s exceeds its order's first CANCELED-class
 confirmation (a Canceled CancelAnswer or a CancelPendingAudit) at or after the request; with the count by first-confirmation class.
 Nothing else changes.
Usage (run_light, PF_SLOT=claude): pf_cancel_race_study.py selftest | run
"""
import collections
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import pf_keep_pull as KP                                 # noqa: E402
import pf_timing_case_audit as TCA                        # noqa: E402  the export paths, cutoff, answer classes, step regex

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'cancel_race_study_v2_20261118'     # v2: sends, C5 on the case table's requests, C3b; v1 stays as provenance
KNOWN = Path(I.K.IO) / 'keep_pull_exec_windows_known_v7_20261118'   # K7 (status-clock offset) and K9 per leg
CENSUS = Path(I.K.IO) / 'known_clock_census_v4_20261118'           # C5's reference
RUNS = ('v6', 'basis')
H_MIN = 60_000.                                           # ms after the first request: below it a no-fill order is CENSORED
POSITIVE = ('Matched', 'Confirmed')
WANT = {('Control', 'Place'), ('Control', 'Cancel'), ('Exec', 'Signed'), ('Exec', 'Accepted'), ('Exec', 'CancelAnswer'), ('Exec', 'CancelPendingAudit'),
        ('Exec', 'UserCancel'), ('Exec', 'CancelQuantityReconciled'), ('Exec', 'Fill')}
GATE = re.compile(b'"(?:Place|Cancel|Signed|Accepted|CancelAnswer|CancelPendingAudit|UserCancel|CancelQuantityReconciled|Fill)"')
ORDER_RANK = ('AFTER-ANSWER', 'AFTER-REQUEST', 'UNORDERED', 'BEFORE-REQUEST')
CLASSES = ORDER_RANK + ('NO-FILL', 'CENSORED')


def journal(path, cutoff_ms=TCA.CUTOFF_MS):
    """One run's kernel journal up to cutoff_ms -> ([(at_ms, (k0, k1), payload)], dict(steps, last_at_ms, stopped_at_cutoff)). Only step
    frames are read; a gated line is parsed and kept when its input kind is one of WANT."""
    ev, last, n, stopped = [], 0, 0, False
    with open(path, 'rb') as f:
        for line in f:
            m = TCA.STEP_RE.match(line)
            if not m:
                if last and b'"type":"outbox"' in line[-60:] and b'"cancels":[]' not in line:          # v2: every cancel send, at its step's time
                    for c_ in json.loads(line)['frame'].get('cancels', []):
                        if isinstance(c_, dict) and 'One' in c_:
                            ev.append((last / 1e6, ('Outbox', 'One'), c_['One']))
                        elif c_ == 'All':
                            ev.append((last / 1e6, ('Outbox', 'All'), {}))
                continue
            at = int(m[1])
            if at >= cutoff_ms * 1e6:
                stopped = True; break
            C.require(at >= last, 'kernel UTC clock regressed'); last = at; n += 1
            if GATE.search(line):
                inp = json.loads(line)['frame'].get('input')
                if isinstance(inp, dict) and len(inp) == 1:
                    k0 = next(iter(inp)); v = inp[k0]
                    if isinstance(v, dict) and len(v) == 1 and (k0, next(iter(v))) in WANT:
                        k1 = next(iter(v)); ev.append((at / 1e6, (k0, k1), v[k1]))
    return ev, dict(steps=n, last_at_ms=last / 1e6, stopped_at_cutoff=stopped)


def fill_class(ub_n, lb_s, ub_s, t_req, t_ans):
    """One fill against its order's first request and first confirmation -> (class, bound type)."""
    ub = min(ub_n, ub_s) if np.isfinite(ub_s) else ub_n
    if ub < t_req:
        return 'BEFORE-REQUEST', ('EVIDENCE' if ub_n < t_req else 'CONDITIONAL')
    if np.isfinite(t_ans) and t_ans >= t_req and np.isfinite(lb_s) and lb_s > t_ans:
        return 'AFTER-ANSWER', 'CONDITIONAL'
    if np.isfinite(lb_s) and lb_s > t_req:
        return 'AFTER-REQUEST', 'CONDITIONAL'
    return 'UNORDERED', 'NONE'


def orders_of(ev, run, horizon_ms, k7, k9=None):
    """Per cancelled order of one run (ev: journal events) -> (orders DataFrame, fills DataFrame, statuses). k7: dict(lo, hi) ms (venue status
    - bot); k9: {(run, cid, trade_id): bool} on known legs."""
    C.require(k7 is not None and np.isfinite(k7.get('lo', np.nan)) and np.isfinite(k7.get('hi', np.nan)) and k7['lo'] <= k7['hi'], 'K7 missing or empty: the status bounds fail closed')
    k9 = k9 or {}
    O, o2c, st = {}, {}, collections.Counter()
    for at, (k0, k1), x in ev:
        if (k0, k1) == ('Control', 'Place'):
            if x['cid'] in O:
                st['cid_placed_twice'] += 1; continue
            O[x['cid']] = dict(size=x['size'] / 1e6, t_place=at, reqs=[], sends=[], ans=[], pa=[], uc=[], cq=[], fills={}, other=collections.Counter(), oid=None)
        elif (k0, k1) == ('Exec', 'Signed') and x['cid'] in O:
            O[x['cid']]['oid'] = tuple(x['order_id']); o2c[tuple(x['order_id'])] = x['cid']
        elif (k0, k1) == ('Exec', 'Accepted') and x.get('cid') in O:
            O[x['cid']].setdefault('t_acc', at)
        elif (k0, k1) == ('Outbox', 'One'):
            if x['cid'] in O:
                O[x['cid']]['sends'].append(at)
            else:
                st['send_of_a_cid_never_placed'] += 1
        elif (k0, k1) == ('Outbox', 'All'):
            st['cancel_all_sends'] += 1
        elif (k0, k1) == ('Control', 'Cancel'):
            if x['cid'] in O:
                O[x['cid']]['reqs'].append(at)
            else:
                st['request_of_a_cid_never_placed'] += 1
        elif (k0, k1) == ('Exec', 'CancelAnswer') and x.get('cid') in O:
            O[x['cid']]['ans'].append((at, x['answer']))
        elif k0 == 'Exec' and k1 in ('CancelPendingAudit', 'UserCancel', 'CancelQuantityReconciled', 'Fill'):
            c = o2c.get(tuple(x['order_id']))
            if c is None:
                continue
            o = O[c]
            if k1 == 'CancelPendingAudit':
                o['pa'].append(at)
            elif k1 == 'UserCancel':
                o['uc'].append(at)
            elif k1 == 'CancelQuantityReconciled':
                o['cq'].append((at, int(x['size_matched'])))
            elif x['status'] in POSITIVE:
                a_, q_, m_ = o['fills'].get(x['trade_id'], (np.inf, 0, np.inf))
                m2 = min(m_, x['match_time'] / 1e9) if x['status'] == 'Matched' else m_
                o['fills'][x['trade_id']] = (min(a_, at), max(q_, int(x['size'])), m2)
            else:
                o['other'][x['status']] += 1
    rows, frows = [], []
    for cid, o in O.items():
        if not o['reqs']:
            continue
        t_req = o['reqs'][0]
        if o['oid'] is None:
            st['cancelled_order_never_signed'] += 1
        conf = sorted([(t, 'answer', a) for t, a in o['ans']] + [(t, 'pending_audit', None) for t in o['pa']], key=lambda z: z[0])
        cls = TCA.answer_class(conf[0][2], conf[0][1] == 'pending_audit') if conf else 'NO-ANSWER'; t_ans = conf[0][0] if conf else np.nan
        if conf and t_ans < t_req:
            st['first_confirmation_before_the_first_request'] += 1
        cc = [t for t, kd, a in conf if TCA.answer_class(a, kd == 'pending_audit') == 'CANCELED' and t >= t_req]; t_canc = min(cc) if cc else np.nan   # v2
        h = horizon_ms - t_req; fl = []; after_c = []
        for tid, (a_, q_, m_) in sorted(o['fills'].items(), key=lambda z: z[1][0]):
            mf = np.isfinite(m_); lb = m_ * 1000. - k7['hi'] if mf else np.nan; ub = (m_ + 1.) * 1000. - k7['lo'] if mf else np.nan
            fc, bt = fill_class(a_, lb, ub, t_req, t_ans); kk = k9.get((run, int(cid), tid))
            contra = bool(kk) and fc in ('AFTER-ANSWER', 'AFTER-REQUEST')
            fl.append(fc); ac = bool(np.isfinite(t_canc) and np.isfinite(lb) and lb > t_canc); after_c.append(ac)
            frows.append(dict(run=run, cid=int(cid), trade_id=tid, notice=a_, size=q_ / 1e6, M=m_ if mf else np.nan, lb_s=lb, ub_s=ub, cls=fc, bound=bt,
                              k9=(np.nan if kk is None else bool(kk)), contradictory=contra, after_canceled=ac))
        filled = sum(q for _, q, _ in o['fills'].values()) / 1e6
        ocls = next((c_ for c_ in ORDER_RANK if c_ in fl), None) or ('NO-FILL' if h >= H_MIN else 'CENSORED')
        rows.append(dict(run=run, cid=int(cid), size=o['size'], signed=o['oid'] is not None, t_place=o['t_place'], t_acc=o.get('t_acc', np.nan), t_req=t_req,
                         n_req=len(o['reqs']), t_req_last=o['reqs'][-1], n_send=len(o['sends']), t_send=min(o['sends'], default=np.nan), n_ans=len(o['ans']),
                         cls_first=cls, t_ans=t_ans, latency=t_ans - t_req, t_canceled=t_canc, match_after_canceled=any(after_c),
                         answers=json.dumps([TCA.answer_class(a) for _, a in o['ans']]), t_pa=min(o['pa'], default=np.nan), t_uc=min(o['uc'], default=np.nan),
                         t_cq=min((t for t, _ in o['cq']), default=np.nan), fills=len(o['fills']), filled=filled, full=bool(filled >= o['size'] - 1e-9),
                         first_notice=min((a for a, _, _ in o['fills'].values()), default=np.nan), other_status=json.dumps(dict(o['other'])),
                         horizon=h, censored=bool(h < H_MIN), cls=ocls))
    st['orders_placed'] = len(O); st['orders_signed'] = sum(o['oid'] is not None for o in O.values())
    return pd.DataFrame(rows), pd.DataFrame(frows, columns=['run', 'cid', 'trade_id', 'notice', 'size', 'M', 'lb_s', 'ub_s', 'cls', 'bound', 'k9', 'contradictory',
                                                            'after_canceled']), dict(st)


def q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {k: (float(np.quantile(x, p)) if len(x) else None) for k, p in (('p10', .1), ('p50', .5), ('p90', .9), ('p99', .99), ('max', 1.))}


def readings(O, F, pop, census, requested):
    """C0-C5 on the per-order table O, the per-fill table F, the per-run population pop, census v4's orders and the (run, cid) the case table
    records a cancel request for (v2)."""
    read, R = {}, {}
    R['C0'] = pop
    read['C0 POPULATION'] = '; '.join(f"{r}: {p['cancelled_orders']:,} cancelled orders of {p['orders_placed']:,} placed, {p['requests']:,} requests, "
                                      f"{p['orders_with_retries']:,} with re-sent cancels ({p['sends']:,} sends; {p.get('cancel_all_sends', 0):,} cancel-all sends)"
                                      for r, p in pop.items())
    R['C1'] = {c: dict(orders=int((O.cls_first == c).sum()), latency_ms=q(O.latency[O.cls_first == c]),
                       above_ms={str(t): int((O.latency[O.cls_first == c] > t).sum()) for t in (100, 1000, 4000)}) for c in TCA.ANSWER}
    lat = O.latency.values.astype(float); fin = np.isfinite(lat)
    read['C1 RESPONSE'] = (f"first confirmation after the first request: p50 {np.median(lat[fin]):,.0f} ms, p90 {np.quantile(lat[fin], .9):,.0f} ms, "
                           f"above 1 s {int((lat[fin] > 1000).sum()):,} and above 4 s {int((lat[fin] > 4000).sum()):,} of {int(fin.sum()):,} answered; "
                           f"no answer {int((O.cls_first == 'NO-ANSWER').sum()):,}") if fin.any() else 'NO ANSWERED CANCEL'
    unc = ~O.censored.values.astype(bool)
    R['C2'] = dict(uncensored={c: int(((O.cls == c) & unc).sum()) for c in CLASSES}, censored={c: int(((O.cls == c) & ~unc).sum()) for c in CLASSES},
                   by_first_confirmation={a: {c: int(((O.cls_first == a) & (O.cls == c)).sum()) for c in CLASSES} for a in TCA.ANSWER},
                   fills={c: {b: int(((F.cls == c) & (F.bound == b)).sum()) for b in ('EVIDENCE', 'CONDITIONAL', 'NONE')} for c in ORDER_RANK})
    read['C2 RACE'] = ', '.join(f"{c} {R['C2']['uncensored'][c]:,}" for c in CLASSES) + f" of {int(unc.sum()):,} uncensored cancelled orders"
    aa = O.cls == 'AFTER-ANSWER'; can = O.cls_first == 'CANCELED'
    R['C3'] = dict(orders=int(aa.sum()), uncensored=int((aa & unc).sum()), of_uncensored=int(unc.sum()), canceled_first=int((aa & can).sum()), of_canceled_first=int(can.sum()),
                   by_first_confirmation={a: int((aa & (O.cls_first == a)).sum()) for a in TCA.ANSWER})
    ac = O.match_after_canceled.values.astype(bool); hc = np.isfinite(O.t_canceled.values.astype(float))
    R['C3b'] = dict(orders=int(ac.sum()), of_with_canceled_confirmation=int(hc.sum()), by_first_confirmation={a: int((ac & (O.cls_first == a)).sum()) for a in TCA.ANSWER})
    read['C3b MATCH-AFTER-A-CANCELED-CONFIRMATION'] = (f"YES (CONDITIONAL on K7): {int(ac.sum()):,} of {int(hc.sum()):,} orders with a Canceled confirmation matched after it"
                                                       if ac.any() else f"NO (CONDITIONAL on K7): none of {int(hc.sum()):,} orders with a Canceled confirmation")
    read['C3 MATCH-AFTER-ANSWER'] = (f"YES (CONDITIONAL on K7): {int(aa.sum()):,} cancelled orders matched after the venue answered their cancel "
                                     f"({int((aa & unc).sum()):,} of {int(unc.sum()):,} uncensored; {int((aa & can).sum()):,} of {int(can.sum()):,} first answered Canceled)"
                                     if aa.any() else 'NO')
    kn = F[F.k9.notna()] if len(F) else F
    R['C4'] = dict(known_fills=int(len(kn)), by_class={c: dict(k9_before=int(((kn.cls == c) & (kn.k9 == True)).sum()), k9_not=int(((kn.cls == c) & (kn.k9 == False)).sum()))   # noqa: E712
                                                         for c in ORDER_RANK}, contradictory=int(F.contradictory.sum()) if len(F) else 0)
    read['C4 BOOK-EVIDENCE'] = ('CONSISTENT' if R['C4']['contradictory'] == 0 else 'CONTRADICTORY') + f" on {len(kn)} known fills ({R['C4']['contradictory']} contradictory)"
    m = census.merge(O, on=['run', 'cid'], how='left', suffixes=('_c', ''), indicator='how')
    eq = lambda a, b: (np.isnan(a) and np.isnan(b)) or (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-6)
    bad = []
    for r in m.itertuples():
        asked = (r.run, int(r.cid)) in requested
        if r.how != 'both':
            if asked:
                bad.append(dict(run=r.run, cid=int(r.cid), field='requested in the case table, not a cancelled order here'))
            continue
        if not asked:
            bad.append(dict(run=r.run, cid=int(r.cid), field='a cancelled order here, no request in the case table')); continue
        for fld, ok in (('class', r.cls_c == r.cls_first), ('answer_at', eq(float(r.answer_at), float(r.t_ans))),
                        ('first_notice', eq(float(r.first_notice_c), float(r.first_notice))), ('full', bool(r.full_c) == bool(r.full))):
            if not ok:
                bad.append(dict(run=r.run, cid=int(r.cid), field=fld))
    R['C5_mismatches'] = bad
    nreq = sum((r_, int(c_)) in requested for r_, c_ in zip(census.run, census.cid))
    read['C5 REPRODUCES-THE-CENSUS'] = (('YES' if not bad and nreq > 0 else 'NO') + f" ({len(census) - len({(b['run'], b['cid']) for b in bad})} of {len(census)} census orders; "
                                        f"{nreq} with a cancel request in the case table)")
    return read, R


def k9_of(K):
    return {(r.run, int(r.cid), r.trade_id): bool(r.K9_match_before_request) for r in K.itertuples()}


def run(out=OUT):
    """Both runs' journals to the cutoff, the per-order and per-fill tables, the readings and the receipt."""
    start = time.time(); out = Path(out); C.require(not out.exists(), 'output exists')
    jk = json.loads((KNOWN / 'known.json').read_text()); C.require(C.sha(KNOWN / 'known_cases.parquet') == jk['outputs']['known_cases.parquet'], 'known v7 cases differ from their receipt')
    jc = json.loads((CENSUS / 'census.json').read_text()); C.require(C.sha(CENSUS / 'orders.parquet') == jc['outputs']['orders.parquet'], 'census v4 orders differ from their receipt')
    KC = pd.read_parquet(KNOWN / 'known_cases.parquet'); k7 = jk['R']['K7']; k9 = k9_of(KC); census = pd.read_parquet(CENSUS / 'orders.parquet')
    requested = {(r_, int(c_)) for r_, c_, q_ in zip(KC.run, KC.cid, KC.req.astype(float)) if np.isfinite(q_)}
    inputs = {str(KNOWN / 'known.json'): C.sha(KNOWN / 'known.json'), str(CENSUS / 'census.json'): C.sha(CENSUS / 'census.json')}
    Os, Fs, pop = [], [], {}
    for r in RUNS:
        p = TCA.KX / f'{r}.kernel.jsonl'; inputs[str(p)] = C.sha(p)
        ev, s = journal(p); O, F, st = orders_of(ev, r, s['last_at_ms'], k7, k9); del ev; C.guard('cancel race study run')
        pop[r] = dict(s, **st, cancelled_orders=int(len(O)), requests=int(O.n_req.sum()) if len(O) else 0, sends=int(O.n_send.sum()) if len(O) else 0,
                      orders_with_retries=int((O.n_send > 1).sum()) if len(O) else 0, orders_with_several_answers=int((O.n_ans > 1).sum()) if len(O) else 0)
        Os.append(O); Fs.append(F)
        print(json.dumps(dict(stage='run', run=r, cancelled=len(O), fills=len(F), seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    O = pd.concat(Os, ignore_index=True); F = pd.concat(Fs, ignore_index=True); read, R = readings(O, F, pop, census, requested)
    out.mkdir(parents=True); O.to_parquet(out / 'orders.parquet', index=False); F.to_parquet(out / 'fills.parquet', index=False)
    res = dict(asof=pd.Timestamp.now(tz='UTC').isoformat(), readings=read, R=R, k7=k7, h_min_ms=H_MIN, cutoff_ms=TCA.CUTOFF_MS, inputs=inputs,
               outputs={f: C.sha(out / f) for f in ('orders.parquet', 'fills.parquet')}, source_sha256=C.sha(Path(__file__)), seconds=round(time.time() - start))
    (out / 'study.json').write_text(json.dumps(res, indent=1, default=str)); txt = report(res); (out / 'study.txt').write_text(txt); print(txt)
    return 0


def report(res):
    L = [f"PER-CANCEL RACE STUDY (Nov 8 to 23:02Z, runs {', '.join(RUNS)}; K7 [{res['k7']['lo']:+.1f}, {res['k7']['hi']:+.1f}] ms; censored below "
         f"{res['h_min_ms'] / 1000:.0f} s of horizon)", '']
    L += [f'  {k}: {v}' for k, v in res['readings'].items()]
    L += ['', '  C1 by first confirmation: orders, latency p10 / p50 / p90 / p99 / max (ms), above 100 ms / 1 s / 4 s']
    for c, x in res['R']['C1'].items():
        lq = x['latency_ms']; f_ = lambda v: '-' if v is None else f'{v:,.0f}'
        L.append(f"    {c:<16} {x['orders']:>7,}   {' / '.join(f_(lq[k]) for k in ('p10', 'p50', 'p90', 'p99', 'max'))}   "
                 f"{' / '.join(str(x['above_ms'][t]) for t in ('100', '1000', '4000'))}")
    L += ['', '  C2 order class by first confirmation (all cancelled orders): ' + ' | '.join(CLASSES)]
    for a, x in res['R']['C2']['by_first_confirmation'].items():
        L.append(f"    {a:<16} " + ' | '.join(f'{x[c]:,}' for c in CLASSES))
    L += ['', '  NOT A CANCEL-FAILURE MODEL: classes bound the match against the request and the answer; the cancel\'s effect time is not observed.']
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ selftest (synthetic only)
def _line(ms, inp, q):
    return json.dumps({"frame": {"at": int(round(ms * 1e6)), "input": inp, "mono": 1, "seq": q}, "i": q, "type": "step", "version": 1}, separators=(',', ':'))


def _outbox(q, cancels):
    return json.dumps({"frame": {"alerts": [], "cancels": cancels, "control": [], "places": [], "requests": [], "seq": q, "timers": []}, "i": q, "type": "outbox",
                       "version": 1}, separators=(',', ':'))


def _journal_lines(base, cut):
    """Planted orders (cid -> what it plants). Times in ms from base; K7 = [-200, +50]."""
    L, q = [], [0]

    def st(t, inp):
        q[0] += 1; L.append((base + t, _line(base + t, inp, q[0])))

    def send(c):                                             # an outbox cancel send after the last step (v2)
        L.append((L[-1][0], _outbox(q[0], [{"One": {"cid": c, "market": 0}}] if c is not None else ["All"])))
    oid = lambda c: [c % 256] * 32
    place = lambda t, c: st(t, {"Control": {"Place": {"cid": c, "market": 0, "price": 500000, "side": "BuyUp", "size": 5000000}}})
    sign = lambda t, c: st(t, {"Exec": {"Signed": {"cid": c, "order_id": oid(c)}}})
    acc = lambda t, c: st(t, {"Exec": {"Accepted": {"cid": c, "status": "Live"}}})
    canc = lambda t, c: (st(t, {"Control": {"Cancel": {"cid": c}}}), send(c))
    ans = lambda t, c, a: st(t, {"Exec": {"CancelAnswer": {"answer": a, "cid": c}}})
    pa = lambda t, c: st(t, {"Exec": {"CancelPendingAudit": {"order_id": oid(c)}}})
    fill = lambda t, c, tid, m_s, status='Matched', size=5000000, o=None: st(t, {"Exec": {"Fill": {"match_time": int(m_s) * 10 ** 9, "order_id": o or oid(c), "price": 1,
                                                                                                   "role": "Maker", "side": "Buy", "size": size, "status": status, "token": [1], "trade_id": tid}}})
    S = (base // 1000)                                       # base is a whole second
    st(500., {"Meta": {"start": 1}}); send(None)                # a cancel-all at the start (a status)
    for c in range(1, 15):
        place(c * 1000., c); sign(c * 1000. + 1, c); acc(c * 1000. + 30, c)
    # 1 retries, no fill: NotCanceled then Canceled
    canc(20_000., 1); ans(20_040., 1, {"NotCanceled": "order is not live"}); canc(20_100., 1); ans(20_140., 1, "Canceled")
    # 2 a fill notified before the request (EVIDENCE)
    fill(21_500., 2, 't2', S + 21); canc(22_000., 2); ans(22_040., 2, "Canceled")
    # 3 Canceled at 23_040; Matched second 25 -> lb = 25_000 - 50 = 24_950 > 23_040: AFTER-ANSWER
    canc(23_000., 3); ans(23_040., 3, "Canceled"); fill(25_300., 3, 't3', S + 25)
    # 3's sibling: another order in the same trade (its fill must not be 3's)
    fill(25_301., 3, 't3', S + 25, o=[99] * 32)
    # 4 a slow answer: request 26_000, answer 30_000; Matched second 27 -> lb 26_950 in (26_000, 30_000): AFTER-REQUEST
    canc(26_000., 4); fill(27_400., 4, 't4', S + 27); ans(30_000., 4, "Canceled")
    # 5 Matched second 31 straddles the request at 31_100 (lb 30_950, ub_s 32_200, notice 31_900): UNORDERED (lb from K7.lo would be 31_200)
    canc(31_100., 5); ans(31_140., 5, "Canceled"); fill(31_900., 5, 't5', S + 31)
    # 6 no fill, requested 8 s before the horizon: CENSORED (the horizon is the last step, planted below)
    # 7 an early fill (notice before the request) and a match after the answer: AFTER-ANSWER by priority, two fills
    fill(32_200., 7, 't7a', S + 32, size=2000000); canc(33_000., 7); ans(33_040., 7, "Canceled"); fill(35_200., 7, 't7b', S + 35, size=3000000)
    # 8 the first confirmation by CancelPendingAudit, no fill: CANCELED / NO-FILL
    canc(36_000., 8); pa(36_070., 8)
    # 9 no answer at all, no fill: NO-ANSWER / NO-FILL
    canc(37_000., 9)
    # 10 a request for a cid never placed
    canc(38_000., 777)
    # 11 a Confirmed-only fill after the request (no Matched second: no status bound) -> UNORDERED
    canc(39_000., 11); ans(39_040., 11, "Canceled"); fill(41_000., 11, 't11', S + 40, status='Confirmed')
    # 12 Matched second 42, request 43_000, notice 43_300: ub_s 43_200 keeps it UNORDERED (ub_s from K7.hi would be 42_950: before)
    canc(43_000., 12); ans(43_040., 12, "Canceled"); fill(43_300., 12, 't12', S + 42)
    # 13 Matched second 44, request 45_500, notice 45_700: ub_s 45_200 < 45_500 - BEFORE-REQUEST on the CONDITIONAL status bound
    canc(45_500., 13); ans(45_540., 13, "Canceled"); fill(45_700., 13, 't13', S + 44)
    # 14 (v2) an HTTP 429 refusal at 47_030 and a re-send; Matched second 48 (lb 47_950): AFTER-ANSWER, but before its Canceled confirmation
    #    (the pending audit at 49_000): no match after a Canceled confirmation
    canc(47_000., 14); ans(47_030., 14, {"Http": 429}); send(14); fill(48_600., 14, 't14', S + 48); pa(49_000., 14)
    canc(cut - base - 8_000., 6)
    st(cut - base - 1., {"Meta": {"tick": 1}})                # the horizon: the last step before the cutoff
    fill(cut - base, 1, 't1late', S + 60)                     # at the cutoff: never read
    fill(cut - base + 5., 6, 't6late', S + 60)
    return [x for _, x in L]


def _syn(base, cut, lines=None):
    tmp = Path(tempfile.mkdtemp(prefix='pf_crs_')); p = tmp / 'syn.kernel.jsonl'
    p.write_text('\n'.join(lines if lines is not None else _journal_lines(base, cut)) + '\n')
    return p


def selftest():
    """Synthetic only."""
    bad = []; base = 1794100000000.; cut = base + 120_000.; k7 = dict(lo=-200., hi=50.)
    p = _syn(base, cut); ev, s = journal(p, cut)
    if not (s['stopped_at_cutoff'] and s['last_at_ms'] == cut - 1.):
        bad.append(f'the read did not stop at the cutoff or lost the horizon: {s}')
    k9 = {('syn', 4, 't4'): True, ('syn', 5, 't5'): True, ('syn', 2, 't2'): True}
    O, F, st = orders_of(ev, 'syn', s['last_at_ms'], k7, k9); o = O.set_index('cid')
    want = {1: 'NO-FILL', 2: 'BEFORE-REQUEST', 3: 'AFTER-ANSWER', 4: 'AFTER-REQUEST', 5: 'UNORDERED', 6: 'CENSORED', 7: 'AFTER-ANSWER', 8: 'NO-FILL', 9: 'NO-FILL',
            11: 'UNORDERED', 12: 'UNORDERED', 13: 'BEFORE-REQUEST', 14: 'AFTER-ANSWER'}
    got = {c: o.loc[c, 'cls'] if c in o.index else None for c in want}
    if got != want:
        bad.append(f'order classes {got} != {want}')
    wf = {1: 'NO-EFFECT', 2: 'CANCELED', 3: 'CANCELED', 4: 'CANCELED', 7: 'CANCELED', 8: 'CANCELED', 9: 'NO-ANSWER', 14: 'NO-EFFECT'}
    if {c: o.loc[c, 'cls_first'] for c in wf} != wf:
        bad.append(f"first confirmations {({c: o.loc[c, 'cls_first'] for c in wf})} != {wf}")
    near = lambda a, b: abs(float(a) - float(b)) <= 1e-3                 # ns -> ms through a double: ~2.4e-4 ms spacing near 1.8e12
    if not (o.loc[1, 'n_req'] == 2 and json.loads(o.loc[1, 'answers']) == ['NO-EFFECT', 'CANCELED'] and near(o.loc[1, 't_req'], base + 20_000.) and near(o.loc[1, 't_req_last'], base + 20_100.)):
        bad.append('retries: request count, answers in order or request times wrong')
    if not (near(o.loc[4, 'latency'], 4000.) and near(o.loc[3, 'latency'], 40.) and near(o.loc[8, 'latency'], 70.) and np.isnan(o.loc[9, 'latency'])):
        bad.append('response latencies wrong')
    if not (o.loc[3, 'fills'] == 1 and abs(o.loc[3, 'filled'] - 5.) < 1e-9 and o.loc[7, 'fills'] == 2 and bool(o.loc[7, 'full']) and bool(o.loc[11, 'full'])):
        bad.append(f"fills per order: a sibling's fill attributed or sizes wrong ({o.loc[3, 'fills']}, {o.loc[3, 'filled']}, {o.loc[7, 'fills']})")
    f2 = F[(F.cid == 2)].iloc[0]; f7 = F[F.cid == 7].set_index('trade_id')
    if not (f2.bound == 'EVIDENCE' and F[F.cid == 13].bound.tolist() == ['CONDITIONAL'] and f7.loc['t7a', 'cls'] == 'BEFORE-REQUEST' and f7.loc['t7b', 'cls'] == 'AFTER-ANSWER'):
        bad.append('fill classes or bound types wrong')
    if not (o.loc[1, 'n_send'] == 2 and o.loc[14, 'n_send'] == 2 and o.loc[14, 'n_ans'] == 1 and o.loc[3, 'n_send'] == 1 and st.get('cancel_all_sends') == 1
            and st.get('send_of_a_cid_never_placed') == 1 and bool(o.loc[3, 'match_after_canceled']) and bool(o.loc[7, 'match_after_canceled'])
            and not bool(o.loc[14, 'match_after_canceled']) and not bool(o.loc[4, 'match_after_canceled']) and near(o.loc[14, 't_canceled'], base + 49_000.)):
        bad.append(f"v2 sends / Canceled confirmations wrong ({o.loc[[1, 3, 14], ['n_send', 'n_ans', 'match_after_canceled', 't_canceled']].to_dict('index')}, {st})")
    if not (bool(o.loc[6, 'censored']) and not bool(o.loc[1, 'censored']) and not bool(o.loc[14, 'censored']) and 777 not in o.index and st.get('request_of_a_cid_never_placed') == 1):
        bad.append(f'censoring or the unplaced request status wrong ({st})')
    if F[(F.trade_id == 't1late') | (F.trade_id == 't6late')].shape[0]:
        bad.append('a fill at / after the cutoff was read')
    fk = F.set_index(['cid', 'trade_id'])
    if not (bool(fk.loc[(4, 't4'), 'contradictory']) and not bool(fk.loc[(5, 't5'), 'contradictory']) and not bool(fk.loc[(2, 't2'), 'contradictory'])
            and np.isnan(fk.loc[(3, 't3'), 'k9'])):
        bad.append('K9 consistency wrong')
    # readings on the planted tables; C5 against a planted census equal to the study, then with one answer receipt moved
    cen = O[O.fills > 0][['run', 'cid']].assign(cls=O[O.fills > 0].cls_first.values, answer_at=O[O.fills > 0].t_ans.values,
                                                first_notice=O[O.fills > 0].first_notice.values, full=O[O.fills > 0].full.values)
    req = {('syn', int(c)) for c in cen.cid}
    cen = pd.concat([cen, pd.DataFrame([dict(run='syn', cid=10, cls='NO-ANSWER', answer_at=np.nan, first_notice=base + 10_500., full=True)])], ignore_index=True)  # never cancelled
    pop = {'syn': dict(s, **st, cancelled_orders=len(O), requests=int(O.n_req.sum()), sends=int(O.n_send.sum()), orders_with_retries=int((O.n_send > 1).sum()))}
    read, R = readings(O, F, pop, cen, req)
    if not (read['C3 MATCH-AFTER-ANSWER'].startswith('YES') and R['C3']['orders'] == 3 and R['C3']['canceled_first'] == 2 and R['C3']['by_first_confirmation']['NO-EFFECT'] == 1
            and read['C3b MATCH-AFTER-A-CANCELED-CONFIRMATION'].startswith('YES') and R['C3b']['orders'] == 2 and read['C5 REPRODUCES-THE-CENSUS'].startswith('YES')
            and read['C4 BOOK-EVIDENCE'].startswith('CONTRADICTORY') and R['C4']['contradictory'] == 1 and R['C2']['censored']['CENSORED'] == 1
            and R['C1']['CANCELED']['orders'] == 9 and R['C1']['CANCELED']['above_ms']['1000'] == 1 and pop['syn']['orders_with_retries'] == 2):
        bad.append(f'readings on planted tables: {read} {R["C3"]} {R["C4"]}')
    c2 = cen.copy(); c2.loc[c2.index[0], 'answer_at'] = c2.answer_at.iloc[0] + 1.
    r2, R2 = readings(O, F, pop, c2, req)
    if not (r2['C5 REPRODUCES-THE-CENSUS'].startswith('NO') and R2['C5_mismatches'] == [dict(run='syn', cid=int(c2.cid.iloc[0]), field='answer_at')]):
        bad.append(f"C5 accepted a moved answer receipt: {r2['C5 REPRODUCES-THE-CENSUS']}")
    O3 = O[O.cls != 'AFTER-ANSWER']; r3, _ = readings(O3, F[~F.cid.isin([3, 7, 14])], pop, cen[~cen.cid.isin([3, 7, 14])], req)
    if r3['C3 MATCH-AFTER-ANSWER'] != 'NO' or not r3['C3b MATCH-AFTER-A-CANCELED-CONFIRMATION'].startswith('NO') or not r3['C4 BOOK-EVIDENCE'].startswith('CONTRADICTORY'):
        bad.append(f"C3 / C3b without after-answer orders: {r3['C3 MATCH-AFTER-ANSWER']} / {r3['C3b MATCH-AFTER-A-CANCELED-CONFIRMATION']}")
    for nm_, rq_, fld in (('the never-cancelled census order marked requested', req | {('syn', 10)}, 'requested in the case table, not a cancelled order here'),
                          ('a cancelled census order without a request', req - {('syn', 2)}, 'a cancelled order here, no request in the case table')):
        r4, R4 = readings(O, F, pop, cen, rq_)
        if not (r4['C5 REPRODUCES-THE-CENSUS'].startswith('NO') and [m_['field'] for m_ in R4['C5_mismatches']] == [fld]):
            bad.append(f"C5 accepted {nm_}: {r4['C5 REPRODUCES-THE-CENSUS']} {R4['C5_mismatches']}")
    # refusals
    for nm, fn in (('an empty K7', lambda: orders_of(ev, 'syn', s['last_at_ms'], dict(lo=10., hi=-10.))),
                   ('a missing K7', lambda: orders_of(ev, 'syn', s['last_at_ms'], None)),
                   ('a regressing clock', lambda: journal(_syn(base, cut, [_line(base + 10., {"Meta": {}}, 1), _line(base + 5., {"Meta": {}}, 2)]), cut))):
        try:
            fn(); bad.append(f'accepted {nm}')
        except (SystemExit, ValueError) as e:                       # C.require raises ValueError
            if nm != 'a regressing clock' and 'K7' not in str(e) or nm == 'a regressing clock' and 'regressed' not in str(e):
                bad.append(f'{nm} refused by another check: {e}')
    if bad:
        print('CANCEL RACE STUDY SELFTEST FAILED: ' + ' | '.join(bad)); return False
    print('cancel race study selftests passed (synthetic only): every fill and order class incl. the priority rule; retries in order; a confirmation by '
          'CancelPendingAudit; no answer; censoring; a sibling order in the same trade; records at / after the cutoff unread; an unplaced request '
          'as a status; K9 consistent and contradictory; C1-C5 on planted tables, C5 refusing a moved receipt; refusals: an empty or missing K7, a '
          'regressing clock; v2: outbox sends and re-sends after an HTTP 429, a cancel-all as a status, the first Canceled confirmation (C3b YES / '
          'NO), C5 against the case table\'s requests (both directions refused)')
    return True


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    elif cmd == 'run':
        sys.exit(run())
    else:
        sys.exit(__doc__)
