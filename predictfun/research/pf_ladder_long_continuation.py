"""LONG-CONTINUATION-PAIRED-VALUE-v1 (P-2026-005; 2026-11-22, session aea5bfb9). Review 32d102bd's ONE next priority, specified by
reviews/ladder_long_continuation_contract_20261122/contract_v2.md (findings 1-2 folded in). USER 2026-11-22: approved <= 0.40 additional
core-hours including controls and a 0.03 h review reserve, ladder line ceiling 2.85. A paired diagnostic on the consumed pilot markets: not a
strategy, a validation, a skill or a profitability claim; no new day, fit or untouched validation; POSITIVE-PRESSURE-GUARD-v1 stays stopped.
POPULATION (census frozen in the manifest before any pair): CURRENT_ASYNC_FAST10 on the 8 pilot markets (first two complete per date, Oct
 24-27) from ASYNC-EXECUTION-SENSITIVITY-v1's hash-bound records. REVIEW = per generation its first policy-consulted recorded live KEEP (reason
 'keep'; a fallback KEEP never consulted the policy); ADMISSION = every recorded POST. Per market x side x kind the PER_CELL = 2 states with
 the smallest sha256(arm|day|m|kind|side|gid) (ADMISSION adds |t); every state keeps a status.
FORK: one roll-in per market with the frozen ASYNC client (ProfileBranch + PressureGuard through LineageGuard, inert until armed, + QueueObs),
 ticks to the state's tick; deep copies of (branch, policy, observer) per arm; the ONE forced action is applied at the identified callback
 (REVIEW: KEEP / PULL for the gid; ADMISSION: POST / WAIT at the side's rung), after that tick's trades, notices, structural requests and any
 earlier side or earlier order (the common prefix is replayed; the context at the callback is fingerprinted); both arms continue to the same
 market END. ONE holding rule in BOTH arms after the fork: every live order of the focal lineage (side, birth rung: the focal gid and every
 gid submitted into that slot at or after the fork) is exempt from pressure_pull; pressure-withholding, structural / inventory removals,
 caps, sizes, notices and price locks stay; other slots follow the pinned policy with their own arm's feedback.
VALUE: Q = sum over post-fork fills s q (M_END - p) + I_fork (M_END - M_fork) (= value_END - cash_fork - I_fork M_fork, reconciled per arm);
 Y = Q_retain - Q_alternative (the carried inventory cancels). Economic axis: retain_better (Y > TOL) / alternative_better (Y < -TOL) / tie;
 invalid statuses apart. Sequence axis (each arm): the focal generation's first post-fork fill group = early, later lineage fills = later,
 END-signed: no_focal_fill / early_loss_later_benefit / early_loss_no_later_benefit / no_early_loss_later_benefit /
 no_early_loss_no_later_benefit. Mechanism: retention_beneficial_with(out)_early_loss / alternative_better / tie; queue dependence
 'unidentified' (the pilot does not separate queue loss from absence, price, size or other-order feedback). Decomposition: focal lineage vs
 other orders; execution edge vs inventory movement; buy / sell shares, FIFO matched volume (the carried inventory opens a lot at the fork)
 and lot holding time; time-weighted / peak / terminal |I|; participation; requests, reposts, pressure pulls, suppressed pulls, post-to-pull
 reversals within 100 ms; lineage replacements' delay, price and activation queue; 1 / 5 / 30 / 60 s markouts (native, common cohort).
GATES (computed; a failure is a status or a refusal, never absorbed): the base roll-in completed to END = the saved record (fingerprint,
 fills, orders, requests, value, guard log, command array); every retain arm whose exemption suppressed nothing = the saved record (the
 unused-exemption control); the forced action applied exactly once in each arm at a context fingerprint equal in both; pre-fork fills
 identical; every lineage order's activation queue = the shown size recomputed from its activation frame; no pressure_pull on a lineage
 order after the fork in either arm; per-arm cash / inventory reconciliation; the inherited per-command invariants.
PARTS: subset = the first date's two pilot markets (16 pairs), then the write-once projection (this batch's spent + the measured subset cost
 x 3 for the remaining six markets + measure / checks + tail <= the batch cap net of the review reserve; the line + that <= 2.85), then rest.
Usage (run_light): pf_ladder_long_continuation.py selftest | manifest | subset | project | rest <day> | measure [out]"""
import copy
import gzip
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_async_fast as AF                         # noqa: E402  frozen (09418aff): ProfileBranch, the ASYNC inputs

QF, P, M, LG, LB, BE, C, pd = AF.QF, AF.P, AF.M, AF.LG, AF.LB, AF.BE, AF.C, AF.pd
AR = M.AR
VERSION = 'long_continuation_paired_value_v1'
ARM, PROFILE, KINDS, PER_CELL, TOL, REVERSAL_MS = 'CURRENT_ASYNC_FAST10', 'ASYNC_FAST10', ('REVIEW', 'ADMISSION'), 2, 1e-6, 100.
ACTIONS = dict(REVIEW=('KEEP', 'PULL'), ADMISSION=('POST', 'WAIT'))
HORIZONS = (1000., 5000., 30000., 60000.)
LIVE, SLOT, POST, KEEP, KEEP_REASON = 0, 1, 0, 2, 1
SUBSET_DAY = '20261024'
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_long_continuation_20261122'
OUT = REPO / 'data/predict_fun/research/ladder_long_continuation_v1_20261122'
CAP_BATCH, REVIEW_RESERVE, LINE_CEILING, TAIL_S, PRIOR_UNLOGGED_S = 0.40, 0.03, 2.85, 60., 30.
SOURCES = ('pf_ladder_long_continuation.py', 'pf_ladder_long_continuation_selftest.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_long_continuation.sh', 'lc_checks.py')
SG = {0: 1., 1: -1.}


def need(cond, msg):
    C.require(cond, msg)


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def committed(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def sources():
    return {n: sha_b((HERE / n).read_bytes()) for n in SOURCES}


def frozen():
    return bool(all(committed(f'predictfun/research/{n}') == (HERE / n).read_bytes() for n in SOURCES)
                and all(committed(f'{RECEIPT}/{n}') == (REPO / RECEIPT / n).read_bytes() for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the fork controls
def ctx_hash(v, o=None):
    """The fingerprint of the decision context at the forced callback (the client view and, for REVIEW, the order view)."""
    return hashlib.sha256(json.dumps([v, o], sort_keys=True, default=str).encode()).hexdigest()


class LineageGuard(P.PressureGuard):
    """The frozen PressureGuard, inert until arm(): then the ONE forced action at the identified callback and the focal-lineage pressure-pull
    exemption (the same rule in both arms)."""

    def __init__(self, B, cache, edges):
        super().__init__(B, cache, edges); self.fork, self.applied, self.suppressed = None, [], []

    def arm(self, st, action, exempt=True):
        self.fork = dict(t=float(st['t']), side=int(st['side']), kind=st['kind'], gid=int(st['gid']), rung=float(st['rung']), action=action, exempt=bool(exempt))

    def member(self, side, o):
        f = self.fork
        return bool(f['exempt'] and side == f['side'] and ((f['kind'] == 'REVIEW' and int(o['gid']) == f['gid'])
                                                          or (float(o['rung']) == f['rung'] and float(o['submit']) >= f['t'])))

    def live(self, v, o):
        f = self.fork; t = float(v['t'])
        if f is None or t < f['t']:
            return super().live(v, o)
        if f['kind'] == 'REVIEW' and t == f['t'] and int(v['side']) == f['side'] and int(o['gid']) == f['gid'] and not self.applied:
            if int(o['gid']) not in self.seen:
                self.seen.add(int(o['gid'])); self.q_first.append((int(o['gid']), t))
            self.inner.live(v, o); self.signal(v); self.applied.append(('live', t, int(o['gid']), f['action'], ctx_hash(v, o)))
            return f['action'], ('keep' if f['action'] == 'KEEP' else 'forced_pull')
        if self.member(int(v['side']), o):
            if int(o['gid']) not in self.seen:
                self.seen.add(int(o['gid'])); self.q_first.append((int(o['gid']), t))
            a, why = self.inner.live(v, o)
            if a == 'KEEP' and self.signal(v)[0]:
                self.suppressed.append((t, int(o['gid'])))
            return a, why
        return super().live(v, o)

    def slot(self, v, r, price):
        f = self.fork; t = float(v['t'])
        if f is not None and f['kind'] == 'ADMISSION' and t == f['t'] and int(v['side']) == f['side'] and float(r) == f['rung'] and not self.applied:
            a, q, why = self.inner.slot(v, r, price)
            if a == 'POST':
                self.signal(v)
            self.applied.append(('slot', t, float(r), f['action'], ctx_hash(v)))
            if f['action'] == 'POST':
                need(a == 'POST', f'the recorded POST at {t} is not the inner decision at the fork ({a})'); return a, q, why
            return 'WAIT', None, 'forced_wait'
        return super().slot(v, r, price)


def fork_key(B, pol):
    """A cheap identity of a fork state (a copy must equal its context; a reused arm state does not)."""
    return (B.k, len(B.E.fills), len(B.E.o), len(B.cmd_log), len(B.E.events), B.E.it, B.E.known, len(pol.log), len(pol.seen),
            hashlib.sha256(json.dumps([B.E.o[-3:], B.unclosed, B.E.fills[-3:]], sort_keys=True, default=str).encode()).hexdigest())


def fork_copy(ctx):
    """(branch, policy, observer) copied together; the read-only tape and cache are shared, never copied."""
    B, pol, obs = ctx; memo = {id(B.E.f): B.E.f, id(B.E.tr): B.E.tr, id(pol.cache): pol.cache, id(pol.edges): pol.edges}
    return copy.deepcopy(ctx, memo)


# ---------------------------------------------------------------------------------------------------------------------- per-arm accounting
def mid(f, t):
    return AR.mid_at(f, t)


def shown_at(f, side, price, act):
    """The activation queue recomputed from the frame rule (the last frame with venue time <= act; beyond the 5 shown levels -> 0)."""
    j = max(0, int(np.searchsorted(f[:, 0], act, side='right')) - 1); p = 2 + side * 10; last = f[j, p + 4]
    if last >= 0 and (price < last if side == 0 else price > last):
        return 0., j
    return float(sum(f[j, p + 5 + k] for k in range(5) if f[j, p + k] >= 0 and f[j, p + k] == price)), j


def fifo(fills0, I0, t0):
    """FIFO lots: the carried inventory opens one lot at the fork; -> (matched shares, quantity-weighted mean holding ms, open lots at END)."""
    lots = [] if abs(I0) < 1e-12 else [[np.sign(I0), abs(I0), t0]]; matched = held = 0.
    for t, s, q in fills0:
        d, q = (1. if s == 0 else -1.), float(q)
        while q > 1e-12 and lots and lots[0][0] == -d:
            x = min(q, lots[0][1]); matched += x; held += x * (t - lots[0][2]); lots[0][1] -= x; q -= x
            if lots[0][1] <= 1e-12:
                lots.pop(0)
        if q > 1e-12:
            lots.append([d, q, t])
    return matched, (held / matched if matched > 0 else np.nan), float(sum(x[0] * x[1] for x in lots))


def active_both(orders, fills, g0, end):
    """Fractions of [g0, END) with a venue-active order on side 0, side 1 and both (active: act <= t < remove, before its full fill)."""
    full = {}
    rem = {o['gid']: o['size'] for o in orders}
    for t, gid, _, q, _, _ in sorted(fills, key=lambda x: x[0]):
        rem[gid] -= q
        if rem[gid] <= 1e-9 and gid not in full:
            full[gid] = t
    iv = {0: [], 1: []}
    for o in orders:
        if o['refused']:
            continue
        a, b = max(o['act'], g0), min(o['remove'], full.get(o['gid'], np.inf), end)
        if b > a:
            iv[o['side']].append((a, b))
    pts = sorted({g0, end} | {x for s in (0, 1) for a, b in iv[s] for x in (a, b)}); tot = {0: 0., 1: 0., 'both': 0.}
    for a, b in zip(pts, pts[1:]):
        m_ = (a + b) / 2.; on = {s: any(x <= m_ < y for x, y in iv[s]) for s in (0, 1)}
        for s in (0, 1):
            tot[s] += (b - a) * on[s]
        tot['both'] += (b - a) * (on[0] and on[1])
    L_ = max(end - g0, 1e-9); return {k: v / L_ for k, v in tot.items()}


def reconciles(value, cash0, I0, M0, Q):
    """value_END - (cash_fork + I_fork M_fork) = Q (the carried inventory counted once)."""
    return bool(M0 is None or abs(value - (cash0 + I0 * M0) - Q) <= 1e-6 * max(1., abs(value)))


def sequence_axis(post, lin, focal, mark):
    """(class, early, later): the focal generation's first post-fork fill group (early) and the lineage fills after it (later), END-signed."""
    c = lambda x: SG[x[2]] * x[3] * (mark - x[4]); ff = sorted([x for x in post if x[1] == focal], key=lambda y: y[0])
    if not ff:
        return 'no_focal_fill', 0., 0.
    t1 = ff[0][0]; early = sum(c(x) for x in ff if x[0] == t1); later = sum(c(x) for x in post if x[1] in lin and x[0] > t1)
    return ('early_loss' if early < -TOL else 'no_early_loss') + '_' + ('later_benefit' if later > TOL else 'no_later_benefit'), early, later


def arm_result(B, pol, obs, st, end):
    """One arm's outcome and decomposition (header); the gates' raw facts."""
    Lg = LB.finish(B); value = float(Lg['value']); g = float(st['t']); f = B.E.f; mark = float(B.mark); fills = [tuple(x) for x in B.E.fills]
    pre = [x for x in fills if x[0] <= g]; post = [x for x in fills if x[0] > g]
    I0 = sum(SG[x[2]] * x[3] for x in pre); cash0 = -sum(SG[x[2]] * x[3] * x[4] for x in pre); M0 = mid(f, g)
    cmds = P.command_arrays(B)['commands'] if all(x[8] in P.REASONS for x in B.cmd_log) else None
    lin = {o['gid'] for o in B.E.o if (o['side'] == st['side'] and ((st['kind'] == 'REVIEW' and o['gid'] == st['gid']) or (o['rung'] == st['rung'] and o['submit'] >= g)))}
    c = lambda x: SG[x[2]] * x[3] * (mark - x[4])
    post_c = sum(c(x) for x in post); Q = post_c + (I0 * (mark - M0) if M0 is not None else np.nan)
    rec_ok = reconciles(value, cash0, I0, M0, Q)
    mids = [mid(f, x[0]) for x in post]; edge = sum(SG[x[2]] * x[3] * (m_ - x[4]) for x, m_ in zip(post, mids) if m_ is not None)
    move = (I0 * (mark - M0) if M0 is not None else np.nan) + sum(SG[x[2]] * x[3] * (mark - m_) for x, m_ in zip(post, mids) if m_ is not None)
    I, path_I, tw, peak = I0, [], 0., abs(I0); last = g
    for x in sorted(post, key=lambda y: y[0]):
        tw += abs(I) * (x[0] - last); I += SG[x[2]] * x[3]; last = x[0]; peak = max(peak, abs(I))
    tw += abs(I) * (end - last); matched, hold, open_lots = fifo([(x[0], x[2], x[3]) for x in sorted(post, key=lambda y: y[0])], I0, g)
    hz = {}
    for h in HORIZONS:
        vals = [(SG[x[2]] * x[3] * (m_ - x[4]) if (x[0] + h <= end and m_ is not None) else None) for x in post for m_ in [mid(f, x[0] + h)]]
        hz[h] = vals
    cc = [all(hz[h][i] is not None for h in HORIZONS) for i in range(len(post))]
    focal = st['gid'] if st['kind'] == 'REVIEW' else next((o['gid'] for o in B.E.o if o['side'] == st['side'] and o['rung'] == st['rung'] and o['submit'] == g), None)
    seq, early, later = sequence_axis(post, lin, focal, mark)
    reps = []
    for o in sorted(B.E.o, key=lambda o: o['submit']):
        if o['gid'] in lin and o['gid'] != st['gid'] or (st['kind'] == 'ADMISSION' and o['gid'] in lin):
            a0 = obs.ahead0.get(int(o['gid'])); want, j = shown_at(f, o['side'], o['price'], o['act'])
            reps.append(dict(gid=o['gid'], submit=o['submit'], act=o['act'], price=o['price'], side=o['side'], refused=bool(o['refused']), ahead0=None if a0 is None else a0[1],
                             shown=want, frame=j, book=[float(x) for x in f[j, 2 + o['side'] * 10:12 + o['side'] * 10]], delay=o['submit'] - g))
    q_ok = all(r['refused'] or r['ahead0'] is None or abs(r['ahead0'] - r['shown']) <= 1e-9 for r in reps)
    reqs = [e for e in B.E.events if e[1].endswith('_request') and e[0] >= g]; new = [o for o in B.E.o if o['submit'] >= g]
    rev = sum(1 for o in new if o['req'] is not None and o['why'] == 'policy' and o['req'] - o['act'] <= REVERSAL_MS)
    pp = [x for x in pol.log if x[10] == 'pressure_pull' and x[0] >= g]; viol = [x for x in pp if x[3] in lin]
    part = active_both([dict(gid=o['gid'], side=o['side'], act=o['act'], remove=o['remove'], size=o['size'], refused=o['refused']) for o in B.E.o], fills, g, end)
    return dict(value=value, Q=Q, I_fork=I0, cash_fork=cash0, M_fork=M0, mark=mark, post_fills=len(post), post_c=post_c, reconciled=rec_ok,
                focal_c=sum(c(x) for x in post if x[1] in lin), other_c=sum(c(x) for x in post if x[1] not in lin), edge=edge, movement=move, edge_missing=sum(m_ is None for m_ in mids),
                buy_q=sum(x[3] for x in post if x[2] == 0), sell_q=sum(x[3] for x in post if x[2] == 1), matched=matched, hold_ms=hold, open_lots=open_lots,
                tw_abs_I=tw / max(end - g, 1e-9), peak_abs_I=peak, terminal_I=I, both_sides=part['both'], side0=part[0], side1=part[1],
                requests=len(reqs), requests_policy=sum(1 for e in reqs if e[1] == 'policy_request'), reposts=len(new), reversals=rev, pressure_pulls=len(pp),
                lineage_pressure_pulls=len(viol), suppressed=len(pol.suppressed), applied=list(pol.applied), lineage=sorted(lin), seq=seq, early=early, later=later,
                replacements=reps, queue_reset_ok=bool(q_ok), first_rep_delay=next((r['delay'] for r in reps if r['gid'] != st['gid']), np.nan),
                hz={str(int(h)): dict(native=float(sum(v for v in hz[h] if v is not None)), n=int(sum(v is not None for v in hz[h])),
                                      common=float(sum(hz[h][i] for i in range(len(post)) if cc[i]))) for h in HORIZONS}, common_n=int(sum(cc)),
                pre_fills=[list(x) for x in pre], ledger=M.ledger_record(B, Lg), cmds=cmds, focal=focal, post_raw=[list(x) for x in post],
                pp_log=[[x[0], x[3]] for x in pp], suppressed_log=[list(x) for x in pol.suppressed])


def pair_result(st, R_, A_):
    """The pair's labels and gates (header)."""
    gate = dict(applied_once=bool(len(R_['applied']) == 1 and len(A_['applied']) == 1 and R_['applied'][0][4] == A_['applied'][0][4]),
                pre_fork_identical=bool(R_['pre_fills'] == A_['pre_fills']), reconciled=bool(R_['reconciled'] and A_['reconciled']),
                queue_reset=bool(R_['queue_reset_ok'] and A_['queue_reset_ok']), no_lineage_pull=bool(R_['lineage_pressure_pulls'] == 0 and A_['lineage_pressure_pulls'] == 0))
    status = 'ok' if all(gate.values()) else ('callback_not_reached' if not gate['applied_once'] else 'gate_failed')
    Y = R_['Q'] - A_['Q'] if (R_['M_fork'] is not None) else R_['post_c'] - A_['post_c']
    econ = 'retain_better' if Y > TOL else ('alternative_better' if Y < -TOL else 'tie')
    mech = ('retention_beneficial_with_early_loss' if R_['seq'].startswith('early_loss') else 'retention_beneficial_without_early_loss') if econ == 'retain_better' else econ
    return dict(day=st['day'], m=st['m'], kind=st['kind'], side=st['side'], gid=st['gid'], rung=st['rung'], t=st['t'], key=st['key'], status=status, Y=Y, Y_post=R_['post_c'] - A_['post_c'],
                Y_focal=R_['focal_c'] - A_['focal_c'], Y_other=R_['other_c'] - A_['other_c'], econ=econ, mechanism=mech, queue_mechanism='unidentified', seq_retain=R_['seq'],
                seq_alternative=A_['seq'], **{f'gate_{k}': v for k, v in gate.items()})


# ---------------------------------------------------------------------------------------------------------------------- census, inputs, the day run
def af_manifest():
    return json.loads((REPO / AF.RECEIPT / 'manifest.json').read_text())


def census(gm_af=None):
    """Every eligible REVIEW / ADMISSION state on the pilot markets (from the saved command arrays) with its key and the PER_CELL selection."""
    gm_af = gm_af or af_manifest(); rows = []
    for day in gm_af['days']:
        z = np.load(AF.OUT / 'pilot' / f'{day}_arrays.npz')
        for m in gm_af['parts']['pilot'][day]:
            cm = z[f'{int(m)}__{ARM}__commands']; lk = cm[(cm[:, 2] == LIVE) & (cm[:, 7] == KEEP) & (cm[:, 8] == KEEP_REASON)]
            for gid in np.unique(lk[:, 3]).astype(int):
                x = lk[lk[:, 3] == gid]; i = int(np.argmin(x[:, 0])); s = int(x[i, 1])
                rows.append(dict(day=day, m=int(m), kind='REVIEW', side=s, gid=int(gid), rung=float(x[i, 4]), t=float(x[i, 0]),
                                 key=hashlib.sha256(f'{ARM}|{day}|{int(m)}|REVIEW|{s}|{int(gid)}'.encode()).hexdigest()))
            sp = cm[(cm[:, 2] == SLOT) & (cm[:, 7] == POST) & (cm[:, 3] >= 0)]
            for x in sp:
                s, gid = int(x[1]), int(x[3])
                rows.append(dict(day=day, m=int(m), kind='ADMISSION', side=s, gid=gid, rung=float(x[4]), t=float(x[0]),
                                 key=hashlib.sha256(f'{ARM}|{day}|{int(m)}|ADMISSION|{s}|{float(x[0])}|{gid}'.encode()).hexdigest()))
    D_ = pd.DataFrame(rows); D_['n_cell'] = D_.groupby(['day', 'm', 'kind', 'side']).key.transform('size')
    D_['rank'] = D_.groupby(['day', 'm', 'kind', 'side']).key.rank(method='first').astype(int); D_['selected'] = D_['rank'] <= PER_CELL
    D_['p_draw'] = np.minimum(PER_CELL / D_.n_cell, 1.); return D_.sort_values(['day', 'm', 'kind', 'side', 'rank']).reset_index(drop=True)


def market_inputs(day, mans, gm_af):
    """The ASYNC day's inputs for markets mans (as pf_ladder_async_fast.run_day loads them) -> {m: (path, ident, cache)}, edges."""
    edges = P.frozen_edges(); _, pmk, pz = LG.parent_day(day, gm_af['parent']['day_records'][day]); man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text())
    members = [x for x in man_run['selected'] if x['day'] == day and int(x['m']) in mans]; need(sorted(int(x['m']) for x in members) == sorted(mans), f'{day}: a member is not in the run')
    prs, fair, depth, feat_fn, allw, opens = M.load_day(day, members, man_run); out = {}
    for x in members:
        m = int(x['m']); path, st, info = M.market_path(prs[m], float(x['ws']), opens[m][1]); need(st == 'complete' and info == pmk[m]['path_info'], f'{day} {m}: the path differs')
        t_abs0, end = float(info['t_start']), float(info['end_rel'])
        ident = dict(day=day, instrument_id=m, start_abs_ms=t_abs0, end_abs_ms=t_abs0 + end, end_rel=end, opening_ms=float(x['ws']), mark_source='QP.build_path supported-end mark',
                     tape_id=BE.tape_id(path['frames'], path['trades'], path['mark']))
        out[m] = (path, ident, LG.cache_of(pz, m))
    return out, edges


def saved_record(day, m):
    """The saved ASYNC CURRENT record and command array of one pilot market (hash-checked by the manifest)."""
    for ln in gzip.decompress((AF.OUT / 'pilot' / f'{day}.jsonl.gz').read_bytes()).decode().splitlines():
        r = json.loads(ln)
        if int(r['m']) == int(m):
            return r[ARM], np.load(AF.OUT / 'pilot' / f'{day}_arrays.npz')[f'{int(m)}__{ARM}__commands']
    raise ValueError(f'{day} {m}: no saved record')


def same_as_saved(ld, cmds, rec, cm):
    """The gate fields of a finished arm (its ledger record and command array) against the saved record."""
    return dict(fingerprint=ld['fingerprint'] == rec['fingerprint'], fills=ld['fills'] == rec['fills'], orders=ld['orders'] == rec['orders'], value=ld['value'] == rec['value'],
                requests=ld['requests'] == rec['requests'], commands=bool(np.array_equal(cmds, cm, equal_nan=True)))


def run_market(day, m, path, ident, cache, edges, states):
    """One market: the roll-in, a pair per state, the base completed to END (parity) -> (pairs, arms, parity, replacements)."""
    end = float(ident['end_rel']); B = AF.ProfileBranch(path, end, ident, False, PROFILE); pol = LineageGuard(B, cache, edges); obs = P.QueueObs(B, pol)
    rec, cm = saved_record(day, m); pairs, arms, reps, ledgers, k_prev = [], [], [], [], 0
    for st in sorted(states, key=lambda s: (s['t'], s['kind'], s['side'], s['gid'])):
        k = int(round(st['t'] / BE.STEP)); need(abs(k * BE.STEP - st['t']) < 1e-9 and k >= k_prev, f'{day} {m}: a state off the grid or out of order')
        LB.run_episode(B, pol, k0=k_prev, until=k, observe=obs); k_prev = k; ctx = (B, pol, obs); key0 = fork_key(B, pol); out = {}
        for name, action in zip(('retain', 'alternative'), ACTIONS[st['kind']]):
            Bx, px, ox = fork_copy(ctx); need(fork_key(Bx, px) == key0, 'a fork copy differs from its context'); px.arm(st, action)
            LB.run_episode(Bx, px, k0=k, observe=ox); out[name] = arm_result(Bx, px, ox, st, end)
        pr = pair_result(st, out['retain'], out['alternative']); R_ = out['retain']
        if R_['suppressed'] == 0:                                     # the exemption engaged and never fired: the retain arm IS the recorded path
            ss = same_as_saved(R_['ledger'], R_['cmds'], rec, cm) if R_['cmds'] is not None else dict(commands=False); pr['gate_unused_exemption'] = bool(all(ss.values()))
            pr['status'] = pr['status'] if pr['gate_unused_exemption'] else 'gate_failed'
        else:
            pr['gate_unused_exemption'] = None
        pr['retain_suppressed'] = R_['suppressed']; pr['alternative_suppressed'] = out['alternative']['suppressed']; pairs.append(pr)
        for name in ('retain', 'alternative'):
            a = {k_: v for k_, v in out[name].items() if k_ not in ('ledger', 'pre_fills', 'replacements', 'applied', 'hz', 'lineage', 'cmds', 'post_raw', 'pp_log', 'suppressed_log')}
            a.update({f'hz{h}_{x}': v for h, d in out[name]['hz'].items() for x, v in d.items()}); arms.append(dict(day=day, m=m, key=st['key'], kind=st['kind'], arm=name, **a))
            reps += [dict(day=day, m=m, key=st['key'], arm=name, **r) for r in out[name]['replacements']]
        ledgers.append(dict(key=st['key'], day=day, m=m, kind=st['kind'], **{name: {k_: out[name][k_] for k_ in ('value', 'Q', 'I_fork', 'cash_fork', 'M_fork', 'mark', 'lineage', 'focal',
                                                                                                             'applied', 'suppressed', 'pre_fills', 'post_raw', 'pp_log', 'suppressed_log',
                                                                                                             'replacements')}
                                                                               for name in ('retain', 'alternative')}))
    LB.run_episode(B, pol, k0=k_prev, observe=obs); Lb = LB.finish(B); ld = M.ledger_record(B, Lb); par = same_as_saved(ld, P.command_arrays(B)['commands'], rec, cm)
    par.update(guard_log=[list(x) for x in pol.log] == rec['guard_log']); par = dict(day=day, m=m, **par, all=bool(all(par.values())))
    ledgers.append(dict(key=f'BASE|{day}|{m}', day=day, m=m, kind='BASE', base=dict(fills=ld['fills'], orders=ld['orders'], value=ld['value'], fingerprint=ld['fingerprint'])))
    return pairs, arms, par, reps, ledgers


# ---------------------------------------------------------------------------------------------------------------------- manifest, parts, projection, measurement
VARIED_SHARE, VARIED_ABS_C = 0.10, 1000.


def pilot_records(gm_af):
    """sha256 of each pilot day record (its gzip / npz bytes = the record's own hashes)."""
    out = {}
    for day in gm_af['days']:
        p_ = AF.OUT / 'pilot' / f'{day}.json'; r = json.loads(p_.read_text())
        need(sha_b((AF.OUT / 'pilot' / f'{day}.jsonl.gz').read_bytes()) == r['sha256']['gz'] and sha_b((AF.OUT / 'pilot' / f'{day}_arrays.npz').read_bytes()) == r['sha256']['npz'],
             f'{day}: pilot bytes differ from the record')
        out[day] = sha_b(p_.read_bytes())
    return out


def build_manifest(path=None, census_path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); cpath = Path(census_path or REPO / RECEIPT / 'census.csv')
    need(not path.exists() and not cpath.exists(), 'the manifest / census exists (write-once)'); gm_af = af_manifest()
    need(M.CMD_KINDS.index('live') == LIVE and M.CMD_KINDS.index('slot') == SLOT and M.CMD_ACTIONS.index('POST') == POST and M.CMD_ACTIONS.index('KEEP') == KEEP
         and P.REASONS.index('keep') == KEEP_REASON, 'the command encoding differs from the declared one')
    D_ = census(gm_af); cpath.parent.mkdir(parents=True, exist_ok=True); D_.to_csv(cpath, index=False); hm = json.loads((REPO / 'predictfun/reviews/ladder_cancel_headroom_20261122/manifest.json').read_text())
    sel = D_[D_.selected]
    man = dict(version=VERSION, review='32d102bd', contract='reviews/ladder_long_continuation_contract_20261122/contract_v2.md', approval='USER 2026-11-22: approve 0.40 h / line 2.85',
               arm=ARM, profile=PROFILE, kinds=list(KINDS), per_cell=PER_CELL, tol=TOL, horizons=list(HORIZONS), reversal_ms=REVERSAL_MS, varied=dict(share=VARIED_SHARE, abs_c=VARIED_ABS_C),
               census_sha256=sha_b(cpath.read_bytes()), census_rows=int(len(D_)), selected=sel[['day', 'm', 'kind', 'side', 'gid', 'rung', 't', 'key']].to_dict('records'),
               af_manifest_sha256=sha_b((REPO / AF.RECEIPT / 'manifest.json').read_bytes()), pilot_records=pilot_records(gm_af), parts=dict(subset=[SUBSET_DAY], rest=[d for d in gm_af['days'] if d != SUBSET_DAY]),
               pilot_markets={d: [int(m) for m in v] for d, v in gm_af['parts']['pilot'].items()}, sources=sources(),
               line_receipts=hm['line_receipts'] + ['ladder_cancel_headroom_20261122'], line_reviews=hm['line_reviews'] + ['20261122_cancel_headroom_review'],
               caps=dict(batch_core_hours=CAP_BATCH, review_reserve=REVIEW_RESERVE, line_ceiling=LINE_CEILING, tail_s=TAIL_S, prior_unlogged_s=PRIOR_UNLOGGED_S),
               asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(part, day, out=None):
    d = Path(out or OUT) / part
    return dict(rec=d / f'{day}.json', pairs=d / f'{day}_pairs.csv', arms=d / f'{day}_arms.csv', reps=d / f'{day}_reps.csv', parity=d / f'{day}_parity.json',
                ledgers=d / f'{day}_ledgers.jsonl.gz')


def run_part(part, day, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date of one part: every selected state of its pilot markets -> pairs / arms / replacements / parity, write-once."""
    if require_frozen:
        need(frozen(), 'the continuation builder / selftest / declaration / manifest / launcher / checker are not the committed ones')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); need(gm['sources'] == sources(), 'a source differs from the frozen one')
    need(day in gm['parts'][part], f'{day} is not in the {part} part'); gm_af = af_manifest(); need(pilot_records(gm_af) == gm['pilot_records'], 'a pilot record differs from the manifest')
    pp = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json')
    need(part == 'subset' or (pp.exists() and json.loads(pp.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')
    P_ = day_paths(part, day, out); need(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    mans = [int(m) for m in gm['pilot_markets'][day]]; inp, edges = market_inputs(day, mans, gm_af); pairs, arms, par, reps, lgs = [], [], [], [], []
    for m in mans:
        path, ident, cache = inp[m]; states = [dict(s, day=day) for s in gm['selected'] if s['day'] == day and int(s['m']) == m]
        a_, b_, c_, d_, e_ = run_market(day, m, path, ident, cache, edges, states); pairs += a_; arms += b_; par.append(c_); reps += d_; lgs += e_
        need(AF.LS.hwm() < 1450., f'{day} {m}: memory above the self-guard')
    pd.DataFrame(pairs).to_csv(P_['pairs'], index=False); pd.DataFrame(arms).to_csv(P_['arms'], index=False); pd.DataFrame(reps).to_csv(P_['reps'], index=False)
    P_['parity'].write_text(json.dumps(par, indent=1, default=str) + '\n')
    P_['ledgers'].write_bytes(gzip.compress('\n'.join(json.dumps(x, default=str) for x in lgs).encode(), mtime=0))
    P_['rec'].write_text(json.dumps(dict(version=VERSION, part=part, day=day, markets=mans, pairs=len(pairs), sha256={k: sha_b(P_[k].read_bytes()) for k in ('pairs', 'arms', 'reps', 'parity', 'ledgers')},
                                         sources=sources(), elapsed_s=time.time() - t0, hwm_mib=AF.LS.hwm(), asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1) + '\n')
    print(json.dumps(dict(part=part, day=day, pairs=len(pairs), parity_all=all(x['all'] for x in par), statuses=pd.Series([x['status'] for x in pairs]).value_counts().to_dict()), indent=1))


def cpu_of(globs):
    import glob as G
    import re
    rl = re.compile(r'^\S+ tag=\S+ rc=\d+ peak_rss_mib=(\d+) wall_s=[\d.]+ cpu_user_s=([\d.]+)'); tot, peak = 0., 0
    for p in sorted({x for gb in globs for x in G.glob(gb, recursive=True)}):
        for ln in Path(p).read_text(errors='replace').splitlines():
            mm = rl.match(ln)
            if mm:
                tot += float(mm.group(2)); peak = max(peak, int(mm.group(1)))
    return tot / 3600., peak


def ledger(gm):
    """(this batch's logged core-hours, the line before this batch incl. the prior declared allowance, this batch's peak)."""
    own, peak = cpu_of([str(REPO / RECEIPT / 'logs' / '**' / '*.log')])
    line, _ = cpu_of([str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log') for n in gm['line_receipts']])
    rev, _ = cpu_of([str(REPO / 'predictfun/reviews/automatic' / n / '*.log') for n in gm['line_reviews']])
    return own, line + rev + gm['caps']['prior_unlogged_s'] / 3600., peak


def project(path=None):
    """The write-once projection: spent + the measured subset (its 2 markets) x 3 for the 6 rest markets + the tail <= the batch cap net of the
    review reserve; the line + spent + projected + the reserve <= the ceiling; the peak below the 1,536 MiB checkpoint."""
    path = Path(path or REPO / RECEIPT / 'logs' / 'projection.json'); need(not path.exists(), 'the projection exists (write-once)')
    gm = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); own, line, peak = ledger(gm); sub, _ = cpu_of([str(REPO / RECEIPT / 'logs' / 'lc_subset.log')])
    need(sub > 0, 'the subset has not run'); cp = gm['caps']; rest = 3. * sub; proj = own + rest + cp['tail_s'] / 3600.
    res = dict(spent=own, subset_core_hours=sub, projected_rest=rest, projected_batch=proj, line_before=line, projected_line=line + proj + cp['review_reserve'], peak_mib=peak,
               run_rest=bool(proj + cp['review_reserve'] <= cp['batch_core_hours'] and line + proj + cp['review_reserve'] <= cp['line_ceiling'] and peak < 1536))
    path.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)); return res


SUM_COLS = ('Y', 'Y_post', 'Y_focal', 'Y_other')


def summarize(Pr, Ar):
    """summary.csv by kind x day x side, kind and all (ok pairs; statuses counted on all) and the declared reading."""
    rows = []
    for ks in (('kind', 'day', 'side'), ('kind', 'day'), ('kind',), ()):
        groups = Pr.groupby(list(ks)) if ks else [((), Pr)]
        for k, g in groups:
            k = k if isinstance(k, tuple) else (k,); d = dict(zip(ks, k)); d.setdefault('kind', 'all'); d.setdefault('day', 'all'); d.setdefault('side', 'all'); d['side'] = str(d['side'])
            o = g[g.status == 'ok']; d.update(pairs=int(len(g)), ok=int(len(o)), **{f'status_{s}': int((g.status == s).sum()) for s in ('callback_not_reached', 'gate_failed')})
            d.update({f'econ_{e}': int((o.econ == e).sum()) for e in ('retain_better', 'alternative_better', 'tie')})
            d.update({f'mech_{e}': int((o.mechanism == e).sum()) for e in ('retention_beneficial_with_early_loss', 'retention_beneficial_without_early_loss')})
            d.update({f'seq_{e}': int((o.seq_retain == e).sum()) for e in ('no_focal_fill', 'early_loss_later_benefit', 'early_loss_no_later_benefit', 'no_early_loss_later_benefit',
                                                                         'no_early_loss_no_later_benefit')})
            d.update({f'{c}_sum': float(o[c].sum()) for c in SUM_COLS}); d.update(Y_mean=float(o.Y.mean()) if len(o) else np.nan, Y_median=float(o.Y.median()) if len(o) else np.nan,
                                                                               Y_abs_sum=float(o.Y.abs().sum()))
            a = Ar.merge(o[['key']], on='key')
            for arm in ('retain', 'alternative'):
                x = a[a.arm == arm]
                d.update({f'{arm}_{c}': float(x[c].mean()) if len(x) else np.nan for c in ('Q', 'edge', 'movement', 'buy_q', 'sell_q', 'matched', 'hold_ms', 'tw_abs_I', 'peak_abs_I',
                                                                                              'terminal_I', 'both_sides', 'requests', 'reposts', 'reversals', 'pressure_pulls', 'suppressed',
                                                                                              'first_rep_delay', 'common_n')})
                d.update({f'{arm}_hz{h}_{x_}': float(x[f'hz{h}_{x_}'].sum()) for h in ('1000', '5000', '30000', '60000') for x_ in ('native', 'common')})
            rows.append(d)
    T = pd.DataFrame(rows); allr = T[(T.kind == 'all') & (T.day == 'all')].iloc[0]
    reliable = bool(allr.status_callback_not_reached == 0 and allr.status_gate_failed == 0)
    varied = bool(allr.ok > 0 and allr.econ_retain_better >= VARIED_SHARE * allr.ok and allr.econ_alternative_better >= VARIED_SHARE * allr.ok and allr.Y_abs_sum >= VARIED_ABS_C)
    return T, dict(reliable=reliable, varied=varied, reading='reliable_and_varied' if (reliable and varied) else ('reliable_not_varied' if reliable else 'not_reliable'))


def measure(out=None, require_frozen=True, man_path=None):
    """All parts -> pairs / arms / replacements / parity / summary .csv and audit.json (write-once)."""
    if require_frozen:
        need(frozen(), 'the continuation builder / selftest / declaration / manifest / launcher / checker are not the committed ones')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); od = Path(out or OUT); md = od / 'measure'; need(not (md / 'audit.json').exists(), 'the audit exists (write-once)')
    md.mkdir(parents=True, exist_ok=True); tabs = dict(pairs=[], arms=[], reps=[]); par = []
    for part, days in gm['parts'].items():
        for day in days:
            P_ = day_paths(part, day, out); need(P_['rec'].exists(), f'{part} {day}: missing'); r = json.loads(P_['rec'].read_text())
            need(all(r['sha256'][k] == sha_b(P_[k].read_bytes()) for k in ('pairs', 'arms', 'reps', 'parity', 'ledgers')), f'{part} {day}: bytes differ from the record')
            for k in tabs:
                tabs[k].append(pd.read_csv(P_[k], dtype={'day': str}) if P_[k].stat().st_size > 1 else pd.DataFrame())
            par += json.loads(P_['parity'].read_text())
    Pr, Ar, Rp = (pd.concat(tabs[k], ignore_index=True) for k in ('pairs', 'arms', 'reps')); need(len(Pr) == len(gm['selected']), 'the pairs are not the selected states')
    T, rd = summarize(Pr, Ar); rd['parity_all'] = bool(all(x['all'] for x in par)); rd['reliable'] = bool(rd['reliable'] and rd['parity_all'])
    if not rd['reliable']:
        rd['reading'] = 'not_reliable'
    for n, D_ in (('pairs.csv', Pr), ('arms.csv', Ar), ('replacements.csv', Rp), ('summary.csv', T), ('parity.csv', pd.DataFrame(par))):
        D_.to_csv(md / n, index=False)
    (md / 'audit.json').write_text(json.dumps(dict(version=VERSION, reading=rd, pairs=int(len(Pr)), outputs={n: sha_b((md / n).read_bytes()) for n in ('pairs.csv', 'arms.csv', 'replacements.csv',
                                                                                                                                                       'summary.csv', 'parity.csv')},
                                                   sources=sources(), asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1) + '\n')
    print(json.dumps(rd, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_long_continuation_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps({k: v for k, v in build_manifest().items() if k != 'selected'}, indent=1)[:2000])
    elif cmd == 'subset':
        run_part('subset', SUBSET_DAY)
    elif cmd == 'project':
        project()
    elif cmd == 'rest':
        run_part('rest', sys.argv[2])
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [None]))
    else:
        sys.exit('usage: pf_ladder_long_continuation.py selftest | manifest | subset | project | rest <day> | measure [out]')
