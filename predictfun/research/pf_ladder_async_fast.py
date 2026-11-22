"""ASYNC-EXECUTION-SENSITIVITY-v1 (P-2026-005; 2026-11-22, session aea5bfb9). Review b8ab077a (target c7404f21) accepted QUEUE-PRIORITY-SENSITIVITY-v1
(MIXED, every arm negative) and named ONE next priority: the two ASYNC_FAST10 cases of the USER's concurrency amendment c3fe1d4a (superseding the
FAST10 notice changes of 9f7fe703), current queue and the identical one-time front treatment, reusing that round's two baseline ledgers. The USER
approved it on 2026-11-22 with <= 0.25 additional core-hours for the whole batch and a 2.50 ladder-line ceiling including review. A hypothetical
execution sensitivity on the 19 complete consumed Oct 24-27 markets: not a measured achievable latency, an executable contract, an upper bound,
a validation or a profitability claim; no new data, threshold, size, cap or parameter search; POSITIVE-PRESSURE-GUARD-v1 stays stopped.
PROFILES (ProfileEngine / ProfileBranch; their source sha256 in the manifest): BASELINE = the frozen timing (activation +58 ms, live cancel effect
 +37 ms, pending-cancel minimum submit + 95 ms) and the frozen slot model; ASYNC_FAST10 = activation +10 ms, live cancel effect +10 ms, a
 cancel of a still-pending post at max(request + 10, submit + 20) (the per-order race rule, never a global wait), and the DESIRED-LEVELS slot
 model: a cancel-pending (retiring) generation no longer occupies one of the side's two desired quote slots, so an independent POST may be sent
 while its cancel is in flight, but it stays unclosed until its acknowledgement: its known remaining stays reserved (the 200-share cap and the
 desired-exposure sizing unchanged: no early release, no overlap allowance), its price stays locked (no duplicate proposal) and own-cross
 protection still sees it; the per-side invariant counts desired generations. Unchanged in both: fill notice +100 ms, cancel acknowledgement
 +100 ms after the effect, post notice +0 ms, the 50 ms grid, the received Predict / Binance clocks and cache (no event is exposed sooner), the
 policy (the frozen PressureGuard), trades-before-commands ties, inclusive activation and strict removal. The near-END feasibility uses the
 profile's activation; the branch settings record the effective timing and slot model (BASELINE keeps the frozen settings for parity).
ARMS: PARITY_BASELINE (BASELINE profile, current queue, through the wrapper; pilot markets only) must equal the accepted pressure arm on every
 pilot market (fingerprint, fills, orders, requests, value, guard log, command array, queue records); CURRENT_ASYNC_FAST10 (ASYNC_FAST10, the
 engine's current queue); FRONT_ASYNC_FAST10 (ASYNC_FAST10, pf_ladder_queue_front's one-time zero external queue at activation). Reused,
 hash-bound: CURRENT_BASELINE = QUEUE-PRIORITY-SENSITIVITY-v1's CURRENT_QUEUE and FRONT_BASELINE = its FRONT_AT_ACTIVATION records.
PARTS / GATE: pilot = the first two complete markets of each date (reused in the result); rest = the other 11, only if the write-once projection
 fits (this batch's spent + the projected rest + a 60 s tail <= 0.25; the line + those + a 0.03 review reserve <= 2.50).
MEASUREMENT (measure): per arm the frozen arm_row (its withholds_near_end label uses the frozen 58 ms window for every arm), the class table
 (pf_ladder_queue_front.class_rows), the treatment census and blocks(): per market and arm, from the saved orders and closures, the desired
 quote occupancy (end-of-tick desired = unclosed and not retiring, capped at the levels, over 2 x levels x ticks), retiring order-ticks, and
 the cap / near_end / own_cross / unavailable_price constraints with the cap refusals made while that side had a retiring generation.
READING (predicates; declared): contrasts EXEC_AT_CURRENT = CURRENT_ASYNC - CURRENT_BASELINE, EXEC_AT_FRONT = FRONT_ASYNC - FRONT_BASELINE,
 QUEUE_AT_FAST = FRONT_ASYNC - CURRENT_ASYNC and INTERACTION = FRONT_ASYNC - CURRENT_ASYNC - FRONT_BASELINE + CURRENT_BASELINE, each with the
 END and eligible +1 s totals per day (POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED) and economic IMPROVES / WORSENS / MIXED; every arm's
 absolute END per day and pooled; reading = nonnegative_development_result if any arm's pooled END >= 0, else assumptions_do_not_rescue. The
 execution contrasts include concurrency and cannot isolate wire latency; a mixed or negative result starts no tuned speed / queue variant; a
 nonnegative one would motivate timing feasibility and separately approved untouched validation, not promotion. Point estimates (4 days).
Usage (run_light): pf_ladder_async_fast.py selftest | manifest | pilot <day> | project | rest <day> | measure [out]"""
import glob
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_queue_front as QF                        # noqa: E402  frozen (b4ecbc34): FrontLadderEngine, class_rows, fill_classes, cpu, the parent helpers

AB, LG, P, M, C, pd, LB, BE, LS, A = QF.AB, QF.LG, QF.P, QF.M, QF.C, QF.pd, QF.LB, QF.BE, QF.LS, QF.A
VERSION = 'async_execution_sensitivity_v1'
PROFILES = dict(BASELINE=dict(place=float(BE.PLACE), cancel=float(BE.CANCEL), pending=float(BE.PENDING), slot_model='frozen'),
                ASYNC_FAST10=dict(place=10., cancel=10., pending=20., slot_model='desired_levels'))
NOTICES = dict(fill_notice=float(BE.FILL_NOTICE), cancel_ack=float(BE.CANCEL_ACK), post_notice=float(BE.POST_NOTICE))
RUN_ARMS = dict(PARITY_BASELINE=(False, 'BASELINE'), CURRENT_ASYNC_FAST10=(False, 'ASYNC_FAST10'), FRONT_ASYNC_FAST10=(True, 'ASYNC_FAST10'))
REUSED = dict(CURRENT_BASELINE='CURRENT_QUEUE', FRONT_BASELINE='FRONT_AT_ACTIVATION')
MEASURED = ('CURRENT_BASELINE', 'FRONT_BASELINE', 'CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')
PARTS = ('pilot', 'rest')
CAP_BATCH, LINE_CEILING, REVIEW_RESERVE, TAIL_S = 0.25, 2.50, 0.03, 60.
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_async_fast_20261122'
OUT = A.DATA / 'ladder_async_fast_v1_20261122'
QF_OUT = A.DATA / 'ladder_queue_front_v1_20261122'
SOURCES = ('pf_ladder_async_fast.py', 'pf_ladder_async_fast_selftest.py', 'pf_ladder_queue_front.py') + AB.SOURCES
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_async_fast.sh')
LINE = QF.LINE + ['ladder_queue_front_20261122']
REVIEWS = QF.REVIEWS + ['20261122_queue_latency_amendment', '20261122_async_execution_amendment', '20261122_queue_front_review']


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_async_fast.py', 'pf_ladder_async_fast_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the profiles
class ProfileEngine(QF.FrontLadderEngine):
    """The frozen engine with the queue-front flag and a declared timing profile (header)."""

    def __init__(self, path, end, front, prof):
        super().__init__(path, end, front); self.prof = dict(prof)

    def add_order(self, side, price, d, q, rung):
        return self.add9([side, float(rung), price, float(q), 0., d + self.prof['place'], 1e30, d, 1.])

    def request(self, gid, g, why):
        o = self.o[gid]; arr = max(g + self.prof['cancel'], o['submit'] + self.prof['pending']) if g < o['act'] else g + self.prof['cancel']
        if arr < o['remove']:
            o['remove'] = arr
        if o['req'] is None:
            o['req'] = float(g); o['why'] = why
        self.events.append((float(g), f'{why}_request', gid, dict(effect=o['remove'])))


class ProfileBranch(LB.LadderBranch):
    """The frozen ladder client with a declared profile (header): the timing in its engine, the near-END check and the slot model."""

    def __init__(self, path, end, ident, front, profile):
        super().__init__(path, LB.LADDER, end=end, episode=0, ident=ident)
        C.require(not self.E.o and not self.E.fills and self.E.it == 0, 'the engine can only be replaced before any order or trade')
        self.profile = profile; self.prof = dict(PROFILES[profile]); self.E = ProfileEngine({'frames': self.E.f, 'trades': self.E.tr}, self.end, front, self.prof)
        if profile != 'BASELINE':
            self.settings = dict(self.settings, contract='branch_ladder_async_v1', activation_ms=self.prof['place'], cancel_effect_ms=self.prof['cancel'],
                                 pending_cancel_ms=self.prof['pending'], slot_model=self.prof['slot_model'], profile=profile)

    def desired(self, s):
        return [gid for gid in self.unclosed[s] if self.E.o[gid]['req'] is None]

    def side_view(self, g, s):
        v = super().side_view(g, s)
        if self.prof['slot_model'] == 'desired_levels':
            v['vacancy'] = self.levels - len(self.desired(s))
        return v

    def check_position(self, g):
        if self.prof['slot_model'] == 'frozen':
            return super().check_position(g)
        Ik, Bk, Ak = self.reserve(g); Pp = sum((1. if s == 0 else -1.) * q for t, _, s, q, _, _ in self.E.fills if t <= g)
        C.require(Ik + Bk <= self.cap + LB.TOL and -Ik + Ak <= self.cap + LB.TOL, f'reservation inequality at {g}: I {Ik}, B {Bk}, A {Ak}, cap {self.cap}')
        C.require(Ik - Ak - LB.TOL <= Pp <= Ik + Bk + LB.TOL, f'position check at {g}: engine {Pp} outside [{Ik - Ak}, {Ik + Bk}]')
        C.require(all(len(self.desired(s)) <= self.levels for s in (0, 1)), f'more than {self.levels} desired generations on a side at {g}')

    def side_commands(self, g, s, pol):
        """pf_branch_ladder.LadderBranch.side_commands with the profile's activation in the near-END check (the slot model enters via side_view)."""
        C.require(s not in self.acted, f'side {s} already processed at {g}'); self.acted.add(s); v = self.side_view(g, s)
        if not v['book_ok']:
            for o in v['orders']:
                if o['state'] == 'live':
                    self.fallbacks.append((float(g), s, 'live', 'feature_unavailable')); self.decisions.append((float(g), s, 'live', 'KEEP', o['gid']))
                    self._log(g, s, 'live', o['gid'], o['rung'], o['price'], None, 'KEEP', 'fallback')
            if v['vacancy'] > 0:
                self.fallbacks.append((float(g), s, 'absent', 'feature_unavailable')); self.decisions.append((float(g), s, 'absent', 'WAIT', None))
                self._log(g, s, 'slot', None, None, None, None, 'WAIT', 'fallback')
            self.check_position(g); return
        for o in [x for x in v['orders'] if x['state'] == 'live']:
            a, why = pol.live(self.side_view(g, s), o); C.require(a in ('KEEP', 'PULL'), f'a live action {a}')
            if a == 'PULL':
                self.E.request(o['gid'], g, 'policy')
            self.decisions.append((float(g), s, 'live', a, o['gid'])); self._log(g, s, 'live', o['gid'], o['rung'], o['price'], None, a, why); self.check_position(g)
        best = v['bid'] if s == 0 else v['ask']
        for r in range(self.levels):
            v = self.side_view(g, s)
            if v['vacancy'] <= 0:
                break
            price = best - LB.TICK * r if s == 0 else best + LB.TICK * r
            if any(o['price'] == price for o in v['orders']):
                continue
            if not 1 <= price <= 99:
                self.constraints.append((float(g), s, 'unavailable_price')); self._log(g, s, 'slot', None, r, price, None, 'WAIT', 'unavailable_price'); continue
            if self.own_cross and any((price >= p_ if s == 0 else price <= p_) for p_ in v['opp_prices']):
                self.constraints.append((float(g), s, 'own_cross_blocked')); self._log(g, s, 'slot', None, r, price, None, 'WAIT', 'own_cross_blocked'); continue
            a, q, why = pol.slot(v, r, price); gid = None
            if a == 'POST':
                C.require(LB.QMIN <= q <= self.q0 and q == math.floor(q), f'a POST quantity {q} outside the whole shares [{LB.QMIN}, {self.q0}]')
                room = (self.cap - v['position'] - v['B']) if s == 0 else (self.cap + v['position'] - v['A'])
                if q > room + LB.TOL:
                    self.constraints.append((float(g), s, 'cap')); a, why = 'WAIT', 'cap'
                elif g + self.prof['place'] >= self.end:
                    self.constraints.append((float(g), s, 'near_end')); a, why = 'WAIT', 'near_end'
                else:
                    gid = self.E.add_order(s, price, g, q, r); self.unclosed[s].append(gid)
            else:
                C.require(a == 'WAIT', f'a slot action {a}')
            self.decisions.append((float(g), s, 'absent', a, gid)); self._log(g, s, 'slot', gid, r, price, q, a, why); self.check_position(g)


def profile_sha():
    return hashlib.sha256((inspect.getsource(ProfileEngine) + inspect.getsource(ProfileBranch)).encode()).hexdigest()


def run_arm(path, ident, cache, edges, arm):
    """One arm of one market on a fresh profile branch -> (branch, ledger record, policy, observer)."""
    front, prof = RUN_ARMS[arm]; B = ProfileBranch(path, ident['end_rel'], ident, front, prof); pol = P.PressureGuard(B, cache, edges); obs = P.QueueObs(B, pol)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    rec['missing'] = {f'{s}|{st}': int(n) for (s, st), n in pol.missing.items()}; rec['queue_venue'] = obs.queue(); rec['fill_class'] = QF.fill_classes(B.E)
    rec['treatment'] = dict(front=B.E.front, accepted_new=B.E.accepted_new, front_set=B.E.front_set); rec['profile'] = dict(name=prof, **PROFILES[prof], **NOTICES)
    return B, rec, pol, obs


# ---------------------------------------------------------------------------------------------------------------------- manifest and the day run
def reused_records():
    """The QUEUE-PRIORITY-SENSITIVITY-v1 day records and their bytes' sha256 (bound in this manifest)."""
    out = {}
    for part in PARTS:
        for p_ in sorted((QF_OUT / part).glob('*.json')):
            if p_.name.endswith('_status.json'):
                continue
            rec = json.loads(p_.read_text()); g_, n_ = p_.with_suffix('.jsonl.gz'), p_.parent / (p_.stem + '_arrays.npz')
            C.require(rec['status'] == 'DAY_DONE' and C.sha(g_) == rec['sha256']['gz'] and C.sha(n_) == rec['sha256']['npz'], f'{p_}: a reused record differs from itself')
            out[f'{part}/{p_.stem}'] = C.sha(p_)
    return out


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    qm = json.loads((REPO / QF.RECEIPT / 'manifest.json').read_text()); C.require(M.committed_equal(f'{QF.RECEIPT}/manifest.json', REPO / QF.RECEIPT / 'manifest.json'), 'the queue-front manifest')
    man = dict(version=VERSION, review='b8ab077a', amendment='c3fe1d4a', approval='USER 2026-11-22: approve ASYNC_FAST10 (<= 0.25 / line 2.50)', profile_sha256=profile_sha(),
               treatment_sha256=QF.treatment_sha(), profiles=PROFILES, notices=NOTICES, run_arms={k: list(v) for k, v in RUN_ARMS.items()}, reused=REUSED,
               reused_records=reused_records(), queue_front_manifest_sha256=C.sha(REPO / QF.RECEIPT / 'manifest.json'), parent=qm['parent'], edges=qm['edges'],
               edges_sha256=qm['edges_sha256'], days=qm['days'], complete=qm['complete'], parts=qm['parts'], sources=sources(),
               caps=dict(batch_core_hours=CAP_BATCH, line_ceiling=LINE_CEILING, review_reserve=REVIEW_RESERVE), line_receipts=LINE, line_reviews=REVIEWS,
               multiplicity=dict(new_sensitivities=['CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10'], controls=['PARITY_BASELINE (pilot)'], reused=list(REUSED)), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, part, out=None):
    return AB.day_paths(day, part, out or OUT)


def run_day(day, part, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date of one part: PARITY_BASELINE (pilot only) and the two ASYNC_FAST10 arms -> records and arrays, write-once."""
    C.require(day in M.DAYS, f'{day} is not a declared day'); C.require(part in PARTS, f'{part} is not a declared part')
    if require_frozen:
        C.require(frozen(), 'the async runner / selftest / declaration / manifest / launcher are not the committed ones')
    man_path = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(man_path.read_text())
    C.require(gm['profile_sha256'] == profile_sha() and gm['treatment_sha256'] == QF.treatment_sha(), 'the profile or the treatment differs from the frozen one')
    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json')
    C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')
    P_ = day_paths(day, part, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True)
    mans = [int(m) for m in gm['parts'][part][day]]; t0 = time.time(); C.require(mans, f'{day}: no members in the {part} part')
    arms = list(RUN_ARMS) if part == 'pilot' else ['CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10']
    edges = P.frozen_edges(); _, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day]); man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text())
    base = {int(r['m']): r for r in P.gz_records(M.day_paths(day, False)['gz'])}
    members = [x for x in man_run['selected'] if x['day'] == day and int(x['m']) in mans]; tim = dict(load=None, markets=[])
    C.require(sorted(int(x['m']) for x in members) == sorted(mans), f'{day}: a member is not in the frozen run')
    G = LS.Guard(1200. + 900. * max(1, len(members)), 900.); mon = LS.RN.Sampler(P_['status'], LS.GUARD); mon.start()
    try:
        prs, fair, depth, feat_fn, allw, opens = M.load_day(day, members, man_run); tim['load'] = time.time() - t0; G.check('load')
        recs, arrs = [], {}
        for x in members:
            tm = time.time(); m = int(x['m']); b, pr_ = base[m], pmk[m]; W_ = prs[m]; path, st, info = M.market_path(W_, float(x['ws']), opens[m][1])
            C.require(st == 'complete' and info == b['path_info'] == pr_['path_info'], f'{day} {m}: the path differs from the baseline / pressure records')
            t_abs0, end = float(info['t_start']), float(info['end_rel'])
            ident = dict(day=day, instrument_id=m, start_abs_ms=t_abs0, end_abs_ms=t_abs0 + end, end_rel=end, opening_ms=float(x['ws']),
                         mark_source='QP.build_path supported-end mark', tape_id=BE.tape_id(path['frames'], path['trades'], path['mark']))
            cache = LG.cache_of(pz, m); out_m = dict(day=day, m=m, path_info=info)
            for arm in arms:
                B, rec, pol, obs = run_arm(path, ident, cache, edges, arm); cm = P.command_arrays(B); rec['queue_known'] = M.queue_summary(W_, rec, cm['commands'], t_abs0)
                out_m[arm] = rec; arrs.update({f'{m}__{arm}__{k}': v for k, v in dict(obs.arrays(), **cm).items()})
            if 'PARITY_BASELINE' in arms:
                z, pp = out_m['PARITY_BASELINE'], pr_['pressure']
                out_m['parity'] = dict(fingerprint=z['fingerprint'] == pp['fingerprint'], fills=z['fills'] == pp['fills'], value=z['value'] == pp['value'], orders=z['orders'] == pp['orders'],
                                       requests=z['requests'] == pp['requests'], guard_log=z['guard_log'] == pp['guard_log'],
                                       commands=bool(np.array_equal(arrs[f'{m}__PARITY_BASELINE__commands'], pz[f'{m}__pressure__commands'], equal_nan=True)),
                                       queue=z['queue_venue'] == pp['queue_venue'] and z['queue_known'] == pp['queue_known'])
                out_m['parity_all'] = bool(all(out_m['parity'].values()))
            recs.append(out_m); tim['markets'].append(dict(m=m, s=time.time() - tm, arms=len(arms))); G.check('market', time.time() - tm)
        b_ = LS.gz_bytes(recs); P_['gz'].write_bytes(b_); np.savez_compressed(P_['npz'], **arrs)
        LS.RN.atomic_json(P_['rec'], dict(version=VERSION, day=day, part=part, members=mans, arms=arms, status='DAY_DONE', sha256=dict(gz=hashlib.sha256(b_).hexdigest(), npz=C.sha(P_['npz'])),
                                          manifest_sha256=C.sha(man_path), profile_sha256=profile_sha(), treatment_sha256=QF.treatment_sha(), sources=sources(), timing=tim,
                                          elapsed_s=time.time() - t0, hwm_mib=LS.hwm(), asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P_['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- the projection gate
def line_used(root=None):
    own = QF.cpu(glob.glob(str((root or REPO / RECEIPT) / 'logs' / '**' / '*.log'), recursive=True))[0]
    line = QF.cpu([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])[0]
    rev = QF.cpu([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])[0]; return own, own + line + rev


def project(out=None, man_path=None, proj_path=None, root=None):
    """The write-once projection after every pilot date -> run_rest (the rest markets run 2 arms; the pilot's 3-arm market time is conservative)."""
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json'); C.require(not proj_path.exists(), 'the projection exists (write-once)')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); recs = {}
    for d in gm['days']:
        p_ = day_paths(d, 'pilot', out)['rec']; C.require(p_.exists(), f'{d}: the pilot is not done'); recs[d] = json.loads(p_.read_text())
        C.require(recs[d].get('status') == 'DAY_DONE', f'{d}: the pilot did not finish')
    loads = [r['timing']['load'] for r in recs.values()]; mkt = [x['s'] for r in recs.values() for x in r['timing']['markets']]
    jobs = QF.cpu(glob.glob(str((root or REPO / RECEIPT) / 'logs' / 'fa_pilot_*.log')))
    cpw = (jobs[0] * 3600.) / jobs[2] if jobs[2] > 0 else 1.; rest = {d: len(gm['parts']['rest'][d]) for d in gm['days']}
    rest_s = sum(float(np.mean(loads)) + n * float(np.mean(mkt)) for n in rest.values() if n) * cpw; spent, line = line_used(root)
    proj = (rest_s + TAIL_S) / 3600.; res = dict(pilot_load_s=loads, pilot_market_s_mean=float(np.mean(mkt)), cpu_per_wall=cpw, rest_markets=rest, projected_rest_cpu_s=rest_s, tail_s=TAIL_S,
                                                  batch_spent=spent, batch_projected=spent + proj, line_now=line, line_projected=line + proj + REVIEW_RESERVE, caps=gm['caps'],
                                                  run_rest=bool(spent + proj <= gm['caps']['batch_core_hours'] and line + proj + REVIEW_RESERVE <= gm['caps']['line_ceiling']))
    proj_path.parent.mkdir(parents=True, exist_ok=True); proj_path.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)); return res


# ---------------------------------------------------------------------------------------------------------------------- measurement
def blocks(day, m, arm, rec, end, levels=2):
    """Slot / risk diagnostics of one arm of one market from its saved orders and closures (header)."""
    nt = int(math.ceil(end / BE.STEP)); g = BE.STEP * np.arange(nt); des = np.zeros((nt, 2)); ret = np.zeros((nt, 2))
    closed = {int(k): float(v[1]) for k, v in rec['closed'].items()}
    for o in rec['orders']:
        s, sub, ct, req = int(o['side']), float(o['submit']), closed.get(int(o['gid']), np.inf), o['req']
        un = (g >= sub) & (g < ct); rt = un & (g >= float(req)) if req is not None else np.zeros(nt, bool); des[:, s] += un & ~rt; ret[:, s] += rt
    cons = [tuple(c) for c in rec['constraints']]; cap_ret = 0
    for t, s, why in cons:
        if why == 'cap':
            k = int(round(float(t) / BE.STEP)); cap_ret += int(0 <= k < nt and ret[k, int(s)] > 0)
    cnt = {w: sum(1 for c in cons if c[2] == w) for w in ('cap', 'near_end', 'own_cross_blocked', 'unavailable_price')}
    return dict(day=day, m=m, arm=arm, desired_occupancy=float(np.minimum(des, levels).sum() / (2 * levels * nt)), retiring_ticks=float(ret.sum()), cap_while_retiring=cap_ret, **cnt)


def arm_totals(R_, arm, day=None):
    g = R_[(R_.arm == arm) & ((R_.day == day) if day is not None else True)]; q = float(g.all_cut_valid_q.sum())
    return dict(END=float(g.value.sum()), mo1s=float(g.all_id_mo1s.sum()), valid_q=q, per_share=float(g.all_cut_mo1s.sum()) / q if q > 0 else np.nan, q=float(g.all_q.sum()))


CONTRASTS = dict(EXEC_AT_CURRENT=(('CURRENT_ASYNC_FAST10', 1.), ('CURRENT_BASELINE', -1.)), EXEC_AT_FRONT=(('FRONT_ASYNC_FAST10', 1.), ('FRONT_BASELINE', -1.)),
                 QUEUE_AT_FAST=(('FRONT_ASYNC_FAST10', 1.), ('CURRENT_ASYNC_FAST10', -1.)),
                 INTERACTION=(('FRONT_ASYNC_FAST10', 1.), ('CURRENT_ASYNC_FAST10', -1.), ('FRONT_BASELINE', -1.), ('CURRENT_BASELINE', 1.)))


def predicates(R_):
    """The declared readings (header)."""
    days = sorted(R_.day.unique()); arms = ('BASELINE',) + MEASURED
    byd = {d: {a: arm_totals(R_, a, d) for a in arms} for d in days}; tot = {a: arm_totals(R_, a) for a in arms}; con = {}
    for nm, terms in CONTRASTS.items():
        dE = {d: sum(w * byd[d][a]['END'] for a, w in terms) for d in days}; d1 = {d: sum(w * byd[d][a]['mo1s'] for a, w in terms) for d in days}
        eA, mA = AB.classify(dE.values()), AB.classify(d1.values())
        con[nm] = dict(END_by_day=dE, mo1s_by_day=d1, END_total=sum(w * tot[a]['END'] for a, w in terms), mo1s_total=sum(w * tot[a]['mo1s'] for a, w in terms), END_answer=eA,
                       mo1s_answer=mA, economic='IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else 'WORSENS' if eA == mA == 'NEGATIVE_EVERY_DAY' else 'MIXED')
    nonneg = [a for a in MEASURED if tot[a]['END'] >= 0]
    return dict(days=days, arm_totals=tot, arm_by_day=byd, contrasts=con, end_negative=dict(total={a: bool(tot[a]['END'] < 0) for a in arms},
                                                                                         by_day={d: {a: bool(byd[d][a]['END'] < 0) for a in arms} for d in days}),
                nonnegative_arms=nonneg, reading='nonnegative_development_result' if nonneg else 'assumptions_do_not_rescue',
                note='a combined execution-profile sensitivity (timing and concurrency together) on 19 consumed-day markets; not wire latency, an executable contract, '
                     'an upper bound, a validation or a profitability claim')


def load_reused(gm, day):
    """The reused baseline rows of a date, bytes checked against this manifest -> ({m: row}, {part: arrays})."""
    rows, arrs = {}, {}
    for part in PARTS:
        p_ = QF_OUT / part / f'{day}.json'
        if f'{part}/{day}' not in gm['reused_records']:
            continue
        C.require(C.sha(p_) == gm['reused_records'][f'{part}/{day}'], f'{day} {part}: a reused record differs from the manifest'); rec = json.loads(p_.read_text())
        b = (QF_OUT / part / f'{day}.jsonl.gz').read_bytes(); C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(QF_OUT / part / f'{day}_arrays.npz') == rec['sha256']['npz'],
                                                                   f'{day} {part}: reused bytes')
        arrs[part] = np.load(QF_OUT / part / f'{day}_arrays.npz'); rows.update({int(r['m']): dict(r, part=part) for r in LS.gz_rows(b)})
    return rows, arrs


def measure(out=OUT, man_path=None, parts=PARTS):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); rows, par, kr, tr_, bl, seen = [], [], [], [], [], set()
    for day in gm['days']:
        _, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day]); ru, ra = load_reused(gm, day)
        for part in parts:
            if not gm['parts'][part][day]:
                continue
            P_ = day_paths(day, part, out); rec = json.loads(P_['rec'].read_text()); b = P_['gz'].read_bytes()
            C.require(rec['profile_sha256'] == gm['profile_sha256'] and rec['members'] == [int(x) for x in gm['parts'][part][day]] and hashlib.sha256(b).hexdigest() == rec['sha256']['gz']
                      and C.sha(P_['npz']) == rec['sha256']['npz'], f'{day} {part}: identity or members')
            z = np.load(P_['npz'])
            for r in LS.gz_rows(b):
                m = int(r['m']); end = float(r['path_info']['end_rel']); C.require((day, m) not in seen, f'{day} {m}: a market twice'); seen.add((day, m))
                if 'parity' in r:
                    par.append(dict(day=day, m=m, **r['parity'], parity_all=r['parity_all']))
                ab = {k.split('__')[2]: pz[k] for k in pz if k.startswith(f'{m}__disabled__')}; rows.append(dict(LG.arm_row(day, m, 'BASELINE', pmk[m]['disabled'], ab, end), part=part))
                q_ = ru[m]; qa = ra[q_['part']]
                srcs = [(a, q_[src], {k.split('__')[2]: qa[k] for k in qa.files if k.startswith(f'{m}__{src}__')}) for a, src in REUSED.items()]
                srcs += [(a, r[a], {k.split('__')[2]: z[k] for k in z.files if k.startswith(f'{m}__{a}__')}) for a in ('CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')]
                for arm, rc, a in srcs:
                    rows.append(dict(LG.arm_row(day, m, arm, rc, a, end), part=part)); kr += QF.class_rows(day, m, arm, rc, end); bl.append(blocks(day, m, arm, rc, end))
                    q0 = rc['queue_venue']['ahead0']; tr_.append(dict(day=day, m=m, arm=arm, **rc['treatment'], accepted=int(sum(1 for v in q0.values() if float(v[2]) > 0)),
                                                                     accepted_with_ahead=int(sum(1 for v in q0.values() if float(v[2]) > 0 and float(v[1]) > 0))))
    C.require(sorted(seen) == sorted((d, int(m)) for d in gm['days'] for m in gm['complete'][d]), 'the measured markets are not the 19 complete ones')
    R_, PA, K_, TR, BL = pd.DataFrame(rows), pd.DataFrame(par), pd.DataFrame(kr), pd.DataFrame(tr_), pd.DataFrame(bl); pr = predicates(R_)
    for n, D_ in (('arms.csv', R_), ('parity.csv', PA), ('classes.csv', K_), ('treatment.csv', TR), ('blocks.csv', BL)):
        D_.to_csv(md / n, index=False)
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, parity_all=bool(PA.parity_all.all()), parity_markets=int(len(PA)), markets=int(len(seen)), parts=list(parts),
                                   outputs={n: C.sha(md / n) for n in ('arms.csv', 'parity.csv', 'classes.csv', 'treatment.csv', 'blocks.csv')}, sources=sources(),
                                   profile_sha256=gm['profile_sha256']))
    print(json.dumps(dict(markets=len(seen), parity_all=bool(PA.parity_all.all()), parity_markets=len(PA), reading=pr['reading']), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_async_fast_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd == 'pilot':
        print(run_day(sys.argv[2], 'pilot'))
    elif cmd == 'project':
        project()
    elif cmd == 'rest':
        print(run_day(sys.argv[2], 'rest'))
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [OUT]))
    else:
        sys.exit('usage: pf_ladder_async_fast.py selftest | manifest | pilot <day> | project | rest <day> | measure [out]')
