"""QUEUE-PRIORITY-SENSITIVITY-v1 (P-2026-005; 2026-11-22, session aea5bfb9). The USER's plan 425e9fbc (reviews/automatic/20261122_queue_sweep_direction/
plan.md), approved by the USER on 2026-11-22 with its caps (<= 0.25 additional core-hours for the whole development batch, the ladder line <= 2.25
including review) and its pilot (the first two complete markets of each date). One paired execution sensitivity on the 19 complete markets of the
consumed Oct 24-27 (all 96 statuses preserved in the parent receipts); no new day, fit, price ladder, size, threshold, inventory parameter or
trained guard. Hypothetical execution sensitivity, NOT an executable policy, a monetary upper bound or exchange-truth priority; the unchanged
external flow assumption is strongest here (fills diverted from resting makers would change real book behaviour).
ARMS (run_day): CURRENT_QUEUE = the frozen PressureGuard (the accepted FULL_11 / Q5 pressure ladder: two nominal 100-share levels, inventory and
 reservation controls, 58 / 37 / 100 ms clocks) on FrontLadderEngine(front=False) - the same override engaged, never triggered - which must
 equal the accepted pressure arm on every market (fingerprint, fills, orders, requests, value, guard log, command array, queue records);
 FRONT_AT_ACTIVATION = the identical policy on FrontLadderEngine(front=True). Both are complete feedback runs: each arm's own fills feed its
 notices, reservations, inventory, vacancies and later decisions; nothing replays a recorded schedule.
TREATMENT (FrontLadderEngine.add9; its source sha256 in the manifest): the frozen engine's add9 unchanged, then for every accepted new generation
 (new, remaining > 0, not refused) the external queue ahead is set to 0 once, at activation; no event or outcome selects the generations. The
 engine's running minimum never raises it, so the order is first at its level for its life; same-price prints fill it before the external
 queue, through-price trades as before; own-order price / time priority, partial remainders, public-volume conservation (the engine's
 per-trade check), post-only refusal and strict removal unchanged. The frozen engine file is not edited.
FILL CLASS (fill_classes): per fill from the engine's own trade row (its saved index): same_price (trade price = the order price) or through
 (better for the aggressor), with the trade price and size; any other relation refuses. Explanatory after execution, never an input.
PARTS: 'pilot' = the first two complete markets of each date (manifest order), reused in the full result, never rerun; 'rest' = the other
 complete markets, run only if the write-once projection (project) fits: this batch's spent core-hours + the projected rest + measurement +
 checks <= 0.25, and the line + those + a review reserve (0.03) <= 2.25; otherwise stop (no outcome-based continuation, no repeated pilot).
MEASUREMENT (measure): per arm pf_ladder_pressure_lag.arm_row (the frozen metrics: END, eligible +1 s, per share, participation, inventory,
 pulls / withholds, first-review churn, generations; BASELINE = the pressure receipt's disabled arm, context); lost / added fills
 (pf_ladder_pressure.lost_added); the class table per arm x market x class (fills, shares, END, eligible +1 s value / shares, positive /
 negative parts, fills sharing a public trade; 5 / 30 s marks where supported with their censored counts); the treatment census.
READING (predicates): each arm's absolute END sign per day and pooled; FRONT - CURRENT END and eligible +1 s per day (POSITIVE_EVERY_DAY /
 NEGATIVE_EVERY_DAY / MIXED) -> economic IMPROVES / WORSENS / MIXED; adds_fills (more filled shares pooled); recovered = the END change /
 |CURRENT END|; the mix decomposition of the +1 s change into a volume term (share changes at CURRENT's per-share by class) and a value term;
 reading = queue_helps_policy_nonnegative (IMPROVES and FRONT END >= 0) / queue_helps_but_insufficient (IMPROVES, FRONT END < 0) /
 front_priority_does_not_rescue (more fills, pooled END change <= 0) / no_added_fills / mixed. Four consumed days: point estimates only.
Usage (run_light): pf_ladder_queue_front.py selftest | manifest | pilot <day> | project | rest <day> | measure [out]"""
import glob
import hashlib
import inspect
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_pressure_ablation as AB                  # noqa: E402  frozen (8f618ea6): day paths, classify, the parent identities

LG, P, M, AU, C, pd, LB, BE, LS, A = AB.LG, AB.P, AB.M, AB.AU, AB.C, AB.pd, AB.LB, AB.BE, AB.LS, AB.A
VERSION = 'queue_priority_sensitivity_v1'
RUN_ARMS = ('CURRENT_QUEUE', 'FRONT_AT_ACTIVATION')
PARTS, PILOT_PER_DAY = ('pilot', 'rest'), 2
PLAN = dict(plan='425e9fbc', approval='USER 2026-11-22: approve, the plan\'s pilot', parent_freeze='793c706b', parent_results='40a3ee8d')
CAP_BATCH, LINE_CEILING, REVIEW_RESERVE, TAIL_S = 0.25, 2.25, 0.03, 60.
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_queue_front_20261122'
OUT = A.DATA / 'ladder_queue_front_v1_20261122'
SOURCES = ('pf_ladder_queue_front.py', 'pf_ladder_queue_front_selftest.py') + AB.SOURCES
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_queue_front.sh')
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121',
        'ladder_pressure_residual_20261121', 'ladder_pressure_positive_20261121', 'ladder_arrival_queue_20261122', 'ladder_queue_price_20261122']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review', '20261121_ladder_pressure_residual_review', '20261122_horizon_conditions', '20261122_queue_conditions',
           '20261122_arrival_queue_review', '20261122_queue_price_review', '20261122_queue_sweep_direction']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
SG = {0: 1., 1: -1.}


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_queue_front.py', 'pf_ladder_queue_front_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the treatment
class FrontLadderEngine(LB.LadderEngine):
    """The frozen ladder engine; with front=True every accepted new generation starts with no external queue ahead (header)."""

    def __init__(self, path, end, front):
        super().__init__(path, end); self.front = bool(front); self.accepted_new = 0; self.front_set = 0

    def add9(self, o9, action=1):
        gid = super().add9(o9, action); o = self.o[gid]
        if o['new'] and o['rem'] > 0 and not o['refused']:
            self.accepted_new += 1
            if self.front:
                o['ahead'] = 0.; self.front_set += 1
        return gid


def treatment_sha():
    return hashlib.sha256(inspect.getsource(FrontLadderEngine).encode()).hexdigest()


def swap_engine(B, front):
    """Replace a fresh branch's engine (before any order or trade) by the declared wrapper."""
    C.require(not B.E.o and not B.E.fills and B.E.it == 0, 'the engine can only be replaced before any order or trade')
    B.E = FrontLadderEngine({'frames': B.E.f, 'trades': B.E.tr}, B.end, front); return B


def branch(path, ident, front):
    return swap_engine(LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident), front)


def fill_classes(E):
    """Per fill [class, trade price, trade size] from the engine's own trade row (header)."""
    out = []
    for t, gid, s, q, p, it in E.fills:
        tp, tq = float(E.tr[it, 2]), float(E.tr[it, 3])
        cls = 'same_price' if tp == float(p) else ('through' if (tp < float(p) if int(s) == 0 else tp > float(p)) else 'invalid')
        C.require(cls != 'invalid' and int(E.tr[it, 1]) == int(s), 'a fill at a trade the engine cannot fill'); out.append([cls, tp, tq])
    return out


def run_arm(path, ident, cache, edges, arm):
    """One arm of one market on a fresh branch -> (branch, ledger record, policy, observer)."""
    B = branch(path, ident, arm == 'FRONT_AT_ACTIVATION'); pol = P.PressureGuard(B, cache, edges); obs = P.QueueObs(B, pol)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    rec['missing'] = {f'{s}|{st}': int(n) for (s, st), n in pol.missing.items()}; rec['queue_venue'] = obs.queue()
    rec['fill_class'] = fill_classes(B.E); rec['treatment'] = dict(front=B.E.front, accepted_new=B.E.accepted_new, front_set=B.E.front_set)
    return B, rec, pol, obs


# ---------------------------------------------------------------------------------------------------------------------- parts, manifest, the day run
def parts_of(days, complete):
    """'pilot' = the first PILOT_PER_DAY complete markets of every date; 'rest' = the others."""
    return dict(pilot={d: [int(m) for m in complete[d][:PILOT_PER_DAY]] for d in days}, rest={d: [int(m) for m in complete[d][PILOT_PER_DAY:]] for d in days})


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    pm = LG.parent_manifest(); C.require(pm['rule_sha256'] == P.rule_sha(), 'the parent rule differs from its frozen manifest'); e = P.frozen_edges()
    man = dict(version=VERSION, plan=PLAN, treatment_sha256=treatment_sha(), parent_rule_sha256=pm['rule_sha256'], run_arms=list(RUN_ARMS), pilot_per_day=PILOT_PER_DAY,
               parent=dict(manifest_sha256=C.sha(REPO / P.RECEIPT / 'manifest.json'), day_records={d: C.sha(P.day_paths(d, False)['rec']) for d in pm['days']}),
               edges={str(s): e[s] for s in (0, 1)}, edges_sha256=P.EDGES_SHA256, sources=sources(), days=pm['days'], complete=pm['complete'], parts=parts_of(pm['days'], pm['complete']),
               caps=dict(batch_core_hours=CAP_BATCH, line_ceiling=LINE_CEILING, review_reserve=REVIEW_RESERVE), line_receipts=LINE, line_reviews=REVIEWS,
               multiplicity=dict(sensitivities=['FRONT_AT_ACTIVATION'], controls=['CURRENT_QUEUE (= the accepted pressure arm)', 'BASELINE (context)']), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, part, out=None):
    return AB.day_paths(day, part, out or OUT)


def run_day(day, part, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date of one part under the two arms -> records and arrays, write-once (header)."""
    C.require(day in M.DAYS, f'{day} is not a declared day'); C.require(part in PARTS, f'{part} is not a declared part')
    if require_frozen:
        C.require(frozen(), 'the queue-front runner / selftest / declaration / manifest / launcher are not the committed ones')
    man_path = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(man_path.read_text())
    C.require(gm['treatment_sha256'] == treatment_sha(), 'the treatment differs from the frozen one')
    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json')
    C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')
    P_ = day_paths(day, part, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True)
    mans = [int(m) for m in gm['parts'][part][day]]; t0 = time.time(); C.require(mans, f'{day}: no members in the {part} part')
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
            for arm in RUN_ARMS:
                B, rec, pol, obs = run_arm(path, ident, cache, edges, arm); cm = P.command_arrays(B); rec['queue_known'] = M.queue_summary(W_, rec, cm['commands'], t_abs0)
                out_m[arm] = rec; arrs.update({f'{m}__{arm}__{k}': v for k, v in dict(obs.arrays(), **cm).items()})
            z, pp = out_m['CURRENT_QUEUE'], pr_['pressure']
            out_m['control'] = dict(fingerprint=z['fingerprint'] == pp['fingerprint'], fills=z['fills'] == pp['fills'], value=z['value'] == pp['value'], orders=z['orders'] == pp['orders'],
                                    requests=z['requests'] == pp['requests'], guard_log=z['guard_log'] == pp['guard_log'],
                                    commands=bool(np.array_equal(arrs[f'{m}__CURRENT_QUEUE__commands'], pz[f'{m}__pressure__commands'], equal_nan=True)),
                                    queue=z['queue_venue'] == pp['queue_venue'] and z['queue_known'] == pp['queue_known'])
            out_m['control_all'] = bool(all(out_m['control'].values())); recs.append(out_m); tim['markets'].append(dict(m=m, s=time.time() - tm)); G.check('market', time.time() - tm)
        b_ = LS.gz_bytes(recs); P_['gz'].write_bytes(b_); np.savez_compressed(P_['npz'], **arrs)
        LS.RN.atomic_json(P_['rec'], dict(version=VERSION, day=day, part=part, members=mans, status='DAY_DONE', sha256=dict(gz=hashlib.sha256(b_).hexdigest(), npz=C.sha(P_['npz'])),
                                          manifest_sha256=C.sha(man_path), treatment_sha256=treatment_sha(), parent_day_record_sha256=C.sha(P.day_paths(day, False)['rec']),
                                          sources=sources(), timing=tim, elapsed_s=time.time() - t0, hwm_mib=LS.hwm(), asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P_['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- the projection gate
def cpu(paths):
    tot, peak, wall = 0., 0, 0.
    for p in paths:
        for ln in Path(p).read_text(errors='replace').splitlines():
            m_ = RL.match(ln)
            if m_:
                tot += float(m_.group(6)); peak = max(peak, int(m_.group(4))); wall += float(m_.group(5))
    return tot / 3600., peak, wall


def line_used(root=None):
    own = cpu(glob.glob(str((root or REPO / RECEIPT) / 'logs' / '**' / '*.log'), recursive=True))[0]
    line = cpu([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])[0]
    rev = cpu([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])[0]; return own, own + line + rev


def project(out=None, man_path=None, proj_path=None, root=None):
    """The write-once projection (header) after every pilot date -> run_rest."""
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json'); C.require(not proj_path.exists(), 'the projection exists (write-once)')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); recs = {}
    for d in gm['days']:
        p_ = day_paths(d, 'pilot', out)['rec']; C.require(p_.exists(), f'{d}: the pilot is not done'); recs[d] = json.loads(p_.read_text())
        C.require(recs[d].get('status') == 'DAY_DONE', f'{d}: the pilot did not finish')
    loads = [r['timing']['load'] for r in recs.values()]; mkt = [x['s'] for r in recs.values() for x in r['timing']['markets']]
    jobs = cpu(glob.glob(str((root or REPO / RECEIPT) / 'logs' / 'qf_pilot_*.log')))
    cpw = (jobs[0] * 3600.) / jobs[2] if jobs[2] > 0 else 1.; rest = {d: len(gm['parts']['rest'][d]) for d in gm['days']}
    rest_s = sum(float(np.mean(loads)) + n * float(np.mean(mkt)) for n in rest.values() if n) * cpw; spent, line = line_used(root)
    proj = (rest_s + TAIL_S) / 3600.; res = dict(pilot_load_s=loads, pilot_market_s_mean=float(np.mean(mkt)), cpu_per_wall=cpw, rest_markets=rest, projected_rest_cpu_s=rest_s, tail_s=TAIL_S,
                                                  batch_spent=spent, batch_projected=spent + proj, line_now=line, line_projected=line + proj + REVIEW_RESERVE, caps=gm['caps'],
                                                  run_rest=bool(spent + proj <= gm['caps']['batch_core_hours'] and line + proj + REVIEW_RESERVE <= gm['caps']['line_ceiling']))
    proj_path.parent.mkdir(parents=True, exist_ok=True); proj_path.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)); return res


# ---------------------------------------------------------------------------------------------------------------------- measurement
def class_rows(day, m, arm, rec, end):
    """The class table of one arm of one market (header)."""
    mark = float(rec['mark']); out = []; cnt = {}
    for f in rec['fills']:
        cnt[int(f[5])] = cnt.get(int(f[5]), 0) + 1
    for cls in ('same_price', 'through'):
        r = dict(day=day, m=m, arm=arm, cls=cls, fills=0, q=0., end=0., el_q=0., mo1s=0., pos1s=0., neg1s=0., shared_trade_fills=0, mo5s=0., el5_q=0., cens5=0, mo30s=0., el30_q=0., cens30=0)
        for f, mm, c in zip(rec['fills'], rec['fill_mids'], rec['fill_class']):
            if c[0] != cls:
                continue
            t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4]); r['fills'] += 1; r['q'] += q; r['end'] += SG[s] * q * (mark - p); r['shared_trade_fills'] += int(cnt[int(f[5])] > 1)
            if t <= end - 1000. and mm[6] is not None:
                v = SG[s] * q * (float(mm[6]) - p); r['el_q'] += q; r['mo1s'] += v; r['pos1s'] += max(v, 0.); r['neg1s'] += min(v, 0.)
            for h, j in ((5, 8), (30, 10)):
                if mm[j] is None:
                    r[f'cens{h}'] += 1
                else:
                    r[f'mo{h}s'] += SG[s] * q * (float(mm[j]) - p); r[f'el{h}_q'] += q
        out.append(r)
    return out


def arm_totals(R_, arm, day=None):
    g = R_[(R_.arm == arm) & ((R_.day == day) if day is not None else True)]; q = float(g.all_cut_valid_q.sum())
    return dict(END=float(g.value.sum()), mo1s=float(g.all_id_mo1s.sum()), valid_q=q, per_share=float(g.all_cut_mo1s.sum()) / q if q > 0 else np.nan, q=float(g.all_q.sum()))


def mix(K_):
    """The eligible +1 s change FRONT - CURRENT split into a volume term (share changes at CURRENT's per-share, by class) and a value term."""
    F, Cq = 'FRONT_AT_ACTIVATION', 'CURRENT_QUEUE'; out = dict(classes={}, volume=0., value=0.)
    for cls in ('same_price', 'through'):
        a, b = K_[(K_.arm == Cq) & (K_.cls == cls)], K_[(K_.arm == F) & (K_.cls == cls)]
        qa, qb, va, vb = float(a.el_q.sum()), float(b.el_q.sum()), float(a.mo1s.sum()), float(b.mo1s.sum()); pa = va / qa if qa > 0 else 0.; pb = vb / qb if qb > 0 else 0.
        vol, val = (qb - qa) * pa, qb * (pb - pa); out['classes'][cls] = dict(cur_q=qa, front_q=qb, cur_per_share=pa if qa > 0 else np.nan, front_per_share=pb if qb > 0 else np.nan,
                                                                             cur_mo1s=va, front_mo1s=vb, volume=vol, value=val)
        out['volume'] += vol; out['value'] += val
    out['total'] = out['volume'] + out['value']; return out


def predicates(R_, K_):
    """The declared readings (header)."""
    F, Cq = 'FRONT_AT_ACTIVATION', 'CURRENT_QUEUE'; days = sorted(R_.day.unique()); arms = ('BASELINE',) + RUN_ARMS
    byd = {d: {a: arm_totals(R_, a, d) for a in arms} for d in days}; tot = {a: arm_totals(R_, a) for a in arms}
    dE = {d: byd[d][F]['END'] - byd[d][Cq]['END'] for d in days}; d1 = {d: byd[d][F]['mo1s'] - byd[d][Cq]['mo1s'] for d in days}
    dp = {d: byd[d][F]['per_share'] - byd[d][Cq]['per_share'] for d in days}
    eA, mA = AB.classify(dE.values()), AB.classify(d1.values()); econ = 'IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else 'WORSENS' if eA == mA == 'NEGATIVE_EVERY_DAY' else 'MIXED'
    mE = R_[R_.arm == F].set_index(['day', 'm']).value - R_[R_.arm == Cq].set_index(['day', 'm']).value; dT = tot[F]['END'] - tot[Cq]['END']; adds = bool(tot[F]['q'] > tot[Cq]['q'])
    reading = ('queue_helps_policy_nonnegative' if econ == 'IMPROVES' and tot[F]['END'] >= 0 else 'queue_helps_but_insufficient' if econ == 'IMPROVES'
               else 'front_priority_does_not_rescue' if adds and dT <= 0 else 'no_added_fills' if not adds else 'mixed')
    g, b = R_[R_.arm == F], R_[R_.arm == Cq]; delta = lambda c, f='sum': float(getattr(g[c], f)() - getattr(b[c], f)())
    return dict(days=days, arm_totals=tot, arm_by_day=byd,
                END=dict(answer=eA, by_day=dE, total=dT, markets_better=int((mE > 1e-9).sum()), markets_worse=int((mE < -1e-9).sum()), markets_equal=int((mE.abs() <= 1e-9).sum())),
                mo1s=dict(answer=mA, by_day=d1, total=tot[F]['mo1s'] - tot[Cq]['mo1s']),
                per_share=dict(answer=AB.classify(dp.values()) if all(np.isfinite(v) for v in dp.values()) else 'MIXED', by_day=dp, total=tot[F]['per_share'] - tot[Cq]['per_share']),
                end_negative=dict(total={a: bool(tot[a]['END'] < 0) for a in arms}, by_day={d: {a: bool(byd[d][a]['END'] < 0) for a in arms} for d in days}),
                adds_fills=adds, filled_q=dict(current=tot[Cq]['q'], front=tot[F]['q']), recovered=dT / abs(tot[Cq]['END']) if tot[Cq]['END'] != 0 else np.nan, mix=mix(K_),
                participation=dict(both_exact=delta('both_exact', 'mean'), bid_exact=delta('bid_exact', 'mean'), ask_exact=delta('ask_exact', 'mean'), pulls=delta('pulls'),
                                   withholds=delta('withholds'), first_review_pulls=delta('first_review_pulls'), generations=delta('generations'),
                                   structural=delta('bid_structural') + delta('ask_structural')),
                inventory=dict(mean_abs=delta('mean_abs_inv', 'mean'), share_abs_ge_S=delta('share_abs_ge_S', 'mean')), economic=econ, reading=reading,
                note='a hypothetical execution sensitivity on 19 consumed-day markets; not an executable policy, upper bound, validation or profitability claim')


def measure(out=OUT, man_path=None, parts=PARTS):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); rows, ctl, la, kr, tr_, seen = [], [], [], [], [], set()
    for day in gm['days']:
        _, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day])
        for part in parts:
            if not gm['parts'][part][day]:
                continue
            P_ = day_paths(day, part, out); rec = json.loads(P_['rec'].read_text()); b = P_['gz'].read_bytes()
            C.require(rec['treatment_sha256'] == gm['treatment_sha256'] and rec['members'] == [int(x) for x in gm['parts'][part][day]] and hashlib.sha256(b).hexdigest() == rec['sha256']['gz']
                      and C.sha(P_['npz']) == rec['sha256']['npz'], f'{day} {part}: identity or members')
            z = np.load(P_['npz'])
            for r in LS.gz_rows(b):
                m = int(r['m']); end = float(r['path_info']['end_rel']); C.require((day, m) not in seen, f'{day} {m}: a market twice'); seen.add((day, m))
                ctl.append(dict(day=day, m=m, part=part, **r['control'], control_all=r['control_all']))
                ab = {k.split('__')[2]: pz[k] for k in pz if k.startswith(f'{m}__disabled__')}; rows.append(dict(LG.arm_row(day, m, 'BASELINE', pmk[m]['disabled'], ab, end), part=part))
                for arm in RUN_ARMS:
                    a = {k.split('__')[2]: z[k] for k in z.files if k.startswith(f'{m}__{arm}__')}; rows.append(dict(LG.arm_row(day, m, arm, r[arm], a, end), part=part))
                    kr += class_rows(day, m, arm, r[arm], end); q0 = r[arm]['queue_venue']['ahead0']
                    tr_.append(dict(day=day, m=m, arm=arm, **r[arm]['treatment'], accepted_with_ahead=int(sum(1 for v in q0.values() if float(v[2]) > 0 and float(v[1]) > 0)),
                                    accepted=int(sum(1 for v in q0.values() if float(v[2]) > 0))))
                la.append(dict(day=day, m=m, **P.lost_added(r['CURRENT_QUEUE'], r['FRONT_AT_ACTIVATION'], end)))
    C.require(sorted(seen) == sorted((d, int(m)) for d in gm['days'] for m in gm['complete'][d]), 'the measured markets are not the 19 complete ones')
    R_, CT, LA, K_, TR = pd.DataFrame(rows), pd.DataFrame(ctl), pd.DataFrame(la), pd.DataFrame(kr), pd.DataFrame(tr_); pr = predicates(R_, K_)
    for n, D_ in (('arms.csv', R_), ('control.csv', CT), ('lost_added.csv', LA), ('classes.csv', K_), ('treatment.csv', TR)):
        D_.to_csv(md / n, index=False)
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, control_all=bool(CT.control_all.all()), markets=int(len(seen)), parts=list(parts),
                                   outputs={n: C.sha(md / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'classes.csv', 'treatment.csv')}, sources=sources(),
                                   treatment_sha256=gm['treatment_sha256']))
    print(json.dumps(dict(markets=len(seen), control_all=bool(CT.control_all.all()), economic=pr['economic'], reading=pr['reading']), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_queue_front_selftest as ST
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
        sys.exit('usage: pf_ladder_queue_front.py selftest | manifest | pilot <day> | project | rest <day> | measure [out]')
