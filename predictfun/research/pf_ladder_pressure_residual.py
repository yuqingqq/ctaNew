"""PRESSURE-RESIDUAL-TIMING-AUDIT-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review e40bd01d (target 54eab545) accepted the 2 x 2 action
attribution and named ONE next priority: from the saved FULL_11 evidence (the full pressure rule, = the accepted pressure arm), separate the
loss in the cancellation window from the exposure the pressure rule did not flag in time. A mechanism audit on the 19 complete markets (all 96
statuses preserved) - no threshold, feature, size or strategy change, no replay, no raw-tape load; consumed Oct 24-27 development days.
INPUTS (hash-bound): the ablation receipt's part records (each = its committed copy outputs/<part>/<day>.json, with the gz / npz sha256 they
 carry) for the FULL_11 ledgers and command arrays; the pressure receipt's cache arrays (pf_ladder_pressure_lag.parent_day); the frozen edges.
CLOCKS (pf_branch_engine / pf_branch_ladder): fills at a grid timestamp execute before that grid's policy commands, so a request at t is pending
 for a fill at t_f only if t < t_f; removal is strict (a fill exists only while t_f < remove; remove = the request's effect: a live cancel
 +37 ms, a cancel before activation max(request + 37, submit + 95)); a live pressure PULL requested at tick k removes at k + 37, so it can
 prevent a fill at t_f only if k + 37 <= t_f.
FILL STATUSES (classify_fill; every actual fill exactly one, in this precedence):
 1 pending_pressure / pending_inventory / pending_structural - a removal requested before the fill (req < t_f) and not yet effective
   (t_f < remove), by the request's reason (structural = the engine's pre-tick request; pressure / inventory = the PULL command's reason at
   the request time); a fill at or after its order's removal is event_or_action_mismatch;
 2 before_feasible_effect - no pending request and t_f < T1 + 37 (T1 = the order's first live callback, fallback or not; none = never live);
 3 otherwise the feasible ticks are its live callbacks at k with T1 <= k and k + 37 <= t_f: event_or_action_mismatch if the cached signal was
   ON at a feasible tick with a non-fallback callback (the rule would have pulled - investigated, never absorbed); nonevaluable if no
   feasible tick was evaluable (a fallback callback or a non-ok cached status at each); else unflagged_later (pressure OFF at every
   evaluable feasible tick, including an ON that arrives only after the last feasible tick).
 Each fill row carries the signal at its order's submission, at T1 and at the latest feasible tick (value, status, source age), the actions
 there and the counts of feasible / evaluable / ON ticks. Generations: refused, no_fill, filled (all kept).
VALUES (per fill, cents, gross): execution edge, +50 / +100 ms (valid quantity / censored), eligible +1 s (t <= END - 1 s with a +1 s mid)
 split positive / negative, movement from the +1 s mid to the END mark, late / missing-horizon END; END = eligible +1 s + movement + late.
TABLES (by status x day x side x birth rung): generations (distinct gids), fills, shares, exposure (the generations' live time, min(remove,
 full fill, END) - activation, each generation once per status) and the values; generation statuses; the market concentration (top-3 share of
 the absolute negative eligible +1 s) per status. A diagnostic partition, not a fill-selected training population or an avoided-loss estimate.
RECONCILIATION (reconcile): every fill exactly once (a duplicate or a missing fill refuses); per market the fill count, shares, END (= the
 ledger value), eligible +1 s and edge equal the accepted FULL_11 arm row (ablation arms.csv).
READING (reading(), computed): per day and pooled the gross negative eligible +1 s in the timing-limited statuses (pending_* and
 before_feasible_effect) and in unflagged_later as fractions of all statuses' gross negative, with the positive value and quantities beside;
 AUDIT_FAILED if any event_or_action_mismatch; THIN if a day has fewer than MIN_NEG_FILLS negative eligible fills; TIMING_LIMITED_DOMINANT /
 UNFLAGGED_EXPOSURE_DOMINANT if that fraction exceeds one half on every day; MIXED otherwise. Not predictive skill, no threshold inferred.
BUDGET: at most 0.12 core-hours (the saved-input pilot, the pilot markets, verifies it), the ladder line at most 2.0.
Usage (run_light): pf_ladder_pressure_residual.py selftest | manifest | audit pilot|full"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_pressure_ablation as AB                  # noqa: E402  frozen (8f618ea6): the ablation receipt's identities and parts

LG, P, M, AU, C, pd, LS, A = AB.LG, AB.P, AB.M, AB.AU, AB.C, AB.pd, AB.LS, AB.A
VERSION = 'pressure_residual_timing_audit_v1'
STEP, CANCEL_MS, ARM = P.STEP, 37., 'FULL_11'
FILL_STATUSES = ('pending_pressure', 'pending_inventory', 'pending_structural', 'before_feasible_effect', 'unflagged_later', 'nonevaluable', 'event_or_action_mismatch')
TIMING = ('pending_pressure', 'pending_inventory', 'pending_structural', 'before_feasible_effect')
UNFLAGGED = ('unflagged_later',)
MIN_NEG_FILLS = 30
ABL = dict(freeze='8f618ea6', checks='d4b843c3', results='54eab545', review='e40bd01d')
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_pressure_residual_20261121'
OUT = A.DATA / 'ladder_pressure_residual_v1_20261121'
SOURCES = ('pf_ladder_pressure_residual.py', 'pf_ladder_pressure_residual_selftest.py') + AB.SOURCES
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_residual.sh')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_pressure_residual.py', 'pf_ladder_pressure_residual_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the partition
def gid_commands(cm):
    """{gid: [(t, kind, action, reason)]} in command order."""
    out = collections.defaultdict(list)
    for c in cm:
        if int(c[3]) >= 0:
            out[int(c[3])].append((float(c[0]), M.CMD_KINDS[int(c[2])], M.CMD_ACTIONS[int(c[7])], P.REASONS[int(c[8])]))
    return out


def request_reason(o, cmds_g):
    """structural (the engine's pre-tick request) / pressure / inventory (the PULL command's reason at the request time) / unknown / None."""
    if o['req'] is None:
        return None
    if o['why'] == 'structural':
        return 'structural'
    rs = [r for t, k, a, r in cmds_g if k == 'live' and a == 'PULL' and t == float(o['req'])]
    return {'pressure_pull': 'pressure', 'inventory_pull': 'inventory'}.get(rs[0] if len(rs) == 1 else None, 'unknown')


def sig(cache, s, k, edges_s):
    """(value, status, source age, ON) of side s at tick k (the current-tick cache of the full rule)."""
    if not 0 <= k < len(cache['x'][s]):
        return (np.nan, 'beyond', np.nan, False)
    st = P.ST_CODES[int(cache['status'][s][k])]; x = float(cache['x'][s][k])
    return (x, st, float(cache['age'][s][k]), bool(st == 'ok' and P.on_value(x, edges_s)))


def classify_fill(t_f, o, cmds_g, cache, edges_s):
    """One fill's status (header precedence) -> (status, diagnostics)."""
    s = int(o['side']); live = [(t, a, r) for t, k, a, r in cmds_g if k == 'live']; T1 = live[0][0] if live else None
    d = dict(sig_submit=sig(cache, s, int(round(float(o['submit']) / STEP)), edges_s), T1=T1, n_feasible=0, n_evaluable=0, n_on=0, k_last=None, sig_T1=None, act_T1=None,
             sig_last=None, act_last=None, why=None)
    if T1 is not None:
        d.update(sig_T1=sig(cache, s, int(round(T1 / STEP)), edges_s), act_T1=f'{live[0][1]}/{live[0][2]}')
    if o['req'] is not None and float(o['req']) < t_f:
        if t_f < float(o['remove']):
            reason = request_reason(o, cmds_g)
            return (f'pending_{reason}', d) if reason in ('pressure', 'inventory', 'structural') else ('event_or_action_mismatch', dict(d, why='pending request of unknown reason'))
        return 'event_or_action_mismatch', dict(d, why='a fill at or after its removal')
    if T1 is None or t_f < T1 + CANCEL_MS:
        return 'before_feasible_effect', d
    feas = [(t, a, r) for t, a, r in live if T1 <= t and t + CANCEL_MS <= t_f]; ev = on = 0
    for t, a, r in feas:
        x = sig(cache, s, int(round(t / STEP)), edges_s); e = r != 'fallback' and x[1] == 'ok'; ev += int(e); on += int(e and x[3])
    t_last, a_last, r_last = feas[-1]
    d.update(n_feasible=len(feas), n_evaluable=ev, n_on=on, k_last=t_last, sig_last=sig(cache, s, int(round(t_last / STEP)), edges_s), act_last=f'{a_last}/{r_last}')
    if on:
        return 'event_or_action_mismatch', dict(d, why='pressure ON at a feasible evaluable tick but kept')
    if not ev:
        return 'nonevaluable', d
    return 'unflagged_later', d


SG = {0: 1., 1: -1.}


def fill_values(f, mm, mark, end):
    """One fill's values (cents): edge, +50 / +100 / +1 s (None if censored), eligible flag, movement to END, late END, END."""
    t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4]); sg = SG[s]; v = lambda i: None if mm[i] is None else sg * q * (float(mm[i]) - p)
    el = t <= end - 1000. and mm[6] is not None; e_ = sg * q * (mark - p)
    return dict(t=t, side=s, q=q, p=p, edge=v(0), mo50=v(2), mo100=v(4), mo1s=v(6) if el else None, eligible=bool(el), move_end=sg * q * (mark - float(mm[6])) if el else 0.,
                late_end=0. if el else e_, end=e_)


def market_rows(day, m, L_, cm, cache, edges):
    """Every fill of one FULL_11 ledger with its status, diagnostics and values; every generation with its status."""
    cg = gid_commands(cm); od = {int(o['gid']): o for o in L_['orders']}; end, mark = float(L_['end']), float(L_['mark']); rows = []; nf = collections.Counter()
    for i, (f, mm) in enumerate(zip(L_['fills'], L_['fill_mids'])):
        o = od[int(f[1])]; st, d = classify_fill(float(f[0]), o, cg.get(int(f[1]), []), cache, edges[int(o['side'])]); nf[int(f[1])] += 1
        rows.append(dict(day=day, m=int(m), fill=i, gid=int(f[1]), rung=int(round(float(o['rung']))), status=st, exposure_ms=float(f[0]) - float(o['act']), **fill_values(f, mm, mark, end),
                         sub_x=d['sig_submit'][0], sub_status=d['sig_submit'][1], sub_age=d['sig_submit'][2], sub_on=d['sig_submit'][3], T1=d['T1'],
                         T1_x=None if d['sig_T1'] is None else d['sig_T1'][0], T1_age=None if d['sig_T1'] is None else d['sig_T1'][2], T1_on=None if d['sig_T1'] is None else d['sig_T1'][3],
                         T1_action=d['act_T1'], k_last=d['k_last'], last_x=None if d['sig_last'] is None else d['sig_last'][0], last_age=None if d['sig_last'] is None else d['sig_last'][2],
                         last_on=None if d['sig_last'] is None else d['sig_last'][3], last_action=d['act_last'], n_feasible=d['n_feasible'], n_evaluable=d['n_evaluable'], n_on=d['n_on'], why=d['why']))
    life = AB.LG.P.lifetimes(L_); lt = {}
    for s in (0, 1):
        oo = [o for o in L_['orders'] if not o['refused'] and int(o['side']) == s]
        for o, v in zip(oo, life[s]):
            lt[int(o['gid'])] = v
    gens = [dict(day=day, m=int(m), gid=int(o['gid']), side=int(o['side']), rung=int(round(float(o['rung']))), gen_status='refused' if o['refused'] else ('no_fill' if not nf[int(o['gid'])] else 'filled'),
                 fills=int(nf[int(o['gid'])]), live_ms=lt.get(int(o['gid'])), req_reason=request_reason(o, cg.get(int(o['gid']), []))) for o in L_['orders']]
    return rows, gens


def reconcile(F, G, arm_rows, ledgers):
    """Every fill exactly once; per market the counts and sums equal the accepted FULL_11 row and the ledger value -> dict (refuses otherwise)."""
    bad = []
    for (day, m), L_ in ledgers.items():
        f = F[(F.day == day) & (F.m == m)]; n = len(L_['fills'])
        C.require(sorted(f.fill.astype(int)) == list(range(n)), f'{day} {m}: the fills are not each assigned exactly once')
        C.require(f.status.isin(FILL_STATUSES).all(), f'{day} {m}: an undeclared status')
        a = arm_rows[(day, m)]; chk = dict(fills=(len(f), a['fills']), q=(f.q.sum(), a['all_q']), end=(f.end.sum(), a['all_end']), value=(f.end.sum(), float(L_['value'])),
                                            mo1s=(f.mo1s.fillna(0.).sum(), a['all_id_mo1s']), edge=(f.edge.fillna(0.).sum(), a['all_edge']))
        bad += [(day, m, k) for k, (x, y) in chk.items() if abs(float(x) - float(y)) > 1e-6 * max(1., abs(float(y)))]
        g = G[(G.day == day) & (G.m == m)]; C.require(sorted(g.gid.astype(int)) == sorted(int(o['gid']) for o in L_['orders']), f'{day} {m}: the generations are not each listed once')
    C.require(not bad, f'the partition does not reconcile: {bad[:5]}'); return dict(markets=len(ledgers), fills=int(len(F)), generations=int(len(G)))


def table(F, G):
    """The status tables (header)."""
    rows = []; lmap = {(d, int(m), int(g_)): (0. if v is None or not np.isfinite(v) else float(v)) for d, m, g_, v in zip(G.day, G.m, G.gid, G.live_ms.astype(float))}
    for key, g in F.groupby(['status', 'day', 'side', 'rung']):
        lt = sum(lmap[(key[1], int(m), int(gid))] for m, gid in g[['m', 'gid']].drop_duplicates().itertuples(index=False))
        el = g[g.eligible]; mo = el.mo1s.astype(float)
        rows.append(dict(status=key[0], day=key[1], side=int(key[2]), rung=int(key[3]), generations=int(len(g[['m', 'gid']].drop_duplicates())), fills=int(len(g)), shares=float(g.q.sum()),
                         exposure_ms=lt, edge=float(g.edge.dropna().sum()), mo50=float(g.mo50.dropna().sum()), mo50_valid_q=float(g.q[g.mo50.notna()].sum()), mo50_censored=int(g.mo50.isna().sum()),
                         mo100=float(g.mo100.dropna().sum()), mo100_valid_q=float(g.q[g.mo100.notna()].sum()), mo100_censored=int(g.mo100.isna().sum()), mo1s=float(mo.sum()),
                         mo1s_pos=float(mo[mo > 0].sum()), mo1s_neg=float(mo[mo < 0].sum()), mo1s_valid_q=float(el.q.sum()), mo1s_neg_fills=int((mo < 0).sum()),
                         mo1s_censored=int((~g.eligible).sum()), move_end=float(g.move_end.sum()), late_end=float(g.late_end.sum()), end=float(g.end.sum())))
    T = pd.DataFrame(rows); conc = {}
    for st, g in F[F.eligible].groupby('status'):
        neg = g.mo1s.astype(float).clip(upper=0.).abs().groupby([g.day, g.m]).sum().sort_values(ascending=False); tot = float(neg.sum())
        conc[st] = dict(markets=int((neg > 0).sum()), top3_share=float(neg.head(3).sum() / tot) if tot > 0 else None)
    gs = G.groupby(['gen_status', 'day', 'side', 'rung']).size().rename('generations').reset_index()
    return T, gs, conc


def reading(F):
    """The declared reading (header)."""
    days = sorted(F.day.unique()); el = F[F.eligible].copy(); el['neg'] = el.mo1s.astype(float).clip(upper=0.); el['pos'] = el.mo1s.astype(float).clip(lower=0.); out = dict(by_day={})
    part = lambda g: dict(neg=float(g.neg.sum()), pos=float(g.pos.sum()), q=float(g.q.sum()), fills=int(len(g)), neg_fills=int((g.neg < 0).sum()))
    for d in days + ['all']:
        g = el if d == 'all' else el[el.day == d]; tot = float(g.neg.sum())
        out['by_day'][d] = dict(total=part(g), timing=part(g[g.status.isin(TIMING)]), unflagged=part(g[g.status.isin(UNFLAGGED)]), other=part(g[~g.status.isin(TIMING + UNFLAGGED)]),
                                frac_timing=float(g[g.status.isin(TIMING)].neg.sum() / tot) if tot < 0 else None,
                                frac_unflagged=float(g[g.status.isin(UNFLAGGED)].neg.sum() / tot) if tot < 0 else None)
    mism = int((F.status == 'event_or_action_mismatch').sum()); thin = [d for d in days if out['by_day'][d]['total']['neg_fills'] < MIN_NEG_FILLS]
    ft = [out['by_day'][d]['frac_timing'] for d in days]; fu = [out['by_day'][d]['frac_unflagged'] for d in days]
    out.update(mismatches=mism, thin_days=thin, answer='AUDIT_FAILED' if mism else 'THIN' if thin else 'TIMING_LIMITED_DOMINANT' if all(x is not None and x > .5 for x in ft)
               else 'UNFLAGGED_EXPOSURE_DOMINANT' if all(x is not None and x > .5 for x in fu) else 'MIXED',
               note='a diagnostic partition of the full rule\'s fills on 19 consumed-day markets; not predictive skill, no threshold inferred')
    return out


# ---------------------------------------------------------------------------------------------------------------------- inputs and the run
def ablation_part(day, part):
    """The ablation receipt's part record (= its committed copy) with its gz rows and arrays, identity-checked."""
    Pp = AB.day_paths(day, part); C.require(M.committed_equal(f'{AB.RECEIPT}/outputs/{part}/{day}.json', Pp['rec']), f'{day} {part}: the ablation record is not the committed copy')
    rec = json.loads(Pp['rec'].read_text())
    if not rec['members']:
        return rec, {}, {}
    b = Pp['gz'].read_bytes(); C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(Pp['npz']) == rec['sha256']['npz'], f'{day} {part}: the ablation outputs differ from its record')
    z = np.load(Pp['npz']); return rec, {int(r['m']): r for r in LS.gz_rows(b)}, {k: z[k] for k in z.files if '__FULL_11__commands' in k}


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    am = json.loads((REPO / AB.RECEIPT / 'manifest.json').read_text())
    man = dict(version=VERSION, ablation=dict(ABL, manifest_sha256=C.sha(REPO / AB.RECEIPT / 'manifest.json'), rule_sha256=am['rule_sha256'],
               part_records={f'{p}|{d}': C.sha(AB.day_paths(d, p)['rec']) for p in AB.PARTS for d in am['days']}, arms_csv_sha256=C.sha(AB.OUT / 'measure' / 'arms.csv')),
               parent_day_records=am['parent']['day_records'], edges={str(s): v for s, v in am['edges'].items()}, edges_sha256=am['edges_sha256'], sources=sources(), days=am['days'],
               parts=am['parts'], statuses=list(FILL_STATUSES), timing=list(TIMING), unflagged=list(UNFLAGGED), min_neg_fills=MIN_NEG_FILLS, cancel_ms=CANCEL_MS, asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def audit(scope, out=None, require_frozen=True, man_path=None):
    """scope 'pilot' (the pilot part's markets: the saved-input pilot) or 'full' (both parts) -> the tables, write-once."""
    C.require(scope in ('pilot', 'full'), f'{scope} is not a declared scope')
    if require_frozen:
        C.require(frozen(), 'the residual audit / selftest / declaration / manifest / launcher are not the committed ones')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); C.require(gm['sources'] == sources(), 'a source differs from the frozen one')
    od = Path(out or OUT) / scope; C.require(not (od / 'audit.json').exists(), f'{scope}: the audit exists (write-once)'); od.mkdir(parents=True, exist_ok=True); t0 = time.time()
    edges = {int(s): v for s, v in gm['edges'].items()}; C.require(edges == {s: v for s, v in P.frozen_edges().items()}, 'the edges differ from the frozen ones')
    arms = pd.read_csv(AB.OUT / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip'); C.require(C.sha(AB.OUT / 'measure' / 'arms.csv') == gm['ablation']['arms_csv_sha256'], 'the ablation arms differ')
    arm_rows = {(r['day'], int(r['m'])): r for r in arms[arms.arm == ARM].to_dict('records')}; Fs, Gs, ledgers = [], [], {}
    for day in gm['days']:
        _, pmk, pz = LG.parent_day(day, gm['parent_day_records'][day])
        for part in (('pilot',) if scope == 'pilot' else AB.PARTS):
            C.require(C.sha(AB.day_paths(day, part)['rec']) == gm['ablation']['part_records'][f'{part}|{day}'], f'{day} {part}: the ablation record differs from the manifest')
            rec, mk, cz = ablation_part(day, part)
            for m, r in mk.items():
                L_ = r[ARM]; rows, gens = market_rows(day, m, L_, cz[f'{m}__FULL_11__commands'], LG.cache_of(pz, m), edges); Fs += rows; Gs += gens; ledgers[(day, m)] = L_
    F, G = pd.DataFrame(Fs), pd.DataFrame(Gs); rc = reconcile(F, G, arm_rows, ledgers); T, gs, conc = table(F, G); rd = reading(F)
    for n, D_ in (('fills.csv', F), ('generations.csv', G), ('status_table.csv', T), ('generation_status.csv', gs)):
        D_.to_csv(od / n, index=False)
    C.dump(od / 'audit.json', dict(version=VERSION, scope=scope, asof=C.stamp(), reconcile=rc, reading=rd, concentration=conc, elapsed_s=time.time() - t0,
                                   outputs={n: C.sha(od / n) for n in ('fills.csv', 'generations.csv', 'status_table.csv', 'generation_status.csv')}, sources=sources(),
                                   manifest_sha256=C.sha(Path(man_path or REPO / RECEIPT / 'manifest.json'))))
    print(json.dumps(dict(scope=scope, reconcile=rc, answer=rd['answer'], mismatches=rd['mismatches'], elapsed_s=round(time.time() - t0, 1)), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_pressure_residual_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1200])
    elif cmd == 'audit':
        audit(sys.argv[2])
    else:
        sys.exit('usage: pf_ladder_pressure_residual.py selftest | manifest | audit pilot|full')
