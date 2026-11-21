"""PRESSURE-LEAD-SENSITIVITY-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 909c671e (target 40a3ee8d) accepted BINANCE-PRESSURE-GUARD-v1
as a consumed-day diagnostic and named ONE next priority: one additional 50 ms of signal-availability delay with the pressure rule otherwise
frozen - a timing-margin sensitivity on the same 19 complete markets (all 96 statuses preserved); not a look-ahead fix, a cooldown, a tuning
step or a market-impact validation. Consumed Oct 24-27 development days; no profitability, validation or promotion claim.
RULE (LaggedPressureGuard; its source sha256 and the parent rule's are in the manifest): pf_ladder_pressure.PressureGuard (frozen at 793c706b)
 with one change - at action tick k the signal / status is read from the hash-bound saved cache at tick k - LAG on the decision's own side
 (LAG = 1: 50 ms); k - LAG < 0 is warm-up (status 'warmup': no guard action, counted); PULL, withhold and clearing all read the delayed
 stream; inventory, structural and every other decision stay current; no timer. Every intervention logs the parent's eleven fields plus the
 source decision time (t - 50 LAG) and the effective age (the source's bn_age + 50 LAG); the cancel effect is the engine's request + 37 ms.
 LAG = 0 (the zero-lag control) must reproduce the accepted pressure receipt per market: fingerprint, fills, orders, requests, value, command
 array, the parent's log fields and the queue records.
CACHE: the pressure receipt's per-market arrays (x / age / status per side), bound by its day records (each equal to the committed receipt copy
 outputs/days/<day>.json; the gz / npz sha256 they carry) - never rebuilt, never re-quantiled.
RUN (run_day; pilot = the first two complete markets of each date in the manifest's (day, m) order): per complete market the baseline path
 (its path_info = the baseline record's = the pressure receipt's), the zero-lag control and the lagged arm on fresh branches with the
 baseline's ident. The baseline arm is the pressure receipt's disabled arm (identity-checked where it is read).
SELECTION (run_selection; the audit's saved live states of both sides): lag_join replaces each live state's adv_bn_r250 by the cached value at
 its own day, market, side and tick k - LAG (warm-up or a non-ok status: NaN, so unselected); the zero-lag join (tick k) must equal every
 live state's own saved value bit for bit (the join-key check); then pf_ladder_pressure.selection unchanged (its strata, 500 draws with seed
 20261121, minimum support and censoring; the windows start after the current action's cancel effect, as saved).
MEASUREMENT (measure(); three arms per market - baseline, lag0, lag1; gross, zero cost, cents): pf_ladder_pressure.arm_metrics per arm (END
 identity, markouts with valid quantities and censor counts, side / rung, interventions, removals, generations, first-review pulls,
 lifetimes, exact inventory) with the queue fields relabelled (QUEUE_LABELS), exact uptime (pf_ladder_audit.durations), lost / added fills
 (lag1 vs baseline and lag1 vs lag0); reentry_pairs (review 909c671e finding 2): for each pressure pull the same side's next submission with
 both gids, prices, rungs and times, the pull's venue ahead at the pull tick, the next order's activation-clock queue snapshot (pf_branch_
 engine.add_order evaluates it at submission from the venue frame at activation = submission + 58 ms), same-price / same-rung flags and the
 ahead difference next - pull - a next-side-order diagnostic, not a matched replacement or a counterfactual queue cost.
READINGS (predicates()): lag1 vs baseline and lag0 vs baseline by pf_ladder_pressure.predicates (END_paired, mo1s_per_share, participation,
 inventory); the retained gain (lag1 - baseline) / (lag0 - baseline) per day and in total for END and per share (descriptive); lag1 - lag0 by
 day; the reading survives_50ms_screen iff lag1-vs-baseline END_paired and pooled mo1s_per_share are IMPROVES_EVERY_DAY and the lagged
 selection screen is SELECTION_BEYOND_EXPOSURE, else not_demonstrated_at_50ms with each failed / thin component (it does not establish that no
 shorter margin works). No lag or threshold is chosen from the outcome.
Usage (run_light): pf_ladder_pressure_lag.py selftest | manifest | pilot <day> | run <day> | selection [out] | measure [out]"""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_pressure as P                            # noqa: E402  frozen (793c706b): the rule, its runner and measurement

M, AU, GD, C, pd, LB, BE, LS, A = P.M, P.AU, P.GD, P.C, P.pd, P.LB, P.BE, P.LS, P.A
VERSION = 'pressure_lead_sensitivity_v1'
LAG, STEP = 1, P.STEP
PARENT = dict(freeze='793c706b', results='40a3ee8d', review='909c671e')
ARMS = ('lag0', 'lag1')
QUEUE_LABELS = {'q_ahead0_mean': 'q_activation_ahead_mean', 'q_first_venue_mean': 'q_first_review_venue_ahead_mean', 'q_first_known_mean': 'q_first_review_known_ahead_mean',
                'q_pulls': 'q_pulls', 'q_pull_venue_mean': 'q_pull_venue_ahead_mean', 'q_reentries': 'q_next_same_side_pairs', 'q_reentry_gap_mean': 'q_next_gap_mean',
                'q_reentry_ahead_mean': 'q_next_activation_ahead_mean', 'q_surrendered_mean': 'q_next_minus_pull_ahead_mean', 'q_reentry_same_price': 'q_next_same_price_share'}
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_pressure_lag_20261121'
OUT = A.DATA / 'ladder_pressure_lag_v1_20261121'
SOURCES = ('pf_ladder_pressure_lag.py', 'pf_ladder_pressure_lag_selftest.py') + P.SOURCES
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_lag.sh')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_pressure_lag.py', 'pf_ladder_pressure_lag_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


class LaggedPressureGuard(P.PressureGuard):
    """The candidate (header): the frozen pressure rule reading its signal LAG ticks late on its own side; lag = 0 is the zero-lag control."""

    def __init__(self, B, cache, edges, lag, enabled=True):
        super().__init__(B, cache, edges, enabled); self.lag = int(lag)

    def value(self, v):
        s = int(v['side']); k = int(round(float(v['t']) / STEP)) - self.lag; xs = self.cache['x'][s]
        if k < 0:
            return np.nan, 'warmup', np.nan
        if k >= len(xs):
            return np.nan, 'beyond_cache', np.nan
        return float(xs[k]), P.ST_CODES[int(self.cache['status'][s][k])], float(self.cache['age'][s][k])

    def _note(self, v, kind, gid, rung, q, x, ag, act):
        super()._note(v, kind, gid, rung, q, x, ag, act); self.log[-1] = self.log[-1] + (float(v['t']) - self.lag * STEP, float(ag) + self.lag * STEP)


def rule_sha():
    """The frozen rule's identity: the sha256 of LaggedPressureGuard's source (the parent rule's sha256 is checked apart)."""
    return hashlib.sha256(inspect.getsource(LaggedPressureGuard).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------- the parent's evidence
def parent_manifest():
    return json.loads((REPO / P.RECEIPT / 'manifest.json').read_text())


def parent_day(day, expect=None):
    """The pressure receipt's day record (= its committed copy), records and arrays, identity-checked -> (record, {m: market record}, arrays)."""
    Pp = P.day_paths(day, False); C.require(M.committed_equal(f'{P.RECEIPT}/outputs/days/{day}.json', Pp['rec']), f'{day}: the pressure day record is not the committed receipt copy')
    rec = json.loads(Pp['rec'].read_text()); b = Pp['gz'].read_bytes()
    C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(Pp['npz']) == rec['sha256']['npz'], f"{day}: the pressure receipt's outputs differ from its record")
    C.require(rec['rule_sha256'] == parent_manifest()['rule_sha256'] == P.rule_sha(), f'{day}: the parent rule differs from the frozen one')
    if expect is not None:
        C.require(C.sha(Pp['rec']) == expect, f'{day}: the pressure day record differs from the manifest')
    z = np.load(Pp['npz']); return rec, {int(r['m']): r for r in LS.gz_rows(b)}, {k: z[k] for k in z.files}


def cache_of(z, m):
    return dict(x={s: z[f'{m}__cache__x{s}'] for s in (0, 1)}, age={s: z[f'{m}__cache__age{s}'] for s in (0, 1)}, status={s: z[f'{m}__cache__status{s}'] for s in (0, 1)})


# ---------------------------------------------------------------------------------------------------------------------- the arms
def run_arm(path, ident, cache, edges, lag):
    """One arm of one market on a fresh branch (the baseline's construction, the lagged pressure policy) -> (branch, ledger record, policy, observer)."""
    B = LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident); pol = LaggedPressureGuard(B, cache, edges, lag); obs = P.QueueObs(B, pol)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    rec['missing'] = {f'{s}|{st}': int(n) for (s, st), n in pol.missing.items()}; rec['queue_venue'] = obs.queue()
    return B, rec, pol, obs


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    pm = parent_manifest(); C.require(pm['rule_sha256'] == P.rule_sha(), 'the parent rule differs from its frozen manifest'); e = P.frozen_edges()
    man = dict(version=VERSION, rule_sha256=rule_sha(), lag_ticks=LAG, lag_ms=LAG * STEP, parent=dict(PARENT, rule_sha256=pm['rule_sha256'], manifest_sha256=C.sha(REPO / P.RECEIPT / 'manifest.json'),
               day_records={d: C.sha(P.day_paths(d, False)['rec']) for d in pm['days']}), edges={str(s): e[s] for s in (0, 1)}, edges_sha256=P.EDGES_SHA256, sources=sources(),
               days=pm['days'], complete=pm['complete'], pilot=pm['pilot'], null=pm['null'], asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, pilot, out=None):
    sub = Path(out or OUT) / ('pilot' if pilot else 'days')
    return {k: sub / f'{day}{s}' for k, s in (('rec', '.json'), ('gz', '.jsonl.gz'), ('npz', '_arrays.npz'), ('status', '_status.json'))}


def run_day(day, pilot=False, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date's complete markets (pilot: its first two) under lag0 and lag1 -> records and arrays, write-once (header)."""
    C.require(day in M.DAYS, f'{day} is not a declared day')
    if require_frozen:
        C.require(frozen(), 'the lag runner / selftest / declaration / manifest / launcher are not the committed ones')
    man_path = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(man_path.read_text())
    C.require(gm['rule_sha256'] == rule_sha(), 'the rule differs from the frozen one')
    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'pilot_projection.json')
    C.require(pilot or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the full run needs the write-once pilot projection with run_rest true')
    P_ = day_paths(day, pilot, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True)
    edges = P.frozen_edges(); prec, pmk, pz = parent_day(day, gm['parent']['day_records'][day])
    mans = [int(m) for m in (gm['pilot'] if pilot else gm['complete'])[day]]; man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text())
    base = {int(r['m']): r for r in P.gz_records(M.day_paths(day, False)['gz'])}
    members = [x for x in man_run['selected'] if x['day'] == day and int(x['m']) in mans]; t0 = time.time(); tim = dict(load=None, markets=[])
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
            cache = cache_of(pz, m); out_m = dict(day=day, m=m, path_info=info)
            for arm, lag in (('lag0', 0), ('lag1', LAG)):
                B, rec, pol, obs = run_arm(path, ident, cache, edges, lag); cm = P.command_arrays(B); rec['queue_known'] = M.queue_summary(W_, rec, cm['commands'], t_abs0)
                out_m[arm] = rec; arrs.update({f'{m}__{arm}__{k}': v for k, v in dict(obs.arrays(), **cm).items()})
            z0, pp = out_m['lag0'], pr_['pressure']
            out_m['control'] = dict(fingerprint=z0['fingerprint'] == pp['fingerprint'], fills=z0['fills'] == pp['fills'], value=z0['value'] == pp['value'], orders=z0['orders'] == pp['orders'],
                                    requests=z0['requests'] == pp['requests'], guard_log=[e[:11] for e in z0['guard_log']] == pp['guard_log'],
                                    commands=bool(np.array_equal(arrs[f'{m}__lag0__commands'], pz[f'{m}__pressure__commands'], equal_nan=True)),
                                    queue=z0['queue_venue'] == pp['queue_venue'] and z0['queue_known'] == pp['queue_known'])
            out_m['control_all'] = bool(all(out_m['control'].values())); recs.append(out_m); tim['markets'].append(dict(m=m, s=time.time() - tm)); G.check('market', time.time() - tm)
        b_ = LS.gz_bytes(recs); P_['gz'].write_bytes(b_); np.savez_compressed(P_['npz'], **arrs)
        LS.RN.atomic_json(P_['rec'], dict(version=VERSION, day=day, pilot=bool(pilot), members=mans, sha256=dict(gz=hashlib.sha256(b_).hexdigest(), npz=C.sha(P_['npz'])),
                                          manifest_sha256=C.sha(man_path), rule_sha256=rule_sha(), parent_day_record_sha256=C.sha(P.day_paths(day, False)['rec']), sources=sources(),
                                          timing=tim, elapsed_s=time.time() - t0, hwm_mib=LS.hwm(), asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P_['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- selection
def lag_join(S_, caches, lag=LAG):
    """Each live state's adv_bn_r250 replaced by the cached value at its own (day, m), side and tick k - lag (warm-up / non-ok: NaN); the zero-lag
    join (tick k) must equal the state's own saved value bit for bit -> (copy, stats)."""
    g = S_.copy(); live = (g.kind == 'live').values; vals = g[P.FEATURE].astype(float).values.copy(); n = mis = off = warm = nok = 0
    for i in np.nonzero(live)[0]:
        c = caches[(str(g.day.iat[i]), int(g.m.iat[i]))]; s = int(g.side.iat[i]); t = float(g.t.iat[i]); k = int(round(t / STEP)); n += 1
        if abs(k * STEP - t) > 1e-6 or not 0 <= k < len(c['x'][s]):
            off += 1; vals[i] = np.nan; continue
        mis += int(P.bits(c['x'][s][k]) != P.bits(g[P.FEATURE].iat[i]))
        j = k - lag
        if j < 0:
            warm += 1; vals[i] = np.nan
        elif int(c['status'][s][j]) != 0:
            nok += 1; vals[i] = np.nan
        else:
            vals[i] = float(c['x'][s][j])
    g[P.FEATURE] = vals; return g, dict(live_states=n, zero_lag_mismatched=mis, off_tick=off, warmup=warm, non_ok=nok)


def run_selection(out=OUT, require_frozen=True, man_path=None):
    out = Path(out); p = out / 'selection.json'
    if require_frozen:
        C.require(frozen(), 'not the committed lag runner')
    C.require(not p.exists(), 'the selection control exists (write-once)'); out.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); S_ = pd.read_csv(AU.OUT / 'states.csv', float_precision='round_trip', dtype={'day': str})
    caches = {}
    for day in gm['days']:
        _, pmk, pz = parent_day(day, gm['parent']['day_records'][day]); caches.update({(day, m): cache_of(pz, m) for m in pmk})
    S_lag, st = lag_join(S_, caches); C.require(st['zero_lag_mismatched'] == 0 and st['off_tick'] == 0, f'the join keys do not reproduce the saved values: {st}')
    res = P.selection(S_lag, P.frozen_edges()); res['join'] = st; res['lag_ticks'] = LAG
    res['inputs'] = dict(states_sha256=C.sha(AU.OUT / 'states.csv'), edges_sha256=P.EDGES_SHA256, parent_day_records=gm['parent']['day_records']); C.dump(p, res)
    print(json.dumps(res, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------------------- measurement
def reentry_pairs(rec):
    """Review 909c671e finding 2 (header): for each pressure pull the same side's next submission - a next-side-order diagnostic."""
    qv = rec['queue_venue']; od = {int(o['gid']): o for o in rec['orders']}; out = []
    for g, v in sorted(qv['pull'].items(), key=lambda kv: (kv[1][0], int(kv[0]))):
        o = od[int(g)]; nxt = sorted((float(x['submit']), int(x['gid'])) for x in rec['orders'] if int(x['side']) == int(o['side']) and float(x['submit']) > v[0])
        row = dict(pull_gid=int(g), side=int(o['side']), pull_t=float(v[0]), pull_price=float(o['price']), pull_rung=float(o['rung']), pull_venue_ahead=float(v[1]), pull_venue_rem=float(v[2]),
                   next_gid=None, next_submit=np.nan, next_act=np.nan, next_price=np.nan, next_rung=np.nan, next_activation_ahead=np.nan, next_refused=None, gap_ms=np.nan,
                   same_price=None, same_rung=None, ahead_difference=np.nan)
        if nxt:
            g2 = nxt[0][1]; o2 = od[g2]; a2 = float(qv['ahead0'][str(g2)][1])
            row.update(next_gid=g2, next_submit=float(o2['submit']), next_act=float(o2['act']), next_price=float(o2['price']), next_rung=float(o2['rung']), next_activation_ahead=a2,
                       next_refused=bool(o2['refused']), gap_ms=float(o2['submit']) - float(v[0]), same_price=float(o2['price']) == float(o['price']),
                       same_rung=float(o2['rung']) == float(o['rung']), ahead_difference=a2 - float(v[1]))
        out.append(row)
    return out


def pair_summary(rows):
    have = [r for r in rows if r['next_gid'] is not None]; sp = [r for r in have if r['same_price']]
    mean = lambda z: float(np.mean(z)) if len(z) else np.nan
    return dict(pairs_pulls=len(rows), pairs_with_next=len(have), pairs_same_price=len(sp), pairs_same_rung=sum(1 for r in have if r['same_rung']),
                pairs_ahead_difference_mean=mean([r['ahead_difference'] for r in have]), pairs_same_price_ahead_difference_mean=mean([r['ahead_difference'] for r in sp]),
                pairs_gap_ms_mean=mean([r['gap_ms'] for r in have]))


def arm_row(day, m, arm, rec, a, end):
    du = AU.durations(dict(day=day, m=m, ledger=rec), {f'{day}|{m}__nva': a['nva']}); x = P.arm_metrics(rec, a, end)
    x = {QUEUE_LABELS.get(k, k): v for k, v in x.items()}
    return dict(day=day, m=m, arm=arm, both_exact=du['both_exact'], ask_exact=du['ask_exact'], bid_exact=du['bid_exact'], **x, **pair_summary(reentry_pairs(rec)) if arm != 'baseline' else {})


def relabel(R_, cand, ref):
    X_ = R_[R_.arm.isin([cand, ref])].copy(); X_['arm'] = X_.arm.map({cand: 'pressure', ref: 'disabled'}); return X_


def retained(R_):
    """(lag1 - baseline) / (lag0 - baseline) per day and in total, for END and the pooled +1 s per share (descriptive)."""
    days = sorted(R_.day.unique()); out = dict(END={}, per_share={})
    tot = lambda arm, d, c: float(R_[(R_.arm == arm) & ((R_.day == d) if d != 'all' else True)][c].sum())
    for d in days + ['all']:
        b0, z0, z1 = tot('baseline', d, 'value'), tot('lag0', d, 'value'), tot('lag1', d, 'value'); out['END'][d] = (z1 - b0) / (z0 - b0) if z0 != b0 else np.nan
        ps = {a: tot(a, d, 'all_cut_mo1s') / tot(a, d, 'all_cut_valid_q') for a in ('baseline', 'lag0', 'lag1')}
        out['per_share'][d] = (ps['lag1'] - ps['baseline']) / (ps['lag0'] - ps['baseline']) if ps['lag0'] != ps['baseline'] else np.nan
    out['lag1_minus_lag0_END'] = {d: tot('lag1', d, 'value') - tot('lag0', d, 'value') for d in days}
    return out


def predicates(R_, sel_lag, sel_parent):
    """The declared readings (header)."""
    vs, ref = P.predicates(relabel(R_, 'lag1', 'baseline'), sel_lag), P.predicates(relabel(R_, 'lag0', 'baseline'), sel_parent)
    fails = [f'END_paired: {vs["END_paired"]["answer"]}'] if vs['END_paired']['answer'] != 'IMPROVES_EVERY_DAY' else []
    fails += [f'mo1s_per_share: {vs["mo1s_per_share"]["all"]["answer"]}'] if vs['mo1s_per_share']['all']['answer'] != 'IMPROVES_EVERY_DAY' else []
    fails += [f'selection: {sel_lag}'] if sel_lag != 'SELECTION_BEYOND_EXPOSURE' else []
    return dict(lag1_vs_baseline=vs, lag0_vs_baseline=ref, retained=retained(R_), reading='survives_50ms_screen' if not fails else 'not_demonstrated_at_50ms', failed=fails,
                note='a 50 ms timing-margin sensitivity on 19 consumed-day markets; it chooses no lag or threshold and claims no validation or profitability')


def measure(out=OUT, man_path=None):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)')
    C.require((out / 'selection.json').exists(), 'the selection control must run before the measurement'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); sel = json.loads((out / 'selection.json').read_text())
    sel_parent = json.loads((P.OUT / 'selection.json').read_text())['answer']; rows, ctl, la, rp = [], [], [], []
    for day in gm['days']:
        P_ = day_paths(day, False, out); rec = json.loads(P_['rec'].read_text()); b = P_['gz'].read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(P_['npz']) == rec['sha256']['npz'] and rec['rule_sha256'] == gm['rule_sha256'], f'{day}: identity')
        _, pmk, pz = parent_day(day, gm['parent']['day_records'][day]); z = np.load(P_['npz'])
        for r in LS.gz_rows(b):
            m = int(r['m']); end = float(r['path_info']['end_rel']); ctl.append(dict(day=day, m=m, **r['control'], control_all=r['control_all']))
            ab = {k.split('__')[2]: pz[k] for k in pz if k.startswith(f'{m}__disabled__')}; rows.append(arm_row(day, m, 'baseline', pmk[m]['disabled'], ab, end))
            for arm in ARMS:
                a = {k.split('__')[2]: z[k] for k in z.files if k.startswith(f'{m}__{arm}__')}; rows.append(arm_row(day, m, arm, r[arm], a, end))
                rp += [dict(p, day=day, m=m, arm=arm) for p in reentry_pairs(r[arm])]
            la.append(dict(day=day, m=m, pair='lag1_vs_baseline', **P.lost_added(pmk[m]['disabled'], r['lag1'], end)))
            la.append(dict(day=day, m=m, pair='lag1_vs_lag0', **P.lost_added(r['lag0'], r['lag1'], end)))
    R_ = pd.DataFrame(rows); CT = pd.DataFrame(ctl); LA = pd.DataFrame(la); RP = pd.DataFrame(rp); pr = predicates(R_, sel['answer'], sel_parent)
    for n, D_ in (('arms.csv', R_), ('control.csv', CT), ('lost_added.csv', LA), ('reentry.csv', RP)):
        D_.to_csv(md / n, index=False)
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, control_all=bool(CT.control_all.all()), markets=int(R_.m.nunique()), selection=sel,
                                   parent_selection_answer=sel_parent, outputs={n: C.sha(md / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'reentry.csv')},
                                   sources=sources(), rule_sha256=gm['rule_sha256']))
    print(json.dumps(dict(reading=pr['reading'], failed=pr['failed'], control_all=bool(CT.control_all.all())), indent=1, default=str))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_pressure_lag_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd in ('pilot', 'run'):
        print(run_day(sys.argv[2], pilot=(cmd == 'pilot')))
    elif cmd == 'selection':
        run_selection(*(sys.argv[2:3] or [OUT]))
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [OUT]))
    else:
        sys.exit('usage: pf_ladder_pressure_lag.py selftest | manifest | pilot <day> | run <day> | selection [out] | measure [out]')
