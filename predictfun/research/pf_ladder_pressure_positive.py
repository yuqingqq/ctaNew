"""POSITIVE-PRESSURE-GUARD-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 4a236235 (target 2063d31a) accepted the residual timing audit
(UNFLAGGED_EXPOSURE_DOMINANT) and named ONE next priority: one declared economic challenger testing whether weaker, same-direction Binance
pressure removes additional harmful exposure without sacrificing too much useful trading - a test of broader signal coverage, not a threshold
search. One new adaptive consumed-day candidate on the same 19 complete markets (all 96 statuses preserved); no promotion, validation or
profitability claim.
RULE (PositivePressureGuard; its source sha256 and the parent rule's are in the manifest): the frozen PressureGuard with one change - ON iff
 the current-tick cached float32 adv_bn_r250 is finite and > 0 (zero ties OFF; a non-ok status takes no guard action and is counted), on both
 sides; pressure PULL and WAIT both enabled; structural / inventory precedence, 58 / 37 / 100 ms clocks, two nominal 100-share levels,
 sizing, paths, no timer and all accounting unchanged.
ARMS (run_day): Q5_CONTROL = the frozen PressureGuard, which must equal the accepted FULL_11 / pressure arm on every market (fingerprint,
 fills, orders, requests, value, log, command array, queue records), and POSITIVE; BASELINE = the pressure receipt's disabled arm (context).
 Parts: 'pilot' = all seven Oct 24 markets (its results are reused, never rerun); 'rest' = the Oct 25-27 complete markets, run only if the
 write-once whole-round projection fits (otherwise stop and ask the USER for resources).
MEASUREMENT (measure()): per arm pf_ladder_pressure_lag.arm_row (the frozen measurement with exact uptime and relabelled queue fields); the
 primary contrast POSITIVE - Q5_CONTROL per market and UTC day for the END and eligible +1 s totals (per share with its denominators apart);
 lost / added fills (the multiset convention), positive / negative contributions, censoring, participation, inventory, reducing-side
 suppressions, structural / pressure pulls, first-review churn, generations; every arm's absolute END and all_arms_negative.
SCREEN (run_screen(); the incremental local control): the audit's saved live states of both sides whose baseline callback at that tick and
 gid was a non-fallback KEEP (joined from the pressure receipt's hash-bound disabled-arm command arrays; a state without that callback is
 excluded and counted), with a finite, currently q5-OFF adv_bn_r250; the challenger-only mask is x > 0 at the snapshot's own tick; the frozen
 pf_ladder_pressure.selection runs unchanged on that population with the mask encoded as adv_bn_r250 = MASK_ON (10.0) where x > 0 else 0.0
 (its q5 test then reproduces the mask): the same strata (day x market x side x remaining x age x inventory sign x B >= 200), 500 draws,
 seed 20261121, support minima, censoring and action-relative windows. No fill-conditioned or future field is an input. Not a full-policy
 P&L null; thin support stays INCONCLUSIVE.
READING (predicates()): the END and eligible +1 s contrasts each POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED against zero; economic =
 IMPROVES iff both are POSITIVE_EVERY_DAY, WORSENS iff both NEGATIVE_EVERY_DAY, else MIXED; the screen's answer apart; stop_candidate unless
 economic is IMPROVES and the screen is SELECTION_BEYOND_EXPOSURE; participation and churn deltas; all_arms_negative. No promotion, no
 threshold search.
DEFERRED (review 4a236235): the residual audit's pending-fill diagnostic counters (uncomputed for pending fills) and its supporting-field
 checker coverage are not used here and stay deferred.
Usage (run_light): pf_ladder_pressure_positive.py selftest | manifest | pilot | rest <day> | screen [out] | measure [out [parts]]"""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_pressure_ablation as AB                  # noqa: E402  frozen (8f618ea6): classify, the parent identities

LG, P, M, AU, C, pd, LB, BE, LS, A = AB.LG, AB.P, AB.M, AB.AU, AB.C, AB.pd, AB.LB, AB.BE, AB.LS, AB.A
VERSION = 'positive_pressure_guard_v1'
STEP, MASK_ON = P.STEP, 10.
RUN_ARMS = ('Q5_CONTROL', 'POSITIVE')
PILOT_DAY = '20261024'
PARENT = dict(freeze='793c706b', results='40a3ee8d', review='4a236235')
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_pressure_positive_20261121'
OUT = A.DATA / 'ladder_pressure_positive_v1_20261121'
SOURCES = ('pf_ladder_pressure_positive.py', 'pf_ladder_pressure_positive_selftest.py') + AB.SOURCES
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_positive.sh')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_pressure_positive.py', 'pf_ladder_pressure_positive_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


class PositivePressureGuard(P.PressureGuard):
    """The challenger (header): the frozen pressure rule with ON = finite float32 adv_bn_r250 > 0 at the current tick, both sides."""

    def signal(self, v):
        x, st, ag = self.value(v)
        if st != 'ok':
            self.missing[(int(v['side']), st)] += 1
        return bool(self.enabled and st == 'ok' and np.isfinite(x) and float(np.float32(x)) > 0.), x, ag


def rule_sha():
    """The frozen rule's identity: the sha256 of PositivePressureGuard's source (the parent rule's sha256 is checked apart)."""
    return hashlib.sha256(inspect.getsource(PositivePressureGuard).encode()).hexdigest()


def run_arm(path, ident, cache, edges, arm):
    """One arm of one market on a fresh branch (the baseline's construction) -> (branch, ledger record, policy, observer)."""
    B = LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident)
    pol = (P.PressureGuard if arm == 'Q5_CONTROL' else PositivePressureGuard)(B, cache, edges); obs = P.QueueObs(B, pol)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    rec['missing'] = {f'{s}|{st}': int(n) for (s, st), n in pol.missing.items()}; rec['queue_venue'] = obs.queue()
    return B, rec, pol, obs


def parts_of(days, complete):
    """'pilot' = every complete market of the pilot date; 'rest' = every complete market of the other dates."""
    return dict(pilot={d: ([int(m) for m in complete[d]] if d == PILOT_DAY else []) for d in days}, rest={d: ([] if d == PILOT_DAY else [int(m) for m in complete[d]]) for d in days})


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    pm = LG.parent_manifest(); C.require(pm['rule_sha256'] == P.rule_sha(), 'the parent rule differs from its frozen manifest'); e = P.frozen_edges()
    man = dict(version=VERSION, rule_sha256=rule_sha(), run_arms=list(RUN_ARMS), mask_on=MASK_ON, pilot_day=PILOT_DAY,
               parent=dict(PARENT, rule_sha256=pm['rule_sha256'], manifest_sha256=C.sha(REPO / P.RECEIPT / 'manifest.json'), day_records={d: C.sha(P.day_paths(d, False)['rec']) for d in pm['days']}),
               edges={str(s): e[s] for s in (0, 1)}, edges_sha256=P.EDGES_SHA256, sources=sources(), days=pm['days'], complete=pm['complete'], parts=parts_of(pm['days'], pm['complete']),
               null=pm['null'], multiplicity=dict(new_candidates=['POSITIVE_PRESSURE_GUARD_v1'], controls=['Q5_CONTROL (= the accepted pressure arm)', 'BASELINE (context)']), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, part, out=None):
    return AB.day_paths(day, part, out or OUT)


def run_day(day, part, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date of one part under the two run arms -> records and arrays, write-once (header)."""
    C.require(day in M.DAYS, f'{day} is not a declared day'); C.require(part in AB.PARTS, f'{part} is not a declared part')
    if require_frozen:
        C.require(frozen(), 'the positive runner / selftest / declaration / manifest / launcher are not the committed ones')
    man_path = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(man_path.read_text())
    C.require(gm['rule_sha256'] == rule_sha(), 'the rule differs from the frozen one')
    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')
    C.require((part == 'pilot') == (day == gm['pilot_day']), f'{day} does not belong to the {part} part')
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'pilot_projection.json')
    C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once whole-round projection with run_rest true')
    P_ = day_paths(day, part, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True)
    mans = [int(m) for m in gm['parts'][part][day]]; t0 = time.time(); C.require(mans, f'{day}: no members in the {part} part')
    edges = P.frozen_edges(); prec, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day]); man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text())
    base = {int(r['m']): r for r in P.gz_records(M.day_paths(day, False)['gz'])}
    members = [x for x in man_run['selected'] if x['day'] == day and int(x['m']) in mans]; tim = dict(load=None, markets=[])
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
            z, pp = out_m['Q5_CONTROL'], pr_['pressure']
            out_m['control'] = dict(fingerprint=z['fingerprint'] == pp['fingerprint'], fills=z['fills'] == pp['fills'], value=z['value'] == pp['value'], orders=z['orders'] == pp['orders'],
                                    requests=z['requests'] == pp['requests'], guard_log=z['guard_log'] == pp['guard_log'],
                                    commands=bool(np.array_equal(arrs[f'{m}__Q5_CONTROL__commands'], pz[f'{m}__pressure__commands'], equal_nan=True)),
                                    queue=z['queue_venue'] == pp['queue_venue'] and z['queue_known'] == pp['queue_known'])
            out_m['control_all'] = bool(all(out_m['control'].values())); recs.append(out_m); tim['markets'].append(dict(m=m, s=time.time() - tm)); G.check('market', time.time() - tm)
        b_ = LS.gz_bytes(recs); P_['gz'].write_bytes(b_); np.savez_compressed(P_['npz'], **arrs)
        LS.RN.atomic_json(P_['rec'], dict(version=VERSION, day=day, part=part, members=mans, status='DAY_DONE', sha256=dict(gz=hashlib.sha256(b_).hexdigest(), npz=C.sha(P_['npz'])),
                                          manifest_sha256=C.sha(man_path), rule_sha256=rule_sha(), parent_day_record_sha256=C.sha(P.day_paths(day, False)['rec']), sources=sources(),
                                          timing=tim, elapsed_s=time.time() - t0, hwm_mib=LS.hwm(), asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P_['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- the screen
def screen_population(S_, cmds, edges):
    """The incremental screen's population and encoded mask (header) -> (frame, counts). cmds = {(day, m): command array of the baseline arm}."""
    g = S_[S_.kind == 'live'].copy(); keep = np.zeros(len(g), bool); nocb = 0; idx = {}
    for key, cm in cmds.items():
        for c in cm:
            if M.CMD_KINDS[int(c[2])] == 'live' and int(c[3]) >= 0:
                idx[(key[0], key[1], int(c[3]), float(c[0]))] = (M.CMD_ACTIONS[int(c[7])], P.REASONS[int(c[8])])
    for i, (d, m, gid, t) in enumerate(zip(g.day.astype(str), g.m.astype(int), g.gid.astype(int), g.t.astype(float))):
        a = idx.get((d, m, gid, t))
        if a is None:
            nocb += 1
        else:
            keep[i] = a == ('KEEP', 'keep')
    x = g[P.FEATURE].astype(float).values; fin = np.isfinite(x); off = np.array([fin[i] and not P.on_value(float(np.float32(x[i])), edges[int(s)]) for i, s in enumerate(g.side.astype(int))], bool)
    pop = keep & fin & off; h = g[pop].copy(); xs = h[P.FEATURE].astype(float).values; mask = np.array([float(np.float32(v)) > 0. for v in xs], bool)
    h[P.FEATURE] = np.where(mask, MASK_ON, 0.)
    return h, dict(live_states=int(len(g)), without_callback=int(nocb), keep_callbacks=int(keep.sum()), finite=int((keep & fin).sum()), q5_off=int(pop.sum()), mask_on=int(mask.sum()))


def run_screen(out=OUT, require_frozen=True, man_path=None):
    out = Path(out); p = out / 'screen.json'
    if require_frozen:
        C.require(frozen(), 'not the committed positive runner')
    C.require(not p.exists(), 'the screen exists (write-once)'); out.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); S_ = pd.read_csv(AU.OUT / 'states.csv', float_precision='round_trip', dtype={'day': str}); cmds = {}
    for day in gm['days']:
        _, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day]); cmds.update({(day, m): pz[f'{m}__disabled__commands'] for m in pmk})
    h, cnt = screen_population(S_, cmds, P.frozen_edges()); res = P.selection(h, P.frozen_edges()); res['population'] = cnt; res['mask_on'] = MASK_ON
    res['inputs'] = dict(states_sha256=C.sha(AU.OUT / 'states.csv'), parent_day_records=gm['parent']['day_records']); C.dump(p, res); print(json.dumps(res, indent=1, default=str)[:1500])


# ---------------------------------------------------------------------------------------------------------------------- measurement
def arm_totals(R_, arm, day=None):
    g = R_[(R_.arm == arm) & ((R_.day == day) if day is not None else True)]; q = float(g.all_cut_valid_q.sum())
    return dict(END=float(g.value.sum()), mo1s=float(g.all_id_mo1s.sum()), valid_q=q, per_share=float(g.all_cut_mo1s.sum()) / q if q > 0 else np.nan, q=float(g.all_q.sum()))


def predicates(R_, screen_answer):
    """The declared readings (header)."""
    days = sorted(R_.day.unique()); byd = {d: {a: arm_totals(R_, a, d) for a in ('BASELINE',) + RUN_ARMS} for d in days}; tot = {a: arm_totals(R_, a) for a in ('BASELINE',) + RUN_ARMS}
    dE = {d: byd[d]['POSITIVE']['END'] - byd[d]['Q5_CONTROL']['END'] for d in days}; d1 = {d: byd[d]['POSITIVE']['mo1s'] - byd[d]['Q5_CONTROL']['mo1s'] for d in days}
    dp = {d: byd[d]['POSITIVE']['per_share'] - byd[d]['Q5_CONTROL']['per_share'] for d in days}
    mE = R_[R_.arm == 'POSITIVE'].set_index(['day', 'm']).value - R_[R_.arm == 'Q5_CONTROL'].set_index(['day', 'm']).value; ab = float(mE.abs().sum())
    g, b = R_[R_.arm == 'POSITIVE'], R_[R_.arm == 'Q5_CONTROL']; delta = lambda c, f='sum': float(getattr(g[c], f)() - getattr(b[c], f)())
    eA, mA = AB.classify(dE.values()), AB.classify(d1.values()); econ = 'IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else 'WORSENS' if eA == mA == 'NEGATIVE_EVERY_DAY' else 'MIXED'
    return dict(days=days, arm_totals=tot, arm_by_day=byd,
                END=dict(answer=eA, by_day=dE, total=tot['POSITIVE']['END'] - tot['Q5_CONTROL']['END'], markets_better=int((mE > 1e-9).sum()), markets_worse=int((mE < -1e-9).sum()),
                         markets_equal=int((mE.abs() <= 1e-9).sum()), top3_abs_share=float(mE.abs().sort_values(ascending=False).head(3).sum() / ab) if ab > 0 else np.nan),
                mo1s=dict(answer=mA, by_day=d1, total=tot['POSITIVE']['mo1s'] - tot['Q5_CONTROL']['mo1s']),
                per_share=dict(answer=AB.classify(dp.values()) if all(np.isfinite(v) for v in dp.values()) else 'MIXED', by_day=dp, total=tot['POSITIVE']['per_share'] - tot['Q5_CONTROL']['per_share']),
                participation=dict(filled_q=delta('all_q'), both_exact=delta('both_exact', 'mean'), bid_exact=delta('bid_exact', 'mean'), ask_exact=delta('ask_exact', 'mean'), pulls=delta('pulls'),
                                   withholds=delta('withholds'), reducing_suppressed=delta('reducing_suppressed'), first_review_pulls=delta('first_review_pulls'), generations=delta('generations'),
                                   structural=delta('bid_structural') + delta('ask_structural')),
                inventory=dict(mean_abs=delta('mean_abs_inv', 'mean'), share_abs_ge_S=delta('share_abs_ge_S', 'mean')),
                all_arms_negative=dict(total=bool(all(tot[a]['END'] < 0 for a in tot)), by_day={d: bool(all(byd[d][a]['END'] < 0 for a in tot)) for d in days}),
                economic=econ, screen=screen_answer, stop_candidate=bool(not (econ == 'IMPROVES' and screen_answer == 'SELECTION_BEYOND_EXPOSURE')),
                note='one adaptive consumed-day challenger; no promotion, validation, threshold search or profitability claim')


def measure(out=OUT, man_path=None, parts=AB.PARTS):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)')
    C.require((out / 'screen.json').exists(), 'the screen must run before the measurement'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); scr = json.loads((out / 'screen.json').read_text()); rows, ctl, la, seen = [], [], [], set()
    for day in gm['days']:
        _, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day])
        for part in parts:
            if not gm['parts'][part][day]:
                continue
            P_ = day_paths(day, part, out); rec = json.loads(P_['rec'].read_text()); b = P_['gz'].read_bytes()
            C.require(rec['rule_sha256'] == gm['rule_sha256'] and rec['members'] == [int(x) for x in gm['parts'][part][day]] and hashlib.sha256(b).hexdigest() == rec['sha256']['gz']
                      and C.sha(P_['npz']) == rec['sha256']['npz'], f'{day} {part}: identity or members')
            z = np.load(P_['npz'])
            for r in LS.gz_rows(b):
                m = int(r['m']); end = float(r['path_info']['end_rel']); C.require((day, m) not in seen, f'{day} {m}: a market twice'); seen.add((day, m))
                ctl.append(dict(day=day, m=m, part=part, **r['control'], control_all=r['control_all']))
                ab = {k.split('__')[2]: pz[k] for k in pz if k.startswith(f'{m}__disabled__')}; rows.append(dict(LG.arm_row(day, m, 'BASELINE', pmk[m]['disabled'], ab, end), part=part))
                for arm in RUN_ARMS:
                    a = {k.split('__')[2]: z[k] for k in z.files if k.startswith(f'{m}__{arm}__')}; rows.append(dict(LG.arm_row(day, m, arm, r[arm], a, end), part=part))
                la.append(dict(day=day, m=m, **P.lost_added(r['Q5_CONTROL'], r['POSITIVE'], end)))
    R_ = pd.DataFrame(rows); CT = pd.DataFrame(ctl); LA = pd.DataFrame(la); pr = predicates(R_, scr['answer'])
    for n, D_ in (('arms.csv', R_), ('control.csv', CT), ('lost_added.csv', LA)):
        D_.to_csv(md / n, index=False)
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, control_all=bool(CT.control_all.all()), markets=int(len(seen)), parts=list(parts), screen=scr,
                                   outputs={n: C.sha(md / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv')}, sources=sources(), rule_sha256=gm['rule_sha256']))
    print(json.dumps(dict(markets=len(seen), control_all=bool(CT.control_all.all()), economic=pr['economic'], screen=pr['screen']), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_pressure_positive_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1200])
    elif cmd == 'pilot':
        print(run_day(PILOT_DAY, 'pilot'))
    elif cmd == 'rest':
        print(run_day(sys.argv[2], 'rest'))
    elif cmd == 'screen':
        run_screen(*(sys.argv[2:3] or [OUT]))
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [OUT]), parts=tuple(sys.argv[3].split(',')) if len(sys.argv) > 3 else AB.PARTS)
    else:
        sys.exit('usage: pf_ladder_pressure_positive.py selftest | manifest | pilot | rest <day> | screen [out] | measure [out [parts]]')
