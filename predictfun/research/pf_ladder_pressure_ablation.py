"""PRESSURE-ACTION-ATTRIBUTION-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 7fca0db4 (target 6b6f977b) accepted the +50 ms timing
sensitivity as a consumed-day diagnostic and named ONE next priority: is the useful effect of the original nominal-clock pressure rule
(BINANCE-PRESSURE-GUARD-v1, frozen 793c706b) primarily from withholding new exposure, pulling existing orders, or their interaction? One
predeclared 2 x 2 ablation on the same 19 complete markets (all 96 statuses preserved); not a lag, threshold or feature search. Consumed Oct
24-27 development days; no profitability, validation, winner or promotion claim.
RULE (AblatedPressureGuard; its source sha256 and the parent rule's are in the manifest): the frozen PressureGuard with two bits - C enables
 the pressure PULL of a would-be-KEEP live order, W the pressure WAIT of a would-be-POST slot; a disabled component delegates exactly to
 LadderInventory (it reads no signal). Arms BASELINE_00, PULL_ONLY_10, WITHHOLD_ONLY_01, FULL_11. Everything else is the parent's: the
 original edges at the current cached tick (no added lag; the declared feed / grid / execution latencies remain), both sides, two levels of
 nominal 100 shares, 200 / 100 / 50 inventory settings, no timer, 58 / 37 / 100 ms clocks, M50 fills, the full paths and marks; structural
 and inventory pulls / restrictions always act with their own reasons. A pull-only arm may submit during pressure (the declared simulation
 comparison).
RUN (run_day; part 'pilot' = the first two complete markets of each date in the manifest's (day, m) order, part 'rest' = the others): per
 market the baseline path (path_info = the baseline record's = the pressure receipt's), the parent cache (pf_ladder_pressure_lag.parent_day /
 cache_of: hash-bound to the committed receipt copies, never rebuilt), the arms FULL_11, PULL_ONLY_10 and WITHHOLD_ONLY_01 on fresh branches
 with the baseline's ident; FULL_11 must equal the accepted pressure arm (fingerprint, fills, orders, requests, value, log, command array,
 queue records). BASELINE_00 is the pressure receipt's disabled arm (= the original ladder), re-identified where it is read; its delegation
 is proven by the selftest. The two parts together are the population (disjoint, union = the 19 complete markets); a part without members
 writes an empty record and loads nothing.
MEASUREMENT (measure(); four arms per market; gross, zero cost, cents): pf_ladder_pressure_lag.arm_row per arm (the frozen pressure
 measurement - END identity, markouts with valid quantities and censor counts, side / rung, good / bad, interventions, reducing-side
 suppressions, removals, generations, first-review pulls, lifetimes, exact inventory - plus exact uptime and the relabelled queue fields);
 the contrasts per market and UTC day for the additive END and eligible +1 s values: cancellation added to withholding R11 - R01,
 withholding added to cancellation R11 - R10, standalone R10 - R00 and R01 - R00, interaction R11 - R10 - R01 + R00 (the standalone effects
 are never summed as the full effect); the pooled +1 s per share per arm with its valid-share denominator apart; lost / added fills per
 contrast (the multiset of (t, side, price, quantity)); the next-same-side re-entry rows of the pulling arms (review 909c671e's labels).
READINGS (predicates(), frozen before any replay): for each of the four incremental contrasts END, eligible +1 s and the pooled +1 s per share
 are each POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED against zero incremental value, with better / worse / equal market counts (END)
 and the top-3 share of the absolute market END contrasts; the interaction's END and eligible +1 s the same; all_arms_negative = every arm's
 total END < 0 (and by day). No interval, winner, promotion, threshold, feature or further candidate. The pressure receipt's 500-draw local
 screen is context only - not an entry-action or whole-policy null.
MULTIPLICITY (manifest, before output): the ladder line's candidates so far and the two new real arms PULL_ONLY_10 and WITHHOLD_ONLY_01;
 FULL_11 and BASELINE_00 are controls.
Usage (run_light): pf_ladder_pressure_ablation.py selftest | manifest | pilot <day> | rest <day> | measure [out [parts]]"""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_pressure_lag as LG                       # noqa: E402  frozen (b19b7ca0): parent_day, cache_of, arm_row, reentry_pairs

P, M, AU, C, pd, LB, BE, LS, A = LG.P, LG.M, LG.AU, LG.C, LG.pd, LG.LB, LG.BE, LG.LS, LG.A
VERSION = 'pressure_action_attribution_v1'
STEP = P.STEP
ARMS = {'BASELINE_00': (False, False), 'PULL_ONLY_10': (True, False), 'WITHHOLD_ONLY_01': (False, True), 'FULL_11': (True, True)}
RUN_ARMS = ('FULL_11', 'PULL_ONLY_10', 'WITHHOLD_ONLY_01')
CONTRASTS = {'cancel_given_withhold': ('FULL_11', 'WITHHOLD_ONLY_01'), 'withhold_given_cancel': ('FULL_11', 'PULL_ONLY_10'),
             'cancel_alone': ('PULL_ONLY_10', 'BASELINE_00'), 'withhold_alone': ('WITHHOLD_ONLY_01', 'BASELINE_00')}
PARTS = ('pilot', 'rest')
PARENT = dict(freeze='793c706b', results='40a3ee8d', review='7fca0db4')
MULTIPLICITY = dict(ladder_line_before=['ASK_RESERVATION_GUARD_v1', 'BINANCE_PRESSURE_GUARD_v1', 'PRESSURE_LEAD_LAG1_v1 (a timing sensitivity of the second)'],
                    new_real_arms=['PULL_ONLY_10', 'WITHHOLD_ONLY_01'], controls=['FULL_11 (= the accepted pressure arm)', 'BASELINE_00 (= the reused baseline)'])
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_pressure_ablation_20261121'
OUT = A.DATA / 'ladder_pressure_ablation_v1_20261121'
SOURCES = ('pf_ladder_pressure_ablation.py', 'pf_ladder_pressure_ablation_selftest.py') + LG.SOURCES
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_ablation.sh')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_pressure_ablation.py', 'pf_ladder_pressure_ablation_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


class AblatedPressureGuard(P.PressureGuard):
    """The ablation (header): C gates the pressure PULL, W the pressure WAIT; a disabled component delegates to LadderInventory."""

    def __init__(self, B, cache, edges, C, W):
        super().__init__(B, cache, edges, enabled=True); self.C, self.W = bool(C), bool(W)

    def live(self, v, o):
        if self.C:
            return super().live(v, o)
        if int(o['gid']) not in self.seen:
            self.seen.add(int(o['gid'])); self.q_first.append((int(o['gid']), float(v['t'])))
        return self.inner.live(v, o)

    def slot(self, v, r, price):
        return super().slot(v, r, price) if self.W else self.inner.slot(v, r, price)


def rule_sha():
    """The frozen rule's identity: the sha256 of AblatedPressureGuard's source (the parent rule's sha256 is checked apart)."""
    return hashlib.sha256(inspect.getsource(AblatedPressureGuard).encode()).hexdigest()


def run_arm(path, ident, cache, edges, arm):
    """One arm of one market on a fresh branch (the baseline's construction) -> (branch, ledger record, policy, observer)."""
    C_, W_ = ARMS[arm]; B = LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident); pol = AblatedPressureGuard(B, cache, edges, C_, W_); obs = P.QueueObs(B, pol)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    rec['missing'] = {f'{s}|{st}': int(n) for (s, st), n in pol.missing.items()}; rec['queue_venue'] = obs.queue()
    return B, rec, pol, obs


def parts_of(complete, pilot):
    """The two parts: the pilot (the first two complete markets of each date) and the rest (the others), disjoint and covering."""
    rest = {d: [int(m) for m in v if int(m) not in {int(x) for x in pilot[d]}] for d, v in complete.items()}
    C.require(all(sorted([int(x) for x in pilot[d]] + rest[d]) == sorted(int(m) for m in complete[d]) for d in complete), 'the parts do not cover the complete markets')
    return dict(pilot={d: [int(x) for x in v] for d, v in pilot.items()}, rest=rest)


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    pm = LG.parent_manifest(); C.require(pm['rule_sha256'] == P.rule_sha(), 'the parent rule differs from its frozen manifest'); e = P.frozen_edges()
    man = dict(version=VERSION, rule_sha256=rule_sha(), arms={a: list(v) for a, v in ARMS.items()}, run_arms=list(RUN_ARMS), contrasts={k: list(v) for k, v in CONTRASTS.items()},
               multiplicity=MULTIPLICITY, parent=dict(PARENT, rule_sha256=pm['rule_sha256'], manifest_sha256=C.sha(REPO / P.RECEIPT / 'manifest.json'),
               day_records={d: C.sha(P.day_paths(d, False)['rec']) for d in pm['days']}), edges={str(s): e[s] for s in (0, 1)}, edges_sha256=P.EDGES_SHA256, sources=sources(),
               days=pm['days'], complete=pm['complete'], parts=parts_of(pm['complete'], pm['pilot']), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, part, out=None):
    sub = Path(out or OUT) / part
    return {k: sub / f'{day}{s}' for k, s in (('rec', '.json'), ('gz', '.jsonl.gz'), ('npz', '_arrays.npz'), ('status', '_status.json'))}


def run_day(day, part, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date's markets of one part under the three run arms -> records and arrays, write-once (header)."""
    C.require(day in M.DAYS, f'{day} is not a declared day'); C.require(part in PARTS, f'{part} is not a declared part')
    if require_frozen:
        C.require(frozen(), 'the ablation runner / selftest / declaration / manifest / launcher are not the committed ones')
    man_path = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(man_path.read_text())
    C.require(gm['rule_sha256'] == rule_sha(), 'the rule differs from the frozen one')
    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'pilot_projection.json')
    C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once pilot projection with run_rest true')
    P_ = day_paths(day, part, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True)
    mans = [int(m) for m in gm['parts'][part][day]]; t0 = time.time()
    if not mans:
        LS.RN.atomic_json(P_['rec'], dict(version=VERSION, day=day, part=part, members=[], status='DAY_EMPTY', manifest_sha256=C.sha(man_path), rule_sha256=rule_sha(), sources=sources(),
                                          asof=C.stamp())); return 'DAY_EMPTY'
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
            z, pp = out_m['FULL_11'], pr_['pressure']
            out_m['control'] = dict(fingerprint=z['fingerprint'] == pp['fingerprint'], fills=z['fills'] == pp['fills'], value=z['value'] == pp['value'], orders=z['orders'] == pp['orders'],
                                    requests=z['requests'] == pp['requests'], guard_log=z['guard_log'] == pp['guard_log'],
                                    commands=bool(np.array_equal(arrs[f'{m}__FULL_11__commands'], pz[f'{m}__pressure__commands'], equal_nan=True)),
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


# ---------------------------------------------------------------------------------------------------------------------- measurement
def classify(vals):
    """Against zero incremental value."""
    vals = list(vals)
    return 'POSITIVE_EVERY_DAY' if all(v > 0 for v in vals) else 'NEGATIVE_EVERY_DAY' if all(v < 0 for v in vals) else 'MIXED'


def arm_totals(R_, arm, day=None):
    g = R_[(R_.arm == arm) & ((R_.day == day) if day is not None else True)]; q = float(g.all_cut_valid_q.sum())
    return dict(END=float(g.value.sum()), mo1s=float(g.all_id_mo1s.sum()), valid_q=q, per_share=float(g.all_cut_mo1s.sum()) / q if q > 0 else np.nan)


def predicates(R_):
    """The declared readings (header)."""
    days = sorted(R_.day.unique()); byd = {d: {a: arm_totals(R_, a, d) for a in ARMS} for d in days}; tot = {a: arm_totals(R_, a) for a in ARMS}
    mk = lambda a: R_[R_.arm == a].set_index(['day', 'm']).value; out = dict(days=days, arm_totals=tot, arm_by_day=byd, contrasts={})
    for name, (a, b) in CONTRASTS.items():
        dE = {d: byd[d][a]['END'] - byd[d][b]['END'] for d in days}; d1 = {d: byd[d][a]['mo1s'] - byd[d][b]['mo1s'] for d in days}
        dp = {d: byd[d][a]['per_share'] - byd[d][b]['per_share'] for d in days}; mE = mk(a) - mk(b); ab = float(mE.abs().sum())
        out['contrasts'][name] = dict(arms=[a, b], END=dict(answer=classify(dE.values()), by_day=dE, total=tot[a]['END'] - tot[b]['END']),
                                      mo1s=dict(answer=classify(d1.values()), by_day=d1, total=tot[a]['mo1s'] - tot[b]['mo1s']),
                                      per_share=dict(answer=classify(dp.values()) if all(np.isfinite(v) for v in dp.values()) else 'MIXED', by_day=dp, total=tot[a]['per_share'] - tot[b]['per_share']),
                                      markets_better=int((mE > 1e-9).sum()), markets_worse=int((mE < -1e-9).sum()), markets_equal=int((mE.abs() <= 1e-9).sum()),
                                      top3_abs_share=float(mE.abs().sort_values(ascending=False).head(3).sum() / ab) if ab > 0 else np.nan)
    inter = lambda k, d: byd[d]['FULL_11'][k] - byd[d]['PULL_ONLY_10'][k] - byd[d]['WITHHOLD_ONLY_01'][k] + byd[d]['BASELINE_00'][k]
    tint = lambda k: tot['FULL_11'][k] - tot['PULL_ONLY_10'][k] - tot['WITHHOLD_ONLY_01'][k] + tot['BASELINE_00'][k]
    out['interaction'] = {k: dict(answer=classify(inter(k, d) for d in days), by_day={d: inter(k, d) for d in days}, total=tint(k)) for k in ('END', 'mo1s')}
    out['all_arms_negative'] = dict(total=bool(all(tot[a]['END'] < 0 for a in ARMS)), by_day={d: bool(all(byd[d][a]['END'] < 0 for a in ARMS)) for d in days})
    out['note'] = 'a 2 x 2 ablation of complete feedback policies on 19 consumed-day markets; no interval, winner, promotion, threshold or new candidate'
    return out


def measure(out=OUT, man_path=None, parts=PARTS):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); rows, ctl, la, rp, seen = [], [], [], [], set()
    for day in gm['days']:
        _, pmk, pz = LG.parent_day(day, gm['parent']['day_records'][day])
        for part in parts:
            P_ = day_paths(day, part, out); rec = json.loads(P_['rec'].read_text()); C.require(rec['rule_sha256'] == gm['rule_sha256'] and rec['members'] == [int(x) for x in gm['parts'][part][day]],
                                                                                                 f'{day} {part}: identity or members')
            if not rec['members']:
                continue
            b = P_['gz'].read_bytes(); C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(P_['npz']) == rec['sha256']['npz'], f'{day} {part}: output identity')
            z = np.load(P_['npz'])
            for r in LS.gz_rows(b):
                m = int(r['m']); end = float(r['path_info']['end_rel']); C.require((day, m) not in seen, f'{day} {m}: a market in two parts'); seen.add((day, m))
                ctl.append(dict(day=day, m=m, part=part, **r['control'], control_all=r['control_all'])); recs = {'BASELINE_00': pmk[m]['disabled']}
                ab = {k.split('__')[2]: pz[k] for k in pz if k.startswith(f'{m}__disabled__')}; rows.append(dict(LG.arm_row(day, m, 'BASELINE_00', recs['BASELINE_00'], ab, end), part=part))
                for arm in RUN_ARMS:
                    a = {k.split('__')[2]: z[k] for k in z.files if k.startswith(f'{m}__{arm}__')}; recs[arm] = r[arm]; rows.append(dict(LG.arm_row(day, m, arm, r[arm], a, end), part=part))
                    rp += [dict(p_, day=day, m=m, arm=arm) for p_ in LG.reentry_pairs(r[arm])]
                for name, (a1, a0) in CONTRASTS.items():
                    la.append(dict(day=day, m=m, contrast=name, **P.lost_added(recs[a0], recs[a1], end)))
    R_ = pd.DataFrame(rows); CT = pd.DataFrame(ctl); LA = pd.DataFrame(la); RP = pd.DataFrame(rp); pr = predicates(R_)
    for n, D_ in (('arms.csv', R_), ('control.csv', CT), ('lost_added.csv', LA), ('reentry.csv', RP)):
        D_.to_csv(md / n, index=False)
    context = json.loads((P.OUT / 'selection.json').read_text())
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, control_all=bool(CT.control_all.all()), markets=int(len(seen)), parts=list(parts),
                                   context_screen=dict(answer=context['answer'], percentile=context['percentile'], note='the pressure receipt\'s local screen: context, not an entry-action or whole-policy null'),
                                   outputs={n: C.sha(md / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'reentry.csv')}, sources=sources(), rule_sha256=gm['rule_sha256']))
    print(json.dumps(dict(markets=len(seen), control_all=bool(CT.control_all.all())), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_pressure_ablation_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd in PARTS:
        print(run_day(sys.argv[2], cmd))
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [OUT]), parts=tuple(sys.argv[3].split(',')) if len(sys.argv) > 3 else PARTS)
    else:
        sys.exit('usage: pf_ladder_pressure_ablation.py selftest | manifest | pilot <day> | rest <day> | measure [out [parts]]')
