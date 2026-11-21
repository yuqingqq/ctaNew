"""ADV2-ENTRY-SMOOTH-v1: THE BOUNDED OWN-STATE ENTRY COMPARISON AND THE REACHED ENTRY-STATE CHECK ON CONSUMED OCT 26-27 (P-2026-005;
2026-11-21, session aea5bfb9). Review 371aab56's next priority, items 5-6, and checklist item 4 (retained evidence). Declared and frozen before
the fit; it refuses to run unless HEAD holds the receipt copy of this round's fit.json. Diagnostic only: not the R4 gate, PI0 is not NEU2.
POPULATION: R3's identical nested manifest (pf_adv2_replay.population of the frozen evaluation plan 8619eec5...: the first eight selected
 worlds per (day, UTC hour), 384, partitions of 32 in canonical order). Each partition's members must equal R3's replay part's; a world
 without a path is a status (R3 had four).
POLICIES (pf_adv2_entry.POLICIES): SMOOTH, ENTRY_ONLY, PI0, N0, each a joint two-sided 5 s episode on its own branch (the binding check at every
 callback, the causal probe every 35th). ENTRY_ONLY / PI0 / N0 are R3's policies re-run: each episode's value and decision / fill hashes must
 equal R3's for the same world (a reproduction control of this module; no frozen output is reused unchecked).
RETAINED EVIDENCE (checklist item 4): every episode keeps its fills, order sizes, decisions, fallbacks, cash / position / value, mark, END,
 settings, hashes, the policy's calls / binding counters and the probe's counters, so a checker recomputes the ledger and the required
 callbacks (pf_r2_label_sample._ledger_bad / _arm_calls_bad) instead of trusting booleans.
REACHED ENTRY STATES (item 6): from SMOOTH's own episode, per world and side the eligible decision (absent, model-decided, finite prediction) of
 minimum sha256(DOMAIN + '|' + key), key = the canonical (day, instrument, original, side, head 'post', generation, decision ms), ties by the
 key, chosen before any pair runs (no score, fill or label enters), with its inclusion count; a side without one is no_eligible_decision.
 The fork is rebuilt by pf_r2_labels.bound_fork with SMOOTH as the roll-in (its fingerprint must equal the recorded one: else fork_mismatch);
 its 107 inputs are rebuilt by pf_r2_label_sample.fork_inputs (hidden / delivered / independent-rebuild controls) and must equal the decision's
 own inputs (else input_mismatch); the POST - WAIT label is pf_r2_labels.pair_bound under the frozen pi0 (audit every 35) with both arms'
 ledgers attached (pf_r2_label_sample.arm_ledgers). Each row has the label-row form, so pf_r2_label_sample._pair_bad validates it. Both the
 new and the old (RICH) POST heads predict on the identical inputs. At most 768 pairs; never added to an episode's P&L.
Usage (run_light): pf_adv2_entry_replay.py selftest | run <part> | summarize"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402  frozen (freeze 2)
import pf_adv2_entry as N                                # noqa: E402
import pf_adv2_eval as AE                                # noqa: E402  frozen (freeze 2)
import pf_adv2_replay as AR                              # noqa: E402  frozen (freeze 2): population, Recorder, metrics, churn, exposure
import pf_adv2_sample as AS                              # noqa: E402  frozen (freeze 1): the day-guarded identity

C, pd, T, LS, L, R2, KP = A.C, A.pd, A.T, A.LS, A.L, A.R2, A.KP
BE, X = L.BE, LS.X
VERSION = 'adv2_entry_smooth_replay_v1'
DOMAIN = 'adv2_entry_smooth_v1_reached_20261121'
AUDIT_EVERY, NT = 35, L.NT
OUT = N.OUT / 'replay_v1'
RECEIPT = Path('predictfun/reviews/r4e_entry_smooth_20261121')
CONTROLS = ('N0', 'PI0', 'ENTRY_ONLY')
SOURCES = ('pf_adv2.py', 'pf_adv2_entry.py', 'pf_adv2_entry_replay.py', 'pf_adv2_replay.py', 'pf_r2_labels.py', 'pf_r2_label_sample.py', 'pf_branch_engine.py',
           'pf_r2_teacher.py', 'pf_r2_shadow.py')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def jl(x):
    return json.loads(json.dumps(list(x), default=str))


def trace(Lx, B, pol, cst):
    """An episode's retained evidence (header), in the label arms' field names."""
    return dict(fill_list=jl(B.E.fills), sizes=[float(o['size']) for o in B.E.o], decision_list=jl(B.decisions), fallback_list=jl(B.fallbacks),
                cash=float(Lx['cash']), position=float(Lx['position']), value=float(Lx['value']), mark=float(B.mark), end=float(B.end), settings=dict(B.settings),
                fills=len(Lx['fills']), fills_sha256=L._h(Lx['fills']), decisions_sha256=L._h(Lx['decisions']),
                teacher=dict(calls=len(pol.log), binding_checked=pol.bind_checked, binding_bad=pol.bind_bad),
                causal={k: v for k, v in cst.items() if k not in ('details', 'n')})


def episode(name, smooth, rich, csup, M, sup, day, w, world, every=AUDIT_EVERY, record=False):
    """One joint own-state episode of a declared policy -> (branch, ledger, policy, causal stats, recorder or None)."""
    probe, cst = T.causal_probe(every)
    pol = N.make_policy(name, smooth, rich, csup, M, sup, causal=probe); B = BE.Branch(world['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol.bind(B, world)
    rcd = AR.Recorder(pol, day, world) if record else None
    for k in range(NT):
        BE.step(B, k, pol)
    return B, BE.finish(B), pol, cst, rcd


def rank_of(key):
    return hashlib.sha256((DOMAIN + '|' + A.ck(key)).encode()).hexdigest()


def select_entry(rec, day, m):
    """SMOOTH's eligible entry decisions (absent, model, finite) -> per side the minimum-rank one with its inclusion count (no outcome enters)."""
    best, n = {}, {}
    for r in rec:
        if r['state'] != 'absent' or r['status'] != 'model' or r['A'] is None or not np.isfinite(r['A']):
            continue
        key = dict(day=str(day), instrument_id=int(m), original_episode_id=int(r['oid']), side=int(r['side']), head='post', generation=int(r['generation']),
                   decision_ms=int(round(r['t'])))
        s = int(r['side']); n[s] = n.get(s, 0) + 1; rk = (rank_of(key), A.ck(key))
        if s not in best or rk < best[s][0]:
            best[s] = (rk, key, r)
    return {s: dict(key=v[1], rank=v[0][0], dec=v[2], eligible=n[s]) for s, v in sorted(best.items())}


def reached_entry(sel, day, w, world, smooth, rich, csup, M, sup, base):
    """The selected entry states -> label rows (header); a side without a selection is no_eligible_decision."""
    out = [dict(base, side=s, head='post', status='no_eligible_decision', eligible=0) for s in (0, 1) if s not in sel]
    for s, x in sorted(sel.items()):
        d = x['dec']; fac = N.factory(N.NAME, smooth, rich, csup, M, sup, world)
        ctx = L.bound_fork(world['path'], fac, d['k'], s, R2.episode_ident(day, w['o'][s]))
        r = dict(base, side=s, oid=int(w['o'][s].oid), head='post', key=x['key'], rank=x['rank'], eligible=x['eligible'], row_key=dict(decision_ms=int(round(d['t']))),
                 fork_fingerprint=ctx['fingerprint'], recorded_fingerprint=d['fingerprint'], A=d['A'], action=d['action'])
        if ctx['fingerprint'] != d['fingerprint']:
            out.append(dict(r, status='fork_mismatch')); continue
        t0 = time.time(); rec_in = [float(d['row'][c]) for c in T.FEATS]
        fi = LS.fork_inputs(ctx, world, M, sup, 'post', int(x['rank'][:8], 16))
        same_in = bool(fi.get('status') == 'ok' and np.asarray(fi['inputs'], np.float32).tobytes() == np.asarray(rec_in, np.float32).tobytes())
        tf = L.teacher_factory(M, sup, world, every=AUDIT_EVERY); res = L.pair_bound(ctx, tf, 'post')
        if res.get('status') == 'ok':
            for n_, v_ in LS.arm_ledgers(tf, res).items():
                res['arms'][n_].update(v_)
        cz = [{k_: v_ for k_, v_ in st.items() if k_ not in ('details', 'n')} for _, st in tf.made]
        xr = np.asarray([rec_in], np.float32)
        preds = {N.NAME: float(N.predict(smooth[(s, 'post')], xr)[0]), 'RICH': float(A.predict(rich['RICH'][(s, 'post')], 'RICH', xr)[0])}
        st_ = 'pair_not_ok' if res.get('status') != 'ok' else ('feature_unavailable' if fi.get('status') != 'ok' else ('input_mismatch' if not same_in else 'ok'))
        out.append(dict(r, status=st_, fork=fi, result=res, causal=cz, inputs_equal_recorded=same_in, pred=preds, seconds=time.time() - t0))
    return out


def r3_reference(k):
    """R3's replay part k (bytes against its record) -> (members, {(m, t_ms): world dict})."""
    rec = json.loads((AR.OUT / 'parts' / f'part_{k:03d}.json').read_text()); b = (AR.OUT / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
    C.require(hashlib.sha256(b).hexdigest() == rec['sha256'], f"R3's replay part {k}: bytes differ from its record")
    return rec['members'], {(int(r['m']), int(r['t_ms'])): r for r in LS.gz_rows(b)}


def r3_match(e, ref):
    """An episode equals R3's (value, decision and fill hashes)."""
    return bool(ref is not None and e.get('trace') is not None and ref.get('value') == e['value'] and ref.get('decisions_sha256') == e['trace']['decisions_sha256']
                and ref.get('fills_sha256') == e['trace']['fills_sha256'])


def fit_sha():
    """This round's fit manifest hash, or None before the fit (no record can then be this round's)."""
    p = N.OUT / 'fit.json'
    return C.sha(p) if p.exists() else None


def part_record_ok(rec, members, b, fz_sha):
    """An existing part's record is this round's: members, sources, fit, plan and its bytes."""
    return bool(fz_sha is not None and rec.get('members') == [LS.wkey(x) for x in members] and rec.get('sources') == sources() and rec.get('fit_sha256') == fz_sha
                and rec.get('eval_plan_sha256') == AE.EVAL_PLAN_SHA and hashlib.sha256(b).hexdigest() == rec.get('sha256'))


def run_part(k, out=OUT, rec_dir=RECEIPT):
    """One partition (header) -> out/parts/part_<k>.jsonl.gz + its record. An existing record is accepted only with this round's identity."""
    out = Path(out); (out / 'parts').mkdir(parents=True, exist_ok=True); rp = out / 'parts' / f'part_{k:03d}.json'
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); C.require(C.sha(AE.EVAL_DIR / 'plan.json') == AE.EVAL_PLAN_SHA, 'not the frozen evaluation plan')
    day, members = AR.population(plan)[int(k)]
    if rp.exists():
        C.require(part_record_ok(json.loads(rp.read_text()), members, (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes(), fit_sha()),
                  f'part {k}: an existing record of another identity')
        return 'DONE'
    C.require(N.committed_equal(rec_dir / 'fit' / 'fit.json', N.OUT / 'fit.json'), 'the fit manifest is not committed in HEAD: no outcome before the model hashes')
    r3m, r3w = r3_reference(int(k)); C.require(r3m == [LS.wkey(x) for x in members], f"part {k}: the members differ from R3's replay part")
    M, sup, _ = L.teacher_assets(); fz, smooth, rich, csup = N.frozen()
    kp = Path(plan['kp_dir']); src = KP.Tape(); G = LS.Guard(600. + 120. * len(members), 120.); mon = LS.RN.Sampler(out / 'parts' / f'part_{k:03d}_status.json', LS.GUARD); mon.start()
    try:
        C.require(LS.identity_now(day, kp, src, AS.guarded_files, LS.hours_tracked) == plan['sample'][day]['inputs'], f'{day}: the inputs differ from the evaluation plan\'s')
        allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
        mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
        C.require(LS.day_identity(day, kp, src, hrs, trk, AS.guarded_files) == plan['sample'][day]['inputs'], 'real_inputs\' hours / tracked differ from the plan')
        data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; G.check('load')
        rows = []
        for x in members:
            aw = allw[LS.ckey(LS.wkey(x))]; w = dict(m=x['m'], t=aw['t'], o=aw['o']); world, wst = L.world_bind(day, w, prs, fair, depth, X.real_features)
            base = dict(day=day, m=int(x['m']), t_ms=int(x['t_ms']), hour=int(x['hour']), rank=x['rank'], pi=x['pi'], oid0=int(aw['o'][0].oid), oid1=int(aw['o'][1].oid))
            ref = r3w.get((int(x['m']), int(x['t_ms'])))
            if world is None:
                rows.append(dict(base, status=f'world:{wst}', r3_status=(ref or {}).get('status'))); G.check('world'); continue
            eps, sel = {}, {}
            for name in N.POLICIES:
                B, Lx, pol, cst, rcd = episode(name, smooth, rich, csup, M, sup, day, w, world, record=(name == N.NAME))
                dec = [{k_: v_ for k_, v_ in d_.items() if k_ != 'row'} for d_ in pol.log]
                e = dict(AR.metrics(B, Lx, dec), ledger_ok=bool(Lx['status'] == 'ok' and L.reconcile_arm(Lx, B)), churn=AR.churn(pol.log, B),
                         trace=trace(Lx, B, pol, cst) if Lx['status'] == 'ok' else None, statuses=AR.count([d_['status'] for d_ in pol.log]),
                         actions=AR.count([d_['action'] for d_ in pol.log]))
                if name in CONTROLS:
                    e['r3_match'] = r3_match(e, ((ref or {}).get('episodes') or {}).get(name))
                if name == N.NAME:
                    sel = select_entry(rcd.rec, day, x['m']); e['exposure'] = AR.exposure(pol.log)
                eps[name] = e; G.check('episode')
            pr = reached_entry(sel, day, w, world, smooth, rich, csup, M, sup, base)
            for r_ in pr:
                G.check('pair', r_.get('seconds', 0.))
            rows.append(dict(base, status='ok', r3_status=(ref or {}).get('status'), episodes=eps, reached=pr)); world = None; G.check('world')
        b = LS.gz_bytes(rows); (out / 'parts' / f'part_{k:03d}.jsonl.gz').write_bytes(b)
        rec = dict(version=VERSION, part=int(k), day=day, members=[LS.wkey(x) for x in members], rows=len(rows), sha256=hashlib.sha256(b).hexdigest(),
                   eval_plan_sha256=AE.EVAL_PLAN_SHA, fit_sha256=C.sha(N.OUT / 'fit.json'), sources=sources(), elapsed_s=time.time() - G.t0, hwm_mib=LS.hwm(),
                   timing=LS.timing(G.log), asof=C.stamp())
        LS.RN.atomic_json(rp, rec); return 'PART_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(out / 'parts' / f'part_{k:03d}_status.json', dict(state='STOP_RESOURCE', part=int(k), latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


def frames(out, parts):
    """Every partition (record identity and complete membership checked) -> (episodes, reached, churn, exposure, statuses)."""
    out = Path(out); ep, rc, chn, exq, st = [], [], [], [], {}; fz = fit_sha()
    for k, (day, members) in enumerate(parts):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(part_record_ok(rec, members, b, fz), f'part {k}: its record is not this round\'s identity (members, sources, fit, plan, bytes)')
        rows = LS.gz_rows(b); C.require([LS.ckey(LS.wkey(r)) for r in rows] == [LS.ckey(LS.wkey(x)) for x in members], f'part {k}: its rows are not exactly its members')
        for r in rows:
            st[r['status']] = st.get(r['status'], 0) + 1
            if r['status'] != 'ok':
                continue
            for name, e in r['episodes'].items():
                ep.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], hour=r['hour'], policy=name, **{k_: v_ for k_, v_ in e.items() if not isinstance(v_, (dict, list))}))
                chn += [dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], policy=name, **c_) for c_ in e.get('churn', [])]
                exq += [dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], policy=name, key=k_, n=v_) for k_, v_ in (e.get('exposure') or {}).items()]
            for x in r['reached']:
                rc.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], side=x['side'], status=x['status'], eligible=x.get('eligible'), Y=(x.get('result') or {}).get('Y'),
                               A=x.get('A'), action=x.get('action'), inputs_equal_recorded=x.get('inputs_equal_recorded'),
                               **{f'pred_{c}': (x.get('pred') or {}).get(c) for c in (N.NAME, 'RICH')}))
    return pd.DataFrame(ep), pd.DataFrame(rc), pd.DataFrame(chn), pd.DataFrame(exq), st


def summarize(out=OUT):
    """Every partition -> episodes / reached / churn / exposure CSVs and replay.json (per day / policy totals, paired differences against
    ENTRY_ONLY and PI0, reached status counts)."""
    out = Path(out); plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); E, R, Ch, Ex, st = frames(out, AR.population(plan))
    for n_, df in (('episodes', E), ('reached', R), ('churn', Ch), ('exposure', Ex)):
        df.to_csv(out / f'{n_}.csv', index=False)
    tot = E.groupby(['day', 'policy']).agg(worlds=('value', 'size'), value=('value', 'sum'), value_bid=('value_bid', 'sum'), value_ask=('value_ask', 'sum'),
                                          fills=('fills', 'sum'), filled_shares=('filled_shares', 'sum'), generations=('generations', 'sum'),
                                          loss=('loss', 'sum'), severe=('severe', 'sum'), ledger_ok=('ledger_ok', 'all'), binding_bad=('binding_bad', 'sum')).reset_index()
    piv = E.pivot_table(index=['day', 'm', 't_ms'], columns='policy', values='value')
    diffs = [dict(day=d, policy=p, versus=b, n=int(g[[p, b]].dropna().shape[0]), diff_sum=float((g[p] - g[b]).sum()), positive=int(((g[p] - g[b]) > 0).sum()),
                  negative=int(((g[p] - g[b]) < 0).sum())) for d, g in piv.groupby(level=0) for p in N.POLICIES for b in ('ENTRY_ONLY', 'PI0') if p != b and p in g and b in g]
    reached = R.groupby(['day', 'status']).size().astype(int).rename('n').reset_index().to_dict('records') if len(R) else []
    js = dict(version=VERSION, statuses=st, totals=tot.to_dict('records'), diffs=diffs, reached=reached, worlds=int(len(piv)), sources=sources(), asof=C.stamp())
    (out / 'replay.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(statuses=st, worlds=js['worlds'], reached=reached), indent=1, default=str))


def selftest():
    import pf_adv2_entry_replay_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['summarize']:
        summarize()
    else:
        sys.exit(__doc__)
