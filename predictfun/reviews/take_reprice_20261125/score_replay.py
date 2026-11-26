"""Take replay of the public score v3 under Predict's venue rules (USER 2026-11-26: "we run the replay test first"). Development evidence on
the four consumed days; no strategy freeze, no orders.
Harness: take_stream_20261125's StreamBranch / one_stream unchanged (the venue-matched harness of take_exit_venue_20261125: a decision-time
list of takes, each a marketable IOC at the Predict best known at its decision, filled at the limit or better at act = decision + network +
150 ms, the rest cancelled; position cap and 50-share takes; the PASSIVE_EXIT line after the take; native tokens, fees; ledger valued at
settlement). Every complete market (the VM summary) of the four days.
Decisions (declared before the run): SCORE1 = the held-out gradient-boosted v3 score's top 1% of each day's random moments (public_score3;
kind top03 for its top 0.3%, top1 for the rest); RAND1 = a seeded draw of the same number of the day's random moments (the control). A
moment t is on the Binance E clock (its features use data up to t): decision = t + 1.5 ms feed + 2 ms decide, network 22 ms.
Readings per config: triggers in complete markets, signal and IOC statuses, taker shares, take markouts at 1 s / 10 s (c / share,
q-weighted), the ledger at settlement (total, take round trips, exits, held) and its unresolved bound; SCORE1 by kind; per day.
CPU: the first day's CPU per market projects the four days; refused above PROJ_CAP (the small-round budget) unless the USER approves more.
Outputs (write-once): OUT/score_replay_<day>.json and per-market ledgers; score_replay.json (merge).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/score_replay.py \
       selftest | rehearse <scratch dir> <day> <n markets> | day <day> | merge
"""
import collections
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_stream_20261125'))
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import stream_replay as SRp  # noqa: E402
import winners_trigger as WT0  # noqa: E402

VR, V, F, OD, MM, TR, X, C = SRp.VR, SRp.V, SRp.F, SRp.OD, SRp.MM, SRp.TR, SRp.X, SRp.C
R = H.parents[2]
OUT = X.DATA / 'take_score_20261126'
DAYS = ('20261023', '20261031', '20261101', '20261102')
NET, Q1, Q3, SEED = 22., .01, .003, 20261126
CONFIGS = ('SCORE1', 'RAND1')
FIRST_DAY, PROJ_CAP, DAY_CPU_CAP = '20261023', 900., 1200.


def bound():
    SRp.bound()
    f = Path(__file__).resolve()
    C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str(f.relative_to(R))], cwd=R) == f.read_bytes(), 'score_replay.py is not committed')


def triggers(Rd, day):
    """Rd: the day's random moments (t, d, s_gbm_v3) -> {config: DataFrame(t, d, kind)}."""
    s = Rd.s_gbm_v3.to_numpy(float)
    thr1, thr3 = np.quantile(s, 1. - Q1), np.quantile(s, 1. - Q3)
    S = Rd[s >= thr1].copy()
    S['kind'] = np.where(S.s_gbm_v3.to_numpy(float) >= thr3, 'top03', 'top1')
    rng = np.random.default_rng(SEED + int(day) % 100000)
    Rn = Rd.iloc[np.sort(rng.choice(len(Rd), len(S), replace=False))].copy()
    Rn['kind'] = 'rand'
    srt = lambda X: X.sort_values(['t', 'd'], ascending=[True, False], kind='stable')[['t', 'd', 'kind']].reset_index(drop=True)   # the branch reads them in time order
    return {'SCORE1': srt(S), 'RAND1': srt(Rn)}


def take_rows(rec, cfg):
    kind_of = {int(r['gid']): r.get('kind') for r in rec['signal_log'] if r.get('gid') is not None}
    return [dict(config=cfg, kind=kind_of.get(int(x['gid'])), q=float(x['q']), price=float(x['price']), fee_c=float(x['fee_c']), side=int(x['side']),
                 m1s_c=x['m1s_c'], m10s_c=x['m10s_c'], end_c=x.get('end_c')) for x in rec['take_marks']]


def day_run(day, out=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (int(DAY_CPU_CAP) + 60, int(DAY_CPU_CAP) + 61))
    bound()
    C.require(day in DAYS and day in OD.DAYS3, f'{day} is not a declared day')
    rehearsal = out is not None
    out = Path(out) if rehearsal else OUT
    fn = out / f'score_replay_{day}.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    if day != FIRST_DAY and not rehearsal:
        pj = json.loads((OUT / 'projection.json').read_text())
        C.require(pj['projected_cpu_s'] <= PROJ_CAP or pj.get('user_approved_cpu_s', 0.) >= pj['projected_cpu_s'], 'the projection exceeds the small-round budget: ask the USER')
    out.mkdir(parents=True, exist_ok=True)
    Z = pd.read_parquet(WT0.OUT / 'public_score3_points.parquet', columns=['day', 'kind', 't', 'd', 's_gbm_v3'])
    Rd = Z[(Z.day == day) & (Z.kind == 'random')].reset_index(drop=True)
    C.require(Rd.s_gbm_v3.notna().all(), 'a random moment without its held-out score')
    TG = triggers(Rd, day)
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']): r['payout'] for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day); t0c = time.process_time()
    rows, takes, used = [], [], {c: 0 for c in CONFIGS}
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in comp]
        if limit is not None:
            mids = mids[:max(0, limit - len(rows))]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            t0, end = float(info['t_start']), float(info['end_rel'])
            cache = OD.make_cache3(t0, end, OD.SEG_OF[day])
            old = dict(end=end, ident=MM.ident_of(day, m, ws, info, p))
            y = comp[m]
            row = dict(day=day, mid=m, hour=int(h), payout=y, arms={})
            for cfg in CONFIGS:
                T = TG[cfg]
                tdec = T.t.to_numpy(float) + SRp.FEED + SRp.DECIDE
                sel = (tdec >= t0) & (tdec < t0 + end)
                used[cfg] += int(sel.sum())
                stream = (tdec[sel] - t0, T.d.to_numpy()[sel].astype(int), T.kind.to_numpy()[sel])
                rec = SRp.one_stream(p, cache, old, edges, 200., V.BASE_SLACK, stream, NET)
                rec.update(day=day, mid=m, config=cfg)
                sha = F.write_once_gz(out / f'{m}_{cfg}.json.gz', rec)
                a = VR.native_account(rec, y)
                a.update(sha256=sha, n_triggers=int(sel.sum()), sig_status=dict(collections.Counter(r_['status'] for r_ in rec['signal_log'])),
                         ioc_status=rec['metrics']['ioc_status'], take_value=SRp.take_value(rec))
                row['arms'][cfg] = a
                takes += [dict(x, day=day, mid=m) for x in take_rows(rec, cfg)]
            rows.append(row)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    spent = time.process_time() - t0c
    n_all = sum(1 for r in vs['rows'] if r['day'] in DAYS and r['status'] == 'complete')
    if rehearsal:
        print('rehearsal cpu per market', spent / max(len(rows), 1), 'projected all days', spent / max(len(rows), 1) * n_all * 1.1, flush=True)
    elif day == FIRST_DAY:
        pfn = OUT / 'projection.json'
        C.require(not pfn.exists(), 'projection.json exists (write-once)')
        TR.dump(pfn, dict(asof='2026-11-26', first_day=day, spent_cpu_s=spent, markets=len(rows), markets_all_days=n_all,
                          projected_cpu_s=spent * n_all / max(len(rows), 1) * 1.1, cap_cpu_s=PROJ_CAP))
    TR.dump(fn, dict(asof='2026-11-26', day=day, rows=rows, takes=takes, triggers={c: int(len(TG[c])) for c in CONFIGS}, triggers_in_complete_markets=used,
                     cpu_s=spent, peak_rss_mib=F.rss_mib()))
    print(f'{day}: {len(rows)} markets, triggers {dict((c, len(TG[c])) for c in CONFIGS)} in markets {used}, cpu {spent:.0f} s', flush=True)


def merge():
    fn = H / 'score_replay.json'
    C.require(not fn.exists(), 'score_replay.json exists (write-once)')
    D = [json.loads((OUT / f'score_replay_{d}.json').read_text()) for d in DAYS]
    out = dict(asof='2026-11-26', days=list(DAYS), net_ms=NET, configs={})
    for cfg in CONFIGS:
        A = [r['arms'][cfg] for dd in D for r in dd['rows']]
        Tk = pd.DataFrame([x for dd in D for x in dd['takes'] if x['config'] == cfg])
        o = dict(markets=len(A), triggers=sum(dd['triggers'][cfg] for dd in D), triggers_in_markets=sum(dd['triggers_in_complete_markets'][cfg] for dd in D),
                 sig_status=dict(sum((collections.Counter(a['sig_status']) for a in A), collections.Counter())),
                 taker_shares=float(sum(a['taker_shares'] for a in A)), takes_filled=int(len(Tk)),
                 total_usd=float(sum(a['total_c'] for a in A)) / 100., take_rt_usd=float(sum(a['rt_take_net_c'] for a in A)) / 100.,
                 exit_rt_usd=float(sum(a['rt_exit_net_c'] for a in A)) / 100., held_usd=float(sum(a['held_settle_net_c'] for a in A)) / 100.,
                 unresolved_lo_usd=float(sum(a['unresolved_lo_c'] for a in A)) / 100., unresolved_hi_usd=float(sum(a['unresolved_hi_c'] for a in A)) / 100.,
                 by_day={dd['day']: dict(total_usd=float(sum(r['arms'][cfg]['total_c'] for r in dd['rows'])) / 100., markets=len(dd['rows']),
                                         takes=int(sum(1 for x in dd['takes'] if x['config'] == cfg))) for dd in D})
        for h in ('m1s_c', 'm10s_c'):
            ok = Tk[h].notna() if len(Tk) else []
            o[f'take_{h}_per_share'] = float((Tk[h][ok] * Tk.q[ok]).sum() / Tk.q[ok].sum()) if len(Tk) and ok.any() else None
        if len(Tk):
            o['by_kind'] = {k: dict(takes=int(len(g)), shares=float(g.q.sum()), m1s_c=float((g.m1s_c * g.q).sum() / g.q.sum()) if g.m1s_c.notna().all() else None,
                                    m10s_c=float((g.m10s_c.fillna(0) * g.q).sum() / g.q.sum()), price_mean=float(g.price.mean())) for k, g in Tk.groupby('kind')}
        out['configs'][cfg] = o
    TR.dump(fn, out)
    print(json.dumps(out, indent=1)[:4000])


def selftest():
    rng = np.random.default_rng(1)
    Rd = pd.DataFrame(dict(t=np.arange(10000.) * 1000., d=np.where(rng.random(10000) < .5, 1., -1.), s_gbm_v3=rng.random(10000)))
    TG = triggers(Rd, '20261101')
    assert len(TG['SCORE1']) == len(TG['RAND1']) == 100, (len(TG['SCORE1']), len(TG['RAND1']))
    assert (TG['SCORE1'].kind == 'top03').sum() == 30 and set(TG['RAND1'].kind) == {'rand'}
    assert TG['SCORE1'].t.isin(Rd.t[Rd.s_gbm_v3 >= np.quantile(Rd.s_gbm_v3, .99)]).all()
    assert all((np.diff(TG[c].t.to_numpy()) >= 0).all() for c in TG), 'triggers out of time order'
    Rb = pd.concat([Rd.assign(d=1.), Rd.assign(d=-1.)], ignore_index=True)                     # two direction blocks, as in the points table
    assert all((np.diff(triggers(Rb, '20261101')[c].t.to_numpy()) >= 0).all() for c in ('SCORE1', 'RAND1')), 'block order not sorted'
    TG2 = triggers(Rd, '20261101')
    assert TG2['RAND1'].equals(TG['RAND1']), 'the control draw is not seeded'
    print('score_replay selftest PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'rehearse':
        day_run(sys.argv[3], out=sys.argv[2], limit=int(sys.argv[4]))
    elif cmd == 'merge':
        merge()
    else:
        dict(selftest=selftest)[cmd]()
