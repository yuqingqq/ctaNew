"""Live Tokyo runs vs the frozen research replay on the same 2026-11-08 windows (user 2026-11-09: "check what's the main
discrepancy and what caused that, how to make live match replay"). Descriptive diagnosis on one consumed day; no test, no tuning.
Inputs: the data branch data/tokyo-live-20261109 (e41bb5c; ledger, core inputs/outputs, live.log refusal reasons), the kernel
journal exports (pmmm-tools journal-export at c54df1f), and the replay of pf_live_samewindow.py (frozen v3 engine, J and X).
  kernel  per order (cid): place, refusal reason, sent, accepted, cancel, CancelPendingAudit, CancelQuantityReconciled,
          CancelAuditComplete, fills. Rebuilds the kernel's NetCap quantity (risk::exposure: net + buy quantity of possibly-live
          orders + remaining of cancelled orders whose cancel audit is pending and unreconciled) at every NetCap refusal, and
          splits it into live vs audit-pending orders. Check: the rebuilt worst case + the order exceeds 150 on the refusals
          (a mismatch is reported, not dropped).
  compare live vs replay per window and in total (P&L, fills, shares, by phase / lane), and for every replay fill whether live
          held an order on that side at that price then (+-250 ms), was refused there, or wanted no order (live decision).
  lifetime  live's fill share of the replay fills it held resting, by the live order's lifetime; live fills with a replay fill.
  markout   replay fills by order age / cancel-pending (9 consumed days and the 220 windows) and live fills by lifetime: markouts.
  phantom   NetCap refusals left under a late-fill window W or a higher net_shares; late fills a window would miss.
  feedtiming live latency vs the replay's; WS pending-message timing of live fills and of replay fills live missed / also got.
  who       touch vs ladder lanes live; makers / takers (REST matches) in the trades the replay credited to us but live missed.
  reasons   every real touch trade 0-240 s: maker markouts and why we were not filled (cancelled / pulled by prediction / ...).
  queue     touch trades we rested at but missed: queue ahead vs traded volume, places lost by cancel + re-place and why.
  frontage  lower bound on how long the front-of-queue orders that took touch trades had been resting.
Usage: python3 -m predictfun.research.pf_live_vs_replay kernel | compare | lifetime | markout | phantom | feedtiming | who | reasons | queue | frontage
"""
import collections, gzip, json, os, re, sys
import numpy as np, pandas as pd

CAP = '/home/yuqing/pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109'
KX = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108-kernel'
RP = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108'
RUNS = (('v6', '1794108465172'), ('basis', '1794147344737'))
CUTOFF_MS = 1794178800000          # 2026-11-08 23:00Z
NET_CAP = 150.0
OUT = f'{RP}/live_vs_replay'


def cohort():
    sc = json.load(open(f'{RP}/scope.json'))
    return {w['market']: (w['ws'], w['run']) for w in sc['windows']}


def kernel(run, rid):
    """Per-order kernel records of one run (cohort windows only)."""
    L = json.load(open(f'{CAP}/{run}/stageA/ledger.json'))
    coh = cohort()
    ix2m = {w['ix']: w['market'] for w in L['windows'] if w['market'] in coh}
    why = {}
    for line in open(f'{CAP}/{run}/live.log.segment'):
        if line.startswith('{"refused"'):
            v = json.loads(line); why[v['refused']] = re.sub(r'\s*\(.*', '', v['why'])
    O = {}; oid2cid = {}; fills = []; tok = {}
    for line in open(f'{KX}/{run}.kernel.jsonl'):
        if '"step"' not in line[-40:]:
            continue
        f = json.loads(line)['frame']; inp = f.get('input'); at = f['at'] / 1e6
        if not isinstance(inp, dict):
            continue
        k0 = next(iter(inp)); v = inp[k0]
        if not isinstance(v, dict):
            continue
        if k0 == 'Meta' and 'up' in v:
            tok[v['market']] = (tuple(v['up']), tuple(v['down']))
        elif k0 == 'Control' and 'Place' in v:
            p = v['Place']
            if p['market'] in ix2m:
                O[p['cid']] = dict(cid=p['cid'], market=ix2m[p['market']], up=p['side'] == 'BuyUp', price=p['price'] / 1e4,
                                   size=p['size'] / 1e6, t_place=at, refused=why.get(p['cid']))
        elif k0 == 'Exec':
            k1 = next(iter(v)); e = v[k1]
            if not isinstance(e, dict):
                continue
            if k1 == 'Signed' and e['cid'] in O:
                oid2cid[tuple(e['order_id'])] = e['cid']; O[e['cid']]['t_signed'] = at
            elif k1 == 'Accepted' and e.get('cid') in O:
                O[e['cid']]['t_accepted'] = at
            elif k1 in ('UserCancel', 'CancelPendingAudit', 'CancelQuantityReconciled', 'CancelAuditComplete'):
                c = oid2cid.get(tuple(e['order_id']))
                if c is not None:
                    O[c].setdefault('t_' + k1, at)
            elif k1 == 'Fill':
                c = oid2cid.get(tuple(e['order_id']))
                if c is not None:
                    fills.append(dict(cid=c, at=at, match_ms=e['match_time'] / 1e6, size=e['size'] / 1e6, price=e['price'] / 1e6, status=e['status']))
    D = pd.DataFrame(O.values()); F = pd.DataFrame(fills)
    D['run'] = run
    return D, F


def rebuild_netcap(D, F):
    """worst_long / worst_short of risk::exposure at each NetCap refusal (booked net from non-failed fills)."""
    rows = []
    D = D.copy()
    D['t_end_live'] = D[['t_UserCancel', 't_CancelPendingAudit']].min(axis=1) if 't_UserCancel' in D else D['t_CancelPendingAudit']
    rec = D[['t_CancelQuantityReconciled', 't_CancelAuditComplete']].min(axis=1)
    D['t_end_audit'] = rec
    ok = F[F.status != 'Failed'] if len(F) else F
    filled = ok.groupby('cid')['size'].sum() if len(ok) else pd.Series(dtype=float)
    D['filled'] = D.cid.map(filled).fillna(0.0)
    for _, r in D[D.refused == 'risk NetCap'].iterrows():
        m = D[(D.market == r.market) & D.refused.isna() & (D.t_place < r.t_place)]
        same = m[m.up == r.up]
        live = same[(same.t_end_live.isna()) | (same.t_end_live > r.t_place)]
        aud = same[(same.t_end_live <= r.t_place) & ((same.t_end_audit.isna()) | (same.t_end_audit > r.t_place))]
        fl = ok[ok.cid.isin(m.cid) & (ok['at'] <= r.t_place)] if len(ok) else ok
        if len(fl):
            fl = fl.merge(D[['cid', 'up']], on='cid')
            net_side = float(fl[fl.up == r.up]['size'].sum() - fl[fl.up != r.up]['size'].sum())
        else:
            net_side = 0.0
        q_live = float((live['size'] - live['filled']).clip(lower=0).sum()); q_aud = float((aud['size'] - aud['filled']).clip(lower=0).sum())
        rows.append(dict(cid=r.cid, market=r.market, t=r.t_place, up=r.up, size=r['size'], net_side=net_side, open_live=q_live,
                         audit_pending=q_aud, n_audit=len(aud), worst=net_side + q_live + q_aud + r['size']))
    return pd.DataFrame(rows)


def main_kernel():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for run, rid in RUNS:
        D, F = kernel(run, rid)
        D.to_parquet(f'{OUT}/kernel_orders_{run}.parquet', index=False); F.to_parquet(f'{OUT}/kernel_fills_{run}.parquet', index=False)
        canc = D[D.t_CancelPendingAudit.notna()]
        rec = canc.t_CancelQuantityReconciled.notna()
        aud = (canc.t_CancelAuditComplete - canc.t_CancelPendingAudit) / 1000.0
        recl = (canc.t_CancelQuantityReconciled - canc.t_CancelPendingAudit)[rec]
        N = rebuild_netcap(D, F); N.to_parquet(f'{OUT}/netcap_{run}.parquet', index=False)
        exceeds = N.worst > NET_CAP + 1e-9
        res[run] = dict(orders=len(D), refused=dict(collections.Counter(D.refused.dropna())), cancels_audited=len(canc),
                        reconciled_share=round(float(rec.mean()), 4), reconcile_ms_median=round(float(recl.median()), 1),
                        audit_s_quantiles={q: round(float(aud.quantile(q)), 2) for q in (0.1, 0.5, 0.9, 0.99)},
                        netcap_refusals=len(N), netcap_rebuilt_over_cap=int(exceeds.sum()),
                        netcap_mean=dict(net_side=round(float(N.net_side.mean()), 1), open_live=round(float(N.open_live.mean()), 1),
                                         audit_pending=round(float(N.audit_pending.mean()), 1), worst=round(float(N.worst.mean()), 1)),
                        netcap_over_cap_without_audit_pending=int(((N.net_side + N.open_live + N['size']) > NET_CAP + 1e-9).sum()))
    json.dump(res, open(f'{OUT}/kernel_summary.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
    return 0


ARM_STEM = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_CAP10_XNA_XMIN1_XV_RO60_S240__mid_szAM-5-25-95'
TOL = 250.0   # ms either side of a replay fill when asking whether live rested there


def lines(path, skip_header=False):
    """Complete lines of a gzip stream; an open (prefix) capture ends at its last complete line (EOFError), as --allow-prefix."""
    with gzip.open(path, 'rt') as f:
        if skip_header:
            next(f)
        try:
            for l in f:
                if l.endswith('\n'):
                    yield l
        except EOFError:
            return


def core_orders(run, rid, markets):
    """Core-level order lifecycle keyed (market, id) from the captured outputs (requests) and inputs (venue / kernel feedback),
    with the kernel refusal reason aligned per market by sequence (core place k <-> kernel Control:Place k)."""
    pl, cx, dec = {}, {}, collections.defaultdict(list)
    if True:
        for l in lines(f'{CAP}/{run}/run-{rid}/outputs.jsonl.gz'):
            if l.startswith('{"Decision"'):
                m = int(l[22:l.index(",", 22)])
                if m in markets:
                    d = json.loads(l)['Decision']; dec[m].append((d['t'], [x[0] for x in d['lanes']]))
            elif l.startswith('{"Request":{"Place"'):
                v = json.loads(l)['Request']['Place']
                if v['market'] in markets:
                    pl[(v['market'], v['id'])] = dict(market=v['market'], id=v['id'], lane=v['lane'], tick=v['tick'], size=v['size'], t_req=v['t'])
            elif l.startswith('{"Request":{"Cancel"'):
                v = json.loads(l)['Request']['Cancel']
                cx.setdefault((v['market'], v['id']), v['t'])
    ev = collections.defaultdict(dict); filled = collections.defaultdict(float)
    if True:
        for l in lines(f'{CAP}/{run}/run-{rid}/inputs.jsonl.gz', skip_header=True):
            k = l[2:l.index('"', 2)]
            if k not in ('Accepted', 'Answered', 'CancelDone', 'Throttled', 'Rejected', 'Filled', 'Hold'):
                continue
            v = json.loads(l)[k]
            if v.get('market') not in markets:
                continue
            if k == 'Hold':
                dec[v['market']].append((v['known_ms'], 'HOLD' if v['held'] else 'RESUME')); continue
            key = (v['market'], v['id'])
            ev[key].setdefault(k, v['known_ms'])
            if k == 'Filled':
                filled[key] += v['qty']; ev[key]['Filled_last'] = v['known_ms']
    P = pd.DataFrame(pl.values())
    for k in ('Accepted', 'Answered', 'CancelDone', 'Throttled', 'Rejected', 'Filled', 'Filled_last'):
        P['t_' + k] = [ev[(a, b)].get(k, np.nan) for a, b in zip(P.market, P.id)]
    P['t_cancel'] = [cx.get((a, b), np.nan) for a, b in zip(P.market, P.id)]
    P['filled'] = [filled[(a, b)] for a, b in zip(P.market, P.id)]
    K = pd.read_parquet(f'{OUT}/kernel_orders_{run}.parquet').sort_values(['market', 't_place', 'cid'], kind='stable')
    P = P.sort_values(['market', 't_req', 'id'], kind='stable')
    P['k'] = P.groupby('market').cumcount(); K['k'] = K.groupby('market').cumcount()
    P = P.merge(K[['market', 'k', 'cid', 'refused', 'size', 'up']].rename(columns={'size': 'ksize'}), on=['market', 'k'], how='left')
    bad = ((P.ksize - P['size']).abs() > 1e-6) | (P.up != P.lane.isin([0, 2]))
    P['align_ok'] = ~bad
    D = {m: (np.array([x[0] for x in v]), [x[1] for x in v]) for m, v in dec.items()}
    return P, D


def live_windows():
    rows = []
    coh = cohort()
    for run, _ in RUNS:
        L = json.load(open(f'{CAP}/{run}/stageA/ledger.json'))
        for w in L['windows']:
            if w['market'] in coh:
                rows.append(dict(run=run, market=w['market'], ws=w['window_start_s'], pnl=w['trading_pnl_excluding_rebates'], legs=w['fill_legs'],
                                 shares=w['up_shares'] + w['down_shares'], up=w['up_shares'], down=w['down_shares'],
                                 refused=w['refused'], places=w['control_places']))
    return pd.DataFrame(rows)


def classify(f, P, D):
    """What live did on the replay fill's side and price around its time."""
    up = f.side == 0; tick = f.pt; T = f.ws_ms + f.t * 1000.0   # the replay's pt is the bought token's own price on both sides (as live)
    o = P[(P.market == f.mid) & (P.lane.isin([0, 2] if up else [1, 3])) & (P.tick == tick)]
    end = o[['t_CancelDone', 't_Filled_last']].max(axis=1).fillna(f.ws_ms + 300_000.0)
    rest = o[o.t_Accepted.notna() & (o.t_Accepted <= T + TOL) & (end >= T - TOL)]
    if len(rest):
        return 'live_rested_and_filled' if (rest.filled > 0).any() else 'live_rested_no_fill'
    near = o[(o.t_req >= T - 2000.0) & (o.t_req <= T + TOL)]
    if near.refused.notna().any():
        return 'refused_' + near.refused.dropna().iloc[-1].replace('risk ', '').replace(' ', '_')
    if near.t_Rejected.notna().any():
        return 'rejected_by_venue'
    if len(near):
        return 'placed_not_live_in_time'
    ts, lanes = D.get(f.mid, (np.array([]), []))
    i = np.searchsorted(ts, T, side='right') - 1
    while i >= 0 and isinstance(lanes[i], str):
        if lanes[i] == 'HOLD':
            return 'live_on_hold'
        i -= 1
    if i < 0:
        return 'no_live_decision'
    want = [lanes[i][j] for j in ((0, 2) if up else (1, 3))]
    if tick in want:
        return 'wanted_not_sent'
    return 'live_quoted_elsewhere' if any(x is not None for x in want) else 'live_quoted_nothing_on_side'


def main_compare():
    coh = cohort(); LW = live_windows()
    R = {}
    for tm in ('J', 'X'):
        base = f'{RP}/research/step4/ladder_tokyoreal_custom_e2_preds_t60k_lw{tm}/20261108'
        F = pd.read_parquet(f'{base}/fills/{ARM_STEM}.parquet'); W = pd.read_parquet(f'{base}/win/{ARM_STEM}.parquet')
        R[tm] = (F[F.mid.isin(coh)].copy(), W[W.mid.isin(coh)].copy())
    out = dict(cohort_windows=len(coh), live={}, replay={}, attribution={})
    for run in ('v6', 'basis', 'all'):
        lw = LW if run == 'all' else LW[LW.run == run]
        out['live'][run] = dict(windows=len(lw), pnl=round(float(lw.pnl.sum()), 2), windows_with_fills=int((lw.legs > 0).sum()),
                                legs=int(lw.legs.sum()), shares=round(float(lw.shares.sum()), 1), places=int(lw.places.sum()), refused=int(lw.refused.sum()))
        mk = set(lw.market)
        for tm, (F, W) in R.items():
            f, w = F[F.mid.isin(mk)], W[W.mid.isin(mk)]
            out['replay'].setdefault(tm, {})[run] = dict(windows=len(w), pnl=round(float(w.pnl.sum()), 2), mark_at_stop=round(float(w.mtm_stop.sum()), 2),
                                                        windows_with_fills=int((w.n_fills > 0).sum()), fills=len(f), shares=round(float(f.q.sum()), 1),
                                                        shares_0_60=round(float(f[f.t < 60].q.sum()), 1), shares_60_240=round(float(f[f.t >= 60].q.sum()), 1),
                                                        zombie_fill_shares=round(float(f[f.zombie].q.sum()), 1))
    rid = dict(RUNS); CL = []
    for run in ('v6', 'basis'):
        mk = set(LW[LW.run == run].market)
        P, D = core_orders(run, rid[run], mk)
        out.setdefault('alignment', {})[run] = dict(core_places=len(P), aligned=int(P.align_ok.sum()))
        P.to_parquet(f'{OUT}/core_orders_{run}.parquet', index=False)
        for tm, (F, W) in R.items():
            f = F[F.mid.isin(mk)].copy(); f['ws_ms'] = f.mid.map(lambda m: coh[m][0] * 1000.0)
            f['why'] = [classify(r, P, D) for r in f.itertuples()]
            f['run'] = run; f['timing'] = tm; CL.append(f)
    C = pd.concat(CL)
    C.to_parquet(f'{OUT}/replay_fill_attribution.parquet', index=False)
    for tm in ('J', 'X'):
        c = C[C.timing == tm]
        g = c.groupby('why').agg(fills=('q', 'size'), shares=('q', 'sum'))
        g['share_pct'] = (100 * g.shares / g.shares.sum()).round(1)
        out['attribution'][tm] = {k: dict(fills=int(v.fills), shares=round(float(v.shares), 1), pct=float(v.share_pct)) for k, v in g.sort_values('shares', ascending=False).iterrows()}
    json.dump(out, open(f'{OUT}/compare_summary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def main_lifetime():
    """Replay fills where live held the same order (side, price) resting at the replay's trade time: live's fill share by that live
    order's lifetime (accepted -> cancel done / last fill), J and X pooled; and live filled shares with a replay fill (same side and
    price within 1.5 s)."""
    C = pd.read_parquet(f'{OUT}/replay_fill_attribution.parquet')
    P = {r: pd.read_parquet(f'{OUT}/core_orders_{r}.parquet') for r, _ in RUNS}
    rows = []
    for r in C[C.why.isin(['live_rested_no_fill', 'live_rested_and_filled'])].itertuples():
        p = P[r.run]; T = r.ws_ms + r.t * 1000.0
        o = p[(p.market == r.mid) & (p.lane.isin([0, 2] if r.side == 0 else [1, 3])) & (p.tick == r.pt) & p.t_Accepted.notna()].copy()
        o['end'] = o[['t_CancelDone', 't_Filled_last']].max(axis=1).fillna(r.ws_ms + 300_000.0)
        ins = o[(o.t_Accepted <= T) & (o.end >= T)]
        rows.append(dict(timing=r.timing, filled=r.why == 'live_rested_and_filled', q=r.q, inside=len(ins) > 0,
                         life=float((ins.end - ins.t_Accepted).max()) if len(ins) else np.nan,
                         age=float((T - ins.t_Accepted).max()) if len(ins) else np.nan, to_cancel=float((ins.end - T).max()) if len(ins) else np.nan))
    R = pd.DataFrame(rows); I = R[R.inside].copy()
    I['life_b'] = pd.cut(I.life, [0, 50, 100, 200, 500, 1e9], labels=['<50ms', '50-100ms', '100-200ms', '200-500ms', '>500ms'])
    tab = {str(k): dict(fills=len(d), shares=round(float(d.q.sum()), 1), live_filled_share=round(float((d.q * d.filled).sum() / d.q.sum()), 3))
           for k, d in I.groupby('life_b', observed=True)}
    mono = [v['live_filled_share'] for v in tab.values()]
    rev = {}
    for tm in ('J', 'X'):
        c = C[C.timing == tm]; tot = hit = 0.0
        for run, _ in RUNS:
            for x in P[run][P[run].filled > 0].itertuples():
                cc = c[(c.mid == x.market) & (c.side == (0 if x.lane in (0, 2) else 1)) & (c.pt == x.tick)]
                tot += x.filled; hit += x.filled * bool(((cc.ws_ms + cc.t * 1000.0 - x.t_Filled).abs() <= 1500.0).any())
        rev[tm] = dict(live_filled_shares=round(tot, 1), with_replay_fill=round(hit, 1))
    out = dict(by_live_lifetime=tab, live_filled_share_increasing=bool(all(a <= b for a, b in zip(mono, mono[1:]))),
               medians={w: dict(live_age_ms=round(float(I[I.filled == f].age.median()), 1), live_ms_to_cancel=round(float(I[I.filled == f].to_cancel.median()), 1))
                        for w, f in (('no_fill', False), ('filled', True))},
               not_inside_shares={tm: round(float(R[(R.timing == tm) & ~R.inside].q.sum()), 1) for tm in ('J', 'X')}, live_fills_vs_replay=rev)
    json.dump(out, open(f'{OUT}/lifetime_summary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def main_markout():
    """Replay fills by order state at the fill (cancel-pending, else age since activation): shares, settle, +1 s / +60 s markout per
    share, on the 9 consumed in-sample days (i2J / i2X, v3 arm) and on the 220 live windows; live booked fills by live order lifetime."""
    from .pf_inv_report import stem, run_dir, DAYS
    from .pf_ladder_v3_build import ARM
    def prof(f, edges, labels):
        b = np.where(f.zombie, 'cancel-pending', pd.cut(f.age_ms, edges, labels=labels).astype(str))
        g = f.assign(b=b).groupby('b').agg(fills=('q', 'size'), shares=('q', 'sum'), settle=('settle', 'sum'), mo1=('mo1', 'sum'), mo60=('mo60', 'sum'))
        return {k: dict(fills=int(v.fills), shares=round(float(v.shares), 1), share_pct=round(float(100 * v.shares / g.shares.sum()), 1),
                        settle_usd=round(float(v.settle), 2), settle_c_sh=round(float(100 * v.settle / v.shares), 2),
                        mo1_usd=round(float(v.mo1), 2), mo1_c_sh=round(float(100 * v.mo1 / v.shares), 2), mo60_c_sh=round(float(100 * v.mo60 / v.shares), 2))
                for k, v in g.iterrows()}
    out = dict(in_sample_9d={}, live_windows={}, live_fills={})
    for tm in ('J', 'X'):
        F = pd.concat([pd.read_parquet(f"{run_dir('i2' + tm, d)}/fills/{stem(ARM)}.parquet") for d in DAYS])
        out['in_sample_9d'][tm] = prof(F, [-1, 50, 100, 200, 500, 1e12], ['<50ms', '50-100ms', '100-200ms', '200-500ms', '>500ms'])
        y = F[F.zombie | (F.age_ms < 100)]
        out['in_sample_9d'][tm + '_young_or_cancel_pending_share_of_mo1'] = round(float(y.mo1.sum() / F.mo1.sum()), 3)
        C = pd.read_parquet(f'{OUT}/replay_fill_attribution.parquet')
        out['live_windows'][tm] = prof(C[C.timing == tm], [-1, 100, 500, 1e12], ['<100ms', '100-500ms', '>500ms'])
    coh = cohort(); rows = []
    for run, _ in RUNS:
        P = pd.read_parquet(f'{OUT}/core_orders_{run}.parquet').set_index('cid')
        for f in json.load(open(f'{CAP}/{run}/stageA/ledger.json'))['fills']:
            if f['market'] in coh and f['booked'] and f['cid'] in P.index:
                p = P.loc[f['cid']]
                rows.append(dict(sh=f['shares'], m1=f['markout_1s'] if f['markout_1s_status'] == 'ok' else np.nan,
                                 m60=f['markout_60s'] if f['markout_60s_status'] == 'ok' else np.nan, pnl=f['shares'] * (f['payout'] - f['price']),
                                 life=np.nanmax([p.t_CancelDone, p.t_Filled_last]) - p.t_Accepted))
    R = pd.DataFrame(rows); R['b'] = pd.cut(R.life, [-1e9, 100, 500, 2000, 1e12], labels=['<100ms', '100-500ms', '0.5-2s', '>2s'])
    for k, d in R.groupby('b', observed=False):
        out['live_fills'][str(k)] = dict(fills=len(d), shares=round(float(d.sh.sum()), 1),
                                        mo1_c_sh=round(float(100 * (d.sh * d.m1).sum() / d.sh[d.m1.notna()].sum()), 2) if len(d) else None,
                                        settle_usd=round(float(d.pnl.sum()), 2))
    out['live_fills']['all_mo1_c_sh'] = round(float(100 * (R.sh * R.m1).sum() / R.sh[R.m1.notna()].sum()), 2)
    json.dump(out, open(f'{OUT}/markout_by_age.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def main_phantom():
    """NetCap fix options on the 11-08 refusals: refusals left if a cancelled order counted only for W ms after its cancel was
    confirmed (the 5 s REST audit stays the backstop), or with a higher net_shares; and late fills (first fill notice after the
    cancel confirmation) a window W would not have covered."""
    res = {}
    for run, _ in RUNS:
        D = pd.read_parquet(f'{OUT}/kernel_orders_{run}.parquet'); F = pd.read_parquet(f'{OUT}/kernel_fills_{run}.parquet')
        ok = F[F.status != 'Failed']
        D['filled'] = D.cid.map(ok.groupby('cid')['size'].max()).fillna(0.0); D['t_first_fill'] = D.cid.map(ok.groupby('cid')['at'].min())
        D['t_live_end'] = D[['t_UserCancel', 't_CancelPendingAudit']].min(axis=1)
        D['t_rec'] = D[['t_CancelQuantityReconciled', 't_CancelAuditComplete']].min(axis=1)
        late = D[D.t_first_fill.notna() & D.t_live_end.notna() & (D.t_first_fill > D.t_live_end)]; dl = late.t_first_fill - late.t_live_end
        ref = D[D.refused == 'risk NetCap']; by_m = {k: g for k, g in D[D.refused.isna()].groupby(['market', 'up'])}
        fo = ok.merge(D[['cid', 'market', 'up']], on='cid')
        def worst(r, W=None):
            g = by_m.get((r.market, r.up))
            if g is None:
                return r['size']
            g = g[g.t_place < r.t_place]
            live = g[g.t_live_end.isna() | (g.t_live_end > r.t_place)]
            aud = g[(g.t_live_end <= r.t_place) & (g.t_rec.isna() | (g.t_rec > r.t_place))]
            if W is not None:
                aud = aud[(r.t_place - aud.t_live_end) < W]
            fl = fo[(fo.market == r.market) & (fo['at'] <= r.t_place)]
            return (fl[fl.up == r.up]['size'].sum() - fl[fl.up != r.up]['size'].sum() + (live['size'] - live.filled).clip(lower=0).sum()
                    + (aud['size'] - aud.filled).clip(lower=0).sum() + r['size'])
        base = np.array([worst(r) for _, r in ref.iterrows()])
        o = dict(netcap_refusals=len(ref), rebuilt_over_cap_now=int((base > NET_CAP).sum()), late_first_fill_notices=len(late),
                 late_notice_after_cancel_ms=dict(p50=round(float(dl.median()), 0), p90=round(float(dl.quantile(0.9)), 0), max=round(float(dl.max()), 0)))
        for W in (500, 1000, 2000, 3000, 5000):
            w = np.array([worst(r, W) for _, r in ref.iterrows()])
            o[f'window_{W}ms'] = dict(refusals_left=int((w > NET_CAP).sum()), late_fills_uncovered=int((dl > W).sum()))
        for cap in (200, 300, 400):
            o[f'net_shares_{cap}'] = dict(refusals_left=int((base > cap).sum()))
        res[run] = o
    json.dump(res, open(f'{OUT}/phantom_fix_options.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
    return 0


def pending_increments(run, rid, markets):
    """Positive changes of the Predict WS 'pending' (settlementsPending, summed over makers per price) per (market, price tick), at the
    live bot's receipt time. A sweep's legs at one price arrive as one increment."""
    inc = collections.defaultdict(list); last = {}
    for l in lines(f'{CAP}/{run}/run-{rid}/inputs.jsonl.gz', skip_header=True):
        if l.startswith('{"Frame"'):
            m = int(l[19:l.index(',', 19)])
            if m not in markets:
                continue
            d = json.loads(l)['Frame']; cur = {(a, b): v for a, b, v in d['pending']}; prev = last.get(m, {})
            for k, v in cur.items():
                if v - prev.get(k, 0.0) > 1e-9:
                    inc[(m, k[1])].append((d['known_ms'], v - prev.get(k, 0.0)))
            last[m] = cur
    return inc


def main_feedtiming():
    """Is the short-lived-fill gap live infrastructure or the replay? (1) live latency vs the replay's (decision -> accepted 40 ms,
    cancel 22 ms); (2) live's own fills: when the WS pending message for the fill reached live, relative to the order's acceptance /
    cancel; (3) replay fills where live held the same order resting: when the pending message for the replay's trade (our price or
    through it, nearest to the replay time within 400 ms) reached live, relative to that live order, missed vs also-filled."""
    out = dict(latency={}, live_fills={}, replay_fills={})
    C = pd.read_parquet(f'{OUT}/replay_fill_attribution.parquet'); R1 = []; R2 = []
    for run, rid in RUNS:
        P = pd.read_parquet(f'{OUT}/core_orders_{run}.parquet'); mk = set(P.market)
        a = (P.t_Accepted - P.t_req).dropna(); c = (P.t_CancelDone - P.t_cancel).dropna()
        out['latency'][run] = dict(decision_to_accepted_ms={q: round(float(a.quantile(q)), 1) for q in (0.1, 0.5, 0.9, 0.99)},
                                   cancel_to_done_ms={q: round(float(c.quantile(q)), 1) for q in (0.5, 0.9)})
        inc = pending_increments(run, rid, mk)
        for m, i, kn, q in [(r.market, r.id, r.t_Filled, r.filled) for r in P[P.filled > 0].itertuples()]:
            o = P[(P.market == m) & (P.id == i)].iloc[0]; val = q * o.tick / 100.0
            cand = [t for t, dv in inc.get((m, o.tick), []) if dv >= val - max(0.011 * val, 0.01) and o.t_req - 200 <= t <= kn + 3000]
            if cand:
                W = min(cand); R1.append(dict(W_A=W - o.t_Accepted, W_C=W - o.t_CancelDone))
        for r in C[(C.run == run) & C.why.isin(['live_rested_no_fill', 'live_rested_and_filled'])].itertuples():
            T = r.ws_ms + r.t * 1000.0
            o = P[(P.market == r.mid) & (P.lane.isin([0, 2] if r.side == 0 else [1, 3])) & (P.tick == r.pt) & P.t_Accepted.notna()].copy()
            o['end'] = o[['t_CancelDone', 't_Filled_last']].max(axis=1).fillna(r.ws_ms + 300_000.0)
            ins = o[(o.t_Accepted <= T) & (o.end >= T)]
            ticks = [r.pt - k for k in range(6)] if r.side == 0 else [r.pt + k for k in range(6)]
            ws = [t for tk in ticks for t, _ in inc.get((r.mid, tk), []) if T - 400 <= t <= T + 400]
            if len(ins) and ws:
                A = ins.t_Accepted.max(); E = ins.loc[ins.t_Accepted.idxmax(), 'end']; W = min(ws, key=lambda t: abs(t - T))
                R2.append(dict(filled=r.why == 'live_rested_and_filled', W_A=W - A, W_E=W - E))
    R1 = pd.DataFrame(R1); R2 = pd.DataFrame(R2); cc = R1[R1.W_C.notna()]
    out['live_fills'] = dict(n=len(R1), pending_minus_accepted_ms={q: round(float(R1.W_A.quantile(q)), 0) for q in (0.1, 0.5, 0.9)},
                             cancelled_n=len(cc), pending_after_cancel_done=int((cc.W_C > 0).sum()))
    for f, lab in ((False, 'live_missed'), (True, 'live_also_filled')):
        x = R2[R2.filled == f]
        out['replay_fills'][lab] = dict(n=len(x), pending_minus_live_accepted_ms={q: round(float(x.W_A.quantile(q)), 0) for q in (0.1, 0.25, 0.5, 0.75)},
                                        within_60ms_of_accept_or_after_gone=round(float(((x.W_A < 60) | (x.W_E > 0)).mean()), 3))
    json.dump(out, open(f'{OUT}/feed_timing.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def main_who():
    """Ladder vs touch in live (by lane: places, refusals, lifetime, fills, +1 s markout) and who traded where the replay credited us:
    REST matches of 2026-11-08 (cohort hours / markets only) -> taker and maker addresses; our address from live fills' order hashes;
    for J replay fills live missed (resting, no fill), the real trades on our side at our price or through it (exec second T-1.5..T+1)."""
    from . import pf_io as io
    sc = json.load(open(f'{RP}/scope.json')); coh = {w['market']: w['ws'] for w in sc['windows']}
    assert all(h.startswith('20261108_') for h in sc['hours'])
    legs = {}
    for h in sc['hours']:
        for recv, params, path, pl in io.rest_rows(h, ('/v1/orders/matches',)):
            for x in pl.get('data') or []:
                mid = (x.get('market') or {}).get('id')
                if mid not in coh:
                    continue
                for m in x['makers']:
                    k = (x['transactionHash'], m['hash'])
                    if k not in legs:
                        side, py = io.yes_side(m)
                        legs[k] = dict(mid=mid, exec_s=int(io.iso_s(x['executedAt'])), tx=x['transactionHash'], taker=x['taker']['signer'].lower(),
                                       maker=m['signer'].lower(), maker_hash=m['hash'].lower(), yes_side=side, ptick=int(round(py * 100)),
                                       shares=int(m['amount']) / io.E18)
    M = pd.DataFrame(legs.values()); M.to_parquet(f'{OUT}/matches_20261108.parquet', index=False)
    ours = {f['order'].lower() for run, _ in RUNS for f in json.load(open(f'{CAP}/{run}/stageA/ledger.json'))['fills'] if f['market'] in coh}
    me = M[M.maker_hash.isin(ours)].maker.value_counts()
    ME = me.index[0]
    M['t_in'] = M.exec_s - M.mid.map(coh); W = M[(M.t_in >= 0) & (M.t_in < 240)]
    def top(g, k=6):
        v = g.sort_values(ascending=False)
        return dict(n=len(v), top5_pct=round(float(100 * v.head(5).sum() / v.sum()), 1),
                    top=[(a[:6] + '..' + a[-4:], round(float(100 * x / v.sum()), 1)) for a, x in v.head(k).items()])
    out = dict(our_address=ME, our_legs=int(me.iloc[0]), our_share_of_maker_volume_0_240s_pct=round(float(100 * W[W.maker == ME].shares.sum() / W.shares.sum()), 3),
               market_makers=top(W.groupby('maker').shares.sum()), market_takers=top(W.groupby('taker').shares.sum()))
    C = pd.read_parquet(f'{OUT}/replay_fill_attribution.parquet'); rows = []
    for r in C[(C.timing == 'J') & (C.why == 'live_rested_no_fill')].itertuples():
        T = (r.ws_ms + r.t * 1000.0) / 1000.0; side = 'bid' if r.side == 0 else 'ask'
        x = M[(M.mid == r.mid) & (M.yes_side == side) & M.exec_s.between(T - 1.5, T + 1.0) & ((M.ptick <= r.pt) if side == 'bid' else (M.ptick >= r.pt))]
        rows.append(x.assign(at_our_price=x.ptick == r.pt))
    X = pd.concat(rows).drop_duplicates(['tx', 'maker_hash']); A = X[X.at_our_price]
    topM = set(W.groupby('maker').shares.sum().sort_values(ascending=False).head(10).index)
    topT = set(W.groupby('taker').shares.sum().sort_values(ascending=False).head(10).index)
    out['missed_trades'] = dict(makers_at_our_price=top(A.groupby('maker').shares.sum()), takers=top(X.groupby('taker').shares.sum()),
                                market_top10_makers_share_at_our_price_pct=round(float(100 * A[A.maker.isin(topM)].shares.sum() / A.shares.sum()), 1),
                                market_top10_takers_share_pct=round(float(100 * X[X.taker.isin(topT)].shares.sum() / X.shares.sum()), 1),
                                we_are_maker_legs=int((X.maker == ME).sum()))
    out['takers_hitting_our_live_orders'] = top(M[M.maker == ME].groupby('taker').shares.sum())
    P = pd.concat([pd.read_parquet(f'{OUT}/core_orders_{r}.parquet') for r, _ in RUNS])
    P['life'] = P[['t_CancelDone', 't_Filled_last']].max(axis=1) - P.t_Accepted
    names = {0: 'up_touch', 1: 'down_touch', 2: 'up_rung', 3: 'down_rung'}
    out['live_by_lane'] = {names[ln]: dict(places=len(g), refused=int(g.refused.notna().sum()), accepted=int(g.t_Accepted.notna().sum()),
                                          life_ms_p50=round(float(g.life.median()), 0), life_ms_p90=round(float(g.life.quantile(0.9)), 0),
                                          filled_orders=int((g.filled > 0).sum()), filled_shares=round(float(g.filled.sum()), 1))
                           for ln, g in P.groupby('lane')}
    json.dump(out, open(f'{OUT}/who.json', 'w'), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 0


def live_tape(run, rid, markets, stop_ms):
    """Per cohort market from the live capture (Tokyo receipt clock): frames (t, best bid, best ask), WS pending increments per price,
    own fill notices (t, id, qty) and holds."""
    fr = collections.defaultdict(list); inc = collections.defaultdict(list); last = {}; fills = collections.defaultdict(list)
    holds = collections.defaultdict(list)
    for l in lines(f'{CAP}/{run}/run-{rid}/inputs.jsonl.gz', skip_header=True):
        if l.startswith('{"Advance"'):
            if float(l[21:l.index('}', 21)]) > stop_ms:
                break
            continue
        if l.startswith('{"Frame"'):
            m = int(l[19:l.index(',', 19)])
            if m not in markets:
                continue
            d = json.loads(l)['Frame']; t = d['known_ms']
            fr[m].append((t, d['bids'][0][0] if d['bids'] else np.nan, d['asks'][0][0] if d['asks'] else np.nan))
            cur = {(a, b): v for a, b, v in d['pending']}; prev = last.get(m, {})
            for k, v in cur.items():
                if v - prev.get(k, 0.0) > 1e-9:
                    inc[(m, k[1])].append(t)
            last[m] = cur
        elif l.startswith('{"Filled"') or l.startswith('{"Hold"'):
            k = 'Filled' if l[2] == 'F' else 'Hold'; v = json.loads(l)[k]
            if v.get('market') in markets:
                (fills if k == 'Filled' else holds)[v['market']].append((v['known_ms'], v['id'], v['qty']) if k == 'Filled' else (v['known_ms'], v['held']))
    F = {m: (np.array([x[0] for x in v]), np.array([x[1] for x in v], float), np.array([x[2] for x in v], float)) for m, v in fr.items()}
    return F, {k: np.array(v) for k, v in inc.items()}, fills, holds


def main_reasons():
    """Every real touch trade (makers filled at the pre-trade best bid / ask) in 0-240 s of the 220 windows: maker markouts and why we
    were not filled. Time = the live bot's receipt of the WS pending increment at that price (first within exec second -0.5..+1.5 s)
    minus 50 ms (sensitivity 0 / 100 ms). Status at that time, in order: we were filled in the trade; our order rested at the price
    (queue); our order there was cancelled <= 500 ms before (re-priced / pulled, by our decision right after the cancel request); an
    order for the price was in flight / refused; else our decision: other price on that side, or nothing - split into feed hold, stop
    (>= 240 s), reduce-only (>= 60 s, side does not reduce known net), CAP10 (< 60 s, |known net| >= 10, adding side), prediction
    (research-recomputed guard value <= the minute's pull line or no fresh value), else other guards (Binance moves, book)."""
    sc = json.load(open(f'{RP}/scope.json')); coh = {w['market']: w['ws'] * 1000.0 for w in sc['windows']}
    M = pd.read_parquet(f'{OUT}/matches_20261108.parquet'); ME = json.load(open(f'{OUT}/who.json'))['our_address']
    reg = pd.read_parquet(f'{RP}/research/step1/20261108/registry.parquet').set_index('mid')
    y_up = {m: {'Up': 1.0, 'Down': 0.0}.get(reg.outcome[m], np.nan) for m in reg.index}
    pv = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    PV = {m: (g.tau.values.astype(float), g.v_bid.values, g.v_ask.values) for m, g in pv.groupby('mid')}
    LINE = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']['90']
    rows = []
    for run, rid in RUNS:
        P, D = core_orders(run, rid, {m for m, (ws, r) in cohort().items() if r == run})
        mk = set(P.market)
        F, INC, FL, HO = live_tape(run, rid, mk, CUTOFF_MS + 120_000)
        lane_of = {(a, b): c for a, b, c in zip(P.market, P.id, P.lane)}
        T = M[M.mid.isin(mk)].copy(); T['t_in'] = T.exec_s - (T.mid.map(coh) / 1000.0); T = T[(T.t_in >= 0) & (T.t_in < 240)]
        T['ours'] = T.maker == ME
        E = T.groupby(['mid', 'tx', 'yes_side', 'ptick'], as_index=False).agg(exec_s=('exec_s', 'first'), shares=('shares', 'sum'), ours=('ours', 'any'))
        for e in E.itertuples():
            m = e.mid; ws = coh[m]; up = e.yes_side == 'bid'
            w = INC.get((m, e.ptick))
            if w is None:
                rows.append(dict(run=run, status='untimed', shares=e.shares)); continue
            w = w[(w >= e.exec_s * 1000 - 500) & (w <= e.exec_s * 1000 + 1500)]
            if not len(w):
                rows.append(dict(run=run, status='untimed', shares=e.shares)); continue
            W = float(w.min()); ft, fb, fa = F[m]; i = np.searchsorted(ft, W) - 1
            if i < 0:
                rows.append(dict(run=run, status='untimed', shares=e.shares)); continue
            touch = (fb[i] == e.ptick) if up else (fa[i] == e.ptick)
            if not touch:
                continue
            def mid_at(t):
                j = np.searchsorted(ft, t, side='right') - 1
                return (fb[j] + fa[j]) / 2.0 if j >= 0 else np.nan
            sg = 1.0 if up else -1.0
            mk_ = {h: sg * (mid_at(W + h * 1000) - e.ptick) for h in (1, 5, 60)}
            st = sg * (100 * y_up.get(m, np.nan) - e.ptick)
            res = {}
            for shift in (0, 50, 100):
                tm = W - shift; res[shift] = status_at(e, m, ws, tm, up, P, D, FL, HO, lane_of, PV, LINE)
            rows.append(dict(run=run, mid=m, t_in=(W - ws) / 1000.0, up=up, ptick=e.ptick, shares=e.shares, status=res[50], status_0=res[0], status_100=res[100],
                             mo1=mk_[1], mo5=mk_[5], mo60=mk_[60], settle=st))
    R = pd.DataFrame(rows); R.to_parquet(f'{OUT}/touch_trade_reasons.parquet', index=False)
    V = R[R.status != 'untimed']
    def agg(d):
        w = d.shares
        return pd.Series(dict(trades=len(d), shares=w.sum(), mo1_c=(w * d.mo1).sum() / w[d.mo1.notna()].sum(), mo5_c=(w * d.mo5).sum() / w[d.mo5.notna()].sum(),
                              mo60_c=(w * d.mo60).sum() / w[d.mo60.notna()].sum(), settle_c=(w * d.settle).sum() / w[d.settle.notna()].sum()))
    G = V.groupby('status').apply(agg, include_groups=False).sort_values('shares', ascending=False)
    G['share_pct'] = 100 * G.shares / G.shares.sum()
    out = dict(touch_trades=len(V), untimed=int((R.status == 'untimed').sum()), by_reason={k: {c: round(float(v[c]), 2) for c in G.columns} for k, v in G.iterrows()},
               stable_across_time_shift=round(float(((V.status == V.status_0) & (V.status == V.status_100)).mean()), 3),
               all=dict(mo1_c=round(float((V.shares * V.mo1).sum() / V.shares[V.mo1.notna()].sum()), 2), settle_c=round(float((V.shares * V.settle).sum() / V.shares[V.settle.notna()].sum()), 2)))
    json.dump(out, open(f'{OUT}/touch_trade_reasons.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def status_at(e, m, ws, tm, up, P, D, FL, HO, lane_of, PV, LINE):
    lanes = (0, 2) if up else (1, 3); touch_lane = 0 if up else 1
    if e.ours:
        return 'we_were_filled'
    o = P[(P.market == m) & P.lane.isin(lanes) & (P.tick == e.ptick)]
    end = o[['t_CancelDone', 't_Filled_last']].max(axis=1).fillna(ws + 300_000.0)
    if ((o.t_Accepted <= tm) & (end >= tm)).any():
        return 'resting_not_filled (queue)'
    c = o[(end < tm) & (end >= tm - 500) & o.t_cancel.notna()]
    if len(c):
        tc = float(c.t_cancel.max()); want = decision(D, m, tc + 0.5, touch_lane)
        if want == 'HOLD':
            return 'cancelled<=500ms: feed hold'
        if want is not None:
            return 'cancelled<=500ms: re-priced'
        return 'cancelled<=500ms: ' + why_none(e, m, ws, tc + 0.5, up, D, FL, HO, lane_of, PV, LINE)
    f = o[(o.t_req <= tm) & (o.t_req >= tm - 500)]
    if len(f):
        if f.refused.notna().any():
            return 'refused: ' + f.refused.dropna().iloc[-1].replace('risk ', '')
        return 'order in flight (placed, not yet live)'
    want = decision(D, m, tm, touch_lane)
    if want == 'HOLD':
        return 'not quoting: feed hold'
    if want is not None:
        return 'quoting another price'
    return 'not quoting: ' + why_none(e, m, ws, tm, up, D, FL, HO, lane_of, PV, LINE)


def decision(D, m, t, lane):
    ts, ln = D.get(m, (np.array([]), []))
    i = np.searchsorted(ts, t, side='right') - 1
    while i >= 0:
        x = ln[i]
        if isinstance(x, str):
            if x == 'HOLD':
                return 'HOLD'
            i -= 1; continue
        return x[lane]
    return None


def why_none(e, m, ws, t, up, D, FL, HO, lane_of, PV, LINE):
    rel = (t - ws) / 1000.0
    if rel >= 240:
        return 'stopped (>= 240 s)'
    net = sum((q if lane_of.get((m, i)) in (0, 2) else -q) for k, i, q in FL.get(m, []) if k <= t)
    reducing = (up and net < 0) or ((not up) and net > 0)
    if rel >= 60 and not reducing:
        return 'reduce-only phase (nothing to reduce on this side)'
    if rel < 60 and not reducing and abs(net) >= 10 - 1e-9:
        return 'CAP10 (inventory limit)'
    tau, vb, va = PV.get(m, (np.array([]), None, None))
    j = np.searchsorted(tau, t, side='right') - 1
    if j < 0 or t - tau[j] > 100 or not np.isfinite((vb if up else va)[j]):
        return 'prediction: no fresh guard value'
    v = (vb if up else va)[j]
    if v <= LINE[min(int(rel // 60), 3)]:
        return 'prediction: guard value below the pull line'
    return 'other guards (Binance move / book checks)'


def main_queue():
    """The touch trades where our order rested at the price but was not filled ('reasons' status queue): queue ahead at our join
    (displayed size at the price in the last live frame before our acceptance) vs the volume traded at the price from our join to the
    trade (REST matches, all makers, exec second >= join second); our order's age; whether we had held the price earlier in an unbroken
    run of our orders (gaps <= 500 ms) and lost the place by cancel + re-place, the reason of each such cancel (our decision at the cancel
    request: side pulled by the prediction line / other guards / inventory, re-priced, resized 5 <-> 25, or kept tick and size), and
    whether the run's first order would have been reached (its displayed queue <= the volume traded at the price since then)."""
    sc = json.load(open(f'{RP}/scope.json')); coh = {w['market']: w['ws'] * 1000.0 for w in sc['windows']}
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet'); Qm = R[R.status == 'resting_not_filled (queue)']
    M = pd.read_parquet(f'{OUT}/matches_20261108.parquet')
    pv = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    PV = {m: (g.tau.values.astype(float), g.v_bid.values, g.v_ask.values) for m, g in pv.groupby('mid')}
    LINE = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']['90']
    rows = []
    for run, rid in RUNS:
        q = Qm[Qm.run == run]; mk = set(q.mid)
        if not mk:
            continue
        P, _ = core_orders(run, rid, mk)
        dec = collections.defaultdict(list); books = collections.defaultdict(list); fills = collections.defaultdict(list); holds = collections.defaultdict(list)
        for l in lines(f'{CAP}/{run}/run-{rid}/outputs.jsonl.gz'):
            if l.startswith('{"Decision"'):
                m = int(l[22:l.index(',', 22)])
                if m in mk:
                    d = json.loads(l)['Decision']; dec[m].append((d['t'], d['lanes']))
        for l in lines(f'{CAP}/{run}/run-{rid}/inputs.jsonl.gz', skip_header=True):
            if l.startswith('{"Advance"'):
                if float(l[21:l.index('}', 21)]) > CUTOFF_MS + 120_000:
                    break
            elif l.startswith('{"Frame"'):
                m = int(l[19:l.index(',', 19)])
                if m in mk:
                    d = json.loads(l)['Frame']; books[m].append((d['known_ms'], dict(d['bids']), dict(d['asks'])))
            elif l.startswith('{"Filled"'):
                v = json.loads(l)['Filled']
                if v['market'] in mk:
                    fills[v['market']].append((v['known_ms'], v['id'], v['qty']))
        DT = {m: np.array([x[0] for x in v]) for m, v in dec.items()}; BT = {m: np.array([x[0] for x in v]) for m, v in books.items()}
        lane_of = {(a, b): c for a, b, c in zip(P.market, P.id, P.lane)}
        Dsimple = {m: (DT[m], [[x[0] for x in ln] for _, ln in dec[m]]) for m in dec}
        def size_at(m, t, up, tick):
            i = np.searchsorted(BT[m], t, side='left') - 1
            return float(books[m][i][1 if up else 2].get(tick, 0.0)) if i >= 0 else np.nan
        def lanes_at(m, t):
            i = np.searchsorted(DT[m], t + 0.01, side='right') - 1
            return dec[m][i][1] if i >= 0 else None
        for e in q.itertuples():
            m = e.mid; ws = coh[m]; W = ws + e.t_in * 1000.0; tm = W - 50.0; up = e.up
            o = P[(P.market == m) & P.lane.isin((0, 2) if up else (1, 3)) & (P.tick == e.ptick) & P.t_Accepted.notna()].sort_values('t_Accepted')
            end = o[['t_CancelDone', 't_Filled_last']].max(axis=1).fillna(ws + 300_000.0)
            cur = o[(o.t_Accepted <= tm) & (end >= tm)]
            if not len(cur):
                continue
            c0 = cur.iloc[-1]
            # unbroken run of our orders at this price before the current one (gap <= 500 ms between one ending and the next accepted)
            run_ = [c0]; prev = o[o.t_Accepted < c0.t_Accepted]; pend = end[prev.index]
            t_head = c0.t_Accepted
            for j in reversed(prev.index.tolist()):
                if pend[j] >= t_head - 500.0:
                    run_.append(o.loc[j]); t_head = min(t_head, o.loc[j].t_Accepted)
                else:
                    break
            first = run_[-1]
            reasons = []
            for x in run_[1:]:
                if pd.isna(x.t_cancel):
                    reasons.append('filled or ended'); continue
                ln = lanes_at(m, x.t_cancel)
                if ln is None:
                    reasons.append('unknown'); continue
                want = ln[x.lane]
                if want[0] is None:
                    reasons.append('pulled: ' + why_none(e, m, ws, x.t_cancel + 0.5, up, Dsimple, fills, holds, lane_of, PV, LINE))
                elif want[0] != x.tick:
                    reasons.append('re-priced')
                elif abs(want[1] - x['size']) > 1e-9:
                    reasons.append('resized')
                else:
                    reasons.append('kept tick and size (other)')
            vol = lambda t0: float(M[(M.mid == m) & (M.yes_side == ('bid' if up else 'ask')) & (M.ptick == e.ptick)
                                     & (M.exec_s >= np.floor(t0 / 1000.0)) & (M.exec_s <= np.floor(W / 1000.0))].shares.sum())
            q_cur = size_at(m, c0.t_Accepted, up, e.ptick); v_cur = vol(c0.t_Accepted)
            q_first = size_at(m, first.t_Accepted, up, e.ptick); v_first = vol(first.t_Accepted)
            rows.append(dict(run=run, mid=m, lane=int(c0.lane), size=c0['size'], age_ms=tm - c0.t_Accepted, q_ahead=q_cur, traded_since=v_cur,
                             trade_shares=e.shares, reached=v_cur >= q_cur, run_len=len(run_), run_age_ms=tm - first.t_Accepted,
                             q_ahead_first=q_first, traded_since_first=v_first, first_reached=v_first >= q_first,
                             cancel_reasons='|'.join(reasons), mo1=e.mo1, mo5=e.mo5, settle=e.settle))
    X = pd.DataFrame(rows); X.to_parquet(f'{OUT}/queue_misses.parquet', index=False)
    rs = collections.Counter(r for x in X.cancel_reasons for r in x.split('|') if r)
    out = dict(queue_misses=len(X), shares=round(float(X.trade_shares.sum()), 0),
               our_order_age_ms={k: round(float(X.age_ms.quantile(k)), 0) for k in (0.25, 0.5, 0.75)},
               queue_ahead_at_join_sh={k: round(float(X.q_ahead.quantile(k)), 0) for k in (0.25, 0.5, 0.75)},
               traded_at_price_since_join_sh={k: round(float(X.traded_since.quantile(k)), 0) for k in (0.25, 0.5, 0.75)},
               reached_but_not_filled=int(X.reached.sum()),
               lost_an_earlier_place_at_price=int((X.run_len > 1).sum()),
               run_age_ms_when_lost={k: round(float(X[X.run_len > 1].run_age_ms.quantile(k)), 0) for k in (0.25, 0.5, 0.75)} if (X.run_len > 1).any() else None,
               first_order_would_be_reached=int((X.run_len > 1).sum() and (X[X.run_len > 1].first_reached).sum()),
               cancels_in_lost_runs=dict(rs.most_common()), by_lane=X.groupby('lane').size().to_dict(),
               mo5_c_sh=round(float((X.trade_shares * X.mo5).sum() / X.trade_shares[X.mo5.notna()].sum()), 2))
    json.dump(out, open(f'{OUT}/queue_misses.json', 'w'), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 0


def main_frontage():
    """How long had the orders that took the trades been resting? For each touch trade (time = live receipt of the pending message
    - 50 ms), the front of the queue was already resting D seconds earlier if, at every grid step back to D, the displayed size at the
    price (live frame) exceeded the volume traded at the price since then (REST matches, exec-second resolution). Lower bound on
    the front orders' age, assuming no cancels at the front. Queue misses vs a sample of other 0-60 s touch trades."""
    sc = json.load(open(f'{RP}/scope.json')); coh = {w['market']: w['ws'] * 1000.0 for w in sc['windows']}
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet'); R = R[R.status != 'untimed']
    Qm = R[R.status == 'resting_not_filled (queue)']
    oth = R[(R.status != 'resting_not_filled (queue)') & (R.t_in < 60)].sample(n=3000, random_state=7)
    S = pd.concat([Qm.assign(grp='queue_miss'), oth.assign(grp='other_0_60s')])
    M = pd.read_parquet(f'{OUT}/matches_20261108.parquet')
    GRID = [0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60]
    rows = []
    for run, rid in RUNS:
        sub = S[S.run == run]; mk = set(sub.mid)
        books = collections.defaultdict(list)
        for l in lines(f'{CAP}/{run}/run-{rid}/inputs.jsonl.gz', skip_header=True):
            if l.startswith('{"Advance"'):
                if float(l[21:l.index('}', 21)]) > CUTOFF_MS + 120_000:
                    break
            elif l.startswith('{"Frame"'):
                m = int(l[19:l.index(',', 19)])
                if m in mk:
                    d = json.loads(l)['Frame']; books[m].append((d['known_ms'], dict(d['bids']), dict(d['asks'])))
        BT = {m: np.array([x[0] for x in v]) for m, v in books.items()}
        for e in sub.itertuples():
            m = e.mid; W = coh[m] + e.t_in * 1000.0 - 50.0; side = 'bid' if e.up else 'ask'
            mm = M[(M.mid == m) & (M.yes_side == side) & (M.ptick == e.ptick)]
            age = 0.0
            for D in GRID:
                i = np.searchsorted(BT[m], W - D * 1000.0, side='right') - 1
                if i < 0:
                    break
                size = books[m][i][1 if e.up else 2].get(e.ptick, 0.0)
                traded = float(mm[(mm.exec_s >= np.floor((W - D * 1000.0) / 1000.0)) & (mm.exec_s <= np.floor(W / 1000.0))].shares.sum())
                if size > traded:
                    age = D
                else:
                    break
            rows.append(dict(grp=e.grp, front_age_s=age, mo5=e.mo5, shares=e.shares))
    X = pd.DataFrame(rows)
    out = {}
    for g, d in X.groupby('grp'):
        out[g] = dict(n=len(d), front_age_s={q: float(d.front_age_s.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                      share_front_resting_ge_2s=round(float((d.front_age_s >= 2).mean()), 3), share_ge_10s=round(float((d.front_age_s >= 10).mean()), 3),
                      mo5_by_front_age={k: round(float((v.shares * v.mo5).sum() / v.shares[v.mo5.notna()].sum()), 2)
                                        for k, v in d.groupby(pd.cut(d.front_age_s, [-1, 0.5, 2, 10, 1e9], labels=['<=0.5s', '1-2s', '3-10s', '>10s']), observed=True)})
    json.dump(out, open(f'{OUT}/front_age.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit({'kernel': main_kernel, 'compare': main_compare, 'lifetime': main_lifetime, 'markout': main_markout, 'phantom': main_phantom, 'feedtiming': main_feedtiming, 'who': main_who, 'reasons': main_reasons, 'queue': main_queue, 'frontage': main_frontage}.get((sys.argv[1:] or [''])[0], lambda: sys.exit('usage: kernel | compare | lifetime'))())
