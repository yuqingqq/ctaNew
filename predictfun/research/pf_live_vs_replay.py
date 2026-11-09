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
Usage: python3 -m predictfun.research.pf_live_vs_replay kernel | compare | lifetime | markout | phantom
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


if __name__ == '__main__':
    sys.exit({'kernel': main_kernel, 'compare': main_compare, 'lifetime': main_lifetime, 'markout': main_markout, 'phantom': main_phantom}.get((sys.argv[1:] or [''])[0], lambda: sys.exit('usage: kernel | compare | lifetime'))())
