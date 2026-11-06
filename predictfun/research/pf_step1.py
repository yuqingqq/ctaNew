"""Step 1 (exploratory): registry, trade legs, book frames, trade-timing
reconciliation and book structure for one UTC day of Predict BTC 5m windows.

Usage: python3 -m predictfun.research.pf_step1 20261023
Outputs: data/predict_fun/research/step1/<day>/
  registry.parquet   one row per window (market id, ws, start/end price + first-seen receipt, outcome, fees, rewards)
  legs.parquet       de-duplicated maker legs (tx, maker hash) with executedAt second, first-seen receipt, YES side/price, shares
  books/part_HH.parquet  top-5 book per WS frame, [ws-300 s, we+5 s]
  pend_inc.parquet   positive settlementsPending increments per frame (YES side, price tick, amount)
  windows.parquet    per-window structure + trade stats (live window [ws, we))
  summary.json       day totals, reconciliation, statuses
"""
import sys, collections, math, bisect
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]
OUT = f'{io.OUT}/step1/{day}'
D0 = io.iso_s(f'{day[:4]}-{day[4:6]}-{day[6:]}T00:00:00Z')
HOURS_META = io.hours_of(day, extra_after=2, extra_before=1)
status = collections.Counter()


def tick(p):
    return int(round(p * 100))


# ---------------- Pass A+B: one REST pass: registry (categories) + maker legs (match pages) ----------------
reg = {}
legs = {}
signers = {}
for h in HOURS_META:
    for recv, params, path, pl in io.rest_rows(h, ('/v1/categories/', '/v1/orders/matches')):
        if path.startswith('/v1/categories/'):
            d = pl.get('data') or {}
            try:
                ws = io.iso_s(d['startsAt']); we = io.iso_s(d['endsAt'])
            except (KeyError, TypeError, ValueError):
                status['category_without_times'] += 1
                continue
            if not (D0 <= ws < D0 + 86400) or we - ws != 300:
                continue
            cvd = d.get('variantData') or {}
            for m in d.get('markets') or []:
                R = reg.get(m['id'])
                if R is None:
                    R = reg[m['id']] = dict(
                        mid=m['id'], slug=d.get('slug'), ws=ws, we=we, first_seen_ns=recv,
                        up_index=[o['indexSet'] for o in m.get('outcomes', []) if o.get('name') == 'Up'],
                        fee_bps=m.get('feeRateBps'), maker_fee_bps=m.get('makerFeeBps'), decimal_precision=m.get('decimalPrecision'),
                        spread_thr=m.get('spreadThreshold'), share_thr=m.get('shareThreshold'), reward_rate=None,
                        feed=cvd.get('priceFeedId'), provider=cvd.get('priceFeedProvider'), symbol=cvd.get('priceFeedSymbol'),
                        start_price=None, start_seen_ns=None, end_price=None, end_seen_ns=None,
                        outcome=None, outcome_seen_ns=None)
                mv = m.get('variantData') or cvd
                for fld, key in (('start_price', 'startPrice'), ('end_price', 'endPrice')):
                    v = mv.get(key)
                    seen = fld.replace('_price', '_seen_ns')
                    if v is not None and (R[seen] is None or recv < R[seen]):
                        if R[fld] is not None and R[fld] != v:
                            status['price_changed_after_first_seen'] += 1
                        R[fld] = v; R[seen] = recv
                res = m.get('resolution')
                if isinstance(res, dict) and res.get('status') == 'WON' and (R['outcome_seen_ns'] is None or recv < R['outcome_seen_ns']):
                    R['outcome'] = res.get('name'); R['outcome_seen_ns'] = recv
                for rw in ((m.get('rewards') or {}).get('schedule') or []):
                    try:
                        if io.iso_s(rw['startsAt']) == ws:
                            R['reward_rate'] = rw.get('hourlyRate')
                    except (KeyError, TypeError, ValueError):
                        pass
            continue
        for x in pl.get('data') or []:
            mid = (x.get('market') or {}).get('id')
            try:
                ts = int(io.iso_s(x['executedAt']))
            except (KeyError, TypeError, ValueError):
                status['match_bad_time'] += 1
                continue
            tx = x['transactionHash'][2:18]
            t = x['taker']
            for m in x['makers']:
                k = tx + m['hash'][2:18]
                old = legs.get(k)
                if old is not None:
                    if recv < old[2]:
                        legs[k] = old[:2] + (recv,) + old[3:]
                    continue
                side, py = io.yes_side(m)
                sg = signers.setdefault(m['signer'], len(signers))
                legs[k] = (mid, ts, recv, 0 if side == 'bid' else 1, tick(py), int(m['amount']) / io.E18,
                           int(x['priceExecuted']) / io.E18, sg, t['outcome']['indexSet'], 0 if t['quoteType'] == 'Bid' else 1)
    io.mem_check(f'AB {h}')
    print(f'[AB] {h} registry {len(reg)} legs {len(legs)} [mem] {io.rss_mib()} MiB', flush=True)
mids = set(reg)
L = pd.DataFrame(list(legs.values()), columns=['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed',
                                                'signer', 'taker_index', 'taker_quote'])
del legs
L = L[L.mid.isin(mids)].reset_index(drop=True)
ws_of = {m: R['ws'] for m, R in reg.items()}
L['t_rel'] = L['exec_s'] - L['mid'].map(ws_of)
L['phase'] = np.select([L.t_rel < 0, L.t_rel < 300], ['pre', 'live'], 'post')

# ---------------- events: collector statuses ----------------
ev = collections.Counter()
for h in io.hours_of(day):
    for line in io._lines(f'{io.PF_RUN}/events/{h}.jsonl.gz'):
        r = io.json.loads(line)
        ev[r.get('event') or r.get('kind')] += 1

# ---------------- Pass C: WS frames ----------------
win = {m: (R['ws'] * 1000 - 300_000, R['we'] * 1000 + 5_000) for m, R in reg.items()}
import os
os.makedirs(f'{OUT}/books', exist_ok=True)
prev_pend = {}
inc_rows = []
COLS = ['mid', 't_ms', 'recv_ns', 'order_count'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps'] + ['pend_bid', 'pend_ask']
nframes = 0
for h in io.hours_of(day, extra_after=1, extra_before=1):
    rows = []
    for recv, d in io.ws_books(h):
        mid = d.get('marketId'); t = d['updateTimestampMs']
        w = win.get(mid)
        if w is None or not (w[0] <= t <= w[1]):
            continue
        bids = sorted(d.get('bids') or [], key=lambda z: -z[0])[:5]
        asks = sorted(d.get('asks') or [], key=lambda z: z[0])[:5]
        r = [mid, t, recv, d.get('orderCount') or 0]
        for lv in (bids, asks):
            for i in range(5):
                if i < len(lv):
                    r += [tick(lv[i][0]), float(lv[i][1])]
                else:
                    r += [-1, 0.0]
        sp = d.get('settlementsPending') or {}
        cur = {}
        for si, key in ((0, 'bids'), (1, 'asks')):
            for p, v in sp.get(key) or []:
                cur[(si, tick(p))] = cur.get((si, tick(p)), 0.0) + float(v)
        old = prev_pend.get(mid, {})
        for k, v in cur.items():
            dlt = v - old.get(k, 0.0)
            if dlt > 1e-9:
                inc_rows.append((mid, t, recv, k[0], k[1], dlt))
        prev_pend[mid] = cur
        r += [sum(v for k, v in cur.items() if k[0] == 0), sum(v for k, v in cur.items() if k[0] == 1)]
        rows.append(r)
    if rows:
        df = pd.DataFrame(rows, columns=COLS)
        df.to_parquet(f'{OUT}/books/part_{h}.parquet', index=False)
        nframes += len(df)
    del rows
    io.mem_check(f'C {h}')
    print(f'[C] {h} frames {nframes} inc {len(inc_rows)} [mem] {io.rss_mib()} MiB', flush=True)
P = pd.DataFrame(inc_rows, columns=['mid', 't_ms', 'recv_ns', 'side', 'ptick', 'inc'])
del inc_rows

# ---------------- Pass D: timing reconciliation (live window only) ----------------
live = L[L.phase == 'live']
cells = live.groupby(['mid', 'exec_s', 'side', 'ptick'], as_index=False)['shares'].sum()
P['t_rel_ms'] = P['t_ms'] - P['mid'].map(ws_of) * 1000
Pl = P[(P.t_rel_ms >= -2000) & (P.t_rel_ms < 302_000)]
idx = collections.defaultdict(list)
for mid, t, side, pt, inc in zip(Pl.mid.values, Pl.t_ms.values, Pl.side.values, Pl.ptick.values, Pl.inc.values):
    idx[(mid, side, pt)].append((t, inc))
for v in idx.values():
    v.sort()
keys_of = {k: [a for a, _ in v] for k, v in idx.items()}

def offsets(cells_df, flip=False, shift_s=0):
    """Histogram (100 ms bins, -2 s..+3 s) of WS increment times relative to the REST second, weighted by increment."""
    H = np.zeros(50); matched = 0.0; total = 0.0; any_inc = 0
    for mid, s, side, pt, sh in zip(cells_df.mid.values, cells_df.exec_s.values, cells_df.side.values, cells_df.ptick.values, cells_df.shares.values):
        k = (mid, 1 - side if flip else side, pt)
        total += sh
        if k not in idx:
            continue
        base = (s + shift_s) * 1000
        ks = keys_of[k]; v = idx[k]
        i0 = bisect.bisect_left(ks, base - 2000); i1 = bisect.bisect_left(ks, base + 3000)
        got = 0.0
        for t, inc in v[i0:i1]:
            H[int((t - base + 2000) // 100)] += inc
            got += inc
        if got > 0:
            any_inc += 1
        matched += min(got, sh)
    return H, matched / max(total, 1e-12), any_inc / max(len(cells_df), 1)

H, cov, anyf = offsets(cells)
Hf, covf, anyff = offsets(cells, flip=True)
Hp, covp, anyp = offsets(cells, shift_s=17)          # placebo: same cells 17 s later
recon = dict(
    n_live_cells=int(len(cells)), live_rest_shares=float(cells.shares.sum()),
    live_ws_inc_shares=float(Pl[(Pl.t_rel_ms >= 0) & (Pl.t_rel_ms < 300_000)].inc.sum()),
    offset_hist_bins_ms=list(range(-2000, 3000, 100)),
    offset_hist_same_side=[round(x, 3) for x in H], offset_hist_flipped=[round(x, 3) for x in Hf], offset_hist_placebo17s=[round(x, 3) for x in Hp],
    share_cells_with_ws_inc=dict(same=anyf, flipped=anyff, placebo=anyp),
    volume_coverage_min_ratio=dict(same=cov, flipped=covf, placebo=covp),
)

# ---------------- Pass E: per-window structure ----------------
books = pd.concat([pd.read_parquet(f'{OUT}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'b1_s', 'a1_p', 'a1_s', 'b2_p', 'a2_p']) for f in sorted(os.listdir(f'{OUT}/books'))], ignore_index=True)
books.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
out_rows = []
for mid, g in books.groupby('mid', sort=False):
    R = reg[mid]; ws_ms = R['ws'] * 1000
    t = g.t_ms.values; tr = (t - ws_ms) / 1000.0
    m = (tr >= 0) & (tr < 300)
    if m.sum() == 0:
        out_rows.append(dict(mid=mid, ws=R['ws'], n_live_frames=0)); continue
    # time weights: duration until the next frame, clipped to the live window
    nxt = np.append(tr[1:], 300.0)
    dur = np.clip(np.minimum(nxt, 300.0) - np.maximum(tr, 0.0), 0, None) * m
    b1 = g.b1_p.values; a1 = g.a1_p.values
    two = (b1 >= 0) & (a1 >= 0)
    locked = two & (b1 == a1); crossed = two & (b1 > a1); ok = two & (b1 < a1)
    spread = np.where(ok, a1 - b1, 0)
    W = dur.sum()
    def tw(mask):
        return float((dur * mask).sum() / W) if W > 0 else float('nan')
    b2 = g.b2_p.values; a2 = g.a2_p.values
    gapb = ok & ((b2 < 0) | (b1 - b2 >= 2)); gapa = ok & ((a2 < 0) | (a2 - a1 >= 2))
    mid_px = np.where(ok, (a1 + b1) / 200.0, np.nan)
    gaps = np.diff(tr[m]) if m.sum() > 1 else np.array([0.0])
    lw = live[live.mid == mid]
    row = dict(mid=mid, ws=R['ws'], n_live_frames=int(m.sum()), frames_per_s=float(m.sum() / 300.0),
               max_frame_gap_s=float(max(gaps.max() if len(gaps) else 0, tr[m][0] if m.any() else 300, 300 - tr[m][-1] if m.any() else 300)),
               tw_two_sided_ok=tw(ok), tw_locked=tw(locked), tw_crossed=tw(crossed), tw_one_or_empty=tw(~two),
               tw_spread1=tw(ok & (spread == 1)), tw_spread2=tw(ok & (spread == 2)), tw_spread3p=tw(ok & (spread >= 3)),
               tw_gap_behind_bid=tw(gapb), tw_gap_behind_ask=tw(gapa),
               med_touch_bid_sh=float(np.median(g.b1_s.values[m & ok])) if (m & ok).any() else float('nan'),
               med_touch_ask_sh=float(np.median(g.a1_s.values[m & ok])) if (m & ok).any() else float('nan'),
               mid_at_open=float(mid_px[m][~np.isnan(mid_px[m])][0]) if (~np.isnan(mid_px[m])).any() else float('nan'),
               mid_at_240=float(pd.Series(mid_px[m & (tr <= 240)]).dropna().iloc[-1]) if (~np.isnan(mid_px[m & (tr <= 240)])).any() else float('nan'),
               live_legs=int(len(lw)), live_shares=float(lw.shares.sum()), live_yes_notional=float((lw.shares * lw.ptick / 100).sum()),
               live_makers=int(lw.signer.nunique()),
               top_maker_share=float(lw.groupby('signer').shares.sum().max() / lw.shares.sum()) if len(lw) else float('nan'),
               aggr_buy_share=float(lw.shares[lw.side == 1].sum() / lw.shares.sum()) if len(lw) else float('nan'),
               pre_shares=float(L[(L.mid == mid) & (L.phase == 'pre')].shares.sum()),
               start_price=R['start_price'], end_price=R['end_price'], outcome=R['outcome'],
               start_known_after_ws_s=(R['start_seen_ns'] / 1e9 - R['ws']) if R['start_seen_ns'] else None,
               outcome_known_after_we_s=(R['outcome_seen_ns'] / 1e9 - R['we']) if R['outcome_seen_ns'] else None,
               reward_rate=R['reward_rate'])
    for b0 in range(0, 300, 60):
        mb = m & (tr >= b0) & (tr < b0 + 60)
        row[f'tw_spread1_{b0}'] = float((dur * mb * (ok & (spread == 1))).sum() / max((dur * mb).sum(), 1e-9))
        row[f'shares_{b0}'] = float(lw.shares[(lw.t_rel >= b0) & (lw.t_rel < b0 + 60)].sum())
    out_rows.append(row)
    io.mem_check('E')
W = pd.DataFrame(out_rows)

# ---------------- write ----------------
RG = pd.DataFrame(list(reg.values()))
RG['up_index'] = RG['up_index'].map(lambda v: v[0] if len(v) == 1 else -1)
RG.to_parquet(f'{OUT}/registry.parquet', index=False)
L.to_parquet(f'{OUT}/legs.parquet', index=False)
P.to_parquet(f'{OUT}/pend_inc.parquet', index=False)
W.to_parquet(f'{OUT}/windows.parquet', index=False)
expected = 288
summary = dict(
    day=day, mode='EXPLORATORY (user 2026-11-01: no held-out rules yet)',
    n_windows_expected=expected, n_windows_registry=len(reg),
    n_windows_with_live_frames=int((W.n_live_frames > 0).sum()) if len(W) else 0,
    n_windows_with_outcome=int(RG.outcome.notna().sum()), n_windows_with_start_price=int(RG.start_price.notna().sum()),
    up_is_index1=int((RG.up_index == 1).sum()), fee_bps=RG.fee_bps.value_counts().to_dict(), maker_fee_bps=RG.maker_fee_bps.value_counts().to_dict(),
    decimal_precision=RG.decimal_precision.value_counts().to_dict(), reward_rate=RG.reward_rate.value_counts(dropna=False).to_dict(),
    n_frames=int(nframes), n_legs=int(len(L)), legs_by_phase=L.phase.value_counts().to_dict(),
    n_distinct_makers_live=int(live.signer.nunique()), n_distinct_signers_all=len(signers),
    first_seen_lag_s_live=dict(zip(['p05', 'p50', 'p95', 'p99'], np.percentile(live.first_recv_ns / 1e9 - live.exec_s, [5, 50, 95, 99]).round(2).tolist())) if len(live) else None,
    recon=recon, collector_events=dict(ev), statuses=dict(status), mem_peak_mib_selfreport=io.rss_mib(),
)
io.write_json(f'{OUT}/summary.json', summary)
print('[done]', {k: summary[k] for k in ('n_windows_registry', 'n_windows_with_live_frames', 'n_windows_with_outcome', 'n_frames', 'n_legs')})
print('[recon]', {k: recon[k] for k in ('n_live_cells', 'live_rest_shares', 'live_ws_inc_shares', 'share_cells_with_ws_inc', 'volume_coverage_min_ratio')})
