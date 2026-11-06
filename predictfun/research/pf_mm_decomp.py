"""MM vs settlement P&L, +1 s against the top makers, and why longer windows fail - FIXED live engine, round i (invwin runs).
Declared in STATUS.yml 2026-11-06 (user: "the 1 s gain compared with top makers and settlement pnl given different configs; check the mm
performance and settlement pnl separately, and why longer windows don't work"). Descriptive, in-sample; reads only existing round-i fills
(run_invwin.sh) and the step-1 legs / books of the same days; no replay.
Per fill (side s = +1 buy / -1 sell, shares q, price p; m0 = venue mid of the last valid frame strictly before the fill, m1 = mid at +1 s,
mS = mid at the arm's stop + 100 ms, y = outcome 0 / 100; mid lookups = the engine's mid_at on the same frames and input bounds), USD:
  spread   s q (m0 - p) / 100     earned against the mid at the fill
  informed s q (m1 - m0) / 100    the mid moving against the fill within 1 s (adverse selection)
  drift    s q (mS - m1) / 100    inventory P&L from 1 s after the fill to the stop
  carry    s q (y - mS) / 100     the leftover's settlement against the mid at the stop
MM P&L = spread + informed + drift (= mark@stop); settlement P&L = carry; total = settle. Reconciliation against the engine fails the
run (per arm-day: settle = engine gross, spread + informed = engine mo1, spread + informed + drift = engine mtm_stop, |diff| <= $0.01);
falsifier: a corrupted copy (one fill's shares doubled) must fail it. Makers: real legs, timed like the engine (tx ws anchor, taker-order
key; the retimed-leg count must equal the engine's), bands top 5 / 6-20 / rest by live-window volume per day, and all; same terms, the
part after 1 s = s q (y - m1) / 100. Positive control: legs_timed2-timed maker +1 s c/sh = report_maker_bad_fills.json (|diff| <= 0.01).
Usage: python3 -m predictfun.research.pf_mm_decomp   (PF_MMD_SMOKE=1: one day, reconciliation + falsifier only, no report)
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io

S4 = f'{io.OUT}/step4'; RUN = f'{S4}/live_tokyoreal_custom_e2_preds_t60k_'
D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; BES = '20261103'; D9 = D8 + [BES]
SIZES = {'s5': 'AV:5:25:0.06', 's20': 'A1:20:100:0'}
PRE = 'MICRO+G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+'
NB = 4; BLAB = ['0-60', '60-120', '120-180', '180-240']
T7 = 2.365; TOL = 0.01
BANDS = ['top 5', '6-20', 'rest', 'all makers']
COMP = ['spread', 'informed', 'plus1', 'drift', 'mark', 'carry', 'settle']
SMOKE = os.environ.get('PF_MMD_SMOKE') == '1'


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def stop_of(name):
    for r in name.split('/')[0].split('+'):
        if r.startswith('S') and r[1:].isdigit():
            return float(r[1:])
    return 240.0


def extra(name):
    assert name.startswith(PRE) and '/mid@' in name, name
    return name[len(PRE):].split('/mid@')[0]


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T7 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(m, 2), lo=round(m - h, 2), hi=round(m + h, 2), excludes_0=bool(m - h > 0 or m + h < 0), days_pos=int((x > 0).sum()))


def mid_maps(day, lo, hi, engine_order=True):
    """{mid: (t_ms of valid frames, mid)} - valid = b1 >= 0 and a1 > b1. engine_order: the engine's sort (mid, t_ms, recv_ns) and its
    input bounds, so searchsorted(..., 'right') - 1 on valid frames = the engine's mid_at (which steps back over invalid frames)."""
    S1 = f'{io.OUT}/step1/{day}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))],
                   ignore_index=True)
    if engine_order:
        bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable'); bk = bk[(bk.t_ms.values >= lo) & (bk.t_ms.values < hi)]
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)]
    if not engine_order:
        bk = bk.sort_values(['mid', 't_ms'], kind='stable')
    return {m: (g.t_ms.values.astype(np.float64), (g.b1_p.values + g.a1_p.values) / 2.0) for m, g in bk.groupby('mid', sort=False)}


def lookup(BY, mids, t, side):
    """mid of the last valid frame at or before t (side='right') or strictly before t (side='left'); NaN if none."""
    out = np.full(len(t), np.nan)
    for m, ix in pd.Series(np.arange(len(t))).groupby(mids, sort=False).indices.items():
        v = BY.get(m)
        if v is None:
            continue
        tv, mv = v; j = np.searchsorted(tv, t[ix], side=side) - 1; ok = j >= 0
        out[ix[ok]] = mv[j[ok]]
    return out


def decompose(f, BY, reg, stop):
    """per-fill USD terms (NaN where a mid is missing)."""
    ws = reg.loc[f.mid.values, 'ws'].values * 1000.0; ta = np.round(ws + f.t.values * 1000.0, 3)
    y = np.where(reg.loc[f.mid.values, 'outcome'].values == 'Up', 100.0, 0.0)
    s = np.where(f.side.values == 0, 1.0, -1.0); q = f.q.values.astype(float); p = f.pt.values.astype(float); mids = f.mid.values
    m0 = lookup(BY, mids, ta, 'left'); m1 = lookup(BY, mids, ta + 1000.0, 'right'); mS = lookup(BY, mids, ws + stop * 1000.0 + 100.0, 'right')
    D = pd.DataFrame(dict(mid=mids, b=np.clip((f.t.values // 60).astype(int), 0, NB - 1), q=q,
                          settle=s * q * (y - p) / 100.0, plus1=s * q * (m1 - p) / 100.0, spread=s * q * (m0 - p) / 100.0,
                          mark=s * q * (mS - p) / 100.0, post1=s * q * (y - m1) / 100.0, dist=np.abs(m0 - 50.0) * q))
    D['informed'] = D.plus1 - D.spread; D['drift'] = D.mark - D.plus1; D['carry'] = D.settle - D.mark
    return D


def recon(D, f, W):
    """failures of the reconciliation against the engine's own numbers (empty = pass)."""
    bad = []
    for lab, a, b in (('settle = engine fill settle', D.settle.sum(), f.settle.sum()), ('settle = engine gross', D.settle.sum(), W.pnl.sum()),
                      ('spread + informed = engine mo1', np.nansum(D.plus1), np.nansum(f.mo1.values)),
                      ('mark = engine mtm_stop', np.nansum(D.mark), np.nansum(W.mtm_stop.values))):
        if not abs(a - b) <= TOL:
            bad.append(f'{lab}: {a:.4f} vs {b:.4f}')
    if D.plus1.isna().sum() != f.mo1.isna().sum():
        bad.append(f'missing +1 s mid: {int(D.plus1.isna().sum())} vs engine {int(f.mo1.isna().sum())}')
    return bad


def maker_legs(day, tx_retimed):
    """live-window maker legs with engine timing (tx ws anchor + taker-order key) and legs_timed2 timing (control)."""
    S1 = f'{io.OUT}/step1/{day}'
    L = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet', columns=['tk']).tk.values
    assert len(tk) == len(L), 'legs_tk not aligned with legs_timed2'
    ws_t = np.where(L.how.values > 0, L.t_ms.values.astype(float), np.nan)
    e2_t = np.where((L.how.values == 0) & np.isfinite(L.t_e2.values), L.t_e2.values, np.nan)
    g = np.where(tk >= 0, tk, -1 - np.arange(len(L)))
    a_ = pd.Series(ws_t).groupby(g, sort=False).transform('min').values; b_ = pd.Series(e2_t).groupby(g, sort=False).transform('min').values
    anc = np.where(np.isfinite(a_), a_, b_); old = np.where(L.how.values > 0, ws_t, e2_t)
    n_re = int((np.isfinite(anc) & (np.abs(anc - old) > 1e-6)).sum())
    if n_re != tx_retimed:
        sys.exit(f'REFUSED: {day} maker timing retimed {n_re} legs, engine {tx_retimed}')
    L['t_eng'] = np.where(np.isfinite(anc), anc, L.t_ms.values); L['t_old'] = old
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid'])
    L = L[L.mid.isin(reg.mid) & (L.t_rel >= 0) & (L.t_rel < 240)].copy()
    vol = L.groupby('signer').shares.sum().sort_values(ascending=False); rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = L.signer.map(rk).values; L['band'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest'))
    return L


def maker_terms(L, t, BY, reg):
    ok = np.isfinite(t); L = L[ok]; t = t[ok]
    y = np.where(reg.loc[L.mid.values, 'outcome'].values == 'Up', 100.0, 0.0); ws = reg.loc[L.mid.values, 'ws'].values * 1000.0
    s = np.where(L.side.values == 0, 1.0, -1.0); q = L.shares.values.astype(float); p = L.ptick.values.astype(float); mids = L.mid.values
    m0 = lookup(BY, mids, t, 'left'); m1 = lookup(BY, mids, t + 1000.0, 'right')
    D = pd.DataFrame(dict(band=L.band.values, b=np.clip(((t - ws) // 60000).astype(int), 0, NB - 1), w60=L.t_rel.values < 60, q=q,
                          settle=s * q * (y - p) / 100.0, plus1=s * q * (m1 - p) / 100.0, spread=s * q * (m0 - p) / 100.0,
                          post1=s * q * (y - m1) / 100.0, dist=np.abs(m0 - 50.0) * q, c1=s * (m1 - p)))
    D['informed'] = D.plus1 - D.spread
    return D, int((~ok).sum())


def agg(D, keys):
    """sums with the shares that carry each term (a term is NaN where its mid is missing)."""
    out = {}
    for k, g in D.groupby(keys, sort=False):
        r = dict(q=float(g.q.sum()))
        for c in ('spread', 'informed', 'plus1', 'drift', 'mark', 'carry', 'settle', 'post1', 'dist'):
            if c in g:
                okc = g[c].notna(); r[c] = float(g[c][okc].sum()); r[f'q_{c}'] = float(g.q[okc].sum())
        out[k if isinstance(k, tuple) else (k,)] = r
    return out


def total(D):
    return agg(D.assign(z=0), ['z']).get((0,), dict(q=0.0))


def main():
    ours = {}        # (k, sz, extra, day) -> {'all': sums, 'b': {bucket: sums}}
    makers = {}      # (timing, day) -> {(band, bucket): sums}
    ctrl = {}        # (part, band) -> [usd, q] over the 8 read days (legs_timed2 timing, maker_bad_fills' book order)
    st = dict(fills=0, no_m0=0, no_m1=0, no_mS=0, maker_legs=0, maker_untimed=0, maker_no_m0=0, maker_no_m1=0)
    first60 = {}     # (k, sz, day) -> equal / not: S240 fills before 60 s vs S60 fills
    falsifier = None
    for d in (['20261024'] if SMOKE else D9):
        S1 = f'{io.OUT}/step1/{d}'
        reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'outcome']).set_index('mid')
        s0 = json.load(open(f'{RUN}iwJ_s5a/{d}/summary.json'))
        ib = s0['input_bounds']; BY = mid_maps(d, ib['t_lo_ms'], ib['t_hi_ms'])
        for k in ('J', 'X'):
            for sz in SIZES:
                fr = {}
                for b in 'ab':
                    base = f'{RUN}iw{k}_{sz}{b}/{d}'; sm = json.load(open(f'{base}/summary.json'))
                    assert sm['input_bounds'] == ib, (base, sm['input_bounds'], ib)
                    for name in sm['arms']:
                        f = pd.read_parquet(f'{base}/fills/{stem(name)}.parquet'); W = pd.read_parquet(f'{base}/win/{stem(name)}.parquet')
                        D = decompose(f, BY, reg, stop_of(name))
                        bad = recon(D, f, W)
                        if bad:
                            sys.exit(f'RECONCILIATION FAILED {k}/{sz}/{d}/{extra(name)}: {bad}')
                        if falsifier is None and len(f) > 1:
                            fb = f.copy(); fb.loc[fb.index[0], 'q'] *= 2.0
                            falsifier = recon(decompose(fb, BY, reg, stop_of(name)), f, W)
                            if not falsifier:
                                sys.exit('FALSIFIER DID NOT FIRE: the reconciliation passed a corrupted copy')
                        st['fills'] += len(D); st['no_m0'] += int(D.spread.isna().sum()); st['no_m1'] += int(D.plus1.isna().sum()); st['no_mS'] += int(D.mark.isna().sum())
                        ours[(k, sz, extra(name), d)] = dict(all=total(D), b={kk[0]: v for kk, v in agg(D, ['b']).items()})
                        fr[extra(name)] = f
                f60 = lambda x: x[x.t < 60].sort_values(['mid', 't', 'side', 'pt', 'q'], kind='stable').reset_index(drop=True)
                first60[(k, sz, d)] = bool(f60(fr['S240']).equals(f60(fr['S60'])))
        # makers
        L = maker_legs(d, s0['tx_timing']['legs_retimed']); st['maker_legs'] += len(L)
        Dm, unt = maker_terms(L, L.t_eng.values.astype(float), BY, reg); st['maker_untimed'] += unt
        st['maker_no_m0'] += int(Dm.spread.isna().sum()); st['maker_no_m1'] += int(Dm.plus1.isna().sum())
        mk = agg(Dm, ['band', 'b']); mk.update({('all makers', kk[0]): v for kk, v in agg(Dm, ['b']).items()})
        mk.update({(bd, 'all'): v for (bd,), v in agg(Dm, ['band']).items()}); mk[('all makers', 'all')] = total(Dm)
        makers[d] = mk
        if d in D8:          # positive control: report_maker_bad_fills' timing, book order and +1 s definition
            BYc = mid_maps(d, -np.inf, np.inf, engine_order=False)
            Dc, _ = maker_terms(L, L.t_old.values.astype(float), BYc, reg); Dc = Dc[np.isfinite(Dc.c1.values)]
            for part, m in (('first 60 s', Dc.w60.values), ('0-240 s', np.ones(len(Dc), bool))):
                for bd, g in Dc[m].groupby('band'):
                    a = ctrl.setdefault(f'{part}|{bd}', [0.0, 0.0]); a[0] += float((g.q * g.c1).sum() / 100.0); a[1] += float(g.q.sum())
            del BYc, Dc
        del BY, L, Dm
        io.mem_check(f'day {d}')

    if SMOKE:
        print(f'SMOKE: statuses {st}; reconciliation PASS; falsifier fired: {falsifier}; first60 {first60}'); return 0
    # ---- positive control ----
    ref = json.load(open(f'{S4}/report_maker_bad_fills.json'))
    cdiff = {kk: round(100 * v[0] / v[1] - ref[kk]['net_c'], 4) for kk, v in ctrl.items()}
    if not all(abs(x) <= 0.01 for x in cdiff.values()):
        sys.exit(f'POSITIVE CONTROL FAILED (maker +1 s vs report_maker_bad_fills): {cdiff}')

    # ---- tables ----
    def per_day(k, sz, e, field, days=D8, b=None):
        out = []
        for d in days:
            r = ours[(k, sz, e, d)]; r = r['all'] if b is None else r['b'].get(b, dict(q=0.0))
            out.append(r.get(field, 0.0))
        return np.array(out)

    def cps(r, c):      # c / share on the shares that carry the term
        return 100.0 * r[c] / r[f'q_{c}'] if r.get(f'q_{c}', 0) > 0 else np.nan

    def pool(rs):
        o = {}
        for r in rs:
            for kk, v in r.items():
                o[kk] = o.get(kk, 0.0) + v
        return o

    rep = dict(note='descriptive, in-sample; fixed engine round i (invwin runs); USD / day over the 8 read days; 11-03 beside', statuses=st,
               reconciliation='PASS (every arm-day)', falsifier_fired=falsifier, positive_control_diff_c=cdiff, first60_equal=sum(first60.values()),
               first60_checked=len(first60), configs={}, buckets={}, makers={}, window={}, vs_top5={})
    exs = sorted({kk[2] for kk in ours}, key=lambda e: (stop_of('x+' + e.replace('+', '+')), e))
    print(f"statuses {st}\nreconciliation PASS on every arm-day; falsifier fired: {falsifier}\npositive control (maker +1 s vs "
          f"report_maker_bad_fills, c/sh): {cdiff}\nS240 fills before 60 s == S60 fills: {sum(first60.values())} of {len(first60)} (timing, size, day)")
    print('\nA. MM vs settlement, $ / day (8 read days; 11-03 beside): spread + informed = +1 s; + drift = MM (mark@stop); carry = settlement; MM + carry = settle')
    for sz in SIZES:
        for k in ('J', 'X'):
            print(f"\n{k} {sz:4s} {'config':22s} {'spread':>8s} {'informed':>9s} {'+1 s':>8s} {'drift':>8s} {'MM':>8s} | {'carry':>8s} | {'settle':>8s} | "
                  f"{'+1s c/sh':>8s} {'MM d>0':>6s} {'carry sd':>8s} | 11-03 MM / carry")
            for e in exs:
                if (k, sz, e, D8[0]) not in ours:
                    continue
                v = {c: per_day(k, sz, e, c).mean() for c in COMP}; sh = per_day(k, sz, e, 'q').sum()
                p1 = 100.0 * per_day(k, sz, e, 'plus1').sum() / per_day(k, sz, e, 'q_plus1').sum()
                mm = per_day(k, sz, e, 'mark'); cr = per_day(k, sz, e, 'carry'); b_ = ours[(k, sz, e, BES)]['all']
                rep['configs'][f'{k}/{sz}/{e}'] = dict({c: round(float(v[c]), 1) for c in COMP}, shares_day=round(sh / 8.0), plus1_c_sh=round(float(p1), 3),
                                                       mm_days_pos=int((mm > 0).sum()), carry_day_sd=round(float(cr.std(ddof=1)), 1),
                                                       mm_1103=round(b_['mark'], 1), carry_1103=round(b_['carry'], 1),
                                                       mm_ci=ci(mm), carry_ci=ci(cr), settle_ci=ci(per_day(k, sz, e, 'settle')))
                print(f"       {e:22s} {v['spread']:+8.1f} {v['informed']:+9.1f} {v['plus1']:+8.1f} {v['drift']:+8.1f} {v['mark']:+8.1f} | {v['carry']:+8.1f} | "
                      f"{v['settle']:+8.1f} | {p1:+8.3f} {int((mm > 0).sum()):4d}/8 {cr.std(ddof=1):8.1f} | {b_['mark']:+7.1f} / {b_['carry']:+7.1f}")

    print('\nB. by fill time in the window, 8 read days pooled: shares / day; c / share: spread, informed, +1 s, after 1 s to the outcome, settle; mean |mid - 50| at the fill')
    hdr = f"{'':26s} {'bucket':8s} {'sh/day':>8s} {'spread':>7s} {'inform':>7s} {'+1 s':>7s} {'post1s':>7s} {'settle':>7s} {'|m-50|':>6s}"
    def brow(lab, bl, r):
        print(f"{lab:26s} {bl:8s} {r['q'] / 8.0:8.0f} {cps(r, 'spread'):+7.2f} {cps(r, 'informed'):+7.2f} {cps(r, 'plus1'):+7.2f} {cps(r, 'post1'):+7.2f} "
              f"{cps(r, 'settle'):+7.2f} {r['dist'] / r['q_dist'] if r.get('q_dist') else np.nan:6.1f}")
    print(hdr)
    for bd in BANDS:
        for bi, bl in enumerate(BLAB):
            r = pool([makers[d][(bd, bi)] for d in D8 if (bd, bi) in makers[d]])
            rep['makers'][f'{bd}/{bl}'] = {c: (round(cps(r, c), 3) if c != 'q' else round(r['q'] / 8.0)) for c in ('q', 'spread', 'informed', 'plus1', 'post1', 'settle')}
            brow(f'maker {bd}', bl, r)
    for sz in SIZES:
        for k in ('J', 'X'):
            for e in exs:
                if (k, sz, e, D8[0]) not in ours:
                    continue
                for bi, bl in enumerate(BLAB):
                    rs = [ours[(k, sz, e, d)]['b'][bi] for d in D8 if bi in ours[(k, sz, e, d)]['b']]
                    if not rs:
                        continue
                    r = pool(rs)
                    rep['buckets'][f'{k}/{sz}/{e}/{bl}'] = {c: (round(cps(r, c), 3) if c != 'q' else round(r['q'] / 8.0)) for c in ('q', 'spread', 'informed', 'plus1', 'post1', 'settle')}
                    if e in ('S60', 'S240', f"S240+CAP{SIZES[sz].split(':')[2]}"):
                        brow(f'ours {k} {sz} {e}', bl, r)

    print('\nC. ours (S240, quotes all 240 s) minus top 5 makers, +1 s c / share per bucket: day-clustered mean (t interval, 7 df), days ours > top 5')
    for sz in SIZES:
        for k in ('J', 'X'):
            for bi, bl in enumerate(BLAB):
                dd = []
                for d in D8:
                    r = ours[(k, sz, 'S240', d)]['b'].get(bi); m = makers[d].get(('top 5', bi))
                    if r and m and r.get('q_plus1', 0) > 0 and m.get('q_plus1', 0) > 0:
                        dd.append(cps(r, 'plus1') - cps(m, 'plus1'))
                if len(dd) == len(D8):
                    c = ci(dd); rep['vs_top5'][f'{k}/{sz}/{bl}'] = c
                    print(f"  {k} {sz:4s} {bl:8s} {c['mean']:+6.2f} ({c['lo']:+.2f}..{c['hi']:+.2f}) {c['days_pos']}/8")
                else:
                    rep['vs_top5'][f'{k}/{sz}/{bl}'] = f'INCOMPLETE ({len(dd)} days)'

    print('\nD. why longer windows: config minus S60, $ / day, day-clustered mean (t interval, 7 df) over the 8 read days; 11-03 beside')
    for sz in SIZES:
        hi = SIZES[sz].split(':')[2]
        for k in ('J', 'X'):
            for a, b in (('S120', 'S60'), ('S180', 'S60'), ('S240', 'S60'), (f'S240+CAP{hi}', f'S60+CAP{hi}')):
                row = {}
                for c in COMP:
                    x = per_day(k, sz, a, c) - per_day(k, sz, b, c); row[c] = ci(x)
                    row[c]['d1103'] = round(ours[(k, sz, a, BES)]['all'][c] - ours[(k, sz, b, BES)]['all'][c], 1)
                rep['window'][f'{k}/{sz}/{a} - {b}'] = row
                print(f"  {k} {sz:4s} {a + ' - ' + b:22s} " + '  '.join(f"{c} {row[c]['mean']:+.1f} ({row[c]['lo']:+.1f}..{row[c]['hi']:+.1f})" for c in
                                                             ('spread', 'informed', 'drift', 'mark', 'carry', 'settle')))
    io.write_json(f'{S4}/report_mm_decomp.json', rep)
    return 0


if __name__ == '__main__':
    sys.exit(main())
