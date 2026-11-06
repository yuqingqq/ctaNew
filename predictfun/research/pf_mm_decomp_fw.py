"""MM / settlement decomposition and +1 s against the top makers for the whole-window feed refresh (section 58) - FIXED live engine, round i.
Declared in STATUS.yml 2026-11-06 05:04Z (user: "measure the +1s gain and compare with top makers, together with settlement pnl").
Descriptive, in-sample; reads only existing fills, no replay: refreshed F = fwJ / fwX runs (preds_t60kF), grid G = invwin runs (preds_t60k),
arms MICRO + D1 stack + V0 + GS + XNA + XMIN1 + XV + S60 / S120 / S180 / S240 at AV:5:25:0.06 (s5) and A1:20:100:0 (s20), J and X timing.
Per-fill terms as pf_mm_decomp (section 57): spread s(m0 - p), informed s(m1 - m0), drift s(mS - m1), carry s(y - mS); +1 s = spread +
informed, MM = +1 s + drift (mark@stop), settlement = carry, MM + carry = settle; post-1 s = s(y - m1) (drift + carry).
Checks (each fails the run): reconciliation against the engine on every arm-day of both sources, with a falsifier that must fire; the grid
configs reproduce report_mm_decomp.json (every term, |diff| <= 0.051 $ / day on 1-decimal values) and a perturbed copy must not; maker band x
minute +1 s c/sh reproduces report_mm_decomp.json (3 decimals); refreshed and grid runs share input bounds and the maker retimed-leg count.
Usage: python3 -m predictfun.research.pf_mm_decomp_fw
"""
import json, sys
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_mm_decomp import (S4, D8, BES, D9, SIZES, PRE, NB, BLAB, BANDS, COMP, stem, stop_of, ci, mid_maps, decompose, recon, maker_legs,
                           maker_terms, agg, total)

RUN = f'{S4}/live_tokyoreal_custom_e2_'
WINS = ['S60', 'S120', 'S180', 'S240']; BATCH = {'S60': 'a', 'S120': 'a', 'S180': 'b', 'S240': 'b'}
TERMS = ['spread', 'informed', 'plus1', 'drift', 'mark', 'carry', 'settle']


def base(src, k, sz, e, d):
    return f'{RUN}preds_t60kF_fw{k}/{d}' if src == 'F' else f'{RUN}preds_t60k_iw{k}_{sz}{BATCH[e]}/{d}'


def name(e, sz):
    return f'{PRE}{e}/mid@{SIZES[sz]}'


def cps(r, c):
    return 100.0 * r[c] / r[f'q_{c}'] if r and r.get(f'q_{c}', 0) > 0 else np.nan


def pool(rs):
    o = {}
    for r in rs:
        for kk, v in r.items():
            o[kk] = o.get(kk, 0.0) + v
    return o


def main():
    ours = {}; makers = {}; falsifier = None
    st = {s: dict(fills=0, no_m0=0, no_m1=0, no_mS=0) for s in 'FG'}; st.update(maker_legs=0, maker_untimed=0)
    for d in D9:
        reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws', 'outcome']).set_index('mid')
        s0 = json.load(open(f'{base("F", "J", "s5", "S60", d)}/summary.json')); ib = s0['input_bounds']
        BY = mid_maps(d, ib['t_lo_ms'], ib['t_hi_ms']); rt = set()
        for src in 'FG':
            for k in ('J', 'X'):
                for sz in SIZES:
                    for e in WINS:
                        b = base(src, k, sz, e, d); sm = json.load(open(f'{b}/summary.json'))
                        if sm['input_bounds'] != ib:
                            sys.exit(f'REFUSED: input bounds differ {b}')
                        assert sm['env'].get('PF_PREDS') == ('preds_t60kF' if src == 'F' else 'preds_t60k'), b
                        rt.add(sm['tx_timing']['legs_retimed'])
                        n = name(e, sz); f = pd.read_parquet(f'{b}/fills/{stem(n)}.parquet'); W = pd.read_parquet(f'{b}/win/{stem(n)}.parquet')
                        D = decompose(f, BY, reg, stop_of(n)); bad = recon(D, f, W)
                        if bad:
                            sys.exit(f'RECONCILIATION FAILED {src}/{k}/{sz}/{e}/{d}: {bad}')
                        if falsifier is None and len(f) > 1:
                            fb = f.copy(); fb.loc[fb.index[0], 'q'] *= 2.0
                            falsifier = recon(decompose(fb, BY, reg, stop_of(n)), f, W)
                            if not falsifier:
                                sys.exit('FALSIFIER DID NOT FIRE: the reconciliation passed a corrupted copy')
                        s_ = st[src]; s_['fills'] += len(D); s_['no_m0'] += int(D.spread.isna().sum()); s_['no_m1'] += int(D.plus1.isna().sum())
                        s_['no_mS'] += int(D.mark.isna().sum())
                        ours[(src, k, sz, e, d)] = dict(all=total(D), b={kk[0]: v for kk, v in agg(D, ['b']).items()})
        if len(rt) != 1:
            sys.exit(f'REFUSED: {d} retimed-leg counts differ across runs {rt}')
        L = maker_legs(d, rt.pop()); st['maker_legs'] += len(L)
        Dm, unt = maker_terms(L, L.t_eng.values.astype(float), BY, reg); st['maker_untimed'] += unt
        mk = agg(Dm, ['band', 'b']); mk.update({('all makers', kk[0]): v for kk, v in agg(Dm, ['b']).items()})
        makers[d] = mk
        del BY, L, Dm
        io.mem_check(f'day {d}')

    pd_ = lambda src, k, sz, e, c, days=D8: np.array([ours[(src, k, sz, e, d)]['all'].get(c, 0.0) for d in days])
    # ---- checks against section 57 ----
    ref = json.load(open(f'{S4}/report_mm_decomp.json'))
    def gdiff(shift=0.0):
        out = {}
        for k in ('J', 'X'):
            for sz in SIZES:
                for e in WINS:
                    r = ref['configs'][f'{k}/{sz}/{e}']
                    for c in TERMS:
                        v = round(float(pd_('G', k, sz, e, c).mean()) + (shift if (k, sz, e, c) == ('J', 's5', 'S60', 'mark') else 0.0), 1)
                        if abs(v - r[c]) > 0.051:
                            out[f'{k}/{sz}/{e}/{c}'] = (v, r[c])
        return out
    gd = gdiff()
    if gd:
        sys.exit(f'REFUSED: grid configs do not reproduce report_mm_decomp.json: {gd}')
    if not gdiff(1.0):
        sys.exit('FALSIFIER DID NOT FIRE: the grid reproduction check passed a perturbed copy')
    md = {}
    for bd in BANDS:
        for bi, bl in enumerate(BLAB):
            v = round(cps(pool([makers[d][(bd, bi)] for d in D8 if (bd, bi) in makers[d]]), 'plus1'), 3)
            if v != ref['makers'][f'{bd}/{bl}']['plus1']:
                md[f'{bd}/{bl}'] = (v, ref['makers'][f'{bd}/{bl}']['plus1'])
    if md:
        sys.exit(f'REFUSED: maker +1 s does not reproduce report_mm_decomp.json: {md}')

    rep = dict(note='descriptive, in-sample; fixed engine round i; F = refreshed (preds_t60kF), G = grid (preds_t60k); $ / day over the 8 read days; 11-03 beside',
               statuses=st, reconciliation='PASS (every arm-day, both sources)', falsifier_fired=falsifier, grid_reproduces_section57=True,
               makers_reproduce_section57=True, configs={}, diff={}, minute={}, vs_top5={}, post1_diff={})
    print(f"statuses {st}\nreconciliation PASS on every arm-day (F and G); falsifier fired: {falsifier}\n"
          f"grid configs = report_mm_decomp.json (all {len(WINS) * 4 * len(TERMS)} terms; perturbed copy refused); maker band x minute +1 s = report_mm_decomp.json")
    print('\nA. per config, $ / day (8 read days): spread + informed = +1 s; + drift = MM (mark@stop); carry = settlement; MM + carry = settle; 11-03 beside')
    print(f"{'':16s} {'spread':>7s} {'inform':>7s} {'+1 s':>7s} {'drift':>7s} {'MM':>7s} | {'carry':>7s} | {'settle':>7s} | {'MM d>0':>6s} {'carry sd':>8s} | 11-03 MM / carry")
    for sz in SIZES:
        for k in ('J', 'X'):
            for e in WINS:
                for src in 'GF':
                    v = {c: float(pd_(src, k, sz, e, c).mean()) for c in TERMS}; mm = pd_(src, k, sz, e, 'mark'); cr = pd_(src, k, sz, e, 'carry')
                    b_ = ours[(src, k, sz, e, BES)]['all']
                    rep['configs'][f'{k}/{sz}/{e}/{src}'] = dict({c: round(v[c], 1) for c in TERMS}, mm_days_pos=int((mm > 0).sum()),
                                                                 carry_day_sd=round(float(cr.std(ddof=1)), 1), mm_ci=ci(mm), carry_ci=ci(cr),
                                                                 settle_ci=ci(pd_(src, k, sz, e, 'settle')), mm_1103=round(b_['mark'], 1), carry_1103=round(b_['carry'], 1))
                    print(f"{k} {sz:3s} {e:4s} {src} {v['spread']:+7.1f} {v['informed']:+7.1f} {v['plus1']:+7.1f} {v['drift']:+7.1f} {v['mark']:+7.1f} | "
                          f"{v['carry']:+7.1f} | {v['settle']:+7.1f} | {int((mm > 0).sum()):4d}/8 {cr.std(ddof=1):8.1f} | {b_['mark']:+.1f} / {b_['carry']:+.1f}")
    print('\nB. refreshed minus grid per term, day-clustered mean (t interval, 7 df) over the 8 read days; 11-03 beside')
    for sz in SIZES:
        for k in ('J', 'X'):
            for e in WINS:
                row = {}
                for c in TERMS:
                    row[c] = ci(pd_('F', k, sz, e, c) - pd_('G', k, sz, e, c))
                    row[c]['d1103'] = round(ours[('F', k, sz, e, BES)]['all'][c] - ours[('G', k, sz, e, BES)]['all'][c], 1)
                rep['diff'][f'{k}/{sz}/{e}'] = row
                print(f"  {k} {sz:3s} {e:4s} " + '  '.join(f"{c} {row[c]['mean']:+.1f} ({row[c]['lo']:+.1f}..{row[c]['hi']:+.1f})" for c in
                                                         ('spread', 'informed', 'drift', 'mark', 'carry', 'settle')))
    print('\nC. by minute of the window, 8 read days pooled, c / share: +1 s and post-1 s (to the outcome); makers vs ours (S240 arm, quotes all 240 s)')
    print(f"{'':22s} {'minute':8s} {'sh/day':>8s} {'+1 s':>7s} {'post1s':>7s} {'settle':>7s}")
    for bd in ('top 5', 'all makers'):
        for bi, bl in enumerate(BLAB):
            r = pool([makers[d][(bd, bi)] for d in D8 if (bd, bi) in makers[d]])
            rep['minute'][f'maker {bd}/{bl}'] = dict(q=round(r['q'] / 8.0), plus1=round(cps(r, 'plus1'), 3), post1=round(cps(r, 'post1'), 3), settle=round(cps(r, 'settle'), 3))
            print(f"maker {bd:16s} {bl:8s} {r['q'] / 8.0:8.0f} {cps(r, 'plus1'):+7.2f} {cps(r, 'post1'):+7.2f} {cps(r, 'settle'):+7.2f}")
    for sz in SIZES:
        for k in ('J', 'X'):
            for src in 'GF':
                for bi, bl in enumerate(BLAB):
                    r = pool([ours[(src, k, sz, 'S240', d)]['b'][bi] for d in D8 if bi in ours[(src, k, sz, 'S240', d)]['b']])
                    rep['minute'][f'ours {k}/{sz}/{src}/{bl}'] = dict(q=round(r['q'] / 8.0), plus1=round(cps(r, 'plus1'), 3), post1=round(cps(r, 'post1'), 3),
                                                                     settle=round(cps(r, 'settle'), 3))
                    print(f"ours {k} {sz:3s} {src:12s} {bl:8s} {r['q'] / 8.0:8.0f} {cps(r, 'plus1'):+7.2f} {cps(r, 'post1'):+7.2f} {cps(r, 'settle'):+7.2f}")
    print('\nD. ours (S240 arm) minus top 5 makers, +1 s c / share per minute: day-clustered mean (t interval, 7 df), days ours > top 5;'
          '\n   and refreshed minus grid post-1 s c / share per minute')
    for sz in SIZES:
        for k in ('J', 'X'):
            for bi, bl in enumerate(BLAB):
                line = f"  {k} {sz:3s} {bl:8s}"
                for src in 'GF':
                    dd = [cps(ours[(src, k, sz, 'S240', d)]['b'].get(bi), 'plus1') - cps(makers[d].get(('top 5', bi)), 'plus1') for d in D8]
                    if np.all(np.isfinite(dd)):
                        c = ci(dd); rep['vs_top5'][f'{k}/{sz}/{src}/{bl}'] = c; line += f" | {src} {c['mean']:+.2f} ({c['lo']:+.2f}..{c['hi']:+.2f}) {c['days_pos']}/8"
                    else:
                        rep['vs_top5'][f'{k}/{sz}/{src}/{bl}'] = 'INCOMPLETE'; line += f' | {src} INCOMPLETE'
                pp = [cps(ours[('F', k, sz, 'S240', d)]['b'].get(bi), 'post1') - cps(ours[('G', k, sz, 'S240', d)]['b'].get(bi), 'post1') for d in D8]
                if np.all(np.isfinite(pp)):
                    c = ci(pp); rep['post1_diff'][f'{k}/{sz}/{bl}'] = c; line += f" | post-1 s F - G {c['mean']:+.2f} ({c['lo']:+.2f}..{c['hi']:+.2f})"
                else:
                    rep['post1_diff'][f'{k}/{sz}/{bl}'] = 'INCOMPLETE'; line += ' | post-1 s F - G INCOMPLETE'
                print(line)
    io.write_json(f'{S4}/report_mm_decomp_fw.json', rep)
    return 0


if __name__ == '__main__':
    sys.exit(main())
