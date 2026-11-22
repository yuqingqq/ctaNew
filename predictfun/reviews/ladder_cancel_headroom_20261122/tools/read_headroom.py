"""POST-RUN READOUT of FEASIBLE-CANCEL-HEADROOM-v1 (session aea5bfb9, 2026-11-22; declared and frozen before the run): descriptive arithmetic on the
frozen outputs, no new reading rule. Reads data/.../{generations_core, summary, sequences, panel_summary}.csv (their sha256 = audit.json) and the
Codex review's committed classification (c72f0d37). Writes outputs/read_headroom.json (write-once):
 T1 anchor (a) per arm: classes with the Codex split of good_only_or_no_fill (NO_SUFFIX_FILLS: no suffix fill; GOOD_ONLY: G > TOL, L <= TOL;
    NEAR_ZERO: the rest), generations and suffix G / L / net; equal to the Codex counts and nets per arm (computed);
 T2 oracle (b) per arm x day and pooled: recorded-KEEP, first-cut and oracle retained +1 s, headroom, cuts, removed L, sacrificed G and positive
    shares, residual L, END recorded vs oracle, chosen-cut classes and loss shares, top-3 market share, through / same-price parts;
 T3 prefix link (USER correction 90e82f8e): anchor-(a) class x prefix status (prefix_loss: a_pre_L > TOL; prefix_gain_only: a_pre_G > TOL, no
    loss; no_valued_prefix) x end cause: generations, prefix G / L, suffix G / L (the suffix anchor is not the whole-life question);
 T4 horizons at the oracle cut: native removed G / L at 5 s, 30 s, END and the common-cohort removed value per horizon with its sign;
 T5 sequences (generation-level flags, overlapping); T6 the decision-state panel by class (labeled: decision states, not samples);
 T7 recorded queue fields of oracle-cut vs other feasible generations (NaN counted as unknown, never front). Usage: read_headroom.py"""
import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); D = REPO / 'data/predict_fun/research/ladder_cancel_headroom_v1_20261122'; OUT = REPO / 'predictfun/reviews/ladder_cancel_headroom_20261122/outputs'
CODEX = 'predictfun/reviews/automatic/20261122_queue_fill_classes/outputs/generations.csv.gz'; TOL = 1e-6; ARMS = ('CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')
R2 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def sub_class(G):
    return np.where(G.a_cls != 'good_only_or_no_fill', G.a_cls, np.where(G.a_suf_n == 0, 'NO_SUFFIX_FILLS', np.where((G.a_suf_G > TOL) & (G.a_suf_L <= TOL), 'GOOD_ONLY', 'NEAR_ZERO')))


def main():
    p = OUT / 'read_headroom.json'; assert not p.exists(), 'read_headroom.json exists (write-once)'; au = json.loads((D / 'audit.json').read_text())
    for n in ('generations_core.csv', 'summary.csv', 'sequences.csv', 'panel_summary.csv'):
        assert hashlib.sha256((D / n).read_bytes()).hexdigest() == au['outputs'][n], f'{n} differs from the audit'
    G = pd.read_csv(D / 'generations_core.csv', dtype={'day': str}); S = pd.read_csv(D / 'summary.csv', dtype={'day': str, 'side': str})
    Q = pd.read_csv(D / 'sequences.csv', dtype={'day': str}); P = pd.read_csv(D / 'panel_summary.csv', dtype={'day': str, 'side': str})
    C = pd.read_csv(io.BytesIO(gzip.decompress(subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{CODEX}'], capture_output=True, check=True).stdout)), dtype={'day': str})
    G['sub'] = sub_class(G); res = dict(asof=au['asof'], reading=au['reading'])
    t1, match = {}, True
    for arm in ARMS:
        g = G[G.arm == arm]; c = C[C.arm == arm]; rows = {}
        for k, h in g.groupby('sub'):
            rows[k] = dict(generations=int(len(h)), G=R2(h.a_suf_G.sum()), L=R2(h.a_suf_L.sum()), net=R2(h.a_suf_G.sum() - h.a_suf_L.sum()))
        cm = {k: (int(len(h)), float(h.net.sum())) for k, h in c.groupby('category')}
        name = dict(no_feasible_callback='NO_OPPORTUNITY', incomplete_valuation='INCOMPLETE', pure_observed_loss='PURE_BAD', compensated_mixed='MIXED_COMPENSATED',
                    uncompensated_mixed='MIXED_UNCOMPENSATED', mixed_tie='MIXED_TIE')
        mine = {name.get(k, k): (v['generations'], float(g[g['sub'] == k].a_suf_G.sum() - g[g['sub'] == k].a_suf_L.sum())) for k, v in rows.items()}
        ok = set(mine) == set(cm) and all(mine[k][0] == cm[k][0] and abs(mine[k][1] - cm[k][1]) <= 1e-6 * max(1., abs(cm[k][1])) for k in cm if k not in ('NO_OPPORTUNITY', 'INCOMPLETE'))
        ok = ok and all(mine[k][0] == cm[k][0] for k in ('NO_OPPORTUNITY', 'INCOMPLETE') if k in cm); match &= bool(ok)
        t1[arm] = dict(classes=rows, all_L=R2(g.all_L.sum()), suffix_L_at_a=R2(g.a_suf_L.sum()), pure_share_of_suffix_L=R2(100. * g[g['sub'] == 'pure_observed_loss'].a_suf_L.sum()
                                                                                                              / g.a_suf_L.sum()), matches_codex=bool(ok))
    res['T1_anchor_a'] = t1; res['T1_matches_codex_every_arm'] = bool(match)
    keep = ['v_keep', 'a_v', 'b_v', 'headroom_oracle', 'headroom_first', 'oracle_cuts', 'b_suf_L', 'b_suf_G', 'b_suf_q_pos', 'b_suf_q_neg', 'all_L', 'all_G', 'end_keep', 'b_end',
            'a_end', 'late_end_keep', 'b_late_end', 'b_pure_observed_loss_n', 'b_pure_observed_loss_L', 'b_uncompensated_mixed_n', 'b_uncompensated_mixed_L', 'b_uncompensated_mixed_G',
            'b_incomplete_valuation_n', 'b_incomplete_valuation_L', 'b_oracle_keep_n', 'b_no_feasible_callback_n', 'b_removed_top3_market_share', 'b_suf_thr_L', 'b_suf_thr_G',
            'b_suf_same_L', 'b_suf_same_G', 'all_thr_L', 'all_same_L', 'reading', 'generations', 'feasible', 'b_fallback_cuts']
    t2 = {}
    for arm in ARMS:
        s = S[(S.arm == arm) & (S.side == 'all')]
        t2[arm] = {r['day']: {k: (r[k] if isinstance(r[k], str) else R2(r[k])) for k in keep} for r in s.to_dict('records')}
        for d, r in t2[arm].items():
            r['residual_L'] = R2(r['all_L'] - r['b_suf_L']); r['removed_L_share'] = R2(100. * r['b_suf_L'] / r['all_L']) if r['all_L'] else None
    res['T2_oracle'] = t2
    G['prefix'] = np.where(G.a_pre_L > TOL, 'prefix_loss', np.where(G.a_pre_G > TOL, 'prefix_gain_only', 'no_valued_prefix'))
    t3 = {}
    for arm in ARMS:
        g = G[(G.arm == arm) & (G.status == 'feasible')]; rows = []
        for (k, pf, ec), h in g.groupby(['sub', 'prefix', 'end_cause']):
            rows.append(dict(cls=k, prefix=pf, end_cause=ec, generations=int(len(h)), prefix_G=R2(h.a_pre_G.sum()), prefix_L=R2(h.a_pre_L.sum()), suffix_G=R2(h.a_suf_G.sum()),
                             suffix_L=R2(h.a_suf_L.sum())))
        bycls = {k: dict(generations=int(len(h)), with_prefix_loss=int((h.prefix == 'prefix_loss').sum()), prefix_L=R2(h.a_pre_L.sum())) for k, h in g.groupby('sub')}
        t3[arm] = dict(by_class=bycls, cells=rows)
    res['T3_prefix_link'] = t3
    t4 = {}
    for arm in ARMS:
        r = S[(S.arm == arm) & (S.day == 'all') & (S.side == 'all')].iloc[0]
        t4[arm] = dict(native={h: dict(G=R2(r[f'b_suf_G{h}']), L=R2(r[f'b_suf_L{h}'])) for h in ('5', '30', 'end')},
                       common_cohort=dict(n=int(r.b_suf_cc_n), **{f'removed_v{h}': R2(r[f'b_suf_cc_v{h}']) for h in ('1', '5', '30', 'end')},
                                          **{f'sign_{h}': int(r[f'b_cc_sign_{h}']) for h in ('1', '5', '30', 'end')}, **{f'loss_recovers_by_{h}': bool(r[f'b_loss_recovers_by_{h}'])
                                                                                                                  for h in ('5', '30', 'end')}),
                       retained=dict(keep_v5=R2(r.v5_keep), oracle_v5=R2(r.b_v5), keep_v30=R2(r.v30_keep), oracle_v30=R2(r.b_v30), keep_end=R2(r.end_keep), oracle_end=R2(r.b_end)))
    res['T4_horizons_at_oracle_cut'] = t4
    res['T5_sequences'] = {arm: {r['flag']: {k: (R2(r[k]) if k in ('G', 'L') else (None if pd.isna(r[k]) else int(r[k]))) for k in ('generations', 'feasible', 'G', 'L', 'compensated', 'uncompensated')
                                             if k in r} for r in Q[(Q.arm == arm) & (Q.day == 'all')].to_dict('records')} for arm in ARMS}
    res['T6_panel'] = {arm: {r['cls']: dict(rows=int(r['rows']), generations=int(r['generations']), G=R2(r['G']), L=R2(r['L'])) for r in P[(P.arm == arm) & (P.day == 'all')].to_dict('records')}
                       for arm in ARMS}
    t7 = {}
    for arm in ARMS:
        g = G[(G.arm == arm) & (G.status == 'feasible')]
        for nm, h in (('oracle_cut', g[g.b_cut.notna()]), ('not_cut', g[g.b_cut.isna()])):
            t7[f'{arm}/{nm}'] = dict(generations=int(len(h)), ahead_act_median=R2(h.q_ahead_act.median()), ahead_act_zero=int((h.q_ahead_act == 0).sum()),
                                     first_venue_ahead_median=R2(h.q_first_venue_ahead.median()), first_known_ahead_unknown=int(h.q_first_known_ahead.isna().sum()),
                                     b_age_median=R2(h.b_age.median()), size_median=R2(h['size'].median()))
    res['T7_queue'] = t7
    p.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)[:9000])


if __name__ == '__main__':
    main()
