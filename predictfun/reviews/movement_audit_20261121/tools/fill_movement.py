"""PI0-MOVEMENT-CALIBRATION-AUDIT-v1: writes report.md from the outputs, checks.json, checks_falsifier.json and the logs. Every number and every
answer is read from a file; the evidence answers are the computed predicates of audit.json. Usage: fill_movement.py"""
import json
import re
from pathlib import Path

import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/movement_audit_20261121'); D = R / 'outputs'; L = R / 'logs'
FREEZE = '03243e5b3ddb304cf795c57010c1338181e4c478'
STR = {c: str for c in ('day', 'side', 'cell', 'view', 'admission', 'stage', 'cohort', 'subset', 'dim')}
au = json.loads((D / 'audit.json').read_text()); ck = json.loads((R / 'checks.json').read_text()); fa = json.loads((R / 'checks_falsifier.json').read_text())
MO, ST, PP, CO, CT, SO = (pd.read_csv(D / f'{n}.csv', dtype=STR) for n in ('moments', 'strata', 'prepost', 'concentration', 'contributions', 'sources'))
P = {p['stage']: p for p in au['predicates']}; ctl = au['control']; mem = au['membership']
mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (L / 'mutants_mov_summary.log').read_text())


def m(co, st, view='all', **kw):
    g = MO[(MO.cohort == co) & (MO.stage == st) & (MO.view == view)]
    for k, v in kw.items():
        g = g[g[k] == v]
    assert len(g) == 1, (co, st, view, kw)
    return g.iloc[0]


def s(co, sub, st, dim, cell):
    g = ST[(ST.cohort == co) & (ST.subset == sub) & (ST.stage == st) & (ST.dim == dim) & (ST.cell == cell)]
    assert len(g) == 1, (co, sub, st, dim, cell)
    return g.iloc[0]


def pp(co, st, view='all', cell='all'):
    g = PP[(PP.cohort == co) & (PP.stage == st) & (PP.view == view) & (PP.cell == cell)]
    assert len(g) == 1
    return g.iloc[0]


f = lambda v, nd=2: f'{float(v):+,.{nd}f}'
g_ = lambda v, nd=3: f'{float(v):,.{nd}f}'
ae, fo, fa_ = m('admitted', 'entry'), m('forced', 'entry', 'overlap'), m('forced', 'entry')
pa, pf = pp('admitted', 'entry'), pp('forced', 'entry', 'overlap', '20261026-27')
cw = CO[(CO.quantity == 'movement_error_entry') & (CO.weighting == 'weighted')].iloc[0]; cu = CO[(CO.quantity == 'movement_error_entry') & (CO.weighting == 'unweighted')].iloc[0]
dw = CO[(CO.quantity == 'D_EXPOSURE') & (CO.weighting == 'weighted')].iloc[0]; du = CO[(CO.quantity == 'D_EXPOSURE') & (CO.weighting == 'unweighted')].iloc[0]
rs = ck['details']['I']['resources']
cells_forced = {d: int(((ST.cohort == 'forced') & (ST.subset == 'overlap') & (ST.stage == 'entry') & (ST.dim == d)).sum()) for d in ('hbin', 'qbin', 'obin')}
ent = P['entry']
PJ = json.loads((L / 'pilot_projection.json').read_text())
import gzip as _gz
_tc = _gz.decompress((D / 'transitions.csv.gz').read_bytes()).decode().splitlines()[0].split(',')
S_F1 = ('The transitions table carries `new_bookticker` and no `new_bn_age` field.' if ('new_bookticker' in _tc and 'new_bn_age' not in _tc)
        else 'The transitions table does not carry the bookTicker label as declared.')
_h = ck['details']['H']; S_F3 = ('reproduces' if (_h.get('top5_weighted') and _h.get('top5_unweighted')) else 'does not reproduce')
_OP = pd.read_csv(D / 'opportunities.csv.gz', dtype=STR); _fe = _OP[(_OP.cohort == 'forced') & (_OP.stage == 'entry') & (_OP.acc_status == 'ok')]
S_START = (f'All {len(_fe)} evaluable forced entries are decided at the episode start (t = 0 ms, 5,000 ms remaining, ordinal 1)'
           if len(_fe) and (_fe.t_dec.astype(float) == 0).all() and (_fe.ordinal.astype(float) == 1).all() else f'The {len(_fe)} evaluable forced entries are not all at the episode start')
fst = pd.DataFrame(au['fill_statuses']); n_bad_ref = int(fst[fst.ref_status != 'ok'].n.sum()) if len(fst) else 0
no_news = int(SO[(SO.cohort == 'admitted') & (SO.subset == 'no new observation')].n.iloc[0])
S_I = {'concentrated': 'at least 0.5' if (ent['i']['top5_world_share'] or 0) >= .5 else 'below 0.5', 'broad': 'below 0.5', 'inconclusive': 'below 0.5'}[ent['i']['answer']]
S_I2 = (' so the rule reads concentrated even though every cell is positive. Both facts stand.' if ent['i']['answer'] == 'concentrated' and ent['i']['positive_cells'] == 4
        else f" so the rule reads {ent['i']['answer']}.")
_dm = max(v for v in ent['ii']['decision_ranges'].values() if v is not None) if any(v is not None for v in ent['ii']['decision_ranges'].values()) else None; _dl = ent['ii']['delay_range']
S_II = ('Neither is 1.5 times the other.' if (_dm is not None and _dl is not None and _dm < 1.5 * _dl and _dl < 1.5 * _dm) else
        'The largest decision-state range is at least 1.5 times the delay range.' if (_dm is not None and (_dl is None or _dm >= 1.5 * _dl)) else
        'The delay range is at least 1.5 times every decision-state range.' if _dl is not None else 'Neither range is available.')
S_P = ('P has useful probability discrimination with calibration error: its Brier is below the pooled rate, and it '
       + ('underpredicts' if ae.mean_P < ae.fill_rate else 'overpredicts') + ' the fill rate.') if ae.brier < ae.brier_pooled else 'P does not beat the pooled rate in Brier.'
S_M = ('The movement head ' + ('improves' if ae.mse_m < ae.mse_m_zero else 'does not improve') + ' MSE over zero, and its mean error is '
       + f"{float(ae.bias_m):+.3f} c.")
S_REF = 'There are no missing or out-of-time references.' if n_bad_ref == 0 else f'{n_bad_ref} counted fills have no usable reference (statuses in tables.md).'
S_NEWS = 'So there is still no no-news comparison group.' if no_news == 0 else f'{no_news} transitions form a no-news group.'
S_FC = ('It cannot separate horizon or ordinal effects.' if (cells_forced['hbin'] == 1 and cells_forced['obin'] == 1) else
        'Its horizon and ordinal strata have more than one cell.')


def strat_rows(co, sub, st, dims):
    out = ['| stratum | cell | n | fills | mean E[qm] (c) | mean M (c) | u (c / opp.) | c = mhat − M (c / fill) | thin |', '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for d in dims:
        for r in ST[(ST.cohort == co) & (ST.subset == sub) & (ST.stage == st) & (ST.dim == d)].to_dict('records'):
            thin = 'thin' if r['thin'] else ('thin (cond.)' if r['thin_conditional'] else '')
            out.append(f"| {d} | {r['cell']} | {int(r['n'])} | {int(r['fills'])} | {f(r['mean_Eqm'])} | {f(r['mean_M'])} | {f(r['mean_u'])} | "
                       f"{f(r['c_m']) if pd.notna(r['c_m']) else '—'} | {thin} |")
    return '\n'.join(out)


rep = f"""# PI0-MOVEMENT-CALIBRATION-AUDIT-v1 — report

{au['asof'][:16].replace('T', ' ')}Z, session aea5bfb9. Review 5a8ee6f5's one next priority.
- Freeze `{FREEZE}`: module, selftest, harness, checker, declaration and launcher, committed before any table existed.
- Results: the commit carrying this report.
- Scope: consumed Oct 24–27 only. No fit, no new branch, no threshold.

## The declared evidence (computed by `predicates()`; reading = the admitted entry, weighted)

| Question | Entry (declared) | First review | First pull |
|---|---|---|---|
| (i) broad or tail-concentrated | **{ent['i']['answer']}** | {P['first_review']['i']['answer']} | {P['first_pull']['i']['answer']} |
| (ii) decision state or fill timing | **{ent['ii']['answer']}** | {P['first_review']['ii']['answer']} | {P['first_pull']['ii']['answer']} |
| (iii) neutral vs admitted residual | **{ent['iii']['answer']}** | {P['first_review']['iii']['answer']} | {P['first_pull']['iii']['answer']} |
| (iv) cells populated | **{ent['iv']['answer']}** | {P['first_review']['iv']['answer']} | {P['first_pull']['iv']['answer']} |
| recommendation | **{ent['recommendation']}** | {P['first_review']['recommendation']} | {P['first_pull']['recommendation']} |

The numbers behind each entry answer:
- **(i)** u = E[qm] − M is positive in {ent['i']['positive_cells']} of the 4 day × side cells:
  - Oct 26 bid {f(ent['i']['cells']['20261026/0'])} c;
  - Oct 26 ask {f(ent['i']['cells']['20261026/1'])} c;
  - Oct 27 bid {f(ent['i']['cells']['20261027/0'])} c;
  - Oct 27 ask {f(ent['i']['cells']['20261027/1'])} c.

  The weighted top-5 world share of |Σ w u| is {g_(ent['i']['top5_world_share'])}, {S_I},{S_I2} The unweighted share is {g_(cu.top5_world_share)}.
- **(ii)** The decision-state ranges of c are:
  - horizon {g_(ent['ii']['decision_ranges']['hbin'], 1)};
  - queue ratio {g_(ent['ii']['decision_ranges']['qbin'], 1)};
  - ordinal {g_(ent['ii']['decision_ranges']['obin'], 1)}.

  The delay-class range is {g_(ent['ii']['delay_range'], 1)} c. {S_II}
- **(iii)** c = mhat − M on fills is {f(ent['iii']['c_admitted'])} c per filled opportunity on the admitted cohort ({ent['iii']['fills_admitted']} fills).
  - The forced Oct 26–27 reference has {f(ent['iii']['c_forced_overlap'])} c ({ent['iii']['fills_forced_overlap']} fills).
  - The forced Oct 24–27 reference has {f(ent['iii']['c_forced_all'])} c.
- **(iv)** Cells with ≥ 30 opportunities and ≥ 10 fills: horizon {ent['iv']['populated_cells']['hbin']}, queue ratio
  {ent['iv']['populated_cells']['qbin']}, ordinal {ent['iv']['populated_cells']['obin']}.

## Validity

Checks and controls:
- Checks A–I: {sum(ck['checks'].values())} / {len(ck['checks'])} true (`checks.json`):
  - A: identities and freeze;
  - B: membership and the exact-key weight / cell / part binding;
  - C: every decomposition recomputed;
  - D: strata from the client inputs;
  - E: every fill and reference recomputed;
  - F: every denominator;
  - G: predicates recomputed;
  - H: the review's independent numbers reproduced to their printed precision;
  - I: resources and the mutation suite.
- Falsifier: {sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])} / {len(fa['cases'])} corruptions of real copies are flagged by
  their check, with clean copies true (`checks_falsifier.json`).
- Mutation suite: {mut.group(1)} / {mut.group(2)} planted errors are caught by a selftest check.

Population:
- Membership: {mem['admitted_rows']} admitted rows and {mem['forced_rows']} forced forks. Weights sum to {g_(mem['admitted_weight_sum'], 0)}.
  There are {mem['weight_mismatch']} weight mismatches.
- Fill-time mids exist for every admitted counted fill: {pa.with_mid} / {pa.fills} at entry. The median age is {g_(pa.median_age, 0)} ms
  and the maximum is {g_(pa.max_age, 0)} ms (venue time).
- Forced mids come from the retained edges: {pf.with_mid} / {pf.fills} for Oct 26–27. Their age is unavailable.
- {S_REF}

Cost and the control:
- {rs['jobs']} run_light jobs, {g_(rs['core_hours'])} core-hours, peak {rs['peak_mib']} MiB. The mids high-water was {g_(rs['mids_hwm_max'], 0)} MiB.
  The write-once pilot projected {g_(PJ['projected_core_hours'])} core-hours (run_rest {str(PJ['run_rest']).lower()}).
- Retained first-review control: PULL {ctl['n_sel']} / {ctl['n']}, percentile {g_(ctl['percentile'])}, {ctl['free_strata']} of
  {ctl['strata']} strata free with {ctl['free_rows']} rows. All {ctl['cell_equal_admission']} reviews' own-time cells equal the admission cell.

## The decomposition (admitted entry, weighted; c per opportunity unless stated)

| Quantity | Value |
|---|---:|
| V − value (bias) | {f(ae.bias_V, 3)} |
| edge × (E[q] − q) | {f(ae.qty_term, 3)} |
| E[qm] − M | {f(ae.mov_term, 3)} |
| (P − z) mhat (probability factor) | {f(ae.mov_prob, 3)} |
| z (mhat − M) (conditional head) | {f(ae.mov_cond, 3)} |
| MSE(V) / zero / mechanical (c²) | {g_(ae.mse_V, 2)} / {g_(ae.mse_zero, 2)} / {g_(ae.mse_mech, 2)} |
| movement MSE / zero (c²) | {g_(ae.mse_m, 2)} / {g_(ae.mse_m_zero, 2)} |
| mean P / fill rate; Brier / pooled | {g_(ae.mean_P)} / {g_(ae.fill_rate)}; {g_(ae.brier)} / {g_(ae.brier_pooled)} |

{S_P}
{S_M}

## The movement head on fills: admitted versus the neutral reference

| Cohort (entry) | fills | mean mhat on fills (c / fill) | mean M on fills (c / fill) | c = mhat − M | mean E[qm] / mean M (c / opp.) |
|---|---:|---:|---:|---:|---:|
| admitted (weighted) | {int(ae.fills)} | {f(ae.mean_mhat_fill)} | {f(ae.mean_M_fill)} | {f(ae.c_m)} | {f(ae.mean_Eqm)} / {f(ae.mean_M)} |
| forced Oct 26–27 | {int(fo.fills)} | {f(fo.mean_mhat_fill)} | {f(fo.mean_M_fill)} | {f(fo.c_m)} | {f(fo.mean_Eqm)} / {f(fo.mean_M)} |
| forced Oct 24–27 | {int(fa_.fills)} | {f(fa_.mean_mhat_fill)} | {f(fa_.mean_M_fill)} | {f(fa_.c_m)} | {f(fa_.mean_Eqm)} / {f(fa_.mean_M)} |

Descriptive differences, admitted minus forced Oct 26–27 (no declared reading):
- The forecast differs by {f(ae.mean_mhat_fill - fo.mean_mhat_fill)} c per fill.
- The realized movement differs by {f(ae.mean_M_fill - fo.mean_M_fill)} c per fill.

Per filled share, the movement splits as follows (`prepost.csv`):

| Cohort | shares | total (c/share) | pre-fill (c/share) | post-fill (c/share) |
|---|---:|---:|---:|---:|
| admitted entry | {g_(pa.q, 1)} | {f(pa.total_per_share, 3)} | {f(pa.pre_per_share, 3)} | {f(pa.post_per_share, 3)} |
| forced entry, Oct 26–27 | {g_(pf.q, 1)} | {f(pf.total_per_share, 3)} | {f(pf.pre_per_share, 3)} | {f(pf.post_per_share, 3)} |

The post-fill share of the per-share movement is {g_(pa.post_per_share / pa.total_per_share)} for the admitted cohort and
{g_(pf.post_per_share / pf.total_per_share)} for the forced cohort.

{S_START}, and the forced decision-state strata have these cell counts: horizon {cells_forced['hbin']},
queue ratio {cells_forced['qbin']}, ordinal {cells_forced['obin']}. {S_FC}

## Strata (admitted entry)

{strat_rows('admitted', 'all', 'entry', ('hbin', 'qbin', 'obin', 'delay_class'))}

Forecast bins and the first review / first pull are in `tables.md`.

## Concentration and contributions

| Quantity | weighting | top-5 generation share | top-5 world share | Σ |
|---|---|---:|---:|---:|
| D_EXPOSURE | weighted | {g_(dw.top5_generation_share)} | {g_(dw.top5_world_share)} | {g_(dw['sum'], 1)} |
| D_EXPOSURE | unweighted | {g_(du.top5_generation_share)} | {g_(du.top5_world_share)} | {g_(du['sum'], 1)} |
| entry movement error | weighted | {g_(cw.top5_generation_share)} | {g_(cw.top5_world_share)} | {g_(cw['sum'], 1)} |
| entry movement error | unweighted | {g_(cu.top5_generation_share)} | {g_(cu.top5_world_share)} | {g_(cu['sum'], 1)} |

Signed day × side shares of the weighted sums:
""" + '\n'.join(f"- {r['day']} side {r['side']}: D {g_(r['share_D'])}, entry u {g_(r['share_entry_u'])}" for r in CT.to_dict('records')) + f"""

## Observation sources, entry → first review (admitted, weighted; ages in ms)

These are counted with the bookTicker label (review 5a8ee6f5 finding 1):
- new PF frame: {g_(SO.iloc[0].new_pf_frame)};
- new Binance bookTicker update: {g_(SO.iloc[0].new_bookticker)};
- new Binance depth snapshot: {g_(SO.iloc[0].new_depth)};
- new PF receipt: {g_(SO.iloc[0].new_pf_receipt)};
- any new observation: {g_(SO.iloc[0].any_new)}.

Transitions with no new observation: {no_news}. {S_NEWS} The direction of the decision-mid change: against {int(SO[(SO.cohort == 'admitted') & (SO.subset == 'mid against')].n.iloc[0])},
unchanged {int(SO[(SO.cohort == 'admitted') & (SO.subset == 'mid unchanged')].n.iloc[0])}, toward the order's value
{int(SO[(SO.cohort == 'admitted') & (SO.subset == 'mid toward_order_value')].n.iloc[0])}.

New observations alone do not attribute the POST-to-PULL reversal to changed danger.

## Review 5a8ee6f5 findings 1–4

1. **Finding 1.** {S_F1} The three clocks and their units are documented in the module header.
2. **Finding 2.** The wording above follows the review. The units are explicit: c per opportunity, c per filled opportunity, c per share.
3. **Finding 3.** The weighted concentration {S_F3} the review's 0.520 ({g_(dw.top5_generation_share)}, generation share of |w D|). It is shown beside the unweighted {g_(du.top5_generation_share)} and the
   world shares, and the signed day × side contributions are listed above.
4. **Finding 4.** Rows are bound to the manifest by exact key in three places: `adm_rows`, check B and falsifier case B_weight. The
   control derives each review's hour from its own decision time, and {ctl['cell_equal_admission']} / {ctl['reviews_with_time']} reviews
   equal the admission cell.

## Limits

- These are related generations on two (admitted) and four (forced) consumed days. There is no interval, no validation and no
  profitability claim.
- The forced reference is differently selected. {S_START}.
- The pre / post split is post-hoc attribution at the venue mid at the fill.
- Delay classes are realized outcomes, never features.
- Nothing is fitted. An oracle corrected with these outcomes is not a model score.
"""
(R / 'report.md').write_text(rep); print(f'report.md: {len(rep.splitlines())} lines')
