"""Post-hoc CAP10 audit on the nine consumed inventory-experiment days.

Accounting: spread + first-second price move + drift to 60.1s + carry to
settlement. Carry is also marked at 120/180/240/299s without assuming an exit.
Robustness: paired UTC days, leave-one-day-out totals, early/late blocks,
and all 512 whole-day sign flips, including a max-|t| screen over the original
10 variants x 2 timings. The sign-flip null assumes symmetric day effects;
it is descriptive after selection and does not cover earlier research searches.
No forward data, parameter tuning, replays, or new trading policy.

Run both declare and run through run_light.sh. Imports reuse the existing
engine-reconciled mark lookup/decomposition; corrupted fill controls must fail.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .pf_mm_decomp import mid_maps, lookup, decompose, recon
from .pf_inv_report import DAYS, arm, name, stem, run_dir

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / 'data/predict_fun/research/step4'
DEST = S4 / 'cap10_attribution_20261107'
ARMS = {'v1': arm(''), 'CAP10': arm('CAP10')}
TERMS = ('spread', 'informed', 'plus1', 'drift', 'mark', 'carry', 'settle')
HORIZONS = (120, 180, 240, 299)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def identities():
    paths = [Path(__file__), ROOT / 'predictfun/research/pf_mm_decomp.py',
             ROOT / 'predictfun/research/pf_inv_report.py', ROOT / 'predictfun/research/pf_step4_ladder_inv.py',
             ROOT / 'predictfun/research/pf_step4_ladder.py', S4 / 'report_inv_v1.json', S4 / 'inv_check.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def declare():
    if DEST.exists():
        raise ValueError('audit destination already exists')
    DEST.mkdir()
    d = dict(declared_at=datetime.now(timezone.utc).isoformat(), days=DAYS, arms=ARMS, timings=['J', 'X'],
             code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             identities=identities(), stop_mark_seconds=60.1, carry_mark_seconds=HORIZONS,
             unit='UTC day; all matched windows including zero-fill windows',
             null='paired policy difference has zero mean and symmetric whole-day effects; all 512 day sign flips',
             null_samples=512, multiplicity='max absolute studentized statistic over original 10 variants x J/X',
             confidence_interval='paired day t interval, df=8; exploratory and unadjusted',
             carry_control='scale v1 inventory per day to CAP10 market-implied binary variance q^2*m60*(1-m60); ex-post diagnostic, not an executable policy',
             limitation='CAP10 was already selected on these consumed days. This is not out-of-sample validation; multiplicity screen omits earlier searches.',
             controls=['accounting reconciles to engine', 'all baseline windows match round i',
                       'corrupt fill quantity is refused', 'identical/opposed paired effects exercise sign-flip checker'],
             forward_data_read=False)
    write(DEST / 'declaration.json', d)
    print('Declared', DEST)


def paired(x):
    x = np.asarray(x, float)
    mean = float(x.mean()); half = float(2.306004135 * x.std(ddof=1) / np.sqrt(len(x)))
    omit = x.sum() - x
    return dict(n_days=len(x), sum=float(x.sum()), mean_per_day=mean,
                day_t_interval=[mean - half, mean + half], positive_days=int((x > 0).sum()),
                leave_one_day_out_min=float(omit.min()), leave_one_day_out_max=float(omit.max()),
                largest_positive_day=float(x.max()), sum_without_best_day=float(x.sum() - x.max()),
                first_five_sum=float(x[:5].sum()), last_four_sum=float(x[5:].sum()), values=x.tolist())


def studentized(a, axis):
    mean = a.mean(axis=axis)
    scale = a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])
    # An exactly constant nonzero effect gets an arbitrarily large finite t;
    # an exactly zero effect has t=0 (finite so the receipt is valid JSON).
    return np.divide(mean, np.maximum(scale, 1e-12))


def signflip(matrix):
    if matrix.shape[0] != 9 or not np.isfinite(matrix).all():
        raise ValueError('sign-flip input must contain nine finite day rows')
    signs = np.array(list(itertools.product((-1., 1.), repeat=9)))
    observed = studentized(matrix, axis=0)
    null = studentized(signs[:, :, None] * matrix[None, :, :], axis=1)
    raw = (np.abs(null) >= np.abs(observed)[None, :] - 1e-10).mean(axis=0)
    maxima = np.abs(null).max(axis=1)
    adjusted = (maxima[:, None] >= np.abs(observed)[None, :] - 1e-10).mean(axis=0)
    return observed, raw, adjusted


def controls():
    t, p, adj = signflip(np.c_[np.zeros(9), np.ones(9), -np.ones(9)])
    assert p[0] == 1 and p[1] == p[2] == 2 / 512 and (adj >= p).all()
    try:
        signflip(np.zeros((8, 1)))
    except ValueError:
        return dict(zero_effect_p=1., signed_constant_effect_p=float(p[1]), invalid_day_count_refused=True)
    raise ValueError('sign-flip refusal control failed')


def run():
    declaration = json.loads((DEST / 'declaration.json').read_text())
    if declaration['identities'] != identities():
        raise ValueError('code/input identity changed after declaration')
    if not json.loads((S4 / 'inv_check.json').read_text())['PASS']:
        raise ValueError('original inventory check did not pass')
    prior = json.loads((S4 / 'report_inv_v1.json').read_text())
    family = [(tm, label) for tm in ('J', 'X') for label in prior['arms'] if label != 'v1']
    if len(family) != 20:
        raise ValueError('original comparison family changed')
    family_days = {key: [] for key in family}
    per = {tm: {} for tm in ('J', 'X')}
    inputs = {}; refusals = 0; status = dict(fills=0, missing_fill_mid=0, missing_1s_mid=0, missing_stop_mid=0, missing_preexpiry_mid=0)
    for day in DAYS:
        s1 = ROOT / 'data/predict_fun/research/step1' / day
        regpath = s1 / 'registry.parquet'
        inputs[str(regpath.relative_to(ROOT))] = digest(regpath)
        reg = pd.read_parquet(regpath, columns=['mid', 'ws', 'outcome']).set_index('mid')
        for p in sorted((s1 / 'books').iterdir()):
            inputs[str(p.relative_to(ROOT))] = digest(p)
        sm0 = json.loads((Path(run_dir('ivJ', day)) / 'summary.json').read_text())
        bounds = sm0['input_bounds']
        by = mid_maps(day, bounds['t_lo_ms'], bounds['t_hi_ms'])
        for tm in ('J', 'X'):
            base = Path(run_dir('iv' + tm, day))
            sp = base / 'summary.json'; inputs[str(sp.relative_to(ROOT))] = digest(sp)
            sm = json.loads(sp.read_text())
            if sm['input_bounds'] != bounds or sm['day'] != day:
                raise ValueError('input bounds/day mismatch')
            env = sm['env']
            if (tm == 'J') != (env.get('PF_TXJIT') == 'frame') or (tm == 'X') != (env.get('PF_TXSHIFT') == '-50' and env.get('PF_QACT') == 'max'):
                raise ValueError('timing identity mismatch')
            terms = {}; window = {}; weights = {}; ending = {}
            for label, spec in ARMS.items():
                fp, wp = base / 'fills' / (stem(spec) + '.parquet'), base / 'win' / (stem(spec) + '.parquet')
                for p in (fp, wp): inputs[str(p.relative_to(ROOT))] = digest(p)
                f, w = pd.read_parquet(fp), pd.read_parquet(wp).sort_values('ws').reset_index(drop=True)
                if len(w) != 288 or w.mid.duplicated().any() or not reg.loc[w.mid, 'outcome'].isin(['Up', 'Down']).all():
                    raise ValueError('invalid day population')
                if len(f) != sm['arms'][name(spec)]['fills'] or (f.t > 60.1 + 1e-9).any():
                    raise ValueError('fill population/stop time mismatch')
                dec = decompose(f, by, reg, 60.)
                errs = recon(dec, f, w)
                if errs: raise ValueError(f'{day} {tm} {label}: {errs}')
                status['fills'] += len(f)
                status['missing_fill_mid'] += int(dec.spread.isna().sum())
                status['missing_1s_mid'] += int(dec.plus1.isna().sum())
                status['missing_stop_mid'] += int(dec.mark.isna().sum())
                if not np.isfinite(dec[list(TERMS)]).all().all():
                    raise ValueError('missing decomposition marks; cannot silently exclude fills')
                bad = f.copy(); bad.loc[bad.index[0], 'q'] += 10.
                if not recon(decompose(bad, by, reg, 60.), f, w):
                    raise ValueError('corrupt quantity falsifier did not fire')
                refusals += 1
                if label == 'v1':
                    ref = pd.read_parquet(Path(run_dir('l' + tm + 'i', day)) / 'win' / (stem(spec) + '.parquet')).sort_values('ws').reset_index(drop=True)
                    if not w.equals(ref): raise ValueError('baseline window reference changed')
                ws = reg.loc[w.mid, 'ws'].to_numpy() * 1000
                m60 = lookup(by, w.mid.to_numpy(), ws + 60100., 'right') / 100
                y = (reg.loc[w.mid, 'outcome'].to_numpy() == 'Up').astype(float)
                q = w.net_end.to_numpy()
                if not np.allclose(w.carry_stop, q * (y - m60), atol=1e-8, rtol=0):
                    raise ValueError('inventory carry does not reconcile')
                if not np.isfinite(m60).all(): raise ValueError('missing common stop marks')
                late = {}
                for sec in HORIZONS:
                    m = lookup(by, w.mid.to_numpy(), ws + sec * 1000, 'right') / 100
                    status['missing_preexpiry_mid'] += int(np.isnan(m).sum())
                    if not np.isfinite(m).all(): raise ValueError('missing late marks')
                    late[str(sec)] = dict(price_drift=float(np.sum(q * (m - m60))),
                                          settlement_residual=float(np.sum(q * (y - m))),
                                          marked_pnl=float(w.mtm_stop.sum() + np.sum(q * (m - m60))))
                ages = []
                for mid, start in zip(w.mid, ws):
                    tt = by[mid][0]; j = np.searchsorted(tt, start + 299000., side='right') - 1
                    ages.append(start + 299000. - tt[j])
                # Exposure at the common mark; p=m is a risk proxy, not an alpha model.
                risk = float(np.sum(q*q*m60*(1-m60)))
                terms[label] = dict(components={k: float(dec[k].sum()) for k in TERMS}, windows=len(w), fills=len(f), shares=float(f.q.sum()),
                                    abs_inventory_mean=float(np.abs(q).mean()), abs_inventory_p90=float(np.quantile(np.abs(q), .9)),
                                    inventory_risk_proxy=risk, carry_sd=float(w.carry_stop.std()), settlement_sd=float(w.pnl.std()),
                                    late=late, last_mark_age_ms=dict(median=float(np.median(ages)), p90=float(np.quantile(ages,.9)),
                                                                   maximum=float(np.max(ages)), over_1s=int((np.array(ages)>1000).sum())))
                window[label] = w
                weights[label] = risk
                ending[label] = q
            if not window['v1'][['mid','ws']].equals(window['CAP10'][['mid','ws']]):
                raise ValueError('policies have different window populations')
            scale = float(np.sqrt(weights['CAP10']/weights['v1'])) if weights['v1'] else 0.
            control = dict(v1_inventory_scale=scale, matched_proxy=weights['v1']*scale*scale,
                           scaled_v1_carry=float(window['v1'].carry_stop.sum()*scale),
                           cap10_carry=float(window['CAP10'].carry_stop.sum()))
            control['cap10_minus_scaled_v1_carry'] = control['cap10_carry'] - control['scaled_v1_carry']
            per[tm][day] = dict(arms=terms, delta={k: terms['CAP10']['components'][k]-terms['v1']['components'][k] for k in TERMS}, risk_matched_carry_control=control)
            for timing, label in family:
                if timing != tm: continue
                wp = base / 'win' / (stem(prior['arms'][label]) + '.parquet')
                inputs[str(wp.relative_to(ROOT))] = digest(wp)
                wf = pd.read_parquet(wp, columns=['mid','ws','pnl']).sort_values('ws').reset_index(drop=True)
                if not wf[['mid','ws']].equals(window['v1'][['mid','ws']]): raise ValueError('multiplicity family windows differ')
                family_days[(tm,label)].append(float(wf.pnl.sum()-window['v1'].pnl.sum()))
        del by
        print('Reconciled', day, flush=True)
    matrix = np.array([family_days[key] for key in family]).T
    test_control = controls()
    t, raw, adjusted = signflip(matrix)
    result = {}
    for tm in ('J','X'):
        result[tm] = dict(totals={label:{k:sum(per[tm][d]['arms'][label]['components'][k] for d in DAYS) for k in TERMS} for label in ARMS},
                          differences={k:paired([per[tm][d]['delta'][k] for d in DAYS]) for k in TERMS},
                          late_differences={str(sec):{k:paired([per[tm][d]['arms']['CAP10']['late'][str(sec)][k]-per[tm][d]['arms']['v1']['late'][str(sec)][k] for d in DAYS])
                                                      for k in ('price_drift','settlement_residual','marked_pnl')} for sec in HORIZONS},
                          risk_matched_carry=paired([per[tm][d]['risk_matched_carry_control']['cap10_minus_scaled_v1_carry'] for d in DAYS]),
                          abs_inventory_lower_days=sum(per[tm][d]['arms']['CAP10']['abs_inventory_mean'] < per[tm][d]['arms']['v1']['abs_inventory_mean'] for d in DAYS),
                          risk_proxy_lower_days=sum(per[tm][d]['arms']['CAP10']['inventory_risk_proxy'] < per[tm][d]['arms']['v1']['inventory_risk_proxy'] for d in DAYS),
                          carry_sd_lower_days=sum(per[tm][d]['arms']['CAP10']['carry_sd'] < per[tm][d]['arms']['v1']['carry_sd'] for d in DAYS))
        j = family.index((tm,'CAP10'))
        result[tm]['signflip_settle'] = dict(t=float(t[j]), exact_p=float(raw[j]), max_abs_t_family_p=float(adjusted[j]), patterns=512)
    out = dict(as_of=datetime.now(timezone.utc).isoformat(), declaration=declaration, checks=dict(reconciliation_pass=True,
               corrupted_fill_refusals=refusals, signflip_controls=test_control), statuses=status, n_days=9, windows_per_arm_per_timing=2592,
               input_sha256=inputs, per_day=per, summary=result,
               family_signflip={f'{tm}/{label}':dict(t=float(t[j]), p=float(raw[j]), family_p=float(adjusted[j])) for j,(tm,label) in enumerate(family)},
               forward_data_read=False, validated_out_of_sample=False)
    write(DEST / 'report.json', out)
    print(json.dumps(dict(report=str(DEST/'report.json'), summary=result, statuses=status), indent=2))


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'declare': declare()
        elif sys.argv[1] == 'run': run()
        else: raise ValueError('expected declare or run')
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f'REFUSED: {exc}')
