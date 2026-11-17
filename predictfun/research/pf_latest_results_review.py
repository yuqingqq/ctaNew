"""Reproduce Nov17 severe-cal/info/queue results, without changing any candidate.

Review only, consumed Oct23-27. Refit the identical information-test recipe in
a separate directory because its original per-row scores/models were not saved.
No new candidate, threshold, policy, intervals, or promotion. Run via run_light.
"""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import pf_severe_cal as SC
import pf_severe_info as SI
import pf_queue_place_test as QP

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/predict_fun/research/latest_results_review_20261117'
RN, GL, C = SC.RN, SC.GL, SC.C
HASHES = {}


def track(path):
    path = Path(path)
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(track(path).read_text())


def close(a, b, tol=1e-10):
    if not np.allclose(a, b, atol=tol, rtol=0, equal_nan=False):
        raise ValueError(f'reproduction mismatch: {a} != {b}')


def selftest():
    close([1., 2.], [1., 2.])
    try:
        close([1., 2.], [1., 3.])
    except ValueError:
        pass
    else:
        raise ValueError('planted changed result accepted')
    SC.selftest(); SI.selftest(); QP.selftest()


def calibration(fr):
    reference = read_json(SC.OUT / 'severe_cal.json')
    track(SC.OUT / 'severe_cal.txt')
    feats = GL.feature_list('full'); folds = {}
    for fold, (tdays, cal, evals) in SC.FOLDS.items():
        assert max(tdays) < cal < min(evals)
        names = ['g_p', 'g_amt', 'l1_amt', 'l2_amt', 'BASE_A', 'BASE_B', 'SMOOTH_A', 'SMOOTH_B']
        heads = {n: lgb.Booster(model_file=str(track(SC.OUT / f'fold_{fold}' / f'{n}.txt'))) for n in names}
        xcal = fr[cal][feats].to_numpy(np.float32)
        loss = (fr[cal].loss > 1e-9).to_numpy()
        maps = {'A': SC.fit_map(heads['BASE_A'].predict(xcal), loss),
                'B': SC.fit_map(heads['BASE_B'].predict(xcal)[loss], fr[cal].sev.to_numpy()[loss])}
        assert all(b > 0 for a, b in maps.values())
        cells = {}; scores = {a: [] for a in ('BASE', 'SMOOTH', 'CAL')}
        for day in evals:
            z = fr[day]; x = z[feats].to_numpy(np.float32)
            gp = heads['g_p'].predict(x)
            gain = gp * np.maximum(heads['g_amt'].predict(x), 0) * GL.SCALE
            a1 = np.clip(heads['l1_amt'].predict(x) * GL.SCALE, 0, GL.SEVERE)
            a2 = np.maximum(heads['l2_amt'].predict(x), 0) * GL.SCALE
            sign = 1 - 2 * z.side.to_numpy(); price = z.price.to_numpy(); rem = z.known_remaining.to_numpy()
            low = np.minimum(-sign * price, sign * (100 - price)) * rem
            high = np.maximum(-sign * price, sign * (100 - price)) * rem
            for arm in scores:
                base = 'BASE' if arm == 'CAL' else arm
                pa = heads[f'{base}_A'].predict(x); pb = heads[f'{base}_B'].predict(x)
                if arm == 'CAL':
                    pa = SC.apply_map(pa, maps['A']); pb = SC.apply_map(pb, maps['B'])
                ps = pa * pb; pred = np.clip(gain - pa * a1 - ps * a2, low, high)
                assert np.all(ps <= pa + 1e-12)
                scores[arm].append(pd.DataFrame(dict(p=ps, y=z.sev.to_numpy())))
                for kind in ('cancel', 'post'):
                    mask = (z.kind == kind).to_numpy(); g = z[mask].copy()
                    p = ps[mask]; y = g.sev.to_numpy(); w = pred[mask]
                    g['pred'] = w
                    top = {}
                    for rate in GL.RATES:
                        selected = [s.sort_values(['pred', 'hash'], ascending=[False, True]).head(math.ceil(rate * len(s)))
                                    for _, s in g.groupby(['m', 'side'])]
                        top[str(rate)] = float(pd.concat(selected).w.mean())
                    cells[f'{day}/{kind}/{arm}'] = dict(n=len(g), events=int(y.sum()),
                        ll=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))), oe=float(y.sum()/p.sum()),
                        rmse=float(np.sqrt(np.mean((w-g.w.to_numpy())**2))), ce=GL.calibration_error(w, g.w),
                        top=top, gain_plus_loss_probability_gt_one=int((gp[mask]+pa[mask]>1+1e-12).sum()))
        lowbands = {}
        for arm, parts in scores.items():
            z = pd.concat(parts, ignore_index=True); z = z[z.p < .05]
            lowbands[arm] = dict(n=len(z), events=int(z.y.sum()), expected=float(z.p.sum()), oe=float(z.y.sum()/z.p.sum()))
        readings = {}
        for arm in ('SMOOTH', 'CAL'):
            pairs = [(cells[f'{d}/{k}/{arm}'], cells[f'{d}/{k}/BASE']) for d in evals for k in ('cancel', 'post')]
            better = {metric: sum(a[metric] < b[metric] for a, b in pairs) for metric in ('ll', 'rmse', 'ce')}
            better['oe'] = sum(abs(np.log(a['oe'])) < abs(np.log(b['oe'])) for a, b in pairs)
            r = dict(S1=better['ll']>=3 and better['oe']>=3,
                S2=abs(np.log(lowbands[arm]['oe']))<abs(np.log(lowbands['BASE']['oe'])),
                S3=better['rmse']>=3 and better['ce']>=3,
                S4=sum(sum(a['top'][str(q)]>=b['top'][str(q)] for a,b in pairs)>=3 for q in GL.RATES)>=2)
            r['PASS'] = r['S1'] and r['S3'] and r['S4']
            if fold == 'A':
                for key in r: assert r[key] == reference['readings'][arm][key], (arm, key)
                close(lowbands[arm]['oe'], reference['readings'][arm]['low_oe'])
                close(lowbands['BASE']['oe'], reference['readings'][arm]['base_low_oe'])
            readings[arm] = dict(predicates=r if fold=='A' else None, improved_cells=better)
        folds[fold] = dict(train_days=tdays, calibration_day=cal, evaluation_days=evals, maps=maps,
            n=sum(len(fr[d]) for d in evals), cells=cells, low_bands=lowbands, readings=readings)
    return folds


def information(fr):
    reference = read_json(SI.OUT / 'severe_info.json'); track(SI.OUT / 'severe_info.txt')
    parts = []
    for day in RN.PRED_DAYS:
        z = fr[day].copy()
        m = lgb.Booster(model_file=str(track(RN.OUT / f'fold_{day}' / 'full_l2_p.txt')))
        z['p'] = m.predict(z[GL.feature_list('full')].to_numpy(np.float32))
        parts.append(z[z.p < .05])
    band = pd.concat(parts, ignore_index=True)
    assert len(band)==30080 and int(band.sev.sum())==675
    sets = {}
    for tag, days in [('all_four_days', RN.PRED_DAYS), ('three_model_comparison_days', RN.PRED_DAYS[1:])]:
        z = band[band.day.isin(days)]
        sets[tag] = dict(days=days, n=len(z), events=int(z.sev.sum()), auc=SI.auc(z.sev,z.p))
    close(sets['three_model_comparison_days']['auc'], reference['pooled_auc']['P'])
    # Exactly reproduce the original six fits, not a new model candidate. Original outputs stay untouched.
    SI.OUT = OUT / 'info_exact_recipe_reproduction'
    log = io.StringIO()
    with contextlib.redirect_stdout(log): SI.run()
    (OUT / 'info_reproduction.log').write_text(log.getvalue())
    replay = read_json(SI.OUT / 'severe_info.json')
    for arm in reference['pooled_auc']: close(replay['pooled_auc'][arm], reference['pooled_auc'][arm])
    assert replay['predicates']==reference['predicates']
    return dict(populations=sets, pooled_auc=replay['pooled_auc'], predicates=replay['predicates'], exact_recipe_refits=6)


def queue():
    frames=[]; receipts=[]
    for day in QP.DAYS:
        r=read_json(QP.OUT/day/'receipt.json'); path=track(QP.OUT/day/'rows.parquet')
        assert HASHES[str(path.relative_to(ROOT))]==r['outputs']['rows.parquet']
        frames.append(pd.read_parquet(path)); receipts.append(r)
    z=pd.concat(frames,ignore_index=True); ref=read_json(QP.OUT/'queue_place.json'); track(QP.OUT/'queue_place.txt')
    eligible=z[z.v.notna()].copy()
    assert np.isfinite(eligible[['v','q','vy','qy']]).all().all()
    assert eligible.q.between(-1e-9,30+1e-9).all()
    assert (eligible.qy<=eligible.q+1e-9).all() and (eligible.qy>=-1e-9).all()
    close(eligible.loc[eligible.q==0,'v'], np.zeros((eligible.q==0).sum()))
    close(eligible.vy*eligible.q, eligible.v*eligible.qy, tol=1e-7)
    def summ(g):
        return dict(n=len(g), mean_c=float(g.v.mean()), mean_without_young_c=float((g.v-g.vy).mean()),
            filled=int((g.q>1e-9).sum()), shares=float(g.q.sum()), per_share_c=float(g.v.sum()/g.q.sum()))
    tables={f'{k}/{a}':summ(g) for (k,a),g in eligible.groupby(['kind','arm'])}
    paired={}; assertions={}
    for name,a,b,event in [('F1 FIRST-BEATS-LATE','FIRST','LATE',True),('E1 EARLY-BEATS-FIRST (ex post)','EARLY','FIRST',True),
                          ('R1 REST1-BEATS-JOIN','REST1','JOIN',False),('R2 IMPROVE-BEATS-JOIN','IMPROVE','JOIN',False)]:
        g=eligible[(eligible.kind!='state') if event else (eligible.kind=='state')].copy()
        keys=['day','m','side','b' if event else 't']; g['v_old']=g.v-g.vy
        x=g[g.arm==a]; y=g[g.arm==b]
        assert not x.duplicated(keys).any() and not y.duplicated(keys).any()
        pair=x.merge(y,on=keys,suffixes=('_a','_b'),validate='one_to_one')
        val={}
        for col in ('v','v_old'):
            diff=pair[f'{col}_a']-pair[f'{col}_b']; daily=diff.groupby(pair.day).mean()
            passed=bool(diff.mean()>0 and (daily>0).sum()>=4)
            assert passed==ref['readings'][f'{name} [{col}]']
            val[col]=dict(n=len(pair), mean=float(diff.mean()), by_day=daily.to_dict(), passes=passed)
        paired[name]=val
    for name,arm in [('F2 FIRST-PROFITABLE','FIRST'),('E2 EARLY-PROFITABLE (ex post)','EARLY'),
                     ('R3 JOIN-PROFITABLE','JOIN'),('R3 REST1-PROFITABLE','REST1'),('R3 IMPROVE-PROFITABLE','IMPROVE')]:
        g=eligible[eligible.arm==arm]
        for col in ('v','v_old'):
            v=g.v if col=='v' else g.v-g.vy
            passed=bool(v.mean()>0 and (v.groupby(g.day).mean()>0).sum()>=4)
            assert passed==ref['readings'][f'{name} [{col}]']
            assertions[f'{name} [{col}]']=passed
    states=z[z.kind=='state']; keys=['day','m','side','t']
    return dict(rows=len(z), eligible_rows=len(eligible), tables=tables, paired=paired, profitability=assertions,
        recorded_statuses=z.groupby(['kind','arm','status']).size().to_dict(),
        event_short_statuses=z[z.kind!='state'].groupby('status_2s').size().to_dict(),
        causal_declared_states=sum(r['stat']['states'] for r in receipts),
        causal_saved_unique_states=len(states[keys].drop_duplicates()),
        causal_short_missing_rows=int(states.v_2s.isna().sum()),
        path_controls={k:sum(r['stat'][k] for r in receipts) for k in ('parity_checked','parity_bad','mut_checked','mut_differs')})


def main():
    OUT.mkdir(parents=True,exist_ok=False); selftest(); fr=RN.load()
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in ('pf_latest_results_review.py','pf_severe_cal.py','pf_severe_info.py','pf_queue_place_test.py',
                 'pf_flow_control_kernel.py','pf_flow_control_kernel.cpp','pf_gain_loss.py','pf_gain_loss_running.py'):
        track(ROOT/'predictfun/research'/name)
    for day in RN.DAYS:
        path=(GL.J.OUT/day/'actions.parquet') if day in RN.DAYS[:3] else (GL.OUT/f'label_{day}'/'actions.parquet')
        track(path)
    result=dict(asof=C.stamp(),review_source_commit=source_commit,calibration=calibration(fr),information=information(fr),queue=queue(),
        controls=dict(valid_result_accepted=True,changed_result_refused=True,original_selftests_passed=True),hashes=HASHES,
        scope='Consumed Oct23-27; candidate reproduction only, no new policy or validation claim. Queue tape not re-simulated.')
    # C.dump supports the existing tuple-key status maps.
    C.dump(OUT/'receipt_v1.json',result)
    print(json.dumps(dict(asof=result['asof'],output=str(OUT/'receipt_v1.json'),calibration='reproduced',
        information='reproduced',queue='saved rows and predicates reproduced')),flush=True)


if __name__=='__main__': main()
