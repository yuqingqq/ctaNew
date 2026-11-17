"""Verify the severe-probability review from saved heads; no candidate fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pf_gain_loss_running as R
import pf_severe_support as S

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT/'predictfun/reviews/severe_candidate_review_20261117/receipt_v1.json'


def verify():
    C=R.C; C.require(RECEIPT.exists(), 'missing receipt')
    saved=json.loads(RECEIPT.read_text()); R.selftest(); R.J.verify()
    # The subset relation is compulsory for probabilities, regardless of model fit.
    def coherent(severe, loss): C.require(np.all(np.asarray(severe)<=np.asarray(loss)+1e-9), 'incoherent probabilities')
    coherent([.1,.2],[.2,.2])
    try: coherent([.3],[.2])
    except ValueError: pass
    else: raise ValueError('incoherent control accepted')
    frames=R.load();feats=R.GL.feature_list('full');parts=[]
    for day in R.PRED_DAYS:
        z=frames[day].copy()
        for head in ('l1_p','l2_p'):
            path=R.OUT/('fold_'+day)/('full_'+head+'.txt')
            C.require(C.sha(path)==saved['model_hashes'][str(path)], 'model changed')
            model=lgb.Booster(model_file=str(path))
            z[head]=model.predict(z[feats].to_numpy(np.float32),num_threads=1)
            C.require(np.isfinite(z[head]).all() and z[head].between(0,1).all(),'invalid probabilities')
        entry=saved['folds'][day]
        C.require(max(entry['train_days'])<day,'nonforward fold')
        C.require(len(z)==entry['n'] and int(z.sev.sum())==entry['events'],'population changed')
        C.require(abs(z.l2_p.sum()-entry['expected'])<1e-9,'expected event count changed')
        C.require(int((z.l2_p>z.l1_p+1e-9).sum())==entry['severe_gt_any_loss_n'],'coherence count changed')
        parts.append(z)
    t=pd.concat(parts,ignore_index=True);bins=pd.cut(t.l2_p,S.BANDS,right=False)
    original=json.loads((S.OUT/'severe_support.json').read_text())
    for band in bins.cat.categories:
        z=t[bins==band];entry=saved['bands'][str(band)]
        C.require(len(z)==entry['n'],'band population changed')
        C.require(np.allclose([z.l2_p.sum(),z.sev.sum()],original['bands'][str(band)],atol=1e-9,rtol=0),'original band mismatch')
        C.require(np.allclose([z.l2_p.sum(),z.sev.sum()],[entry['expected'],entry['observed']],atol=1e-9,rtol=0),'review band mismatch')
    print(json.dumps(dict(asof=C.stamp(),n=len(t),events=int(t.sev.sum()),
        coherence_violations=int((t.l2_p>t.l1_p+1e-9).sum()),
        controls=dict(valid_probabilities_accepted=True,incoherent_probabilities_refused=True),
        all_bands_and_model_hashes_verified=True)),flush=True)


if __name__=='__main__':verify()
