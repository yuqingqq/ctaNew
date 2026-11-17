"""Recompute published moment predictions from models and identity-bound inputs."""
import json
from pathlib import Path
import subprocess
import pf_joint_moment as J
import pf_joint_moment_fit as FIT

C, np, pd = J.C, J.np, J.pd


def equal(left, right, name):
    C.require(np.allclose(np.asarray(left,float), np.asarray(right,float),
                          rtol=0, atol=1e-9, equal_nan=True), name)


def controls():
    equal([1., np.nan], [1., np.nan], 'positive equality control')
    return dict(equal_positive=True,
                planted_difference_refused=FIT.CHECK.refused(
                    lambda: equal([1.], [2.], 'planted difference'), 'difference'))


def run():
    J.verify()
    result = dict(asof=C.stamp(), controls=controls(), days={},
                  source_sha256=C.sha(__file__),
                  source_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=J.F.ROOT, text=True).strip())
    fr = json.loads((J.OUT/'fit/receipt.json').read_text())
    train = []
    for day in J.F.TRAIN:
        a, receipt = FIT.read(day)
        candidates = pd.read_parquet(J.OUT/day/'candidates.parquet')
        all_actions = pd.read_parquet(J.OUT/day/'actions.parquet')
        C.require(C.sha(J.OUT/day/'candidates.parquet') == receipt['outputs']['candidates.parquet'], 'candidate hash')
        C.require(set(candidates.loc[candidates.selected, 'uid']) == set(all_actions.uid), 'selected action accounting')
        C.require(len(all_actions) == all_actions.uid.nunique(), 'duplicate selected identity')
        train.append(a)
    train = pd.concat(train, ignore_index=True)
    C.require(len(train) == fr['n'], 'training count')
    equal(train[J.FEATURES].min(), [fr['feature_min'][f] for f in J.FEATURES], 'feature minimum')
    equal(train[J.FEATURES].max(), [fr['feature_max'][f] for f in J.FEATURES], 'feature maximum')
    result['training'] = dict(n=len(train), selected_identity_accounting=True,
                             severe=int(train.y_severe.sum()), zero_fill=int((train.y_q<=1e-9).sum()))
    import lightgbm as lgb
    for day in (J.F.CAL, J.F.TEST):
        out = J.OUT/('eval_'+day)
        r = json.loads((out/'receipt.json').read_text())
        for name, digest in r['outputs'].items():
            C.require(C.sha(out/name) == digest, 'evaluation artifact hash')
        rep = json.loads((out/'report.json').read_text())
        a = pd.read_parquet(out/'heads.parquet').set_index('uid')
        actions, _ = FIT.read(day)
        equal(a[J.FEATURES], actions.set_index('uid').loc[a.index, J.FEATURES], 'evaluation feature identity')
        for name, artifact in fr['artifacts'].items():
            C.require(C.sha(artifact['path']) == artifact['sha256'], 'model artifact hash')
            model = lgb.Booster(model_file=artifact['path'])
            pred = model.predict(a[artifact['features']].to_numpy(np.float32), num_threads=1)*artifact['scale']
            equal(pred, a['raw_'+name], 'saved model output '+name)
        for variant, raw in [('full','full'), ('mechanical','mechanical')]:
            q,m = J.bounded(a['raw_'+raw+'_q'].to_numpy(), a['raw_'+raw+'_m'].to_numpy(),
                            a.known_remaining.to_numpy(), a.mid.to_numpy(), a.side.to_numpy())
            equal(q, a['q_'+variant], 'quantity bounds'); equal(m, a['m_'+variant], 'motion bounds')
        sign = 1.-2*a.side.to_numpy(); orient = np.where(a.kind=='post',1.,-1.)
        edge = sign*(a.mid-a.price).to_numpy()
        base = pd.read_parquet(J.D.OUT/day/'values.parquet')
        base = base[(base.case=='SOFT_JOINT')&(base.channel=='ordinary')].set_index('uid').loc[a.index]
        old = (base.pred_focal_edge+base.pred_focal_motion).to_numpy()
        rest = base.predicted.to_numpy()-old
        low = np.minimum(-sign*a.price, sign*(100.-a.price))*a.known_remaining
        high = np.maximum(-sign*a.price, sign*(100.-a.price))*a.known_remaining
        direct = np.clip(a.raw_direct_w, low, high)
        fraction = np.where(a.kind=='cancel', np.clip(a.raw_remaining,0.,1.), 1.)
        expected = dict(SOFT_JOINT=base.predicted.to_numpy(),
                        JOINT_MOMENT=rest+orient*(edge*a.q_full+a.m_full),
                        MECHANICAL_MOMENT=rest+orient*(edge*a.q_mechanical+a.m_mechanical),
                        DIRECT_WEALTH=rest+orient*direct,
                        REMAINDER_ONLY=rest+old*fraction)
        values = pd.read_parquet(out/'values.parquet')
        for case, prediction in expected.items():
            z = values[values.case==case].set_index('uid').loc[a.index]
            equal(z.predicted, prediction, 'economic reconstruction '+case)
            equal(z.actual, a.y_value, 'target identity '+case)
        for kind, z in a.groupby('kind'):
            for variant in ('full','mechanical'):
                expected_moments = dict(n=len(z), quantity_mae_shares=float((z['q_'+variant]-z.y_q).abs().mean()),
                    motion_rmse_cents=float(np.sqrt(np.mean((z['m_'+variant]-z.y_m)**2))))
                C.require(expected_moments == rep['moment_scores'][variant+':'+kind], 'moment metrics')
        z=a[a.kind=='cancel'];error=z.remaining_fraction-z.y_remaining;naive=1.-z.y_remaining;hidden=z.y_remaining<1.-1e-9
        remaining = dict(n=len(z), unreceived_n=int(hidden.sum()),
            model_rmse_fraction=float(np.sqrt(np.mean(error**2))), known_upper_bound_rmse_fraction=float(np.sqrt(np.mean(naive**2))),
            model_mae_fraction=float(error.abs().mean()), model_hidden_rmse=float(np.sqrt(np.mean(error[hidden]**2))) if hidden.any() else None,
            naive_hidden_rmse=float(np.sqrt(np.mean(naive[hidden]**2))) if hidden.any() else None)
        C.require(remaining == rep['remainder'], 'remaining metrics')
        coverage={}
        for name in J.FEATURES:
            x=a[name].to_numpy();lo=fr['feature_min'][name];hi=fr['feature_max'][name]
            coverage[name]=dict(missing=int((~np.isfinite(x)).sum()),below=int((x<lo).sum()) if lo is not None else None,
                                above=int((x>hi).sum()) if hi is not None else None)
        C.require(coverage==rep['coverage'], 'coverage metrics')
        result['days'][day]=dict(n=len(a), saved_models_reproduced=True, economic_values_reconstructed=True,
                                 moment_remaining_coverage_metrics_recomputed=True, report_sha256=C.sha(out/'report.json'))
    dest=J.REVIEW/'reconstruction_v1.json';C.require(not dest.exists(), 'reconstruction exists')
    C.dump(dest,result)
    print(json.dumps(C.clean(result)),flush=True)


if __name__=='__main__':
    run()
