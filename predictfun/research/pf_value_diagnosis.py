"""Frozen calibration/history and current-controller diagnostics; research only."""
import hashlib
import json
from pathlib import Path
import subprocess
import pf_joint_moment as J
import pf_joint_moment_fit as FIT
import pf_flow_churn_diagnostic as CH
F,C,np,pd,I,V=J.F,J.C,J.np,J.pd,J.I,J.V
OUT=Path(I.K.IO)/'value_diagnosis_20261117_v1'
REVIEW=F.HERE.parent/'reviews/value_diagnosis_20261117'
FREEZE=REVIEW/'freeze_v1.json'
SOURCES=('pf_value_diagnosis.py','pf_value_diagnosis_calibration.py',
         'pf_value_diagnosis_controller.py','pf_value_diagnosis_audit.py','run_value_diagnosis.sh')
HISTORY={'H1':('20261025',),'H2':('20261024','20261025'),
         'H3':F.TRAIN,'H3_MATCHED_N1':F.TRAIN}
GEOMETRY=['pf_mid','pf_spread','sigma','adv_bn_r100','adv_bn_r1000','adv_bn_imb',
          'adv_bn_flow250','adv_pf_micro_dev','dfv250','jm_depth','jm_remaining',
          'jm_ahead','jm_scale','jm_seconds_left','jm_shown','jm_cum_displayed_ahead']
VALUE_BINS=[-np.inf,-30.,-5.,-1.,0.,1.,5.,30.,np.inf]
PROB_BINS=[0.,.01,.03,.1,.3,.6,1.00000001]


def identity_hash(uid):
    return hashlib.sha256((str(uid)+':value-diagnosis-v1').encode()).hexdigest()


def check_equal(a,b,name):
    C.require(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=0,atol=1e-8,equal_nan=True),name)


def files(directory):
    return {p.name:C.sha(p) for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'}


def checked_receipt(directory):
    r=json.loads((directory/'receipt.json').read_text())
    for name,digest in r['outputs'].items():C.require(C.sha(directory/name)==digest,'artifact changed '+str(directory/name))
    return r


def freeze():
    J.verify();C.require(not FREEZE.exists(),'diagnostic freeze exists')
    inherited={str(Path(CH.__file__).relative_to(F.ROOT)):C.sha(CH.__file__)}
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        sources={str((F.HERE/n).relative_to(F.ROOT)):C.sha(F.HERE/n) for n in SOURCES},inherited_sources=inherited,
        joint_moment_freeze_sha256=C.sha(J.FREEZE),history=HISTORY,features=J.FEATURES,mechanical=J.MECHANICAL,
        geometry=GEOMETRY,parameters=C.V.PARAMS,rounds=200,
        population='Reuse all exact eligible neutral training rows Oct23-25 and861 archived Oct26/27 actions. Same targets/features/clocks/size. Original statuses remain attached. No outcome sampling. Controller study uses own-policy SOFT_JOINT entries, kept separate from neutral training.',
        history_design='H1 Oct25, H2 Oct24-25, H3 Oct23-25. H3_MATCHED_N1 hash-samples H3 without outcomes to match H1 counts in kind/side/phase. H3 original seven heads reused by hashes. No history winner selected or policy rollout.',
        additional_fill_heads='Full/mechanical binary affected-fill heads fitted on ALL eligible actions for each history, diagnostic only. They do not change value predictions. Q/p is checked for physical incoherence, never silently clipped.',
        calibration='Fixed action-value bins -inf,-30,-5,-1,0,1,5,30,inf cents and probability bins0,.01,.03,.1,.3,.6,1. No fitted recalibration or policy threshold. Report daily kind/side, depth<=0/(0,2]/>2, phase, no-fill, severe partitions. Training summaries explicitly in-sample. Group counts/means, no row-level standard errors.',
        coverage='Fixed16 dimensions; full-three-day training median/IQR with fallback1; asinh robust scaling plus missing indicators. Nearest same-kind/side training action; reference95th percentile from min-hash500 training queries per kind/side, excluding same market. No target in geometry. Also kind/side/phase/depth/mid cells and training severe counts. All diagnostic, not admission filters.',
        controller='Reproduce exact SOFT_JOINT Oct27 pilot on12 predeclared markets; preserve every post status. Up to8 identity-hash pairs per market/side/first-review-cancel status. Same existing16-cell Shapley with SOFT context; endpoint sensitivities, pending-state truth diagnostic, and current bank versus complement-transformed query. No new controller.',
        symmetry='Check exact kernel value invariance after transforming BOTH paths and state by side swap and p->100-p, with fixed per-path weights. Separately measure original-bank predictions on complemented queries; this is representation sensitivity, not proof of profitable mirroring or symmetric market law.',
        null=dict(draws=200,seed=20261116,economic='Permute selected indicators within market/side/kind, preserving positive-action counts; inherited matched_null.',controller='Cancel/keep label permutation within market/side on forecast L1 changes; descriptive association, not decision value or causal proof.'),
        multiplicity=dict(history_windows=4,new_history_refits=3,diagnostic_fill_heads=8,new_controller_candidates=0,new_policy_thresholds=0,diagnostic_designs=1),
        limits=['Consumed Oct23-27 development only; no confidence intervals, protected dates or promotion',
            'History test spans at most3 days and mixes extra coverage with temporal regime changes; does not resolve longer-history sufficiency',
            'Single realized outcomes do not reveal conditional means; pooled calibration also has noise and conditional-mixing limits',
            'Controller hybrid-state sensitivities are not realized economic losses; global RMSE is not a calibrated decision margin',
            'Symmetry sensitivity cannot establish bid conditioning as the cause of ask losses',
            'No new strategy P&L, cooldown, minimum age, risk coefficient or continuation model'],output=str(OUT)))


def verify():
    J.verify();r=json.loads(FREEZE.read_text());C.require(r['joint_moment_freeze_sha256']==C.sha(J.FREEZE),'parent freeze changed')
    for p,h in {**r['sources'],**r['inherited_sources']}.items():C.require(C.sha(F.ROOT/p)==h,'diagnostic source changed '+p)
    return r
