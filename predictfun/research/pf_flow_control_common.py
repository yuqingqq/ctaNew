"""Frozen minimal controller experiment; historical research only."""
import json
import subprocess
import sys
from pathlib import Path
import pf_flow_price_common as C
np, pd, I = C.np, C.I.pd, C.I
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(I.K.IO) / 'flow_control_20261116_v2'
REVIEW = HERE.parent / 'reviews/flow_control_20261116'
FREEZE = REVIEW / 'freeze_v2.json'
TRAIN = ('20261023', '20261024', '20261025')
CAL = '20261026'
TEST = '20261027'
SIZE = 30.
CAP = 180.
RISK = 1. / (2. * CAP)  # cents/share^2: marginal penalty 1 cent/share at CAP
HORIZON = 1000.
BANK_PER_MARKET = 24
CELL_TEMPLATES = 3
IPF_ITERATIONS = 24
PILOT_MARKETS = 12  # chronological, one selected market per two-hour block
SOURCES = ['pf_flow_control_common.py','pf_flow_control_paths.py','pf_flow_control_kernel.cpp',
           'pf_flow_control_kernel.py','pf_flow_control_lane.py','pf_flow_control_policy.py',
           'pf_flow_control_run.py','pf_flow_control_checks.py','run_flow_controller.sh']


def freeze():
    C.verify()
    path=FREEZE
    C.require(not path.exists(), 'controller freeze exists')
    files=[HERE/n for n in SOURCES]
    files += [ROOT/p for p in C.verify()['sources']]
    files += [Path(m.__file__).resolve() for m in list(sys.modules.values())
              if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(HERE))]
    C.dump(path, dict(asof=C.stamp(), source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={str(p.relative_to(ROOT)):C.sha(p) for p in sorted(set(files))},
        inherited_prediction_freeze_sha256=C.sha(C.REVIEW/'freeze_v1.json'),
        supersedes=dict(freeze='freeze_v1.json',sha256=C.sha(REVIEW/'freeze_v1.json'),
            reason='Review found that the generic book-minimum queue estimator ignored acknowledged partial fills. Within queueM a known positive fill proves ahead is zero and later snapshots cannot increase it. Correct the controller-only state mapping; old smoke and interrupted pilot retained. No threshold, forecast, path selection or risk parameter changed.'),
        train=TRAIN,calibration=CAL,test=TEST,population='12 predeclared markets: first eligible in each two-hour block; continuous five minutes; consumed development only',
        paths='All eligible neutral bid/rung0/at-best probes; deterministic 24-per-market hash sampling before outcomes; 3 representative paths per phase/spread/bid-class/ask-class, weighted by cell counts. Paired both-side tape and book paths, no independent rung samples.',
        controller='Greedy simultaneous action set, keep/cancel and post/wait, common scenarios and endpoint; repeat coordinate improvement until no positive increment, no timer or minimum rest. No replacement or size search.',
        objective='Expected portfolio wealth change at t+1s minus lambda*(terminal net inventory squared - current squared), cents. No continuation value beyond endpoint. Same t+2s extension is a separately declared horizon sensitivity.',
        risk=dict(size=SIZE,net_worst_case_cap=CAP,lambda_cents_per_share2=RISK,interpretation='marginal inventory penalty 1 cent/share at cap; design choice, not fitted'),
        timing=dict(place_ms=58,cancel_ms=37,pending_cancel_ms=95,notice_ms=100,review_ms=50,trade_shift_ms=50),
        scenarios=dict(max_paths_per_joint_cell=CELL_TEMPLATES,ipf_iterations=IPF_ITERATIONS,
            alignment='Exact initial spread and phase; integer price translation; reject any out-of-payoff-range path. Preserve paired timing, quantities and subsequent book evolution.',
            weighting='Empirical cell weights vs iterative proportional fitting to two canonical fresh at-best JOINT forecasts. Unsupported mass and achieved marginal residual reported. No tail class fabricated.',
            unsupported='If no paths: keep existing and wait empty; structural engine safety remains active. Missing class mass is reported, not called calibrated.'),
        cases=['KEEP_ALL_CAP','EMPIRICAL','JOINT','JOINT_H2'],
        null=dict(draws=200,seed=20261116,method='Within market/side/action-kind/hour, permute forecast action-advantage against realized branch advantage on one hash-selected neutral generation probe; same positive-action count; descriptive only'),
        multiplicity=dict(controller_designs=1,forecast_arms=2,horizon_sensitivities=1,no_parameter_search=True),
        exclusions='No model/invalid current book/missing scenario support/horizon near expiry recorded as statuses. All actual fill markout exclusions retained.',
        limits=['One consumed test day; no confidence intervals, untouched validation or promotion',
            'Training paths transplanted into current books are an empirical approximation, not a validated event generator',
            'Both-side joint marginals may be incompatible; projection residual and unsupported mass measured',
            'Historical estimated trade timing and queueM assumptions remain; no live performance claim',
            'Known order remainder and inventory omit unreceived fills; action cap conservatively reserves outstanding orders',
            'Existing joint classifier is queried at canonical fresh best orders; this transport requires diagnostics',
            'Zero endpoint continuation; compare declared 2s horizon, do not claim queue continuation solved'],output=str(OUT)))


def verify():
    f=json.loads(FREEZE.read_text())
    for p,h in f['sources'].items():C.require(C.sha(ROOT/p)==h,'changed controller source '+p)
    return f
