"""Version-one prediction-only experiment. No order action or policy selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_stop_ev_minutes_inputs as I
import pf_stop_ev_model as V
import pf_stop_ev_accounting as AC

ROOT = HERE.parents[1]
OUT = Path(I.K.IO) / 'flow_price_20261116_v1'
REVIEW = HERE.parent / 'reviews/flow_price_20261116'
DAYS = tuple(V.DAYS9)
TESTS = tuple(V.TESTS)
FOLDS = V.FOLDS
AGES = (0., 250., 500., 1000., 2000.)
SIZE = 30.
SHIFT = 50.
SETTINGS = AC.Settings(cancel_ms=37., notification_ms=100., timing='all_anchors_sensitivity')
MARKET = list(V.F71)
ORDER = list(V.ORDER_IN)
EXTRA = ['fp_size', 'fp_remaining', 'fp_queue', 'fp_shown', 'fp_scale', 'fp_phase']
FEATURES = MARKET + ORDER + EXTRA
MECHANICAL = ORDER + EXTRA + ['fside', 'pf_mid', 'pf_spread', 'pf_age', 't_rel']
TARGETS = ['flow_quantity', 'own_quantity', 'value_cents', 'severe', 'own_fill']
COND = ['scenario_flow', 'scenario_move']
MOVE_EDGES = np.array([-2., -.5, .5, 2.])
TEMPERATURES = [.5, .75, 1., 1.5, 2., 3.]
NCLASS = 15
SOURCE_NAMES = ['pf_flow_price_common.py', 'pf_flow_price_data.py', 'pf_flow_price_model.py',
                'pf_flow_price_report.py', 'pf_flow_price_check.py']


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer, np.bool_)): return x.item()
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x


def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(x), indent=2, allow_nan=False) + '\n')


def classes(q, move):
    q, move = np.asarray(q), np.asarray(move)
    require(np.isfinite(q).all() and np.isfinite(move).all() and (q >= 0).all(), 'invalid targets')
    flow = np.where(q <= 1e-10, 0, np.where(q <= SIZE, 1, 2)).astype(int)
    price = np.searchsorted(MOVE_EDGES, move, side='right')
    return flow, price, 5 * flow + price


def verify():
    f = json.loads((REVIEW / 'freeze_v1.json').read_text())
    for p, h in f['sources'].items(): require(sha(ROOT / p) == h, 'source changed: ' + p)
    return f


def freeze():
    require(not (REVIEW / 'freeze_v1.json').exists(), 'freeze exists')
    paths = [HERE / n for n in SOURCE_NAMES]
    # Pin every imported local implementation, including inherited loaders/engine.
    paths += [Path(m.__file__).resolve() for m in list(sys.modules.values())
              if getattr(m, '__file__', None) and str(Path(m.__file__).resolve()).startswith(str(HERE))]
    paths = sorted(set(paths))
    require(all(p.exists() for p in paths), 'source missing')
    dump(REVIEW / 'freeze_v1.json', dict(asof=stamp(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources={str(p.relative_to(ROOT)): sha(p) for p in paths}, days=DAYS, folds=FOLDS,
        features=FEATURES, mechanical=MECHANICAL, parameters=V.PARAMS, rounds=200,
        targets=dict(flow='public shares at or through the focal price in [t+37ms,t+1s]',
            move='side-signed midpoint(t+1s) minus CLIENT-KNOWN midpoint at decision; cents',
            own='neutral order tranches in [t+37ms,t+1s], each marked at fill+1s',
            severe='own local value < -30 cents per original 30-share order',
            joint='3 flow classes (zero, <=30, >30) x 5 signed price-change classes',
            move_edges_cents=MOVE_EDGES, no_fill='retained; value zero, not omitted'),
        clocks=dict(place_ms=58, cancel_ms=37, notification_ms=100, review_ms=50, trade_shift_ms=50),
        neutral=dict(size=SIZE, queue='M', unknown='zero', nr=6, cap=None, ages_ms=AGES, stop_ms=299800),
        model='joint multiclass; conditional mean/binary response heads integrate over all 15 predicted scenarios',
        comparators=['empirical side/rung/phase joint', 'mechanical joint', 'independent full-feature marginals', 'direct full-feature value'],
        calibration=dict(temperature_grid=TEMPERATURES, criterion='calibration-day categorical logloss', value='no post-fit correction'),
        population='one deterministic probe per neutral generation; all statuses saved; full contract, all selected historical markets',
        null=dict(draws=200, seed=20261116, method='permute local values within day/market/side/rung; descriptive within-stratum ranking test'),
        acceptance='prediction diagnostics only; no strategy P&L or promotion. Compare joint logloss, quantity/value MSE, mean calibration and tail Brier to baselines by day/side/phase.',
        multiplicity=dict(learned_joint_candidates=1, independence_ablation=1, trained_mechanical_baseline=1, direct_value_baseline=1),
        limits=['consumed dates, no untouched validation', 'estimated/imputed trade clocks and queue assumptions',
                'coarse joint outcomes, not a full path generator', 'neutral resting orders; not yet calibrated entry/action advantage',
                'future mark coverage required uniformly even for zero-fill actions', 'public trade prints are labels, not timely input features'],
        output=str(OUT)))


def guard(where):
    I.M.trim()
    require(I.K.rss_mib() < 2048., 'RSS cap at ' + where)
