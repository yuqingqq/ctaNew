"""Predeclared all-order reservation experiment on consumed historical data."""
import datetime as dt
import subprocess
from .pf_sizing_study import ROOT, HERE, BASE, DAYS, sha, write
from .pf_entry_study import DATA as INPUT, DEST as PRIOR, ARMS as OLD_ARMS

DEST = ROOT/'data/predict_fun/research/inflight_budget_20261109'
DATA = DEST/'research'
ARMS = {'baseline': BASE, 'reserved10': OLD_ARMS['reserved10'],
        'reduce_reserved': BASE.replace('+CAP10', '+CAP10+IRED'),
        'both_reserved': BASE.replace('+CAP10', '+CAP10+IBOTH')}
PAIRS = [('reduce_reserved', 'baseline'), ('both_reserved', 'reserved10'), ('both_reserved', 'reduce_reserved')]


def context():
    from . import pf_io
    pf_io.OUT = str(DATA)


def setup():
    if DEST.exists():
        raise ValueError('refuse existing experiment')
    (DEST/'logs').mkdir(parents=True); (DATA/'step4').mkdir(parents=True)
    files = []
    for day in DAYS:
        for stage in ['step1', 'step5']:
            p = DATA/stage/day; p.parent.mkdir(exist_ok=True); p.symlink_to(INPUT/stage/day, target_is_directory=True)
        files += [DATA/'step5'/day/'preds_basis_study.parquet']
        files += [DATA/'step1'/day/f for f in ['registry.parquet', 'clock_rl_d2.json', 'legs_timed2.parquet', 'legs_tk.parquet']]
        files += sorted((DATA/'step1'/day/'books').glob('*.parquet'))
    for f in ['minute_lines_t60k.json', 'inv_r60.json']:
        (DATA/'step4'/f).symlink_to(INPUT/'step4'/f); files.append(DATA/'step4'/f)
    d = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), days=DAYS, arms=ARMS, pairs=PAIRS,
        population='all 288 resolved markets per day, 2592 per arm/timing; identical flat starting state',
        selection='all nine consumed dates, fixed guards trained on eight; exploratory historical comparison',
        reduce_reserved='original adding policy; reducing desired/submitted quantities capped to max(0,abs(known net)-all same-side remaining reservations)',
        both_reserved='same reducing policy plus adding cap max(0,10-signed known net-all same-side remaining reservations)',
        reservations='resting, pending creates, pending cancels; size minus reported fills; deduplicate order identity; no opposite-side credit',
        cancellation='new arms reserve through simulated cancellation confirmation plus 100ms fill-report delay plus .001ms event-order cushion; no credit from unreported actual fills',
        retention='new arms keep eligible same-price reducing orders if known remainder fits desired quantity and aggregate budget; both_reserved also does so for adding orders',
        allocation='existing reservations first; new orders in lane order 0,1,2,3; recheck on send; floor .01 and enforce existing >=1 share/$1 rules',
        unchanged='all price guards, RO60/S240, original size maximum, execution/latency/queue models, no taker or urgency',
        limits=['reduce_reserved retains soft adding-side CAP10 and cannot promise a global 10-share bound',
                'both_reserved aims for actual absolute inventory <=10 in the simulated lifecycle; checked on every fill',
                'changes include partial-order retention and conservative post-cancel reservation grace; not isolated to a single arithmetic term',
                'realized turnover not matched; report per-share markout and volume; no pure execution-quality claim',
                'no own market impact or full live OMS; no independent validation or deployment claim'],
        reference_reuse='fresh baseline and reserved10 on first full day J/X plus eight-market smoke; other days copied by hash from prior experiment',
        metrics=['settlement', '+1s', 'time-weighted squared inventory', 'full-path peaks/time>10/zero-crossings',
                 'daily SD/worst day/drawdown', 'FIFO matched and residual PnL', 'shares/requests/remaining inventory'],
        attribution='FIFO matching includes price drift, not pure spread; inventory-square is an exposure proxy',
        null=dict(draws=2000, bootstrap_draws=2000, seed=20261108,
            design='shared paired UTC-day sign flips, sign-invariant RMS max-T, paired day bootstrap',
            family='three declared pairs x J/X x settlement/+1s/inventory-square = 18'),
        prior_selection='two new variants chosen after entry-budget results; prior entry study had two candidates / 18 comparisons; no threshold sweep',
        timings=['J','X'], latency_ms=dict(place=40, cancel=22, pending_cancel=95, fill_known=100),
        loss_budget=None, exposure_backstop_shares=150, rebates='excluded',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        prepared_input_sha={str(p):sha(p) for p in files}, prior_receipt_sha=sha(PRIOR/'report.json'),
        forward_data_read=False, live_settings_changed=False)
    write(DEST/'declaration.json', d)
    print('DECLARED', DAYS, flush=True)


if __name__ == '__main__':
    setup()
