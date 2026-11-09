"""Two fixed closing-liquidity hypotheses; no model or price-threshold search."""
import datetime as dt
import subprocess
from .pf_sizing_study import ROOT,HERE,BASE,DAYS,sha,write
from .pf_late_study import DATA as INPUT,DEST as PRIOR,ARMS as OLD_ARMS

DEST=ROOT/'data/predict_fun/research/exit_liquidity_20261109'
DATA=DEST/'research'
REFERENCES=['baseline','late_reserved']
ARMS={k:OLD_ARMS[k] for k in REFERENCES}
ARMS.update(exit_keep=BASE.replace('+CAP10','+CAP10+EKEEP'),exit_close25=BASE.replace('+CAP10','+CAP10+ECLOSE25'))
PAIRS=[(v,b) for v in ['exit_keep','exit_close25'] for b in REFERENCES]+[('exit_close25','exit_keep')]
LIVE_REVIEW=ROOT.parent/'pmmm-data/review-exports/live-samewindow-20261108/live_vs_replay'


def context():
    from . import pf_io
    pf_io.OUT=str(DATA)


def setup():
    if DEST.exists():raise ValueError('refuse existing experiment')
    (DEST/'logs').mkdir(parents=True);(DATA/'step4').mkdir(parents=True)
    files=[]
    for day in DAYS:
        for stage in ['step1','step5']:
            p=DATA/stage/day;p.parent.mkdir(exist_ok=True);p.symlink_to(INPUT/stage/day,target_is_directory=True)
        files += [DATA/'step5'/day/'preds_basis_study.parquet']
        files += [DATA/'step1'/day/f for f in ['registry.parquet','clock_rl_d2.json','legs_timed2.parquet','legs_tk.parquet']]
        files += sorted((DATA/'step1'/day/'books').glob('*.parquet'))
    for f in ['minute_lines_t60k.json','inv_r60.json']:
        (DATA/'step4'/f).symlink_to(INPUT/'step4'/f);files.append(DATA/'step4'/f)
    sources=sorted(HERE.glob('pf_exit*.py'))+[HERE/'run_exit_liquidity.sh',HERE/'tests/test_exit_liquidity.py']
    evidence=[LIVE_REVIEW/f for f in ['markout_by_age.json','compare_summary.json','lifetime_summary.json','feed_timing.json']]
    (DEST/'evidence').mkdir()
    for p in evidence:
        (DEST/'evidence'/p.name).write_bytes(p.read_bytes())
    declaration=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,arms=ARMS,pairs=PAIRS,
        population='all 288 resolved markets/day;2592 per arm/timing, no outcome-selected subgroup',
        selection='nine already-consumed dates, guards trained on eight; conditional engineering experiment, no independent validation',
        hypothesis='prediction-dependent 5/25 adding size need not govern an otherwise safe inventory-reducing order',
        exit_keep='from60 retain same-price guard-eligible existing closing remainder whenever it fits aggregate closing room, even above current AM target; new sizes unchanged',
        exit_close25='same retention; new closing quantity=min(25,remaining aggregate closing room), ignoring AM low5 sizing; never top up a safe existing order',
        unchanged='all pre60 decisions/fills; all quote prices/prediction guards; original RO60/S240; minimums applied to actual new target size; cancelling adverse/ineligible orders; no quote-price concession',
        ordering='safe retention or closing-size target is applied before minimum-notional evaluation so an obsolete AM low5 target cannot veto a legal larger closing order; existing partial orders retain original size identity',
        reservations='same late_reserved aggregate reservation, pending create/resting/cancel; no opposite-side credit; cancel acknowledgement+100ms fill-report grace+.001ms',
        admission='same late_reserved submission recheck: never create inventory-increasing orders after60; no new orders from240',
        reference_reuse='fresh first full day J/X and smoke baseline/late_reserved exact; other references hash-verified',
        metrics=['settlement','+1s','mark-at240','carry','FIFO matched/residual','residual and integrated inventory',
                 'reducing/opening flow','orders/cancels','new-order reverse opening','retention/size engagement',
                 'young/cancel-pending fill contribution (descriptive, NOT filtered counterfactual PnL)'],
        null=dict(draws=2000,bootstrap_draws=2000,seed=20261108,
            design='shared UTC-day sign flips with sign-invariant RMS max-T and paired day bootstrap',
            family='five pairs x J/X x settlement/+1s/inventory-square=30'),
        multiplicity='two additional policies after prior late2/family18 and earlier experiments; no outcome-driven retuning',
        live_fill_warning='new Nov8 same-window review questions young/cancel-pending replay fills; this study does NOT calibrate/correct fill timing or claim live-achievable edge',
        evidence_limits='live order total lifetime differs from replay fill age; nearest pending-message matching is not exact transaction identity; no blind age-filter calibration',
        controls=['safe same-price retention positive/over-cap refusal','new closing size positive/minimum refusal','no unreported physical-fill credit',
                  'new-order capacity and no-reverse rechecks','exact pre60 and reference parity','per-fill intent/cash/carry/path reconciliation','wrong source refusal'],
        loss_budget=None,exposure_backstop_shares=150,rebates='excluded',timings=['J','X'],
        latency_ms=dict(place=40,cancel=22,pending_cancel=95,fill_known=100),
        limitations=['no own market impact/full live OMS','prediction guard still controls whether reduction is quoted',
                     'not forced liquidation; inherited inventory and pending orders persist','candidate objective is unproven until measured'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        prepared_source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},
        prepared_input_sha={str(p):sha(p) for p in files},prior_receipt_sha=sha(PRIOR/'report.json'),
        external_review_sha={str(p):sha(DEST/'evidence'/p.name) for p in evidence},
        evidence_snapshot_sha={str(DEST/'evidence'/p.name):sha(DEST/'evidence'/p.name) for p in evidence},
        forward_data_read=False,live_settings_changed=False,deployment=False)
    write(DEST/'declaration.json',declaration)
    print('DECLARED',DAYS,flush=True)


if __name__=='__main__':setup()
