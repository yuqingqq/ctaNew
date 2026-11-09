"""One predeclared allowed-inventory-range candidate, no threshold search."""
import datetime as dt
import subprocess
from .pf_sizing_study import ROOT,HERE,BASE,DAYS,sha,write
from .pf_inflight_study import DATA as INPUT,DEST as PRIOR,ARMS as OLD_ARMS

DEST=ROOT/'data/predict_fun/research/inventory_range_20261109'
DATA=DEST/'research'
ARMS={k:OLD_ARMS[k] for k in ['baseline','reserved10','both_reserved']}
ARMS['range10']=BASE.replace('+CAP10','+CAP10+IRANGE10')
PAIRS=[('range10','both_reserved'),('range10','reserved10'),('range10','baseline')]
REFERENCES=['baseline','reserved10','both_reserved']


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
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,arms=ARMS,pairs=PAIRS,
        population='all 288 resolved markets per day, 2592 per arm/timing; identical flat initial state and tape',
        selection='all nine already-consumed dates; fixed guards trained on eight; exploratory historical test',
        candidate='one fixed +/-10-share range; for both sides room=max(0,10-signed known net-all same-side known remaining reservations)',
        example='known net +10, no pending sells: allow up to 20 total sell shares, reaching -10; existing sell reservations subtract from 20',
        retention='same eligible remaining-quantity retention on both sides as strict both_reserved; no upsize of eligible resting order',
        reservations='resting, pending create, pending cancel; all lanes, unique object identity; size-reported fills; no opposite-side credit',
        cancellation='same as strict control: reserve until cancel confirmation +100ms fill-report delay +.001ms; conservative simulator approximation',
        unchanged='original desired quotes, guards, AM5/25 sizes, RO60 per-order clipping and rung withdrawal, S240, minimum lots/notional, execution timings',
        late_phase='no new MM-cycle restart after60; existing per-order RO controls remain, but outstanding orders may collectively reverse within the allowed range',
        invariants=['each submission respects remaining worst-side range','every physical fill stays within +/-10 for range10 and strict both_reserved',
                    'independent full-path reconstruction and per-market settlement/carry reconciliation'],
        reference_reuse='fresh first full day J/X plus eight-market smoke for all three references; other references read by artifact hash',
        metrics=['settlement','+1s markout','time-weighted squared inventory','residual/peak inventory/time>10',
                 'direct zero-crossing fills','day SD/worst day/drawdown','FIFO matched/residual PnL','volume/requests/per-share markout'],
        attribution='FIFO includes price drift and is not pure spread; exposure-square is a risk proxy; turnover not matched',
        null=dict(draws=2000,bootstrap_draws=2000,seed=20261108,
            design='paired shared UTC-day sign flips with sign-invariant RMS max-T; paired day bootstrap',
            family='range10 versus three fixed controls x J/X x settlement/+1s/inventory-square =18'),
        prior_selection='candidate proposed after entry-budget and inflight studies (two new candidates and 18 comparisons each); no new forward-race entry',
        limitations=['nine consumed days cannot independently validate a selected policy','no own market impact/full live OMS','fixed delay bound is not a live guarantee'],
        timings=['J','X'],latency_ms=dict(place=40,cancel=22,pending_cancel=95,fill_known=100),
        loss_budget=None,exposure_backstop_shares=150,rebates='excluded',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        prepared_input_sha={str(p):sha(p) for p in files},prior_receipt_sha=sha(PRIOR/'report.json'),
        forward_data_read=False,live_settings_changed=False)
    write(DEST/'declaration.json',d);print('DECLARED',DAYS,flush=True)


if __name__=='__main__':setup()
