"""Fixed late-reduction comparison on nine already consumed dates."""
import datetime as dt
import subprocess
from .pf_sizing_study import ROOT, HERE, BASE, DAYS, sha, write
from .pf_range_study import DATA as INPUT, DEST as PRIOR

DEST = ROOT/'data/predict_fun/research/late_reduction_20261109'
DATA = DEST/'research'
ARMS = {'baseline': BASE, 'late_reserved': BASE.replace('+CAP10', '+CAP10+LRES'),
        'late_taper1': BASE.replace('+CAP10', '+CAP10+LTAPER')}
PAIRS = [('late_reserved', 'baseline'), ('late_taper1', 'baseline'), ('late_taper1', 'late_reserved')]


def context():
    from . import pf_io
    pf_io.OUT = str(DATA)


def setup():
    if DEST.exists():
        raise ValueError('refuse existing experiment')
    (DEST/'logs').mkdir(parents=True)
    (DATA/'step4').mkdir(parents=True)
    files = []
    for day in DAYS:
        for stage in ['step1', 'step5']:
            p = DATA/stage/day
            p.parent.mkdir(exist_ok=True)
            p.symlink_to(INPUT/stage/day, target_is_directory=True)
        files += [DATA/'step5'/day/'preds_basis_study.parquet']
        files += [DATA/'step1'/day/f for f in ['registry.parquet','clock_rl_d2.json','legs_timed2.parquet','legs_tk.parquet']]
        files += sorted((DATA/'step1'/day/'books').glob('*.parquet'))
    for f in ['minute_lines_t60k.json', 'inv_r60.json']:
        (DATA/'step4'/f).symlink_to(INPUT/'step4'/f)
        files.append(DATA/'step4'/f)
    sources = sorted(HERE.glob('pf_late*.py')) + [HERE/'run_late_reduction.sh', HERE/'tests/test_late_reduction.py']
    declaration = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), days=DAYS, arms=ARMS, pairs=PAIRS,
        population='all 288 markets/day, 2592 per arm/timing; no outcome-selected subgroup',
        selection='nine consumed dates; guards trained on eight; exploratory, no untouched validation',
        early='exact original orders/fills for t<60s in both variants; inherited orders may remain fillable after60',
        late_reserved='from60, aggregate reducing allowance=max(0,abs(known net)-same-side known outstanding remainder); retain eligible same-price remainder',
        late_taper1='same reservations; when abs(known net)>10*(240-t)/180 for60<=t<240, improve otherwise guard-eligible reducing quote by exactly1 tick only if strictly inside current known spread',
        target='linear urgency trigger from10 shares at60 to0 at240, NOT a guaranteed exposure bound or liquidation schedule',
        guards='all original price guards retained; no quoting when original guard withholds; no taker crossing; original AM5/25 sizes and RO60/S240',
        reservations='all lanes, pending create/resting/pending cancel, reported fills only, no opposite-side credit; no additional reverse budget',
        migration='retain cancellation history through fill-report grace while exposing original lifecycle before60; from60 expose conservative grace, including inherited cancels',
        cancellation='ack+100ms fill-report grace+.001ms, same conservative simulator approximation as previous strict study',
        timer='target evaluated at existing engine decisions; no additional polling clock',
        price_attribution='record actual order origin and every fill; concession=q*signed(price-original eligible price)/100 on shifted fills, descriptive same-fill comparison, NOT a counterfactual fill at original price',
        accounting='existing maker execution cash prices include concessions; no second subtraction; rebates excluded; no hypothetical midpoint liquidation',
        metrics=['settlement','+1s','mark-at240','carry','FIFO matched/residual','residual inventory','time-weighted q^2',
                 'late reducing/opening volume','concession dollars and fill quantity','day SD/tails/drawdown','pre60 exact parity'],
        null=dict(draws=2000,bootstrap_draws=2000,seed=20261108,
                  design='paired shared UTC-day sign flips with sign-invariant RMS max-T and paired UTC-day bootstrap',
                  family='three pairs x J/X x settlement/+1s/inventory-square =18; carry/SD/concentration descriptive'),
        multiplicity='two new policies after earlier sizing, passive, urgency, reservation and range selection; no parameter search after results',
        timings=['J','X'],latency_ms=dict(place=40,cancel=22,pending_cancel=95,fill_known=100),
        loss_budget=None,exposure_backstop_shares=150,rebates='excluded',
        controls=['synthetic price engagement and no-cross refusal','pending cancellation migration and known-state checks',
                  'wrong-source refusal','fresh baseline parity at first full day J/X','all-market pre60 exact fills/requests',
                  'fill identity/cash/carry/path reconciliation and shifted-record refusal'],
        limitations=['early baseline inventory can exceed10 and cannot be removed instantly',
                     'passive exit does not guarantee fill or flat terminal position','no own market impact/full live OMS',
                     'fixed guards selected on these dates; no independent expected-profit claim'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        prepared_source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},
        prepared_input_sha={str(p):sha(p) for p in files},prior_receipt_sha=sha(PRIOR/'report.json'),
        forward_data_read=False,live_settings_changed=False,deployment=False)
    write(DEST/'declaration.json',declaration)
    print('DECLARED',DAYS,flush=True)


if __name__ == '__main__':
    setup()
