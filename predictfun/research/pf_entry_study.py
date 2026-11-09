"""Predeclared entry-sizing experiment; all nine dates were already consumed."""
import datetime as dt
import json
import subprocess
from .pf_sizing_study import ROOT,HERE,BASE,DAYS,DATA as INPUT,sha,write

DEST=ROOT/'data/predict_fun/research/entry_budget_20261109'
DATA=DEST/'research'
ARMS={'baseline':BASE,'uniform10':BASE.replace('+CAP10','+CAP10+EUN10'),
      'reserved10':BASE.replace('+CAP10','+CAP10+ERES10')}


def context():
    from . import pf_io
    pf_io.OUT=str(DATA)


def setup():
    if DEST.exists():raise ValueError('refuse existing study')
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
    declaration=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,arms=ARMS,
        population='all 288 venue-resolved markets/day; 2592 per arm/timing; same flat initial state and fixed tape',
        selection='all nine previously prepared consumed days; fixed guards trained on eight; exploratory in-sample',
        uniform10='cap new inventory-adding orders at 10 shares per order; retain all original reduction sizing and guards',
        reserved10='size inventory-adding orders to max(0,10 - signed known net - all same-side known remaining reservations); recheck at submission',
        adding='side sign times known net >= 0; orders currently reducing are unchanged even if they could cross zero',
        common_new_arm_mechanics='keep same-price guard-eligible existing adding order if its known remainder fits desired size and applicable room; no upsize while it remains eligible; current and pending orders remain reserved until terminal knowledge',
        allocation='existing orders reserve first, then new desired orders in engine lane order 0,1,2,3; no netting opposite-side reservations; quantity floored to 0.01 and existing 1-share/1-USDT checks',
        limitations=['not a global hard 10-share cap: unchanged reducing orders and delayed fills can cross zero',
                     'new-arm versus baseline includes shared remaining-quantity retention; reserved versus uniform isolates the budget rule',
                     'uniform control fixes order cap, not realized turnover; no ex-post activity matching or tuning',
                     'historical tape has no own market impact or full OMS; no independent validation claim'],
        loss_budget=None,exposure_backstop_shares=150,rebates='excluded',timings=['J','X'],
        baseline_reuse='fresh first full date at J/X plus eight-market smoke; other main baselines copied from verified prior artifacts with hashes',
        controls=['baseline exact win/fill/request parity','common market identity','per-market cash/settlement/carry reconciliation',
                  'remaining-quantity and pending-cancel positive/refusal controls','adding-order submission invariant',
                  'FIFO matched/residual accounting and full-path inventory controls'],
        metrics=['settlement PnL','one-second markout','mean time-weighted squared inventory 0..300s',
                 'mean/peak/residual absolute inventory','time above 10 shares','daily PnL SD/worst day/drawdown',
                 'FIFO matched-trade PnL and residual PnL','shares/requests/min-notional withholding/refusals'],
        attribution='FIFO matched PnL includes price drift between fills and is not pure spread; squared inventory is a risk proxy, not calibrated outcome variance',
        null=dict(design='2000 paired shared UTC-day sign flips with sign-invariant RMS max-T; 2000 paired day bootstrap draws',
                  seed=20261108,draws=2000,bootstrap_draws=2000,
                  family='reserved versus baseline, uniform versus baseline, reserved versus uniform; J/X x settlement/+1s/inventory-square = 18'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        prepared_input_sha={str(p):sha(p) for p in files},forward_data_read=False,live_settings_changed=False,
        prior_sizing_receipt_sha=sha(INPUT.parent/'receipt.json'))
    write(DEST/'declaration.json',declaration);print('DECLARED',DAYS,flush=True)


if __name__=='__main__':setup()
