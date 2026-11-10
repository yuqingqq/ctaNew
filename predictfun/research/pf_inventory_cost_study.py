"""Declare and hash a bounded, consumed-data experiment before reading outcomes."""
from pathlib import Path
import datetime as dt
import json
import subprocess
from .pf_sizing_study import ROOT,HERE,BASE,DAYS,sha,write
from .pf_late_study import DATA as INPUT

DEST=ROOT/'data/predict_fun/research/inventory_cost_20261110'
DATA=DEST/'research'
REVIEW=HERE.parent/'reviews/inventory_cost_20261110'
ARMS={'baseline':BASE,'late_reserved':BASE.replace('+CAP10','+CAP10+LRES'),
      'soft_exit':BASE.replace('+CAP10','+CAP10+ESOFT'),
      'budget_exit':BASE.replace('+CAP10','+CAP10+ECOST'),
      'marginal_entry':BASE.replace('+CAP10','+CAP10+ERISK')}
PAIRS=[(a,b) for a in ('soft_exit','budget_exit','marginal_entry') for b in ('baseline','late_reserved')]+[('budget_exit','soft_exit')]
PRIOR=ROOT/'data/predict_fun/research/late_reduction_20261109/report.json'


def source_files():
    # The full Python pipeline is frozen, including transitive helpers and controls.
    return sorted(HERE.glob('*.py'))+sorted((HERE/'tests').glob('*.py'))+[HERE/'run_inventory_cost.sh']


def check():
    d=json.loads((DEST/'declaration.json').read_text())
    for name,h in d['source_sha'].items():
        if sha(HERE.parents[1]/name)!=h:raise ValueError('source changed: '+name)
    return d


def setup():
    if DEST.exists():raise ValueError('refuse existing experiment')
    (DEST/'logs').mkdir(parents=True);(DATA/'step4').mkdir(parents=True)
    files=[]
    for day in DAYS:
        for stage in ('step1','step5'):
            p=DATA/stage/day;p.parent.mkdir(exist_ok=True);p.symlink_to(INPUT/stage/day,target_is_directory=True)
        files += [DATA/'step5'/day/'preds_basis_study.parquet']
        files += [DATA/'step1'/day/f for f in ('registry.parquet','clock_rl_d2.json','legs_timed2.parquet','legs_tk.parquet')]
        files += sorted((DATA/'step1'/day/'books').glob('*.parquet'))
    for f in ('minute_lines_t60k.json','inv_r60.json'):
        (DATA/'step4'/f).symlink_to(INPUT/'step4'/f);files.append(DATA/'step4'/f)
    prior=json.loads(PRIOR.read_text())
    files += [Path(p) for p in prior['binance_input_sha']]+[PRIOR]
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,arms=ARMS,pairs=PAIRS,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in source_files()},
        input_sha={str(p):sha(p) for p in files},
        population='all288 markets/day;2592 per arm/J or X,9 previously consumed dates; no outcome-based exclusions; guards trained on8 dates',
        baseline='same continuous-basis, single-order execution approximation as prior exit studies; not native incremental LZ/live parity',
        controls='baseline and late_reserved freshly replayed every day; both exact to prior win/fill/request artifacts',
        audit='baseline only, every normal desired invocation from60s; consecutive identical feasibility states compressed into episodes, all predicates plus first blocking reason; durations are decision-sampled states, not continuously observable queue availability',
        soft_exit='late_reserved plus reducing-side PM90 cutoff lower by .01 score units during60..240; preserve Binance,GS,price,book,notional,room,stop; new threshold-crossing events only in this interval',
        budget_exit='soft_exit plus one tick strictly inside spread during180..240 when abs known net>=10; cumulative submitted concessions <=$0.25/window including all uncertain orders, no refunds for cancels/rejections; cap at send and floor lots; retain paid-for existing order',
        budget_limits='bounds deliberate quote-price concession versus originating unshifted quote only; not adverse selection, total exit loss, or causal incremental cost; never subtract twice from cash PnL',
        marginal_entry='late_reserved after60; before60 use premium .01 score per625 increase in possible inventory radius squared; clip quantity using fresh score minus PM90 line, pending both sides independently including cancellation grace, all lanes and retry recheck; preserves same-price feasible remainder; no settlement forecast/dollar-score interpretation',
        timing={'J':'transaction frame jitter,40/22/95/100ms place/cancel/pending/fill-known',
                'X':'transaction -50ms,max activation queue,same delays',
                'stress_X':'Nov3 only,X plus80/22/95/300ms delays; full-path paired replay,assumptions not calibration'},
        null=dict(draws=2000,bootstrap_draws=2000,seed=20261108,cluster='UTC day',
                  design='shared paired day sign flips, sign-invariant RMS max-T, paired bootstrap',
                  family='7pairs x2timings x3metrics(settlement,+1s,inventory-square)=42; stress descriptive only'),
        multiplicity='3new policies,one fixed setting each after prior consumed-data selection; no tuning based on this run',
        metrics=['spread','adverse_1s','plus1','mark240','carry','settlement','FIFO matched/residual','concession committed/filled','shares','inventory_square','inventory duration/terminal/peak','requests'],
        limits='historical fill clocks uncalibrated; no independent validation, no market impact/full native OMS, no live-achievable claim',
        loss_budget=None,backstop_shares=150,rebates='excluded',forward_data_read=False,live_changes=False)
    write(DEST/'declaration.json',d);write(REVIEW/'declaration.json',d)
    print('DECLARED',d['population'])


if __name__=='__main__':setup()
