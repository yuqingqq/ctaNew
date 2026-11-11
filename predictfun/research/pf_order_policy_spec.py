"""Own-order continuation and calm-entry experiment, isolated from frozen v3."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

ROOT=Path('/home/yuqing/ctaNew')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/predict_fun/research/own_orders_calm_20261111'
REVIEW=HERE.parent/'reviews/own_orders_calm_20261111'
DAYS=['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']
TEST=DAYS[3:]
BASE='MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95'
MODES={'reference':'','hold_rule':'OHOLD','continuation':'OLEARN','calm_filter':'OFILTER',
       'calm_seed':'OSEED','calm_seed_continuation':'OSEED+OLEARN'}
TIMINGS={'X':dict(shift=-50,place=40,cancel=22,notice=100,queue='M'),
         'stress':dict(shift=-250,place=80,cancel=100,notice=250,queue='L')}
HOLD_MS=1000.
BAND_C=.01
RISK_C=.01
CONT_MARGIN_C=.01


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def arm(mode):
    p,r,m,s=BASE.split('|');extra=MODES[mode]
    return '|'.join([p,r+('+'+extra if extra else ''),m,s])


def name(mode):
    p,r,m,s=arm(mode).split('|');return f'{p}+{r}/{m}@{s}'


def stem(mode):return name(mode).replace('/','__').replace('+','_').replace('@','_sz').replace(':','-')+'.parquet'


def declare():
    if OUT.exists():raise ValueError('study exists; supersede rather than overwrite')
    files=sorted(HERE.glob('pf_order_policy_*.py'))+[HERE/'tests/test_order_policy.py',HERE/'run_order_policy.sh',
        HERE/'pf_step4_ladder_v3.py',HERE/'pf_clock_rl.py',HERE/'pf_io.py',HERE/'pf_model_spec.py',HERE/'pf_inv_r60.py',
        HERE.parent/'freeze/pm_ladder_v3/params.json']
    source={str(p.relative_to(HERE.parents[1])):sha(p) for p in files}
    prior=json.loads((HERE.parent/'reviews/queue_model_20261111/declaration.json').read_text())
    inputs=prior['input_sha'].copy()
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('prior input changed: '+p)
    from .pf_io import hours_of
    for day in DAYS:
        for hour in hours_of(day,extra_before=1,extra_after=1):
            if hour[:8] in ('20261028','20261029','20261030') or hour[:8]>='20261104':continue
            p=ROOT/'data/mm_hf/raw/bookTicker/BTCUSDT'/f'{hour}.csv.gz'
            if p.exists():inputs[str(p)]=sha(p)
    lines=ROOT/'data/predict_fun/research/step4/minute_lines_t60k.json';inputs[str(lines)]=sha(lines)
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha=source,input_sha=inputs,days=DAYS,test_days=TEST,base=BASE,modes=MODES,timings=TIMINGS,
        population='All sent generations of the frozen v3 research engine, including refusals/unfilled orders/statuses. One first cancellation decision per generation. Learned continuation applies only to accepted touch orders with client-known remaining size<=5, in the existing0.01c soft band, with all hard price/market-data/inventory guards intact.',
        labels='Reference and calm-seed neutral paths have no learned cancellation override. At actual cancellation effectiveness snapshot the order after all unavoidable fills. Independently extend its queue up to1s from the decision or the next market hard stop, respecting existing same-side orders and pending creates, including pending cancellations. Those competitors are held until already known expiries; future reference reprices/admissions are not forked into the label. Full sequential replays evaluate the policy interaction. Incremental future fills valued at their own1s and5s marks equally, less0.01c incremental inventory-square penalty. Latent queue/remain are labels only; features use notified fills and observed same-price depth.',
        continuation='Pinned LightGBM parameters/300 rounds. Expanding chronology train first3days then6test days; reference+seed neutral generation rows, including no-fill outcomes. KEEP only if predicted incremental value exceeds0.01c per remaining share; otherwise original cancel. One override per generation;1s cap, hard guards always win. Compare simple hold rule, unchanged reference, and age-matched market-only diagnostic.',
        calm='Causal250ms stability: unchanged Predict touch prices, neither side loses20% displayed size at the same price, abs(Predict imbalance)<1/3, abs250ms Binance imbalance change<0.3, abs250ms Binance return<0.5bps, current1s activity<2x trailing arithmetic30s mean (>=5s history). All required observations finite/fresh; missing is unknown, never calm. Require five consecutive50ms calm observations before seed entry.',
        entry='Calm filter gates only new exposure-increasing baseline orders, never rests or reduces. Calm seed permits5share touch orders in first60s when only PM entry line withholds and v is within0.01c below it. Seed rests in that band while calm for at most1s, then original logic resumes. No new rung, size increase, crossed quote or risk exception.',
        evaluation='Exact baseline win/fill/request parity against stored b14rX; passive hooks must not change engine. Full sequential engine portfolios: primary six arms on six testdays; stress reference/continuation/seed/seed+continuation. Compare trading markouts/spread, cash/mark/settlement carry, inventory-square time and terminal inventory, requests and fills. Independent selected-fill reconciliation. No rebates or live changes.',
        null=dict(draws=500,seed=20261111,method='Continuation KEEP indicators shuffled within day,side,UTC hour,age bucket and inventory sign, preserving action counts on neutral generation rows. Calm-filter diagnostic permutes entry calm labels within day/side/hour, preserving count, using reference generation values. Both are neutral-action diagnostics, not full-policy counterfactual inference.'),
        inference='Day-cluster paired bootstrap1000 on six consumed test dates. No untouched validation. Primary success needs improved marked trading value with nonincreasing inventory-square exposure, and consistent direction under stress. Missing marks remain statuses; common-window comparison alongside full settlement; no promotion.',
        multiplicity=dict(primary_policy_arms=6,new_learned_candidates=1,entry_variants=2,scenarios=2,posthoc_tuning=0),
        limits='Frozen v3 research admissions, not the uncalibrated current live incremental-size implementation. Existing cached guard uses prior LODO predictions; new continuation layer is chronological conditional on that guard. Queue/fill calibration remains open. Protected Oct28-30/Nov4-6/Nov9-13 payloads not read. One run_light process.')
    write(OUT/'declaration.json',d);write(REVIEW/'declaration.json',d)
    print('DECLARED',len(inputs),'inputs',flush=True)


def check():
    d=json.loads((OUT/'declaration.json').read_text())
    for p,h in d['source_sha'].items():
        if sha(HERE.parents[1]/p)!=h:raise ValueError('frozen source changed: '+p)
    return d


if __name__=='__main__':declare()
