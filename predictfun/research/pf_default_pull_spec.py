"""Frozen experiment: quote by default, estimate the value of pulling an order."""
import datetime as dt
import json
import subprocess
from pathlib import Path
from . import pf_order_policy_spec as OLD

ROOT=OLD.ROOT
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/predict_fun/research/default_pull_20261111'
REVIEW=HERE.parent/'reviews/default_pull_20261111'
DAYS=OLD.DAYS
TEST=OLD.TEST
TIMINGS=OLD.TIMINGS
MODES={'reference':'', 'matched_guard':'DMATCH', 'default':'DDEFAULT',
       'pressure':'DPRESS', 'floor':'DFLOOR', 'market':'DMARKET',
       'queue':'DQUEUE', 'queue_cost':'DCOST'}
GROUPS={'smoke':['reference','default'], 'build':['reference','default'],
        'eval':['matched_guard','pressure','floor','market','queue','queue_cost'],
        'stress':['reference','matched_guard','default','pressure','floor','market','queue','queue_cost']}
COOLDOWN=1000.
LABEL_MS=2000.
PROBE_AGES=(100.,250.,500.,1000.,2000.,5000.)
RISK_C=.01
sha=OLD.sha

def native(x):
    import numpy as np
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    if isinstance(x,np.generic):return native(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write(path,obj):
    obj=native(obj);OLD.write(path,obj)

def arm(mode):
    p,r,m,s=OLD.BASE.split('|')
    return '|'.join([p,r+('+'+MODES[mode] if MODES[mode] else ''),m,s])

def stem(mode):
    p,r,m,s=arm(mode).split('|')
    return f'{p}+{r}/{m}@{s}'.replace('/','__').replace('+','_').replace('@','_sz').replace(':','-')+'.parquet'

def folder(day,group):return OUT/'runs'/group/day

def declare():
    if OUT.exists():raise ValueError('output already exists; supersede')
    # Freeze every imported research source, not just the new builder.
    sources={str(p.relative_to(HERE.parents[1])):sha(p) for p in sorted(HERE.glob('pf_*.py'))}
    for p in [HERE/'run_default_pull.sh',HERE/'tests/test_default_pull.py']:
        sources[str(p.relative_to(HERE.parents[1]))]=sha(p)
    prior=json.loads((OLD.OUT/'declaration.json').read_text())
    for p,h in prior['input_sha'].items():
        if sha(p)!=h:raise ValueError('input changed: '+p)
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha=sources,input_sha=prior['input_sha'],days=DAYS,test_days=TEST,timings=TIMINGS,modes=MODES,
        prior_complete='Own-order/calm study d475eccb; comparison_v4.json. No promoted candidate; this broader default-quote experiment is separate.',
        design='Frozen v3 benchmark plus matched 5-share JOIN touch strategies, no ladder; same-price original quantity retained after partial fills. Rest within 2 ticks behind best, never cross. New default strategies remove PM90/G1/G2 admission gates. Data freshness, valid prices, minimum $1 order, client-known in-flight/cancel reservations, inventory range +/-10, RO60, S240 remain enforced. These structural differences are shared by matched_guard and all new strategies; benchmark differences alone do not identify model benefit.',
        actions='One neutral decision row per sent generation, including refusals/unfilled/no-probe statuses. Probe age chosen at submission by deterministic hash from 100/250/500/1000/2000/5000ms; first causal decision at/after that age, independent of outcomes. Neutral default path has no learned or signal cancellations. Features only client-known remaining, inventory, actual order age/price, observed depth and market state; physical queue only for labels. Policy scores actual accepted orders every100ms from age100ms; re-entry waits1000ms after learned cancellation.',
        label='Local KEEP minus CANCEL+RESET value over2s after probe. Snapshot at t+effective cancel latency removes unavoidable prefix. KEEP extends actual latent queue; CANCEL has no order until probe+1000ms+place, then a same-price reset of remaining quantity known by reset submission, initialized at queue back under identical execution assumptions. Reset notices include only unavoidable prefix fills from the CANCEL branch. Both obey fresh-book/noncrossed/2-tick bounds and horizon. Value half1s+half5s per-fill markouts less0.01c change in inventory-square from decision-time known inventory. Existing same-side active/pending/cancelled orders compete; future independent admissions and opposite-side inventory responses are not forked. This is a local training approximation, not full portfolio causal P&L; sequential replay evaluates actual cooldown/re-entry interactions.',
        policies='reference frozen v3; matched_guard same new sizing/price/holding/cap architecture with original PM90 and G1/G2; default no signal pull; pressure corrected same-price thinning OR side imbalance OR adverse burst/imbalance shift; floor v<=-0.10c; market/queue models pull when predicted KEEP-minus-RESET<0; queue_cost only if prediction<-0.01c*known remaining. Hard/data exits override every policy; pressure/floor wait1000ms and require trigger to clear for entry. Models estimate; thresholds live in policy.',
        features='Pinned original32 market features plus actual order state, age, guard value, causal250ms deltas/arithmetic activity ratio; queue model adds4 observed depth/progress fields. Market model includes age and guard value to remove tautological benefit. Unknown required market/pressure data means withdraw, never calm/safe. Same-price thinning compares exact order price with elapsed250ms, fresh old frame; price change is not thinning.',
        training='Pinned LightGBM PARAMS and300rounds; expanding first3days then6testdays; one neutral generation row per action, no filled-only selection. Separate market+age and queue models. Scores recomputed from saved model artifacts; chronological layer conditional on old LODO guard caches. All dates consumed, no validation/promotion claim.',
        null=dict(draws=500,seed=20261111,method='Pull indicators shuffled on neutral test actions within day/side/UTC hour/order-age bucket/inventory sign; counts preserved; decision value=-label, not harm rate. Report shuffleable groups and zero-fill actions.'),
        evaluation='Exact passive reference win/fill/request parity on9days. Independently reconcile all portfolio generations/fills/cash/settlement/inventory area. Compare marked value on paired common windows, full settlement, strict1s/5s markouts and missing statuses, spread/carry, inventory area/terminal exposure, presence/lifetime/churn. Day-cluster1000draw intervals on6complete consumed testdays. Same8arms under conservative combined timing/queue stress; success needs improved marked trading value with nonincreasing inventory area versus matched_guard and reference under both scenarios.',
        controls='Synthetic positive/falsifier pairs: unknown data, same-price thinning, elapsed-time deltas, effective cancel prefix, reset queue priority, quantity/reservation cap, causal warmup, engine source anchors, accounting, duplicated actions, model artifact identity. Refuse on failures; preserve failed artifacts and supersede corrections.',
        multiplicity=dict(primary_arms=8,learned_models=2,learned_policy_thresholds=2,scenarios=2,posthoc_tuning=0),
        operations='One numeric process through run_light, core8/4GiB AS/2GiB RSS. Durable driver waits for other numeric processes, tracks every stage and receipts, no silent retries/overwrite. No live change, no rebate/loss-budget changes, no protected payload reads. Subsequent ideas require new declarations/receipts.')
    write(OUT/'declaration.json',d);write(REVIEW/'declaration.json',d)
    print('DECLARED',len(d['input_sha']),flush=True)

def check():
    d=json.loads((OUT/'declaration.json').read_text())
    for p,h in d['source_sha'].items():
        if sha(HERE.parents[1]/p)!=h:raise ValueError('frozen source changed: '+p)
    return d

if __name__=='__main__':declare()
