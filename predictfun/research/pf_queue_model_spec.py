"""Declared controlled queue-continuation experiment; separate from frozen v3."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).resolve().parent
OUT = ROOT/'data/predict_fun/research/queue_model_20261111'
REVIEW = HERE.parent/'reviews/queue_model_20261111'
DAYS = ['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']
STARTS = list(range(5, 226, 10))
AGES = [250., 1000., 2000.]
SIZE = 5.
LEASE = 5000.
RISK_C = .01  # cents per share squared; fixed design choice, not fitted
CONFIGS = {'L50':dict(rule='L',shift=50.,place=40.,cancel=22.,notice=100.),
           'M50':dict(rule='M',shift=50.,place=40.,cancel=22.,notice=100.),
           'L250slow':dict(rule='L',shift=250.,place=80.,cancel=100.,notice=250.)}
ACTIONS = ['keep','cancel','rejoin']
L90 = [.0492,.0623,.0789,.0917]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sources():
    return sorted(HERE.glob('pf_queue_model_*.py'))+[HERE/'tests/test_queue_model.py',HERE/'pf_model_spec.py',HERE/'pf_io.py',HERE/'run_queue_model.sh']


def declare():
    if OUT.exists():raise ValueError('study already exists; supersede, never overwrite')
    source={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources()}
    inputs={}
    for day in DAYS:
        s=ROOT/f'data/predict_fun/research/step1/{day}'
        paths=[s/'registry.parquet',s/'legs_timed2.parquet',s/'legs_tk.parquet',s/'clock_rl_d2.json']
        paths+=sorted((s/'books').glob('*.parquet'))
        paths += [ROOT/f'data/predict_fun/research/step5/{day}/{n}.parquet' for n in ('rows_rl_d2_pk','preds_t60k')]
        for p in paths:inputs[str(p)]=sha(p)
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha=source,input_sha=inputs,days=DAYS,starts_s=STARTS,ages_ms=AGES,configs=CONFIGS,
        design='Controlled fixed-5-share queue-continuation model. One neutral generation per side every10s from5..225s in every BTC5m window;5s lease, no replenishment except the explicitly tested sequential cancel/rejoin.',
        unit='One action row per neutral generation at a deterministic age250/1000/2000ms after activation; all scheduled generations retained as statuses. All actions within each generation share one marking horizon; portfolio arms share240s. Not a full v3 replacement.',
        actions='KEEP queue; CANCEL with effective latency; REJOIN same price after cancel takes effect plus create latency, size=client-known remaining. Same unavoidable fills before cancel. All arms use the same post-only,1USD minimum notional,stale/book,gap-to-touch and Binance price guards. Maker fees must be0 in registry; no rebates.',
        information='Features use cached causal Tokyo-clock market rows and last-received books. Queue features use initial displayed size, minimum observed size at this exact price, order age and notified fills, never latent simulated queue or unnotified fills.',
        target='Two regressions: KEEP minus CANCEL and REJOIN minus CANCEL. Cash plus inventory marked at min(decision+30s,window240s), less0.01c times change in squared terminal inventory, under neutral other-side continuation. Initial admissions are neutral; model controls continuation, not a learned entry policy.',
        feature_arms=['market: current32 features + causal pressure changes/activity ratio + side/order price/known remaining/signed inventory','queue: market + order age/initial queue/minimum same-price queue/current same-price queue/progress'],
        fit='Pinned pf_model_spec LightGBM parameters/300 rounds; expanding chronological train first3dates then test6. Only L50 trains; same fixed fitted models applied to M50 and L250slow. No horizon,threshold or hyperparameter search.',
        baselines=['always_keep','always_cancel','legacy_score at the same scheduled decisions (diagnostic, existing LODO score is not chronological)','market model without queue features'],
        evaluation='Neutral-action value and a separate sequential window portfolio,5shares,soft10/hard25 shares with full outstanding-size reservation,reduce-only180..240s. Report cash/marked/settlement PnL,spread and1/5/60s markouts,inventory-square time,terminal/max inventory,fill and order counts. No rebates. This controlled policy is not full v3 or live performance.',
        null=dict(draws=500,seed=20261111,method='Shuffle queue-model actions within test day,side,UTC hour,minute,known-inventory sign and rejoin availability, preserving action counts. Compare its sum of neutral-action utility gains with the max across three fixed timing/queue scenarios. Exploratory, not untouched validation.'),
        inference='Day-unit point estimates and paired day bootstrap with1000 draws,6 complete scheduled-policy test days. All dates previously consumed; no independent validation or deployment claim.',
        assessment='Primary comparison queue vs market-only model: common240s marked portfolio value with inventory-square time and terminal absolute inventory alongside. Improvement requires positive paired marked-value evidence and no increased inventory-square point estimate in L50, with same direction under both sensitivities; otherwise inconclusive or rejected, never deployment promotion.',
        controls='Positive queue-progress advantage; toxic keep; cancel-latency common fills; sequential rejoin; partial/unnotified fills; post-only refusal; bid/ask mirror; no cross-window/stale mark backfill; chronological split; bad clocks/identities/protected dates refused; independent saved-artifact arithmetic.',
        multiplicity=dict(feature_candidates=2,queue_candidate=1,timing_scenarios=3,posthoc_tuning=0),
        protections='No Oct28-30/Nov4-6/Nov9-13 quote/outcome reads. Frozen strategy/loss budget/live code unchanged. One light numeric process.')
    write(OUT/'declaration.json',d);write(REVIEW/'declaration.json',d)
    print('DECLARED',len(inputs),'inputs',flush=True)


def check():
    d=json.loads((OUT/'declaration.json').read_text())
    for p,h in d['source_sha'].items():
        if sha(HERE.parents[1]/p)!=h:raise ValueError('frozen source changed: '+p)
    return d


if __name__=='__main__':declare()
