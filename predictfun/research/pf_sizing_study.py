"""Declared exploratory sizing and settlement study. Frozen inputs stay read-only."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).parent
OLD = ROOT / 'data/predict_fun/research'
DEST = OLD / 'sizing_settlement_20261108'
DATA = DEST / 'research'
DAYS = ['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']
BASE = 'MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95'
ARMS = {'baseline':BASE, 'capacity':BASE.replace('+CAP10','+CAP10+HCAP'),
        'hysteresis':BASE.replace('+CAP10','+CAP10+SHYST'),
        'combined':BASE.replace('+CAP10','+CAP10+HCAP+SHYST')}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def setup():
    if DEST.exists():raise ValueError('refuse existing study output')
    DEST.mkdir(parents=True)
    declaration=dict(declared_at=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,
      scope='exploratory on consumed days; no selection, deployment or forward freeze',
      strategy=ARMS,loss_budget_usdt=[150,None],net_share_backstop=150,
      capacity='clip new orders to worst-side share headroom including pending cancels and creates; keep reservations',
      hysteresis='AM size 5/25 switches up at line + 0.01c and down below line - 0.01c; guards remain immediate',
      common='continuous basis; fixed frozen deployed guard models; 0.01 share floor; >=1USDT; risk refusals retry 250ms..4s',
      timings=dict(J='transaction frame jitter, normal activation queue',X='transaction -50ms, max activation queue'),
      latency_ms=dict(place=40,cancel=22,pending_cancel=95,fill_known=100),
      probability=dict(train_initial=DAYS[:3],expanding_test=DAYS[3:],anchors_s=[15,30,45,60,90,120,180,240],
        models=['market_mid','brownian_basis','market_calibrated_logistic','market_plus_physical_logistic'],
        target='venue-resolved Up outcome; no Binance-derived label; nonbinary outcomes explicit exclusions',
        unit='neutral scheduled decision per market/anchor; never selected on strategy fills; per-market total training weight one',
        parameters='L2 logistic C=1, max_iter=2000; train-only median imputation/scaling; no tuning',
        primary='day-balanced Brier and log-loss improvement over market and calibrated-market baseline',
        inventory_probe='one-share hold versus immediately hedge at executable opposite ask, 40ms delay and actual fee_bps; no liquidity claim',
        inference='2000 fixed-seed day sign flips + 2000 day bootstrap draws; multiplicity includes both fitted models and two baselines'),
      strategy_metrics=['settlement PnL excluding rebates','1s markout','mark at 240s','drawdown','inventory','requests','refusals','halted windows'],
      controls=['unmodified engine parity','synthetic hysteresis and capacity engagement','unknown version refusal','time split refusal','oracle probability positive control','shuffled-label refusal'],
      limitations=['frozen guard trained on eight of these days: strategy comparisons in-sample',
        'fixed measured delays, not full independent live OMS parity; no own market impact',
        'settlement study is chronological but days were previously consumed; later untouched validation still needed',
        'loss caps carry over studied days in order; gaps skip excluded days, not continuous live history'],
      forward_data_read=False,live_limit_changed=False,
      code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip())
    write(DEST/'declaration.json',declaration)
    (DEST/'logs').mkdir()
    (DATA/'step4').mkdir(parents=True)
    for f in ['minute_lines_t60k.json','inv_r60.json']:
        (DATA/'step4'/f).symlink_to(OLD/'step4'/f)
    for day in DAYS:
        dst=DATA/'step1'/day;dst.mkdir(parents=True)
        for p in (OLD/'step1'/day).iterdir():
            if p.name.startswith('clock_rl') and p.is_file():shutil.copy2(p,dst/p.name)
            else:(dst/p.name).symlink_to(p)
        (DATA/'step5'/day).mkdir(parents=True)
    print(DEST)

def context():
    from . import pf_io as io
    io.OUT=str(DATA)
    return io

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['declare']);a=p.parse_args()
    setup()

if __name__=='__main__':main()
