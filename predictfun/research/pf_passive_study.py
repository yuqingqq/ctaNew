"""Predeclared, bounded passive-inventory experiment on consumed dates only."""
import datetime as dt
import json
from pathlib import Path
import subprocess
from .pf_sizing_study import BASE, ROOT, HERE, sha, write, DATA as INPUT

DAYS=['20261027','20261031','20261101','20261102','20261103']
DEST=ROOT/'data/predict_fun/research/passive_inventory_20261109'
DATA=DEST/'research'
ARMS={k:BASE.replace('+CAP10','+CAP10'+v) for k,v in {
    'baseline':'','keep_queue':'+RKEEP','small_reduce':'+RSMALL',
    'urgent_quote':'+RPRICE','combined':'+RKEEP+RSMALL+RPRICE'}.items()}
ACK_MS=143.3536010742191


def setup():
    if DEST.exists():raise ValueError('refuse existing experiment')
    DEST.mkdir();(DEST/'logs').mkdir();(DATA/'step4').mkdir(parents=True)
    for d in DAYS:
        for stage in ['step1','step5']:
            p=DATA/stage/d;p.parent.mkdir(exist_ok=True);p.symlink_to(INPUT/stage/d,target_is_directory=True)
    for f in ['minute_lines_t60k.json','inv_r60.json']:
        (DATA/'step4'/f).symlink_to(INPUT/'step4'/f)
    declaration=dict(declared_at=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,
        selection='five latest complete modelled dates from previous consumed-day study, fixed before new results',
        population='all 288 venue-resolved markets/day; every arm starts flat per market; same pre-60s policy',
        arms=ARMS,loss_budget=None,exposure_cap_shares=150,rebates='excluded',
        keep_queue='after 60s, preserve same-price guard-eligible current order despite size changes only when its known remainder plus all other reducing reservations fits known inventory; combined also limits remainder to 5',
        small_reduce='after 60s, cap the original reduce-only desired size at 5 shares; no new no-flip guarantee',
        urgent_quote='180<=t<240 and abs(known net)>=10: improve reducing quote by one 1-cent tick if strictly passive; original quote must pass all original guards',
        urgency_cost='Deliberately spends 1 cent/share for faster reduction. Original model score is not claimed calibrated to improved price; full price cost enters actual fill PnL.',
        timings=['J','X'],latency_ms=dict(place=40,cancel=22,pending_cancel=95,fill_known=100),
        stress=dict(day='20261103',cancel_ack_ms=ACK_MS,physical_removal='unchanged',timings=['J','X'],
                    scope='one-day uniform completed-cancel p99 reservation sensitivity; no interval or calibrated live OMS claim'),
        metrics=['settlement excluding rebates','one-second markout','mark at 240s','carry','drawdown',
                 'inventory area 60..300s','net inventory at end','reducing fills and zero-crossing residual',
                 'request counts','risk/throttle/post-only refusals','queue age of fills (descriptive)'],
        null=dict(design='paired UTC-day sign flips; shared flips across arms, metrics and J/X; sign-invariant scaling',
                  draws=2000,bootstrap_draws=2000,seed=20261108,
                  family='4 candidates x 2 timings x 2 metrics (settlement and +1s) = 16; all other metrics descriptive'),
        controls=['disabled adapter exact baseline parity with previous study',
                  'pre-60s fills and requests exact between all arms in every window',
                  'per-market fill/settlement and carry reconciliation',
                  'synthetic queue/partial-fill/reservation/size/price engagement and known-bad refusal',
                  'day-null signal positive control and invalid population refusal'],
        limitations=['fixed guard trained on four of these five days; exploratory in-sample comparison',
            'all dates already consumed; no untouched validation or new strategy selection',
            'historical tape has no own market impact or full OMS/watchdog replay',
            'known-state order reservations retained; fills can still cross zero before knowledge/cancel catches up',
            'post-fill diagnostics are attribution, not an outcome-selected learning or causal-fill comparison',
            'no quoting beyond 240s: frozen guard coverage ends there'],
        forward_data_read=False,live_settings_changed=False,
        prior_receipt_sha=sha(INPUT.parents[0]/'receipt.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip())
    write(DEST/'declaration.json',declaration);print(json.dumps(declaration,indent=2))


def context():
    from . import pf_io as io
    io.OUT=str(DATA)


if __name__=='__main__':setup()
