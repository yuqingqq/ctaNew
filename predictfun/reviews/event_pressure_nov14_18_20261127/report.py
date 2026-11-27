"""Publish the November block from checked paired rows and its final receipt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'data/predict_fun/research/event_pressure_nov14_18'
DAYS = tuple(f'202611{d}' for d in range(14, 19))


def amount(x):
    return f'{x:+,.2f}'


def main():
    result = json.loads((SOURCE/'comparison.json').read_text())
    assert set(result['coverage']) == set(DAYS)
    lines = ['# Event-driven pressure cancels on November 14–18',
             '', f'Receipt prepared {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. '
             'Five archived UTC days released before scoring, earlier than the November 19–23 development results. '
             'One candidate, using modeled fills and zero passive fees or rebates.',
             '', '| UTC day | Eligible / scheduled | Other statuses |',
             '|---|---:|---|']
    for day, coverage in result['coverage'].items():
        assert sum(coverage.values()) == 288
        excluded = ', '.join(f'{status} {count}' for status,count in coverage.items() if status!='complete')
        lines.append(f'| {day} | {coverage.get("complete",0)} / 288 | {excluded} |')
    if result['sufficient']:
        lines += ['', '| UTC day | Grid settlement, USD | Event settlement, USD | Difference, USD | Grid / event +1 s, cents/share |',
                  '|---|---:|---:|---:|---:|']
        frames=[]
        hashes={}
        for day in DAYS:
            part = SOURCE/day
            path = part/f'paired_{day}.csv'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert json.loads((part/'pair_hashes.json').read_text())[path.name]==digest
            hashes[day]=digest
            frame = pd.read_csv(path,float_precision='round_trip')
            assert frame.mid.is_unique and frame.day.eq(int(day)).all()
            for arm in ('grid','event'):
                np.testing.assert_allclose(frame[f'{arm}_turnover']+frame[f'{arm}_residual'],frame[f'{arm}_settle'],atol=1e-6,rtol=0)
                np.testing.assert_allclose(2*frame[f'{arm}_closed']+frame[f'{arm}_abs_end'],frame[f'{arm}_shares'],atol=1e-6,rtol=0)
            rec=result['daily'][day]
            assert len(frame)==rec['markets']==result['coverage'][day]['complete']
            assert abs(frame.event_settle.sum()-frame.grid_settle.sum()-rec['difference'])<1e-6
            lines.append(f'| {day} | {amount(rec["grid_settle"])} | {amount(rec["event_settle"])} | {amount(rec["difference"])} | {rec["grid_mark_c_per_sh"]:.3f} / {rec["event_mark_c_per_sh"]:.3f} |')
            frames.append(frame)
        df = pd.concat(frames,ignore_index=True)
        daily = df.groupby('day')[['grid_settle','event_settle']].sum()
        delta = daily.event_settle-daily.grid_settle
        p=result['pooled']
        assert p['directional_pass']==bool(delta.sum()>0 and delta.sum()-delta.max()>0 and (delta>0).sum()>=3)
        assert p['absolute_pass']==bool(daily.event_settle.sum()>0 and (daily.event_settle>0).sum()>=3 and daily.event_settle.sum()-daily.event_settle.max()>0 and df.event_settle.sum()-df.event_settle.max()>0)
        assert len(df)==p['markets']
        lines+=['', '| Measure | Grid | Event |',
                '|---|---:|---:|']
        for name,metric in [('Settlement, USD','settle'),('Completed FIFO profit, USD','turnover'),
                            ('Residual at payout, USD','residual'),('One-second markout, cents per valued share','mark_c_per_sh'),
                            ('Gross adverse +1 s, USD','marks_adverse'),('Gross favorable +1 s, USD','marks_favorable'),
                            ('Shares lacking a +1 s mark','marks_censored'),('Filled shares','shares'),
                            ('Mean absolute inventory at payout, shares','mean_end'),('Cancel requests','requests'),('Reposts','reposts')]:
            lines.append(f'| {name} | {p["grid_"+metric]:,.3f} | {p["event_"+metric]:,.3f} |')
        lines += ['', f'Event minus grid settlement was {amount(p["difference"])} USD. Excluding its best improvement day, the difference was {amount(p["delta_without_best_day"])} USD. '
                  f'The event arm improved {p["improved_days"]}/5 days and settled positive on {p["profitable_days"]}/5. '
                  f'The directional screen {"passed" if p["directional_pass"] else "failed"} and the absolute screen {"passed" if p["absolute_pass"] else "failed"}.',
                  '', 'The two arms generated separate order and inventory paths. The historical feed delay was 4 ms, the nominal venue cancel effect was 37 ms, and placement and reposts remained on the 50 ms grid. '
                  'The live wallet fill rate and venue removal time remain unresolved.']
        abs_checks = {
            'pooled_settlement':float(daily.event_settle.sum()),
            'settlement_without_best_day':float(daily.event_settle.sum()-daily.event_settle.max()),
            'settlement_without_best_market':float(df.event_settle.sum()-df.event_settle.max()),
            'profitable_days':int((daily.event_settle>0).sum())
        }
        lines += ['', f'Event settlement after removing its best day was {amount(abs_checks["settlement_without_best_day"])} USD and after removing its best market was {amount(abs_checks["settlement_without_best_market"])} USD.']
        result['absolute_components']=abs_checks
        result['paired_sha256']=hashes
    else:
        lines+=['','The block has insufficient coverage under the frozen rule.']
    (HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
