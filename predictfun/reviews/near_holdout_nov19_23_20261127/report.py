"""Render the frozen test screens and freeze a digest of the inputs used for scoring."""
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'data/predict_fun/research/near_holdout_nov19_23'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def usd(value):
    return f"${value:,.2f}" if value>=0 else f"−${-value:,.2f}"


def main():
    receipt=json.loads((HERE/'result.json').read_text())
    for name,expected in receipt['hashes'].items():
        assert digest(HERE/name)==expected
    days=sorted(receipt['coverage'])
    coverage=receipt['coverage']
    complete=sum(c.get('complete',0) for c in coverage.values())
    lines=['# Frozen passive inventory rule on November 19–23, 2026',
           '',f"All five test dates were assigned before reading outcomes. The census includes {complete} complete market paths from 1,440 scheduled windows.",
           '', '## Five-day decision','']
    panels=[]
    check_labels={
        'positive':'positive total settlement','without_best_market':'positive without best market','without_best_day':'positive without best day',
        'majority':'profitable on a majority of dates','lower_average_inventory':'lower mean absolute ending position',
        'lower_tail_inventory':'lower worst-decile ending position','no_worse_tail_loss':'no increase in worst-decile settlement loss',
        'no_worse_settlement':'settlement at least as high as original','no_fewer_winning_days':'at least as many profitable dates as original',
        'beats_time_only':'higher settlement than the midpoint time-only control'}
    if (HERE/'comparison.csv').exists():
        p=pd.read_csv(HERE/'comparison.csv',float_precision='round_trip',dtype={'period':str})
        for arm in ('nominal','short'):
            chk=receipt['checks'][arm]
            absolute=all(chk[k] for k in ('positive','without_best_market','without_best_day','majority'))
            comparative=all(chk[k] for k in chk if k not in ('positive','without_best_market','without_best_day','majority'))
            lines.append(f"- {arm.capitalize()} execution. Absolute profit screen {'passed' if absolute else 'failed'}. Comparative screen {'passed' if comparative else 'failed'}.")
            fail=[label for k,label in check_labels.items() if not chk[k]]
            if fail:
                lines.append('  Failed criteria were '+', '.join(fail)+'.')
            for r in p[(p.arm==arm)&(p.period=='pooled')].itertuples():
                panels.append('| '+' | '.join([arm,r.policy,usd(r.turnover),usd(r.residual),usd(r.settle),
                   str(r.positive_days)+'/5',f'{r.mean_end:.2f}',f'{r.tail_end:.2f}',usd(r.tail_loss),f'{r.closed:,.0f}'])+' |')
        lines.extend(['','| Timing | Policy | FIFO | Residual at payout | Settlement | Positive dates | Mean abs end, shares | Worst 10% abs end, shares | Worst 10% mean loss | Matched shares |',
                      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',*panels,
                      '','| UTC date | Eligible / 288 | Original nominal | Time-only nominal | Near nominal | Original short | Time-only short | Near short |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for d in days:
            by=p[p.period==d].set_index(['arm','policy'])
            vals=[usd(by.loc[(a,k),'settle']) if (a,k) in by.index else 'unavailable' for a in ('nominal','short') for k in ('original','time_only','near')]
            lines.append('| '+' | '.join([d,str(coverage[d].get('complete',0)),*vals])+' |')
    lines.extend(['',f"Coverage screen {'passed' if receipt['sufficient'] else 'insufficient'}. Overall predeclared decision: **{'pass' if receipt['passes'] else 'insufficient coverage' if not receipt['sufficient'] else 'did not pass'}**.",
                  '','## Coverage, execution and verification',''])
    for d in days:
        lines.append('- '+d+' — '+', '.join(f'{key} {value}' for key,value in coverage[d].items())+'.')
    lines.extend(['','Economics apply to eligible tapes and include zero-fill markets. Missing outcomes carry unknown P&L. Reported dollars use zero maker fees/rebates, fixed external flow, modeled queue and the recorded YES payout including ties. Nominal placement takes 58 ms with 37 ms live cancellation. Short eligibility suppresses matches after a cancel request while retaining nominal reservation and notice timing. Neither profile measures real wallet fills. The absolute screen removes the best day and best market separately. The tail mean uses the largest ceil(n/10) positions or losses in each pooled policy.',
                 '', 'The October 26 control reproduced every census status, all 70 tape identities, timing calibration, BTC grid and intraday pressure series, and the original and selected policy ledgers on the first eligible market. The readback checks pair membership, the inventory cap, FIFO plus residual at settlement, and a corrupted residual. The UTC day is the unit for interpreting signs. The December 1–5 prospective test is separate.',
                 '', '[Frozen protocol](protocol.md) · [Per-day coverage and computed checks](result.json) · [Input hashes](inputs.json)'])
    files={}
    for day in days:
        folder=OUT/'step1'/day
        paths=[folder/f for f in ('registry.parquet','legs_timed2.parquet','legs_tk.parquet')]
        paths += sorted((folder/'books').glob('part_*.parquet'))
        paths += [OUT/f'grid_{day}.json',OUT/f'pressure_{day}.json',OUT/'btc'/f'btc_{day}.parquet',OUT/f'pressure_{day}.parquet']
        files.update({str(path.relative_to(OUT)):digest(path) for path in paths})
    files['btc/btc_20261118.parquet']=digest(OUT/'btc/btc_20261118.parquet')
    (HERE/'inputs.json').write_text(json.dumps(files,indent=2)+'\n')
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
