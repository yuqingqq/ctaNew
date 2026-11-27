"""Read back the ledgers, compare to the control receipt, and append the held-out-from-summary development date."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def close(left, right):
    if not math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-6):
        raise ValueError((left, right))


def ledger(row):
    close(float(row['turnover']) + float(row['residual']), row['settle'])
    close(2 * float(row['closed']) + float(row['abs_end']), row['shares'])


def main():
    rows = read(HERE / 'markets.csv')
    control = read(HERE.parent / 'urgency_cap150_eight_dates_20261126/markets_20261026.csv')
    keys = {(r['mid'],r['tape_id']) for r in control}
    ledger(rows[0])
    try:
        ledger(dict(rows[0], settle=float(rows[0]['settle']) + 1))
    except ValueError:
        pass
    else:
        raise AssertionError('changed settlement accepted')
    assert len(keys) == len(control) == 70 and len(rows) == 4 * len(keys)
    assert all(r['day'] == '20261026' for r in rows)
    previous = HERE.parent / 'urgency_far_extension_20261127/comparison.csv'
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == '1a63447d4d42fde92026e1cc582aa0f664e3579db003fb1de1557366c128bb8e'
    eight = {(r['arm'], r['policy']): r for r in read(previous) if r['period'] == 'all8'}
    oct26 = {(r['arm'],r['policy']):r for r in read(HERE / 'comparison.csv')}
    all_days = []
    for arm in ('nominal','short'):
        for policy in ('near','original'):
            sample = [r for r in rows if r['arm'] == arm and r['policy'] == policy]
            assert len(sample) == len(keys) and {(r['mid'],r['tape_id']) for r in sample} == keys
            for row in sample:
                ledger(row)
                assert float(row['peak_position']) <= 150 + 1e-6
                if arm == 'short':
                    close(row['post_request_shares'], 0)
            metrics = oct26[arm,policy]
            prior = eight[arm,'previous_btc' if policy == 'near' else policy]
            n = len(sample) + int(prior['markets'])
            combined = dict(arm=arm, policy=policy, markets=n)
            for column in ('settle','turnover','residual','closed','shares'):
                value = sum(float(r[column]) for r in sample)
                close(value, metrics[column])
                combined[column] = value + float(prior[column])
            exposure = sum(float(r['abs_end']) for r in sample)
            close(exposure / len(sample), metrics['mean_end'])
            combined['mean_end'] = (exposure + float(prior['mean_end']) * int(prior['markets'])) / n
            combined['positive_days'] = int(prior['positive_days']) + int(float(metrics['settle']) > 0)
            all_days.append(combined)
    (HERE / 'nine_days.json').write_text(json.dumps(all_days, indent=2) + '\n')
    print(json.dumps(all_days, indent=2))


if __name__ == '__main__':
    main()
