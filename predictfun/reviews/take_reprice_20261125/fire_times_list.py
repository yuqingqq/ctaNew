"""The top taker's fire times, nothing else (USER 2026-11-26: "Just list the top takers fires with time, and the metrics we need from other
exchanges orderbooks"). From smart_fires_export's top_taker_fires: fire_id, utc_time, t_ms (arrival at Predict, estimate), bet (UP / DOWN).
Output (write-once): smart_fires/top_taker_fire_times.csv.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_times_list.py [<rehearsal dir>]
"""
import sys
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import winners_trigger as WT  # noqa: E402

C = WT.C


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'smart_fires/top_taker_fire_times.csv'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    X = pd.read_csv(WT.OUT / 'smart_fires_export/top_taker_fires.csv.gz', usecols=['fire_id', 'utc_time', 't_ms', 'bet'])
    C.require(len(X) == 5930 and X.fire_id.is_unique and X.bet.isin(['UP', 'DOWN']).all(), 'not the 5,930 fires')
    X['t_ms'] = X.t_ms.round().astype('int64')
    X = X.sort_values('t_ms', kind='stable')
    fn.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(fn, index=False)
    print(fn, len(X), fn.stat().st_size, 'bytes', X.utc_time.iloc[0], '->', X.utc_time.iloc[-1])


if __name__ == '__main__':
    main()
