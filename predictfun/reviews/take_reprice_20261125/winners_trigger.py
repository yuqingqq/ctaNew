"""What triggers the takers who beat us? (USER 2026-11-25: "Proceed to find their signal"). Diagnostic on four consumed days (20261023,
20261031..1102: every venue present); no strategy change.

Stage `legs <day>`: every BTC 5m maker leg of the day (step1 legs_timed2) joined to its taker (REST match records: taker signer and order hash,
takers.taker_table, pf_step1's key); legs whose key maps to more than one taker are flagged `ambiguous` and kept out of per-taker results.
Output (write-once): OUT/winners_legs_<day>.parquet + OUT/winners_legs_<day>.json (counts).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/winners_trigger.py legs <day>
"""
import json
import resource
import sys
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import reprice_replay as RP  # noqa: E402
import takers as TK  # noqa: E402

S, OD, MM, TR, C = RP.S, RP.OD, RP.MM, RP.TR, RP.C
OUT = RP.OUT
DAYS = ('20261023', '20261031', '20261101', '20261102')


def legs(day):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    C.require(day in DAYS, f'{day} is not a declared day')
    fn, rc = OUT / f'winners_legs_{day}.parquet', OUT / f'winners_legs_{day}.json'
    C.require(not fn.exists() and not rc.exists(), f'{fn.name} exists (write-once)')
    L = pd.read_parquet(MM.S1 / day / 'legs_timed2.parquet', columns=TK.KEY + ['t_ms2', 'how2'])
    T = TK.taker_table(day)
    dup = T.duplicated(TK.KEY, keep=False)
    amb = T[dup].groupby(TK.KEY).taker.nunique()
    amb_keys = set(amb[amb > 1].index)
    T = T.drop_duplicates(TK.KEY)
    J = L.merge(T, on=TK.KEY, how='left')
    C.require(len(J) == len(L), 'the join duplicated legs')
    J['ambiguous'] = [k in amb_keys for k in zip(*(J[c] for c in TK.KEY))]
    J.to_parquet(fn, index=False)
    TR.dump(rc, dict(asof='2026-11-25', day=day, legs=len(J), joined=int(J.taker.notna().sum()), ambiguous_legs=int(J.ambiguous.sum()),
                     takers=int(J.taker.nunique()), orders=int(J.taker_hash.nunique())))
    print(day, len(J), int(J.taker.notna().sum()), int(J.ambiguous.sum()), flush=True)


if __name__ == '__main__':
    dict(legs=lambda: legs(sys.argv[2]))[sys.argv[1]]()
