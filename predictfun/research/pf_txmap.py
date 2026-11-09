"""Group full transaction/taker identities; never infer identity from price/time/rank.

Run after identity-preserving step1. Legacy content-only inputs are refused.
This table groups events; it never supplies or combines matching timestamps.
"""
import sys
import pandas as pd
from . import pf_io as io
from .pf_trade_tape import transaction_groups

def main():
    day=sys.argv[1];root=f'{io.OUT}/step1/{day}'
    legs=pd.read_parquet(f'{root}/legs.parquet')
    groups=transaction_groups(legs)
    pd.DataFrame(dict(leg_id=legs.leg_id,tk=groups,tk_status=0)).to_parquet(f'{root}/legs_tk_identity.parquet',index=False)
    io.write_json(f'{root}/txmap_identity.json',dict(schema='full_transaction_taker_identity_v1',legs=len(legs),
        transaction_groups=int(len(set(groups))),missing_identities=0,clock_substitution=False))

if __name__=='__main__':main()
