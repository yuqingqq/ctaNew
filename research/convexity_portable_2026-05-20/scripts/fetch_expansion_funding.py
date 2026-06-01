"""Fetch funding rates 2021-01..2026-05 for the 106 expansion Binance symbols (V0 BASE features need funding)."""
from __future__ import annotations
import json, time
from pathlib import Path
from data_collectors.funding_rate_loader import load_funding_rate
SYMS=sorted(set(json.loads(Path("/tmp/addable.json").read_text()).values()))
def main():
    t0=time.time(); print(f"funding {len(SYMS)} expansion syms 2021-01..2026-05",flush=True)
    ok=0
    for i,s in enumerate(SYMS,1):
        try:
            df=load_funding_rate(s,start_month="2021-01",end_month="2026-05"); ok+=(df is not None and len(df)>0)
            if i%20==0: print(f"  {i}/{len(SYMS)} [{time.time()-t0:.0f}s]",flush=True)
        except Exception as e: print(f"  {s} ERR {str(e)[:50]}",flush=True)
    print(f"DONE funding {ok}/{len(SYMS)} [{time.time()-t0:.0f}s]",flush=True)
if __name__=="__main__": main()
