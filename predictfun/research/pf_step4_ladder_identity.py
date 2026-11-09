"""Research replay entrypoint with identity/clock preflight; frozen v3 remains intact.

PF_TRADE_TAPE points to a pf_trade_tape manifest directory. Missing/coarse match
times stop before market/model data or replay output are touched.
"""
import hashlib
import os
from pathlib import Path
import sys
from .pf_trade_tape import load_replay_tape

ENGINE_SHA='d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17'
HERE=Path(__file__).parent


def engine_source():
    raw=(HERE/'pf_step4_ladder_v3.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ENGINE_SHA:raise ValueError('frozen engine source changed')
    s=raw.decode()
    reg="reg = pd.read_parquet(f'{S1}/registry.parquet').set_index('mid')"
    if s.count(reg)!=1:raise ValueError('engine registry anchor changed')
    s=s.replace(reg,reg+"\nif set(reg.index) != set(IDENTITY_TAPE[0].mid):\n    raise ValueError('registry and exact trade tape market populations differ')")
    start=s.index("GID = None\nTIMING = ")
    end=s.index('LT_BY = {m: g for m, g in LT.groupby',start)
    s=s[:start]+("GID = None\nTIMING = 'identity'\nLT, TXSTAT = IDENTITY_TAPE\n"
        "from .pf_trade_windows import select_trade_windows\n"
        "LT, TXSTAT['window_population'] = select_trade_windows(LT, reg)\n")+s[end:]
    start=s.index('if TXJIT:                                            # within-frame match time')
    end=s.index('# ---- Binance futures: knowledge time',start)
    s=s[:start]+s[end:]
    old="tt = np.where(g.how.values > 0, g.t_ms.values, g.exec_s.values * 1000.0 + shift)"
    if s.count(old)!=1:raise ValueError('engine trade-time anchor changed')
    return s.replace(old,'tt = g.t_ms.values')


def main():
    if len(sys.argv)!=4:raise SystemExit('usage: pf_step4_ladder_identity DAY SCENARIO ARMSET')
    for name in ['PF_TXTIME','PF_TXKEY','PF_TXJIT','PF_TXSHIFT','PF_TXSHIFT_E2']:
        if os.environ.get(name) not in (None,'','0'):raise SystemExit('REFUSED: heuristic clock option '+name)
    path=os.environ.get('PF_TRADE_TAPE')
    if not path:raise SystemExit('REFUSED: PF_TRADE_TAPE is required; no legacy fallback')
    try:tape=load_replay_tape(path,sys.argv[1])
    except (ValueError,KeyError,FileNotFoundError) as e:raise SystemExit('REFUSED: '+str(e)) from e
    os.environ['PF_TIMING']='identity'
    ns=dict(__name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),IDENTITY_TAPE=tape)
    exec(compile(engine_source(),'identity_clock_engine','exec'),ns)


if __name__=='__main__':main()
