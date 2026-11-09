"""Describe actual admission waits on the saved re-entry action population."""
import datetime as dt
import json
import subprocess
import pandas as pd
from . import pf_reentry_study as s


def waiting_times(actions):
    eligible=actions.admitted_at.notna()
    delta=actions.loc[eligible,'admitted_at']-actions.loc[eligible,'start']
    if (delta<0).any():raise ValueError('admission precedes opportunity')
    delays=delta[delta>0]
    return dict(actions=len(actions),admitted=int(eligible.sum()),delayed=len(delays),
        delay_ms_p50=float(delays.median()) if len(delays) else None,
        delay_ms_p90=float(delays.quantile(.9)) if len(delays) else None)


def main():
    if (s.DEST/'diagnostics.json').exists():raise ValueError('diagnostics exist')
    r=json.loads((s.DEST/'receipt.json').read_text())
    if s.a.sha(s.DEST/'actions.parquet')!=r['artifacts']['actions.parquet']:raise ValueError('action artifact changed')
    control=pd.DataFrame(dict(start=[0.,10.,20.],admitted_at=[50.,None,20.]))
    positive=waiting_times(control)==dict(actions=3,admitted=2,delayed=1,delay_ms_p50=50.,delay_ms_p90=50.)
    bad=control.copy();bad.loc[0,'admitted_at']=-1;refused=False
    try:waiting_times(bad)
    except ValueError:refused=True
    if not positive or not refused:raise ValueError('diagnostic controls fail')
    act=pd.read_parquet(s.DEST/'actions.parquet');summary={arm:waiting_times(g) for arm,g in act.groupby('arm')}
    for arm,q in summary.items():
        if q['delayed']!=r['summary'][arm]['delayed_actions']:raise ValueError('delay population differs')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.a.HERE,text=True).strip(),
        source_sha256=s.a.sha(s.a.HERE/'pf_reentry_diagnostics.py'),receipt_sha256=s.a.sha(s.DEST/'receipt.json'),
        controls=dict(positive=positive,negative_wait_refused=refused),summary=summary,
        interpretation='minimum elapsed confirmation, evaluated at original decision events; no new timer is introduced. Delay quantiles condition on delayed admissions and are not performance metrics.')
    s.a.write(s.DEST/'diagnostics.json',out);s.a.write(s.REVIEW/'diagnostics.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
