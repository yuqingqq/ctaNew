"""Standalone comparison figure from the audited strategy scorecard."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'reviews/strategy_scorecard_20261116/results_v1'


def main():
    raw=(ROOT/'report.json').read_bytes()
    r=json.loads(raw);a=json.loads((ROOT/'audit_v1.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==a['report_sha256']
    assert a['fills_verified']==r['checks']['ledger_fills']
    names=['STOP_EV','ARRIVAL_EV','FUTURE_ENTRY'];labels=['STOP-EV entry + guard','Arrival forecast + STOP-EV guard','Future-informed entry (noncausal)']
    colors=['#2364aa','#c14953','#6b8e23'];tot=[r['totals'][n+'__recorded58'] for n in names]
    fields=['spread_capture_cents_per_share','markout_100_cents_per_share','markout_500_cents_per_share','markout_1000_cents_per_share','markout_5000_cents_per_share']
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.25,1]})
    for v,label,color in zip(tot,labels,colors):
        ax.plot(range(5),[v[f] for f in fields],marker='o',lw=2,color=color,label=label,linestyle='--' if 'noncausal' in label else '-')
    ax.axhline(0,color='#888888',lw=.8);ax.set_xticks(range(5),['Pre-fill\nexecution edge','+100ms','+500ms','+1s','+5s'])
    ax.set_ylabel('Cents per filled share');ax.set_title('Execution edge retained after fills');ax.grid(axis='y',alpha=.2)
    ax.legend(loc='best',fontsize=8,frameon=False)
    x=np.arange(3);w=.34
    bx.bar(x-w/2,[v['terminal_pnl_usd'] for v in tot],w,label='Marked at 60.2s',color='#587b95')
    bx.bar(x+w/2,[v['settlement_pnl_usd'] for v in tot],w,label='Held to settlement',color='#c7a76c')
    bx.axhline(0,color='#888888',lw=.8);bx.set_xticks(x,['STOP_EV','ARRIVAL_EV','FUTURE_ENTRY\n(noncausal)'])
    bx.set_ylabel('Gross simulated dollars');bx.set_title('Terminal value and settlement value');bx.grid(axis='y',alpha=.2);bx.legend(frameon=False,fontsize=8)
    for bars in bx.containers:bx.bar_label(bars,fmt='%+.0f',padding=3,fontsize=8)
    fig.suptitle('Same execution settings: 58ms placement / 37ms cancellation',fontsize=13)
    fig.text(.5,.01,'Five consumed dates · 1,434 first-minute windows per case · all fills · M50 / queue M · 50ms review\nSettlement holds residual inventory after 60.2s; fees/rewards not added. As of '+r['asof'][:19]+' UTC.',ha='center',fontsize=8,color='#444444')
    fig.tight_layout(rect=(0,.08,1,.93))
    fig.savefig(ROOT/'entry_comparison.png',dpi=160);fig.savefig(ROOT/'entry_comparison.svg')
    svg=ROOT/'entry_comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
