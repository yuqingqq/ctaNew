"""Plot verified paired trading-value/inventory comparisons, without new selection."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=Path('/home/yuqing/ctaNew/data/predict_fun/research/default_pull_20261111_v3')
REVIEW=HERE.parent/'reviews/default_pull_20261111_v3'
MODES={'default':('Default, no signal pull','#777777'),
       'pressure':('Pressure rules','#d99315'),'floor':('Guard floor','#c45e46'),
       'market':('Market + age model','#8c62aa'),'queue':('Queue model','#2878b5'),
       'queue_cost':('Queue model + pull margin','#309478')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(report):
    out=[]
    if report['stage']!='final' or report['validation'] is not False or len(report['days'])!=6:
        raise ValueError('plot requires complete exploratory six-day report')
    for baseline in ('reference','matched_guard'):
        for scenario in ('X','stress'):
            for mode in MODES:
                r=report['comparisons'][f'{scenario}:{mode}-{baseline}']
                values=[r['marked_delta'],r['exposure_pct']]+r['marked_day_interval']
                if r['windows']!=1728 or not 0<r['common_mark_windows']<=1728 or not np.isfinite(values).all():
                    raise ValueError('invalid plotted population/value')
                if r['marked_day_interval'][0]>r['marked_day_interval'][1]:raise ValueError('reversed interval')
                out.append(dict(baseline=baseline,scenario=scenario,mode=mode,windows=r['windows'],
                    common_marks=r['common_mark_windows'],marked_delta=r['marked_delta'],
                    exposure_pct=r['exposure_pct'],day_interval=r['marked_day_interval']))
    return out


def controls():
    r=dict(stage='final',validation=False,days=list(range(6)),comparisons={})
    for b in ('reference','matched_guard'):
        for s in ('X','stress'):
            for m in MODES:r['comparisons'][f'{s}:{m}-{b}']=dict(windows=1728,common_mark_windows=1700,marked_delta=1.,exposure_pct=-1.,marked_day_interval=[-2.,4.])
    if len(rows(r))!=24:raise ValueError('plot population positive control')
    r['comparisons']['X:queue-reference']['common_mark_windows']=1729
    try:rows(r)
    except ValueError:pass
    else:raise ValueError('bad plot population accepted')


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    controls()
    report=json.loads((OUT/'comparison_final.json').read_text())
    audit=json.loads((OUT/'completion_runtime_v3.json').read_text())
    if audit['status']!='PASS' or sha(OUT/'comparison_final.json')!=audit['comparison_sha']:
        raise ValueError('report is not the audited artifact')
    if sha(OUT/'completion_audit.json')!=audit['completion_audit_sha'] or sha(OUT/'runtime_parity.json')!=audit['runtime_parity_sha']:
        raise ValueError('audit chain changed')
    data=rows(report);targets=[REVIEW/f'tradeoff.{ext}' for ext in ('png','pdf')]+[REVIEW/'tradeoff_receipt.json']
    if any(p.exists() for p in targets):raise ValueError('refuse plot overwrite')
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,baseline,title in zip(axes,('reference','matched_guard'),('Against frozen ladder v3','Against matched guard strategy')):
        group=[r for r in data if r['baseline']==baseline]
        for mode,(label,color) in MODES.items():
            pair=[r for r in group if r['mode']==mode]
            ax.plot([r['exposure_pct'] for r in pair],[r['marked_delta'] for r in pair],color=color,alpha=.3,lw=1)
            for r in pair:
                x,y=r['exposure_pct'],r['marked_delta'];lo,hi=r['day_interval']
                ax.vlines(x,lo,hi,color=color,alpha=.42,lw=1.4)
                ax.scatter(x,y,marker='o' if r['scenario']=='X' else 'X',s=65,color=color,zorder=3,edgecolors='white',linewidths=.5)
        ax.axhline(0,color='#333333',lw=.8);ax.axvline(0,color='#333333',lw=.8)
        ax.grid(alpha=.15);ax.set_title(title,fontsize=12)
        ax.set_xlabel('Inventory-square time change (%)')
        ax.set_ylabel('Paired marked trading-value change ($)')
    handles=[Line2D([0],[0],color=c,marker='o',lw=0,label=n) for n,c in MODES.values()]
    handles += [Line2D([0],[0],color='#333333',marker='o',lw=0,label='Primary timing / queue M'),
                Line2D([0],[0],color='#333333',marker='X',lw=0,label='Slower timing / queue L')]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Quote by default, learn when to pull\nSix consumed dates — exploratory, not live validation',fontsize=14)
    for p in targets[:2]:fig.savefig(p,dpi=170)
    plt.close(fig)
    receipt=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha=sha(__file__),comparison_sha=sha(OUT/'comparison_final.json'),audit_sha=sha(OUT/'completion_runtime_v3.json'),
        artifacts={p.name:sha(p) for p in targets[:2]},rows=data,
        caption='Upper-left direction means more paired marked trading value with less inventory-square exposure. Lines connect primary/stress assumptions, not a time trend. Vertical ranges resample six UTC days; they do not adjust for selection on consumed history. Each point uses its own common-mark windows, with counts in this receipt and the full report. Inventory exposure is a risk proxy, not measured future P&L variance. Spread/settlement are separate tables; neither is added to marked value.',
        controls='All24 expected populations accepted; a planted impossible common-window count is refused.')
    with targets[2].open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print('VERIFIED_TRADEOFF_PLOT',targets[0],flush=True)


if __name__=='__main__':run()
