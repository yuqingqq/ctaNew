"""Static scientific figures of the frozen exhaustive attribution; no new screen."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import common as C
def run():
 C.limits();r=json.loads((C.H/'readings.json').read_text());b=json.loads((C.H/'bounds.json').read_text());df=pd.read_parquet(C.OUT/'pairs.parquet');ex=pd.read_parquet(C.OUT/'exposure.parquet');vols=list(C.VOLS);colors=['#5799bf','#89ad8a','#d88761','#b6b6b6']
 def cell(asset,axis,role='all',day='all',clock='pre',scope='all19'):return next(x for x in r['attribution'] if(x['asset'],x['axis'],x['role'],x['day'],x['clock'],x['scope'])==(asset,axis,role,day,clock,scope))['rows']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(3,2,figsize=(14,12),layout='constrained');xx=np.arange(4)
 for j,a in enumerate(C.ASSETS):
  ax=axs[0,j]
  for off,axis,col in [(-.18,'entry','#447d9e'),(.18,'exit','#c47751')]:
   y=[v['net_c_per_share'] if v['net_c_per_share'] is not None else np.nan for v in cell(a,axis)];bars=ax.bar(xx+off,y,.34,label=axis.capitalize(),color=col);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
  ax.set_xticks(xx,vols);ax.axhline(0,color='#777',lw=.7);ax.set_ylabel('Net cents / matched share');ax.set_title(('Predict mid movement' if a=='pf' else 'Binance return movement')+' at entry vs exit');ax.legend()
 ac=[x for x in r['acquisition'] if(x['scope'],x['day'],x['clock'],x['asset'])==('all19','all','pre','pf')];ax=axs[1,0]
 for off,key,col in [(-.18,'completed_usd','#447d9e'),(.18,'residual_usd','#c47751')]:
  yy=[next(x[key] for x in ac if x['vol']==v) for v in vols];bars=ax.bar(xx+off,yy,.34,label=key.replace('_usd','').capitalize(),color=col);ax.bar_label(bars,fmt='%.0f',fontsize=8,padding=2)
 ax.set_xticks(xx,vols);ax.axhline(0,color='#777',lw=.7);ax.set_ylabel('USD');ax.set_title('Acquisition conditions: turnover and residual');ax.legend()
 a=df[(df.clock=='pre')&(df.asset=='pf')&(df.axis=='entry')];ax=axs[1,1];ss=['narrow','intermediate','wide'];matrix=np.full((3,3),np.nan);labels={}
 for i,v in enumerate(vols[:3]):
  for j,s in enumerate(ss):
   g=a[(a.vol==v)&(a.spread==s)];q=g.q.sum()
   if q:matrix[i,j]=g.net_c.sum()/q;labels[i,j]=f'{matrix[i,j]:+.2f}c\n{q:,.0f} sh'
 lim=max(1,np.nanmax(abs(matrix)));im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim)
 for (i,j),txt in labels.items():ax.text(j,i,txt,ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>lim*.5 else 'black')
 ax.set_xticks(range(3),['<=1c','1-4c','>4c']);ax.set_yticks(range(3),vols[:3]);ax.set_xlabel('Displayed spread at entry');ax.set_title('Entry joint cells: net c/share, matched shares');fig.colorbar(im,ax=ax,shrink=.65)
 ax=axs[2,0];matrix=np.zeros((3,3))
 for i,v in enumerate(vols[:3]):
  for j,w in enumerate(vols[:3]):matrix[i,j]=a[(a.entry_vol==v)&(a.exit_vol==w)].net_c.sum()/100
 lim=max(1,np.max(abs(matrix)));im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'${matrix[i,j]:+,.0f}',ha='center',va='center',color='white' if abs(matrix[i,j])>lim*.5 else 'black')
 ax.set_xticks(range(3),vols[:3]);ax.set_yticks(range(3),vols[:3]);ax.set_xlabel('Exit condition');ax.set_ylabel('Entry condition');ax.set_title('Completed net turnover by transition (USD)');fig.colorbar(im,ax=ax,shrink=.65)
 ax=axs[2,1];ee=ex[ex.asset=='pf'];ev=[ee.loc[ee.vol==v,'seconds'].sum()/ee.seconds.sum()*100 for v in vols];entry=cell('pf','entry');exit=cell('pf','exit')
 for off,label,vals,col in [(-.24,'Reference time',ev,'#9a9a9a'),(0,'Matched entry shares',[x['matched_share']*100 for x in entry],'#447d9e'),(.24,'Matched exit shares',[x['matched_share']*100 for x in exit],'#c47751')]:ax.bar(xx+off,vals,.23,label=label,color=col)
 ax.set_xticks(xx,vols);ax.set_ylabel('Share (%)');ax.set_title('Activity versus neutral reference time');ax.legend(fontsize=8)
 fig.suptitle('Maker 3 | calm vs volatile, separately from wide books\n19 selected complete markets, Oct 24-27 | pre-window clock | all completed roles',fontsize=16)
 fig.supxlabel(f"Calm / volatile cutoffs from neutral Oct24-25: Predict {b['pf']['q25']:.3f} / {b['pf']['q75']:.3f} cents RMS; BTC {b['btc']['q25']:.3f} / {b['btc']['q75']:.3f} bps RMS.\nTrailing10s; fees included. FIFO is accounting; sparse cells and execution-time uncertainty remain. Unknown transition rows omitted only from 3x3 plot; retained in tables.",fontsize=9)
 fig.savefig(C.H/'regimes.png',dpi=150);fig.savefig(C.H/'regimes.svg');plt.close(fig)
 C.dump(C.H/'headline.json',dict(bounds=b,pf_entry=cell('pf','entry'),pf_exit=cell('pf','exit'),btc_entry=cell('btc','entry'),btc_exit=cell('btc','exit'),passive_entry=cell('pf','entry','MM'),passive_exit=cell('pf','exit','MM'),acquisition=ac))
 print('figures and headline saved; no fitted rule')
if __name__=='__main__':run()
