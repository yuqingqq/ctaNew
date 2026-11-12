import pandas as pd, numpy as np, sys, datetime as dt
R='/home/yuqing/ctaNew/data/mm_hf/raw'; d=sys.argv[1]
d0=dt.datetime.strptime(d,'%Y%m%d')
hrs=[(d0+dt.timedelta(hours=h)).strftime('%Y%m%d_%H') for h in range(0,24)]
for s in ('bookTicker','trade'):
    prev=None; bad=[]
    for h in hrs:
        E=pd.read_csv(f'{R}/{s}/BTCUSDT/{h}.csv.gz',header=None,usecols=[1]).iloc[:,0].values
        if prev is not None and E.min()<prev: bad.append((h,int(prev-E.min()),int((E<prev).sum())))
        prev=E.max()
    print(d,s,'hour boundaries where the next file starts before the previous file ends (hour, ms overlap, rows):',bad)
