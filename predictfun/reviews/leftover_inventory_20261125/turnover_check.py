"""Post-hoc diagnostic on the frozen C anchors. No change to sizing or to the primary reading."""
import json
from collections import Counter
import numpy as np
from controller_module import controller as R


def tally(metric, anchors, model):
    """Reconstruct inventory at t by subtracting all post-t net fills from END inventory."""
    counts=Counter()
    end=metric['end_pos']
    for bid,ask in zip(anchors[::2],anchors[1::2],strict=True):
        assert (bid[0][0],ask[0][0])==(0,1)
        sells,buys=bid[1],ask[1]
        pos=end-(buys-sells)
        if abs(pos)<1e-7:
            continue
        counts['holding_anchors']+=1
        key,exits,_=bid if pos>0 else ask
        forecast,_=R.forecast(model,key)
        if forecast+1e-7<abs(pos):
            continue
        counts['projected_can_clear_current_holding']+=1
        gross_enough=exits+1e-7>=abs(pos)
        net_cleared=end*np.sign(pos)<=1e-7
        assert not net_cleared or gross_enough
        counts['gross_exits_enough']+=int(gross_enough)
        counts['same_direction_position_at_end']+=int(not net_cleared)
        counts['gross_exits_enough_and_same_direction_at_end']+=int(gross_enough and not net_cleared)
    return counts


def main():
    # Long 100, buys another 100, sells 120, ending long 80. Enough sell volume to
    # clear the old shares, yet further buying leaves an open long at expiry.
    k=[0,0,0,0]
    test=tally({'end_pos':80},[[k,120,None],[[1,0,0,0],100,None]],{R.groups(k)[0]:dict(markets=8,q25=120)})
    assert test['gross_exits_enough_and_same_direction_at_end']==1
    s=json.loads((R.HERE/'results_v2.json').read_text())
    fit=json.loads((R.HERE/'fit_v2.json').read_text())
    out=Counter()
    for file,h in s['inputs'].items():
        path=R.OUT/'eval'/file
        assert R.sha(path)==h
        v=json.loads(path.read_text())['arms']['C']
        out.update(tally(v['metrics'],v['anchors'],fit['cells']))
    R.write(R.HERE/'turnover_check.json',dict(observation='C 5-second anchors, repeat observations within market',counts=out))


if __name__=='__main__':
    main()
