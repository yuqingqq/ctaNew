"""Own-order calibration ingestion; precise status clocks are not match clocks.

Reads an explicitly declared, closed capture only. No strategy simulation,
parameter fitting, exchange access, or inferred public-trade timestamps.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re

WALLET_CLOCK = re.compile(rb'^\{"recv_unix_ns":\s*([0-9]+),')
KERNEL_CLOCK = re.compile(rb'^\{"frame":\{"at":([0-9]+),')
FILL_TYPES = {'orderTransactionSubmitted': 'Matched', 'orderTransactionSuccess': 'Confirmed', 'orderTransactionFailed': 'Failed'}
ORDER_TYPES = {'orderAccepted', 'orderNotAccepted', 'orderExpired', 'orderCancelled'}
E18 = Decimal(10)**18
MILLION = Decimal(10)**6


def need(condition, message):
    if not condition: raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def signed_hash(value):
    if isinstance(value,list):
        need(len(value)==32 and all(type(x) is int and 0<=x<=255 for x in value),'invalid signed hash bytes')
        value='0x'+bytes(value).hex()
    need(isinstance(value,str) and re.fullmatch(r'0x[0-9a-fA-F]{64}',value) is not None,'invalid signed order hash')
    return value.lower()


def records(path,start,end,kind):
    """Reject out-of-scope event envelopes before decoding their payloads."""
    path=Path(path); opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as f:
        for line in f:
            if kind=='wallet' and b'"data":' in line:
                match=WALLET_CLOCK.match(line)
                need(match is not None,'unrecognized wallet timestamp envelope')
                need(start<=int(match[1])<end,'wallet payload outside declared time range')
            elif kind=='kernel':
                match=KERNEL_CLOCK.match(line)
                if match:
                    need(start<=int(match[1])<end,'kernel payload outside declared time range')
                else:
                    # Non-step journal header/footer is not an order event.
                    need(b'"kind":"step"' not in line and b'"type":"step"' not in line,'unrecognized kernel timestamp envelope')
            yield json.loads(line)


def kernel_actions(rows):
    orders={}; hashes={}; fills=[]; last=0
    for row in rows:
        kind=row.get('type',row.get('kind'))
        need(kind in ('step','header','outbox','market_ref'),'unknown kernel record kind')
        if kind!='step':continue
        frame=row.get('frame')
        need(isinstance(frame,dict),'kernel step missing frame')
        at=frame['at']; need(type(at) is int and at>=last,'kernel clock regressed');last=at
        inp=frame.get('input')
        if not isinstance(inp,dict):continue
        control=inp.get('Control',{}); ex=inp.get('Exec',{})
        if isinstance(control,dict) and 'Place' in control:
            p=control['Place'];cid=p['cid']
            need(cid not in orders,'duplicate client order action')
            need(p['side'] in ('BuyUp','BuyDown'),'unsupported order side')
            need(0<p['price']<1_000_000 and p['size']>0,'invalid order price/size')
            orders[cid]=dict(cid=cid,market_index=p['market'],side=p['side'],price_micros=p['price'],size_micros=p['size'],
                             placed_ns=at,signed_ns=None,accepted_ns=None,order_hash=None,terminal_ns=None)
        if not isinstance(ex,dict):continue
        if 'Signed' in ex:
            e=ex['Signed'];cid=e['cid'];h=signed_hash(e['order_id'])
            need(cid in orders,'signed order absent from complete action population')
            need(h not in hashes and orders[cid]['order_hash'] is None,'duplicate signed order identity')
            hashes[h]=cid;orders[cid].update(order_hash=h,signed_ns=at)
        if 'Accepted' in ex:
            cid=ex['Accepted']['cid'];need(cid in orders,'accepted order absent from action population')
            orders[cid]['accepted_ns']=orders[cid]['accepted_ns'] or at
        for key in ('UserCancel','CancelPendingAudit','CancelQuantityReconciled','CancelAuditComplete'):
            if key in ex:
                h=signed_hash(ex[key]['order_id'])
                if h in hashes:
                    o=orders[hashes[h]];o['terminal_ns']=o['terminal_ns'] or at
        if 'Fill' in ex:
            e=ex['Fill'];h=signed_hash(e['order_id'])
            fills.append(dict(order_hash=h,settlement_id=str(e['trade_id']),status=e['status'],receipt_ns=at,
                              reported_status_timestamp_ns=e.get('match_time'),
                              quantity=str(Decimal(e['size'])/MILLION),price=str(Decimal(e['price'])/MILLION)))
    need(bool(orders),'empty order-action population')
    return list(orders.values()),fills


def wallet_events(rows,start,end):
    events=[];sessions=[];footer=None;written=0;last=0
    for row in rows:
        need(footer is None,'records after wallet footer')
        if set(row)=={'written','dropped'}:
            need(type(row['written']) is int and type(row['dropped']) is int and row['dropped']>=0,'invalid wallet footer')
            need(row['written']==written,'wallet footer count mismatch');footer=row;continue
        written+=1;at=row.get('recv_unix_ns')
        need(type(at) is int and start<=at<end and at>=last,'wallet receipt clock out of scope or regressed');last=at
        if 'session' in row:
            need(row['session'] in ('connected','live','not_live','ended'),'unknown wallet session marker')
            sessions.append(row);continue
        d=row.get('data');need(isinstance(d,dict),'wallet event missing data')
        ty=d.get('type');need(ty in ORDER_TYPES|FILL_TYPES.keys(),'unknown wallet event type')
        h=signed_hash(d.get('orderHash'));ts=d.get('timestamp')
        need(type(ts) is int and ts>=0,'invalid venue event timestamp')
        details=d.get('details');need(isinstance(details,dict),'wallet event missing order details')
        need(isinstance(d.get('orderId'),str) and d['orderId'],'wallet order id missing')
        event=dict(order_hash=h,venue_order_id=d['orderId'],type=ty,event_timestamp_ms=ts,receipt_ns=at,
                   receipt_minus_event_ms=str(Decimal(at)/Decimal(1_000_000)-ts),
                   market_id=details['marketId'],quantity=str(Decimal(details['quantity'])),order_price=str(Decimal(details['price'])))
        if ty in FILL_TYPES:
            fill=d['fill'];need(isinstance(d.get('settlementId'),str) and d['settlementId'],'fill without settlement identity')
            event.update(settlement_id=d['settlementId'],status=FILL_TYPES[ty],
                         fill_quantity=str(Decimal(fill['executedSizeWei'])/E18),fill_price=str(Decimal(fill['executedPriceWei'])/E18))
            need(Decimal(event['fill_quantity'])>0,'nonpositive fill quantity')
        events.append(event)
    return events,sessions,footer


def reconcile(orders,kernel_fills,events,sessions,footer):
    by_hash={o['order_hash']:o for o in orders if o['order_hash'] is not None}
    need(len(by_hash)==sum(o['order_hash'] is not None for o in orders),'duplicate order-action hash')
    wallet=defaultdict(list);unknown=[]
    for e in events:
        if e['order_hash'] not in by_hash:unknown.append(e);continue
        o=by_hash[e['order_hash']]
        token=Decimal(o['price_micros'])/MILLION
        if o['side']=='BuyDown':token=1-token
        need(Decimal(e['quantity'])==Decimal(o['size_micros'])/MILLION and Decimal(e['order_price'])==token,'wallet/kernel order price or size differs')
        need(e['receipt_ns']>=o['placed_ns'],'wallet receipt precedes order action')
        wallet[e['order_hash']].append(e)
    grouped=defaultdict(list)
    for e in events:
        if 'settlement_id' in e:grouped[e['order_hash'],e['settlement_id']].append(e)
    fills=[]
    for (h,sid),group in grouped.items():
        need(len({(Decimal(e['fill_quantity']),Decimal(e['fill_price'])) for e in group})==1,'settlement quantity/price changed')
        statuses={status:[e for e in group if e['status']==status] for status in ('Matched','Confirmed','Failed')}
        clocks={status:dict(count=len(v),event_timestamps_ms=sorted({e['event_timestamp_ms'] for e in v}),
                           first_receipt_ns=min((e['receipt_ns'] for e in v),default=None)) for status,v in statuses.items()}
        k=[x for x in kernel_fills if x['order_hash']==h and x['settlement_id']==sid]
        need(all(Decimal(x['quantity'])==Decimal(group[0]['fill_quantity']) and Decimal(x['price'])==Decimal(group[0]['fill_price']) for x in k),'kernel/wallet fill quantity or price differs')
        issues=[]
        if h not in by_hash:issues.append('not_in_declared_action_population')
        if not statuses['Matched']:issues.append('missing_submitted_status')
        if statuses['Failed']:issues.append('failed_or_reversed_settlement')
        if not statuses['Confirmed']:issues.append('not_confirmed')
        if not k:issues.append('missing_kernel_fill')
        if h in by_hash and statuses['Matched']:
            o=by_hash[h]
            hi=min(o['terminal_ns'] or 2**63,clocks['Matched']['first_receipt_ns'])
            if any(not o['placed_ns']<=ts*1_000_000<=hi for ts in clocks['Matched']['event_timestamps_ms']):
                issues.append('submitted_timestamp_outside_local_lifecycle')
        fills.append(dict(order_hash=h,settlement_id=sid,quantity=group[0]['fill_quantity'],price=group[0]['fill_price'],
                          status_clocks=clocks,kernel_notices=k,kernel_notice_count=len(k),issues=issues,execution_time_verified=False))
    # Never classify a missing notice as a verified unfilled order when the tap
    # dropped records, is unfinished, or has any reconnect/liveness uncertainty.
    connected=sum(s['session']=='connected' for s in sessions)
    ends=[s for s in sessions if s['session']=='ended']
    clean=(footer is not None and footer['dropped']==0 and connected==1 and len(ends)==1 and ends[0].get('why')=='stopped'
           and any(s['session']=='live' for s in sessions) and not any(s['session']=='not_live' for s in sessions))
    actions=[]
    connected_at=next((s['recv_unix_ns'] for s in sessions if s['session']=='connected'),None)
    ended_at=ends[0]['recv_unix_ns'] if len(ends)==1 else None
    for o in orders:
        es=wallet[o['order_hash']];fs=[f for f in fills if f['order_hash']==o['order_hash']]
        confirmed=sum((Decimal(f['quantity']) for f in fs if f['status_clocks']['Confirmed']['count'] and not f['status_clocks']['Failed']['count']),Decimal(0))
        need(confirmed<=Decimal(o['size_micros'])/MILLION,'confirmed fills exceed submitted order size')
        terminal=any(e['type'] in ('orderCancelled','orderExpired','orderNotAccepted') for e in es)
        action_covered=clean and connected_at<=o['placed_ns'] and all(e['receipt_ns']<=ended_at for e in es)
        status=('not_signed' if o['order_hash'] is None else 'capture_incomplete' if not action_covered else
                'fill_identity_or_status_issue' if any(f['issues'] for f in fs) else 'confirmed_filled' if confirmed>0 else
                'kernel_fill_without_wallet_leg' if any(k['order_hash']==o['order_hash'] for k in kernel_fills) else
                'observed_terminal_without_fill' if terminal else 'open_or_missing_wallet_terminal')
        actions.append(dict(**o,calibration_status=status,confirmed_shares=str(confirmed),wallet_records=len(es)))
    unknown_kernel=[k for k in kernel_fills if k['order_hash'] not in by_hash or (k['order_hash'],k['settlement_id']) not in grouped]
    return dict(actions=actions,fills=fills,unknown_wallet_events=unknown,unmatched_kernel_fills=unknown_kernel,
                summary=dict(order_actions=len(actions),deduplicated_fill_legs=len(fills),unknown_wallet_events=len(unknown),
                             unmatched_kernel_fill_notices=len(unknown_kernel),
                             action_status_counts=dict(Counter(a['calibration_status'] for a in actions)),wallet_capture_complete=clean,
                             wallet_footer=footer,execution_timing_calibrated=False,queue_model_calibrated=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();scope=json.loads(Path(args.scope).read_text())
    start,end=scope['start_ns'],scope['end_ns'];need(type(start) is int and type(end) is int and start<end,'invalid scope')
    # Preserve the existing forward block. An explicit later change of scope
    # requires its own reviewed declaration; this tool has no bypass switch.
    protected_start,protected_end=1794182400000000000,1794614400000000000
    need(end<=protected_start or start>=protected_end,'scope intersects protected Nov9–13 forward block')
    output=Path(args.out);need(not output.exists(),'refuse existing output destination')
    for name in ('kernel','wallet'):
        p=Path(scope[name]['path']);need(p.is_file(),f'missing {name} capture')
        need(digest(p)==scope[name]['sha256'],f'{name} capture hash mismatch')
    orders,kernel=kernel_actions(records(scope['kernel']['path'],start,end,'kernel'))
    events,sessions,footer=wallet_events(records(scope['wallet']['path'],start,end,'wallet'),start,end)
    result=reconcile(orders,kernel,events,sessions,footer);output.mkdir(parents=True)
    for name in ('actions','fills','unknown_wallet_events','unmatched_kernel_fills'):
        with (output/f'{name}.jsonl').open('x') as f:
            for row in result[name]:f.write(json.dumps(row,separators=(',',':'))+'\n')
    receipt=dict(scope=scope,scope_sha256=digest(args.scope),source_sha256=digest(__file__),summary=result['summary'],
                 limitations=['Receipt minus event time combines source processing and cross-clock offset; not pure network or matching latency.',
                              'Lifecycle compatibility is necessary only. Venue Submitted timestamp is not automatically an exchange matching timestamp.',
                              'No queue fit or strategy replay. All order actions retained with explicit coverage/status issues.'],
                 artifacts={p.name:digest(p) for p in output.glob('*.jsonl')})
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
