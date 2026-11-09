"""Identity-preserving trade tape. Confirmation and status-event clocks are not match clocks.

All source times are integer nanoseconds. A coarse status timestamp stays an
interval; price/size proximity, book changes and other legs never supply a point.
"""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'predict_trade_identity_v1'
HASH = re.compile(r'0x[0-9a-f]{64}\Z')
E18 = 10 ** 18


def hash32(value):
    if not isinstance(value, str) or not HASH.fullmatch(value.lower()):
        raise ValueError('missing or malformed full order/transaction hash')
    return value.lower()


def iso_ns(value):
    t = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.utcoffset() is None:
        raise ValueError('timestamp lacks UTC offset')
    d = t - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (d.days * 86400 + d.seconds) * 10**9 + d.microseconds * 1000


def rest_leg(trade, maker, recv_ns):
    tx = hash32(trade['transactionHash']); taker = hash32(trade['taker']['hash']); order = hash32(maker['hash'])
    index = maker['outcome']['indexSet']; quote = maker['quoteType']
    if index not in (1, 2) or quote not in ('Bid', 'Ask'):
        raise ValueError('unknown outcome/quote coordinate')
    if trade['taker']['outcome']['indexSet'] not in (1,2) or trade['taker']['quoteType'] not in ('Bid','Ask'):
        raise ValueError('unknown taker coordinate')
    amount = int(maker['amount']); price = int(maker['price'])
    if amount <= 0 or not 0 < price < E18 or type(recv_ns) is not int or recv_ns <= 0:
        raise ValueError('invalid trade amount, price or receipt')
    yes_price = price if index == 1 else E18 - price
    if yes_price % (E18 // 100):
        raise ValueError('maker price is not on the supported cent grid')
    side = 0 if (index == 1) == (quote == 'Bid') else 1
    return dict(schema=SCHEMA, leg_id=tx+'|'+taker+'|'+order, transaction_hash=tx,
        taker_order_hash=taker, maker_order_hash=order, settlement_id=trade.get('settlementId'),
        mid=int(trade['market']['id']), side=side, ptick=yes_price//(E18//100),
        shares=amount/E18, amount_wei=str(amount), token_price_wei=str(price),
        price_executed=int(trade['priceExecuted'])/E18, maker_signer=maker['signer'].lower(),
        taker_index=int(trade['taker']['outcome']['indexSet']),
        taker_quote=0 if trade['taker']['quoteType']=='Bid' else 1,
        confirmation_raw=trade['executedAt'], confirmation_ns=iso_ns(trade['executedAt']),
        first_recv_ns=recv_ns, last_recv_ns=recv_ns, observation_count=1)


def merge_leg(old, new):
    """Repeated REST pages cannot change economics or invent a new fill."""
    mutable = {'first_recv_ns','last_recv_ns','observation_count','settlement_id'}
    if any(old[k] != new[k] for k in old if k not in mutable):
        raise ValueError('conflicting observations of an exact trade leg')
    if old['settlement_id'] and new['settlement_id'] and old['settlement_id'] != new['settlement_id']:
        raise ValueError('conflicting settlement identity')
    old['settlement_id'] = old['settlement_id'] or new['settlement_id']
    old['first_recv_ns'] = min(old['first_recv_ns'],new['first_recv_ns'])
    old['last_recv_ns'] = max(old['last_recv_ns'],new['last_recv_ns'])
    old['observation_count'] += new['observation_count']


def status_event(order_hash, trade_id, status, source_ns, recv_ns, size_micros, price_micros):
    hash32(order_hash)
    if not trade_id or status not in ('Matched','Confirmed','Failed','Retrying'):
        raise ValueError('invalid trade status identity')
    if any(type(v) is not int or v < 0 for v in [source_ns,recv_ns,size_micros,price_micros]):
        raise ValueError('status event requires integer source and receipt times')
    if source_ns % 10**9:
        raise ValueError('kernel match_time is expected to be truncated seconds')
    return dict(maker_order_hash=order_hash,trade_id=trade_id,status=status,
        source_lo_ns=source_ns,source_hi_ns=source_ns+10**9,receipt_ns=recv_ns,
        quantity_micros=size_micros,token_price_micros=price_micros,
        timestamp_kind='status_event_floor_second')


def resolve_status(events):
    """One exact leg's status history. Never replace Matched with Confirmed."""
    if not events:
        return dict(clock_status='missing_matched_event',match_lo_ns=None,match_hi_ns=None,
                    matched_notice_ns=None,confirmed_notice_ns=None,status_changed_ns=None)
    if len({(e['maker_order_hash'],e['trade_id']) for e in events}) != 1:
        raise ValueError('mixed identities in status history')
    if len({(e['quantity_micros'],e['token_price_micros']) for e in events}) != 1:
        raise ValueError('changing economics in status history')
    if any(e['timestamp_kind'] != 'status_event_floor_second' or e['source_hi_ns']-e['source_lo_ns'] != 10**9 for e in events):
        raise ValueError('unsupported status clock precision')
    matched = [e for e in events if e['status']=='Matched']
    confirmed = [e for e in events if e['status']=='Confirmed']
    failed = [e for e in events if e['status']=='Failed']
    spans = {(e['source_lo_ns'],e['source_hi_ns']) for e in matched}
    kind = ('conflicting_terminal_status' if failed and confirmed else 'failed_trade' if failed else
            'conflicting_matched_event' if len(spans)>1 else 'matched_status_interval' if spans else 'missing_matched_event')
    lo,hi = next(iter(spans)) if len(spans)==1 else (None,None)
    return dict(clock_status=kind,match_lo_ns=lo,match_hi_ns=hi,
        matched_notice_ns=min((e['receipt_ns'] for e in matched),default=None),
        confirmed_notice_ns=min((e['receipt_ns'] for e in confirmed),default=None),
        status_changed_ns=(min(e['source_lo_ns'] for e in confirmed)-lo) if confirmed and lo is not None else None)


def attach_status(legs, events):
    """Exact hash + settlement ID only. No propagation to other maker legs."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for e in events:by_key[(e['maker_order_hash'],e['trade_id'])].append(e)
    used = set();out = []
    for leg in legs:
        k=(leg['maker_order_hash'],leg['settlement_id'] or leg['transaction_hash'])
        ev=by_key.get(k,[])
        if ev:
            if k in used:raise ValueError('one status identity maps to multiple REST legs')
            used.add(k)
            for e in ev:
                # Venue fixed point truncation can differ by up to one microshare.
                if abs(e['quantity_micros']*10**12-int(leg['amount_wei']))>10**12:
                    raise ValueError('exact status/REST quantity mismatch')
                if e['token_price_micros']*10**12!=int(leg['token_price_wei']):
                    raise ValueError('exact status/REST token price mismatch')
        out.append(dict(leg,**resolve_status(ev)))
    return out,used


def transaction_groups(frame):
    """Group full transaction AND taker identities without modifying any clock."""
    import pandas as pd
    required={'transaction_hash','taker_order_hash','maker_order_hash','leg_id'}
    if not required<=set(frame.columns):raise ValueError('legacy tape lacks full identities; rebuild from REST')
    if frame.leg_id.duplicated().any():raise ValueError('duplicate exact leg')
    for col in required-{'leg_id'}:
        for h in frame[col].unique():hash32(h)
    expected=frame.transaction_hash+'|'+frame.taker_order_hash+'|'+frame.maker_order_hash
    if not (expected==frame.leg_id).all():raise ValueError('leg ID does not match full identities')
    key=pd.MultiIndex.from_frame(frame[['transaction_hash','taker_order_hash']])
    return pd.factorize(key,sort=False)[0]


def replay_times(frame):
    """Point replay requires actual match events. A status interval is insufficient.

    An eventual matching-engine event adapter must provide match_time_ns and its
    precision/provenance. The current wallet/REST adapters deliberately cannot.
    """
    import numpy as np
    transaction_groups(frame)
    if not len(frame):raise ValueError('empty trade population')
    if 'clock_status' not in frame:raise ValueError('legacy heuristic clock is inadmissible')
    bad=frame.clock_status!='matching_engine_event'
    if bad.any():
        raise ValueError('point replay requires match-event timestamps; unresolved population '+str(frame.loc[bad,'clock_status'].value_counts().to_dict()))
    required={'match_time_ns','match_precision_ns','match_event_id','match_source_sha256'}
    if not required<=set(frame.columns):raise ValueError('match event provenance/precision is absent')
    for x in frame.itertuples():
        if not isinstance(x.match_time_ns,(int,np.integer)) or x.match_time_ns<=0 or not 0<x.match_precision_ns<=10**6:
            raise ValueError('point replay cannot round or sample a coarse timestamp')
        if not x.match_event_id or not re.fullmatch('[0-9a-f]{64}',x.match_source_sha256):
            raise ValueError('missing match event provenance')
    # An atomic transaction has one authoritative match time; disagreement is not min/median-resolved.
    g=transaction_groups(frame)
    if (frame.groupby(g).match_time_ns.nunique()>1).any():raise ValueError('conflicting exact transaction clocks')
    return frame.match_time_ns.to_numpy(dtype='int64')/10**6


def load_replay_tape(directory,day):
    import pandas as pd
    p=Path(directory);m=json.loads((p/'manifest.json').read_text())
    if m.get('schema')!=SCHEMA or m.get('day')!=day:raise ValueError('wrong tape schema or day')
    path=p/'legs.parquet'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=m['legs_sha256']:raise ValueError('trade tape hash mismatch')
    f=pd.read_parquet(path)
    if len(f)!=m['legs'] or sorted(map(int,f.mid.unique()))!=m['markets']:raise ValueError('trade population differs from manifest')
    t=replay_times(f)
    return f.assign(t_ms=t,how=1),dict(schema=SCHEMA,legs=len(f),manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest())


def refuse_legacy_clock():
    raise SystemExit('REFUSED: REST confirmation time, nearest pending increments and E2 book drops are not match clocks. '
                     'Use pf_trade_tape_build and pf_step4_ladder_identity; frozen historical reproduction remains on the prior commit.')
