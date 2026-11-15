"""Conservative timing certificates for the uncapped neutral STOP-EV path.

Opposite-side trades cannot change focal-side state in this specific engine
with no cap, learned policy or inventory input. On the focal side, an uncertain
event is harmless only if no accepted order can be reached anywhere in its
possible time interval before the label exposure ends. This covers queue drain
as well as fills. Unknown timestamps use unbounded intervals, never empirical
quantiles. All missing source records remain in the checked source ledger.

Proof: before the earliest possible uncertain state-changing event, the replay
states are identical by induction. A same-side event cannot change any queue or
fill when no nominal live order is reachable. Opposite-side events are isolated
under the stated reference. Multiple certified no-ops therefore remain no-ops.
No reset/recovery after a relevant uncertain event is assumed.
"""
import copy
from dataclasses import dataclass, replace
import math

import numpy as np

import pf_stop_ev_accounting as A

CONTRACT = 'stop-ev-side-prefix-invariance-v1'


@dataclass(frozen=True)
class MissingInfo:
    source_id: str
    side: int
    price: float
    quantity: float


class Certificate:
    def __init__(self, ledger, missing_info):
        self.ledger = ledger
        ref = ledger.result['local_reference']
        if ref['contract'] != A.REFERENCE or ref['mode'] != 'none' or ref['cap'] is not None:
            raise ValueError('side independence requires the uncapped neutral reference')
        self.missing = {x.source_id: x for x in missing_info}
        if len(self.missing) != len(missing_info) or set(self.missing) != {g.source_id for g in ledger.coverage.missing}:
            raise ValueError('missing-source metadata does not reconcile')
        for x in missing_info:
            if x.side not in (0, 1) or not (math.isfinite(x.price) and 1 <= x.price <= 99
                    and math.isfinite(x.quantity) and x.quantity > 0):
                raise ValueError('invalid missing-source metadata')
        self.orders = {}
        for side in (0, 1):
            orders = [o for k,o in ledger.orders.items() if k[0] == side and not o['ref']]
            self.orders[side] = (
                np.asarray([o['ta'] for o in orders], float),
                np.asarray([min(o['tr'], o['tfill'], ledger.end) for o in orders], float),
                np.asarray([o['pt'] for o in orders], float))
        self.events = []
        for g in ledger.coverage.missing:
            x = self.missing[g.source_id]
            # Even a supplied finite empirical interval is not a bound on a
            # missing anchor. Other exclusions carry their actual known time.
            lo, hi = (-math.inf, math.inf) if g.reason == 'untimed' else (g.earliest_ms, g.latest_ms)
            self.events.append(dict(id=x.source_id, side=x.side, kind='missing', lo=lo, hi=hi,
                                    earliest=self.first_reach(x.side, x.price, lo, hi)))
        for key, info in ledger.leg_info.items():
            if info.timing != 'e2': continue
            t, p, q = ledger.legs[key]; lo=t-info.uncertainty_ms; hi=t+info.uncertainty_ms
            self.events.append(dict(id=info.source_id, side=key[0], kind='e2', lo=lo, hi=hi,
                                    earliest=self.first_reach(key[0], p, lo, hi)))

    def first_reach(self, side, price, lo, hi):
        ta, end, p = self.orders[side]
        begin = np.maximum(ta, max(lo, self.ledger.ws))
        stop = np.minimum(end, hi)
        reaches = (price <= p) if side == 0 else (price >= p)
        # Closed endpoints are conservative for activation/cancel/fill ties.
        ok = reaches & (begin <= stop)
        return float(begin[ok].min()) if ok.any() else math.inf

    def assess(self, side, decision, end, *, estimated=False):
        relevant = [e for e in self.events if e['side']==side and e['earliest'] <= end
                    and (not estimated or e['kind']=='missing')]
        return dict(contract=CONTRACT, certified=not relevant,
                    mode='untimed_invariant_given_e2_anchors' if estimated else 'value_invariant_within_e2_intervals',
                    unresolved_ids=[e['id'] for e in relevant],
                    prior_state_ids=[e['id'] for e in relevant if e['earliest'] <= decision],
                    opposite_side_ids=[e['id'] for e in self.events if e['side'] != side],
                    no_reach_ids=[e['id'] for e in self.events if e['side']==side and e['earliest'] > end],
                    e2_interval_is_assumption=True, cap=None)

    def label(self, key, decision, *, day, estimated=False, settings=A.Settings()):
        l = self.ledger; side=key[0]; end=min(decision+settings.exposure_ms,l.end)
        proof=self.assess(side,decision,end,estimated=estimated)
        # First classify invalid/non-resting/mark conditions without making a
        # claim about timing. This is a local evaluation view, not a new source
        # ledger: the original raw count + every excluded identity stay below.
        view=copy.copy(l); view.coverage=replace(l.coverage,missing=())
        row=A.label_action(view,key,decision,day=day,settings=replace(settings,timing='all_anchors_sensitivity'))
        row['timing_certificate']=proof
        row['source_accounting']=dict(expected_legs=l.coverage.expected_legs,
            timed_legs=len(l.legs),missing_legs=len(l.coverage.missing),
            excluded_source_ids=[g.source_id for g in l.coverage.missing])
        row['timing']='certified_estimated' if estimated else 'certified_ws'
        row['label_spec']['timing']=row['timing']
        if row['status'] not in ('eligible','eligible_estimated_timing'): return row
        if not proof['certified']:
            row['status']='uncertain_prior_reference' if proof['prior_state_ids'] else 'uncertain_future_exposure'
            row['label']=None
        else:
            row['status']='eligible_estimated_timing' if estimated else 'eligible_timing_invariant'
        return row
