"""Partition authoritative match events by market window; preserve exclusion counts."""
import numpy as np


def select_trade_windows(legs, registry):
    if not registry.index.is_unique or set(legs.mid)!=set(registry.index):
        raise ValueError('registry and trade tape populations differ')
    if not np.isfinite(registry[['ws','we']].to_numpy()).all() or not ((registry.we-registry.ws)==300).all():
        raise ValueError('invalid market windows')
    if not np.isfinite(legs.t_ms.to_numpy()).all():raise ValueError('missing match event time')
    before=legs.t_ms<legs.mid.map(registry.ws)*1000
    after=legs.t_ms>=legs.mid.map(registry.we)*1000
    inside=~(before|after)
    counts=dict(before_window=int(before.sum()),in_window=int(inside.sum()),after_window=int(after.sum()))
    return legs.loc[inside].copy(),counts
