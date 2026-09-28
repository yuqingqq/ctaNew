"""Trigger math shared by the live engine and the historical replay.

A sudden move at horizon h fires when ALL hold for an asset:
  |e_h| >= floor_h                 e_h = r_h - m_h, the move net of the market
                                    (cross-sectional median r_h of liquid instruments)
  |z_h| >= zmin_h                  z_h = e_h / (sigma_1m * sqrt(h / 60 s))
  vm_h  >= vmin_h                  USD volume over h / (baseline 1m USD volume * h/60)
  liq   >= liq_min_usd             24h USD volume summed across venues

r is a log return. Floors are configured as simple percents and converted with
log(1 + p), so a 10% floor fires at +10.0% up and -9.1% down.

Baselines exclude the most recent hour so an ongoing move never raises its own bar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HORIZONS = (60, 300, 900, 3600, 14400, 86400)
H_NAME = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 14400: "4h", 86400: "24h"}
NAME_H = {v: k for k, v in H_NAME.items()}


@dataclass
class HorizonRule:
    floor_pct: float   # e.g. 10.0 for 10%
    zmin: float
    vmin: float

    @property
    def floor(self) -> float:
        return math.log1p(self.floor_pct / 100.0)


@dataclass
class Rules:
    horizons: dict[int, HorizonRule]
    liq_min_usd: float = 2_000_000.0
    market_liq_usd: float = 5_000_000.0     # instruments counted in the market median
    market_floor_pct: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Rules":
        hz = {NAME_H[k]: HorizonRule(**v) for k, v in d["horizons"].items()}
        mk = {NAME_H[k]: float(v) for k, v in d.get("market_floor_pct", {}).items()}
        return cls(horizons=hz, liq_min_usd=float(d.get("liq_min_usd", 2e6)),
                   market_liq_usd=float(d.get("market_liq_usd", 5e6)), market_floor_pct=mk)

    def scaled(self, floor_mult: float = 1.0, z_mult: float = 1.0, v_mult: float = 1.0) -> "Rules":
        hz = {h: HorizonRule(r.floor_pct * floor_mult, r.zmin * z_mult, r.vmin * v_mult)
              for h, r in self.horizons.items()}
        return Rules(horizons=hz, liq_min_usd=self.liq_min_usd, market_liq_usd=self.market_liq_usd,
                     market_floor_pct=dict(self.market_floor_pct))


DEFAULT_RULES = Rules(
    horizons={
        60: HorizonRule(3.0, 6.0, 5.0),
        300: HorizonRule(5.0, 6.0, 4.0),
        900: HorizonRule(7.0, 5.0, 3.0),
        3600: HorizonRule(10.0, 5.0, 3.0),
        14400: HorizonRule(15.0, 4.0, 2.5),
        86400: HorizonRule(25.0, 4.0, 2.0),
    },
    market_floor_pct={900: 2.5, 3600: 4.0, 14400: 6.0},
)


def baselines_from_minutes(C: np.ndarray, V: np.ndarray, min_hist: int = 120,
                           sigma_floor: float = 5e-4, vol_floor_usd: float = 50.0):
    """Per-row baseline from a window of 1m closes C and USD volumes V (n x L).

    sigma_1m: RMS of 1m log returns with each row's squared returns clipped at its
      own 99th percentile (one wick does not set the scale), floored at 5 bps.
    vol_1m:   max(median, 0.3 * mean, $50) of 1m USD volume over minutes where the
      instrument had a price -- the median alone is 0 for thin names, which would
      make any trade an infinite surge.
    Rows with < min_hist valid 1m returns get NaN (not eligible to trigger alone).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(C), axis=1)
    valid = np.isfinite(lr)
    nh = valid.sum(axis=1)
    r2 = np.where(valid, lr * lr, np.nan)
    ok = nh > 0
    sig = np.full(C.shape[0], np.nan)
    v1m = np.full(C.shape[0], np.nan)
    if ok.any():
        r2o = r2[ok]
        with np.errstate(invalid="ignore"):
            p99 = np.nanpercentile(r2o, 99, axis=1, keepdims=True)
        sig[ok] = np.sqrt(np.nanmean(np.minimum(r2o, p99), axis=1))
        Vv = np.where(np.isfinite(C[ok]), V[ok], np.nan)
        med = np.nanmedian(Vv, axis=1)
        mean = np.nanmean(Vv, axis=1)
        v1m[ok] = np.fmax(np.fmax(med, 0.3 * mean), vol_floor_usd)
    sig = np.fmax(sig, sigma_floor)
    bad = nh < min_hist
    sig[bad] = np.nan
    v1m[bad] = np.nan
    return sig, v1m, nh


def zscore(e: np.ndarray, sigma1m: np.ndarray, h: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return e / (sigma1m * math.sqrt(h / 60.0))


def vol_multiple(vol_h: np.ndarray, vol1m: np.ndarray, h: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return vol_h / (vol1m * (h / 60.0))


def fire(e: np.ndarray, z: np.ndarray, vm: np.ndarray, liq: np.ndarray, rule: HorizonRule,
         liq_min: float) -> np.ndarray:
    """Boolean trigger for one horizon (NaN inputs never fire)."""
    with np.errstate(invalid="ignore"):
        return ((np.abs(e) >= rule.floor) & (np.abs(z) >= rule.zmin) & (vm >= rule.vmin)
                & (liq >= liq_min))


def market_median(r: np.ndarray, liq: np.ndarray, liq_min: float) -> float:
    m = r[(liq >= liq_min) & np.isfinite(r)]
    return float(np.median(m)) if m.size >= 10 else 0.0


def pct(logret: float) -> float:
    """Log return -> simple percent."""
    return (math.exp(logret) - 1.0) * 100.0
