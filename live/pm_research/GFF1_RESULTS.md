# G-FF1 — WS `side` vs on-chain taker direction (`gff1_v2`)

Verdict: **INSUFFICIENT_EVIDENCE**. Threshold 0.99 on the Wilson lower bound.

## Sample

- protocol SHA-256: `9cb97da773a0c7450e1082ef0057ecfd0f6ccdf2bca7a41316cc188d89c8271f`
- script SHA-256: `b4dad60d3eb2e6ef0bb602347f01c47074522c8830fe9f7069d67a3a0bcc1f3a`
- candidate digest: `469509e34507136f5d9396bca9f67c7063c4dc9c9b7c4a7ff70f2f401acf35a2`
- UTC days: 20260819, 20260820; seed 20260821
- source archives: 2938; candidate legs 1402887; candidate transactions 1401267
- sampled transactions: 500; validated: 473

## Result

Agreement **1.0000** (Wilson 95% [0.9919, 1.0000])

| coin | agree | n | rate |
|---|---:|---:|---:|
| bnb | 78 | 78 | 1.0000 |
| btc | 67 | 67 | 1.0000 |
| doge | 69 | 69 | 1.0000 |
| eth | 66 | 66 | 1.0000 |
| hype | 63 | 63 | 1.0000 |
| sol | 65 | 65 | 1.0000 |
| xrp | 65 | 65 | 1.0000 |

| moneyness | agree | n | rate |
|---|---:|---:|---:|
| 0.15-0.35 | 96 | 96 | 1.0000 |
| 0.35-0.65 | 95 | 95 | 1.0000 |
| 0.65-0.85 | 93 | 93 | 1.0000 |
| p<0.15 | 95 | 95 | 1.0000 |
| p>=0.85 | 94 | 94 | 1.0000 |

## Excluded (reported beside the retained set)

| reason | count |
|---|---:|
| `JOIN_MISMATCH` | 27 |

## Why not PASS

- only 473 validated tx-clusters, protocol requires 500
- JOIN_MISMATCH rate 0.054 exceeds 0.05

Protocol: `live/pm_research/GFF1_PROTOCOL.md`.
