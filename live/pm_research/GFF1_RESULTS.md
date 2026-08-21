# G-FF1 — WS `side` vs on-chain taker direction (`gff1_v3`)

Verdict: **PASS**. Threshold 0.99 on the Wilson lower bound.

## Sample

- protocol SHA-256: `82293cca11d34cee04602dd11693d3b1595b852c2f57fbb518e651b080916518`
- script SHA-256: `fbc867234aaee1b7633b2a19535706a88298fc2d8082ac65fcbe870fe2b093c8`
- candidate digest: `469509e34507136f5d9396bca9f67c7063c4dc9c9b7c4a7ff70f2f401acf35a2`
- UTC days: 20260819, 20260820; seed 20260821
- source archives: 2938; candidate legs 1402887; candidate transactions 1401267
- sampled transactions: 600; validated: 600

## Result

Agreement **1.0000** (Wilson 95% [0.9936, 1.0000])

| coin | agree | n | rate |
|---|---:|---:|---:|
| bnb | 90 | 90 | 1.0000 |
| btc | 90 | 90 | 1.0000 |
| doge | 90 | 90 | 1.0000 |
| eth | 90 | 90 | 1.0000 |
| hype | 80 | 80 | 1.0000 |
| sol | 80 | 80 | 1.0000 |
| xrp | 80 | 80 | 1.0000 |

| moneyness | agree | n | rate |
|---|---:|---:|---:|
| 0.15-0.35 | 120 | 120 | 1.0000 |
| 0.35-0.65 | 120 | 120 | 1.0000 |
| 0.65-0.85 | 120 | 120 | 1.0000 |
| p<0.15 | 120 | 120 | 1.0000 |
| p>=0.85 | 120 | 120 | 1.0000 |

## Excluded (reported beside the retained set)

| reason | count |
|---|---:|
| — | 0 |

Protocol: `live/pm_research/GFF1_PROTOCOL.md`.
