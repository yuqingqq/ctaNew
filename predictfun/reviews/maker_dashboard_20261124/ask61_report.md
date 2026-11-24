# 61-cent disappearance: partial execution, then likely withdrawal

2026-11-24, user-selected market 3789484, opening 1792977000000ms.
This follow-up narrows the previous statement that the 61c disappearance
was unresolved. Saved records only; no strategy replay or new market.

At settlement upper bound 8.537s, a buyer filled 24.5901639344 shares at 61c:
10 shares from another maker and 14.5901639344 from rank3's hash
`0x0f062c64c08f6fc1aebafcc64794d61d71e4e05662c94cb3278a0438fa3d2905`.
This is a different order from rank3's later 63c order. The full 24.59-shares
record is observed, not inferred from a book decline.

The native book then shows 205.4098360656 shares at 61c at 8.539s, and zero
at 8.592s. It had shown 60 shares at 8.494s; therefore the 61c sequence also
contains replenishment, and cannot be described as one static queue exhausted
by the later 200-share buyer.

An independent query of every canonical 61c ask maker leg in this market
(36 legs) finds no execution whose accepted hard window overlaps the
8.539–8.592s disappearance. This includes settlement timestamps up to 10.592s
under the existing <=2s settlement-lag premise. The earlier 24.59-share record
has upper bound 8.537s and precedes this drop. No invalid source status was
accepted. The book frames and price-level quantities were independently read
from the canonical parquet files rather than trusted from the first probe.

Conclusion: **61c traded earlier, but the subsequent 205.41-share disappearance
is consistent with cancellation/withdrawal and has no matching recorded fill.**
Under complete trade capture, the existing clock bound and faithful book
snapshots it is a nontrade removal. Without a private cancel feed this is a
conditional inference, not an observed cancellation command or identified
maker. We cannot assign all 205.41 shares to rank3 or prove a 61c-to-63c reprice.
The later 200-share buyer's actual record contains only 45 at 62c and 155 at
63c; the disappearance is not a missing 61c leg in that record.

Evidence: ask61_evidence.json and ask61_checks.json. Positive case passes;
three falsifiers (same-window execution, delayed execution and invalid book)
refuse the nontrade-removal reading. The query uses all market/price legs,
not only the dashboard's top-three identities.

Resource: mt_dash_ask61_probe and mt_dash_ask61_verify use 0.30+0.30 CPU seconds;
allow 1s publication overhead. Packet upper 71.45+1.60=73.05s of 180s. No new
simulation, model, untouched data, or policy conclusion. The underlying
dashboard payload remains unchanged.
