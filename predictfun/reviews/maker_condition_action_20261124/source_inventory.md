# Documented action-source coverage — 2026-11-24

Read-only documentation check prompted by the USER's request to bridge the full condition-to-action chain. These are current documented interfaces, not a claim that the October collector recorded their complete semantics. No authenticated API request, credential use, subscription, live collection or trading was performed.

| Source | What it provides | Does it close the competitor action join? |
|---|---|---|
| Public book WebSocket | Full aggregate book snapshots; nearby changes can be combined, omitting intermediate states. | No maker-attributed add/cancel stream or complete queue chronology is documented. |
| Wallet events | Order identity, lifecycle events and event timestamps, through a JWT belonging to that wallet. | Useful for the wallet's own actions; not a public feed of competitors' lifecycle events. |
| Order lookup by hash | Details/status for one's own order, requiring wallet JWT authentication. | Knowing a competitor's signed hash does not grant access to this route. |
| Recorded transaction receipts | Successful settlement events and original signed terms in supported calldata; some explicit on-chain cancellation events. | Useful for identity, fills and budget exhaustion. Ordinary quote removal can happen off-chain. |

Primary-source support:

- [Order book topic](https://dev.predict.fun/websocket/topics/order-book): whole-book updates, with intermediate changes potentially combined. [WebSocket overview](https://dev.predict.fun/websocket): distinguishes aggregate market topics from wallet events.
- [Follow your orders](https://dev.predict.fun/websocket/how-tos/follow-your-orders): wallet-specific JWT, order ID/hash, millisecond event timestamp, quantity and fills; disconnected events are not replayed.
- [Get order by hash](https://dev.predict.fun/api/orders/get-order-by-hash): explicitly restricted to the authenticated wallet's own orders.
- [Market making](https://dev.predict.fun/trading/market-making): quotes are changed by removing and replacing them; removal is off-chain. This explains why a complete cancellation narrative cannot generally be recovered from settlement receipts alone.

Conclusion from the documented coverage: these standard public interfaces do not provide the missing historical, competitor-attributed command stream. That is a source-coverage result, not proof that no separately shared exchange/maker dataset could exist. Another archive RPC would not reveal ordinary off-chain withdrawals. Do not spend the next round treating the 47 unresolved signed remainders as if archival state alone could recover their private pull logic.

To complete the identification task rather than just observable associations, obtain an authorized maker/exchange export containing both filled and unfilled order lifecycles, or keep posting/pulling as partially identified hypotheses. Own logs can validate the reconstruction method but cannot establish a competitor's decisions. No external request has been sent.
