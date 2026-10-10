# Forward tests unblinded: F1 paper P&L shown live (2026-10-10 02:30Z, user decision)

**What changed.** The user chose to see F1's paper results before its read, history included.
From this commit the service shows:
- every F1 trade's result in its COVER message (entry and exit price, gross, cost, funding, net);
- the running paper book (trades, wins, average gross and net, worst, total at 1% of capital per
  trade) on SHORT and COVER messages, and in a daily summary at 00:05 UTC;
- the open positions with their unrealized gross.

The ledger `data/alerts/fadebook.jsonl` was backfilled with every F1 trade since F1 began
(2026-10-02 14:40Z). `python3 -m research.forward_fade --paper` prints it.

**What this unblinds.**
- **F1 (`docs/FORWARD_FADE_F1_v2.md`):** every outcome, as it happens.
- **Forward test v2, H1 (`docs/FORWARD_TEST_v2.md`), partly.** F1's trades are the heavy pumps'
  raw returns from +4 h to +24 h. H1 ranks all pumps by volume ratio against the
  market-adjusted return from +4 h to +72 h, so the heavy end of H1's outcome becomes partly
  visible. H2 (breadth, alert → +4 h) and B1 are not shown.

**What stays locked** (stated at the user's choice, before any result was shown):
- F1's population, trade, cost, funding source, H3 and its pass rule, the minimum (≥ 400
  complete rows, ≥ 12 ISO weeks) and the single `--final` read are unchanged. Nothing is stopped
  early or changed because of interim results.
- The same holds for v2's H1, H2, B1, minimum and `--final` read.
- A change to either test now needs a new declaration. Data already seen then counts as
  consumed for that question.
- Both evaluators write this note into their receipts (`unblinded` / `notes`), so an automated
  reader of a receipt sees it.

**How the live numbers are made** (`movewatch/watchers.py` PaperBook, outside the v2 identity):
- **Price and cost:** the same entry and exit records and population as the evaluator; a test
  checks that the trades and gross match. Cost is a flat 0.40%.
- **Funding:** while a short is open, the perp's markPrice stream is polled every 10 min. Each
  settlement in the hold is charged at the last rate seen before it. If the polls leave a gap over
  50 min, funding is pending.
- **Exact funding:** once Binance Vision publishes the hold's month (early the next month), the
  exact settled rates replace the estimate. Until then, history trades show funding as pending,
  counted at 0, with the research average of −0.78% per trade noted beside the total.
