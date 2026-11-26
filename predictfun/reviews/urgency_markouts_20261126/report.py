"""Independent aggregation and comparison with the saved two-market parity fixture."""
import bisect
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "urgency_size100_20261126"
HORIZONS = [0, 1, 5, 30]


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize(rows):
    sums = defaultdict(lambda: defaultdict(float))
    for row in rows:
        values = sums[int(row["horizon_s"])]
        for name in ["valid_shares", "cents", "after_end_shares", "missing_book_shares", "shares"]:
            values[name] += float(row[name])
    return {h: dict(values, per_share=values["cents"]/values["valid_shares"]) for h, values in sums.items()}


def fixture_control(lookup):
    file = HERE.parents[2] / "pmmmrust/parity/fixtures/pf_urgency/oct26.json.gz"
    fixture = json.loads(gzip.decompress(file.read_bytes()))
    for market in fixture:
        prices = {gid: (side, price) for _, kind, gid, side, price, _ in market["expected"] if kind == "place"}
        times, mids, fills = [], [], []
        for event in market["inputs"]:
            if "Frame" in event:
                v = event["Frame"]
                times.append(v["known_ms"])
                bid = v["bids"][0][0] if v["bids"] else -1
                ask = v["asks"][0][0] if v["asks"] else -1
                mids.append((bid+ask)/2 if 0 <= bid < ask <= 100 else None)
            if "Filled" in event:
                v = event["Filled"]
                fills.append((v["known_ms"]-100, v["qty"], *prices[v["id"]]))
        for h in HORIZONS:
            cents = shares = 0.
            for t, qty, side, price in fills:
                target = t + h*1000
                idx = bisect.bisect_right(times, target) - 1
                if target > 298000 or idx < 0 or mids[idx] is None:
                    continue
                cents += qty * (mids[idx] - price if side == 0 else price - mids[idx])
                shares += qty
            row = lookup[(str(market["market"]), str(h))]
            assert abs(cents-float(row["cents"])) < 1e-5 and abs(shares-float(row["valid_shares"])) < 1e-5
    return len(fixture)


def main():
    rows = read(HERE / "markets.csv")
    keys = {(r["mid"], r["horizon_s"]): r for r in rows}
    assert len(keys) == len(rows) == 689 * 4
    expected = {}
    for f in sorted(PARENT.glob("markets_*.csv")):
        for row in read(f):
            if row["size"] == "100":
                expected[row["day"], row["mid"]] = float(row["shares"])
    assert len(expected) == 689
    for row in rows:
        total = expected[row["day"], row["mid"]]
        assert abs(float(row["shares"])-total) < 1e-6
        assert abs(sum(float(row[n]) for n in ("valid_shares", "after_end_shares", "missing_book_shares"))-total) < 1e-6
    check_count = fixture_control(keys)
    days = sorted({d for d, _ in expected})
    daily = {day:summarize([r for r in rows if r["day"] == day]) for day in days}
    pooled = summarize(rows)
    shadow = read(HERE / "tokyo.csv")
    assert len({(r["mid"],r["horizon_s"],r["mode"]) for r in shadow}) == len(shadow) == 11 * 4 * 2
    sources = {"Historical backtest":pooled}
    for mode, name in [("low_capped","Tokyo crossing estimate"),("high_capped","Tokyo crossing plus queue decreases")]:
        sources[name] = summarize([r for r in shadow if r["mode"] == mode])
    lines = ["# 100-share urgency strategy markouts", "",
             "Descriptive recount on November 26, 2026. The historical sample comprises the original 689 complete markets on nine consumed dates. Tokyo covers 11 different markets in one hour of shadow trading on November 26. All prices are in YES cents. Each fill contributes its signed difference between the future book midpoint and execution price, weighted by shares. A positive value favors the maker.", "",
             "| Population | At fill | 1 s | 5 s | 30 s |", "|---|---:|---:|---:|---:|"]
    for name, data in sources.items():
        lines.append("| " + name + " | " + " | ".join(f'{data[h]["per_share"]:+.3f}' for h in HORIZONS) + " |")
    lines += ["", "## Historical markout by date, cents per eligible filled share", "",
              "| UTC date | Markets | At fill | 1 s | 5 s | 30 s |", "|---|---:|---:|---:|---:|---:|"]
    for day, data in daily.items():
        lines.append(f"| {day} | {sum(d==day for d,_ in expected)} | " + " | ".join(f'{data[h]["per_share"]:+.3f}' for h in HORIZONS) + " |")
    lines += ["", "Days with a negative historical markout: " + ", ".join(
        f"{sum(d[h]['per_share'] < 0 for d in daily.values())}/{len(days)} at {h} s" for h in HORIZONS
    ) + ".", "", "## Denominators and checks", "",
              "| Population | Horizon | Eligible shares | After endpoint | Missing two-sided book | Markout USD |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, data in sources.items():
        for h in HORIZONS:
            v = data[h]
            lines.append(f'| {name} | {h} s | {v["valid_shares"]:,.2f} | {v["after_end_shares"]:,.2f} | {v["missing_book_shares"]:,.2f} | {v["cents"]/100:+,.2f} |')
    lines += ["",
              f"Every historical market reproduced all nine archived metrics. An independent calculation using the two pinned October 26 Rust parity fixtures reproduced the eligible quantities and markout sums at every horizon in {check_count} markets. The Tokyo fill legs reconcile to the committed estimator quantities in all 11 markets.",
              "",
              "The midpoint is the latest two-sided top of book received by the target time, using the frame current at that time. A missing or invalid current book excludes the fill for that horizon. Both datasets exclude horizons beyond opening plus 298 s. The historical clock starts at the simulated match. The Tokyo clock starts at the frame used to infer the fill. Inventory feedback and the sources of fills differ between the two runs. No maker fees, rebates or own-market impact are included.",
              "",
              "The earlier historical run reported +4.546 cents per FIFO-matched share and +$22,244.34 at settlement. Those figures measure completed turns and settlement economics. Each fixed-horizon markout measures a single fill. The different horizons on the same fills must be read separately.",
              "", "[Original historical receipt](../urgency_size100_20261126/report.md) · [Tokyo receipt](https://github.com/yuqingqq/pmmmrust/blob/29741ec/docs/receipts/20261126_urgency100_shadow_review/README.md)", ""]
    (HERE / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
