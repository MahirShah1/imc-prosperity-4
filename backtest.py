"""Round 3 backtester.

One file containing the data loaders, matching engine, fill accounting,
result container, and a CLI entry point. Run from the project root:

    .venv/bin/python backtest.py --days 0,1,2 --out results/run.json

The trader is plug-and-play: any object with ``run(state) -> (orders, conversions, traderData)``
works.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from datamodel import Order, OrderDepth, TradingState
from Observation import Observation


# ---------- round 3 config ----------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"

DEFAULT_POSITION_LIMIT = 20
TS_PER_DAY = 1_000_000

VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_UNDERLYING = "VELVETFRUIT_EXTRACT"
VEV_PRODUCTS = [f"VEV_{k}" for k in VEV_STRIKES]
ALL_PRODUCTS = ["HYDROGEL_PACK", VEV_UNDERLYING, *VEV_PRODUCTS]
POSITION_LIMITS: Dict[str, int] = {p: DEFAULT_POSITION_LIMIT for p in ALL_PRODUCTS}


# ---------- record schemas ----------

@dataclass(frozen=True)
class TradeRecord:
    day: int
    timestamp: int
    product: str
    side: str  # "BUY" | "SELL"
    qty: int
    price: int
    mid: float
    position_after: int
    cash_after: float
    mtm_pnl_after: float


@dataclass(frozen=True)
class MarkRecord:
    day: int
    timestamp: int
    product: str
    mid: float
    best_bid: Optional[int]
    best_ask: Optional[int]
    position: int
    mtm_pnl: float


@dataclass
class BacktestResult:
    summary: Dict
    trades: List[TradeRecord] = field(default_factory=list)
    marks: List[MarkRecord] = field(default_factory=list)

    def trades_df(self) -> pd.DataFrame:
        cols = list(TradeRecord.__annotations__)
        if not self.trades:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([asdict(t) for t in self.trades])

    def marks_df(self) -> pd.DataFrame:
        cols = list(MarkRecord.__annotations__)
        if not self.marks:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([asdict(m) for m in self.marks])

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "summary": self.summary,
            "trades": [asdict(t) for t in self.trades],
            "marks": [asdict(m) for m in self.marks],
        }, indent=2, default=float))

    @classmethod
    def from_json(cls, path) -> "BacktestResult":
        payload = json.loads(Path(path).read_text())
        return cls(
            summary=payload["summary"],
            trades=[TradeRecord(**t) for t in payload["trades"]],
            marks=[MarkRecord(**m) for m in payload["marks"]],
        )


# ---------- CSV loaders ----------

Snapshot = Tuple[int, int, Dict[str, OrderDepth]]


def load_price_events(days: Iterable[int] = (0, 1, 2),
                      data_dir: Path = DATA_DIR) -> List[Snapshot]:
    """Yield ``(day, timestamp, {product: OrderDepth})`` snapshots in time order."""
    days = set(days)
    products = set(ALL_PRODUCTS)
    events: Dict[Tuple[int, int], Dict[str, OrderDepth]] = {}

    for path in sorted(data_dir.glob("prices_round_3_day_*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                if row["product"] not in products:
                    continue
                day = int(row["day"])
                if day not in days:
                    continue
                ts = int(row["timestamp"])
                key = (day, ts)
                events.setdefault(key, {})

                depth = OrderDepth()
                for level in (1, 2, 3):
                    bp = row.get(f"bid_price_{level}", "")
                    bv = row.get(f"bid_volume_{level}", "")
                    if bp and bv:
                        depth.buy_orders[int(float(bp))] = int(float(bv))
                    ap = row.get(f"ask_price_{level}", "")
                    av = row.get(f"ask_volume_{level}", "")
                    if ap and av:
                        depth.sell_orders[int(float(ap))] = -int(float(av))
                events[key][row["product"]] = depth

    return [(d, t, depths) for (d, t), depths in sorted(events.items())]


def load_prices_df(days: Iterable[int] = (0, 1, 2),
                   data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Long-form prices DataFrame for research."""
    days = set(days)
    frames = []
    for path in sorted(data_dir.glob("prices_round_3_day_*.csv")):
        df = pd.read_csv(path, sep=";")
        df = df[df["product"].isin(ALL_PRODUCTS) & df["day"].isin(days)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_trades_df(days: Iterable[int] = (0, 1, 2),
                   data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Long-form market-trades DataFrame; ``day`` inferred from filename."""
    days = set(days)
    frames = []
    for path in sorted(data_dir.glob("trades_round_3_day_*.csv")):
        try:
            day = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if day not in days:
            continue
        df = pd.read_csv(path, sep=";")
        df = df[df["symbol"].isin(ALL_PRODUCTS)]
        df.insert(0, "day", day)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------- matching ----------

def _consume_ask(depth: OrderDepth, price: int, qty: int) -> None:
    new_vol = depth.sell_orders[price] + qty  # sell volumes are negative
    if new_vol >= 0:
        del depth.sell_orders[price]
    else:
        depth.sell_orders[price] = new_vol


def _consume_bid(depth: OrderDepth, price: int, qty: int) -> None:
    new_vol = depth.buy_orders[price] - qty
    if new_vol <= 0:
        del depth.buy_orders[price]
    else:
        depth.buy_orders[price] = new_vol


def match_depth(order: Order, depth: OrderDepth,
                position: int, position_limit: int) -> List[Tuple[str, int, int]]:
    """Walk all 3 depth levels until the order's price or size runs out.

    Returns a list of ``(side, qty, price)`` fills. Mutates ``depth`` in place.
    """
    fills: List[Tuple[str, int, int]] = []
    if order.quantity > 0:
        remaining = order.quantity
        for ask in sorted(depth.sell_orders):
            if ask > order.price:
                break
            avail = -depth.sell_orders[ask]
            room = position_limit - position
            if avail <= 0 or room <= 0:
                if room <= 0:
                    break
                continue
            qty = min(remaining, avail, room)
            if qty <= 0:
                break
            fills.append(("BUY", qty, ask))
            _consume_ask(depth, ask, qty)
            position += qty
            remaining -= qty
            if remaining <= 0:
                break
    elif order.quantity < 0:
        remaining = -order.quantity
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < order.price:
                break
            avail = depth.buy_orders[bid]
            room = position_limit + position
            if avail <= 0 or room <= 0:
                if room <= 0:
                    break
                continue
            qty = min(remaining, avail, room)
            if qty <= 0:
                break
            fills.append(("SELL", qty, bid))
            _consume_bid(depth, bid, qty)
            position -= qty
            remaining -= qty
            if remaining <= 0:
                break
    return fills


def match_top(order: Order, depth: OrderDepth,
              position: int, position_limit: int) -> List[Tuple[str, int, int]]:
    """Conservative matcher that only crosses the very best bid/ask."""
    if order.quantity > 0 and depth.sell_orders:
        best = min(depth.sell_orders)
        avail = -depth.sell_orders[best]
        room = position_limit - position
        if order.price >= best and avail > 0 and room > 0:
            qty = min(order.quantity, avail, room)
            if qty > 0:
                _consume_ask(depth, best, qty)
                return [("BUY", qty, best)]
    elif order.quantity < 0 and depth.buy_orders:
        best = max(depth.buy_orders)
        avail = depth.buy_orders[best]
        room = position_limit + position
        if order.price <= best and avail > 0 and room > 0:
            qty = min(-order.quantity, avail, room)
            if qty > 0:
                _consume_bid(depth, best, qty)
                return [("SELL", qty, best)]
    return []


MATCHERS = {"depth": match_depth, "top": match_top}


# ---------- engine ----------

def _best_bid_ask(depth: OrderDepth):
    bb = max(depth.buy_orders) if depth.buy_orders else None
    ba = min(depth.sell_orders) if depth.sell_orders else None
    return bb, ba


def _mid(bb, ba):
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None


def run_backtest(trader, days: Iterable[int] = (0, 1, 2),
                 matcher: str = "depth",
                 record_marks: bool = True,
                 data_dir: Path = DATA_DIR) -> BacktestResult:
    match_fn = MATCHERS[matcher]
    events = load_price_events(days, data_dir=data_dir)

    positions: Dict[str, int] = {p: 0 for p in ALL_PRODUCTS}
    cash: Dict[str, float] = {p: 0.0 for p in ALL_PRODUCTS}
    last_mid: Dict[str, float] = {}
    trades: List[TradeRecord] = []
    marks: List[MarkRecord] = []

    trader_data = ""
    listings: Dict = {}
    own_trades: Dict = {}
    market_trades: Dict = {}
    obs = Observation()

    start = time.perf_counter()
    for day, ts, depths in events:
        global_ts = day * TS_PER_DAY + ts
        # Trader gets a copy so it can't mutate the book mid-tick.
        trader_view = {p: copy.deepcopy(d) for p, d in depths.items()}
        state = TradingState(
            traderData=trader_data,
            timestamp=global_ts,
            listings=listings,
            order_depths=trader_view,
            own_trades=own_trades,
            market_trades=market_trades,
            position={p: positions[p] for p in ALL_PRODUCTS},
            observations=obs,
        )

        result, _conversions, trader_data = trader.run(state)

        for product, orders in (result or {}).items():
            if product not in POSITION_LIMITS or product not in depths:
                continue
            book = copy.deepcopy(depths[product])
            limit = POSITION_LIMITS[product]
            for order in orders or []:
                fills = match_fn(order, book, positions[product], limit)
                for side, qty, price in fills:
                    if side == "BUY":
                        positions[product] += qty
                        cash[product] -= qty * price
                    else:
                        positions[product] -= qty
                        cash[product] += qty * price
                    bb, ba = _best_bid_ask(depths[product])
                    mid_now = _mid(bb, ba) or 0.0
                    trades.append(TradeRecord(
                        day=day, timestamp=ts, product=product,
                        side=side, qty=qty, price=price, mid=mid_now,
                        position_after=positions[product],
                        cash_after=cash[product],
                        mtm_pnl_after=cash[product] + positions[product] * mid_now,
                    ))

        for product, depth in depths.items():
            bb, ba = _best_bid_ask(depth)
            mid = _mid(bb, ba)
            if mid is not None:
                last_mid[product] = mid
            if record_marks and mid is not None:
                marks.append(MarkRecord(
                    day=day, timestamp=ts, product=product, mid=mid,
                    best_bid=bb, best_ask=ba,
                    position=positions[product],
                    mtm_pnl=cash[product] + positions[product] * mid,
                ))

    elapsed = time.perf_counter() - start

    per_product_pnl = {
        p: cash[p] + positions[p] * last_mid.get(p, 0.0) for p in ALL_PRODUCTS
    }
    summary = {
        "matcher": matcher,
        "days": sorted(set(days)),
        "ticks": len(events),
        "trades": len(trades),
        "elapsed_sec": elapsed,
        "net_pnl": sum(per_product_pnl.values()),
        "per_product_pnl": per_product_pnl,
        "final_positions": dict(positions),
        "last_mid": last_mid,
    }
    return BacktestResult(summary=summary, trades=trades, marks=marks)


# ---------- CLI ----------

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", default="0,1,2",
                   help="Comma-separated day numbers (default: 0,1,2)")
    p.add_argument("--matcher", default="depth", choices=("depth", "top"))
    p.add_argument("--out", default="results/run.json",
                   help="Output JSON path (default: results/run.json)")
    p.add_argument("--no-marks", action="store_true",
                   help="Skip per-tick marks (smaller output, no equity curve)")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT))
    from trader import Trader

    days = [int(x) for x in args.days.split(",") if x.strip()]
    result = run_backtest(
        Trader(), days=days, matcher=args.matcher,
        record_marks=not args.no_marks,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    result.to_json(out_path)

    s = result.summary
    print(f"Matcher          : {s['matcher']}")
    print(f"Days             : {s['days']}")
    print(f"Ticks replayed   : {s['ticks']:,}")
    print(f"Trades executed  : {s['trades']:,}")
    print(f"Net PnL          : {s['net_pnl']:.2f}")
    print(f"Wall time        : {s['elapsed_sec']:.2f}s")
    print(f"Output           : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
