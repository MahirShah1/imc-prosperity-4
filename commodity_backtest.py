import csv
from pathlib import Path
from typing import Dict, List, Optional

from datamodel import OrderDepth, TradingState
from Observation import Observation
from trader import Trader

PRODUCTS = {"ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"}
DATA_DIR = Path(__file__).resolve().parent / "Data"


def load_price_events(data_dir: Path):
    events = {}

    for path in sorted(data_dir.glob("prices_round_1_day_*.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                product = row["product"]
                if product not in PRODUCTS:
                    continue

                day = int(row["day"])
                timestamp = int(row["timestamp"])
                key = (day, timestamp)
                events.setdefault(key, {})

                depth = OrderDepth()
                for level in (1, 2, 3):
                    bid_price = row.get(f"bid_price_{level}", "")
                    bid_volume = row.get(f"bid_volume_{level}", "")
                    if bid_price and bid_volume:
                        depth.buy_orders[int(bid_price)] = int(bid_volume)

                    ask_price = row.get(f"ask_price_{level}", "")
                    ask_volume = row.get(f"ask_volume_{level}", "")
                    if ask_price and ask_volume:
                        depth.sell_orders[int(ask_price)] = -int(ask_volume)

                events[key][product] = depth

    return sorted(events.items())


def simulate():
    trader = Trader()
    state = TradingState(
        traderData="",
        timestamp=0,
        listings={},
        order_depths={},
        own_trades={},
        market_trades={},
        position={product: 0 for product in PRODUCTS},
        observations=Observation(),
    )

    positions = {product: 0 for product in PRODUCTS}
    realized_pnl = 0.0
    last_mid: Dict[str, float] = {}
    trade_log = []

    events = load_price_events(DATA_DIR)
    for (day, timestamp), depths in events:
        state.timestamp = day * 100000 + timestamp
        state.order_depths = depths
        state.position = positions.copy()

        result, _, state.traderData = trader.run(state)

        for product, orders in result.items():
            depth = depths.get(product)
            if depth is None:
                continue

            best_bid, best_bid_vol, best_ask, best_ask_vol = trader.best_bid_ask(depth)
            for order in orders:
                if order.quantity > 0 and best_ask is not None and order.price >= best_ask:
                    qty = min(order.quantity, -best_ask_vol)
                    if qty > 0:
                        positions[product] += qty
                        realized_pnl -= qty * best_ask
                        trade_log.append((day, timestamp, product, "BUY", qty, best_ask))
                elif order.quantity < 0 and best_bid is not None and order.price <= best_bid:
                    qty = min(-order.quantity, best_bid_vol)
                    if qty > 0:
                        positions[product] -= qty
                        realized_pnl += qty * best_bid
                        trade_log.append((day, timestamp, product, "SELL", qty, best_bid))

        for product, depth in depths.items():
            best_bid, _, best_ask, _ = trader.best_bid_ask(depth)
            mid = trader.mid_price(best_bid, best_ask)
            if mid is not None:
                last_mid[product] = mid

    unrealized = sum(positions[product] * last_mid.get(product, 0.0) for product in PRODUCTS)
    net_pnl = realized_pnl + unrealized

    return {
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized,
        "net_pnl": net_pnl,
        "positions": positions,
        "last_mid": last_mid,
        "trade_count": len(trade_log),
        "trade_log": trade_log,
    }


if __name__ == "__main__":
    results = simulate()
    print("Backtest summary")
    print("----------------")
    print(f"Realized PnL: {results['realized_pnl']:.2f}")
    print(f"Unrealized PnL: {results['unrealized_pnl']:.2f}")
    print(f"Net PnL: {results['net_pnl']:.2f}")
    print(f"Final positions: {results['positions']}")
    print(f"Last observed mid prices: {results['last_mid']}")
    print(f"Trades executed: {results['trade_count']}")
    print("\nSample trades:")
    for trade in results["trade_log"][0:20]:
        day, timestamp, product, side, qty, price = trade
        print(f"day={day} ts={timestamp} {product} {side} {qty} @ {price}")
