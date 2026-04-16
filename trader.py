from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    POSITION_LIMIT = {
        "ASH_COATED_OSMIUM": 20,
        "INTARIAN_PEPPER_ROOT": 20,
    }
    MEMORY_LENGTH = 100

    # ---------- persistence helpers ----------

    def load_memory(self, trader_data: str):
        if trader_data:
            try:
                memory = json.loads(trader_data)
                if "mid_hist" not in memory:
                    memory["mid_hist"] = {}
                if "prev_mid" not in memory:
                    memory["prev_mid"] = {}
                return memory
            except Exception:
                pass

        return {
            "mid_hist": {
                "ASH_COATED_OSMIUM": [],
                "INTARIAN_PEPPER_ROOT": [],
            },
            "prev_mid": {},
        }

    def save_memory(self, memory) -> str:
        return json.dumps(memory)

    # ---------- market helpers ----------

    def best_bid_ask(
        self, order_depth: OrderDepth
    ) -> Tuple[Optional[int], int, Optional[int], int]:
        best_bid = None
        best_bid_vol = 0
        best_ask = None
        best_ask_vol = 0

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = order_depth.sell_orders[best_ask]

        return best_bid, best_bid_vol, best_ask, best_ask_vol

    def mid_price(self, best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def microprice(
        self,
        best_bid: Optional[int],
        best_bid_vol: int,
        best_ask: Optional[int],
        best_ask_vol: int,
    ) -> Optional[float]:
        if best_bid is None or best_ask is None:
            return None

        bid_size = max(best_bid_vol, 0)
        ask_size = abs(min(best_ask_vol, 0))

        denom = bid_size + ask_size
        if denom == 0:
            return (best_bid + best_ask) / 2

        return (best_ask * bid_size + best_bid * ask_size) / denom

    def ema(self, values: List[float], span: int = 20) -> Optional[float]:
        if not values:
            return None
        alpha = 2 / (span + 1)
        e = values[0]
        for x in values[1:]:
            e = alpha * x + (1 - alpha) * e
        return e

    # ---------- fair values ----------

    def fair_ash(self, mid: float, position: int) -> float:
        return 10000 + 0.20 * (mid - 10000) - 0.18 * position

    def fair_pepper(
        self,
        mid: float,
        micro: Optional[float],
        prev_mid: Optional[float],
        hist: List[float],
        position: int,
    ) -> float:
        ema20 = self.ema(hist[-20:], span=20) if hist else mid
        if ema20 is None:
            ema20 = mid

        if micro is None:
            micro = mid

        if prev_mid is None:
            prev_mid = mid

        fair = (
            mid
            + 0.40 * (mid - ema20)
            + 0.55 * (micro - mid)
            + 0.10 * (mid - prev_mid)
        )
        fair -= 0.22 * position
        return fair

    # ---------- execution helpers ----------

    def market_take(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        position: int,
        limit: int,
        take_threshold: float,
        max_clip: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        pos = position

        for ask in sorted(order_depth.sell_orders.keys()):
            ask_vol = -order_depth.sell_orders[ask]
            if ask <= fair - take_threshold and pos < limit:
                qty = min(ask_vol, limit - pos, max_clip)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty
            else:
                break

        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_vol = order_depth.buy_orders[bid]
            if bid >= fair + take_threshold and pos > -limit:
                qty = min(bid_vol, pos + limit, max_clip)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty
            else:
                break

        return orders, pos

    def market_make(
        self,
        product: str,
        best_bid: Optional[int],
        best_ask: Optional[int],
        fair: float,
        position: int,
        limit: int,
        base_size: int,
    ) -> List[Order]:
        orders: List[Order] = []
        if best_bid is None or best_ask is None:
            return orders

        bid_quote = min(best_bid + 1, math.floor(fair - 1))
        ask_quote = max(best_ask - 1, math.ceil(fair + 1))

        if bid_quote >= ask_quote:
            bid_quote = math.floor(fair - 1)
            ask_quote = math.ceil(fair + 1)

        if bid_quote >= ask_quote:
            return orders

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        buy_size = min(base_size, buy_capacity)
        sell_size = min(base_size, sell_capacity)

        if position > 0.6 * limit:
            buy_size = min(buy_size, 1)
        if position < -0.6 * limit:
            sell_size = min(sell_size, 1)

        if buy_size > 0:
            orders.append(Order(product, bid_quote, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_quote, -sell_size))

        return orders

    # ---------- per-product strategies ----------

    def trade_ash(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMIT[product]

        best_bid, best_bid_vol, best_ask, best_ask_vol = self.best_bid_ask(order_depth)
        mid = self.mid_price(best_bid, best_ask)
        if mid is None:
            return []

        fair = self.fair_ash(mid, position)
        take_orders, new_pos = self.market_take(
            product=product,
            order_depth=order_depth,
            fair=fair,
            position=position,
            limit=limit,
            take_threshold=1.5,
            max_clip=5,
        )
        orders.extend(take_orders)

        mm_orders = self.market_make(
            product=product,
            best_bid=best_bid,
            best_ask=best_ask,
            fair=fair,
            position=new_pos,
            limit=limit,
            base_size=4,
        )
        orders.extend(mm_orders)
        return orders

    def trade_pepper(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        mid_hist: List[float],
        prev_mid: Optional[float],
    ) -> Tuple[List[Order], Optional[float]]:
        orders: List[Order] = []
        limit = self.POSITION_LIMIT[product]

        best_bid, best_bid_vol, best_ask, best_ask_vol = self.best_bid_ask(order_depth)
        mid = self.mid_price(best_bid, best_ask)
        if mid is None:
            return [], None

        micro = self.microprice(best_bid, best_bid_vol, best_ask, best_ask_vol)
        fair = self.fair_pepper(mid, micro, prev_mid, mid_hist, position)

        take_orders, new_pos = self.market_take(
            product=product,
            order_depth=order_depth,
            fair=fair,
            position=position,
            limit=limit,
            take_threshold=2.5,
            max_clip=5,
        )
        orders.extend(take_orders)

        mm_orders = self.market_make(
            product=product,
            best_bid=best_bid,
            best_ask=best_ask,
            fair=fair,
            position=new_pos,
            limit=limit,
            base_size=5,
        )
        orders.extend(mm_orders)
        return orders, mid

    # ---------- main entry point ----------

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        memory = self.load_memory(state.traderData)

        for product, order_depth in state.order_depths.items():
            if product not in self.POSITION_LIMIT:
                continue

            position = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_ash(product, order_depth, position)
            elif product == "INTARIAN_PEPPER_ROOT":
                hist = memory["mid_hist"].setdefault(product, [])
                prev_mid = memory["prev_mid"].get(product)
                orders, new_mid = self.trade_pepper(
                    product,
                    order_depth,
                    position,
                    hist,
                    prev_mid,
                )
                result[product] = orders
                if new_mid is not None:
                    hist.append(new_mid)
                    memory["mid_hist"][product] = hist[-self.MEMORY_LENGTH:]
                    memory["prev_mid"][product] = new_mid
                continue

            best_bid, _, best_ask, _ = self.best_bid_ask(order_depth)
            mid = self.mid_price(best_bid, best_ask)
            if mid is not None:
                memory["mid_hist"].setdefault(product, []).append(mid)
                memory["mid_hist"][product] = memory["mid_hist"][product][-self.MEMORY_LENGTH:]
                memory["prev_mid"][product] = mid

        trader_data = self.save_memory(memory)
        return result, conversions, trader_data
