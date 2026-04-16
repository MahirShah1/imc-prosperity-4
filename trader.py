from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    POSITION_LIMIT = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }
    MEMORY_LENGTH = 200

    # ── helpers ───────────────────────────────────────────────────────────────

    def load_memory(self, trader_data: str) -> dict:
        if trader_data:
            try:
                m = json.loads(trader_data)
                m.setdefault("mid_hist", {})
                return m
            except Exception:
                pass
        return {"mid_hist": {p: [] for p in self.POSITION_LIMIT}}

    def save_memory(self, memory: dict) -> str:
        return json.dumps(memory)

    def best_bid_ask(self, od: OrderDepth):
        bb = bv = ba = av = None
        if od.buy_orders:
            bb = max(od.buy_orders)
            bv = od.buy_orders[bb]
        if od.sell_orders:
            ba = min(od.sell_orders)
            av = od.sell_orders[ba]
        return bb, bv or 0, ba, av or 0

    def mid_price(self, bb, ba) -> Optional[float]:
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return float(bb)
        if ba is not None:
            return float(ba)
        return None

    def ema(self, vals: list, span: int) -> Optional[float]:
        if not vals:
            return None
        a = 2.0 / (span + 1)
        e = vals[0]
        for v in vals[1:]:
            e = a * v + (1 - a) * e
        return e

    def stddev(self, vals: list) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

    # ── PEPPER: strong uptrend, always stay max long ──────────────────────────
    #
    # Data pattern:
    #   - price rises ~+1000 per day (slope ≈ 0.001/timestamp within 0..999900)
    #   - jumps +1000 overnight between days
    #   - holding +20 from day -2 to end of day 0 captures ~60,000 PnL
    #
    # Strategy:
    #   1. Accumulate to +20 as quickly as possible
    #   2. Hold; only sell passively into large spikes above fair
    #   3. Fair value = momentum-adjusted mid, with hard long bias

    def trade_pepper(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        hist: list,
    ) -> Tuple[List[Order], Optional[float]]:
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        bb, bv, ba, av = self.best_bid_ask(od)
        mid = self.mid_price(bb, ba)
        if mid is None:
            return [], None

        # EMA-based momentum (captures the persistent uptrend)
        ema_fast = self.ema(hist[-10:], 10) if len(hist) >= 3 else mid
        ema_slow = self.ema(hist[-40:], 40) if len(hist) >= 10 else mid
        if ema_fast is None:
            ema_fast = mid
        if ema_slow is None:
            ema_slow = mid

        momentum = ema_fast - ema_slow  # positive in uptrend
        velocity = (mid - hist[-1]) if hist else 0.0

        # Base fair value with trend and velocity
        fair = mid + 0.60 * momentum + 0.20 * velocity

        # Asymmetric inventory bias: aggressively long, never short
        # Thresholds scaled to limit (80)
        if position < 0:
            fair += 10.0               # emergency buy-back when short
        elif position < limit * 0.10:
            fair += 5.0                # build fast from near-flat
        elif position < limit * 0.40:
            fair += 3.0                # continue accumulating
        elif position < limit * 0.70:
            fair += 1.5                # approaching limit, still nudge long
        # above 70% of limit: momentum signal alone drives buys

        # ── market-take: hit asks up to fair ─────────────────────────────────
        pos = position
        for ask in sorted(od.sell_orders.keys()):
            vol = -od.sell_orders[ask]
            if ask <= fair and pos < limit:
                qty = min(vol, limit - pos, 20)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty
            else:
                break

        # Sell ONLY into large spikes (≥10 above fair) and only from a long base
        SPIKE_THRESHOLD = 10.0
        MIN_LONG_FLOOR = limit // 4   # never go below 25% long on a spike
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            vol = od.buy_orders[bid]
            if bid >= fair + SPIKE_THRESHOLD and pos > MIN_LONG_FLOOR:
                qty = min(vol, pos - MIN_LONG_FLOOR, 10)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty
            else:
                break

        # ── market-make: aggressive bid, very passive ask ─────────────────────
        if bb is not None and ba is not None:
            bid_q = min(bb + 1, math.floor(fair - 0.5))
            ask_q = max(ba - 1, math.ceil(fair + SPIKE_THRESHOLD))

            buy_cap = max(0, limit - pos)
            # Only quote asks from surplus above 50% of limit
            sell_cap = max(0, pos - limit // 2)

            buy_sz = min(20, buy_cap)
            sell_sz = min(5, sell_cap)

            if bid_q < ask_q:
                if buy_sz > 0:
                    orders.append(Order(product, bid_q, buy_sz))
                if sell_sz > 0:
                    orders.append(Order(product, ask_q, -sell_sz))

        return orders, mid

    # ── ASH: tight mean-reversion / market-making around 10 000 ───────────────
    #
    # Data pattern:
    #   - true fair value = 10000, perfectly stable across all days
    #   - std dev ≈ 5.35, bid-ask spread ≈ 16
    #
    # Strategy:
    #   1. Quote inside the spread symmetrically (capture ~14 per round trip)
    #   2. Bollinger-band aggressive takes when price deviates > 1.5 std
    #   3. Symmetric inventory penalty keeps position near 0

    ASH_FAIR = 10_000.0

    def trade_ash(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        hist: list,
    ) -> List[Order]:
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        bb, bv, ba, av = self.best_bid_ask(od)
        mid = self.mid_price(bb, ba)
        if mid is None:
            return []

        rolling_std = self.stddev(hist[-50:]) if len(hist) >= 20 else 5.35
        if rolling_std < 1.5:
            rolling_std = 5.35

        deviation = mid - self.ASH_FAIR
        z = deviation / rolling_std

        # Mean-reversion fair value (pulls toward 10000 proportionally)
        fair = self.ASH_FAIR - 0.40 * deviation - 0.25 * position

        # Bollinger: tighten take threshold when price is extreme
        take_threshold = 1.5
        if abs(z) > 1.5:
            take_threshold = 0.5  # much more aggressive at extremes

        # ── market-take ────────────────────────────────────────────────────────
        pos = position
        for ask in sorted(od.sell_orders.keys()):
            vol = -od.sell_orders[ask]
            if ask <= fair - take_threshold and pos < limit:
                qty = min(vol, limit - pos, 20)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty
            else:
                break

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            vol = od.buy_orders[bid]
            if bid >= fair + take_threshold and pos > -limit:
                qty = min(vol, pos + limit, 20)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty
            else:
                break

        # ── market-make: symmetric inside spread ───────────────────────────────
        if bb is not None and ba is not None:
            bid_q = min(bb + 1, math.floor(fair - 1.0))
            ask_q = max(ba - 1, math.ceil(fair + 1.0))

            if bid_q >= ask_q:
                bid_q = math.floor(fair) - 1
                ask_q = math.ceil(fair) + 1

            buy_cap = max(0, limit - pos)
            sell_cap = max(0, limit + pos)

            buy_sz = min(15, buy_cap)
            sell_sz = min(15, sell_cap)

            # Scale down the heavy side to avoid inventory drift
            if pos > 0.5 * limit:
                buy_sz = min(5, buy_cap)
            if pos < -0.5 * limit:
                sell_sz = min(5, sell_cap)

            if pos >= limit:
                buy_sz = 0
            if pos <= -limit:
                sell_sz = 0

            if bid_q < ask_q:
                if buy_sz > 0:
                    orders.append(Order(product, bid_q, buy_sz))
                if sell_sz > 0:
                    orders.append(Order(product, ask_q, -sell_sz))

        return orders

    # ── main entry point ───────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        memory = self.load_memory(state.traderData)

        for product, od in state.order_depths.items():
            if product not in self.POSITION_LIMIT:
                continue

            pos = state.position.get(product, 0)
            hist = memory["mid_hist"].setdefault(product, [])

            if product == "INTARIAN_PEPPER_ROOT":
                orders, new_mid = self.trade_pepper(product, od, pos, hist)
                result[product] = orders
                if new_mid is not None:
                    hist.append(new_mid)
                    memory["mid_hist"][product] = hist[-self.MEMORY_LENGTH:]
                continue

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_ash(product, od, pos, hist)

            bb, _, ba, _ = self.best_bid_ask(od)
            m = self.mid_price(bb, ba)
            if m is not None:
                hist.append(m)
                memory["mid_hist"][product] = hist[-self.MEMORY_LENGTH:]

        return result, conversions, self.save_memory(memory)
