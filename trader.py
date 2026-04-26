from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    HYDROGEL = "HYDROGEL_PACK"
    UNDERLYING = "VELVETFRUIT_EXTRACT"
    OPTION_STRIKES = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }
    PRODUCTS = (HYDROGEL, UNDERLYING, *OPTION_STRIKES.keys())
    POSITION_LIMIT = {
        HYDROGEL: 200,
        UNDERLYING: 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }
    MEMORY_LENGTH = 240
    DEFAULT_IV = 0.02
    LIVE_ROUND3_TTE = 5.0
    HISTORICAL_DAY0_TTE = 8.0
    ACTIVE_OPTION_DISTANCE = 260
    COUNTERPARTY_DECAY = 0.82
    COUNTERPARTY_SIGNAL_CAP = 8.0
    DECAY_SHORT_TARGET = {
        "VEV_5000": -300,
        "VEV_5100": -300,
        "VEV_5200": -300,
        "VEV_5300": -300,
        "VEV_5400": -300,
        "VEV_5500": -300,
    }

    def load_memory(self, trader_data: str) -> dict:
        if trader_data:
            try:
                memory = json.loads(trader_data)
                memory.setdefault("mid_hist", {})
                memory.setdefault("sigma", self.DEFAULT_IV)
                memory.setdefault("day_index", 0)
                memory.setdefault("last_ts", None)
                memory.setdefault("counterparty_signal", {})
                return memory
            except Exception:
                pass

        return {
            "mid_hist": {product: [] for product in self.PRODUCTS},
            "sigma": self.DEFAULT_IV,
            "day_index": 0,
            "last_ts": None,
            "counterparty_signal": {},
        }

    def save_memory(self, memory: dict) -> str:
        return json.dumps(memory)

    def clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], int, Optional[int], int]:
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        bid_volume = od.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_volume = od.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, bid_volume, best_ask, ask_volume

    def mid_price(self, od: OrderDepth) -> Optional[float]:
        best_bid, _, best_ask, _ = self.best_bid_ask(od)
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def ema(self, values: List[float], span: int) -> Optional[float]:
        if not values:
            return None
        alpha = 2.0 / (span + 1.0)
        out = values[0]
        for value in values[1:]:
            out = alpha * value + (1.0 - alpha) * out
        return out

    def stddev(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    def normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call_value(self, spot: float, strike: int, ttm: float, sigma: float) -> float:
        intrinsic = max(spot - strike, 0.0)
        if ttm <= 1e-9 or sigma <= 1e-9 or spot <= 0.0:
            return intrinsic
        root_t = math.sqrt(ttm)
        vol_term = sigma * root_t
        if vol_term <= 1e-9:
            return intrinsic
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * ttm) / vol_term
        d2 = d1 - vol_term
        return spot * self.normal_cdf(d1) - strike * self.normal_cdf(d2)

    def bs_delta(self, spot: float, strike: int, ttm: float, sigma: float) -> float:
        if ttm <= 1e-9 or sigma <= 1e-9 or spot <= 0.0:
            return 1.0 if spot > strike else 0.0
        root_t = math.sqrt(ttm)
        vol_term = sigma * root_t
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * ttm) / vol_term
        return self.normal_cdf(d1)

    def infer_clock(self, state: TradingState, memory: dict) -> Tuple[int, float]:
        raw_timestamp = int(state.timestamp)
        encoded_day = raw_timestamp // 1_000_000
        ts = raw_timestamp % 1_000_000

        if encoded_day > 0:
            day_index = encoded_day
        else:
            last_ts = memory.get("last_ts")
            day_index = int(memory.get("day_index", 0))
            if last_ts is not None and ts < last_ts:
                day_index += 1
            memory["last_ts"] = ts

        memory["day_index"] = day_index
        intra_round = ts / 1_000_000.0

        historical_mode = bool(memory.get("historical_mode", False))
        if encoded_day > 0:
            historical_mode = True
        memory["historical_mode"] = historical_mode

        base_tte = self.HISTORICAL_DAY0_TTE if historical_mode else self.LIVE_ROUND3_TTE
        ttm = max(0.25, base_tte - day_index - intra_round)
        return day_index, ttm

    def implied_vol(self, spot: float, strike: int, ttm: float, price: float) -> Optional[float]:
        intrinsic = max(spot - strike, 0.0)
        if price <= intrinsic + 1e-6 or ttm <= 1e-9 or spot <= 0.0:
            return None
        lo, hi = 0.01, 2.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            value = self.bs_call_value(spot, strike, ttm, mid)
            if value > price:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def update_counterparty_signal(self, state: TradingState, memory: dict) -> Dict[str, float]:
        raw_signal = memory.setdefault("counterparty_signal", {})
        signal = {product: float(raw_signal.get(product, 0.0)) * self.COUNTERPARTY_DECAY for product in self.PRODUCTS}

        for trades in state.market_trades.values():
            for trade in trades:
                product = trade.symbol
                if product not in self.PRODUCTS:
                    continue
                quantity = max(1, int(trade.quantity))
                weight = min(3.0, math.sqrt(quantity))
                impulse = 0.0

                if product in self.OPTION_STRIKES:
                    if trade.buyer == "Mark 01":
                        impulse += 1.25 * weight
                    if trade.seller == "Mark 01":
                        impulse -= 0.75 * weight
                    if trade.seller == "Mark 22":
                        impulse += 0.85 * weight
                    if trade.buyer == "Mark 22":
                        impulse -= 0.45 * weight
                    if trade.buyer == "Mark 14":
                        impulse += 0.35 * weight
                elif product == self.UNDERLYING:
                    if trade.buyer in ("Mark 01", "Mark 67", "Mark 14"):
                        impulse += 0.25 * weight
                    if trade.seller in ("Mark 01", "Mark 67", "Mark 14"):
                        impulse -= 0.20 * weight
                    if trade.buyer == "Mark 55":
                        impulse -= 0.25 * weight
                    if trade.seller == "Mark 55":
                        impulse += 0.25 * weight
                elif product == self.HYDROGEL:
                    if trade.buyer == "Mark 14":
                        impulse += 0.20 * weight
                    if trade.seller == "Mark 14":
                        impulse -= 0.20 * weight
                    if trade.seller == "Mark 22":
                        impulse -= 0.35 * weight

                if impulse:
                    signal[product] = self.clamp(
                        int(round(signal.get(product, 0.0) + impulse)),
                        -int(self.COUNTERPARTY_SIGNAL_CAP),
                        int(self.COUNTERPARTY_SIGNAL_CAP),
                    )

        memory["counterparty_signal"] = signal
        return signal

    def fit_surface_sigma(
        self,
        state: TradingState,
        underlying_mid: float,
        ttm: float,
        memory: dict,
    ) -> float:
        samples: List[Tuple[float, int]] = []
        for product, strike in self.OPTION_STRIKES.items():
            od = state.order_depths.get(product)
            if od is None:
                continue
            option_mid = self.mid_price(od)
            if option_mid is None:
                continue
            distance = abs(strike - underlying_mid)
            if distance <= 250 and option_mid > max(1.0, max(underlying_mid - strike, 0.0)):
                iv = self.implied_vol(underlying_mid, strike, ttm, option_mid)
                if iv is not None:
                    samples.append((iv, distance))

        if not samples:
            return float(memory.get("sigma", self.DEFAULT_IV))

        samples.sort(key=lambda item: item[1])
        core = [iv for iv, _ in samples[:4]]
        core.sort()
        fitted = core[len(core) // 2]
        prev = float(memory.get("sigma", self.DEFAULT_IV))
        sigma = 0.15 * prev + 0.85 * fitted
        sigma = min(1.5, max(0.001, sigma))
        memory["sigma"] = sigma
        return sigma

    def trade_mean_reverter(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        history: List[float],
        anchor: float,
        target_position: int,
        base_take_width: float,
        base_make_width: float,
        round_progress: float,
    ) -> Tuple[List[Order], Optional[float]]:
        orders: List[Order] = []
        limit = self.POSITION_LIMIT[product]
        mid = self.mid_price(od)
        if mid is None:
            return orders, None

        best_bid, _, best_ask, _ = self.best_bid_ask(od)
        vol = max(1.0, self.stddev((history + [mid])[-50:]))
        late_stage = round_progress >= 0.85
        if round_progress >= 0.92:
            target_position = 0
        fair = anchor - 0.30 * position
        take_width = max(base_take_width, 0.30 * vol)
        make_width = max(base_make_width, 0.20 * vol)
        if late_stage:
            take_width += 0.8
            make_width += 0.8

        pos = position
        for ask in sorted(od.sell_orders):
            available = -od.sell_orders[ask]
            if ask <= fair - take_width and pos < limit:
                qty = min(available, limit - pos, 10 if not late_stage else 6)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty
            else:
                break

        for bid in sorted(od.buy_orders, reverse=True):
            available = od.buy_orders[bid]
            if bid >= fair + take_width and pos > -limit:
                qty = min(available, pos + limit, 10 if not late_stage else 6)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty
            else:
                break

        buy_room = max(0, limit - pos)
        sell_room = max(0, limit + pos)
        bid_quote = int(math.floor(fair - make_width))
        ask_quote = int(math.ceil(fair + make_width))

        if best_bid is not None and best_ask is not None and best_ask - best_bid >= 2:
            bid_quote = min(best_bid + 1, bid_quote, best_ask - 1)
            ask_quote = max(best_ask - 1, ask_quote, best_bid + 1)
        else:
            if best_bid is not None:
                bid_quote = min(best_bid, bid_quote)
            if best_ask is not None:
                ask_quote = max(best_ask, ask_quote)
        if bid_quote >= ask_quote:
            bid_quote = int(math.floor(fair)) - 1
            ask_quote = int(math.ceil(fair)) + 1

        bid_size = min(8 if not late_stage else 5, buy_room)
        ask_size = min(8 if not late_stage else 5, sell_room)

        if pos > target_position:
            bid_size = min(bid_size, 2)
            ask_size = min(max(6, ask_size), sell_room)
        elif pos < target_position:
            ask_size = min(ask_size, 2)
            bid_size = min(max(6, bid_size), buy_room)

        if late_stage:
            if position > 0:
                bid_size = 0
                ask_size = min(max(ask_size, min(12, sell_room)), sell_room)
            elif position < 0:
                ask_size = 0
                bid_size = min(max(bid_size, min(12, buy_room)), buy_room)

        if bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))
        if ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders, mid

    def trade_underlying(
        self,
        od: OrderDepth,
        position: int,
        history: List[float],
        hedge_target: int,
        round_progress: float,
    ) -> Tuple[List[Order], Optional[float]]:
        mid = self.mid_price(od)
        if mid is None:
            return [], None

        best_bid, _, best_ask, _ = self.best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return [], mid

        # Do not run a standalone underlying strategy; only flatten residual
        # inventory late in the round.
        target = 0 if round_progress >= 0.80 else position
        gap = target - position
        orders: List[Order] = []
        if gap > 0:
            price = best_ask
            qty = min(gap, 20)
            if qty > 0:
                orders.append(Order(self.UNDERLYING, price, qty))
        elif gap < 0:
            price = best_bid
            qty = min(-gap, 20)
            if qty > 0:
                orders.append(Order(self.UNDERLYING, price, -qty))

        return orders, mid

    def trade_hydrogel(
        self,
        od: OrderDepth,
        position: int,
        history: List[float],
        round_progress: float,
        fair_shift: float,
    ) -> Tuple[List[Order], Optional[float]]:
        mid = self.mid_price(od)
        if mid is None:
            return [], None
        if position == 0:
            return [], mid

        best_bid, _, best_ask, _ = self.best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return [], mid

        orders: List[Order] = []
        flatten_phase = round_progress >= 0.50
        if position > 0:
            price = best_bid if flatten_phase else max(best_bid, int(math.floor(mid + fair_shift)))
            qty = min(position, 20 if flatten_phase else 8)
            if qty > 0:
                orders.append(Order(self.HYDROGEL, price, -qty))
        elif position < 0:
            price = best_ask if flatten_phase else min(best_ask, int(math.ceil(mid + fair_shift)))
            qty = min(-position, 20 if flatten_phase else 8)
            if qty > 0:
                orders.append(Order(self.HYDROGEL, price, qty))
        return orders, mid

    def trade_option(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        spot_fair: float,
        sigma: float,
        ttm: float,
        round_progress: float,
        fair_shift: float,
    ) -> Tuple[List[Order], Optional[float], float]:
        strike = self.OPTION_STRIKES[product]
        limit = self.POSITION_LIMIT[product]
        option_mid = self.mid_price(od)
        if option_mid is None:
            return [], None, 0.0

        theoretical = self.bs_call_value(spot_fair, strike, ttm, sigma)
        delta = self.bs_delta(spot_fair, strike, ttm, sigma)
        distance = abs(strike - spot_fair)

        best_bid, _, best_ask, _ = self.best_bid_ask(od)
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else 2
        orders: List[Order] = []
        pos = position

        if product in self.DECAY_SHORT_TARGET and round_progress < 0.92 and best_bid is not None and best_bid > 0:
            short_target = self.DECAY_SHORT_TARGET[product]
            if pos > short_target:
                available = od.buy_orders.get(best_bid, 0)
                qty = min(available, pos - short_target, 15)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    pos -= qty

        if distance > self.ACTIVE_OPTION_DISTANCE:
            return orders, option_mid, delta

        model_cap = option_mid + max(2.0, 0.75 * spread + max(0.0, fair_shift))
        model_floor = option_mid - max(2.0, 0.75 * spread + max(0.0, -fair_shift))
        theoretical = min(model_cap, max(model_floor, theoretical))

        # Lean into the observable edge: near-spot vouchers in the historical
        # tests have positive expectancy with passive quoting. Bias the fair
        # slightly toward the richer theoretical value but keep it anchored to
        # the tape.
        fair = 0.68 * theoretical + 0.32 * option_mid + fair_shift
        late_stage = round_progress >= 0.85
        if round_progress >= 0.92:
            fair = 0.45 * theoretical + 0.55 * option_mid + 0.35 * fair_shift
        inventory_penalty = (0.03 if not late_stage else 0.06) * position
        reservation = fair - inventory_penalty

        edge = max(1.0, 0.55 * spread)
        quote_edge = max(1.0, edge + 0.35)

        buy_room = max(0, limit - pos)
        sell_room = max(0, limit + pos)
        bid_quote = int(math.floor(reservation - quote_edge))
        ask_quote = int(math.ceil(reservation + quote_edge))

        if best_bid is not None and best_ask is not None and best_ask - best_bid >= 2:
            bid_quote = min(best_bid + 1, bid_quote, best_ask - 1)
            ask_quote = max(best_ask - 1, ask_quote, best_bid + 1)
        else:
            if best_bid is not None:
                bid_quote = min(best_bid, bid_quote)
            if best_ask is not None:
                ask_quote = max(best_ask, ask_quote)
        bid_quote = max(0, bid_quote)
        ask_quote = max(bid_quote + 1, ask_quote)

        size_cap = 10 if distance <= 150 else 7
        if late_stage:
            size_cap = max(3, size_cap - 3)

        bid_size = min(size_cap, buy_room)
        ask_size = min(size_cap, sell_room)
        if product != "VEV_5000":
            bid_size = 0
        if position > 0:
            bid_size = min(bid_size, 2)
            ask_size = min(max(ask_size, size_cap + 4), sell_room)
        elif position < 0:
            ask_size = min(ask_size, 2)
            bid_size = min(max(bid_size, size_cap + 4), buy_room)
        if product != "VEV_5000" and position >= 0:
            bid_size = 0
        if position <= 0:
            ask_size = 0
        if product in self.DECAY_SHORT_TARGET:
            bid_size = 0

        if late_stage:
            if position > 0:
                bid_size = 0
                ask_size = min(max(ask_size, size_cap + 6), sell_room)
            elif position < 0:
                ask_size = 0
                bid_size = 0 if product in self.DECAY_SHORT_TARGET else min(max(bid_size, size_cap + 6), buy_room)

        if round_progress >= 0.97:
            bid_size = 0 if position >= 0 or product in self.DECAY_SHORT_TARGET else min(max(12, bid_size), buy_room)
            ask_size = 0 if position <= 0 else min(max(12, ask_size), sell_room)

        # Skip the least profitable active strike seen in the logs.
        if product == "VEV_5300" and ttm <= 5.1:
            bid_size = 0
            ask_size = 0

        if fair > 0.5 or strike <= int(spot_fair + 80):
            if bid_size > 0:
                orders.append(Order(product, bid_quote, bid_size))
            if ask_size > 0:
                orders.append(Order(product, ask_quote, -ask_size))

        return orders, option_mid, delta

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        memory = self.load_memory(state.traderData)
        day_index, ttm = self.infer_clock(state, memory)
        round_progress = (int(state.timestamp) % 1_000_000) / 1_000_000.0

        mid_hist = memory["mid_hist"]
        positions = {product: state.position.get(product, 0) for product in self.PRODUCTS}
        counterparty_signal = self.update_counterparty_signal(state, memory)

        underlying_od = state.order_depths.get(self.UNDERLYING)
        hydro_od = state.order_depths.get(self.HYDROGEL)
        underlying_mid = self.mid_price(underlying_od) if underlying_od is not None else None

        sigma = float(memory.get("sigma", self.DEFAULT_IV))
        if underlying_mid is not None:
            sigma = self.fit_surface_sigma(state, underlying_mid, ttm, memory)

        option_delta_exposure = 0.0
        spot_for_delta = underlying_mid if underlying_mid is not None else 5250.0
        for product, strike in self.OPTION_STRIKES.items():
            option_delta_exposure += positions[product] * self.bs_delta(spot_for_delta, strike, ttm, sigma)

        hedge_target = 0

        if hydro_od is not None:
            orders, new_mid = self.trade_hydrogel(
                hydro_od,
                positions[self.HYDROGEL],
                mid_hist.setdefault(self.HYDROGEL, []),
                round_progress,
                0.35 * counterparty_signal.get(self.HYDROGEL, 0.0),
            )
            result[self.HYDROGEL] = orders
            if new_mid is not None:
                hist = mid_hist.setdefault(self.HYDROGEL, [])
                hist.append(new_mid)
                mid_hist[self.HYDROGEL] = hist[-self.MEMORY_LENGTH :]

        spot_fair = underlying_mid if underlying_mid is not None else 5250.0
        if underlying_od is not None:
            orders, new_mid = self.trade_underlying(
                underlying_od,
                positions[self.UNDERLYING],
                mid_hist.setdefault(self.UNDERLYING, []),
                hedge_target,
                round_progress,
            )
            result[self.UNDERLYING] = orders
            if new_mid is not None:
                hist = mid_hist.setdefault(self.UNDERLYING, [])
                hist.append(new_mid)
                mid_hist[self.UNDERLYING] = hist[-self.MEMORY_LENGTH :]
                spot_fair = new_mid

        spot_hist = mid_hist.setdefault(self.UNDERLYING, [])
        if spot_hist:
            fast = self.ema(spot_hist[-20:], 20) or spot_fair
            slow = self.ema(spot_hist[-80:], 80) or spot_fair
            spot_fair = 0.65 * fast + 0.35 * slow

        for product in self.OPTION_STRIKES:
            od = state.order_depths.get(product)
            if od is None:
                continue
            orders, option_mid, _ = self.trade_option(
                product,
                od,
                positions[product],
                spot_fair,
                sigma,
                ttm,
                round_progress,
                counterparty_signal.get(product, 0.0),
            )
            result[product] = orders
            if option_mid is not None:
                hist = mid_hist.setdefault(product, [])
                hist.append(option_mid)
                mid_hist[product] = hist[-self.MEMORY_LENGTH :]

        memory["day_index"] = day_index
        memory["sigma"] = sigma
        return result, conversions, self.save_memory(memory)
