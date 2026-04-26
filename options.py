"""Black-Scholes pricing, greeks, IV, and time-to-expiry helpers.

Used by ``backtest.py`` and ``research.ipynb``. ``trader.py`` carries a
verbatim copy of the BS block between its ``BS_INLINE_BEGIN/END`` markers,
matching the same block here. ``tests/test_options.py`` enforces the
mirror.

Conventions:
    S    underlying spot
    K    strike
    T    time to expiry in *days* (Solvenarian)
    sigma volatility per sqrt(day) — matches T's unit
    r    continuously compounded rate per day, default 0
    call True for call, False for put
"""

from __future__ import annotations

import math


# BS_INLINE_BEGIN
import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float):
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(S: float, K: float, T: float, sigma: float,
             r: float = 0.0, call: bool = True) -> float:
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
        return intrinsic * math.exp(-r * max(T, 0.0))
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    disc = math.exp(-r * T)
    if call:
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def delta(S: float, K: float, T: float, sigma: float,
          r: float = 0.0, call: bool = True) -> float:
    if T <= 0.0 or sigma <= 0.0:
        if call:
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        return -1.0 if S < K else (-0.5 if S == K else 0.0)
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return _norm_cdf(d1) if call else _norm_cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return S * _norm_pdf(d1) * math.sqrt(T)


def theta(S: float, K: float, T: float, sigma: float,
          r: float = 0.0, call: bool = True) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    disc = math.exp(-r * T)
    first = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if call:
        return first - r * K * disc * _norm_cdf(d2)
    return first + r * K * disc * _norm_cdf(-d2)
# BS_INLINE_END


# ---------- Time to expiry ----------

# Round 3: VEVs expire 7 Solvenarian days after day 1.
# In-day timestamps span 0..999_900 → one day = 1_000_000 ts units.
TS_PER_DAY = 1_000_000
TTE_ANCHOR_DAY = 1
TTE_TOTAL_DAYS = 7


def tte_days(day: int, timestamp: int,
             anchor_day: int = TTE_ANCHOR_DAY,
             total_days: int = TTE_TOTAL_DAYS,
             ts_per_day: int = TS_PER_DAY) -> float:
    """Days remaining until expiry. Clamped at 0."""
    elapsed = (day - anchor_day) + (timestamp / ts_per_day)
    return max(total_days - elapsed, 0.0)


# ---------- Implied volatility ----------

def implied_vol(price: float, S: float, K: float, T: float,
                r: float = 0.0, call: bool = True,
                sigma_lo: float = 1e-4, sigma_hi: float = 5.0,
                tol: float = 1e-6) -> float:
    """Solve BS for sigma. Returns ``nan`` if no solution in the bracket.

    Uses scipy's Brent solver when available; falls back to bisection.
    """
    if T <= 0.0 or price <= 0.0:
        return math.nan

    intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
    if price < intrinsic * math.exp(-r * T) - tol:
        return math.nan

    def f(sigma: float) -> float:
        return bs_price(S, K, T, sigma, r=r, call=call) - price

    if f(sigma_lo) > 0:
        return sigma_lo
    if f(sigma_hi) < 0:
        return math.nan

    try:
        from scipy.optimize import brentq
        return float(brentq(f, sigma_lo, sigma_hi, xtol=tol, maxiter=200))
    except Exception:
        return _bisect(f, sigma_lo, sigma_hi, tol)


def _bisect(f, lo: float, hi: float, tol: float, max_iter: int = 200) -> float:
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return math.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) * 0.5 < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
