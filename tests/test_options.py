"""Tests for options.py and the BS block inlined into trader.py.

Run with: ``.venv/bin/python -m pytest -q``
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

import options
import trader

ROOT = Path(__file__).resolve().parent
MARKER_RE = re.compile(
    r"# BS_INLINE_BEGIN\n(.*?)\n# BS_INLINE_END",
    re.DOTALL,
)


def _extract_block(path: Path) -> str:
    text = path.read_text()
    m = MARKER_RE.search(text)
    assert m is not None, f"No BS_INLINE markers in {path}"
    return m.group(1)


def test_inline_block_matches_canonical():
    canonical = _extract_block(ROOT / "options.py")
    inlined = _extract_block(ROOT / "trader.py")
    assert canonical == inlined, (
        "BS block in trader.py has drifted from options.py. "
        "Re-inline before submitting."
    )


@pytest.mark.parametrize("S,K,T,sigma,r,call", [
    (100, 100, 30, 0.2, 0.0, True),
    (100, 110, 30, 0.2, 0.0, True),
    (100, 90, 30, 0.2, 0.0, False),
    (5246, 5000, 7, 0.05, 0.0, True),
    (5246, 5500, 1, 0.04, 0.0, True),
    (5246, 5500, 0.0, 0.05, 0.0, True),  # expired
    (5246, 5500, 7, 0.0, 0.0, True),     # zero vol
])
def test_numeric_parity(S, K, T, sigma, r, call):
    expected = options.bs_price(S, K, T, sigma, r=r, call=call)
    actual = trader.bs_price(S, K, T, sigma, r=r, call=call)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_put_call_parity():
    S, K, T, sigma, r = 5246, 5300, 5, 0.05, 0.01
    c = options.bs_price(S, K, T, sigma, r=r, call=True)
    p = options.bs_price(S, K, T, sigma, r=r, call=False)
    assert (c - p) == pytest.approx(S - K * math.exp(-r * T), rel=1e-10, abs=1e-10)


def test_implied_vol_round_trip():
    S, K, T, true_sigma = 5246, 5200, 5, 0.06
    price = options.bs_price(S, K, T, true_sigma)
    iv = options.implied_vol(price, S, K, T)
    assert iv == pytest.approx(true_sigma, abs=1e-5)


def test_tte_anchor():
    assert options.tte_days(1, 0) == 7.0
    assert options.tte_days(8, 0) == 0.0
    assert options.tte_days(4, 500_000) == pytest.approx(3.5)
