import pytest
import numpy as np
from quant_pricing.black_scholes import BlackScholesPricer
from quant_pricing.implied_volatility import implied_volatility


def test_implied_volatility_roundtrip_call():
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.01
    known_sigma = 0.25

    # Compute exact Black-Scholes market price
    pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=known_sigma, q=q)
    market_price = pricer.price("call")

    iv = implied_volatility(market_price=market_price, S0=S0, K=K, T=T, r=r, q=q, option_type="call")
    assert pytest.approx(iv, abs=1e-5) == known_sigma


def test_implied_volatility_roundtrip_put():
    S0, K, T, r, q = 120.0, 100.0, 0.5, 0.03, 0.0
    known_sigma = 0.40

    pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=known_sigma, q=q)
    market_price = pricer.price("put")

    iv = implied_volatility(market_price=market_price, S0=S0, K=K, T=T, r=r, q=q, option_type="put")
    assert pytest.approx(iv, abs=1e-5) == known_sigma


def test_implied_volatility_various_sigmas():
    S0, K, T, r, q = 100.0, 105.0, 2.0, 0.02, 0.01
    for known_sigma in [0.05, 0.15, 0.30, 0.60, 0.90]:
        bs_price = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=known_sigma, q=q).price("call")
        iv = implied_volatility(market_price=bs_price, S0=S0, K=K, T=T, r=r, q=q, option_type="call")
        assert pytest.approx(iv, abs=1e-4) == known_sigma


def test_invalid_market_price():
    # Zero or negative market price returns NaN
    assert np.isnan(implied_volatility(market_price=0.0, S0=100, K=100, T=1, r=0.05))
    assert np.isnan(implied_volatility(market_price=-2.5, S0=100, K=100, T=1, r=0.05))


def test_case_insensitivity():
    S0, K, T, r, known_sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_call = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=known_sigma).price("call")
    bs_put = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=known_sigma).price("put")

    assert pytest.approx(implied_volatility(bs_call, S0, K, T, r, option_type="CALL"), abs=1e-5) == known_sigma
    assert pytest.approx(implied_volatility(bs_put, S0, K, T, r, option_type="Put"), abs=1e-5) == known_sigma


def test_invalid_parameters():
    # Invalid parameters (e.g. negative S0 or K) raise ValueError from BlackScholesPricer
    with pytest.raises(ValueError):
        implied_volatility(market_price=10.0, S0=-100, K=100, T=1, r=0.05)

    with pytest.raises(ValueError):
        implied_volatility(market_price=10.0, S0=100, K=100, T=-1, r=0.05)

    with pytest.raises(ValueError):
        implied_volatility(market_price=10.0, S0=100, K=100, T=1, r=0.05, option_type="invalid")


def test_unreachable_target_returns_nan():
    # If market price is below intrinsic value or impossible, solver fails and returns NaN
    # For S0=100, K=50, T=1, r=0.05, intrinsic call value is ~ 52.43.
    # Market price of 1.0 is far below intrinsic value.
    iv = implied_volatility(market_price=1.0, S0=100, K=50, T=1, r=0.05, option_type="call", max_iter=20)
    assert np.isnan(iv)
