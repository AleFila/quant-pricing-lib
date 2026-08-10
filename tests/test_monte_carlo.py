import pytest
import numpy as np
from quant_pricing.monte_carlo import MonteCarloPricer
from quant_pricing.black_scholes import BlackScholesPricer


def test_init_valid():
    pricer = MonteCarloPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02)
    assert pricer.S0 == 100.0
    assert pricer.K == 100.0
    assert pricer.T == 1.0
    assert pricer.r == 0.05
    assert pricer.sigma == 0.2
    assert pricer.q == 0.02


def test_init_invalid_values():
    with pytest.raises(ValueError, match=r"S0 and K must be positive."):
        MonteCarloPricer(S0=0, K=100, T=1, r=0.05, sigma=0.2)

    with pytest.raises(ValueError, match=r"S0 and K must be positive."):
        MonteCarloPricer(S0=100, K=-5, T=1, r=0.05, sigma=0.2)

    with pytest.raises(ValueError, match=r"T must be positive."):
        MonteCarloPricer(S0=100, K=100, T=0, r=0.05, sigma=0.2)

    with pytest.raises(ValueError, match=r"sigma must be positive."):
        MonteCarloPricer(S0=100, K=100, T=1, r=0.05, sigma=-0.2)


def test_reproducibility_with_seed():
    pricer = MonteCarloPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    p1, se1 = pricer.price("call", num_simulations=50_000, seed=42)
    p2, se2 = pricer.price("call", num_simulations=50_000, seed=42)
    assert p1 == p2
    assert se1 == se2


def test_convergence_to_black_scholes():
    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.01
    mc_pricer = MonteCarloPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    bs_pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)

    # Call comparison
    mc_call_price, call_se = mc_pricer.price("call", num_simulations=200_000, seed=123)
    bs_call_price = bs_pricer.price("call")
    # MC estimate should be within 3 standard errors of analytical BS price
    assert abs(mc_call_price - bs_call_price) < 3.0 * call_se

    # Put comparison
    mc_put_price, put_se = mc_pricer.price("put", num_simulations=200_000, seed=123)
    bs_put_price = bs_pricer.price("put")
    assert abs(mc_put_price - bs_put_price) < 3.0 * put_se


def test_put_call_parity():
    S0, K, T, r, sigma, q = 100.0, 95.0, 0.5, 0.04, 0.3, 0.02
    mc_pricer = MonteCarloPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    call_price, call_se = mc_pricer.price("call", num_simulations=200_000, seed=999)
    put_price, put_se = mc_pricer.price("put", num_simulations=200_000, seed=999)

    # Theoretical parity: S0 * exp(-q*T) - K * exp(-r*T)
    theoretical_parity = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    mc_parity_diff = call_price - put_price

    # MC difference should converge to theoretical parity within sampling tolerance (~0.5%)
    assert pytest.approx(mc_parity_diff, rel=0.01) == theoretical_parity


def test_standard_error_reduction():
    pricer = MonteCarloPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    _, se_small = pricer.price("call", num_simulations=10_000, seed=42)
    _, se_large = pricer.price("call", num_simulations=100_000, seed=42)

    # Increasing simulations by 10x should decrease SE by ~sqrt(10) (~3.16x)
    assert se_large < se_small
    ratio = se_small / se_large
    assert pytest.approx(ratio, rel=0.15) == np.sqrt(10)


def test_case_insensitivity():
    pricer = MonteCarloPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    p_call_lower, _ = pricer.price("call", num_simulations=1000, seed=1)
    p_call_upper, _ = pricer.price("CALL", num_simulations=1000, seed=1)
    assert p_call_lower == p_call_upper

    p_put_lower, _ = pricer.price("put", num_simulations=1000, seed=1)
    p_put_upper, _ = pricer.price("PUT", num_simulations=1000, seed=1)
    assert p_put_lower == p_put_upper


def test_invalid_option_type():
    pricer = MonteCarloPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        pricer.price("binary", num_simulations=100)
