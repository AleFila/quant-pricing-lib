import pytest
import numpy as np
from quant_pricing.heston import HestonMonteCarloPricer
from quant_pricing.black_scholes import BlackScholesPricer


def test_init_valid():
    pricer = HestonMonteCarloPricer(
        S0=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, q=0.01
    )
    assert pricer.S0 == 100.0
    assert pricer.K == 100.0
    assert pricer.T == 1.0
    assert pricer.r == 0.05
    assert pricer.v0 == 0.04
    assert pricer.kappa == 2.0
    assert pricer.theta == 0.04
    assert pricer.xi == 0.3
    assert pricer.rho == -0.7
    assert pricer.q == 0.01


def test_init_invalid_values():
    # S0, K, T positive check
    with pytest.raises(ValueError, match=r"S0, K and T must be positive."):
        HestonMonteCarloPricer(S0=0, K=100, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.5)

    with pytest.raises(ValueError, match=r"S0, K and T must be positive."):
        HestonMonteCarloPricer(S0=100, K=-10, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.5)

    with pytest.raises(ValueError, match=r"S0, K and T must be positive."):
        HestonMonteCarloPricer(S0=100, K=100, T=0, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.5)

    # Variance parameters positive check
    with pytest.raises(ValueError, match=r"The variance parameters \(v0, theta, kappa, xi\) must be positive."):
        HestonMonteCarloPricer(S0=100, K=100, T=1, r=0.05, v0=-0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.5)

    with pytest.raises(ValueError, match=r"The variance parameters \(v0, theta, kappa, xi\) must be positive."):
        HestonMonteCarloPricer(S0=100, K=100, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.0, rho=-0.5)

    # Correlation rho range check [-1, 1]
    with pytest.raises(ValueError, match=r"The correlation rho must be between -1.0 and 1.0."):
        HestonMonteCarloPricer(S0=100, K=100, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-1.5)

    with pytest.raises(ValueError, match=r"The correlation rho must be between -1.0 and 1.0."):
        HestonMonteCarloPricer(S0=100, K=100, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=1.2)


def test_reproducibility_with_seed():
    pricer = HestonMonteCarloPricer(
        S0=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
    )
    p1, se1 = pricer.price("call", num_simulations=20_000, num_steps=50, seed=42)
    p2, se2 = pricer.price("call", num_simulations=20_000, num_steps=50, seed=42)

    assert p1 == p2
    assert se1 == se2


def test_degeneration_to_black_scholes():
    """When vol-of-vol xi -> 0 and v0 == theta == sigma^2, Heston converges to Black-Scholes."""
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    sigma = 0.20
    v0 = sigma ** 2  # 0.04
    theta = sigma ** 2

    heston_pricer = HestonMonteCarloPricer(
        S0=S0, K=K, T=T, r=r, v0=v0, kappa=2.0, theta=theta, xi=1e-4, rho=0.0, q=q
    )
    bs_pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)

    call_h, se_h = heston_pricer.price("call", num_simulations=100_000, num_steps=100, seed=123)
    call_bs = bs_pricer.price("call")

    # Heston price estimate should be within 3 standard errors of Black-Scholes price
    assert abs(call_h - call_bs) < 3.0 * se_h


def test_put_call_parity():
    S0, K, T, r, q = 100.0, 95.0, 0.5, 0.03, 0.01
    pricer = HestonMonteCarloPricer(
        S0=S0, K=K, T=T, r=r, v0=0.04, kappa=1.5, theta=0.04, xi=0.2, rho=-0.5, q=q
    )

    c_price, _ = pricer.price("call", num_simulations=50_000, num_steps=50, seed=777)
    p_price, _ = pricer.price("put", num_simulations=50_000, num_steps=50, seed=777)

    theoretical_parity = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    assert pytest.approx(c_price - p_price, rel=0.01) == theoretical_parity


def test_case_insensitivity():
    pricer = HestonMonteCarloPricer(
        S0=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5
    )

    p_call_lower, _ = pricer.price("call", num_simulations=1000, num_steps=20, seed=1)
    p_call_upper, _ = pricer.price("CALL", num_simulations=1000, num_steps=20, seed=1)
    assert p_call_lower == p_call_upper

    p_put_lower, _ = pricer.price("put", num_simulations=1000, num_steps=20, seed=1)
    p_put_upper, _ = pricer.price("PUT", num_simulations=1000, num_steps=20, seed=1)
    assert p_put_lower == p_put_upper


def test_invalid_option_type():
    pricer = HestonMonteCarloPricer(
        S0=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5
    )
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        pricer.price("exotic", num_simulations=100, num_steps=10)
