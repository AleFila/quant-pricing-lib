import pytest
import numpy as np
from quant_pricing.black_scholes import BlackScholesPricer


def test_init_valid():
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02)
    assert pricer.S0 == 100.0
    assert pricer.K == 100.0
    assert pricer.T == 1.0
    assert pricer.r == 0.05
    assert pricer.sigma == 0.2
    assert pricer.q == 0.02


def test_init_invalid_values():
    with pytest.raises(ValueError, match=r"Stock price \(S0\) and strike \(K\) must be positive."):
        BlackScholesPricer(S0=0, K=100, T=1, r=0.05, sigma=0.2)

    with pytest.raises(ValueError, match=r"Stock price \(S0\) and strike \(K\) must be positive."):
        BlackScholesPricer(S0=100, K=-10, T=1, r=0.05, sigma=0.2)

    with pytest.raises(ValueError, match=r"Time to expiration \(T\) must be positive."):
        BlackScholesPricer(S0=100, K=100, T=0, r=0.05, sigma=0.2)

    with pytest.raises(ValueError, match=r"Volatility \(sigma\) must be positive."):
        BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=-0.1)


def test_price_known_values():
    # Benchmark case: S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0.0)
    
    # Expected call price ~ 10.45058
    # Expected put price ~ 5.57353
    call_price = pricer.price("call")
    put_price = pricer.price("put")
    
    assert pytest.approx(call_price, rel=1e-4) == 10.45058
    assert pytest.approx(put_price, rel=1e-4) == 5.57353


def test_put_call_parity():
    test_params = [
        (100, 100, 1.0, 0.05, 0.20, 0.00),
        (120, 100, 0.5, 0.03, 0.25, 0.01),
        (80, 100, 2.0, 0.02, 0.15, 0.03),
    ]
    for S0, K, T, r, sigma, q in test_params:
        pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        call = pricer.price("call")
        put = pricer.price("put")
        
        # Put-Call Parity: C - P = S0 * exp(-q*T) - K * exp(-r*T)
        parity_left = call - put
        parity_right = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        assert pytest.approx(parity_left, abs=1e-10) == parity_right


def test_delta_properties():
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0.01)
    call_delta = pricer.delta("call")
    put_delta = pricer.delta("put")

    # Call delta for q=0.01 should be slightly below 1 * N(d1)
    assert 0 < call_delta < 1
    assert -1 < put_delta < 0
    # Parity: delta_call - delta_put = exp(-q * T)
    assert pytest.approx(call_delta - put_delta, abs=1e-10) == np.exp(-0.01 * 1)


def test_gamma_vega_properties():
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0.0)
    gamma = pricer.gamma()
    vega = pricer.vega()

    assert gamma > 0
    assert vega > 0
    # Benchmark values for S=100, K=100, T=1, r=0.05, sigma=0.2
    # gamma ~ 0.01876, vega ~ 37.524
    assert pytest.approx(gamma, rel=1e-3) == 0.01876
    assert pytest.approx(vega, rel=1e-3) == 37.524


def test_theta_rho_properties():
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2, q=0.0)
    call_theta = pricer.theta("call")
    put_theta = pricer.theta("put")
    call_rho = pricer.rho("call")
    put_rho = pricer.rho("put")

    # Call theta is typically negative for non-dividend ATM option
    assert call_theta < 0
    assert call_rho > 0
    assert put_rho < 0


def test_case_insensitivity():
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    assert pricer.price("CALL") == pricer.price("call")
    assert pricer.price("PuT") == pricer.price("put")
    assert pricer.delta("Call") == pricer.delta("call")
    assert pricer.theta("PUT") == pricer.theta("put")
    assert pricer.rho("CaLl") == pricer.rho("call")


def test_invalid_option_type():
    pricer = BlackScholesPricer(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    invalid_type = "straddle"

    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        pricer.price(invalid_type)

    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        pricer.delta(invalid_type)

    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        pricer.theta(invalid_type)

    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        pricer.rho(invalid_type)


def test_greeks_finite_difference():
    """Verify analytical greeks match finite difference numerical approximations."""
    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.01
    pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    h = 1e-4

    # Delta wrt S0
    p_plus = BlackScholesPricer(S0=S0 + h, K=K, T=T, r=r, sigma=sigma, q=q).price("call")
    p_minus = BlackScholesPricer(S0=S0 - h, K=K, T=T, r=r, sigma=sigma, q=q).price("call")
    fd_delta = (p_plus - p_minus) / (2 * h)
    assert pytest.approx(pricer.delta("call"), abs=1e-4) == fd_delta

    # Gamma wrt S0
    p_center = pricer.price("call")
    fd_gamma = (p_plus - 2 * p_center + p_minus) / (h ** 2)
    assert pytest.approx(pricer.gamma(), abs=1e-4) == fd_gamma

    # Vega wrt sigma
    p_sig_plus = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma + h, q=q).price("call")
    p_sig_minus = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma - h, q=q).price("call")
    fd_vega = (p_sig_plus - p_sig_minus) / (2 * h)
    assert pytest.approx(pricer.vega(), abs=1e-4) == fd_vega

    # Rho wrt r
    p_r_plus = BlackScholesPricer(S0=S0, K=K, T=T, r=r + h, sigma=sigma, q=q).price("call")
    p_r_minus = BlackScholesPricer(S0=S0, K=K, T=T, r=r - h, sigma=sigma, q=q).price("call")
    fd_rho = (p_r_plus - p_r_minus) / (2 * h)
    assert pytest.approx(pricer.rho("call"), abs=1e-4) == fd_rho
