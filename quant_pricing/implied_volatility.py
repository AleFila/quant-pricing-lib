import numpy as np
from quant_pricing.black_scholes import BlackScholesPricer

def implied_volatility(
    market_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """
    Compute the Implied Volatility using the Newton-Raphson algorithm.
    Use the option's Vega as the first derivative.
    """
    if market_price <= 0:
        return np.nan

    # Initial volatility estimate (e.g., 20%)
    sigma = 0.20

    for _ in range(max_iter):
        pricer = BlackScholesPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        bs_price = pricer.price(option_type=option_type)
        vega = pricer.vega()

        diff = bs_price - market_price

        # If the difference is lower than the tolerance, we found the IV
        if abs(diff) < tol:
            return sigma

        # Avoid division by zero if Vega is close to 0
        if abs(vega) < 1e-8:
            break

        # Newton-Raphson update: x_new = x_old - f(x) / f'(x)
        sigma = sigma - diff / vega

        # Volatility cannot be negative or zero
        if sigma <= 0:
            sigma = 1e-4

    return np.nan
