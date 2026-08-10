import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    """
    Compute the price and the Greeks of a European option
    using Black-Scholes-Merton model.
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        if S0 <= 0 or K <= 0:
            raise ValueError("Stock price (S0) and strike (K) must be positive.")
        if T <= 0:
            raise ValueError("Time to expiration (T) must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive.")

        self.S0 = float(S0)         #Spot Price
        self.K = float(K)           #Strike Price
        self.T = float(T)           #Time to Expiration (in years)
        self.r = float(r)           #Risk-Free Rate
        self.sigma = float(sigma)   #Volatility
        self.q = float(q)           #Dividend Yield

    def _d1_d2(self) -> tuple[float, float]: 
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type: str = "call") -> float:
        d1, d2 = self._d1_d2()
        option_type = option_type.lower()

        if option_type == "call":
            return self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == "put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def delta(self, option_type: str = "call") -> float:
        d1, _ = self._d1_d2()
        option_type = option_type.lower()

        if option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        elif option_type == "put":
            return -np.exp(-self.q * self.T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        d1, _ = self._d1_d2()
        return self.S0 * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self, option_type: str = "call") -> float:
        d1, d2 = self._d1_d2()
        option_type = option_type.lower()
        term1 = -(self.S0 * self.sigma * np.exp(-self.q * self.T) * norm.pdf(d1)) / (2 * np.sqrt(self.T))

        if option_type == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            term3 = self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1)
            return (term1 + term2 + term3) / 365.0
        elif option_type == "put":
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            term3 = -self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
            return (term1 + term2 + term3) / 365.0
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def rho(self, option_type: str = "call") -> float:
        _, d2 = self._d1_d2()
        option_type = option_type.lower()

        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == "put":
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")