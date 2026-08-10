import numpy as np

class MonteCarloPricer:
    """
    Compute the price of a European option using Monte Carlo Simulation
    based on Geometric Brownian Motion (GBM).
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)

    def price(self, option_type: str = "call", num_simulations: int = 100_000, seed: int = None) -> tuple[float, float]:
        if seed is not None:
            np.random.seed(seed)

        option_type = option_type.lower()
        
        # Simulation of N paths at expiration T
        Z = np.random.standard_normal(num_simulations)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        ST = self.S0 * np.exp(drift + diffusion)

        # Payoff
        if option_type == "call":
            payoffs = np.maximum(ST - self.K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(self.K - ST, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Discount and Standard Error
        discount_factor = np.exp(-self.r * self.T)
        price_estimate = discount_factor * np.mean(payoffs)
        standard_error = discount_factor * (np.std(payoffs, ddof=1) / np.sqrt(num_simulations))

        return price_estimate, standard_error