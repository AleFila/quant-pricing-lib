import numpy as np

class HestonMonteCarloPricer:
    """
    Monte Carlo Pricer for European options under the Heston Model (Stochastic Volatility).
    Uses Euler-Maruyama discretization to simulate S_t and v_t.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        q: float = 0.0,
    ):
        if S0 <= 0 or K <= 0 or T <= 0:
            raise ValueError("S0, K and T must be positive.")
        if v0 <= 0 or theta <= 0 or kappa <= 0 or xi <= 0:
            raise ValueError("The variance parameters (v0, theta, kappa, xi) must be positive.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("The correlation rho must be between -1.0 and 1.0.")

        # Heston Model Parameters
        self.S0 = float(S0)      # Initial asset price (spot price)
        self.K = float(K)        # Strike price of the option
        self.T = float(T)        # Time to expiration in years
        self.r = float(r)        # Risk-free interest rate (annualized)
        self.v0 = float(v0)      # Initial variance (v_0 = sigma_0^2)
        self.kappa = float(kappa)# Rate of mean reversion for variance
        self.theta = float(theta)# Long-term mean variance level
        self.xi = float(xi)      # Volatility of volatility (vol-of-vol parameter)
        self.rho = float(rho)    # Correlation between price and variance shocks (-1 <= rho <= 1)
        self.q = float(q)        # Continuous dividend yield (annualized)

    def price(
        self,
        option_type: str = "call",
        num_simulations: int = 50_000,
        num_steps: int = 100,
        seed: int = None,
    ) -> tuple[float, float]:
        """
        Performs multi-step Monte Carlo simulation.
        
        Returns:
            tuple[float, float]: (Estimated Price, Standard Error)
        """
        if seed is not None:
            np.random.seed(seed)

        option_type = option_type.lower()
        dt = self.T / num_steps

        # Initialize vectors for N simulations
        S = np.full(num_simulations, self.S0, dtype=float)
        v = np.full(num_simulations, self.v0, dtype=float)

        # Correlation matrix for generating correlated normals
        cov_matrix = np.array([[1.0, self.rho], [self.rho, 1.0]])
        
        # Step-by-step simulation (Time-Stepping)
        for _ in range(num_steps):
            # Generate correlated shocks Z1 and Z2 ~ N(0, 1)
            Z = np.random.multivariate_normal([0.0, 0.0], cov_matrix, size=num_simulations)
            Z1 = Z[:, 0]
            Z2 = Z[:, 1]

            # Current variance truncated to 0 for numerical stability
            v_curr = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_curr)

            # 1. Update Price S_t (Log-Euler to avoid S <= 0)
            drift_S = (self.r - self.q - 0.5 * v_curr) * dt
            diffusion_S = sqrt_v * np.sqrt(dt) * Z1
            S = S * np.exp(drift_S + diffusion_S)

            # 2. Update Variance v_t (CIR Process with Euler)
            drift_v = self.kappa * (self.theta - v_curr) * dt
            diffusion_v = self.xi * sqrt_v * np.sqrt(dt) * Z2
            v = v_curr + drift_v + diffusion_v

        # Calculate Payoffs at maturity T
        if option_type == "call":
            payoffs = np.maximum(S - self.K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(self.K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Financial discount and price estimation
        discount_factor = np.exp(-self.r * self.T)
        price_estimate = discount_factor * np.mean(payoffs)
        standard_error = discount_factor * (np.std(payoffs, ddof=1) / np.sqrt(num_simulations))

        return price_estimate, standard_error