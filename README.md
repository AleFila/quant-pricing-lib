# 📈 Quantitative Pricing & Volatility Engine
### Financial Engineering & Derivatives Pricing Library in Python

A Python framework for European option pricing, risk sensitivity analysis (Greeks), stochastic volatility modeling, and empirical market calibration against real-world data.

## 📋 Executive Summary

This repository provides a modular, end-to-end quantitative engine built to address the classical limitations of analytical derivatives pricing. While the **Black-Scholes-Merton** framework offers an elegant closed-form baseline under constant volatility assumptions, real-world financial markets display fat tails, jump risks, and asymmetric volatility structures.

To bridge the gap between theoretical modeling and market realities, this project implements:

1. **Analytical & Numerical Benchmarking:** Closed-form Black-Scholes pricing and full analytical Greeks derivation ($\Delta, \Gamma, \Theta, \text{Vega}, \rho$), validated against a **Geometric Brownian Motion (GBM) Monte Carlo** simulation engine equipped with standard error tracking and 95% confidence bounds.
2. **Stochastic Volatility Modeling:** Implementation of the **Heston model** using bivariate stochastic differential equations (SDEs) linked via a mean-reverting Cox-Ingersoll-Ross (CIR) variance process. By tuning the price-variance correlation parameter ($\rho$), the engine successfully reproduces the synthetic **Volatility Skew/Smile**.
3. **Implied Volatility & Real Market Calibration:** A hybrid root-finding solver (Newton-Raphson with Bisection fallback) designed to back out Implied Volatility (IV) from live option chains. Using real-time market data from **Apple Inc. (AAPL)** via `yfinance`, the engine reconstructs the empirical Volatility Skew, proving why stochastic volatility models are indispensable for institutional risk management.

## 🛠️ Project Structure

```text
quant-pricing-lib/
│
├── quant_pricing/              # Core Quantitative Library
│   ├── __init__.py
│   ├── black_scholes.py        # Analytical pricing and Greeks
│   ├── monte_carlo.py          # GBM Monte Carlo simulator & SEM
│   ├── heston.py               # Heston stochastic volatility engine
│   └── implied_volatility.py   # Newton-Raphson & Bisection IV solver
│
├── notebooks/                  # Visual Analysis & Reports
│   └── demo_pricing.ipynb      # Convergence plots, 3D surfaces & market data
│
├── tests/                      # Automated Unit Test Suite (PyTest)
│   ├── test_black_scholes.py
│   ├── test_monte_carlo.py
│   ├── test_heston.py
│   └── test_implied_volatility.py
│
├── docs/                       # Rendered Charts & Assets for Documentation
│   ├── mc_convergence.png
│   ├── delta_surface.png
│   ├── heston_skew.png
│   └── apple_real_skew.png
│
├── requirements.txt            # Environment dependencies
├── .gitignore                
└── README.md                   # Project documentation
```

## 🧮 Mathematical Foundations

### 1. Black-Scholes-Merton Baseline

The Black-Scholes model assumes that the underlying asset price $S_t$ follows a Geometric Brownian Motion (GBM) under the risk-neutral measure $\mathbb{Q}$:

$$
d S_t = r S_t d t + \sigma S_t d W_t
$$

Where $r$ is the risk-free interest rate, $\sigma$ is the constant volatility, and $W_t$ is a standard Wiener process.

The analytical closed-form price for a European Call option $C(S_0, t)$ with strike price $K$ and time-to-expiration $T$ is given by:

$$
C(S_0, t) = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

Where $N(\cdot)$ denotes the Cumulative Distribution Function (CDF) of a standard Normal distribution $\mathcal{N}(0, 1)$, and $d_1, d_2$ are defined as:

$$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
$$

#### Analytical Greeks
Option sensitivities (Greeks) measure the partial derivatives of the option price with respect to market parameters:

* **Delta ($\Delta$):** Sensitivity to the underlying asset price:

$$\Delta_{\text{Call}} = \frac{\partial C}{\partial S_0} = N(d_1)$$

* **Gamma ($\Gamma$):** Second-order sensitivity to the underlying asset price:

$$\Gamma = \frac{\partial^2 C}{\partial S_0^2} = \frac{N'(d_1)}{S_0 \sigma \sqrt{T}}$$

* **Vega ($\nu$):** Sensitivity to volatility:

$$\text{Vega} = \frac{\partial C}{\partial \sigma} = S_0 \sqrt{T} N'(d_1)$$
---

### 2. Monte Carlo Framework & Standard Error

Under the risk-neutral measure, the fair price of an option is the expected discounted payoff at maturity:

$$
C(S_0) = e^{-rT} \mathbb{E}^{\mathbb{Q}}\left[ \max(S_T - K, 0) \right]
$$

The Monte Carlo estimator simulates $N$ independent asset price paths at maturity $T$:

$$
S_T^{(i)} = S_0 \exp\left( \left(r - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} Z^{(i)} \right), \quad Z^{(i)} \sim \mathcal{N}(0, 1)
$$

The estimated option price is:

$$
\hat{C}_{\text{MC}} = e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \max\left(S_T^{(i)} - K, 0\right)
$$

By the Central Limit Theorem, the statistical uncertainty of the estimate is bounded by the **Standard Error of the Mean (SEM)**:

$$
\text{SE} = e^{-rT} \cdot \frac{s_{\text{payoff}}}{\sqrt{N}}
$$

Where $s_{\text{payoff}}$ is the sample standard deviation of simulated payoffs (using Bessel's correction $N-1$). The 95% confidence interval is given by $\hat{C}_{\text{MC}} \pm 1.96 \cdot \text{SE}$.

---

### 3. Heston Stochastic Volatility Model

To resolve the constant-volatility limitation of Black-Scholes, the Heston (1993) model introduces a secondary stochastic process for variance $v_t$:

$$
\begin{aligned}
d S_t &= r S_t d t + \sqrt{v_t} S_t d W_t^S \\
d v_t &= \kappa (\theta - v_t) d t + \xi \sqrt{v_t} d W_t^v
\end{aligned}
$$

Where:

* $v_t$: Instantaneous variance at time $t$ ($v_0$ is the initial variance).
* $\kappa > 0$: Rate of mean reversion.
* $\theta > 0$: Long-term mean variance target.
* $\xi > 0$: Volatility of variance (*vol-of-vol*).
* $d W_t^S, d W_t^v$: Standard Wiener processes with correlation $\mathbb{E}[d W_t^S d W_t^v] = \rho d t$.

#### Feller Condition

To guarantee that the variance process $v_t$ remains strictly positive ($v_t > 0$), the parameters should satisfy the **Feller condition**:

$$
2 \kappa \theta > \xi^2
$$

#### The Correlation Parameter ($\rho$)

* **$\rho < 0$ (Leverage Effect):** When the asset price falls ($d W_t^S < 0$), variance tends to increase ($d W_t^v > 0$). This generates asymmetric fat tails on the left side of the distribution, reproducing the empirical **Volatility Skew** observed in equity markets.
* **$\rho = 0$:** Price and variance shocks are uncorrelated, generating a symmetric **Volatility Smile**.

---

### 4. Implied Volatility & Root-Finding (Newton-Raphson)

Implied Volatility $\sigma_{\text{implied}}$ is the unique volatility value that equates the theoretical Black-Scholes price $C_{\text{BS}}(\sigma)$ to the observed market price $C_{\text{Market}}$:

$$
f(\sigma) = C_{\text{BS}}(\sigma) - C_{\text{Market}} = 0
$$

Using the **Newton-Raphson method**, the root is found iteratively using Vega ($\nu = \frac{\partial C_{\text{BS}}}{\partial \sigma}$) as the derivative:

$$
\sigma_{n+1} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{Market}}}{\text{Vega}(\sigma_n)}
$$

If Vega approaches zero ($\nu \to 0$) or the iteration fails to converge within tolerance $\epsilon = 10^{-7}$, the engine automatically falls back to a robust **Bisection Method** search algorithm within the bracket $[\sigma_{\text{min}}, \sigma_{\text{max}}] = [0.001, 5.0]$.

## 📊 Model Validation & Empirical Results

The models and algorithms implemented in `quant_pricing` were validated through numerical experiments, convergence tests, and empirical market calibration against live Wall Street option chains.

---

### 1. Monte Carlo Convergence vs. Black-Scholes Analytical Baseline
To verify the numerical precision and statistical consistency of the Geometric Brownian Motion (GBM) simulation engine, we evaluate a European Call option with parameters $S_0 = 100$, $K = 100$, $T = 1.0$, $r = 0.05$, and $\sigma = 0.20$.

The analytical Black-Scholes baseline yields an exact price of **$10.4506$**.

![Monte Carlo Convergence][def]

#### Statistical Analysis:
* **Asymptotic Behavior:** As the number of simulated paths increases from $N = 1,000$ to $N = 500,000$, the Monte Carlo estimate $\hat{C}_{\text{MC}}$ converges smoothly toward the analytical benchmark.
* **Variance Reduction:** The 95% confidence interval ($\pm 1.96 \cdot \text{SE}$) shrinks strictly according to the theoretical rate of $O(1/\sqrt{N})$. At $N = 500,000$, the Standard Error drops to less than **$0.01\$$**, confirming the statistical stability of the estimator.

---

### 2. Delta Sensitivity Surface ($\Delta$)
Option Greeks serve as the primary tool for dynamic hedging (Delta-Neutral strategies). We construct a 3D surface mapping the European Call Delta ($\Delta = N(d_1)$) across spot prices $S_0 \in [50, 150]$ and times-to-expiration $T \in [0.05, 1.5]$ years (with Strike $K = 100$).

![Delta Surface 3D][def2]

#### Quantitative Insights:
* **Deep Out-of-the-Money ($S_0 < 80$):** Delta approaches $0.0$, indicating negligible price sensitivity to underlying asset movements.
* **At-the-Money ($S_0 \approx 100$):** Delta centers around $0.50$, representing an approximate 50% risk-neutral probability of expiring in-the-money.
* **Deep In-the-Money ($S_0 > 120$):** Delta saturates at $1.0$, meaning the option tracks the underlying asset on a 1:1 basis.
* **Pin Risk / Gamma Peak ($T \to 0$):** As time-to-expiration approaches zero ($T \to 0.05$), the sigmoidal S-curve transitions into a steep step-function around $K = 100$. This vertical gradient illustrates the extreme instability of Delta (peak Gamma $\Gamma$) near expiration.

---

### 3. Synthetic Volatility Skew (Heston Correlation Effect)
Black-Scholes assumes constant volatility ($\sigma$), predicting a flat implied volatility profile across strikes. To test the Heston stochastic volatility engine, we simulate option prices across strikes $K \in [80, 120]$ under two scenarios for the correlation parameter $\rho$:

1. **Uncorrelated Case ($\rho = 0.0$):** Independent price and variance shocks.
2. **Correlated Case ($\rho = -0.70$):** Strong negative correlation (Leverage Effect).

Using our Newton-Raphson solver, we extract the Implied Volatility (IV) from the resulting Heston prices.

![Heston Volatility Skew][def3]

#### Model Insights:
* **Symmetric Smile ($\rho = 0.0$):** Produces a nearly flat, symmetric profile around $19.5\%$ IV.
* **Asymmetric Skew ($\rho = -0.70$):** Generates a pronounced downward slope (**Volatility Skew**). Out-of-the-Money Put / In-the-Money Call options ($K = 80$) exhibit an IV of **$\sim 23\%$**, while Out-of-the-Money Calls ($K = 120$) drop to **$\sim 17\%$**. 
* **Conclusion:** Tuning $\rho < 0$ allows the Heston model to successfully capture asymmetric fat-tailed risk and market crash aversion.

---

### 4. Empirical Market Calibration: Apple Inc. (AAPL)
To validate the framework against real-world market dynamics, live option chains for **Apple Inc. (AAPL)** were fetched via `yfinance` for options expiring in September 2026 (Spot Price $S_0 = \$302.72$).

Mid-market prices $\frac{\text{Bid} + \text{Ask}}{2}$ across liquid strikes ($K \in [275, 345]$) were passed to our `implied_volatility` solver under a risk-free rate of $r = 4.5\%$.

![Apple Real Volatility Skew][def4]

#### Market Evidence:
* **Empirical Skew Confirmation:** The real-world implied volatility curve for AAPL exhibits a classic asymmetric "Smirk" shape, with IV peaking at **$\sim 27.8\%$** for lower strikes ($K = 275$) and bottoming around **$\sim 23.5\%$** At-The-Money ($K \approx 305$).
* **Structural Failure of Black-Scholes:** The non-flat nature of the empirical green curve proves that the market actively prices in downside tail risk (protective Put demand).
* **Engineering Validation:** The successful extraction of the empirical skew demonstrates that our hybrid Newton-Raphson / Bisection solver operates robustly on noisy, real-world exchange data.

[def]: docs/mc_convergence.png
[def2]: docs/delta_surface.png
[def3]: docs/heston_skew.png
[def4]: docs/apple_real_skew.png

---

## 🚀 Quickstart & Setup

### 1. Clone the repository
```bash
git clone https://github.com/AleFila/quant-pricing-lib.git
cd quant-pricing-lib
```
### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Automated Unit Tests
```bash
pytest tests/
```
### 5. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/demo_pricing.ipynb
```

---

## 🧪 Testing & Code Quality

The core library is fully covered by automated unit tests built with `pytest` (located in the `tests/` directory). The test suite ensures mathematical correctness, boundary stability, and numerical tolerance across all pricing engines:

* **Black-Scholes & Greeks (`test_black_scholes.py`):** Validates Call-Put parity relations, extreme boundary conditions ($T \to 0$, $S_0 \to \infty$), and verifies analytical Greeks against finite-difference numerical approximations.
* **Monte Carlo Engine (`test_monte_carlo.py`):** Ensures simulated estimates fall within the theoretical $1.96 \cdot \text{SE}$ confidence bounds relative to analytical baselines and verifies seed reproducibility.
* **Heston Model (`test_heston.py`):** Verifies correct correlation matrix decomposition ($\rho$), volatility trajectory generation, and stability near the Feller condition threshold.
* **Implied Volatility Solver (`test_implied_volatility.py`):** Tests Newton-Raphson convergence speed, tolerance precision limits ($\epsilon = 10^{-7}$), and automatic fallback safety to the Bisection method under near-zero Vega scenarios.

---

## 📖 References & Bibliography

1. **Black, F., & Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637-654.
2. **Heston, S. L. (1993).** *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond, Currency, and Equity Options.* The Review of Financial Studies, 6(2), 327-343.
3. **Glasserman, P. (2004).** *Monte Carlo Methods in Financial Engineering.* Springer Science & Business Media.

---

## 👨‍💻 Author

* **Alex Filaferro** — [LinkedIn](https://www.linkedin.com/in/alex-filaferro) | [GitHub](https://github.com/AleFila)
