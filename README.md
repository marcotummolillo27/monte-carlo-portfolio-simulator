# Monte Carlo Portfolio Simulator

A Monte Carlo simulation engine for a multi-asset equity portfolio, modelling 1,000 possible price paths over a 252-day horizon using correlated Geometric Brownian Motion (GBM). Includes portfolio vs S&P 500 benchmarking and distribution analysis of final portfolio values.

---

## Overview

This project simulates the future value of a $1,000,000 portfolio across 9 equities using historically calibrated return distributions. Correlations between assets are preserved via a multivariate normal return model, making the simulation more realistic than independent per-asset simulations.

**Portfolio holdings:**

| Ticker | Company |
|---|---|
| ORCL | Oracle Corporation |
| NVDA | NVIDIA Corporation |
| VUAA | Vanguard S&P 500 UCITS ETF |
| SBUX | Starbucks Corporation |
| RY | Royal Bank of Canada |
| DAL | Delta Air Lines |
| KO | The Coca-Cola Company |
| BAC | Bank of America |
| CNQ | Canadian Natural Resources |

---

## Methodology

### 1. Data Retrieval

```python
import yfinance as yf

tickers = ['ORCL', 'NVDA', 'VUAA', 'SBUX', 'RY', 'DAL', 'KO', 'BAC', 'CNQ']
price_data = yf.download(tickers, start="2024-01-01", end="2026-04-17")['Close']
price_data = price_data.dropna()
```

Daily closing prices from January 2024 to April 2026 via `yfinance`. Tickers that fail to download are automatically detected and removed.

---

### 2. Return Modelling

```python
log_returns = np.log(price_data / price_data.shift(1)).dropna()

mu  = log_returns.mean().values   # daily mean return vector
cov = log_returns.cov().values    # daily covariance matrix (captures correlations)
```

Log returns are used to estimate the daily mean return vector **μ** and the full covariance matrix **Σ**, which encodes both individual asset volatilities and cross-asset correlations.

---

### 3. Monte Carlo Simulation (Correlated GBM)

```python
for sim in range(num_simulations):
    for day in range(1, num_simulation_days):
        simulated_log_return = np.random.multivariate_normal(mu, cov)
        simulated_prices[day] = simulated_prices[day-1] * np.exp(simulated_log_return)
```

**Parameters:**

| Parameter | Value |
|---|---|
| Simulations | 1,000 |
| Horizon | 252 trading days (~1 year) |
| Starting portfolio value | $1,000,000 |
| Weighting | Equal weight across all assets |
| Return model | Multivariate normal (correlated GBM) |

Each simulation draws correlated daily log returns from `np.random.multivariate_normal(μ, Σ)`, ensuring realistic co-movement between assets.

---

### 4. Portfolio vs S&P 500 Benchmark

```python
sp500 = yf.download("^GSPC", start="2020-01-01", end="2026-01-01")['Close']
portfolio_index = (normalized_stocks * weights).sum(axis=1)
```

The portfolio's historical performance is compared against the S&P 500, both normalised to a $1,000,000 starting value, and plotted interactively via Plotly.

---

### 5. Outputs

- **Distribution histogram** of 1,000 final portfolio values with mean, 5th and 95th percentile markers
- **Interactive Plotly chart** — portfolio value vs S&P 500 over the historical period
- **Console output** — mean daily log returns and daily volatilities per asset

---

## Tech Stack

| Library | Role |
|---|---|
| `yfinance` | Market data retrieval |
| `pandas` | Data cleaning and manipulation |
| `numpy` | Log returns, covariance matrix, GBM simulation |
| `matplotlib` | Final value distribution histogram |
| `plotly` | Interactive portfolio vs benchmark chart |

---

## Installation

```bash
git clone https://github.com/marcotummolillo27/monte-carlo-portfolio-simulator.git
cd monte-carlo-portfolio-simulator
pip install yfinance pandas numpy matplotlib plotly
jupyter notebook simulation.ipynb
```

---

## Project Structure

```
├── simulation.ipynb    # Full simulation: data, GBM model, plots, benchmark
└── README.md
```

---

## About

Built as part of an independent exploration of quantitative finance and risk modelling.  
**Marco Tummolillo** — Actuarial Science, University of Amsterdam  
[GitHub](https://github.com/marcotummolillo27)

> *This project is for educational purposes only and does not constitute financial advice.*
