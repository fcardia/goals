import pandas as pd
import yfinance as yf
import yaml
import numpy as np

np.random.seed(42)

def load_data(config: dict, start="2000-01-01", end="2026-01-01") -> pd.DataFrame:
    return yf.download(config.get("INDEXES"), start=start, end=end)["Close"]

# def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
#     return data.pct_change().dropna()

def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
    return np.log(data / data.shift(1)).dropna()

def compute_annualized_returns(data: pd.DataFrame, periods_per_year: int=252) -> pd.DataFrame:
    return data.pct_change().dropna() * 252

def build_scenario_matrix(data: pd.DataFrame, n_years: int=10, n_scenarios: int=5000, annualized: bool=True) -> pd.DataFrame:
    daily_returns = compute_returns(data)  # now log returns

    simulated_returns = pd.DataFrame(index=data.columns, columns=range(n_scenarios))

    for ticker in data.columns:
        returns_series = daily_returns[ticker].values
        for scenario in range(n_scenarios):
            sampled_returns = np.random.choice(returns_series, size=252*n_years, replace=True)
            cumulative_log_return = np.sum(sampled_returns)          # sum, not product
            simulated_returns.loc[ticker, scenario] = cumulative_log_return

    simulated_returns = simulated_returns.astype(float)
    if annualized: simulated_returns = simulated_returns / n_years   # divide, not geometric root

    return simulated_returns

# def build_scenario_matrix(data: pd.DataFrame, n_years: int=10, n_scenarios: int=5000, annualized: bool=True) -> pd.DataFrame:
#     daily_returns = compute_returns(data)

#     # DataFrame finale dei rendimenti cumulati simulati
#     simulated_returns = pd.DataFrame(index=data.columns, columns=range(n_scenarios))

#     # Simula n_scenarios scenari a n_years anni
#     for ticker in data.columns: # Per ogni titolo (n)
#         returns_series = daily_returns[ticker].values
#         for scenario in range(n_scenarios): # Per ogni scenario (T)
#             sampled_returns = np.random.choice(returns_series, size=252*n_years, replace=True)
#             cumulative_return = np.prod(1 + sampled_returns) - 1
#             simulated_returns.loc[ticker, scenario] = cumulative_return

#     # convertiamo tutto in float
#     simulated_returns = simulated_returns.astype(float)
#     if annualized: simulated_returns = (1 + simulated_returns)**(1/n_years) - 1

#     return simulated_returns

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        