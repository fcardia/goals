import numpy as np
import pandas as pd
from typing import Dict
 
 
import numpy as np
import pandas as pd
from typing import Dict

# def evaluate_portfolio(
#     weights: pd.Series,
#     returns_matrix: pd.DataFrame,
#     risk_free_rate: float = 0.0,
#     beta_cvar: float = 0.05,
#     beta_var: float = 0.05,
# ) -> Dict[str, float]:

#     returns_matrix = returns_matrix[weights.index]
#     returns = returns_matrix.values @ weights.values  # daily, shape (n_periods,)

#     # ── Daily building blocks ─────────────────────
#     mean_ret_daily = returns.mean()
#     std_ret_daily = returns.std(ddof=1)
#     rfr_daily = (1 + risk_free_rate) ** (1/252) - 1

#     # ── Annualized base metrics ───────────────────
#     mean_ret = mean_ret_daily * 252
#     std_ret = std_ret_daily * np.sqrt(252)
#     variance = returns.var(ddof=1) * 252

#     # ── Sharpe ───────────────────────────────────
#     sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 1e-12 else np.nan

#     # ── Sortino ──────────────────────────────────
#     downside_diff = np.minimum(returns - rfr_daily, 0)
#     downside_std = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252)
#     sortino = (mean_ret - risk_free_rate) / downside_std if downside_std > 1e-12 else np.nan

#     # ── VaR / CVaR (annualized) ───────────────────
#     var_hist = float(np.percentile(returns, beta_var * 100)) * np.sqrt(252)
#     cvar = float(returns[returns <= np.percentile(returns, beta_var * 100)].mean()) * np.sqrt(252)

#     # ── Max Drawdown ──────────────────────────────
#     cumulative = np.cumprod(1 + returns)
#     running_max = np.maximum.accumulate(cumulative)
#     drawdowns = (cumulative - running_max) / running_max
#     max_drawdown = float(drawdowns.min())

#     # ── Calmar ───────────────────────────────────
#     calmar = mean_ret / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else np.nan

#     # ── Skewness & Kurtosis (daily space) ─────────
#     n = len(returns)
#     z = (returns - mean_ret_daily) / std_ret_daily  # standardize in daily space

#     skewness = (
#         (n / ((n - 1) * (n - 2))) * np.sum(z ** 3)
#     ) if std_ret_daily > 1e-12 and n > 2 else 0.0

#     return {
#         "expected_return": round(mean_ret, 6),
#         "volatility": round(std_ret, 6),
#         "variance": round(variance, 6),
#         "sharpe_ratio": round(sharpe, 4),
#         "sortino_ratio": round(sortino, 4),
#         "calmar_ratio": round(calmar, 4),
#         f"var_{int((1-beta_var)*100)}pct": round(var_hist, 6),
#         f"cvar_{int((1-beta_cvar)*100)}pct": round(cvar, 6),
#         "max_drawdown": round(max_drawdown, 6),
#         "skewness": round(skewness, 4),
#     }

def evaluate_portfolio(
    weights: pd.Series,
    returns_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    beta_cvar: float = 0.05,
    beta_var: float = 0.05,
) -> Dict[str, float]:

    returns_matrix = returns_matrix[weights.index]
    
    # ── Use log returns to match optimization space ──
    log_returns_matrix = np.log(1 + returns_matrix)
    returns = log_returns_matrix.values @ weights.values  # daily log returns

    # ── Daily building blocks ─────────────────────
    mean_ret_daily = returns.mean()
    std_ret_daily = returns.std(ddof=1)
    rfr_daily = np.log(1 + risk_free_rate) / 252  # log risk-free daily

    # ── Annualized base metrics ───────────────────
    mean_ret = mean_ret_daily * 252                # exact for log returns
    std_ret = std_ret_daily * np.sqrt(252)         # exact for log returns
    variance = returns.var(ddof=1) * 252

    # ── Sharpe ───────────────────────────────────
    rfr_annual_log = np.log(1 + risk_free_rate)
    sharpe = (mean_ret - rfr_annual_log) / std_ret if std_ret > 1e-12 else np.nan

    # ── Sortino ──────────────────────────────────
    downside_diff = np.minimum(returns - rfr_daily, 0)
    downside_std = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252)
    sortino = (mean_ret - rfr_annual_log) / downside_std if downside_std > 1e-12 else np.nan

    # ── VaR / CVaR (annualized) ───────────────────
    # Correct: aggregate to annual log returns, then take quantile
    n_days = len(returns)
    n_years = n_days // 252
    if n_years > 1:
        annual_returns = np.array([
            returns[i*252:(i+1)*252].sum() for i in range(n_years)
        ])
    else:
        annual_returns = np.array([returns.sum()])  # fallback: single window

    var_hist = float(np.percentile(annual_returns, beta_var * 100))
    cvar = float(annual_returns[annual_returns <= np.percentile(annual_returns, beta_var * 100)].mean())

    # ── Max Drawdown ──────────────────────────────
    cumulative = np.exp(np.cumsum(returns))        # correct for log returns
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    # ── Calmar ───────────────────────────────────
    calmar = mean_ret / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else np.nan

    # ── Skewness (daily log space) ────────────────
    n = len(returns)
    z = (returns - mean_ret_daily) / std_ret_daily
    skewness = (
        (n / ((n - 1) * (n - 2))) * np.sum(z ** 3)
    ) if std_ret_daily > 1e-12 and n > 2 else 0.0

    return {
        "expected_return": round(mean_ret, 6),
        "volatility": round(std_ret, 6),
        "variance": round(variance, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        f"var_{int((1-beta_var)*100)}pct": round(var_hist, 6),
        f"cvar_{int((1-beta_cvar)*100)}pct": round(cvar, 6),
        "max_drawdown": round(max_drawdown, 6),
        "skewness": round(skewness, 4),
    }
 
 
def benchmark_models(
    model_weights: Dict[str, pd.Series],
    returns_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    beta_cvar: float = 0.05,
    beta_var: float = 0.05,
) -> pd.DataFrame:

    results = {}

    for name, w in model_weights.items():
        results[name] = evaluate_portfolio(
            w,
            returns_matrix,
            risk_free_rate=risk_free_rate,
            beta_cvar=beta_cvar,
            beta_var=beta_var,
        )

    return pd.DataFrame(results)