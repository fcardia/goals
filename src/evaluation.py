import numpy as np
import pandas as pd
from typing import Dict
 
 
def evaluate_portfolio(
    weights: pd.Series,
    scenarios_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    beta_cvar: float = 0.05,
    beta_var: float = 0.05,
) -> Dict[str, float]:
    """
    Evaluate a portfolio given its weights and a scenario matrix.
 
    Parameters
    ----------
    weights          : pd.Series — asset weights (must sum to 1)
    scenarios_matrix : pd.DataFrame — shape (n_assets, n_scenarios)
    risk_free_rate   : annualised risk-free rate (same units as returns)
    beta_cvar        : tail probability for CVaR  (e.g. 0.05 → 95% CVaR)
    beta_var         : tail probability for VaR   (e.g. 0.05 → 95% VaR)
 
    Returns
    -------
    dict with all metrics
    """
    w = weights.values
    returns = scenarios_matrix.values.T @ w          # shape (n_scenarios,)
 
    mean_ret   = returns.mean()
    std_ret    = returns.std(ddof=1)
    variance   = returns.var(ddof=1)
 
    # ── Sharpe Ratio ──────────────────────────────────────────────────────────
    sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 1e-12 else np.nan
 
    # ── Sortino Ratio ─────────────────────────────────────────────────────────
    downside_returns = returns[returns < risk_free_rate] - risk_free_rate
    downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) else 1e-12
    sortino = (mean_ret - risk_free_rate) / downside_std
 
    # ── MAD ───────────────────────────────────────────────────────────────────
    mad = np.mean(np.abs(returns - mean_ret))
 
    # ── Semi-MAD (downside only) ───────────────────────────────────────────────
    below_mean = returns[returns < mean_ret]
    semi_mad = np.mean(np.abs(below_mean - mean_ret)) if len(below_mean) else 0.0
 
    # ── VaR (historical) ──────────────────────────────────────────────────────
    var_hist = float(np.percentile(returns, beta_var * 100))
 
    # ── CVaR / Expected Shortfall ─────────────────────────────────────────────
    cvar = float(returns[returns <= var_hist].mean()) if np.any(returns <= var_hist) else var_hist
 
    # ── Max Drawdown (on cumulative wealth path, scenario-ordered) ────────────
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(drawdowns.min())
 
    # ── Calmar Ratio ──────────────────────────────────────────────────────────
    calmar = mean_ret / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else np.nan
 
    # ── Skewness & Kurtosis ───────────────────────────────────────────────────
    n = len(returns)
    skewness = float(
        (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean_ret) / std_ret) ** 3)
    ) if std_ret > 1e-12 else 0.0
    excess_kurtosis = float(
        ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)))
        * np.sum(((returns - mean_ret) / std_ret) ** 4)
        - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    ) if std_ret > 1e-12 else 0.0
 
    return {
        "expected_return":   round(mean_ret,        6),
        "volatility":        round(std_ret,          6),
        "variance":          round(variance,         6),
        "sharpe_ratio":      round(sharpe,           4),
        "sortino_ratio":     round(sortino,          4),
        "calmar_ratio":      round(calmar,           4),
        "mad":               round(mad,              6),
        "semi_mad":          round(semi_mad,         6),
        f"var_{int((1-beta_var)*100)}pct":   round(var_hist, 6),
        f"cvar_{int((1-beta_cvar)*100)}pct": round(cvar,     6),
        "max_drawdown":      round(max_drawdown,     6),
        "skewness":          round(skewness,         4),
        "excess_kurtosis":   round(excess_kurtosis,  4),
    }
 
 
def benchmark_models(
    model_weights: Dict[str, pd.Series],
    scenarios_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    beta_cvar: float = 0.05,
    beta_var: float = 0.05,
) -> pd.DataFrame:
    """
    Compare multiple portfolio models side-by-side.
 
    Parameters
    ----------
    model_weights    : dict  {model_name: weights_series}
    scenarios_matrix : pd.DataFrame — shape (n_assets, n_scenarios)
 
    Returns
    -------
    pd.DataFrame — metrics as rows, models as columns
    """
    results = {}
    for name, w in model_weights.items():
        results[name] = evaluate_portfolio(
            w, scenarios_matrix,
            risk_free_rate=risk_free_rate,
            beta_cvar=beta_cvar,
            beta_var=beta_var,
        )
    return pd.DataFrame(results)