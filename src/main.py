import argparse
import yaml

from data_loader import load_data, build_scenario_matrix, compute_returns
from data_analysis import analyze_returns, covariance_matrix
from optimization import mad_optimization, semi_mad_optimization, gini_optimization, minmax_optimization, var_optimization, var_rev_optimization, cvar_optimization, approx_markowitz_optimization, mean_variance_optimization
from evaluation import benchmark_models
import matplotlib.pyplot as plt 
import numpy as np

from time import time

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config", 
    type=str, 
    default="config.yaml", 
    help="Path to the configuration file"
    )


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data = load_data(config)
    returns = compute_returns(data)
    scenarios = build_scenario_matrix(data, n_years=10, n_scenarios=100, annualized=True)

    cov_matrix = covariance_matrix(returns, annualized=True)

    start = time()
    approx_weights = approx_markowitz_optimization(scenarios, cov_matrix, desired_return=0.07, n_intervals=10)
    print(f"Approx M-V time: {time()-start}\nApprox M-V variance: {approx_weights.values.T @ cov_matrix @ approx_weights.values}", "\n\n\n")

    start = time()
    correct_weights = mean_variance_optimization(scenarios.mean(axis=1), cov_matrix, desired_return=0.07)
    print(f"Correct M-V time: {time()-start}\nCorrect M-V variance: {correct_weights.values.T @ cov_matrix @ correct_weights.values}", "\n\n\n")
    
    start = time()
    mad_opt_weights = mad_optimization(scenarios, desired_return=0.07)
    print(f"MAD time: {time()-start}\nMAD variance: {mad_opt_weights.values.T @ cov_matrix @ mad_opt_weights.values}", "\n\n\n")

    start = time()
    semi_mad_opt_weights = semi_mad_optimization(scenarios, desired_return=0.07)
    print(f"semi-MAD time: {time()-start}\nsemi-MAD variance: {semi_mad_opt_weights.values.T @ cov_matrix @ semi_mad_opt_weights.values}", "\n\n\n")

    start = time()
    gini_opt_weights = gini_optimization(scenarios, desired_return=0.07)
    print(f"gini time: {time()-start}\ngini variance: {gini_opt_weights.values.T @ cov_matrix @ gini_opt_weights.values}", "\n\n\n")

    start = time()
    minmax_opt_weights = minmax_optimization(scenarios, desired_return=0.07)
    print(f"minmax time: {time()-start}\nminmax variance: {minmax_opt_weights.values.T @ cov_matrix @ minmax_opt_weights.values}", "\n\n\n")

    start = time()
    var_opt_weights = var_optimization(scenarios, desired_return=0.07, beta=0.01)
    print(f"VaR time: {time()-start}\nVaR variance: {var_opt_weights.values.T @ cov_matrix @ var_opt_weights.values}", "\n\n\n")

    start = time()
    cvar_opt_weights = cvar_optimization(scenarios, desired_return=0.11, beta=0.01)
    print(f"CVaR time: {time()-start}\nCVaR variance: {cvar_opt_weights.values.T @ cov_matrix @ cvar_opt_weights.values}", "\n\n\n")

    # start = time()
    # var_rev_opt_weights = var_rev_optimization(annual_scenarios, min_var_return=0.01, beta=0.01)
    # print(var_rev_opt_weights)

    models = {
        "Approx M-V": approx_weights,
        "Correct M-V": correct_weights,
        "MAD": mad_opt_weights,
        "semi-MAD": semi_mad_opt_weights,
        "Gini": gini_opt_weights,
        "MinMax": minmax_opt_weights,
        "VaR": var_opt_weights,
        "CVaR": cvar_opt_weights
    }

    benchmarks = benchmark_models(models, scenarios)
    print(benchmarks)
    
if __name__ == "__main__":

    args = parser.parse_args()
    main(
        config_path=args.config
    )