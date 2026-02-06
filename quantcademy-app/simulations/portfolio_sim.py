"""
QuantCademy Simulation Engine
This is your differentiator - teaching through counterfactual outcomes.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Historical parameters (approximate)
ASSET_PARAMS = {
    "us_stocks": {
        "name": "US Stocks (VTI)",
        "expected_return": 0.10,  # 10% annual
        "volatility": 0.18,       # 18% std dev
        "worst_year": -0.37,      # 2008
        "best_year": 0.33,
        "color": "#4CAF50"
    },
    "intl_stocks": {
        "name": "International Stocks (VXUS)",
        "expected_return": 0.08,
        "volatility": 0.20,
        "worst_year": -0.43,
        "best_year": 0.32,
        "color": "#2196F3"
    },
    "bonds": {
        "name": "US Bonds (BND)",
        "expected_return": 0.04,
        "volatility": 0.05,
        "worst_year": -0.13,
        "best_year": 0.12,
        "color": "#FF9800"
    },
    "cash": {
        "name": "Cash/Money Market",
        "expected_return": 0.03,
        "volatility": 0.01,
        "worst_year": 0.0,
        "best_year": 0.05,
        "color": "#9E9E9E"
    }
}

# Correlation matrix (approximate)
CORRELATION_MATRIX = np.array([
    [1.0,  0.85, 0.0,  0.0],   # US Stocks
    [0.85, 1.0,  0.1,  0.0],   # Intl Stocks
    [0.0,  0.1,  1.0,  0.3],   # Bonds
    [0.0,  0.0,  0.3,  1.0]    # Cash
])


def calculate_portfolio_stats(weights: dict) -> dict:
    """Calculate expected return and volatility for a portfolio."""
    assets = ["us_stocks", "intl_stocks", "bonds", "cash"]
    w = np.array([weights.get(a, 0) / 100 for a in assets])
    
    returns = np.array([ASSET_PARAMS[a]["expected_return"] for a in assets])
    vols = np.array([ASSET_PARAMS[a]["volatility"] for a in assets])
    
    # Portfolio expected return
    port_return = np.dot(w, returns)
    
    # Portfolio volatility (using correlation matrix)
    cov_matrix = np.outer(vols, vols) * CORRELATION_MATRIX
    port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    
    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": (port_return - 0.03) / port_vol if port_vol > 0 else 0
    }


def monte_carlo_simulation(
    initial_investment: float,
    monthly_contribution: float,
    weights: dict,
    years: int,
    n_simulations: int = 1000
) -> dict:
    """
    Run Monte Carlo simulation for portfolio outcomes.
    Returns probability distributions and scenario bands.
    """
    stats = calculate_portfolio_stats(weights)
    monthly_return = stats["expected_return"] / 12
    monthly_vol = stats["volatility"] / np.sqrt(12)
    
    months = years * 12
    results = np.zeros((n_simulations, months + 1))
    results[:, 0] = initial_investment
    
    # Generate correlated random returns
    np.random.seed(42)  # For reproducibility
    
    for sim in range(n_simulations):
        portfolio_value = initial_investment
        for month in range(1, months + 1):
            # Random return with drift
            random_return = np.random.normal(monthly_return, monthly_vol)
            portfolio_value = portfolio_value * (1 + random_return) + monthly_contribution
            results[sim, month] = portfolio_value
    
    # Calculate percentiles
    percentiles = {
        "p5": np.percentile(results, 5, axis=0),
        "p10": np.percentile(results, 10, axis=0),
        "p25": np.percentile(results, 25, axis=0),
        "p50": np.percentile(results, 50, axis=0),
        "p75": np.percentile(results, 75, axis=0),
        "p90": np.percentile(results, 90, axis=0),
        "p95": np.percentile(results, 95, axis=0)
    }
    
    # Final value statistics
    final_values = results[:, -1]
    total_contributed = initial_investment + monthly_contribution * months
    
    return {
        "percentiles": percentiles,
        "final_median": np.median(final_values),
        "final_mean": np.mean(final_values),
        "final_p10": np.percentile(final_values, 10),
        "final_p90": np.percentile(final_values, 90),
        "total_contributed": total_contributed,
        "prob_loss": np.mean(final_values < total_contributed) * 100,
        "prob_double": np.mean(final_values > total_contributed * 2) * 100,
        "worst_outcome": np.min(final_values),
        "best_outcome": np.max(final_values),
        "all_simulations": results
    }


def calculate_drawdowns(portfolio_values: np.ndarray) -> dict:
    """Calculate maximum drawdown and recovery time."""
    # Running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    
    max_drawdown = np.min(drawdowns)
    max_dd_idx = np.argmin(drawdowns)
    
    # Find recovery (if any)
    recovery_idx = None
    for i in range(max_dd_idx, len(portfolio_values)):
        if portfolio_values[i] >= running_max[max_dd_idx]:
            recovery_idx = i
            break
    
    recovery_months = (recovery_idx - max_dd_idx) if recovery_idx else None
    
    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": abs(max_drawdown) * 100,
        "recovery_months": recovery_months,
        "drawdown_series": drawdowns
    }


def probability_of_loss_by_horizon(weights: dict, horizons: list = [1, 3, 5, 10, 20]) -> dict:
    """
    Calculate probability of being down at different time horizons.
    This is a KEY teaching tool - shows why time in market matters.
    """
    results = {}
    
    for years in horizons:
        sim = monte_carlo_simulation(
            initial_investment=10000,
            monthly_contribution=0,  # Lump sum to isolate time effect
            weights=weights,
            years=years,
            n_simulations=5000
        )
        results[years] = {
            "prob_loss": sim["prob_loss"],
            "expected_gain": (sim["final_median"] / 10000 - 1) * 100,
            "worst_10pct": (sim["final_p10"] / 10000 - 1) * 100,
            "best_10pct": (sim["final_p90"] / 10000 - 1) * 100
        }
    
    return results


def what_if_stop_contributing(
    initial_investment: float,
    monthly_contribution: float,
    weights: dict,
    years: int,
    stop_after_months: int
) -> dict:
    """
    Compare outcomes: continue contributing vs stop after X months.
    Teaches the cost of stopping during downturns.
    """
    # Full contribution scenario
    full = monte_carlo_simulation(
        initial_investment, monthly_contribution, weights, years
    )
    
    # Stop contributing scenario
    stopped = monte_carlo_simulation(
        initial_investment, monthly_contribution, weights, years
    )
    # Modify to stop contributions (simplified)
    stats = calculate_portfolio_stats(weights)
    monthly_return = stats["expected_return"] / 12
    monthly_vol = stats["volatility"] / np.sqrt(12)
    
    months = years * 12
    stopped_results = np.zeros((1000, months + 1))
    stopped_results[:, 0] = initial_investment
    
    np.random.seed(42)
    for sim in range(1000):
        portfolio_value = initial_investment
        for month in range(1, months + 1):
            random_return = np.random.normal(monthly_return, monthly_vol)
            contribution = monthly_contribution if month <= stop_after_months else 0
            portfolio_value = portfolio_value * (1 + random_return) + contribution
            stopped_results[sim, month] = portfolio_value
    
    stopped_final = stopped_results[:, -1]
    full_final = full["all_simulations"][:, -1]
    
    return {
        "continue_median": np.median(full_final),
        "stopped_median": np.median(stopped_final),
        "difference": np.median(full_final) - np.median(stopped_final),
        "difference_pct": (np.median(full_final) / np.median(stopped_final) - 1) * 100,
        "continue_contributed": initial_investment + monthly_contribution * months,
        "stopped_contributed": initial_investment + monthly_contribution * stop_after_months
    }


def inflation_adjusted_comparison(
    amount: float,
    years: int,
    invest: bool = True,
    weights: dict = None,
    inflation_rate: float = 0.03
) -> dict:
    """
    Compare real (inflation-adjusted) value of investing vs cash.
    Shows the 'cash risk' - losing purchasing power.
    """
    if invest and weights:
        sim = monte_carlo_simulation(amount, 0, weights, years)
        nominal_median = sim["final_median"]
    else:
        # Just cash/savings at ~3%
        nominal_median = amount * (1.03 ** years)
    
    # Inflation-adjusted
    real_value = nominal_median / ((1 + inflation_rate) ** years)
    purchasing_power_change = (real_value / amount - 1) * 100
    
    return {
        "nominal_value": nominal_median,
        "real_value": real_value,
        "purchasing_power_change": purchasing_power_change,
        "inflation_cost": amount * ((1 + inflation_rate) ** years) - amount
    }


def historical_drawdown_examples() -> list:
    """
    Real historical drawdown examples for context.
    """
    return [
        {
            "event": "2008 Financial Crisis",
            "drawdown": -37,
            "recovery_months": 49,
            "lesson": "Even catastrophic drops recovered - those who stayed invested were rewarded"
        },
        {
            "event": "COVID Crash (2020)",
            "drawdown": -34,
            "recovery_months": 5,
            "lesson": "Some crashes recover quickly - panic selling locks in losses"
        },
        {
            "event": "Dot-Com Bust (2000-2002)",
            "drawdown": -49,
            "recovery_months": 56,
            "lesson": "Tech-heavy portfolios suffered most - diversification matters"
        },
        {
            "event": "2022 Bear Market",
            "drawdown": -25,
            "recovery_months": 14,
            "lesson": "Both stocks AND bonds fell - unusual but not unprecedented"
        }
    ]
