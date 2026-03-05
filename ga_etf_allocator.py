import math
import random
from typing import Dict, List, Tuple, Any

import numpy as np


"""
Genetic-algorithm-based ETF allocator.

This module is intentionally self-contained and does not modify or depend on the
existing stock-level portfolio optimisation logic in ``portfolio_optimizer.py``.

It exposes a single public function:

    recommend_etf_allocation(answers: Dict[str, Any]) -> Dict[str, float]

Given a set of risk-profile answers, it returns a dictionary of
{ticker -> weight} where weights are decimals that sum to 1.0.
"""


# ---------------------------------------------------------------------------
# ETF universe (small, curated, ETF-only universe)
# ---------------------------------------------------------------------------

ETF_UNIVERSE: List[Dict[str, Any]] = [
    # Core US equity
    {
        "ticker": "VTI",
        "name": "Vanguard Total Stock Market",
        "asset_class": "equity",
        "region": "US",
        "category": "Core",
        "exp_return": 0.105,
        "volatility": 0.155,
        "expense_ratio": 0.0003,
    },
    {
        "ticker": "VOO",
        "name": "Vanguard S&P 500",
        "asset_class": "equity",
        "region": "US",
        "category": "Core",
        "exp_return": 0.102,
        "volatility": 0.148,
        "expense_ratio": 0.0003,
    },
    # International equity
    {
        "ticker": "VXUS",
        "name": "Vanguard Total International",
        "asset_class": "equity",
        "region": "Intl",
        "category": "Core",
        "exp_return": 0.068,
        "volatility": 0.160,
        "expense_ratio": 0.0007,
    },
    # Growth / tech tilt
    {
        "ticker": "QQQ",
        "name": "Invesco Nasdaq 100",
        "asset_class": "equity",
        "region": "US",
        "category": "Growth",
        "exp_return": 0.145,
        "volatility": 0.205,
        "expense_ratio": 0.0020,
    },
    # Bonds
    {
        "ticker": "BND",
        "name": "Vanguard Total Bond Market",
        "asset_class": "bond",
        "region": "US",
        "category": "Core",
        "exp_return": 0.045,
        "volatility": 0.055,
        "expense_ratio": 0.0003,
    },
    {
        "ticker": "AGG",
        "name": "iShares Core US Aggregate Bond",
        "asset_class": "bond",
        "region": "US",
        "category": "Core",
        "exp_return": 0.043,
        "volatility": 0.052,
        "expense_ratio": 0.0003,
    },
    {
        "ticker": "TIP",
        "name": "iShares TIPS Bond",
        "asset_class": "bond",
        "region": "US",
        "category": "Inflation",
        "exp_return": 0.040,
        "volatility": 0.065,
        "expense_ratio": 0.0019,
    },
    # Real assets
    {
        "ticker": "VNQ",
        "name": "Vanguard Real Estate",
        "asset_class": "equity",
        "region": "US",
        "category": "RealEstate",
        "exp_return": 0.090,
        "volatility": 0.180,
        "expense_ratio": 0.0012,
    },
]


TICKERS = [etf["ticker"] for etf in ETF_UNIVERSE]


def _build_covariance_matrix() -> np.ndarray:
    """Construct a simple covariance matrix using heuristic correlations."""
    vols = np.array([etf["volatility"] for etf in ETF_UNIVERSE])
    n = len(ETF_UNIVERSE)
    corr = np.full((n, n), 0.4)
    np.fill_diagonal(corr, 1.0)

    # Higher correlation within same asset class; slightly lower across.
    asset_classes = [etf["asset_class"] for etf in ETF_UNIVERSE]
    for i in range(n):
        for j in range(i + 1, n):
            if asset_classes[i] == asset_classes[j]:
                corr[i, j] = corr[j, i] = 0.7
            else:
                corr[i, j] = corr[j, i] = 0.3

    return np.outer(vols, vols) * corr


_COV = _build_covariance_matrix()
_EXP_RETURNS = np.array([etf["exp_return"] for etf in ETF_UNIVERSE])
_EXPENSES = np.array([etf["expense_ratio"] for etf in ETF_UNIVERSE])
_ASSET_CLASSES = np.array([1.0 if etf["asset_class"] == "equity" else 0.0 for etf in ETF_UNIVERSE])


# ---------------------------------------------------------------------------
# Questionnaire → risk profile mapping
# ---------------------------------------------------------------------------

def _compute_risk_score(answers: Dict[str, Any]) -> float:
    """
    Map questionnaire answers to a continuous risk score in [0, 1].

    Expected keys (all optional, with sensible defaults):
      - time_horizon_years: int
      - risk_tolerance: 1–5
      - drawdown_tolerance: 1–5
      - investment_knowledge: 1–5
      - income_stability: 1–5
      - primary_goal: string enum
    """
    time_horizon = int(answers.get("time_horizon_years", 10))
    risk_tol = int(answers.get("risk_tolerance", 3))
    drawdown_tol = int(answers.get("drawdown_tolerance", risk_tol))
    knowledge = int(answers.get("investment_knowledge", 3))
    income_stability = int(answers.get("income_stability", 3))
    goal = str(answers.get("primary_goal", "balanced"))

    # Normalise ordinal answers to [0, 1]
    def norm(x: int, lo: int = 1, hi: int = 5) -> float:
        x = max(lo, min(hi, x))
        return (x - lo) / (hi - lo)

    risk_tol_n = norm(risk_tol)
    drawdown_n = norm(drawdown_tol)
    knowledge_n = norm(knowledge)
    income_stab_n = norm(income_stability)

    # Time horizon: <5y very conservative, >20y very aggressive
    if time_horizon <= 3:
        horizon_n = 0.1
    elif time_horizon <= 5:
        horizon_n = 0.25
    elif time_horizon <= 10:
        horizon_n = 0.5
    elif time_horizon <= 20:
        horizon_n = 0.75
    else:
        horizon_n = 0.9

    goal_map = {
        "capital_preservation": 0.1,
        "income": 0.25,
        "balanced": 0.5,
        "growth": 0.75,
        "max_growth": 0.9,
    }
    goal_n = goal_map.get(goal, 0.5)

    # Weighted blend; more weight on explicit risk/drawdown + horizon.
    score = (
        0.25 * risk_tol_n
        + 0.20 * drawdown_n
        + 0.20 * horizon_n
        + 0.15 * goal_n
        + 0.10 * knowledge_n
        + 0.10 * income_stab_n
    )

    return max(0.0, min(1.0, float(score)))


def _target_equity_weight(risk_score: float) -> float:
    """
    Smooth mapping from risk score to target equity allocation.

    Very conservative (~0.0) → ~20% equity, very aggressive (~1.0) → ~95% equity.
    """
    return 0.20 + 0.75 * risk_score


# ---------------------------------------------------------------------------
# GA internals
# ---------------------------------------------------------------------------

def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project arbitrary vector onto the probability simplex."""
    # Implementation of "Efficient Projections onto the L1-Ball for Learning in High Dimensions"
    if np.all(weights == 0):
        return np.ones_like(weights) / len(weights)
    v = np.sort(weights)[::-1]
    cssv = np.cumsum(v)
    rho = np.nonzero(v * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(weights - theta, 0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _portfolio_metrics(w: np.ndarray) -> Tuple[float, float, float]:
    """Return (expected_return, volatility, expense_ratio) for a weight vector."""
    er = float(w @ _EXP_RETURNS)
    var = float(w @ _COV @ w)
    vol = math.sqrt(max(var, 0.0))
    exp_ratio = float(w @ _EXPENSES)
    return er, vol, exp_ratio


def _fitness(
    w: np.ndarray,
    risk_score: float,
    equity_target: float,
    lambda_risk: float = 2.0,
    lambda_expense: float = 10.0,
    lambda_equity: float = 8.0,
) -> float:
    """
    Higher is better.

    Objective is:
        utility = E[R] - lambda_risk * risk_aversion * Vol
                  - lambda_expense * ExpenseRatio
                  - lambda_equity * |equity_weight - target|
    """
    w = _project_to_simplex(w)
    er, vol, exp_ratio = _portfolio_metrics(w)

    equity_weight = float(w @ _ASSET_CLASSES)
    equity_penalty = abs(equity_weight - equity_target)

    # Higher risk_score → lower aversion
    risk_aversion = 1.5 - risk_score  # ~[0.5, 1.5]

    utility = (
        er
        - lambda_risk * risk_aversion * vol
        - lambda_expense * exp_ratio
        - lambda_equity * equity_penalty
    )
    return float(utility)


def _run_ga(
    risk_score: float,
    population_size: int = 80,
    generations: int = 80,
    mutation_rate: float = 0.25,
    crossover_rate: float = 0.75,
    random_seed: int = 42,
) -> np.ndarray:
    """Run a simple real-valued GA to optimise ETF weights."""
    random_state = np.random.RandomState(random_seed)
    n = len(ETF_UNIVERSE)
    equity_target = _target_equity_weight(risk_score)

    # Initial population: random points on simplex, lightly biased to core ETFs.
    base_bias = np.array(
        [1.3 if etf["category"] == "Core" else 1.0 for etf in ETF_UNIVERSE],
        dtype=float,
    )

    def random_individual() -> np.ndarray:
        raw = random_state.rand(n) * base_bias
        return _project_to_simplex(raw)

    population = [random_individual() for _ in range(population_size)]

    best_individual = None
    best_fitness = -1e9

    for _ in range(generations):
        fitnesses = [
            _fitness(ind, risk_score, equity_target) for ind in population
        ]

        # Track global best
        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_individual = ind.copy()

        # Tournament selection
        def select_one() -> np.ndarray:
            i, j = random_state.randint(0, population_size, size=2)
            return population[i] if fitnesses[i] > fitnesses[j] else population[j]

        new_population: List[np.ndarray] = []

        while len(new_population) < population_size:
            parent1 = select_one()
            parent2 = select_one()

            # Crossover (simulated binary crossover style)
            if random.random() < crossover_rate:
                alpha = random_state.uniform(0.2, 0.8)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation: gaussian noise then project
            for child in (child1, child2):
                if random.random() < mutation_rate:
                    noise = random_state.normal(0, 0.05, size=n)
                    child += noise
                    child[:] = _project_to_simplex(child)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        # Elitism: keep the best individual
        if best_individual is not None:
            new_population[0] = best_individual.copy()

        population = new_population

    # Final best
    if best_individual is None:
        best_individual = population[0]
    return _project_to_simplex(best_individual)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_etf_allocation(answers: Dict[str, Any]) -> Dict[str, float]:
    """
    Main entry point for GA-based ETF allocation.

    Parameters
    ----------
    answers : dict
        Questionnaire answers. The following keys are recognised:
          - time_horizon_years (int)
          - risk_tolerance (1–5)
          - drawdown_tolerance (1–5)
          - investment_knowledge (1–5)
          - income_stability (1–5)
          - primary_goal (str): one of
                "capital_preservation", "income",
                "balanced", "growth", "max_growth"

        Missing keys fall back to conservative defaults.

    Returns
    -------
    dict
        {ticker -> weight_percent}, rounded to two decimals and summing to ~100.
    """
    risk_score = _compute_risk_score(answers)
    weights = _run_ga(risk_score=risk_score)

    # Convert to percentage weights and round
    raw_alloc = {ticker: float(w * 100.0) for ticker, w in zip(TICKERS, weights)}

    # Prune tiny allocations (< 1%) and renormalise
    filtered = {t: w for t, w in raw_alloc.items() if w >= 1.0}
    if not filtered:
        filtered = raw_alloc

    total = sum(filtered.values())
    if total <= 0:
        # Fallback to equal weight if something went wrong
        equal = 100.0 / len(TICKERS)
        return {t: round(equal, 2) for t in TICKERS}

    normalized = {t: (w / total) * 100.0 for t, w in filtered.items()}

    # Final rounding and small re-normalisation to avoid drift from 100
    rounded = {t: round(w, 2) for t, w in normalized.items()}
    scale = 100.0 / max(1e-9, sum(rounded.values()))
    final = {t: round(w * scale, 2) for t, w in rounded.items()}
    return final


def explain_allocation(answers: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Helper to build a human-friendly explanation for the allocation.

    Returns a summary dict with risk metrics and high-level commentary
    that can be surfaced in the API or UI.
    """
    risk_score = _compute_risk_score(answers)

    # Rebuild full weight vector for metrics
    w = np.zeros(len(TICKERS))
    for i, t in enumerate(TICKERS):
        if t in weights:
            w[i] = weights[t] / 100.0
    w = _project_to_simplex(w)

    er, vol, exp_ratio = _portfolio_metrics(w)
    equity_weight = float(w @ _ASSET_CLASSES)

    profile = (
        "Conservative" if risk_score < 0.3
        else "Moderate" if risk_score < 0.5
        else "Balanced" if risk_score < 0.65
        else "Growth" if risk_score < 0.8
        else "Aggressive"
    )

    return {
        "risk_score": round(risk_score, 3),
        "inferred_profile": profile,
        "expected_return_annual_pct": round(er * 100.0, 2),
        "expected_volatility_annual_pct": round(vol * 100.0, 2),
        "expense_ratio_pct": round(exp_ratio * 100.0, 2),
        "equity_weight_pct": round(equity_weight * 100.0, 2),
    }

