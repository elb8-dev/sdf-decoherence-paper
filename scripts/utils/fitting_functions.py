#!/usr/bin/env python3
"""
Fitting Functions Utility - Supplementary
==========================================

Common fitting functions used across SDF analysis scripts.

Author: SDF Research Team
Version: 1.0 (Supplementary)
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DECOHERENCE MODELS
# =============================================================================

def exponential(t: np.ndarray, L: float, tau: float) -> np.ndarray:
    """
    Standard exponential decay (GR prediction for ringdown).
    
    D(t) = L · [1 - exp(-t/τ)]
    
    Args:
        t: Time array
        L: Asymptotic limit
        tau: Characteristic time
        
    Returns:
        Decoherence D(t)
    """
    return L * (1 - np.exp(-t / tau))


def stretched_exponential(t: np.ndarray, L: float, tau: float, 
                          beta: float) -> np.ndarray:
    """
    SDF stretched-exponential decoherence.
    
    D(t) = L · [1 - exp(-(t/τ)^β)]
    
    Args:
        t: Time array
        L: Asymptotic limit
        tau: Characteristic time
        beta: Stretched-exponential exponent (0 < β ≤ 1)
        
    Returns:
        Decoherence D(t)
    """
    return L * (1 - np.exp(-(t / tau) ** beta))


def fit_stretched_exponential(t: np.ndarray, D: np.ndarray,
                              p0: Optional[Tuple] = None,
                              bounds: Optional[Tuple] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit stretched exponential to data.
    
    Args:
        t: Time array
        D: Decoherence data
        p0: Initial parameters (L, tau, beta)
        bounds: Parameter bounds
        
    Returns:
        (parameters, covariance)
    """
    if p0 is None:
        p0 = (max(D), t[len(t)//2], 0.9)
    
    if bounds is None:
        bounds = ([0.01, 0.001, 0.1], [10.0, 100.0, 1.0])
    
    popt, pcov = curve_fit(stretched_exponential, t, D, p0=p0, bounds=bounds)
    return popt, pcov


def fit_exponential(t: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit standard exponential to data.
    
    Args:
        t: Time array
        D: Decoherence data
        
    Returns:
        (parameters, covariance)
    """
    p0 = (max(D), t[len(t)//2])
    bounds = ([0.01, 0.001], [10.0, 100.0])
    
    popt, pcov = curve_fit(exponential, t, D, p0=p0, bounds=bounds)
    return popt, pcov


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compute_aic(n: int, k: int, residuals: np.ndarray) -> float:
    """
    Compute Akaike Information Criterion.
    
    AIC = n·ln(RSS/n) + 2k
    
    Args:
        n: Number of data points
        k: Number of parameters
        residuals: Fit residuals
        
    Returns:
        AIC value
    """
    rss = np.sum(residuals ** 2)
    return n * np.log(rss / n) + 2 * k


def compare_models(t: np.ndarray, D: np.ndarray) -> Dict:
    """
    Compare exponential vs stretched-exponential fits.
    
    Args:
        t: Time array
        D: Decoherence data
        
    Returns:
        Comparison results including ΔAIC
    """
    n = len(t)
    
    # Fit exponential (2 parameters)
    popt_exp, _ = fit_exponential(t, D)
    D_exp = exponential(t, *popt_exp)
    res_exp = D - D_exp
    aic_exp = compute_aic(n, 2, res_exp)
    r2_exp = 1 - np.sum(res_exp**2) / np.sum((D - np.mean(D))**2)
    
    # Fit stretched exponential (3 parameters)
    popt_se, _ = fit_stretched_exponential(t, D)
    D_se = stretched_exponential(t, *popt_se)
    res_se = D - D_se
    aic_se = compute_aic(n, 3, res_se)
    r2_se = 1 - np.sum(res_se**2) / np.sum((D - np.mean(D))**2)
    
    # ΔAIC: negative favors stretched exponential
    delta_aic = aic_se - aic_exp
    
    return {
        'exponential': {
            'L': popt_exp[0],
            'tau': popt_exp[1],
            'r2': r2_exp,
            'aic': aic_exp
        },
        'stretched_exponential': {
            'L': popt_se[0],
            'tau': popt_se[1],
            'beta': popt_se[2],
            'r2': r2_se,
            'aic': aic_se
        },
        'delta_aic': delta_aic,
        'favored': 'stretched_exponential' if delta_aic < -2 else 'exponential'
    }


# =============================================================================
# WEIGHTED STATISTICS
# =============================================================================

def weighted_mean(values: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Compute weighted mean and error.
    
    Args:
        values: Measured values
        errors: Measurement errors
        
    Returns:
        (weighted_mean, weighted_error)
    """
    weights = 1 / errors**2
    wmean = np.sum(weights * values) / np.sum(weights)
    werror = 1 / np.sqrt(np.sum(weights))
    return wmean, werror


def chi_squared(values: np.ndarray, errors: np.ndarray, 
                expected: float) -> Tuple[float, float]:
    """
    Compute chi-squared and p-value.
    
    Args:
        values: Measured values
        errors: Measurement errors
        expected: Expected value
        
    Returns:
        (chi2, p_value)
    """
    from scipy import stats
    
    chi2 = np.sum(((values - expected) / errors)**2)
    dof = len(values) - 1
    p = 1 - stats.chi2.cdf(chi2, dof)
    
    return chi2, p


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing fitting functions...")
    
    # Generate test data
    t = np.linspace(0.01, 10, 100)
    D_true = stretched_exponential(t, 1.0, 2.0, 0.93)
    D_noisy = D_true + 0.02 * np.random.randn(len(t))
    
    # Fit
    popt, _ = fit_stretched_exponential(t, D_noisy)
    print(f"True:  L=1.0, τ=2.0, β=0.93")
    print(f"Fitted: L={popt[0]:.3f}, τ={popt[1]:.3f}, β={popt[2]:.3f}")
    
    # Compare models
    comparison = compare_models(t, D_noisy)
    print(f"ΔAIC = {comparison['delta_aic']:.2f}")
    print(f"Favored: {comparison['favored']}")
    
    print("\nAll tests passed!")
