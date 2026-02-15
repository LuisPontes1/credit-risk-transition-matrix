"""
Stress Testing — Apply shocks and generate scenarios on transition matrices.

Implements matrix transformations for IFRS 9 / CPC 4966 stress testing:
- Scalar stress: multiply migration probabilities by a factor
- Z-shift: shift probability mass across buckets
- Scenario generation: base, optimistic, pessimistic
- Weighted scenario: blend multiple matrices by probability weights
"""

from typing import Dict, List, Optional, Any
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_scalar_stress(
    transition_matrix: pd.DataFrame,
    stress_factor: float = 1.5,
    direction: str = 'worsen'
) -> pd.DataFrame:
    """
    Apply scalar stress to a transition matrix.
    
    Multiplies off-diagonal migration probabilities by a stress factor,
    then renormalizes each row to sum to 1.0.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix (rows = from, cols = to).
    stress_factor : float, default=1.5
        Multiplicative factor for migration probabilities.
        > 1.0 increases migration, < 1.0 decreases it.
    direction : str, default='worsen'
        'worsen': scale probabilities of migrating to worse (higher index) buckets.
        'improve': scale probabilities of migrating to better (lower index) buckets.
        'both': scale all off-diagonal probabilities.
        
    Returns
    -------
    pd.DataFrame
        Stressed transition matrix with rows summing to 1.0.
        
    Raises
    ------
    ValueError
        If stress_factor is negative or direction is invalid.
    """
    if stress_factor < 0:
        raise ValueError("stress_factor must be non-negative")
    if direction not in ('worsen', 'improve', 'both'):
        raise ValueError(f"direction must be 'worsen', 'improve', or 'both', got '{direction}'")
    
    buckets = list(transition_matrix.index)
    n = len(buckets)
    stressed = transition_matrix.values.astype(float).copy()
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip diagonal
            
            if direction == 'worsen' and j > i:
                stressed[i, j] *= stress_factor
            elif direction == 'improve' and j < i:
                stressed[i, j] *= stress_factor
            elif direction == 'both':
                stressed[i, j] *= stress_factor
        
        # Renormalize row
        row_sum = stressed[i].sum()
        if row_sum > 0:
            stressed[i] /= row_sum
    
    result = pd.DataFrame(stressed, index=buckets, columns=buckets)
    
    logger.info(
        "Applied scalar stress: factor=%.2f, direction=%s",
        stress_factor, direction
    )
    
    return result


def apply_z_shift(
    transition_matrix: pd.DataFrame,
    z_shift: float = 0.5
) -> pd.DataFrame:
    """
    Apply a z-shift to a transition matrix.
    
    Shifts probability mass toward worse buckets (positive z) or better
    buckets (negative z) by redistributing off-diagonal probabilities.
    
    The shift works by treating each row's off-diagonal probabilities as a
    distribution and shifting the mean of that distribution by z_shift positions.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix.
    z_shift : float, default=0.5
        Number of positions to shift. Positive = worse, negative = better.
        
    Returns
    -------
    pd.DataFrame
        Z-shifted transition matrix.
    """
    buckets = list(transition_matrix.index)
    n = len(buckets)
    shifted = np.zeros((n, n))
    P = transition_matrix.values.astype(float)
    
    for i in range(n):
        for j in range(n):
            # Source index j maps to target j + z_shift
            target = j + z_shift
            
            # Distribute between floor and ceil of target
            lower = int(np.floor(target))
            upper = int(np.ceil(target))
            frac = target - lower
            
            if lower == upper:
                # Exact integer shift
                idx = max(0, min(n - 1, lower))
                shifted[i, idx] += P[i, j]
            else:
                # Weighted split between two adjacent buckets
                if 0 <= lower < n:
                    shifted[i, lower] += P[i, j] * (1 - frac)
                elif lower < 0:
                    shifted[i, 0] += P[i, j] * (1 - frac)
                else:
                    shifted[i, n - 1] += P[i, j] * (1 - frac)
                    
                if 0 <= upper < n:
                    shifted[i, upper] += P[i, j] * frac
                elif upper >= n:
                    shifted[i, n - 1] += P[i, j] * frac
                else:
                    shifted[i, 0] += P[i, j] * frac
        
        # Renormalize
        row_sum = shifted[i].sum()
        if row_sum > 0:
            shifted[i] /= row_sum
    
    result = pd.DataFrame(shifted, index=buckets, columns=buckets)
    logger.info("Applied z-shift: %.2f positions", z_shift)
    
    return result


def generate_scenarios(
    transition_matrix: pd.DataFrame,
    scenarios_config: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple stressed transition matrices for scenario analysis.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Base (TTC) transition matrix.
    scenarios_config : Dict[str, Dict], optional
        Configuration for each scenario. Each entry must contain:
        - 'factor': stress factor to apply (float)
        - 'weight': probability weight for this scenario (float)
        
        Default scenarios:
        - base: factor=1.0, weight=0.40
        - optimistic: factor=0.7, weight=0.30
        - pessimistic: factor=1.8, weight=0.30
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping scenario name to stressed transition matrix.
        Also includes '_weights' key with scenario weights.
    """
    if scenarios_config is None:
        scenarios_config = {
            'base': {'factor': 1.0, 'weight': 0.40},
            'optimistic': {'factor': 0.7, 'weight': 0.30},
            'pessimistic': {'factor': 1.8, 'weight': 0.30}
        }
    
    # Validate weights sum to 1
    total_weight = sum(cfg['weight'] for cfg in scenarios_config.values())
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(
            "Scenario weights sum to %.4f, not 1.0. "
            "Normalizing weights.", total_weight
        )
        for cfg in scenarios_config.values():
            cfg['weight'] /= total_weight
    
    scenarios = {}
    weights = {}
    
    for name, config in scenarios_config.items():
        factor = config['factor']
        weight = config['weight']
        
        if abs(factor - 1.0) < 1e-10:
            scenarios[name] = transition_matrix.copy()
        else:
            scenarios[name] = apply_scalar_stress(
                transition_matrix, 
                stress_factor=factor,
                direction='worsen'
            )
        
        weights[name] = weight
        logger.info("Generated scenario '%s': factor=%.2f, weight=%.2f", name, factor, weight)
    
    scenarios['_weights'] = weights
    return scenarios


def weighted_scenario_matrix(
    scenario_matrices: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Blend multiple scenario matrices into a single forward-looking matrix.
    
    Computes: M_weighted = sum(w_i * M_i) for each scenario i.
    
    Parameters
    ----------
    scenario_matrices : Dict[str, pd.DataFrame]
        Dictionary of scenario name -> transition matrix.
        May contain '_weights' key from generate_scenarios().
    weights : Dict[str, float], optional
        Probability weights for each scenario. If None, uses equal weights
        or '_weights' from scenario_matrices if available.
        
    Returns
    -------
    pd.DataFrame
        Weighted (forward-looking) transition matrix.
    """
    # Extract weights
    if weights is None:
        if '_weights' in scenario_matrices:
            weights = scenario_matrices['_weights']
        else:
            # Equal weights
            matrix_names = [k for k in scenario_matrices if k != '_weights']
            weights = {name: 1.0 / len(matrix_names) for name in matrix_names}
    
    # Filter out non-matrix entries
    matrix_names = [k for k in scenario_matrices if k != '_weights' and k in weights]
    
    if not matrix_names:
        raise ValueError("No valid scenario matrices found")
    
    # Get shape from first matrix
    first_matrix = scenario_matrices[matrix_names[0]]
    result = pd.DataFrame(
        0.0, 
        index=first_matrix.index, 
        columns=first_matrix.columns
    )
    
    total_weight = sum(weights[name] for name in matrix_names)
    
    for name in matrix_names:
        w = weights[name] / total_weight  # normalize
        result += w * scenario_matrices[name]
    
    logger.info(
        "Created weighted scenario matrix from %d scenarios: %s",
        len(matrix_names), 
        {n: f"{weights[n]:.0%}" for n in matrix_names}
    )
    
    return result
