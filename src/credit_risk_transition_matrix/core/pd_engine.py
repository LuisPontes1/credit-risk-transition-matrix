"""
PD Engine — Probability of Default calculations derived from transition matrices.

This module provides functions for computing:
- Cumulative PD (PD acumulada até período t)
- Marginal PD (PD condicional período a período)
- PD Term Structure (curva completa)
- Survival curve (1 - PD cumulativa)
- Cure rates (migração de buckets piores para melhores)
- Absorbing state analysis (tempo esperado até default via forma fundamental)
- EAD-weighted transition matrices
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cumulative_pd(
    transition_matrix: pd.DataFrame,
    n_periods: int = 12,
    default_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate cumulative PD for each starting bucket over multiple periods.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix (rows = from, cols = to)
    n_periods : int, default=12
        Number of periods to project forward
    default_states : List[str], optional
        Bucket labels considered as default. If None, uses the last bucket.
        
    Returns
    -------
    pd.DataFrame
        Index = period (1..n_periods), columns = starting buckets,
        values = cumulative probability of reaching a default state.
    """
    
    buckets = list(transition_matrix.index)
    
    if default_states is None:
        default_states = [buckets[-1]]
    
    default_indices = [buckets.index(s) for s in default_states if s in buckets]
    
    P = transition_matrix.values.astype(float)
    n = len(buckets)
    
    results = np.zeros((n_periods, n))
    
    # P^t for t = 1..n_periods
    Pt = np.eye(n)
    for t in range(n_periods):
        Pt = Pt @ P
        for i in range(n):
            results[t, i] = Pt[i, default_indices].sum()
    
    return pd.DataFrame(
        results,
        index=range(1, n_periods + 1),
        columns=buckets
    )


def marginal_pd(
    transition_matrix: pd.DataFrame,
    n_periods: int = 12,
    default_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate marginal (conditional) PD: PD(t) - PD(t-1).
    
    The marginal PD represents the probability of defaulting in exactly
    period t, given survival up to period t-1.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix
    n_periods : int, default=12
        Number of periods
    default_states : List[str], optional
        Bucket labels considered as default
        
    Returns
    -------
    pd.DataFrame
        Index = period (1..n_periods), columns = starting buckets,
        values = marginal PD for that period.
    """
    
    cpd = cumulative_pd(transition_matrix, n_periods, default_states)
    
    # Marginal = difference between consecutive cumulative PDs
    mpd = cpd.diff()
    mpd.iloc[0] = cpd.iloc[0]  # First period marginal = first period cumulative
    
    return mpd


def pd_term_structure(
    transition_matrix: pd.DataFrame,
    n_periods: int = 12,
    default_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Complete PD term structure: cumulative, marginal, and survival per period.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix
    n_periods : int, default=12
        Number of periods
    default_states : List[str], optional
        Bucket labels considered as default
        
    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (bucket, metric) where metric is one of
        'cumulative_pd', 'marginal_pd', 'survival'.
    """
    
    cpd = cumulative_pd(transition_matrix, n_periods, default_states)
    mpd = marginal_pd(transition_matrix, n_periods, default_states)
    surv = survival_curve(transition_matrix, n_periods, default_states)
    
    buckets = list(transition_matrix.index)
    
    arrays = []
    for bucket in buckets:
        arrays.append(pd.DataFrame({
            (bucket, 'cumulative_pd'): cpd[bucket],
            (bucket, 'marginal_pd'): mpd[bucket],
            (bucket, 'survival'): surv[bucket]
        }))
    
    result = pd.concat(arrays, axis=1)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    result.index.name = 'period'
    
    return result


def survival_curve(
    transition_matrix: pd.DataFrame,
    n_periods: int = 12,
    default_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Survival probability: 1 - cumulative PD.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix
    n_periods : int, default=12
        Number of periods
    default_states : List[str], optional
        Bucket labels considered as default
        
    Returns
    -------
    pd.DataFrame
        Index = period, columns = starting buckets, values = survival probability.
    """
    
    cpd = cumulative_pd(transition_matrix, n_periods, default_states)
    return 1.0 - cpd


def cure_rates(transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cure rates from the transition matrix.
    
    A cure rate is the probability of migrating from a worse (higher) bucket
    to a better (lower) bucket. This is extracted from the lower-triangular 
    portion of the matrix.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: from_bucket, to_bucket, cure_probability.
        Only includes transitions from worse to better buckets.
    """
    
    buckets = list(transition_matrix.index)
    n = len(buckets)
    
    records = []
    for i in range(1, n):  # Start from second bucket (first has no worse state)
        from_bucket = buckets[i]
        total_cure = 0.0
        
        for j in range(i):  # Only buckets better (lower index) than current
            to_bucket = buckets[j]
            prob = transition_matrix.iloc[i, j]
            
            if prob > 0:
                records.append({
                    'from_bucket': from_bucket,
                    'to_bucket': to_bucket,
                    'cure_probability': prob
                })
                total_cure += prob
        
        # Also record total cure rate for this bucket
        records.append({
            'from_bucket': from_bucket,
            'to_bucket': '_TOTAL_CURE_',
            'cure_probability': total_cure
        })
    
    return pd.DataFrame(records)


def absorbing_analysis(
    transition_matrix: pd.DataFrame,
    absorbing_states: Optional[List[str]] = None
) -> Dict[str, pd.Series]:
    """
    Absorbing state analysis using the fundamental matrix of a Markov chain.
    
    Computes:
    - Expected number of steps until absorption from each transient state
    - Absorption probabilities (if multiple absorbing states)
    
    Uses the fundamental matrix: N = (I - Q)^{-1}, where Q is the 
    sub-matrix of transient-to-transient transitions.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix
    absorbing_states : List[str], optional
        States considered absorbing. If None, uses the last bucket.
        
    Returns
    -------
    Dict[str, pd.Series]
        'expected_steps': Expected time to absorption from each transient state.
        'absorption_probabilities': Probability of being absorbed in each absorbing state.
    """
    
    buckets = list(transition_matrix.index)
    
    if absorbing_states is None:
        absorbing_states = [buckets[-1]]
    
    transient_states = [b for b in buckets if b not in absorbing_states]
    
    if not transient_states:
        logger.warning("No transient states found. All states are absorbing.")
        return {
            'expected_steps': pd.Series(dtype=float),
            'absorption_probabilities': pd.DataFrame()
        }
    
    # Extract Q (transient → transient) and R (transient → absorbing)
    Q = transition_matrix.loc[transient_states, transient_states].values.astype(float)
    R = transition_matrix.loc[transient_states, absorbing_states].values.astype(float)
    
    n_transient = len(transient_states)
    
    # Fundamental matrix N = (I - Q)^{-1}
    try:
        I = np.eye(n_transient)
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix (I - Q). Cannot compute fundamental matrix. "
                       "This may indicate the chain is not absorbing.")
        return {
            'expected_steps': pd.Series(np.inf, index=transient_states),
            'absorption_probabilities': pd.DataFrame()
        }
    
    # Expected number of steps to absorption: sum of rows of N
    expected_steps = pd.Series(
        N.sum(axis=1),
        index=transient_states,
        name='expected_steps'
    )
    
    # Absorption probabilities: B = N @ R
    B = N @ R
    absorption_probs = pd.DataFrame(
        B,
        index=transient_states,
        columns=absorbing_states
    )
    
    return {
        'expected_steps': expected_steps,
        'absorption_probabilities': absorption_probs
    }


def weighted_matrix(
    transitions_df: pd.DataFrame,
    bucket_labels: List[str],
    weight_col: str = 'weight'
) -> pd.DataFrame:
    """
    Calculate EAD-weighted transition matrix from transitions data.
    
    Parameters
    ----------
    transitions_df : pd.DataFrame
        Transitions with 'from_bucket', 'to_bucket', and weight column
    bucket_labels : List[str]
        Ordered list of bucket labels
    weight_col : str, default='weight'
        Column containing exposure weights (EAD)
        
    Returns
    -------
    pd.DataFrame
        Weighted transition matrix where probabilities are proportional
        to exposure rather than contract count.
    """
    
    matrix = pd.DataFrame(0.0, index=bucket_labels, columns=bucket_labels)
    
    for from_bucket in bucket_labels:
        mask = transitions_df['from_bucket'] == from_bucket
        subset = transitions_df[mask]
        
        if len(subset) == 0:
            continue
            
        total_weight = subset[weight_col].sum()
        if total_weight == 0:
            continue
            
        for to_bucket in bucket_labels:
            to_mask = subset['to_bucket'] == to_bucket
            matrix.loc[from_bucket, to_bucket] = (
                subset.loc[to_mask, weight_col].sum() / total_weight
            )
    
    return matrix
