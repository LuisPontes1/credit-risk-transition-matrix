"""
ECL Calculator — Expected Credit Loss computation per IFRS 9 / CPC 4966.

Implements ECL calculation by stage:
- Stage 1: ECL = PD_12m × LGD × EAD
- Stage 2: ECL = PD_lifetime × LGD × EAD
- Stage 3: ECL = LGD × EAD (PD = 100%)
"""

from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

from .pd_engine import cumulative_pd

logger = logging.getLogger(__name__)


def calculate_ecl(
    pd_values: pd.Series,
    ead_values: pd.Series,
    lgd: Union[float, pd.Series] = 0.45,
    discount_rate: float = 0.0
) -> pd.Series:
    """
    Calculate ECL = PD × LGD × EAD for a set of observations.
    
    Parameters
    ----------
    pd_values : pd.Series
        Probability of Default for each observation.
    ead_values : pd.Series
        Exposure at Default for each observation.
    lgd : float or pd.Series, default=0.45
        Loss Given Default. Can be a fixed value or per-observation.
    discount_rate : float, default=0.0
        Effective interest rate for discounting (annual). 
        Set to 0.0 for undiscounted ECL.
        
    Returns
    -------
    pd.Series
        ECL for each observation.
    """
    ecl = pd_values * lgd * ead_values
    
    if discount_rate > 0:
        # Simple single-period discount
        ecl = ecl / (1 + discount_rate)
    
    return ecl


def ecl_by_stage(
    transition_matrix: pd.DataFrame,
    df: pd.DataFrame,
    bucket_col: str = 'bucket',
    stage_col: str = 'stage',
    ead_col: str = 'ead',
    lgd: Union[float, str] = 0.45,
    n_periods_12m: int = 12,
    n_periods_lifetime: int = 60,
    default_states: Optional[List[str]] = None,
    discount_rate: float = 0.0
) -> pd.DataFrame:
    """
    Calculate ECL per contract using stage-appropriate PD horizons.
    
    Stage 1: ECL = PD(12m) × LGD × EAD
    Stage 2: ECL = PD(lifetime) × LGD × EAD
    Stage 3: ECL = 1.0 × LGD × EAD  (PD = 100%)
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Fitted transition matrix (or scenario-weighted matrix).
    df : pd.DataFrame
        DataFrame with bucket, stage, and EAD columns.
    bucket_col : str, default='bucket'
        Column with current bucket/risk grade.
    stage_col : str, default='stage'
        Column with IFRS 9 stage (1, 2, 3).
    ead_col : str, default='ead'
        Column with Exposure at Default.
    lgd : float or str, default=0.45
        Loss Given Default. If string, treated as column name in df.
    n_periods_12m : int, default=12
        Number of periods for 12-month PD (Stage 1).
    n_periods_lifetime : int, default=60
        Number of periods for lifetime PD (Stage 2).
    default_states : List[str], optional
        Bucket labels considered as default. Defaults to last bucket.
    discount_rate : float, default=0.0
        Annual discount rate for ECL.
        
    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: 'pd_used', 'lgd_used', 'ecl'.
    """
    result = df.copy()
    
    # Compute PD tables
    cpd_12m = cumulative_pd(transition_matrix, n_periods_12m, default_states)
    cpd_lifetime = cumulative_pd(transition_matrix, n_periods_lifetime, default_states)
    
    # Map bucket -> PD for each horizon
    pd_12m_map = {}
    pd_lifetime_map = {}
    
    for bucket in transition_matrix.index:
        if bucket in cpd_12m.columns:
            pd_12m_map[bucket] = cpd_12m.loc[n_periods_12m, bucket]
        if bucket in cpd_lifetime.columns:
            pd_lifetime_map[bucket] = cpd_lifetime.loc[n_periods_lifetime, bucket]
    
    # Assign PD based on stage
    result['pd_used'] = 0.0
    
    # Stage 1: 12-month PD
    stage1_mask = result[stage_col] == 1
    result.loc[stage1_mask, 'pd_used'] = (
        result.loc[stage1_mask, bucket_col].map(pd_12m_map).fillna(0.0)
    )
    
    # Stage 2: Lifetime PD
    stage2_mask = result[stage_col] == 2
    result.loc[stage2_mask, 'pd_used'] = (
        result.loc[stage2_mask, bucket_col].map(pd_lifetime_map).fillna(0.0)
    )
    
    # Stage 3: PD = 100%
    stage3_mask = result[stage_col] == 3
    result.loc[stage3_mask, 'pd_used'] = 1.0
    
    # Resolve LGD
    if isinstance(lgd, str):
        result['lgd_used'] = result[lgd]
    else:
        result['lgd_used'] = lgd
    
    # Calculate ECL
    result['ecl'] = calculate_ecl(
        pd_values=result['pd_used'],
        ead_values=result[ead_col],
        lgd=result['lgd_used'],
        discount_rate=discount_rate
    )
    
    total_ecl = result['ecl'].sum()
    total_ead = result[ead_col].sum()
    
    logger.info(
        "ECL calculated: total=%.2f, coverage=%.4f%%, "
        "S1=%.2f, S2=%.2f, S3=%.2f",
        total_ecl,
        (total_ecl / total_ead * 100) if total_ead > 0 else 0,
        result.loc[stage1_mask, 'ecl'].sum(),
        result.loc[stage2_mask, 'ecl'].sum(),
        result.loc[stage3_mask, 'ecl'].sum()
    )
    
    return result


def portfolio_ecl(
    ecl_df: pd.DataFrame,
    stage_col: str = 'stage',
    ead_col: str = 'ead',
    ecl_col: str = 'ecl',
    segment_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate ECL at portfolio level with breakdown by stage and optionally segment.
    
    Parameters
    ----------
    ecl_df : pd.DataFrame
        DataFrame with ECL calculated (output of ecl_by_stage).
    stage_col : str, default='stage'
        Stage column.
    ead_col : str, default='ead'
        EAD column.
    ecl_col : str, default='ecl'
        ECL column.
    segment_col : str, optional
        Optional segment column for additional grouping.
        
    Returns
    -------
    pd.DataFrame
        Aggregated ECL summary with columns:
        count, ead_total, ecl_total, coverage_ratio.
    """
    stage_labels = {1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3'}
    
    if segment_col is not None and segment_col in ecl_df.columns:
        group_cols = [segment_col, stage_col]
    else:
        group_cols = [stage_col]
    
    agg = ecl_df.groupby(group_cols).agg(
        count=(ead_col, 'size'),
        ead_total=(ead_col, 'sum'),
        ecl_total=(ecl_col, 'sum'),
        pd_mean=('pd_used', 'mean')
    ).reset_index()
    
    agg['coverage_ratio'] = np.where(
        agg['ead_total'] > 0,
        agg['ecl_total'] / agg['ead_total'],
        0.0
    )
    
    # Map stage numbers to labels
    agg[stage_col] = agg[stage_col].map(stage_labels).fillna(agg[stage_col])
    
    # Add total row
    total = pd.DataFrame([{
        stage_col: 'Total',
        'count': agg['count'].sum(),
        'ead_total': agg['ead_total'].sum(),
        'ecl_total': agg['ecl_total'].sum(),
        'pd_mean': ecl_df['pd_used'].mean(),
        'coverage_ratio': (
            agg['ecl_total'].sum() / agg['ead_total'].sum()
            if agg['ead_total'].sum() > 0 else 0.0
        )
    }])
    
    if segment_col is not None and segment_col in agg.columns:
        total[segment_col] = 'ALL'
    
    result = pd.concat([agg, total], ignore_index=True)
    
    logger.info(
        "Portfolio ECL: %.2f (coverage %.4f%%)",
        result.iloc[-1]['ecl_total'],
        result.iloc[-1]['coverage_ratio'] * 100
    )
    
    return result
