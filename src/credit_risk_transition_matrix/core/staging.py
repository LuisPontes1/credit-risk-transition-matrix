"""
Staging — IFRS 9 / CPC 4966 stage classification.

Classifies contracts into Stage 1 (performing), Stage 2 (underperforming),
and Stage 3 (default/impaired) based on configurable bucket thresholds.

Stage definitions:
- Stage 1: No significant deterioration → 12-month PD
- Stage 2: Significant deterioration → Lifetime PD
- Stage 3: Default/impaired → PD = 100%
"""

from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classify_stage(
    df: pd.DataFrame,
    bucket_col: str = 'bucket',
    bucket_labels: Optional[List[str]] = None,
    stage1_max_bucket: Optional[str] = None,
    default_buckets: Optional[List[str]] = None,
    origin_bucket_col: Optional[str] = None,
    deterioration_threshold: int = 2
) -> pd.DataFrame:
    """
    Classify each observation into IFRS 9 Stage 1, 2, or 3.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a bucket column.
    bucket_col : str, default='bucket'
        Column name containing the current bucket/risk grade.
    bucket_labels : List[str], optional
        Ordered list of bucket labels (best to worst).
        If None, inferred from unique values of bucket_col.
    stage1_max_bucket : str, optional
        Maximum bucket still considered Stage 1.
        Default: second bucket in bucket_labels (e.g., '15-29').
    default_buckets : List[str], optional
        Buckets considered as default (Stage 3).
        Default: last bucket in bucket_labels.
    origin_bucket_col : str, optional
        Column with the origination bucket for each contract.
        If provided, Stage 2 is also triggered when the contract has
        deteriorated by >= deterioration_threshold positions.
    deterioration_threshold : int, default=2
        Number of bucket positions a contract must worsen to be classified
        as Stage 2 (only used if origin_bucket_col is provided).
        
    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with 'stage' column added (values: 1, 2, or 3).
    """
    result = df.copy()
    
    # Determine bucket ordering
    if bucket_labels is None:
        bucket_labels = sorted(result[bucket_col].unique())
        logger.warning(
            "bucket_labels not provided; using sorted unique values: %s",
            bucket_labels
        )
    
    bucket_order = {label: idx for idx, label in enumerate(bucket_labels)}
    n_buckets = len(bucket_labels)
    
    # Defaults
    if default_buckets is None:
        default_buckets = [bucket_labels[-1]]
    
    if stage1_max_bucket is None:
        # Default: second bucket (index 1) if available, else first
        stage1_max_idx = min(1, n_buckets - 1)
        stage1_max_bucket = bucket_labels[stage1_max_idx]
    
    stage1_max_idx = bucket_order.get(stage1_max_bucket, 1)
    
    # Classify
    current_idx = result[bucket_col].map(bucket_order)
    
    # Stage 3: default
    is_default = result[bucket_col].isin(default_buckets)
    
    # Stage 1: within threshold
    is_stage1 = current_idx <= stage1_max_idx
    
    # Check deterioration from origination
    if origin_bucket_col is not None and origin_bucket_col in result.columns:
        origin_idx = result[origin_bucket_col].map(bucket_order)
        deterioration = current_idx - origin_idx
        is_deteriorated = deterioration >= deterioration_threshold
    else:
        is_deteriorated = pd.Series(False, index=result.index)
    
    # Assign stages: Stage 3 > Stage 2 > Stage 1
    result['stage'] = 2  # Default to Stage 2
    result.loc[is_stage1 & ~is_deteriorated, 'stage'] = 1
    result.loc[is_default, 'stage'] = 3
    
    logger.info(
        "Stage classification: S1=%d, S2=%d, S3=%d",
        (result['stage'] == 1).sum(),
        (result['stage'] == 2).sum(),
        (result['stage'] == 3).sum()
    )
    
    return result


def classify_stage_sicr(
    df: pd.DataFrame,
    pd_current_col: str,
    pd_origination_col: str,
    default_flag_col: Optional[str] = None,
    dpd_col: Optional[str] = None,
    sicr_relative_threshold: float = 2.0,
    sicr_absolute_threshold: float = 0.03,
    dpd_stage2_backstop: int = 30,
    dpd_stage3_threshold: int = 90,
    low_risk_exemption: float = 0.003,
) -> pd.DataFrame:
    """
    Classify contracts using PD-based SICR per Resolução 4.966 / IFRS 9.
    
    SICR (Significant Increase in Credit Risk) triggers Stage 2 when:
    
    1. **Relative criterion**: PD_current / PD_origination >= threshold (default: 2x)
    2. **Absolute criterion**: PD_current - PD_origination >= threshold (default: 3pp)
    3. **DPD backstop**: DPD >= 30 days (rebuttable presumption per IFRS 9)
    
    Exceptions:
    - **Low-risk exemption**: if PD_current < low_risk_exemption, stays Stage 1
      regardless of deterioration (investment-grade exception)
    
    Stage 3 = default (DPD >= 90 or explicit default flag).
    
    Parameters
    ----------
    df : pd.DataFrame
        Contract-level snapshot.
    pd_current_col : str
        Column with current PD (from behavioral model).
    pd_origination_col : str
        Column with PD at origination (from application model).
    default_flag_col : str, optional
        Boolean column indicating default. If None, uses dpd_col threshold.
    dpd_col : str, optional
        Days past due column. Used for backstop and Stage 3.
    sicr_relative_threshold : float, default=2.0
        Relative PD increase threshold (e.g., 2.0 = PD doubled).
    sicr_absolute_threshold : float, default=0.03
        Absolute PD increase threshold (e.g., 0.03 = 3 percentage points).
    dpd_stage2_backstop : int, default=30
        DPD threshold for Stage 2 backstop.
    dpd_stage3_threshold : int, default=90
        DPD threshold for Stage 3 (default).
    low_risk_exemption : float, default=0.003
        PD below this threshold = low risk, exempt from SICR 
        (0.003 = 0.3%, investment grade).
        Set to 0.0 to disable.
        
    Returns
    -------
    pd.DataFrame
        Copy of input with columns added:
        - ``stage``: 1, 2, or 3
        - ``sicr_trigger``: reason for Stage 2 ('relative', 'absolute', 
          'dpd_backstop', or None)
          
    Examples
    --------
    >>> staged = classify_stage_sicr(
    ...     df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
    ...     dpd_col='dpd'
    ... )
    """
    result = df.copy()
    
    pd_current = result[pd_current_col].astype(float)
    pd_origin = result[pd_origination_col].astype(float)
    
    # Initialize
    result['stage'] = 1
    result['sicr_trigger'] = None
    
    # ── Stage 3: Default ──
    is_default = pd.Series(False, index=result.index)
    
    if default_flag_col is not None and default_flag_col in result.columns:
        is_default = result[default_flag_col].astype(bool)
    
    if dpd_col is not None and dpd_col in result.columns:
        dpd = result[dpd_col].astype(float)
        is_default = is_default | (dpd >= dpd_stage3_threshold)
    
    result.loc[is_default, 'stage'] = 3
    result.loc[is_default, 'sicr_trigger'] = 'default'
    
    # ── Stage 2: SICR triggers (only for non-default) ──
    non_default = ~is_default
    
    # Low-risk exemption
    is_low_risk = (pd_current < low_risk_exemption) & (low_risk_exemption > 0)
    eligible_for_sicr = non_default & ~is_low_risk
    
    # 1. Relative criterion: PD ratio
    safe_origin = pd_origin.replace(0, np.nan)
    pd_ratio = pd_current / safe_origin
    relative_trigger = eligible_for_sicr & (pd_ratio >= sicr_relative_threshold)
    result.loc[relative_trigger, 'stage'] = 2
    result.loc[relative_trigger & (result['sicr_trigger'].isna()), 'sicr_trigger'] = 'relative'
    
    # 2. Absolute criterion: PD delta
    pd_delta = pd_current - pd_origin
    absolute_trigger = eligible_for_sicr & (pd_delta >= sicr_absolute_threshold)
    result.loc[absolute_trigger, 'stage'] = 2
    result.loc[absolute_trigger & (result['sicr_trigger'].isna()), 'sicr_trigger'] = 'absolute'
    
    # 3. DPD backstop
    if dpd_col is not None and dpd_col in result.columns:
        dpd = result[dpd_col].astype(float)
        dpd_trigger = non_default & (dpd >= dpd_stage2_backstop) & (dpd < dpd_stage3_threshold)
        result.loc[dpd_trigger, 'stage'] = 2
        result.loc[dpd_trigger & (result['sicr_trigger'].isna()), 'sicr_trigger'] = 'dpd_backstop'
    
    logger.info(
        "SICR staging: S1=%d, S2=%d (relative=%d, absolute=%d, dpd=%d), "
        "S3=%d, low_risk_exempt=%d",
        (result['stage'] == 1).sum(),
        (result['stage'] == 2).sum(),
        (result['sicr_trigger'] == 'relative').sum(),
        (result['sicr_trigger'] == 'absolute').sum(),
        (result['sicr_trigger'] == 'dpd_backstop').sum(),
        (result['stage'] == 3).sum(),
        is_low_risk.sum()
    )
    
    return result


def stage_migration_matrix(
    df: pd.DataFrame,
    id_col: str = 'id_contrato',
    time_col: str = 'data_ref',
    stage_col: str = 'stage'
) -> pd.DataFrame:
    """
    Calculate 3x3 migration matrix between IFRS 9 stages.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with stage classification, sorted by id and time.
    id_col : str, default='id_contrato'
        Contract identifier column.
    time_col : str, default='data_ref'
        Time/date column.
    stage_col : str, default='stage'
        Stage classification column (values 1, 2, 3).
        
    Returns
    -------
    pd.DataFrame
        3x3 transition matrix between stages (rows sum to 1).
    """
    sorted_df = df.sort_values([id_col, time_col]).copy()
    
    # Next stage for each observation
    sorted_df['next_stage'] = sorted_df.groupby(id_col)[stage_col].shift(-1)
    
    # Drop last observation per contract (no next stage)
    transitions = sorted_df.dropna(subset=['next_stage']).copy()
    transitions['next_stage'] = transitions['next_stage'].astype(int)
    
    # Build 3x3 matrix
    stages = [1, 2, 3]
    stage_labels = ['Stage 1', 'Stage 2', 'Stage 3']
    
    matrix = pd.DataFrame(0.0, index=stage_labels, columns=stage_labels)
    
    for i, from_stage in enumerate(stages):
        mask = transitions[stage_col] == from_stage
        subset = transitions[mask]
        total = len(subset)
        
        if total == 0:
            continue
        
        for j, to_stage in enumerate(stages):
            count = (subset['next_stage'] == to_stage).sum()
            matrix.iloc[i, j] = count / total
    
    logger.info("Stage migration matrix computed from %d transitions", len(transitions))
    
    return matrix


def stage_summary(
    df: pd.DataFrame,
    stage_col: str = 'stage',
    ead_col: Optional[str] = None,
    pd_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Portfolio summary by IFRS 9 stage.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with stage classification.
    stage_col : str, default='stage'
        Stage column (values 1, 2, 3).
    ead_col : str, optional
        Column with Exposure at Default (EAD).
    pd_col : str, optional
        Column with PD values.
        
    Returns
    -------
    pd.DataFrame
        Summary with count, percentage, and optionally EAD and PD per stage.
    """
    stage_labels = {1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3'}
    
    records = []
    total_count = len(df)
    
    for stage_num, stage_name in stage_labels.items():
        mask = df[stage_col] == stage_num
        subset = df[mask]
        count = len(subset)
        
        record = {
            'stage': stage_name,
            'count': count,
            'pct_count': count / total_count if total_count > 0 else 0.0
        }
        
        if ead_col is not None and ead_col in df.columns:
            record['ead_total'] = subset[ead_col].sum()
            record['ead_mean'] = subset[ead_col].mean() if count > 0 else 0.0
            record['pct_ead'] = (
                subset[ead_col].sum() / df[ead_col].sum() 
                if df[ead_col].sum() > 0 else 0.0
            )
        
        if pd_col is not None and pd_col in df.columns:
            record['pd_mean'] = subset[pd_col].mean() if count > 0 else 0.0
            record['pd_median'] = subset[pd_col].median() if count > 0 else 0.0
        
        records.append(record)
    
    result = pd.DataFrame(records)
    
    logger.info(
        "Stage summary: S1=%d (%.1f%%), S2=%d (%.1f%%), S3=%d (%.1f%%)",
        records[0]['count'], records[0]['pct_count'] * 100,
        records[1]['count'], records[1]['pct_count'] * 100,
        records[2]['count'], records[2]['pct_count'] * 100
    )
    
    return result
