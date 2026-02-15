"""
EAD Engine — Exposure at Default estimation with CCF.

For term loans (capital de giro):
    EAD = outstanding balance (simple)

For revolving products (cartão, cheque especial):
    EAD = drawn + CCF × (limit - drawn)

CCF (Credit Conversion Factor) represents how much of the undrawn
limit gets converted to exposure before default. Estimated empirically
from historical data.
"""

from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_ccf(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    balance_col: str,
    limit_col: str,
    bucket_col: str,
    default_states: List[str],
    lookback_periods: int = 12
) -> pd.DataFrame:
    """
    Estimate empirical CCF from observed pre-default behavior.
    
    CCF measures how much of the unused credit limit gets drawn
    before a contract defaults::
    
        CCF = (EAD_at_default - balance_12m_before) / (limit - balance_12m_before)
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data sorted by id and time.
    id_col : str
        Contract identifier column.
    time_col : str
        Reference date column.
    balance_col : str
        Outstanding balance column.
    limit_col : str
        Credit limit column.
    bucket_col : str
        DPD bucket / rating column.
    default_states : List[str]
        Bucket labels considered as default.
    lookback_periods : int, default=12
        Months to look back from default for the reference balance.
        
    Returns
    -------
    pd.DataFrame
        One row per default event with columns:
        - contract_id, default_date
        - balance_at_default, balance_pre_default
        - limit_at_default, undrawn_pre_default
        - ccf
        
    Examples
    --------
    >>> ccf_df = estimate_ccf(df, 'id', 'data_ref', 'saldo', 'limite',
    ...                        'dpd_bucket', ['90-119', '120+'])
    >>> print(f"Mean CCF: {ccf_df['ccf'].mean():.2%}")
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Identify default entries
    data['_is_default'] = data[bucket_col].astype(str).isin(
        [str(s) for s in default_states]
    )
    data['_prev_default'] = data.groupby(id_col)['_is_default'].shift(1)
    data['_entered_default'] = data['_is_default'] & (
        data['_prev_default'].fillna(False) == False  # noqa: E712
    )
    
    default_entries = data[data['_entered_default']].copy()
    
    if default_entries.empty:
        logger.warning("No default entries found for CCF estimation")
        return pd.DataFrame(columns=[
            'contract_id', 'default_date', 'balance_at_default',
            'balance_pre_default', 'limit_at_default',
            'undrawn_pre_default', 'ccf'
        ])
    
    records = []
    
    for _, entry in default_entries.iterrows():
        cid = entry[id_col]
        default_date = entry[time_col]
        balance_at_default = entry[balance_col]
        limit_at_default = entry[limit_col]
        
        # Get pre-default observation (lookback_periods months before)
        contract_history = data[
            (data[id_col] == cid) & (data[time_col] < default_date)
        ].sort_values(time_col)
        
        if contract_history.empty:
            continue
        
        # Use observation lookback_periods months before default
        if len(contract_history) >= lookback_periods:
            pre_obs = contract_history.iloc[-lookback_periods]
        else:
            pre_obs = contract_history.iloc[0]
        
        balance_pre = pre_obs[balance_col]
        limit_pre = pre_obs[limit_col] if limit_col in pre_obs.index else limit_at_default
        undrawn = limit_pre - balance_pre
        
        if undrawn <= 0:
            # Already fully drawn, CCF isn't meaningful
            ccf = 1.0
        else:
            additional_drawn = balance_at_default - balance_pre
            ccf = max(0.0, min(additional_drawn / undrawn, 1.0))
        
        records.append({
            'contract_id': cid,
            'default_date': default_date,
            'balance_at_default': round(balance_at_default, 2),
            'balance_pre_default': round(balance_pre, 2),
            'limit_at_default': round(limit_at_default, 2),
            'undrawn_pre_default': round(undrawn, 2),
            'ccf': round(ccf, 6)
        })
    
    result = pd.DataFrame(records)
    
    if not result.empty:
        logger.info(
            "CCF estimated from %d default events: mean=%.2f%%, "
            "median=%.2f%%, std=%.2f%%",
            len(result),
            result['ccf'].mean() * 100,
            result['ccf'].median() * 100,
            result['ccf'].std() * 100
        )
    
    return result


def ccf_by_segment(
    ccf_df: pd.DataFrame,
    segment_col: Optional[str] = None,
    ccf_col: str = 'ccf',
    weighted_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate CCF by segment.
    
    Parameters
    ----------
    ccf_df : pd.DataFrame
        Output of ``estimate_ccf()``.
    segment_col : str, optional
        Grouping column. If None, returns overall CCF.
    ccf_col : str, default='ccf'
        CCF value column.
    weighted_col : str, optional
        Column for weighted average (e.g., undrawn amount).
        
    Returns
    -------
    pd.DataFrame
        CCF summary per segment with mean, median, std, and count.
    """
    if ccf_df.empty:
        return pd.DataFrame(columns=['segment', 'count', 'ccf_mean',
                                     'ccf_median', 'ccf_std'])
    
    if segment_col is not None and segment_col in ccf_df.columns:
        groups = ccf_df.groupby(segment_col)
    else:
        temp = ccf_df.copy()
        temp['_segment'] = 'ALL'
        groups = temp.groupby('_segment')
    
    records = []
    for name, group in groups:
        record = {
            'segment': name,
            'count': len(group),
            'ccf_mean': round(group[ccf_col].mean(), 6),
            'ccf_median': round(group[ccf_col].median(), 6),
            'ccf_std': round(group[ccf_col].std(), 6) if len(group) > 1 else 0.0
        }
        
        if weighted_col and weighted_col in group.columns:
            total_w = group[weighted_col].sum()
            if total_w > 0:
                record['ccf_weighted'] = round(
                    (group[ccf_col] * group[weighted_col]).sum() / total_w, 6
                )
        
        records.append(record)
    
    return pd.DataFrame(records)


def compute_ead(
    snapshot: pd.DataFrame,
    balance_col: str,
    limit_col: Optional[str] = None,
    ccf: Union[float, str] = 0.75,
    product_type: str = 'term'
) -> pd.Series:
    """
    Compute EAD for each contract in a snapshot.
    
    - **Term loans**: ``EAD = balance`` (fixed amortization)
    - **Revolving**: ``EAD = balance + CCF × max(0, limit - balance)``
    
    Parameters
    ----------
    snapshot : pd.DataFrame
        Portfolio snapshot (one row per contract).
    balance_col : str
        Outstanding balance column.
    limit_col : str, optional
        Credit limit column (required for revolving).
    ccf : float or str, default=0.75
        Credit Conversion Factor. Float for fixed value, or column name
        for per-contract CCF.
    product_type : str, default='term'
        Product type: 'term' or 'revolving'.
        
    Returns
    -------
    pd.Series
        EAD per contract.
        
    Examples
    --------
    >>> ead = compute_ead(df, 'saldo', 'limite', ccf=0.80, product_type='revolving')
    """
    balance = snapshot[balance_col].copy()
    
    if product_type == 'term':
        return balance.clip(lower=0)
    
    if product_type == 'revolving':
        if limit_col is None or limit_col not in snapshot.columns:
            logger.warning(
                "limit_col not provided for revolving product, "
                "using balance as EAD"
            )
            return balance.clip(lower=0)
        
        limit = snapshot[limit_col]
        undrawn = (limit - balance).clip(lower=0)
        
        if isinstance(ccf, str):
            ccf_values = snapshot[ccf]
        else:
            ccf_values = ccf
        
        ead = balance + ccf_values * undrawn
        
        return ead.clip(lower=0).round(2)
    
    raise ValueError(f"Unknown product_type: '{product_type}'. Use 'term' or 'revolving'.")
