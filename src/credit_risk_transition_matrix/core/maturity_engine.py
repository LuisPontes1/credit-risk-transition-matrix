"""
Maturity Engine — contractual and behavioral maturity estimation.

For term loans (giro): maturity = remaining contractual term.
For revolving products (cartão, cheque): maturity is behavioral,
estimated from historical utilization / balance decay patterns.
"""

from typing import Optional, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def contractual_maturity(
    df: pd.DataFrame,
    maturity_date_col: str,
    ref_date_col: str,
) -> pd.Series:
    """
    Compute remaining contractual maturity in months.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contract-level data.
    maturity_date_col : str
        Column with contract maturity date.
    ref_date_col : str
        Column with reference/snapshot date.
        
    Returns
    -------
    pd.Series
        Remaining maturity in months (float, clipped at 0).
    """
    maturity = pd.to_datetime(df[maturity_date_col])
    ref = pd.to_datetime(df[ref_date_col])
    
    remaining_days = (maturity - ref).dt.total_seconds() / 86400
    remaining = (remaining_days / 30.4375).clip(lower=0)
    
    return remaining.round(1)


def behavioral_maturity(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    balance_col: str,
    method: str = 'balance_halflife'
) -> pd.DataFrame:
    """
    Estimate behavioral maturity for revolving products.
    
    Revolving products have no contractual maturity, so we estimate
    how long the exposure will persist based on observed patterns.
    
    Methods
    -------
    - ``balance_halflife``: Time for balance to decay to 50% for
      closed/inactive accounts. Proxy for expected life.
    - ``median_tenure``: Median observed account lifetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data (multiple periods per contract).
    id_col : str
        Contract identifier.
    time_col : str
        Reference date column.
    balance_col : str
        Outstanding balance column.
    method : str, default='balance_halflife'
        Estimation method.
        
    Returns
    -------
    pd.DataFrame
        One row per contract with estimated ``behavioral_maturity`` in months.
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([id_col, time_col])
    
    if method == 'balance_halflife':
        records = []
        for cid, group in data.groupby(id_col):
            group = group.sort_values(time_col)
            balances = group[balance_col].values
            times = group[time_col].values
            
            if len(balances) < 2 or balances[0] <= 0:
                records.append({
                    'contract_id': cid,
                    'behavioral_maturity': np.nan
                })
                continue
            
            initial_balance = balances[0]
            half_target = initial_balance * 0.5
            
            # Find when balance crosses half
            halflife_months = None
            for i in range(1, len(balances)):
                if balances[i] <= half_target:
                    # Interpolate
                    months_elapsed = (times[i] - times[0]) / np.timedelta64(1, 'D') / 30.4375
                    halflife_months = months_elapsed
                    break
            
            if halflife_months is None:
                # Balance never halved; extrapolate from decay rate
                months_total = (times[-1] - times[0]) / np.timedelta64(1, 'D') / 30.4375
                if months_total > 0 and balances[-1] < initial_balance:
                    decay_rate = 1 - (balances[-1] / initial_balance)
                    if decay_rate > 0:
                        halflife_months = months_total * 0.5 / decay_rate
                    else:
                        halflife_months = 60.0  # fallback: 5 years
                else:
                    halflife_months = 60.0
            
            # Behavioral maturity ≈ 2 × halflife (time for ~full amortization)
            records.append({
                'contract_id': cid,
                'behavioral_maturity': round(min(halflife_months * 2, 120), 1)
            })
        
        return pd.DataFrame(records)
    
    elif method == 'median_tenure':
        tenure = data.groupby(id_col).agg(
            first_date=(time_col, 'min'),
            last_date=(time_col, 'max')
        )
        tenure_days = (tenure['last_date'] - tenure['first_date']).dt.total_seconds() / 86400
        tenure['tenure_months'] = (tenure_days / 30.4375).round(1)
        
        median_val = tenure['tenure_months'].median()
        
        result = tenure[['tenure_months']].reset_index()
        result.columns = ['contract_id', 'behavioral_maturity']
        result['behavioral_maturity'] = result['behavioral_maturity'].clip(
            lower=0
        ).fillna(median_val)
        
        return result
    
    raise ValueError(f"Unknown method: '{method}'. Use 'balance_halflife' or 'median_tenure'.")


def effective_maturity(
    df: pd.DataFrame,
    contractual_col: Optional[str] = None,
    behavioral_col: Optional[str] = None,
    floor_months: float = 12.0,
    cap_months: float = 60.0,
) -> pd.Series:
    """
    Compute effective maturity with regulatory floor and cap.
    
    Uses contractual maturity if available, otherwise behavioral.
    Applies floor (minimum 12m per Basel/4.966) and cap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contract data.
    contractual_col : str, optional
        Column with remaining contractual maturity in months.
    behavioral_col : str, optional
        Column with behavioral maturity in months.
    floor_months : float, default=12.0
        Minimum maturity (Basel: 1 year).
    cap_months : float, default=60.0
        Maximum maturity (Basel: 5 years).
        
    Returns
    -------
    pd.Series
        Effective maturity in months.
    """
    if contractual_col and contractual_col in df.columns:
        maturity = df[contractual_col].astype(float)
    elif behavioral_col and behavioral_col in df.columns:
        maturity = df[behavioral_col].astype(float)
    else:
        logger.warning("No maturity column found, using floor=%d months", floor_months)
        maturity = pd.Series(floor_months, index=df.index)
    
    result = maturity.clip(lower=floor_months, upper=cap_months)
    
    logger.info(
        "Effective maturity: mean=%.1f, median=%.1f, "
        "floor=%.0f, cap=%.0f months",
        result.mean(), result.median(), floor_months, cap_months
    )
    
    return result.round(1)
