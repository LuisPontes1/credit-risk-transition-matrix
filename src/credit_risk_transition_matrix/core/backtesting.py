"""
Backtesting Framework — model validation per CPC 4.966 / IFRS 9.

Compares predicted risk parameters (PD, LGD, ECL) against
realized outcomes to validate model accuracy. Essential for
regulatory compliance and audit.
"""

from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def backtest_pd(
    predicted_pd: pd.Series,
    actual_default: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Backtest PD predictions against actual defaults (calibration test).
    
    Groups contracts into PD buckets and compares predicted vs observed
    default rates. A well-calibrated model should show predicted ≈ observed.
    
    Parameters
    ----------
    predicted_pd : pd.Series
        Model-predicted PD (0 to 1).
    actual_default : pd.Series
        Binary default flag (1=defaulted, 0=survived).
    n_bins : int, default=10
        Number of PD buckets for calibration analysis.
        
    Returns
    -------
    pd.DataFrame
        Calibration table with:
        - pd_bin: PD range
        - count: number of contracts
        - predicted_dr: average predicted PD
        - observed_dr: observed default rate
        - ratio: observed / predicted (1.0 = perfect)
    """
    df = pd.DataFrame({
        'pd': predicted_pd.astype(float),
        'default': actual_default.astype(int)
    })
    
    # Handle edge case: all same PD
    if df['pd'].nunique() <= 1:
        return pd.DataFrame({
            'pd_bin': ['ALL'],
            'count': [len(df)],
            'predicted_dr': [df['pd'].mean()],
            'observed_dr': [df['default'].mean()],
            'ratio': [
                df['default'].mean() / df['pd'].mean()
                if df['pd'].mean() > 0 else np.nan
            ]
        })
    
    df['pd_bin'] = pd.qcut(df['pd'], q=n_bins, duplicates='drop')
    
    result = df.groupby('pd_bin', observed=True).agg(
        count=('default', 'size'),
        predicted_dr=('pd', 'mean'),
        observed_dr=('default', 'mean')
    ).reset_index()
    
    result['ratio'] = np.where(
        result['predicted_dr'] > 0,
        result['observed_dr'] / result['predicted_dr'],
        np.nan
    )
    result['pd_bin'] = result['pd_bin'].astype(str)
    
    logger.info(
        "PD backtest: %d bins, overall predicted=%.4f, observed=%.4f",
        len(result), df['pd'].mean(), df['default'].mean()
    )
    
    return result


def backtest_lgd(
    predicted_lgd: pd.Series,
    actual_lgd: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Backtest LGD predictions against realized loss rates.
    
    Parameters
    ----------
    predicted_lgd : pd.Series
        Model-predicted LGD (0 to 1).
    actual_lgd : pd.Series
        Realized LGD (1 - recovery_rate), only for defaulted contracts.
    n_bins : int, default=5
        Number of LGD bins.
        
    Returns
    -------
    pd.DataFrame
        Calibration table with predicted vs realized LGD.
    """
    df = pd.DataFrame({
        'predicted': predicted_lgd.astype(float),
        'actual': actual_lgd.astype(float)
    }).dropna()
    
    if df.empty:
        return pd.DataFrame(columns=[
            'lgd_bin', 'count', 'predicted_mean', 'actual_mean', 'bias'
        ])
    
    if df['predicted'].nunique() <= 1:
        return pd.DataFrame({
            'lgd_bin': ['ALL'],
            'count': [len(df)],
            'predicted_mean': [df['predicted'].mean()],
            'actual_mean': [df['actual'].mean()],
            'bias': [df['predicted'].mean() - df['actual'].mean()]
        })
    
    df['lgd_bin'] = pd.qcut(df['predicted'], q=n_bins, duplicates='drop')
    
    result = df.groupby('lgd_bin', observed=True).agg(
        count=('actual', 'size'),
        predicted_mean=('predicted', 'mean'),
        actual_mean=('actual', 'mean')
    ).reset_index()
    
    result['bias'] = result['predicted_mean'] - result['actual_mean']
    result['lgd_bin'] = result['lgd_bin'].astype(str)
    
    return result


def backtest_ecl(
    predicted_ecl: pd.Series,
    actual_loss: pd.Series,
    stage_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Backtest ECL against actual credit losses.
    
    Parameters
    ----------
    predicted_ecl : pd.Series
        Model-predicted ECL (monetary value).
    actual_loss : pd.Series
        Realized credit loss (monetary value).
    stage_col : pd.Series, optional
        Stage classification for breakdown.
        
    Returns
    -------
    pd.DataFrame
        ECL accuracy summary with total and optionally by stage.
    """
    df = pd.DataFrame({
        'predicted': predicted_ecl.astype(float),
        'actual': actual_loss.astype(float)
    })
    
    if stage_col is not None:
        df['stage'] = stage_col.values
        groups = ['ALL'] + sorted(df['stage'].unique().tolist())
    else:
        groups = ['ALL']
    
    records = []
    for group in groups:
        if group == 'ALL':
            subset = df
        else:
            subset = df[df['stage'] == group]
        
        if subset.empty:
            continue
        
        pred_total = subset['predicted'].sum()
        actual_total = subset['actual'].sum()
        
        records.append({
            'group': str(group),
            'count': len(subset),
            'ecl_predicted': round(pred_total, 2),
            'ecl_actual': round(actual_total, 2),
            'ratio': round(pred_total / actual_total, 4) if actual_total > 0 else np.nan,
            'bias': round(pred_total - actual_total, 2),
            'bias_pct': round(
                (pred_total - actual_total) / actual_total * 100, 2
            ) if actual_total > 0 else np.nan
        })
    
    return pd.DataFrame(records)


def discrimination_metrics(
    pd_values: pd.Series,
    default_flag: pd.Series,
) -> Dict[str, float]:
    """
    Compute discrimination metrics for PD model.
    
    Parameters
    ----------
    pd_values : pd.Series
        Model-predicted PD.
    default_flag : pd.Series
        Binary default indicator (1=default, 0=survive).
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - auc: Area Under ROC Curve
        - gini: Gini coefficient (2×AUC - 1)
        - ks: Kolmogorov-Smirnov statistic
        - default_rate: observed default rate
        - n_defaults: number of defaults
        - n_total: total contracts
    """
    pd_val = pd_values.astype(float).values
    actual = default_flag.astype(int).values
    
    n_total = len(actual)
    n_defaults = actual.sum()
    n_non_defaults = n_total - n_defaults
    
    if n_defaults == 0 or n_non_defaults == 0:
        logger.warning("Cannot compute discrimination: need both defaults and non-defaults")
        return {
            'auc': np.nan,
            'gini': np.nan,
            'ks': np.nan,
            'default_rate': n_defaults / n_total if n_total > 0 else 0.0,
            'n_defaults': int(n_defaults),
            'n_total': n_total
        }
    
    # Sort by PD descending
    order = np.argsort(-pd_val)
    sorted_actual = actual[order]
    
    # Cumulative distributions for ROC / KS
    cum_defaults = np.cumsum(sorted_actual) / n_defaults
    cum_non_defaults = np.cumsum(1 - sorted_actual) / n_non_defaults
    
    # AUC via trapezoidal rule (compat with numpy 1.x and 2.x)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = _trapz(cum_defaults, cum_non_defaults)
    gini = 2 * auc - 1
    
    # KS statistic
    ks = np.max(np.abs(cum_defaults - cum_non_defaults))
    
    metrics = {
        'auc': round(float(auc), 4),
        'gini': round(float(gini), 4),
        'ks': round(float(ks), 4),
        'default_rate': round(n_defaults / n_total, 6),
        'n_defaults': int(n_defaults),
        'n_total': n_total
    }
    
    logger.info(
        "Discrimination: AUC=%.4f, Gini=%.4f, KS=%.4f "
        "(defaults=%d/%d, DR=%.2f%%)",
        metrics['auc'], metrics['gini'], metrics['ks'],
        n_defaults, n_total, metrics['default_rate'] * 100
    )
    
    return metrics
