"""
LGD Engine — Loss Given Default calculations from transition data.

Provides empirical LGD estimation methods:
- lgd_from_cure_rates: LGD implícito derivado da cure rate da matriz de transição
- lgd_workout: LGD Workout rastreando recuperação real pós-default por contrato
- lgd_by_segment: Agregação de LGD por segmento/produto
"""

from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def lgd_from_cure_rates(
    transition_matrix: pd.DataFrame,
    default_states: Optional[List[str]] = None,
    recovery_haircut: float = 0.0
) -> pd.DataFrame:
    """
    Estimate LGD from the transition matrix cure rates.
    
    For each default state, computes:
      cure_rate = P(exit default → any non-default bucket)
      lgd_implied ≈ 1 − cure_rate × (1 − recovery_haircut)
    
    This is a model-implied LGD that requires no additional data beyond
    the fitted transition matrix.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix (rows = from, cols = to).
    default_states : List[str], optional
        Bucket labels considered as default. Defaults to last bucket.
    recovery_haircut : float, default=0.0
        Reduction applied to cure-based recovery (0.0 = no haircut,
        0.3 = 30% haircut on recovery estimate). Accounts for partial
        recovery costs (legal, operational).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - default_state: the default bucket label
        - cure_rate: probability of exiting default
        - recovery_rate: cure_rate × (1 − haircut)
        - lgd_implied: 1 − recovery_rate
        
    Examples
    --------
    >>> lgd_from_cure_rates(learner.transition_matrix_, default_states=['1'])
      default_state  cure_rate  recovery_rate  lgd_implied
    0             1       0.08          0.08         0.92
    """
    
    buckets = list(transition_matrix.index)
    
    if default_states is None:
        default_states = [buckets[-1]]
    
    non_default = [b for b in buckets if b not in default_states]
    
    records = []
    for ds in default_states:
        if ds not in transition_matrix.index:
            logger.warning("Default state '%s' not in transition matrix", ds)
            continue
        
        # Cure rate = sum of probabilities of transitioning from default
        # to any non-default state
        cure_rate = sum(
            transition_matrix.loc[ds, nd]
            for nd in non_default
            if nd in transition_matrix.columns
        )
        
        recovery_rate = cure_rate * (1.0 - recovery_haircut)
        lgd = 1.0 - recovery_rate
        
        records.append({
            'default_state': ds,
            'cure_rate': round(cure_rate, 6),
            'recovery_rate': round(recovery_rate, 6),
            'lgd_implied': round(lgd, 6)
        })
    
    result = pd.DataFrame(records)
    
    if not result.empty:
        logger.info(
            "LGD from cure rates: %s",
            ", ".join(
                f"{r['default_state']}={r['lgd_implied']:.2%}"
                for r in records
            )
        )
    
    return result


def lgd_workout(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    bucket_col: str,
    ead_col: str,
    default_states: List[str],
    recovery_col: Optional[str] = None,
    workout_periods: int = 24
) -> pd.DataFrame:
    """
    Calculate workout LGD per contract from observed post-default recovery.
    
    For each contract that enters a default state:
    1. Identifies the date of entry into default
    2. Tracks EAD changes over the following workout_periods
    3. Computes: LGD = 1 − (total_recovered / ead_at_default)
    
    If ``recovery_col`` is not provided, recovery is estimated from 
    **decreases** in EAD (negative ΔBalance = payment received).
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data sorted by id and time.
    id_col : str
        Contract identifier column.
    time_col : str
        Reference date column.
    bucket_col : str
        Bucket / rating column.
    ead_col : str
        Exposure at Default (balance) column.
    default_states : List[str]
        Bucket labels considered as default.
    recovery_col : str, optional
        Column with explicit recovery amounts. If None, inferred from
        EAD decreases.
    workout_periods : int, default=24
        Maximum months to track post-default recovery.
        
    Returns
    -------
    pd.DataFrame
        One row per default event with columns:
        - contract_id, default_date, ead_at_default
        - total_recovered, lgd, cure_flag
        - months_in_workout
        
    Examples
    --------
    >>> lgd_df = lgd_workout(
    ...     df, id_col='id', time_col='data_ref',
    ...     bucket_col='pd_rating', ead_col='saldo',
    ...     default_states=['1']
    ... )
    """
    
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Mark default observations
    data['_is_default'] = data[bucket_col].astype(str).isin(
        [str(s) for s in default_states]
    )
    
    # Find first default entry for each contract
    # A default entry = first observation where contract is in default,
    # preceded by a non-default observation (or first obs of contract)
    data['_prev_default'] = data.groupby(id_col)['_is_default'].shift(1)
    data['_entered_default'] = data['_is_default'] & (
        data['_prev_default'].fillna(False) == False  # noqa: E712
    )
    
    default_entries = data[data['_entered_default']].copy()
    
    if default_entries.empty:
        logger.warning("No default entries found in the data")
        return pd.DataFrame(columns=[
            'contract_id', 'default_date', 'ead_at_default',
            'total_recovered', 'lgd', 'cure_flag', 'months_in_workout'
        ])
    
    records = []
    
    for _, entry in default_entries.iterrows():
        cid = entry[id_col]
        default_date = entry[time_col]
        ead_at_default = entry[ead_col]
        
        if ead_at_default <= 0:
            continue
        
        # Get all observations for this contract after default
        contract_data = data[
            (data[id_col] == cid) & 
            (data[time_col] >= default_date)
        ].head(workout_periods + 1)  # +1 includes the default obs itself
        
        if len(contract_data) <= 1:
            # No post-default observations
            records.append({
                'contract_id': cid,
                'default_date': default_date,
                'ead_at_default': ead_at_default,
                'total_recovered': 0.0,
                'lgd': 1.0,
                'cure_flag': False,
                'months_in_workout': 0
            })
            continue
        
        # Calculate recovery
        if recovery_col is not None and recovery_col in data.columns:
            # Use explicit recovery column
            post_default = contract_data.iloc[1:]  # exclude default obs
            total_recovered = post_default[recovery_col].sum()
        else:
            # Infer recovery from EAD decreases
            ead_series = contract_data[ead_col].values
            decreases = np.diff(ead_series)
            # Negative changes = payments (recovery)
            total_recovered = abs(decreases[decreases < 0].sum())
        
        # Check if contract cured (returned to non-default)
        post_default_buckets = contract_data.iloc[1:][bucket_col].astype(str)
        non_default_states = [
            str(s) for s in post_default_buckets 
            if str(s) not in [str(d) for d in default_states]
        ]
        cure_flag = len(non_default_states) > 0
        
        lgd = max(0.0, 1.0 - (total_recovered / ead_at_default))
        lgd = min(lgd, 1.0)
        
        months_in_workout = len(contract_data) - 1
        
        records.append({
            'contract_id': cid,
            'default_date': default_date,
            'ead_at_default': ead_at_default,
            'total_recovered': round(total_recovered, 2),
            'lgd': round(lgd, 6),
            'cure_flag': cure_flag,
            'months_in_workout': months_in_workout
        })
    
    result = pd.DataFrame(records)
    
    if not result.empty:
        logger.info(
            "LGD workout: %d default events, mean LGD=%.2f%%, "
            "cure rate=%.1f%%",
            len(result),
            result['lgd'].mean() * 100,
            result['cure_flag'].mean() * 100
        )
    
    return result


def lgd_by_segment(
    lgd_df: pd.DataFrame,
    segment_col: Optional[str] = None,
    lgd_col: str = 'lgd',
    ead_col: str = 'ead_at_default',
    weighted: bool = True
) -> pd.DataFrame:
    """
    Aggregate LGD by segment (e.g., product type, collateral type).
    
    Parameters
    ----------
    lgd_df : pd.DataFrame
        Output of ``lgd_workout()`` with per-contract LGD.
    segment_col : str, optional
        Column to group by (e.g., 'tipo_produto').
        If None, returns overall portfolio LGD.
    lgd_col : str, default='lgd'
        Column with LGD values.
    ead_col : str, default='ead_at_default'
        Column with EAD for EAD-weighted averaging.
    weighted : bool, default=True
        If True, compute EAD-weighted mean LGD.
        
    Returns
    -------
    pd.DataFrame
        Aggregated LGD with columns: segment, count, lgd_mean,
        lgd_weighted (if weighted), lgd_median, lgd_std.
    """
    
    if lgd_df.empty:
        return pd.DataFrame(columns=[
            'segment', 'count', 'lgd_mean', 'lgd_median', 'lgd_std'
        ])
    
    if segment_col is not None and segment_col in lgd_df.columns:
        groups = lgd_df.groupby(segment_col)
    else:
        lgd_df = lgd_df.copy()
        lgd_df['_segment'] = 'ALL'
        groups = lgd_df.groupby('_segment')
    
    records = []
    for name, group in groups:
        record = {
            'segment': name,
            'count': len(group),
            'lgd_mean': round(group[lgd_col].mean(), 6),
            'lgd_median': round(group[lgd_col].median(), 6),
            'lgd_std': round(group[lgd_col].std(), 6) if len(group) > 1 else 0.0
        }
        
        if weighted and ead_col in group.columns:
            total_ead = group[ead_col].sum()
            if total_ead > 0:
                record['lgd_weighted'] = round(
                    (group[lgd_col] * group[ead_col]).sum() / total_ead, 6
                )
            else:
                record['lgd_weighted'] = record['lgd_mean']
        
        records.append(record)
    
    result = pd.DataFrame(records)
    
    logger.info(
        "LGD by segment: %s",
        ", ".join(
            f"{r['segment']}={r['lgd_mean']:.2%}"
            for r in records
        )
    )
    
    return result


# ──────────────────────────────────────────────────────────────────────────────
# LGD Rating Engine
# ──────────────────────────────────────────────────────────────────────────────

class LGDRatingEngine:
    """
    Product-specific LGD Rating Engine.
    
    Assigns an LGD rating to every contract in a portfolio based on 
    observed transition data and product characteristics. Each product type
    has its own decay curve and collateral treatment.
    
    Workflow::
    
        engine = LGDRatingEngine(GIRO_TPV)
        engine.fit(df, id_col='id', time_col='data_ref',
                   bucket_col='dpd', ead_col='saldo')
        
        # Analyze product behavior
        decay = engine.get_decay_curves()
        
        # Rate every contract
        rated = engine.rate(snapshot, ead_col='saldo',
                            pct_clean_col='pct_clean',
                            collateral_type_col='garantia_tipo')
    
    Parameters
    ----------
    schema : ProductSchema
        Product-specific configuration (buckets, collateral, etc.).
    """
    
    def __init__(self, schema):
        from .product_schema import ProductSchema
        if not isinstance(schema, ProductSchema):
            raise TypeError(
                f"schema must be a ProductSchema, got {type(schema).__name__}"
            )
        self.schema = schema
        self._learner = None
        self._cure_rates = None
        self._implied_lgd = None
        self._workout_lgd = None
        self._is_fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        bucket_col: str,
        ead_col: str,
        segment_col: Optional[str] = None,
        workout_periods: int = 24
    ) -> "LGDRatingEngine":
        """
        Learn transition patterns and LGD from observed data.
        
        Steps:
        1. Fit TransitionMatrixLearner with product-specific buckets
        2. Extract cure rates per bucket
        3. Compute implied LGD from cure rates
        4. Compute workout LGD per defaulted contract
        
        Parameters
        ----------
        df : pd.DataFrame
            Panel data with at least id, time, DPD, and EAD columns.
        id_col : str
            Contract identifier.
        time_col : str
            Reference date column.
        bucket_col : str
            Column with DPD values (numeric).
        ead_col : str
            Exposure at Default / balance column.
        segment_col : str, optional
            Optional segmentation column.
        workout_periods : int, default=24
            Months to track post-default recovery.
            
        Returns
        -------
        self
        """
        from .transition_matrix_learner import TransitionMatrixLearner
        
        # 1. Fit transition matrix
        self._learner = TransitionMatrixLearner(
            buckets=self.schema.buckets,
            min_observations=10
        )
        self._learner.fit(
            df, id_col=id_col, time_col=time_col,
            bucket_col=bucket_col, segment_col=segment_col
        )
        
        # 2. Compute implied LGD from cure rates
        default_labels = self.schema.default_buckets
        self._implied_lgd = lgd_from_cure_rates(
            self._learner.transition_matrix_,
            default_states=default_labels
        )
        
        # 3. Compute cure rates per bucket (using pd_engine)
        from .pd_engine import cure_rates as compute_cure_rates
        self._cure_rates = compute_cure_rates(self._learner.transition_matrix_)
        
        # 4. Workout LGD per defaulted contract
        self._workout_lgd = lgd_workout(
            df, id_col=id_col, time_col=time_col,
            bucket_col='_lgd_bucket',
            ead_col=ead_col,
            default_states=default_labels,
            workout_periods=workout_periods
        ) if '_lgd_bucket' in df.columns else self._compute_workout_from_raw(
            df, id_col, time_col, bucket_col, ead_col,
            default_labels, workout_periods
        )
        
        self._is_fitted = True
        
        logger.info(
            "LGDRatingEngine fitted for product '%s': "
            "%d transitions, implied LGD=%s",
            self.schema.name,
            len(self._learner._transitions) if self._learner._transitions is not None else 0,
            self._implied_lgd['lgd_implied'].to_dict() if not self._implied_lgd.empty else 'N/A'
        )
        
        return self
    
    def _compute_workout_from_raw(
        self, df, id_col, time_col, bucket_col, ead_col,
        default_labels, workout_periods
    ):
        """Compute workout LGD by first assigning buckets to raw DPD data."""
        data = df.copy()
        
        # Assign bucket labels from numeric DPD
        buckets = self.schema.buckets
        labels = self.schema.bucket_labels
        
        def assign_bucket(dpd_val):
            try:
                dpd_val = float(dpd_val)
            except (ValueError, TypeError):
                return str(dpd_val)
            for i in range(len(buckets) - 1, -1, -1):
                if dpd_val >= buckets[i]:
                    return labels[i]
            return labels[0]
        
        data['_lgd_bucket'] = data[bucket_col].apply(assign_bucket)
        
        return lgd_workout(
            data, id_col=id_col, time_col=time_col,
            bucket_col='_lgd_bucket', ead_col=ead_col,
            default_states=default_labels,
            workout_periods=workout_periods
        )
    
    def _check_is_fitted(self, method_name: str):
        if not self._is_fitted:
            raise RuntimeError(
                f"LGDRatingEngine must be fit before calling {method_name}()"
            )
    
    def get_decay_curves(self, n_periods: int = 12) -> pd.DataFrame:
        """
        Compute decay curves: probability of reaching default from each bucket.
        
        Uses the transition matrix to project forward n_periods,
        showing how the probability of default evolves from each starting state.
        
        Parameters
        ----------
        n_periods : int, default=12
            Number of periods to project forward.
            
        Returns
        -------
        pd.DataFrame
            Index = period (1..n_periods), columns = starting buckets,
            values = cumulative probability of default by that period.
        """
        self._check_is_fitted("get_decay_curves")
        from .pd_engine import cumulative_pd
        
        default_labels = self.schema.default_buckets
        cpd = cumulative_pd(
            self._learner.transition_matrix_,
            n_periods=n_periods,
            default_states=default_labels
        )
        
        return cpd
    
    def get_cure_rates_summary(self) -> pd.DataFrame:
        """
        Cure rates summary per bucket.
        
        Returns
        -------
        pd.DataFrame
            Cure rates from worse to better buckets, including total cure.
        """
        self._check_is_fitted("get_cure_rates_summary")
        return self._cure_rates
    
    def get_implied_lgd(self) -> pd.DataFrame:
        """
        Implied LGD from cure rates.
        
        Returns
        -------
        pd.DataFrame
            LGD implied per default state.
        """
        self._check_is_fitted("get_implied_lgd")
        return self._implied_lgd
    
    def get_workout_lgd(self) -> pd.DataFrame:
        """
        Workout LGD per defaulted contract.
        
        Returns
        -------
        pd.DataFrame
            Contract-level workout LGD.
        """
        self._check_is_fitted("get_workout_lgd")
        return self._workout_lgd
    
    def rate(
        self,
        snapshot: pd.DataFrame,
        ead_col: str,
        bucket_col: Optional[str] = None,
        pct_clean_col: Optional[str] = None,
        collateral_type_col: Optional[str] = None,
        collateral_stressed_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Assign LGD rating to every contract in a snapshot.
        
        The rating combines:
        1. Implied LGD from the current DPD bucket (transition matrix)
        2. Blended collateral-based LGD from the product schema
        
        The final LGD per contract is:
            lgd = max(lgd_implied_bucket, lgd_collateral_blend)
        
        LGD ratings (A through E):
          - A: LGD ≤ 20%  (very low loss expected)
          - B: LGD ≤ 40%  (low loss)
          - C: LGD ≤ 60%  (moderate loss)
          - D: LGD ≤ 80%  (high loss)
          - E: LGD > 80%  (very high loss)
        
        Parameters
        ----------
        snapshot : pd.DataFrame
            Current portfolio snapshot (one row per contract).
        ead_col : str
            Exposure at Default column.
        bucket_col : str, optional
            DPD column. If None, lgd_implied is skipped and only
            collateral-based LGD is used.
        pct_clean_col : str, optional
            Column with % clean exposure (0.0–1.0). Default: 1.0 (all clean).
        collateral_type_col : str, optional
            Column with collateral type. Ignored if pct_clean=1.0.
        collateral_stressed_col : str, optional
            Boolean column indicating stressed collateral (e.g., churn).
            
        Returns
        -------
        pd.DataFrame
            Copy of snapshot with added columns:
            - lgd_implied: LGD from transition matrix for current bucket
            - lgd_collateral: LGD from collateral composition
            - lgd_value: final blended LGD (0.0–1.0)
            - lgd_rating: rating label (A–E)
        """
        self._check_is_fitted("rate")
        
        result = snapshot.copy()
        
        # --- 1. LGD from transition matrix (bucket-based) ---
        if bucket_col is not None and bucket_col in result.columns:
            # Map each bucket to its implied P(default) → use as LGD proxy
            decay_1p = self.get_decay_curves(n_periods=12)
            
            # Use 12-period cumulative PD as the implied LGD contribution
            bucket_lgd_map = {}
            for col in decay_1p.columns:
                bucket_lgd_map[col] = decay_1p.loc[12, col] if 12 in decay_1p.index else 0.0
            
            # Assign bucket labels from DPD values
            buckets = self.schema.buckets
            labels = self.schema.bucket_labels
            
            def assign_bucket(dpd_val):
                try:
                    dpd_val = float(dpd_val)
                except (ValueError, TypeError):
                    return str(dpd_val)
                for i in range(len(buckets) - 1, -1, -1):
                    if dpd_val >= buckets[i]:
                        return labels[i]
                return labels[0]
            
            result['_bucket_label'] = result[bucket_col].apply(assign_bucket)
            result['lgd_implied'] = result['_bucket_label'].map(bucket_lgd_map).fillna(
                self.schema.clean_lgd
            )
            
            # For default buckets, force lgd_implied to implied cure-based LGD
            if not self._implied_lgd.empty:
                default_lgd_map = dict(zip(
                    self._implied_lgd['default_state'],
                    self._implied_lgd['lgd_implied']
                ))
                for ds, lgd_val in default_lgd_map.items():
                    mask = result['_bucket_label'] == ds
                    result.loc[mask, 'lgd_implied'] = lgd_val
            
            result.drop(columns=['_bucket_label'], inplace=True)
        else:
            result['lgd_implied'] = self.schema.clean_lgd
        
        # --- 2. LGD from collateral composition ---
        if pct_clean_col is not None and pct_clean_col in result.columns:
            pct_clean_series = result[pct_clean_col]
        else:
            pct_clean_series = pd.Series(1.0, index=result.index)
        
        if collateral_type_col is not None and collateral_type_col in result.columns:
            col_type_series = result[collateral_type_col]
        else:
            col_type_series = pd.Series(None, index=result.index)
        
        if collateral_stressed_col is not None and collateral_stressed_col in result.columns:
            stressed_series = result[collateral_stressed_col].astype(bool)
        else:
            stressed_series = pd.Series(False, index=result.index)
        
        result['lgd_collateral'] = [
            self.schema.compute_contract_lgd(
                pct_clean=pc,
                collateral_type=ct,
                collateral_stressed=st
            )
            for pc, ct, st in zip(pct_clean_series, col_type_series, stressed_series)
        ]
        
        # --- 3. Final LGD = max of implied and collateral-based ---
        result['lgd_value'] = np.maximum(
            result['lgd_implied'], result['lgd_collateral']
        )
        result['lgd_value'] = result['lgd_value'].clip(0.0, 1.0).round(4)
        
        # --- 4. Assign rating labels ---
        conditions = [
            result['lgd_value'] <= 0.20,
            result['lgd_value'] <= 0.40,
            result['lgd_value'] <= 0.60,
            result['lgd_value'] <= 0.80,
        ]
        choices = ['A', 'B', 'C', 'D']
        result['lgd_rating'] = np.select(conditions, choices, default='E')
        
        logger.info(
            "LGD ratings assigned for '%s': "
            "A=%d, B=%d, C=%d, D=%d, E=%d (mean LGD=%.2f%%)",
            self.schema.name,
            (result['lgd_rating'] == 'A').sum(),
            (result['lgd_rating'] == 'B').sum(),
            (result['lgd_rating'] == 'C').sum(),
            (result['lgd_rating'] == 'D').sum(),
            (result['lgd_rating'] == 'E').sum(),
            result['lgd_value'].mean() * 100
        )
        
        return result
    
    @staticmethod
    def compare_products(
        engines: List["LGDRatingEngine"],
        n_periods: int = 12
    ) -> pd.DataFrame:
        """
        Compare decay curves across different product types.
        
        Shows how different products reach default at different speeds,
        which is the core insight for product-specific LGD.
        
        Parameters
        ----------
        engines : List[LGDRatingEngine]
            Fitted engines for different products.
        n_periods : int, default=12
            Number of periods for decay curve comparison.
            
        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with columns:
            product, period, bucket, cumulative_pd
        """
        records = []
        
        for engine in engines:
            engine._check_is_fitted("compare_products")
            decay = engine.get_decay_curves(n_periods=n_periods)
            
            for period in decay.index:
                for bucket in decay.columns:
                    records.append({
                        'product': engine.schema.name,
                        'period': period,
                        'bucket': bucket,
                        'cumulative_pd': round(decay.loc[period, bucket], 6)
                    })
        
        result = pd.DataFrame(records)
        
        if not result.empty:
            logger.info(
                "Product comparison: %s",
                ", ".join(e.schema.name for e in engines)
            )
        
        return result
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LGDRatingEngine("
            f"product='{self.schema.name}', "
            f"buckets={len(self.schema.buckets)}, "
            f"collateral_types={list(self.schema.collateral_types.keys())}, "
            f"status={status})"
        )

