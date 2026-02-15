"""
ContractRiskEngine — CPC 4.966 / IFRS 9 compliant ECL orchestrator.

Integrates all contract-level risk components:
    PD (external) + LGD + EAD + SICR + Maturity + Forward-Looking → ECL

This is the main entry point for computing contract-level Expected Credit Loss.
PD is provided externally (from application/behavioral scoring models).
Everything else — LGD, EAD, staging, maturity — is computed by this engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

from .lgd_engine import LGDRatingEngine
from .ead_engine import compute_ead
from .staging import classify_stage_sicr
from .forward_looking import (
    MacroScenario, DEFAULT_SCENARIOS, apply_macro_overlay,
    scenario_ecl_breakdown
)
from .maturity_engine import effective_maturity
from .product_schema import ProductSchema

logger = logging.getLogger(__name__)


class ContractRiskEngine:
    """
    CPC 4.966-compliant contract risk engine.
    
    Orchestrates the full ECL calculation pipeline::
    
        PD (external) ──┐
                        ├─→ SICR → Stage 1/2/3
        LGD (internal) ──┤
        EAD (internal) ──┤
        Maturity ────────┤
        Macro overlay ───┘
                          → ECL per contract
    
    Parameters
    ----------
    product_schema : ProductSchema
        Product configuration (buckets, collateral, etc.).
    macro_scenarios : List[MacroScenario], optional
        Forward-looking scenarios. Defaults to BCB-aligned 3 scenarios.
    product_type : str, default='term'
        Product type for EAD computation: 'term' or 'revolving'.
        
    Examples
    --------
    >>> from credit_risk_transition_matrix import ContractRiskEngine, GIRO_TPV
    >>> engine = ContractRiskEngine(GIRO_TPV)
    >>> engine.fit(historical_df, id_col='id', time_col='data_ref',
    ...            bucket_col='dpd', ead_col='saldo')
    >>> ecl = engine.compute_ecl(
    ...     snapshot, pd_col='pd_bhv', pd_origination_col='pd_app',
    ...     ead_col='saldo', dpd_col='dpd'
    ... )
    """
    
    def __init__(
        self,
        product_schema: ProductSchema,
        macro_scenarios: Optional[List[MacroScenario]] = None,
        product_type: str = 'term',
    ):
        if not isinstance(product_schema, ProductSchema):
            raise TypeError(
                f"Expected ProductSchema, got {type(product_schema).__name__}"
            )
        
        self.schema = product_schema
        self.lgd_engine = LGDRatingEngine(product_schema)
        self.scenarios = macro_scenarios or DEFAULT_SCENARIOS
        self.product_type = product_type
        self._is_fitted = False
        self._ccf = 0.75  # default; updated when fit with revolving data
    
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        bucket_col: str,
        ead_col: str,
        recovery_col: Optional[str] = None,
        limit_col: Optional[str] = None,
    ) -> "ContractRiskEngine":
        """
        Fit the engine on historical data.
        
        Trains the LGD engine (transition matrix, cure rates, workout LGD)
        and optionally estimates CCF from revolving data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical panel data.
        id_col, time_col, bucket_col, ead_col : str
            Standard column mappings.
        recovery_col : str, optional
            Recovery amount column for workout LGD (reserved for future use).
        limit_col : str, optional
            Credit limit column (for CCF estimation on revolving).
            
        Returns
        -------
        ContractRiskEngine
            Self (fitted).
        """
        # 1. Fit LGD engine
        fit_kwargs = dict(
            id_col=id_col, time_col=time_col,
            bucket_col=bucket_col, ead_col=ead_col,
        )
        self.lgd_engine.fit(df, **fit_kwargs)
        
        # 2. Estimate CCF for revolving (if applicable)
        if self.product_type == 'revolving' and limit_col and limit_col in df.columns:
            from .ead_engine import estimate_ccf
            default_states = self.schema.default_buckets()
            default_labels = [str(s) for s in default_states]
            
            ccf_df = estimate_ccf(
                df, id_col=id_col, time_col=time_col,
                balance_col=ead_col, limit_col=limit_col,
                bucket_col=bucket_col, default_states=default_labels
            )
            
            if not ccf_df.empty:
                self._ccf = float(ccf_df['ccf'].median())
                logger.info("CCF estimated at %.2f%%", self._ccf * 100)
        
        self._is_fitted = True
        logger.info("ContractRiskEngine fitted for product '%s'", self.schema.name)
        
        return self
    
    def compute_ecl(
        self,
        snapshot: pd.DataFrame,
        pd_col: str,
        ead_col: str,
        pd_origination_col: Optional[str] = None,
        dpd_col: Optional[str] = None,
        bucket_col: Optional[str] = None,
        limit_col: Optional[str] = None,
        pct_clean_col: Optional[str] = None,
        collateral_type_col: Optional[str] = None,
        collateral_stressed_col: Optional[str] = None,
        maturity_col: Optional[str] = None,
        discount_rate: float = 0.0,
        sicr_relative_threshold: float = 2.0,
        sicr_absolute_threshold: float = 0.03,
        apply_forward_looking: bool = True,
    ) -> pd.DataFrame:
        """
        Compute contract-level ECL through the full pipeline.
        
        Pipeline steps:
        
        1. **Staging (SICR)**: Classify Stage 1/2/3 using PD-based SICR
        2. **LGD rating**: Assign contract-level LGD from engine
        3. **EAD**: Compute EAD (+ CCF for revolving)
        4. **Maturity**: Determine effective maturity horizon
        5. **Forward-looking**: Apply macro scenario overlay to PD and LGD
        6. **ECL**: Calculate ECL = PD × LGD × EAD, discounted
        
        Parameters
        ----------
        snapshot : pd.DataFrame
            Portfolio snapshot (one row per contract).
        pd_col : str
            Current PD column (from behavioral/application model).
        ead_col : str
            Outstanding balance / exposure column.
        pd_origination_col : str, optional
            PD at origination for SICR. If None, skips PD-based SICR.
        dpd_col : str, optional
            Days past due column. Used for DPD backstop in SICR.
        bucket_col : str, optional
            DPD bucket column for LGD rating.
        limit_col : str, optional
            Credit limit column for revolving EAD.
        pct_clean_col : str, optional
            Clean exposure percentage for collateral blend.
        collateral_type_col : str, optional
            Collateral type column.
        collateral_stressed_col : str, optional
            Collateral stress flag (e.g., churn).
        maturity_col : str, optional
            Remaining maturity in months column.
        discount_rate : float, default=0.0
            Annual effective interest rate for discounting.
        sicr_relative_threshold : float, default=2.0
            SICR relative PD threshold.
        sicr_absolute_threshold : float, default=0.03
            SICR absolute PD threshold (pp).
        apply_forward_looking : bool, default=True
            Whether to apply macro scenario overlay.
            
        Returns
        -------
        pd.DataFrame
            Copy of snapshot with added columns:
            - ``stage``: IFRS 9 stage (1, 2, 3)
            - ``sicr_trigger``: SICR trigger reason
            - ``lgd_value``: Contract-level LGD
            - ``lgd_rating``: LGD rating (A-E)
            - ``ead_computed``: EAD (possibly with CCF)
            - ``maturity_months``: Effective maturity
            - ``pd_adjusted``: PD after forward-looking overlay
            - ``lgd_adjusted``: LGD after forward-looking overlay
            - ``ecl``: Expected Credit Loss
            - ``ecl_coverage``: ECL / EAD ratio
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fit() before compute_ecl()")
        
        result = snapshot.copy()
        
        # ── Step 1: Staging (SICR) ──
        if pd_origination_col and pd_origination_col in result.columns:
            staged = classify_stage_sicr(
                result,
                pd_current_col=pd_col,
                pd_origination_col=pd_origination_col,
                dpd_col=dpd_col,
                sicr_relative_threshold=sicr_relative_threshold,
                sicr_absolute_threshold=sicr_absolute_threshold,
            )
            result['stage'] = staged['stage']
            result['sicr_trigger'] = staged['sicr_trigger']
        else:
            # Fallback: simple DPD-based staging
            if dpd_col and dpd_col in result.columns:
                dpd = result[dpd_col].astype(float)
                result['stage'] = np.where(
                    dpd >= 90, 3,
                    np.where(dpd >= 30, 2, 1)
                )
            else:
                result['stage'] = 1
            result['sicr_trigger'] = None
        
        # ── Step 2: LGD rating ──
        b_col = bucket_col or dpd_col
        if b_col and b_col in result.columns:
            rated = self.lgd_engine.rate(
                result, ead_col=ead_col, bucket_col=b_col,
                pct_clean_col=pct_clean_col,
                collateral_type_col=collateral_type_col,
                collateral_stressed_col=collateral_stressed_col,
            )
            result['lgd_value'] = rated['lgd_value']
            result['lgd_rating'] = rated['lgd_rating']
            result['lgd_implied'] = rated['lgd_implied']
            result['lgd_collateral'] = rated['lgd_collateral']
        else:
            result['lgd_value'] = self.schema.clean_lgd
            result['lgd_rating'] = 'D'
            result['lgd_implied'] = self.schema.clean_lgd
            result['lgd_collateral'] = self.schema.clean_lgd
        
        # ── Step 3: EAD ──
        result['ead_computed'] = compute_ead(
            result, balance_col=ead_col, limit_col=limit_col,
            ccf=self._ccf, product_type=self.product_type
        )
        
        # ── Step 4: Maturity ──
        result['maturity_months'] = effective_maturity(
            result, contractual_col=maturity_col
        )
        
        # ── Step 5: Forward-looking overlay ──
        pd_values = result[pd_col].astype(float)
        lgd_values = result['lgd_value'].astype(float)
        
        if apply_forward_looking:
            pd_adj, lgd_adj = apply_macro_overlay(
                pd_values, lgd_values, self.scenarios
            )
        else:
            pd_adj = pd_values
            lgd_adj = lgd_values
        
        result['pd_adjusted'] = pd_adj
        result['lgd_adjusted'] = lgd_adj
        
        # ── Step 6: ECL calculation ──
        # Adjust PD by stage:
        #   Stage 1: use 12-month PD (pd_adjusted as-is for short horizon)
        #   Stage 2: scale PD by maturity factor (simplified lifetime proxy)
        #   Stage 3: PD = 1.0
        pd_for_ecl = pd_adj.copy()
        
        is_stage2 = result['stage'] == 2
        is_stage3 = result['stage'] == 3
        
        # Stage 2: simple lifetime scaling
        maturity_factor = (result['maturity_months'] / 12.0).clip(upper=5.0)
        pd_for_ecl = np.where(
            is_stage2,
            (1 - (1 - pd_adj) ** maturity_factor).clip(upper=1.0),
            pd_for_ecl
        )
        pd_for_ecl = np.where(is_stage3, 1.0, pd_for_ecl)
        
        result['pd_for_ecl'] = pd_for_ecl
        
        # ECL = PD × LGD × EAD
        ecl = pd_for_ecl * lgd_adj * result['ead_computed']
        
        # Discount
        if discount_rate > 0:
            maturity_years = result['maturity_months'] / 12.0
            discount_factor = 1 / (1 + discount_rate) ** maturity_years
            ecl = ecl * discount_factor
        
        result['ecl'] = ecl.round(2)
        result['ecl_coverage'] = np.where(
            result['ead_computed'] > 0,
            result['ecl'] / result['ead_computed'],
            0.0
        )
        
        # Summary logging
        total_ead = result['ead_computed'].sum()
        total_ecl = result['ecl'].sum()
        
        logger.info(
            "ECL computed: %d contracts, total_EAD=%.0f, total_ECL=%.0f, "
            "coverage=%.4f%%, S1=%d, S2=%d, S3=%d",
            len(result), total_ead, total_ecl,
            (total_ecl / total_ead * 100) if total_ead > 0 else 0,
            (result['stage'] == 1).sum(),
            (result['stage'] == 2).sum(),
            (result['stage'] == 3).sum(),
        )
        
        return result
    
    def ecl_summary(self, ecl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ECL by stage.
        
        Parameters
        ----------
        ecl_df : pd.DataFrame
            Output of ``compute_ecl()``.
            
        Returns
        -------
        pd.DataFrame
            Summary by stage with count, EAD, ECL, coverage.
        """
        records = []
        for stage in [1, 2, 3]:
            mask = ecl_df['stage'] == stage
            subset = ecl_df[mask]
            records.append({
                'stage': f'Stage {stage}',
                'count': len(subset),
                'ead_total': round(subset['ead_computed'].sum(), 2),
                'ecl_total': round(subset['ecl'].sum(), 2),
                'pd_mean': round(subset['pd_adjusted'].mean(), 6) if len(subset) > 0 else 0,
                'lgd_mean': round(subset['lgd_adjusted'].mean(), 4) if len(subset) > 0 else 0,
                'coverage': round(
                    subset['ecl'].sum() / subset['ead_computed'].sum(), 6
                ) if subset['ead_computed'].sum() > 0 else 0,
            })
        
        total_ead = ecl_df['ead_computed'].sum()
        total_ecl = ecl_df['ecl'].sum()
        records.append({
            'stage': 'Total',
            'count': len(ecl_df),
            'ead_total': round(total_ead, 2),
            'ecl_total': round(total_ecl, 2),
            'pd_mean': round(ecl_df['pd_adjusted'].mean(), 6),
            'lgd_mean': round(ecl_df['lgd_adjusted'].mean(), 4),
            'coverage': round(total_ecl / total_ead, 6) if total_ead > 0 else 0,
        })
        
        return pd.DataFrame(records)
    
    def scenario_breakdown(
        self,
        ecl_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Show ECL under each macro scenario for audit/governance.
        
        Parameters
        ----------
        ecl_df : pd.DataFrame
            Output of ``compute_ecl()``.
            
        Returns
        -------
        pd.DataFrame
            ECL per scenario (before and after weighting).
        """
        return scenario_ecl_breakdown(
            ecl_df['pd_for_ecl'],
            ecl_df['lgd_value'],
            ecl_df['ead_computed'],
            self.scenarios
        )
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_scenarios = len(self.scenarios)
        return (
            f"ContractRiskEngine(product='{self.schema.name}', "
            f"type='{self.product_type}', "
            f"scenarios={n_scenarios}, {status})"
        )
