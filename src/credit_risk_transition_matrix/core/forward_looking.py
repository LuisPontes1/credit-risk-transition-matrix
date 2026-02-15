"""
Forward-Looking Macro Overlay — CPC 4.966 / IFRS 9 compliant.

Adjusts PD and LGD using probability-weighted macroeconomic scenarios.
The regulation requires multiple forward-looking scenarios (at minimum:
optimistic, base, pessimistic) to be applied to ECL calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MacroScenario:
    """
    A macroeconomic scenario with probability weight and risk adjustments.
    
    Parameters
    ----------
    name : str
        Scenario name (e.g., 'base', 'otimista', 'pessimista').
    weight : float
        Probability weight (all scenario weights must sum to 1.0).
    pd_adjustment : float
        PD multiplier (1.0 = neutral, >1 = worse, <1 = better).
    lgd_adjustment : float
        LGD multiplier (1.0 = neutral, >1 = worse, <1 = better).
    description : str
        Human-readable scenario narrative.
    """
    name: str
    weight: float
    pd_adjustment: float = 1.0
    lgd_adjustment: float = 1.0
    description: str = ""
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        if self.pd_adjustment < 0:
            raise ValueError(f"pd_adjustment must be >= 0, got {self.pd_adjustment}")
        if self.lgd_adjustment < 0:
            raise ValueError(f"lgd_adjustment must be >= 0, got {self.lgd_adjustment}")


# ── Default scenarios (BCB-aligned) ──
DEFAULT_SCENARIOS = [
    MacroScenario(
        name='otimista',
        weight=0.25,
        pd_adjustment=0.80,
        lgd_adjustment=0.90,
        description='Cenário otimista: crescimento PIB acima do esperado, '
                    'SELIC em queda, desemprego baixo'
    ),
    MacroScenario(
        name='base',
        weight=0.50,
        pd_adjustment=1.00,
        lgd_adjustment=1.00,
        description='Cenário base: projeções do Focus/BCB, '
                    'condições macroeconômicas estáveis'
    ),
    MacroScenario(
        name='pessimista',
        weight=0.25,
        pd_adjustment=1.40,
        lgd_adjustment=1.20,
        description='Cenário pessimista: recessão, SELIC alta, '
                    'deterioração do mercado de trabalho'
    ),
]


def validate_scenarios(scenarios: List[MacroScenario]) -> None:
    """
    Validate that scenarios are internally consistent.
    
    Raises
    ------
    ValueError
        If weights don't sum to ~1.0 or list is empty.
    """
    if not scenarios:
        raise ValueError("At least one scenario is required")
    
    total_weight = sum(s.weight for s in scenarios)
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError(
            f"Scenario weights must sum to 1.0, got {total_weight:.6f}"
        )


def apply_macro_overlay(
    pd_values: pd.Series,
    lgd_values: Union[pd.Series, float],
    scenarios: Optional[List[MacroScenario]] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply probability-weighted macro scenario overlay to PD and LGD.
    
    Computes::
    
        PD_adjusted  = Σ (weight_i × min(PD × pd_adj_i, 1.0))
        LGD_adjusted = Σ (weight_i × min(LGD × lgd_adj_i, 1.0))
    
    Parameters
    ----------
    pd_values : pd.Series
        Per-contract PD values.
    lgd_values : pd.Series or float
        Per-contract or fixed LGD values.
    scenarios : List[MacroScenario], optional
        Scenarios to apply. Defaults to ``DEFAULT_SCENARIOS``.
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (pd_adjusted, lgd_adjusted) — probability-weighted values.
        
    Examples
    --------
    >>> pd_adj, lgd_adj = apply_macro_overlay(df['pd'], df['lgd'])
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    
    validate_scenarios(scenarios)
    
    pd_values = pd_values.astype(float)
    if isinstance(lgd_values, (int, float)):
        lgd_values = pd.Series(lgd_values, index=pd_values.index)
    else:
        lgd_values = lgd_values.astype(float)
    
    pd_adjusted = pd.Series(0.0, index=pd_values.index)
    lgd_adjusted = pd.Series(0.0, index=lgd_values.index)
    
    for scenario in scenarios:
        pd_scenario = (pd_values * scenario.pd_adjustment).clip(upper=1.0)
        lgd_scenario = (lgd_values * scenario.lgd_adjustment).clip(upper=1.0)
        
        pd_adjusted += scenario.weight * pd_scenario
        lgd_adjusted += scenario.weight * lgd_scenario
    
    logger.info(
        "Macro overlay applied (%d scenarios): "
        "PD mean %.4f → %.4f, LGD mean %.4f → %.4f",
        len(scenarios),
        pd_values.mean(), pd_adjusted.mean(),
        lgd_values.mean(), lgd_adjusted.mean()
    )
    
    return pd_adjusted, lgd_adjusted


def scenario_ecl_breakdown(
    pd_values: pd.Series,
    lgd_values: Union[pd.Series, float],
    ead_values: pd.Series,
    scenarios: Optional[List[MacroScenario]] = None,
) -> pd.DataFrame:
    """
    Show ECL under each scenario (before weighting) for transparency.
    
    Parameters
    ----------
    pd_values, lgd_values, ead_values : pd.Series
        Per-contract risk parameters.
    scenarios : List[MacroScenario], optional
        Defaults to ``DEFAULT_SCENARIOS``.
        
    Returns
    -------
    pd.DataFrame
        One row per scenario with total ECL, weight, and weighted ECL.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    
    validate_scenarios(scenarios)
    
    if isinstance(lgd_values, (int, float)):
        lgd_values = pd.Series(lgd_values, index=pd_values.index)
    
    records = []
    for s in scenarios:
        pd_s = (pd_values * s.pd_adjustment).clip(upper=1.0)
        lgd_s = (lgd_values * s.lgd_adjustment).clip(upper=1.0)
        ecl_s = (pd_s * lgd_s * ead_values).sum()
        
        records.append({
            'scenario': s.name,
            'weight': s.weight,
            'pd_adjustment': s.pd_adjustment,
            'lgd_adjustment': s.lgd_adjustment,
            'ecl_scenario': round(ecl_s, 2),
            'ecl_weighted': round(ecl_s * s.weight, 2),
        })
    
    result = pd.DataFrame(records)
    result.loc[len(result)] = {
        'scenario': 'TOTAL_WEIGHTED',
        'weight': 1.0,
        'pd_adjustment': np.nan,
        'lgd_adjustment': np.nan,
        'ecl_scenario': np.nan,
        'ecl_weighted': result['ecl_weighted'].sum(),
    }
    
    return result
