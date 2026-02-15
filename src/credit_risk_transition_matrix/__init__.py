"""
Credit Risk Transition Matrix Library — CPC 4.966 / IFRS 9 Compliant

A professional Python library for contract-level credit risk analysis.
Designed for data scientists, risk analysts, and financial institutions.

Main Components:
- ContractRiskEngine: Full ECL pipeline (PD + LGD + EAD + SICR → ECL)
- LGDRatingEngine: Product-specific LGD rating per contract
- EAD Engine: Exposure at Default with CCF for revolving
- SICR Staging: PD-based stage classification per Resolução 4.966
- Forward-Looking: Macro scenario overlay (optimistic/base/pessimistic)
- Maturity Engine: Contractual and behavioral maturity
- Backtesting: PD/LGD/ECL validation with discrimination metrics
- Product Schema: Product configurations (revolving, giro, collateral)
- TransitionMatrixLearner: Transition matrix analysis
- PD Engine: Marginal PD, lifetime PD, cure rates
- Stress Testing: Scenario generation and matrix stress
- ECL Calculator: Expected Credit Loss computation
- Visualization: Matplotlib heatmaps and Plotly charts
"""

from .core.transition_matrix_learner import TransitionMatrixLearner
from .core.pd_engine import (
    cumulative_pd,
    marginal_pd,
    pd_term_structure,
    survival_curve,
    cure_rates,
    absorbing_analysis,
    weighted_matrix
)
from .core.stress_testing import (
    apply_scalar_stress,
    apply_z_shift,
    generate_scenarios,
    weighted_scenario_matrix
)
from .core.staging import (
    classify_stage,
    classify_stage_sicr,
    stage_migration_matrix,
    stage_summary
)
from .core.ecl_calculator import (
    calculate_ecl,
    ecl_by_stage,
    portfolio_ecl
)
from .core.lgd_engine import (
    lgd_from_cure_rates,
    lgd_workout,
    lgd_by_segment,
    LGDRatingEngine
)
from .core.product_schema import (
    ProductSchema,
    CollateralConfig,
    REVOLVING,
    GIRO_TPV,
    GIRO_TRADITIONAL
)
from .core.ead_engine import (
    estimate_ccf,
    ccf_by_segment,
    compute_ead
)
from .core.forward_looking import (
    MacroScenario,
    DEFAULT_SCENARIOS,
    apply_macro_overlay,
    scenario_ecl_breakdown
)
from .core.maturity_engine import (
    contractual_maturity,
    behavioral_maturity,
    effective_maturity
)
from .core.backtesting import (
    backtest_pd,
    backtest_lgd,
    backtest_ecl,
    discrimination_metrics
)
from .core.contract_risk_engine import ContractRiskEngine
from .visualization.heatmaps import plot_global_heatmap

__version__ = "1.0.0"
__author__ = "Luis Pontes"

__all__ = [
    "TransitionMatrixLearner",
    "cumulative_pd",
    "marginal_pd",
    "pd_term_structure",
    "survival_curve",
    "cure_rates",
    "absorbing_analysis",
    "weighted_matrix",
    "apply_scalar_stress",
    "apply_z_shift",
    "generate_scenarios",
    "weighted_scenario_matrix",
    "classify_stage",
    "classify_stage_sicr",
    "stage_migration_matrix",
    "stage_summary",
    "calculate_ecl",
    "ecl_by_stage",
    "portfolio_ecl",
    "lgd_from_cure_rates",
    "lgd_workout",
    "lgd_by_segment",
    "LGDRatingEngine",
    "ProductSchema",
    "CollateralConfig",
    "REVOLVING",
    "GIRO_TPV",
    "GIRO_TRADITIONAL",
    "estimate_ccf",
    "ccf_by_segment",
    "compute_ead",
    "MacroScenario",
    "DEFAULT_SCENARIOS",
    "apply_macro_overlay",
    "scenario_ecl_breakdown",
    "contractual_maturity",
    "behavioral_maturity",
    "effective_maturity",
    "backtest_pd",
    "backtest_lgd",
    "backtest_ecl",
    "discrimination_metrics",
    "ContractRiskEngine",
    "plot_global_heatmap",
]
