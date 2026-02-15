"""Core module for credit risk transition matrix analysis."""

from .transition_matrix_learner import TransitionMatrixLearner
from .pd_engine import (
    cumulative_pd,
    marginal_pd,
    pd_term_structure,
    survival_curve,
    cure_rates,
    absorbing_analysis,
    weighted_matrix
)
from .stress_testing import (
    apply_scalar_stress,
    apply_z_shift,
    generate_scenarios,
    weighted_scenario_matrix
)
from .staging import (
    classify_stage,
    stage_migration_matrix,
    stage_summary
)
from .ecl_calculator import (
    calculate_ecl,
    ecl_by_stage,
    portfolio_ecl
)

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
    "stage_migration_matrix",
    "stage_summary",
    "calculate_ecl",
    "ecl_by_stage",
    "portfolio_ecl",
]
