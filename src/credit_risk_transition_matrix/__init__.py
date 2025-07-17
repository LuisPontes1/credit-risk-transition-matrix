"""
Credit Risk Transition Matrix Library

A professional Python library for credit risk analysis using transition matrices.
Designed for data scientists, risk analysts, and financial institutions to analyze 
credit portfolio behavior, predict defaults, and generate risk reports.

Main Components:
- TransitionMatrixLearner: Core class for transition matrix analysis
- Visualization tools: Professional heatmaps and charts
- Risk metrics: PD calculations and portfolio analytics
- Model validation: Backtesting and performance metrics
"""

from .core.transition_matrix_learner import TransitionMatrixLearner
from .visualization.heatmaps import plot_global_heatmap

__version__ = "0.1.0"
__author__ = "Luis Pontes"
__email__ = "luis@example.com"

__all__ = [
    "TransitionMatrixLearner",
    "plot_global_heatmap",
]
