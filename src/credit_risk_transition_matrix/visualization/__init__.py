"""Visualization module for transition matrix heatmaps and charts."""

from .heatmaps import plot_global_heatmap, plot_segmented_heatmaps

# Plotly charts are importable but not loaded here to avoid hard dependency.
# Use: from credit_risk_transition_matrix.visualization.plotly_charts import plot_heatmap_plotly

__all__ = ["plot_global_heatmap", "plot_segmented_heatmaps"]
