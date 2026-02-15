"""
Plotly-based visualizations for transition matrices.

Alternative to matplotlib heatmaps — designed for Databricks notebooks 
where Plotly renders natively with interactivity.
"""

from typing import Optional, Dict, Any
import pandas as pd


def plot_heatmap_plotly(
    transition_matrix: pd.DataFrame,
    title: str = "Transition Matrix",
    colorscale: str = "Blues",
    show_text: bool = True,
    height: int = 600,
    width: int = 700
) -> Any:
    """
    Create an interactive Plotly heatmap for a transition matrix.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix
    title : str, default="Transition Matrix"
        Title of the plot
    colorscale : str, default="Blues"
        Plotly colorscale name
    show_text : bool, default=True
        Whether to annotate cells with values
    height : int, default=600
        Figure height in pixels
    width : int, default=700
        Figure width in pixels
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure
        
    Raises
    ------
    ImportError
        If plotly is not installed
    """
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for plot_heatmap_plotly(). "
            "Install with: pip install plotly"
        )
    
    labels = list(transition_matrix.index)
    values = transition_matrix.values
    
    # Format text annotations
    text = [[f"{v:.2%}" for v in row] for row in values]
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=labels,
        y=labels,
        colorscale=colorscale,
        text=text if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont={"size": 11},
        hovertemplate=(
            "From: %{y}<br>"
            "To: %{x}<br>"
            "Probability: %{z:.4f}<extra></extra>"
        ),
        zmin=0,
        zmax=1,
        colorbar=dict(title="Prob", tickformat=".0%")
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="To Bucket",
        yaxis_title="From Bucket",
        height=height,
        width=width,
        yaxis_autorange="reversed"
    )
    
    return fig


def plot_pd_curve_plotly(
    pd_data: pd.DataFrame,
    buckets: Optional[list] = None,
    title: str = "PD Term Structure",
    height: int = 500,
    width: int = 800
) -> Any:
    """
    Plot PD term structure curves using Plotly.
    
    Parameters
    ----------
    pd_data : pd.DataFrame
        Output of cumulative_pd() — index=period, columns=buckets.
    buckets : list, optional
        Specific buckets to plot. If None, plots all.
    title : str, default="PD Term Structure"
        Title of the plot.
    height : int, default=500
        Figure height in pixels.
    width : int, default=800
        Figure width in pixels.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive line chart.
    """
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for plot_pd_curve_plotly(). "
            "Install with: pip install plotly"
        )
    
    if buckets is None:
        buckets = list(pd_data.columns)
    
    fig = go.Figure()
    
    for bucket in buckets:
        if bucket in pd_data.columns:
            fig.add_trace(go.Scatter(
                x=list(pd_data.index),
                y=pd_data[bucket].values,
                mode='lines+markers',
                name=bucket,
                hovertemplate=(
                    f"Bucket: {bucket}<br>"
                    "Period: %{x}<br>"
                    "PD: %{y:.4f}<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Period",
        yaxis_title="Cumulative PD",
        yaxis_tickformat=".1%",
        height=height,
        width=width,
        legend=dict(title="Starting Bucket")
    )
    
    return fig
