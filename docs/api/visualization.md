# Visualization Functions API Reference

## Overview

The visualization module provides professional-grade plotting capabilities for transition matrices and risk analytics. All functions generate publication-ready charts with Brazilian Portuguese labels.

## Core Functions

### plot_global_heatmap()

Generate a global transition matrix heatmap.

```python
def plot_global_heatmap(
    transition_matrix: pd.DataFrame,
    title: str = "Matriz de Transição Global",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdYlBu_r',
    save_path: str = None
) -> plt.Figure
```

**Parameters:**
- `transition_matrix`: Transition matrix DataFrame
- `title`: Plot title in Portuguese
- `figsize`: Figure size (width, height)
- `cmap`: Colormap for heatmap
- `save_path`: Path to save figure (optional)

**Returns:** Matplotlib figure object

**Example:**
```python
from credit_risk_transition_matrix.visualization import plot_global_heatmap

fig = plot_global_heatmap(
    transition_matrix=global_matrix,
    title="Matriz de Transição - Portfolio Total",
    figsize=(14, 12),
    save_path="./outputs/global_heatmap.png"
)
```

### plot_segmented_heatmaps()

Generate multiple heatmaps for different portfolio segments.

```python
def plot_segmented_heatmaps(
    segmented_matrices: Dict[str, pd.DataFrame],
    title_prefix: str = "Matriz de Transição",
    figsize: Tuple[int, int] = (15, 12),
    save_dir: str = None
) -> List[plt.Figure]
```

**Parameters:**
- `segmented_matrices`: Dictionary with segment names and matrices
- `title_prefix`: Prefix for plot titles
- `figsize`: Figure size for each subplot
- `save_dir`: Directory to save figures

**Returns:** List of matplotlib figure objects

**Example:**
```python
figures = plot_segmented_heatmaps(
    segmented_matrices=segment_matrices,
    title_prefix="Análise por Segmento",
    save_dir="./outputs/segments/"
)
```

### plot_comparison_heatmaps()

Generate side-by-side comparison of two transition matrices.

```python
def plot_comparison_heatmaps(
    matrix1: pd.DataFrame,
    matrix2: pd.DataFrame,
    labels: List[str] = ["Período 1", "Período 2"],
    figsize: Tuple[int, int] = (20, 8),
    save_path: str = None
) -> plt.Figure
```

**Parameters:**
- `matrix1`: First transition matrix
- `matrix2`: Second transition matrix  
- `labels`: Labels for comparison
- `figsize`: Figure size
- `save_path`: Path to save figure

**Returns:** Matplotlib figure object

**Example:**
```python
fig = plot_comparison_heatmaps(
    matrix1=pre_covid_matrix,
    matrix2=post_covid_matrix,
    labels=["Pré-COVID", "Pós-COVID"],
    save_path="./analysis/covid_comparison.png"
)
```

### plot_flow_diagram()

Generate a flow diagram showing transition patterns.

```python
def plot_flow_diagram(
    transition_matrix: pd.DataFrame,
    threshold: float = 0.05,
    figsize: Tuple[int, int] = (12, 10),
    save_path: str = None
) -> plt.Figure
```

**Parameters:**
- `transition_matrix`: Transition matrix DataFrame
- `threshold`: Minimum probability to show transitions
- `figsize`: Figure size
- `save_path`: Path to save figure

**Returns:** Matplotlib figure object

**Example:**
```python
fig = plot_flow_diagram(
    transition_matrix=matrix,
    threshold=0.1,  # Show only transitions > 10%
    save_path="./outputs/flow_diagram.png"
)
```

## Customization Options

### Color Schemes

Available color maps for heatmaps:

```python
# Professional color schemes
COLORMAPS = {
    'default': 'RdYlBu_r',      # Red-Yellow-Blue (reversed)
    'risk': 'Reds',             # Red gradient for risk
    'green': 'Greens',          # Green gradient  
    'blue': 'Blues',            # Blue gradient
    'viridis': 'viridis',       # Perceptually uniform
    'plasma': 'plasma'          # High contrast
}
```

### Style Configuration

```python
# Default style parameters
STYLE_CONFIG = {
    'font_family': 'Arial',
    'title_size': 16,
    'label_size': 12,
    'tick_size': 10,
    'annotation_size': 9,
    'dpi': 300
}
```

### Brazilian Portuguese Labels

All plots include proper Portuguese labels:

```python
PORTUGUESE_LABELS = {
    'title': 'Matriz de Transição',
    'xlabel': 'Bucket Destino',
    'ylabel': 'Bucket Origem', 
    'colorbar': 'Probabilidade de Transição',
    'legend': 'Legenda',
    'total': 'Total',
    'percentage': 'Percentual'
}
```

## Advanced Visualizations

### Time Series Plot

```python
def plot_pd_time_series(
    pd_data: Dict[str, List[float]],
    dates: List[str],
    title: str = "Evolução da PD por Bucket",
    save_path: str = None
) -> plt.Figure
```

### Risk Distribution

```python
def plot_risk_distribution(
    portfolio_data: pd.DataFrame,
    bucket_col: str = "bucket",
    value_col: str = "exposure",
    title: str = "Distribuição de Risco",
    save_path: str = None
) -> plt.Figure
```

### Migration Trends

```python
def plot_migration_trends(
    matrices_by_period: Dict[str, pd.DataFrame],
    bucket_from: str,
    bucket_to: str,
    title: str = None,
    save_path: str = None
) -> plt.Figure
```

## Export Options

### Supported Formats

```python
EXPORT_FORMATS = {
    'png': {'dpi': 300, 'bbox_inches': 'tight'},
    'pdf': {'bbox_inches': 'tight', 'format': 'pdf'},
    'svg': {'bbox_inches': 'tight', 'format': 'svg'},
    'eps': {'bbox_inches': 'tight', 'format': 'eps'},
    'jpg': {'dpi': 300, 'bbox_inches': 'tight', 'quality': 95}
}
```

### High-Quality Export

```python
def save_figure(
    fig: plt.Figure,
    path: str,
    format: str = 'png',
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """Save figure with high quality settings."""
    fig.savefig(
        path,
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor='white',
        edgecolor='none'
    )
```

## Interactive Features

### Plotly Integration

For interactive visualizations (Phase 2):

```python
# Future functionality
def plot_interactive_heatmap(
    transition_matrix: pd.DataFrame,
    title: str = "Interactive Transition Matrix"
) -> plotly.graph_objects.Figure:
    """Generate interactive Plotly heatmap."""
    pass  # Implementation in Phase 2
```

## Accessibility Features

### Color Blind Support

```python
# Color blind friendly palettes
COLORBLIND_SAFE = {
    'viridis': 'viridis',
    'cividis': 'cividis', 
    'deuteranopia': ['#1f77b4', '#ff7f0e', '#2ca02c'],
    'protanopia': ['#1f77b4', '#ff7f0e', '#2ca02c']
}
```

### High Contrast Mode

```python
def apply_high_contrast(fig: plt.Figure) -> plt.Figure:
    """Apply high contrast styling for accessibility."""
    # Implementation details
    return fig
```

## Performance Optimization

### Large Matrix Handling

```python
def optimize_large_matrix_plot(
    matrix: pd.DataFrame,
    max_size: int = 20
) -> pd.DataFrame:
    """Optimize large matrices for visualization."""
    if matrix.shape[0] > max_size:
        # Aggregate smaller buckets
        return aggregate_buckets(matrix, target_size=max_size)
    return matrix
```

### Memory Efficient Plotting

```python
def plot_with_memory_optimization(
    matrix: pd.DataFrame,
    chunk_size: int = 1000
) -> plt.Figure:
    """Plot large matrices with memory optimization."""
    # Implementation for chunked processing
    pass
```

## Best Practices

### 1. Figure Sizing

```python
# Recommended figure sizes
FIGURE_SIZES = {
    'small': (8, 6),      # For single matrices
    'medium': (12, 10),   # For detailed analysis
    'large': (16, 12),    # For presentations
    'comparison': (20, 8), # For side-by-side
    'poster': (24, 18)    # For posters/reports
}
```

### 2. Color Selection

```python
# Choose colors based on use case
def select_colormap(use_case: str) -> str:
    """Select appropriate colormap for use case."""
    color_guide = {
        'risk_analysis': 'Reds',
        'general_purpose': 'RdYlBu_r', 
        'presentation': 'viridis',
        'print_friendly': 'Greys',
        'colorblind_safe': 'cividis'
    }
    return color_guide.get(use_case, 'RdYlBu_r')
```

### 3. Title and Labels

```python
# Professional title formatting
def format_title(base_title: str, context: Dict[str, str]) -> str:
    """Format professional titles with context."""
    return f"{base_title} - {context.get('period', '')} - {context.get('segment', 'Total')}"
```

## Error Handling

### Common Issues

```python
# Handle missing data
if transition_matrix.isnull().any().any():
    warnings.warn("Transition matrix contains NaN values")
    
# Handle empty matrices
if transition_matrix.empty:
    raise ValueError("Cannot plot empty transition matrix")
    
# Handle size limitations
if transition_matrix.shape[0] > 30:
    warnings.warn("Large matrix may be difficult to read")
```

## Examples

### Complete Visualization Workflow

```python
from credit_risk_transition_matrix.visualization import (
    plot_global_heatmap,
    plot_segmented_heatmaps,
    plot_comparison_heatmaps
)

# Generate all visualizations
def create_risk_report(learner, save_dir="./outputs/"):
    """Create complete risk visualization report."""
    
    # Global heatmap
    global_fig = plot_global_heatmap(
        learner.global_matrix_,
        title="Matriz de Transição Global - 2024",
        save_path=f"{save_dir}/global_matrix.png"
    )
    
    # Segmented analysis
    if learner.segmented_matrices_:
        segment_figs = plot_segmented_heatmaps(
            learner.segmented_matrices_,
            title_prefix="Análise por Segmento",
            save_dir=f"{save_dir}/segments/"
        )
    
    # Time comparison (if historical data available)
    if historical_matrix is not None:
        comparison_fig = plot_comparison_heatmaps(
            historical_matrix,
            learner.global_matrix_,
            labels=["2023", "2024"],
            save_path=f"{save_dir}/year_comparison.png"
        )
    
    return [global_fig] + segment_figs + [comparison_fig]
```

## See Also

- [TransitionMatrixLearner API](transition_matrix_learner.md)
- [Risk Metrics](risk_metrics.md)
- [Advanced Analysis Tutorial](../tutorials/advanced_analysis.ipynb)
