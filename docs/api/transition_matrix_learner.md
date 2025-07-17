# TransitionMatrixLearner API Reference

## Overview

The `TransitionMatrixLearner` is the core class for credit risk transition matrix analysis. It provides comprehensive functionality for calculating transition probabilities, generating visualizations, and performing risk analytics.

## Class Definition

```python
class TransitionMatrixLearner:
    def __init__(
        self,
        buckets: List[Union[int, float]] = None,
        time_horizon: int = 12,
        min_observations: int = 10
    )
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buckets` | List[Union[int, float]] | [0, 15, 30, 60, 90, 120, 180, 240, 360] | Risk bucket boundaries (days past due) |
| `time_horizon` | int | 12 | Time horizon for analysis (months) |
| `min_observations` | int | 10 | Minimum observations required per bucket |

## Methods

### fit()

Fit the transition matrix model on training data.

```python
def fit(
    self,
    data: pd.DataFrame,
    id_col: str = "client_id",
    time_col: str = "observation_date",
    bucket_col: str = "bucket",
    segment_col: str = None,
    weight_col: str = None
) -> 'TransitionMatrixLearner'
```

**Parameters:**
- `data`: Input DataFrame with panel data
- `id_col`: Column name for client/contract identifier
- `time_col`: Column name for observation date
- `bucket_col`: Column name for risk bucket values
- `segment_col`: Optional column for portfolio segmentation
- `weight_col`: Optional column for exposure weighting

**Returns:** Self (for method chaining)

**Example:**
```python
learner.fit(
    data=df_panel,
    id_col="contract_id",
    time_col="reference_date",
    bucket_col="days_past_due",
    segment_col="product_type"
)
```

### transform()

Generate transition matrices after fitting.

```python
def transform(
    self,
    modes: List[str] = ['global']
) -> Dict[str, pd.DataFrame]
```

**Parameters:**
- `modes`: List of matrix types to generate ['global', 'segmented']

**Returns:** Dictionary with transition matrices

**Example:**
```python
matrices = learner.transform(modes=['global', 'segmented'])
global_matrix = matrices['global']
```

### fit_transform()

Convenience method combining fit and transform.

```python
def fit_transform(
    self,
    data: pd.DataFrame,
    **fit_params
) -> Dict[str, pd.DataFrame]
```

### predict_transitions()

Predict future state distributions based on current states.

```python
def predict_transitions(
    self,
    current_data: Union[pd.DataFrame, str],
    periods: int = 1
) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `current_data`: Current portfolio state or DataFrame
- `periods`: Number of periods to predict forward

**Returns:** Dictionary with prediction probabilities

**Example:**
```python
# Predict from single bucket
predictions = learner.predict_transitions("30-59", periods=3)

# Predict from DataFrame
predictions = learner.predict_transitions(current_portfolio, periods=6)
```

### calculate_pd()

Calculate Probability of Default (PD) metrics.

```python
def calculate_pd(
    self,
    time_horizon: int = None,
    default_buckets: List[str] = None
) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `time_horizon`: Horizon for PD calculation (uses instance default if None)
- `default_buckets`: Buckets considered as default states

**Returns:** Dictionary with PD metrics by bucket

**Example:**
```python
pd_metrics = learner.calculate_pd(
    time_horizon=12,
    default_buckets=["240-359", "360+"]
)
```

### plot_heatmaps()

Generate professional heatmap visualizations.

```python
def plot_heatmaps(
    self,
    modes: List[str] = ["global"],
    save_dir: str = None,
    file_format: str = "png",
    dpi: int = 300
) -> List[plt.Figure]
```

**Parameters:**
- `modes`: Types of heatmaps to generate
- `save_dir`: Directory to save plots (optional)
- `file_format`: Output format ('png', 'pdf', 'svg')
- `dpi`: Resolution for raster formats

**Returns:** List of matplotlib figures

### validate_model()

Validate model performance on test data.

```python
def validate_model(
    self,
    test_data: pd.DataFrame,
    metrics: List[str] = ['accuracy']
) -> Dict[str, float]
```

**Parameters:**
- `test_data`: Test dataset with same structure as training
- `metrics`: List of metrics to calculate ['accuracy', 'stability', 'coverage']

**Returns:** Dictionary with validation results

**Available Metrics:**
- `accuracy`: Prediction accuracy for transitions
- `stability`: Matrix stability (diagonal dominance)
- `coverage`: Data coverage across buckets

## Properties

### Fitted Properties

These properties are available after calling `fit()`:

| Property | Type | Description |
|----------|------|-------------|
| `global_matrix_` | pd.DataFrame | Global transition matrix |
| `segmented_matrices_` | Dict[str, pd.DataFrame] | Segmented matrices by group |
| `bucket_labels_` | List[str] | Formatted bucket labels |
| `buckets_` | List[Union[int, float]] | Bucket boundaries used |
| `is_fitted_` | bool | Whether model has been fitted |

### Example Property Usage

```python
# Check if model is fitted
if learner.is_fitted_:
    # Access global matrix
    global_matrix = learner.global_matrix_
    
    # Get bucket labels
    buckets = learner.bucket_labels_
    
    # Access segmented matrices
    if learner.segmented_matrices_:
        retail_matrix = learner.segmented_matrices_['Retail']
```

## Error Handling

The class provides comprehensive error handling:

```python
# Common exceptions
ValueError: "Model must be fitted before prediction"
ValueError: "Data validation failed: missing required columns"
ValueError: "Insufficient data for matrix calculation"
```

## Thread Safety

The `TransitionMatrixLearner` is not thread-safe. For concurrent usage, create separate instances for each thread.

## Performance Considerations

- Memory usage scales with number of unique clients and time periods
- Matrix calculation complexity: O(n * t) where n=clients, t=time periods
- Visualization generation is memory-intensive for large matrices

## Best Practices

1. **Data Preparation**: Ensure consistent date formatting and complete time series
2. **Bucket Definition**: Choose buckets based on business requirements and data distribution
3. **Validation**: Always validate on out-of-time data
4. **Memory Management**: For large datasets, consider chunked processing

## Examples

### Complete Workflow Example

```python
from credit_risk_transition_matrix import TransitionMatrixLearner
import pandas as pd

# Initialize learner
learner = TransitionMatrixLearner(
    buckets=[0, 15, 30, 60, 90, 120, 180, 240, 360],
    time_horizon=12
)

# Fit and generate matrices
matrices = learner.fit_transform(
    data=df_training,
    id_col="contract_id",
    time_col="reference_date", 
    bucket_col="days_past_due",
    segment_col="product_type"
)

# Generate visualizations
figures = learner.plot_heatmaps(
    modes=["global", "segmented"],
    save_dir="./analysis/matrices/",
    file_format="png"
)

# Calculate risk metrics
pd_results = learner.calculate_pd(
    time_horizon=12,
    default_buckets=["240-359", "360+"]
)

# Validate performance
validation = learner.validate_model(
    test_data=df_test,
    metrics=['accuracy', 'stability', 'coverage']
)

print(f"Model accuracy: {validation['accuracy']:.2%}")
```

## See Also

- [Visualization Functions](visualization.md)
- [Risk Metrics](risk_metrics.md)
- [Getting Started Tutorial](../tutorials/getting_started.ipynb)
