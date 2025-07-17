# Risk Metrics API Reference

## Overview

The risk metrics module provides comprehensive functions for calculating credit risk indicators, probability of default (PD), and portfolio analytics based on transition matrices.

## Core Functions

### calculate_pd()

Calculate Probability of Default for different time horizons.

```python
def calculate_pd(
    transition_matrix: pd.DataFrame,
    time_horizon: int = 12,
    default_buckets: List[str] = None
) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `transition_matrix`: Transition probability matrix
- `time_horizon`: Time horizon in months
- `default_buckets`: Buckets considered as default states

**Returns:** Dictionary with PD metrics by starting bucket

**Example:**
```python
pd_results = calculate_pd(
    transition_matrix=matrix,
    time_horizon=12,
    default_buckets=["240-359", "360+"]
)

# Access 1-year PD for bucket "30-59"
pd_12m = pd_results["30-59"]["pd_12m"]
```

### calculate_lifetime_pd()

Calculate lifetime probability of default.

```python
def calculate_lifetime_pd(
    transition_matrix: pd.DataFrame,
    default_buckets: List[str] = None,
    max_periods: int = 120
) -> Dict[str, float]
```

**Parameters:**
- `transition_matrix`: Transition probability matrix
- `default_buckets`: Buckets considered as default
- `max_periods`: Maximum periods to calculate (months)

**Returns:** Dictionary with lifetime PD by starting bucket

### calculate_migration_rates()

Calculate migration rates between bucket categories.

```python
def calculate_migration_rates(
    transition_matrix: pd.DataFrame,
    bucket_categories: Dict[str, List[str]] = None
) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `transition_matrix`: Transition probability matrix
- `bucket_categories`: Custom bucket groupings

**Returns:** Migration rates between categories

**Default Categories:**
```python
DEFAULT_CATEGORIES = {
    'performing': ['0-14', '15-29'],
    'watch': ['30-59', '60-89'],
    'substandard': ['90-119', '120-179'],
    'doubtful': ['180-239', '240-359'],
    'loss': ['360+']
}
```

### calculate_concentration_risk()

Calculate portfolio concentration metrics.

```python
def calculate_concentration_risk(
    portfolio_data: pd.DataFrame,
    bucket_col: str = "bucket",
    exposure_col: str = "exposure"
) -> Dict[str, float]
```

**Parameters:**
- `portfolio_data`: Portfolio exposure data
- `bucket_col`: Column name for risk buckets
- `exposure_col`: Column name for exposure amounts

**Returns:** Concentration risk metrics

**Metrics Returned:**
- `hhi_index`: Herfindahl-Hirschman Index
- `gini_coefficient`: Gini coefficient
- `top_3_concentration`: Top 3 buckets concentration
- `entropy`: Shannon entropy measure

### calculate_vintage_analysis()

Perform vintage analysis on portfolio data.

```python
def calculate_vintage_analysis(
    portfolio_data: pd.DataFrame,
    vintage_col: str = "origination_date",
    performance_col: str = "bucket",
    periods: int = 24
) -> pd.DataFrame
```

**Parameters:**
- `portfolio_data`: Portfolio data with vintages
- `vintage_col`: Column with origination dates
- `performance_col`: Column with performance buckets
- `periods`: Number of periods to analyze

**Returns:** DataFrame with vintage performance curves

## Advanced Analytics

### Expected Credit Loss (ECL)

Calculate IFRS 9 Expected Credit Loss.

```python
def calculate_ecl(
    portfolio_data: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    lgd_by_bucket: Dict[str, float],
    ead_col: str = "exposure",
    time_horizon: int = 12
) -> Dict[str, float]
```

**Parameters:**
- `portfolio_data`: Current portfolio positions
- `transition_matrix`: Transition probability matrix
- `lgd_by_bucket`: Loss Given Default by bucket
- `ead_col`: Exposure at Default column
- `time_horizon`: ECL calculation horizon

**Returns:** ECL metrics by bucket and total

### Stress Testing

Perform stress testing on transition matrices.

```python
def apply_stress_scenario(
    baseline_matrix: pd.DataFrame,
    stress_factors: Dict[str, float],
    scenario_name: str = "Stress"
) -> pd.DataFrame
```

**Parameters:**
- `baseline_matrix`: Baseline transition matrix
- `stress_factors`: Stress multipliers by transition type
- `scenario_name`: Name for stress scenario

**Returns:** Stressed transition matrix

**Example Stress Factors:**
```python
stress_factors = {
    'deterioration': 1.5,  # 50% increase in downgrades
    'improvement': 0.7,    # 30% decrease in upgrades
    'default': 2.0         # 100% increase in defaults
}
```

### Model Validation Metrics

Calculate model validation statistics.

```python
def calculate_validation_metrics(
    actual_transitions: pd.DataFrame,
    predicted_matrix: pd.DataFrame
) -> Dict[str, float]
```

**Parameters:**
- `actual_transitions`: Observed transition counts
- `predicted_matrix`: Predicted transition probabilities

**Returns:** Validation metrics

**Metrics Included:**
- `chi_square_stat`: Chi-square test statistic
- `p_value`: Statistical significance
- `cramers_v`: Cramér's V association measure
- `prediction_accuracy`: Overall prediction accuracy

## Statistical Tests

### Stability Testing

Test transition matrix stability over time.

```python
def test_matrix_stability(
    matrix_t1: pd.DataFrame,
    matrix_t2: pd.DataFrame,
    significance_level: float = 0.05
) -> Dict[str, Any]
```

**Parameters:**
- `matrix_t1`: Transition matrix period 1
- `matrix_t2`: Transition matrix period 2  
- `significance_level`: Statistical significance level

**Returns:** Stability test results

### Homogeneity Testing

Test homogeneity across portfolio segments.

```python
def test_segment_homogeneity(
    segmented_matrices: Dict[str, pd.DataFrame],
    significance_level: float = 0.05
) -> Dict[str, Any]
```

**Parameters:**
- `segmented_matrices`: Matrices by segment
- `significance_level`: Statistical significance level

**Returns:** Homogeneity test results

## Performance Metrics

### Model Performance

Calculate predictive model performance.

```python
def calculate_model_performance(
    learner: TransitionMatrixLearner,
    test_data: pd.DataFrame,
    metrics: List[str] = ['auc', 'accuracy', 'precision', 'recall']
) -> Dict[str, float]
```

**Parameters:**
- `learner`: Fitted transition matrix learner
- `test_data`: Out-of-sample test data
- `metrics`: Performance metrics to calculate

**Returns:** Model performance metrics

### Backtesting

Perform backtesting analysis.

```python
def perform_backtesting(
    learner: TransitionMatrixLearner,
    historical_data: pd.DataFrame,
    backtest_periods: int = 12
) -> Dict[str, Any]
```

**Parameters:**
- `learner`: Fitted model
- `historical_data`: Historical performance data
- `backtest_periods`: Number of periods for backtesting

**Returns:** Backtesting results and statistics

## Risk-Adjusted Metrics

### Portfolio Risk Metrics

Calculate comprehensive portfolio risk indicators.

```python
def calculate_portfolio_metrics(
    transition_matrix: pd.DataFrame,
    exposure_data: pd.DataFrame,
    risk_weights: Dict[str, float] = None
) -> Dict[str, float]
```

**Parameters:**
- `transition_matrix`: Transition probability matrix
- `exposure_data`: Portfolio exposure by bucket
- `risk_weights`: Basel risk weights by bucket

**Returns:** Portfolio risk metrics

**Metrics Calculated:**
- `expected_loss`: Portfolio expected loss
- `unexpected_loss`: Portfolio unexpected loss
- `risk_weighted_assets`: Basel RWA calculation
- `capital_requirement`: Regulatory capital
- `raroc`: Risk-Adjusted Return on Capital

### Economic Capital

Calculate economic capital requirements.

```python
def calculate_economic_capital(
    transition_matrix: pd.DataFrame,
    portfolio_data: pd.DataFrame,
    confidence_level: float = 0.999,
    correlations: pd.DataFrame = None
) -> Dict[str, float]
```

**Parameters:**
- `transition_matrix`: Transition probabilities
- `portfolio_data`: Portfolio exposures
- `confidence_level`: Confidence level for VaR
- `correlations`: Asset correlation matrix

**Returns:** Economic capital by bucket and total

## Regulatory Compliance

### IFRS 9 Calculations

Calculate IFRS 9 staging and provisions.

```python
def calculate_ifrs9_staging(
    portfolio_data: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    staging_rules: Dict[str, Any] = None
) -> pd.DataFrame
```

**Parameters:**
- `portfolio_data`: Current portfolio
- `transition_matrix`: Transition probabilities
- `staging_rules`: Custom staging criteria

**Returns:** Portfolio with IFRS 9 stages

### CECL Calculations

Calculate CECL lifetime expected losses.

```python
def calculate_cecl_provisions(
    portfolio_data: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    economic_scenarios: List[Dict[str, float]] = None
) -> pd.DataFrame
```

**Parameters:**
- `portfolio_data`: Current portfolio
- `transition_matrix`: Base transition matrix
- `economic_scenarios`: Economic scenario adjustments

**Returns:** CECL provision calculations

## Utility Functions

### Matrix Operations

Common matrix operations for risk calculations.

```python
def normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Normalize matrix rows to sum to 1."""
    
def validate_matrix_properties(matrix: pd.DataFrame) -> Dict[str, bool]:
    """Validate transition matrix properties."""
    
def aggregate_buckets(
    matrix: pd.DataFrame,
    bucket_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Aggregate buckets according to mapping."""
```

### Data Validation

Validate input data for risk calculations.

```python
def validate_risk_data(
    data: pd.DataFrame,
    required_columns: List[str],
    numeric_columns: List[str] = None
) -> bool:
    """Validate data structure for risk calculations."""
```

## Examples

### Complete Risk Analysis

```python
from credit_risk_transition_matrix.risk_metrics import *

def perform_complete_risk_analysis(learner, portfolio_data):
    """Perform comprehensive risk analysis."""
    
    results = {}
    
    # Basic PD calculations
    results['pd_metrics'] = calculate_pd(
        learner.global_matrix_,
        time_horizon=12,
        default_buckets=["240-359", "360+"]
    )
    
    # Migration analysis
    results['migration_rates'] = calculate_migration_rates(
        learner.global_matrix_
    )
    
    # Concentration risk
    results['concentration'] = calculate_concentration_risk(
        portfolio_data,
        bucket_col="bucket",
        exposure_col="exposure"
    )
    
    # IFRS 9 provisions
    results['ifrs9'] = calculate_ifrs9_staging(
        portfolio_data,
        learner.global_matrix_
    )
    
    # Economic capital
    results['economic_capital'] = calculate_economic_capital(
        learner.global_matrix_,
        portfolio_data,
        confidence_level=0.999
    )
    
    return results
```

### Stress Testing Example

```python
def run_stress_tests(baseline_matrix, portfolio_data):
    """Run comprehensive stress testing."""
    
    stress_scenarios = {
        'mild_stress': {'deterioration': 1.2, 'improvement': 0.9},
        'moderate_stress': {'deterioration': 1.5, 'improvement': 0.7},
        'severe_stress': {'deterioration': 2.0, 'improvement': 0.5}
    }
    
    results = {}
    
    for scenario_name, stress_factors in stress_scenarios.items():
        # Apply stress
        stressed_matrix = apply_stress_scenario(
            baseline_matrix,
            stress_factors,
            scenario_name
        )
        
        # Calculate stressed metrics
        results[scenario_name] = {
            'pd_impact': calculate_pd(stressed_matrix),
            'ecl_impact': calculate_ecl(portfolio_data, stressed_matrix),
            'capital_impact': calculate_economic_capital(stressed_matrix, portfolio_data)
        }
    
    return results
```

## Best Practices

### 1. Data Quality
- Ensure complete time series data
- Validate bucket assignments
- Check for outliers and data quality issues

### 2. Model Validation
- Use out-of-time testing
- Perform stability testing
- Document model assumptions

### 3. Regulatory Compliance
- Follow local regulatory guidelines
- Document methodology and assumptions
- Maintain audit trails

## See Also

- [TransitionMatrixLearner API](transition_matrix_learner.md)
- [Visualization Functions](visualization.md)
- [Mathematical Background](../theory/pd_calculations.md)
