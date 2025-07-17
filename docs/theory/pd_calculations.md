# PD Calculation Methods

## Overview

Probability of Default (PD) is a fundamental risk metric representing the likelihood that a borrower will default within a specific time horizon. This document covers the mathematical foundation and implementation methods for PD calculations using transition matrices.

## Mathematical Foundation

### Basic Definition

For a given time horizon h, the PD from bucket i is:

```
PD[i](h) = P(Default by time h | Current state = i)
```

### Absorbing State Model

In transition matrix notation with absorbing default states:

```
PD[i](h) = ∑ P⁽ʰ⁾[i,d]
           d∈D
```

Where:
- P⁽ʰ⁾ = h-period transition matrix
- D = Set of default buckets
- P⁽ʰ⁾[i,d] = Probability of transitioning from bucket i to default bucket d in h periods

## Calculation Methods

### 1. Matrix Power Method

For h-period PD calculation:

```python
def calculate_pd_matrix_power(P, h, default_buckets):
    """
    Calculate PD using matrix power method.
    
    P: Transition matrix
    h: Time horizon
    default_buckets: List of default bucket indices
    """
    P_h = np.linalg.matrix_power(P, h)
    pd_vector = P_h[:, default_buckets].sum(axis=1)
    return pd_vector
```

#### Advantages:
- Exact calculation for discrete time
- Handles multiple default states
- Computationally efficient for small matrices

#### Limitations:
- Requires matrix exponentiation
- Memory intensive for large matrices
- Assumes constant transition probabilities

### 2. Fundamental Matrix Method

For absorbing Markov chains, define:
- **Q**: Transient submatrix (non-default to non-default)
- **R**: Absorption probability matrix (non-default to default)

The fundamental matrix:
```
N = (I - Q)⁻¹
```

Absorption probabilities:
```
B = N × R
```

PD calculation:
```python
def calculate_pd_fundamental_matrix(Q, R):
    """
    Calculate PD using fundamental matrix method.
    
    Q: Transient transition submatrix
    R: Absorption probability matrix
    """
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    B = N @ R
    pd_vector = B.sum(axis=1)  # Sum over all default states
    return pd_vector
```

#### Advantages:
- Direct calculation of ultimate PD
- No need for matrix powers
- Provides expected time to absorption

#### Limitations:
- Requires matrix inversion
- Only for ultimate (lifetime) PD
- Assumes absorbing default states

### 3. Eigenvalue Decomposition Method

For large matrices or repeated calculations:

```python
def calculate_pd_eigenvalue(P, h, default_buckets):
    """
    Calculate PD using eigenvalue decomposition.
    
    P: Transition matrix
    h: Time horizon
    default_buckets: List of default bucket indices
    """
    eigenvalues, eigenvectors = np.linalg.eig(P)
    
    # Reconstruct P^h using eigendecomposition
    P_h = eigenvectors @ np.diag(eigenvalues**h) @ np.linalg.inv(eigenvectors)
    
    pd_vector = P_h[:, default_buckets].sum(axis=1)
    return pd_vector.real  # Take real part due to numerical precision
```

### 4. Continuous-Time Approximation

For continuous monitoring or interpolation:

```python
def calculate_pd_continuous(Q_matrix, t, default_buckets):
    """
    Calculate PD using continuous-time approximation.
    
    Q_matrix: Generator matrix
    t: Time (in years)
    default_buckets: List of default bucket indices
    """
    from scipy.linalg import expm
    
    P_t = expm(Q_matrix * t)
    pd_vector = P_t[:, default_buckets].sum(axis=1)
    return pd_vector
```

## Time Horizon Considerations

### Term Structure of PD

PD typically increases with time horizon:

```
PD(1 month) ≤ PD(3 months) ≤ PD(12 months) ≤ PD(lifetime)
```

#### Marginal PD

Probability of default in period h (given survival to h-1):

```
Marginal_PD(h) = [PD(h) - PD(h-1)] / [1 - PD(h-1)]
```

#### Conditional PD

Probability of default in next period given current survival:

```python
def calculate_conditional_pd(transition_matrix, current_bucket, default_buckets):
    """Calculate 1-period conditional PD."""
    bucket_index = bucket_to_index[current_bucket]
    default_indices = [bucket_to_index[b] for b in default_buckets]
    
    conditional_pd = transition_matrix[bucket_index, default_indices].sum()
    return conditional_pd
```

### Forward PD Rates

Calculate forward default rates between periods:

```python
def calculate_forward_pd(pd_curve):
    """
    Calculate forward PD rates from cumulative PD curve.
    
    pd_curve: Array of cumulative PDs by period
    """
    forward_rates = np.zeros_like(pd_curve)
    forward_rates[0] = pd_curve[0]
    
    for i in range(1, len(pd_curve)):
        survival_rate = 1 - pd_curve[i-1]
        forward_rates[i] = (pd_curve[i] - pd_curve[i-1]) / survival_rate
    
    return forward_rates
```

## Advanced PD Calculations

### Through-the-Cycle (TTC) vs Point-in-Time (PIT) PD

#### TTC PD Calculation

Long-term average PD, adjusted for economic cycle:

```python
def calculate_ttc_pd(pit_pd_series, economic_indicators):
    """
    Calculate TTC PD from PIT observations.
    
    pit_pd_series: Historical PIT PD observations
    economic_indicators: Economic cycle indicators
    """
    # Regression to remove cyclical effects
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(economic_indicators.values.reshape(-1, 1), pit_pd_series)
    
    # TTC PD is intercept (PD at neutral economic conditions)
    ttc_pd = model.intercept_
    return ttc_pd
```

#### PIT PD Calculation

Current PD incorporating latest economic information:

```python
def calculate_pit_pd(ttc_pd, economic_adjustment_factor):
    """
    Calculate PIT PD from TTC PD.
    
    ttc_pd: Through-the-cycle PD
    economic_adjustment_factor: Current economic conditions multiplier
    """
    pit_pd = ttc_pd * economic_adjustment_factor
    return min(pit_pd, 1.0)  # Cap at 100%
```

### Stressed PD Calculations

#### Basel Stressed PD

For regulatory capital calculations:

```python
def calculate_stressed_pd(base_pd, stress_factor=3.0, floor_pd=0.0003):
    """
    Calculate Basel stressed PD.
    
    base_pd: Base case PD
    stress_factor: Stress multiplier (typically 3x)
    floor_pd: Regulatory floor (0.03%)
    """
    stressed_pd = base_pd * stress_factor
    return max(stressed_pd, floor_pd)
```

#### Scenario-Based Stressed PD

```python
def calculate_scenario_pd(baseline_matrix, scenario_adjustments):
    """
    Calculate PD under economic scenarios.
    
    baseline_matrix: Base transition matrix
    scenario_adjustments: Dictionary of adjustment factors
    """
    stressed_matrix = baseline_matrix.copy()
    
    # Apply scenario adjustments
    for transition_type, factor in scenario_adjustments.items():
        if transition_type == 'deterioration':
            # Increase downgrade probabilities
            for i in range(len(stressed_matrix)):
                for j in range(i+1, len(stressed_matrix)):
                    stressed_matrix.iloc[i, j] *= factor
        elif transition_type == 'improvement':
            # Decrease upgrade probabilities
            for i in range(len(stressed_matrix)):
                for j in range(i):
                    stressed_matrix.iloc[i, j] *= factor
    
    # Renormalize rows
    stressed_matrix = stressed_matrix.div(stressed_matrix.sum(axis=1), axis=0)
    
    return stressed_matrix
```

## Regulatory Applications

### IFRS 9 PD Requirements

#### 12-Month PD

For Stage 1 assets:

```python
def calculate_12m_pd(transition_matrix, default_buckets):
    """Calculate 12-month PD for IFRS 9."""
    return calculate_pd_matrix_power(transition_matrix, 12, default_buckets)
```

#### Lifetime PD

For Stage 2 and Stage 3 assets:

```python
def calculate_lifetime_pd(transition_matrix, default_buckets, max_life=360):
    """
    Calculate lifetime PD for IFRS 9.
    
    max_life: Maximum life in months (30 years = 360 months)
    """
    # Use fundamental matrix method for efficiency
    non_default_indices = [i for i in range(len(transition_matrix)) 
                          if i not in default_buckets]
    
    Q = transition_matrix.iloc[non_default_indices, non_default_indices]
    R = transition_matrix.iloc[non_default_indices, default_buckets]
    
    return calculate_pd_fundamental_matrix(Q, R)
```

### CECL PD Requirements

For Current Expected Credit Loss:

```python
def calculate_cecl_pd(transition_matrix, remaining_life_distribution):
    """
    Calculate CECL lifetime PD weighted by remaining life.
    
    remaining_life_distribution: Distribution of remaining asset lives
    """
    weighted_pd = 0
    
    for life, weight in remaining_life_distribution.items():
        life_pd = calculate_pd_matrix_power(transition_matrix, life, default_buckets)
        weighted_pd += life_pd * weight
    
    return weighted_pd
```

### Basel Capital Requirements

#### PD for Capital Calculation

```python
def calculate_basel_pd(ttc_pd, maturity_adjustment=1.0):
    """
    Calculate Basel PD for capital requirements.
    
    ttc_pd: Through-the-cycle PD
    maturity_adjustment: Maturity adjustment factor
    """
    # Apply regulatory floor
    floored_pd = max(ttc_pd, 0.0003)  # 0.03% floor
    
    # Apply maturity adjustment if applicable
    adjusted_pd = floored_pd * maturity_adjustment
    
    return min(adjusted_pd, 0.9999)  # Cap at 99.99%
```

## Model Validation

### PD Validation Metrics

#### Discriminatory Power

```python
def calculate_auc(actual_defaults, predicted_pd):
    """Calculate Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(actual_defaults, predicted_pd)
```

#### Calibration Assessment

```python
def assess_pd_calibration(predicted_pd, actual_defaults, n_bins=10):
    """
    Assess PD calibration using binning approach.
    
    Returns calibration statistics including Hosmer-Lemeshow test.
    """
    import pandas as pd
    
    # Create PD bins
    pd_bins = pd.qcut(predicted_pd, n_bins, duplicates='drop')
    
    # Calculate statistics by bin
    calibration_stats = pd.DataFrame({
        'bin': range(len(pd_bins.cat.categories)),
        'predicted_pd': predicted_pd.groupby(pd_bins).mean(),
        'actual_default_rate': actual_defaults.groupby(pd_bins).mean(),
        'count': predicted_pd.groupby(pd_bins).count()
    })
    
    # Calculate Hosmer-Lemeshow statistic
    expected = calibration_stats['predicted_pd'] * calibration_stats['count']
    observed = calibration_stats['actual_default_rate'] * calibration_stats['count']
    
    hl_statistic = ((observed - expected)**2 / (expected * (1 - calibration_stats['predicted_pd']))).sum()
    
    return calibration_stats, hl_statistic
```

#### Stability Testing

```python
def test_pd_stability(pd_series_t1, pd_series_t2, significance_level=0.05):
    """
    Test PD stability between two periods using Mann-Whitney U test.
    """
    from scipy.stats import mannwhitneyu
    
    statistic, p_value = mannwhitneyu(pd_series_t1, pd_series_t2)
    
    is_stable = p_value > significance_level
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_stable': is_stable,
        'significance_level': significance_level
    }
```

## Implementation Best Practices

### Numerical Considerations

#### Matrix Conditioning

```python
def check_matrix_conditioning(transition_matrix):
    """Check if transition matrix is well-conditioned."""
    condition_number = np.linalg.cond(transition_matrix)
    
    if condition_number > 1e12:
        warnings.warn("Transition matrix is poorly conditioned")
    
    return condition_number
```

#### Handling Near-Singular Matrices

```python
def regularize_transition_matrix(matrix, regularization_param=1e-8):
    """Add regularization to handle near-singular matrices."""
    regularized = matrix + regularization_param * np.eye(matrix.shape[0])
    # Renormalize rows
    return regularized / regularized.sum(axis=1, keepdims=True)
```

### Performance Optimization

#### Sparse Matrix Implementation

```python
def calculate_pd_sparse(sparse_matrix, h, default_buckets):
    """Calculate PD using sparse matrix operations."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import matrix_power
    
    sparse_P = csr_matrix(sparse_matrix)
    sparse_P_h = matrix_power(sparse_P, h)
    
    pd_vector = sparse_P_h[:, default_buckets].sum(axis=1).A1
    return pd_vector
```

#### Vectorized Calculations

```python
def calculate_pd_vectorized(transition_matrices, horizons, default_buckets):
    """Vectorized PD calculation for multiple scenarios."""
    results = np.zeros((len(transition_matrices), len(horizons), len(transition_matrices[0])))
    
    for i, matrix in enumerate(transition_matrices):
        for j, h in enumerate(horizons):
            results[i, j, :] = calculate_pd_matrix_power(matrix, h, default_buckets)
    
    return results
```

## Common Pitfalls and Solutions

### 1. Absorbing State Issues

**Problem**: Default states that allow transitions out
**Solution**: Ensure P[default, default] = 1

### 2. Matrix Convergence

**Problem**: Matrix powers don't converge
**Solution**: Check matrix eigenvalues and stationarity

### 3. Numerical Precision

**Problem**: Small probabilities lost due to floating point precision
**Solution**: Use high-precision arithmetic for small PDs

### 4. Data Sparsity

**Problem**: Insufficient observations for reliable estimation
**Solution**: Apply smoothing techniques or bucket aggregation

## References

### Academic Sources

1. **Markov Chain Applications in Finance**
   - Jarrow, R.A. et al. (1997). "A Markov Model for the Term Structure of Credit Risk Spreads"
   - Lando, D. (1998). "On Cox Processes and Credit Risky Securities"

2. **PD Modeling Methodologies**
   - Altman, E.I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"
   - Merton, R.C. (1974). "On the Pricing of Corporate Debt"

### Regulatory Guidance

1. **Basel Committee Publications**
   - "International Convergence of Capital Measurement and Capital Standards" (Basel II)
   - "Basel III: A Global Regulatory Framework"

2. **Accounting Standards**
   - IFRS 9: Financial Instruments
   - ASC 326: Current Expected Credit Losses (CECL)

### Industry Best Practices

1. **Risk Management Association (RMA)**
2. **International Association of Credit Portfolio Managers (IACPM)**
3. **Professional Risk Managers' International Association (PRMIA)**

## Conclusion

PD calculation using transition matrices provides a robust framework for credit risk quantification. Key considerations include:

- **Method Selection**: Choose appropriate calculation method based on requirements
- **Time Horizon**: Consider both regulatory and business needs
- **Validation**: Implement comprehensive validation framework
- **Numerical Stability**: Ensure robust numerical implementation
- **Regulatory Compliance**: Align with applicable accounting and regulatory standards

Proper implementation of these methodologies enables accurate risk measurement and regulatory compliance in credit portfolio management.
