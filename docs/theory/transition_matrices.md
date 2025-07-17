# Transition Matrix Theory

## Mathematical Foundation

### Definition

A transition matrix **P** is a square matrix where element P[i,j] represents the probability of moving from state i to state j over a specific time period.

For credit risk applications:
- **States**: Risk buckets (e.g., days past due ranges)
- **Time Period**: Typically monthly observations
- **Transitions**: Migration between risk buckets

### Mathematical Properties

#### 1. Stochastic Matrix Properties

A valid transition matrix must satisfy:

```
∑ P[i,j] = 1 for all i  (row sums equal 1)
j

0 ≤ P[i,j] ≤ 1 for all i,j  (probabilities between 0 and 1)
```

#### 2. Matrix Notation

```
P = [P₁₁  P₁₂  ...  P₁ₙ]
    [P₂₁  P₂₂  ...  P₂ₙ]
    [...  ...  ...  ...]
    [Pₙ₁  Pₙ₂  ...  Pₙₙ]
```

Where:
- P[i,j] = Probability of transition from bucket i to bucket j
- n = Number of risk buckets

### Calculation Methodology

#### 1. Empirical Transition Probabilities

The maximum likelihood estimator for transition probabilities:

```
P̂[i,j] = N[i,j] / N[i]
```

Where:
- N[i,j] = Number of observed transitions from bucket i to j
- N[i] = Total number of observations starting in bucket i

#### 2. Confidence Intervals

95% confidence interval for transition probability:

```
P̂[i,j] ± 1.96 × √(P̂[i,j] × (1 - P̂[i,j]) / N[i])
```

#### 3. Smoothing Techniques

For sparse data, apply Laplace smoothing:

```
P̂[i,j] = (N[i,j] + α) / (N[i] + α × n)
```

Where α is the smoothing parameter (typically 0.5 to 1).

## Multi-Period Transitions

### Chapman-Kolmogorov Equation

For multi-period transitions:

```
P⁽ᵏ⁾ = P^k
```

Where P⁽ᵏ⁾ is the k-period transition matrix obtained by matrix multiplication.

#### Example: 2-Period Transitions

```
P⁽²⁾[i,j] = ∑ P[i,k] × P[k,j]
           k
```

This represents the probability of moving from bucket i to bucket j over 2 periods.

### Long-term Behavior

#### Steady-State Distribution

If the matrix is ergodic, there exists a stationary distribution π such that:

```
π × P = π
```

The steady-state probabilities represent long-term bucket distributions.

#### Absorbing States

Default buckets are typically modeled as absorbing states where:

```
P[default, default] = 1
P[default, j] = 0 for j ≠ default
```

## Credit Risk Applications

### Risk Bucket Definition

Typical credit risk buckets based on days past due (DPD):

| Bucket | DPD Range | Risk Level |
|--------|-----------|------------|
| B0 | 0-14 | Current |
| B1 | 15-29 | Early Delinquency |
| B2 | 30-59 | Moderate Delinquency |
| B3 | 60-89 | Significant Delinquency |
| B4 | 90-119 | Substandard |
| B5 | 120-179 | Doubtful |
| B6 | 180-239 | Loss |
| B7 | 240-359 | Default |
| B8 | 360+ | Write-off |

### Transition Patterns

#### Typical Characteristics

1. **Diagonal Dominance**: Most accounts remain in the same bucket
2. **Adjacent Movement**: Transitions typically occur to adjacent buckets
3. **Absorbing Default**: Once in default, accounts rarely return
4. **Seasonality**: Transition rates may vary by time of year

#### Performance Indicators

- **Stability**: High diagonal probabilities indicate portfolio stability
- **Deterioration Rate**: Sum of downward transition probabilities
- **Improvement Rate**: Sum of upward transition probabilities
- **Default Rate**: Transition probabilities to default buckets

## Statistical Properties

### Eigenvalue Decomposition

For matrix P with eigenvalues λ₁, λ₂, ..., λₙ:

```
P^k = ∑ λᵢᵏ × vᵢ × uᵢᵀ
      i
```

Where vᵢ and uᵢ are right and left eigenvectors.

### Convergence Rate

The second-largest eigenvalue determines convergence rate to steady state:

```
Rate = |λ₂|
```

Smaller |λ₂| implies faster convergence.

## Model Validation

### Statistical Tests

#### 1. Chi-Square Goodness of Fit

Test if observed transitions match expected:

```
χ² = ∑∑ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ
     i j
```

Where:
- Oᵢⱼ = Observed transitions
- Eᵢⱼ = Expected transitions under model

#### 2. Markov Property Test

Test if transition probabilities are independent of history beyond current state.

#### 3. Stationarity Test

Test if transition probabilities are stable over time using:

```
H₀: P(t₁) = P(t₂) for different time periods
```

### Performance Metrics

#### 1. Prediction Accuracy

```
Accuracy = ∑ᵢ ∑ⱼ (Predicted[i,j] × Actual[i,j]) / Total Transitions
```

#### 2. Mean Absolute Error (MAE)

```
MAE = (1/n²) × ∑ᵢ ∑ⱼ |P̂[i,j] - P_true[i,j]|
```

#### 3. Kullback-Leibler Divergence

```
KL(P||Q) = ∑ᵢ ∑ⱼ P[i,j] × log(P[i,j] / Q[i,j])
```

## Advanced Topics

### Non-Homogeneous Markov Chains

When transition probabilities depend on external factors:

```
P(t) = P₀ × exp(X(t) × β)
```

Where:
- X(t) = Covariate matrix at time t
- β = Regression coefficients

### Continuous-Time Markov Chains

For continuous monitoring:

```
P(t) = exp(Q × t)
```

Where Q is the generator matrix with:
- Q[i,j] ≥ 0 for i ≠ j (transition rates)
- Q[i,i] = -∑ⱼ≠ᵢ Q[i,j] (negative sum of row)

### Hidden Markov Models

When true risk state is unobservable:

```
Observed Bucket = f(Hidden Risk State, Observation Error)
```

This accounts for measurement error in bucket assignments.

## Regulatory Considerations

### Basel Accord Requirements

- **Data Requirements**: Minimum 5 years of data
- **Validation**: Annual model validation required
- **Documentation**: Complete methodology documentation
- **Governance**: Independent model validation

### IFRS 9 Applications

Transition matrices support:
- **Stage Classification**: 1→2→3 staging based on credit deterioration
- **Forward-Looking Information**: Incorporating economic scenarios
- **Expected Credit Loss**: Multi-period PD calculations

### CECL Implementation

For Current Expected Credit Loss:
- **Lifetime PD**: Calculate probability of default over instrument life
- **Economic Scenarios**: Multiple scenario weighting
- **Vintage Analysis**: Performance by origination cohort

## Implementation Considerations

### Data Requirements

#### Minimum Data Standards

- **Time Series**: At least 2-3 years of monthly data
- **Sample Size**: Minimum 30 transitions per bucket
- **Data Quality**: Complete observation history
- **Representativeness**: Data reflects current portfolio

#### Data Preprocessing

1. **Alignment**: Ensure consistent time intervals
2. **Filtering**: Remove accounts with insufficient history
3. **Bucket Assignment**: Consistent bucket definition rules
4. **Outlier Treatment**: Handle extreme values appropriately

### Computational Complexity

- **Memory**: O(n²) for n buckets
- **Computation**: O(n³) for matrix operations
- **Storage**: O(n² × T) for T time periods

### Software Implementation

Key considerations:
- **Numerical Stability**: Handle near-singular matrices
- **Performance**: Optimize for large datasets
- **Validation**: Built-in statistical tests
- **Visualization**: Professional plotting capabilities

## References

### Academic Literature

1. **Markov Chain Theory**
   - Norris, J.R. (1997). "Markov Chains"
   - Ross, S.M. (2014). "Introduction to Probability Models"

2. **Credit Risk Applications**
   - Lando, D. (2004). "Credit Risk Modeling"
   - McNeil, A.J. et al. (2015). "Quantitative Risk Management"

3. **Regulatory Guidance**
   - Basel Committee on Banking Supervision
   - International Accounting Standards Board (IFRS 9)
   - Financial Accounting Standards Board (CECL)

### Industry Standards

- **ISDA**: Credit Risk Modeling Standards
- **GARP**: Risk Management Standards
- **PRMIA**: Professional Risk Manager Standards

## Conclusion

Transition matrices provide a powerful framework for credit risk analysis, combining:
- **Mathematical Rigor**: Strong theoretical foundation
- **Practical Application**: Direct regulatory compliance
- **Interpretability**: Clear business insight
- **Flexibility**: Adaptable to various portfolios

Understanding the mathematical foundation ensures proper implementation and validation of transition matrix models in credit risk management.
