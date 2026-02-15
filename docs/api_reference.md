# API Reference

This reference documents the primary classes and functions of the `credit_risk_transition_matrix` library.

---

## 1. Core Engine

### `ContractRiskEngine`

The main orchestrator for calculating ECL on a contract level.

```python
class ContractRiskEngine:
    def __init__(self, product_schema: str = 'GIRO_TPV', macro_scenarios: List[MacroScenario] = None):
        """
        Initializes the risk engine with a specific product configuration and macro scenarios.
        
        Args:
            product_schema (str): Key for ProductSchema configuration (e.g., 'GIRO_TPV', 'REVOLVING').
            macro_scenarios (List[MacroScenario], optional): Custom scenarios. Defaults to [Optimistic, Base, Pessimistic].
        """

    def fit(self, df: pd.DataFrame, id_col: str, time_col: str, bucket_col: str, ead_col: str = None) -> 'ContractRiskEngine':
        """
        Trains the internal models (LGD, CCF) using historical data.
        
        Args:
            df (pd.DataFrame): Historical dataset.
            id_col (str): Column name for contract ID.
            time_col (str): Column name for reference date.
            bucket_col (str): Column name for delinquency bucket (DPD or string).
            ead_col (str, optional): Column name for exposure amount (used for LGD weighting).
        """

    def compute_ecl(self, snapshot: pd.DataFrame, pd_col: str, pd_origination_col: str, 
                    ead_col: str = None, limit_col: str = None, 
                    collateral_type_col: str = None, recovery_col: str = None) -> pd.DataFrame:
        """
        Computes the Expected Credit Loss (ECL) for a portfolio snapshot.
        
        Args:
            snapshot (pd.DataFrame): Current portfolio data.
            pd_col (str): Column containing external 12-month PD.
            pd_origination_col (str): PDP at origination (for SICR staging).
            ead_col (str, optional): Current balance.
            limit_col (str, optional): Total credit limit (for revolving products).
            collateral_type_col (str, optional): Column for collateral type (e.g., 'VEICULO', 'IMOVEL').
            
        Returns:
            pd.DataFrame: Snapshot with appended columns: 'stage', 'ecl', 'allowance', 'provision'.
        """
```

---

## 2. Transition Matrix Module

### `TransitionMatrixLearner`

Estimates transition probabilities from historical data.

```python
class TransitionMatrixLearner:
    def __init__(self, buckets: List[Union[int, str]], time_horizon: int = 12):
        """
        Args:
            buckets (List): List of bucket edges (e.g., [0, 30, 60]) or labels.
            time_horizon (int): Number of months for projection.
        """

    def fit(self, df: pd.DataFrame, id_col: str, time_col: str, bucket_col: str, 
            segment_col: str = None, weight_col: str = None):
        """
        Calculates the transition matrix.
        """
```

### Helper Functions

- **`cumulative_pd(matrix, n_periods)`**: Computes cumulative PD curve up to n periods.
- **`marginal_pd(matrix, n_periods)`**: Computes the marginal PD for each period.
- **`survival_curve(matrix, n_periods)`**: Computes the probability of survival (1 - Cumulative PD).
- **`cure_rates(matrix)`**: Calculates the probability of returning to specific performing states from default/worse states.

---

## 3. LGD & EAD Engines

### `LGDRatingEngine`

- **`rate(snapshot, ...)`**: Assigns LGD values based on product schema and collateral. Supports downturn adjustments.

### `ead_engine`

- **`estimate_ccf(df, ...)`**: Empirically estimates the Credit Conversion Factor (CCF) from historical limit utilization data.
- **`compute_ead(snapshot, ...)`**: Calculates EAD for term loans (amortized balance) and revolving lines (Balance + CCF * Unused).

---

## 4. Staging (SICR)

### `classify_stage_sicr`

Classifies contracts into Stage 1, 2, or 3 based on IFRS 9 criteria.

```python
def classify_stage_sicr(
    df: pd.DataFrame, 
    pd_current_col: str, 
    pd_origination_col: str, 
    default_buckets: List[str],
    bucket_col: str,
    sicr_relative_threshold: float = 2.0,   # PD_now / PD_orig > 2.0
    sicr_absolute_threshold: float = 0.03,  # PD_now - PD_orig > 3.0%
    dpd_stage2_threshold: int = 30          # Backstop > 30 days
) -> pd.DataFrame:
    """
    Returns DataFrame with a new 'stage' column (1, 2, 3).
    """
```
