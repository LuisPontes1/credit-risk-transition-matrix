"""
Tests for the LGD Engine module.

Tests lgd_from_cure_rates (matrix-implied LGD), lgd_workout (contract-level
recovery tracking), lgd_by_segment (aggregation), and integration with
TransitionMatrixLearner.
"""

import pytest
import pandas as pd
import numpy as np
from credit_risk_transition_matrix import (
    TransitionMatrixLearner,
    lgd_from_cure_rates,
    lgd_workout,
    lgd_by_segment
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rating_data(
    n_contracts: int = 200,
    n_periods: int = 12,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic panel data with ratings (7=best, 1=default)."""
    rng = np.random.default_rng(seed)
    ratings = ['7', '6', '5', '4', '3', '2', '1']  # best→worst
    
    rows = []
    for cid in range(1, n_contracts + 1):
        current = rng.choice(ratings)
        for t in range(n_periods):
            rows.append({
                'id_contrato': f'C{cid:04d}',
                'data_ref': pd.Timestamp('2023-01-01') + pd.DateOffset(months=t),
                'rating': current,
                'saldo': rng.uniform(1_000, 100_000),
                'tipo_produto': rng.choice(['cartao', 'cheque_especial', 'capital_giro'])
            })
            # Simulate migration
            idx = ratings.index(current)
            r = rng.random()
            if r < 0.10 and idx < len(ratings) - 1:
                current = ratings[idx + 1]  # downgrade
            elif r < 0.15 and idx > 0:
                current = ratings[idx - 1]  # upgrade
    
    return pd.DataFrame(rows)


def _fit_learner(df):
    """Fit a TransitionMatrixLearner with standard settings."""
    scale = ['7', '6', '5', '4', '3', '2', '1']
    learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
    learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
    return learner


# ---------------------------------------------------------------------------
# Tests: lgd_from_cure_rates
# ---------------------------------------------------------------------------

class TestLGDFromCureRates:
    """Test LGD implied by cure rates."""
    
    def test_basic(self):
        df = _make_rating_data()
        learner = _fit_learner(df)
        result = lgd_from_cure_rates(learner.transition_matrix_, default_states=['1'])
        
        assert len(result) == 1
        assert result.iloc[0]['default_state'] == '1'
        assert 0.0 <= result.iloc[0]['lgd_implied'] <= 1.0
        assert 0.0 <= result.iloc[0]['cure_rate'] <= 1.0
    
    def test_recovery_rate_equals_cure_rate_without_haircut(self):
        df = _make_rating_data()
        learner = _fit_learner(df)
        result = lgd_from_cure_rates(
            learner.transition_matrix_, default_states=['1'], recovery_haircut=0.0
        )
        
        row = result.iloc[0]
        assert abs(row['recovery_rate'] - row['cure_rate']) < 1e-6
    
    def test_haircut_reduces_recovery(self):
        df = _make_rating_data()
        learner = _fit_learner(df)
        
        no_haircut = lgd_from_cure_rates(
            learner.transition_matrix_, default_states=['1'], recovery_haircut=0.0
        )
        with_haircut = lgd_from_cure_rates(
            learner.transition_matrix_, default_states=['1'], recovery_haircut=0.3
        )
        
        assert with_haircut.iloc[0]['lgd_implied'] >= no_haircut.iloc[0]['lgd_implied']
        assert with_haircut.iloc[0]['recovery_rate'] < no_haircut.iloc[0]['recovery_rate']
    
    def test_lgd_plus_recovery_equals_one(self):
        df = _make_rating_data()
        learner = _fit_learner(df)
        result = lgd_from_cure_rates(learner.transition_matrix_, default_states=['1'])
        
        row = result.iloc[0]
        assert abs(row['lgd_implied'] + row['recovery_rate'] - 1.0) < 1e-6
    
    def test_default_states_default_to_last_bucket(self):
        df = _make_rating_data()
        learner = _fit_learner(df)
        result = lgd_from_cure_rates(learner.transition_matrix_)
        
        # Last bucket in the scale is '1' (worst)
        assert result.iloc[0]['default_state'] == '1'
    
    def test_via_learner_convenience(self):
        """Test the get_lgd_implied() method on TransitionMatrixLearner."""
        df = _make_rating_data()
        learner = _fit_learner(df)
        result = learner.get_lgd_implied(default_states=['1'])
        
        assert len(result) == 1
        assert 0.0 <= result.iloc[0]['lgd_implied'] <= 1.0


# ---------------------------------------------------------------------------
# Tests: lgd_workout
# ---------------------------------------------------------------------------

class TestLGDWorkout:
    """Test contract-level workout LGD calculation."""
    
    def test_basic(self):
        """Test LGD workout with simulated data."""
        df = _make_rating_data(n_contracts=200, n_periods=12)
        result = lgd_workout(
            df,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='rating',
            ead_col='saldo',
            default_states=['1']
        )
        
        assert 'contract_id' in result.columns
        assert 'lgd' in result.columns
        assert 'cure_flag' in result.columns
        assert (result['lgd'] >= 0).all()
        assert (result['lgd'] <= 1).all()
    
    def test_no_defaults_returns_empty(self):
        """If no contracts enter default, return empty DataFrame."""
        # Create data with only good ratings
        rng = np.random.default_rng(99)
        rows = []
        for cid in range(1, 51):
            for t in range(12):
                rows.append({
                    'id': f'C{cid:04d}',
                    'data_ref': pd.Timestamp('2023-01-01') + pd.DateOffset(months=t),
                    'rating': rng.choice(['7', '6', '5']),  # Never enters '1'
                    'saldo': 50_000.0
                })
        df = pd.DataFrame(rows)
        
        result = lgd_workout(
            df, id_col='id', time_col='data_ref',
            bucket_col='rating', ead_col='saldo',
            default_states=['1']
        )
        
        assert len(result) == 0
    
    def test_workout_periods_limit(self):
        """Test that workout_periods limits tracking window."""
        df = _make_rating_data(n_contracts=200, n_periods=24)
        result = lgd_workout(
            df,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='rating',
            ead_col='saldo',
            default_states=['1'],
            workout_periods=6
        )
        
        if not result.empty:
            assert result['months_in_workout'].max() <= 6


# ---------------------------------------------------------------------------
# Tests: lgd_by_segment
# ---------------------------------------------------------------------------

class TestLGDBySegment:
    """Test LGD aggregation by segment."""
    
    def test_overall(self):
        """Test overall aggregation (no segment)."""
        lgd_data = pd.DataFrame({
            'contract_id': ['C1', 'C2', 'C3'],
            'lgd': [0.6, 0.8, 0.5],
            'ead_at_default': [10_000, 20_000, 30_000]
        })
        
        result = lgd_by_segment(lgd_data)
        
        assert len(result) == 1
        assert result.iloc[0]['count'] == 3
        np.testing.assert_almost_equal(
            result.iloc[0]['lgd_mean'], np.mean([0.6, 0.8, 0.5]), decimal=4
        )
    
    def test_by_product(self):
        """Test aggregation by product type."""
        lgd_data = pd.DataFrame({
            'contract_id': ['C1', 'C2', 'C3', 'C4'],
            'lgd': [0.6, 0.8, 0.3, 0.4],
            'ead_at_default': [10_000, 20_000, 30_000, 15_000],
            'produto': ['cartao', 'cartao', 'capital_giro', 'capital_giro']
        })
        
        result = lgd_by_segment(lgd_data, segment_col='produto')
        
        assert len(result) == 2
        segments = result['segment'].tolist()
        assert 'cartao' in segments
        assert 'capital_giro' in segments
    
    def test_weighted_lgd(self):
        """Test EAD-weighted LGD is different from simple mean."""
        lgd_data = pd.DataFrame({
            'contract_id': ['C1', 'C2'],
            'lgd': [0.2, 0.9],
            'ead_at_default': [100_000, 1_000]  # C1 dominates by EAD
        })
        
        result = lgd_by_segment(lgd_data)
        
        # Weighted LGD should be closer to 0.2 (C1 has 100x more EAD)
        assert result.iloc[0]['lgd_weighted'] < result.iloc[0]['lgd_mean']
    
    def test_empty_input(self):
        """Test empty input returns empty DataFrame."""
        result = lgd_by_segment(pd.DataFrame())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Integration
# ---------------------------------------------------------------------------

class TestLGDIntegration:
    """Test full PD → LGD → ECL integration."""
    
    def test_full_flow(self):
        """End-to-end: fit PD model, compute implied LGD, use in ECL."""
        from credit_risk_transition_matrix import classify_stage, ecl_by_stage
        
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['7', '6', '5', '4', '3', '2', '1']
        
        # 1. Fit PD model
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
        
        # 2. Get implied LGD
        lgd_result = learner.get_lgd_implied(default_states=['1'])
        lgd_value = lgd_result.iloc[0]['lgd_implied']
        
        assert 0.0 < lgd_value < 1.0
        
        # 3. Staging
        snapshot = df.groupby('id_contrato').last().reset_index()
        snapshot = snapshot.rename(columns={'rating': 'bucket'})
        staged = classify_stage(
            snapshot, bucket_col='bucket',
            bucket_labels=scale,
            default_buckets=['1']
        )
        
        # 4. ECL using implied LGD
        result = ecl_by_stage(
            learner.transition_matrix_, staged,
            bucket_col='bucket',
            stage_col='stage',
            ead_col='saldo',
            lgd=lgd_value,
            default_states=['1']
        )
        
        assert 'ecl' in result.columns
        assert (result['ecl'] >= 0).all()
        assert result['ecl'].sum() > 0
