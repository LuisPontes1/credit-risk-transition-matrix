"""
Tests for the rating_scale feature in TransitionMatrixLearner.

Verifies that the learner works correctly when initialized with
pre-assigned rating labels instead of numeric DPD buckets.
"""

import pytest
import pandas as pd
import numpy as np
from credit_risk_transition_matrix import TransitionMatrixLearner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rating_data(
    n_contracts: int = 200,
    n_periods: int = 12,
    ratings: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic panel data with pre-assigned ratings."""
    rng = np.random.default_rng(seed)
    if ratings is None:
        ratings = ['7', '6', '5', '4', '3', '2', '1']  # best→worst
    
    rows = []
    for cid in range(1, n_contracts + 1):
        current = rng.choice(ratings)
        for t in range(n_periods):
            rows.append({
                'id_contrato': f'C{cid:04d}',
                'data_ref': pd.Timestamp('2023-01-01') + pd.DateOffset(months=t),
                'rating': current,
                'ead': rng.uniform(1_000, 100_000),
                'segmento': rng.choice(['PF', 'PJ'])
            })
            # Simulate migration: 15% chance of moving to adjacent rating
            idx = ratings.index(current)
            r = rng.random()
            if r < 0.10 and idx < len(ratings) - 1:
                current = ratings[idx + 1]  # downgrade (higher index = worse)
            elif r < 0.15 and idx > 0:
                current = ratings[idx - 1]  # upgrade (lower index = better)
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRatingScaleInit:
    """Test TransitionMatrixLearner initialization with rating_scale."""
    
    def test_rating_scale_sets_labels(self):
        scale = ['7', '6', '5', '4', '3', '2', '1']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        assert learner.bucket_labels_ == scale
        assert learner.buckets_ is None
        assert learner._pre_assigned is True
    
    def test_rating_scale_casts_to_str(self):
        """Numeric ratings should be cast to string."""
        learner = TransitionMatrixLearner(rating_scale=[1, 2, 3, 4, 5], min_observations=10)
        assert learner.bucket_labels_ == ['1', '2', '3', '4', '5']
    
    def test_dpd_mode_default(self):
        learner = TransitionMatrixLearner()
        assert learner._pre_assigned is False
        assert learner.buckets_ is not None


class TestRatingScaleFit:
    """Test fit() with pre-assigned ratings."""
    
    def test_fit_basic(self):
        df = _make_rating_data(n_contracts=100, n_periods=12)
        scale = ['7', '6', '5', '4', '3', '2', '1']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
        
        assert learner.is_fitted_
        assert learner.transition_matrix_ is not None
        assert list(learner.transition_matrix_.index) == scale
        assert list(learner.transition_matrix_.columns) == scale
    
    def test_matrix_rows_sum_to_one(self):
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
        
        row_sums = learner.transition_matrix_.sum(axis=1)
        for rating in scale:
            if row_sums[rating] > 0:
                np.testing.assert_almost_equal(row_sums[rating], 1.0, decimal=6)
    
    def test_fit_with_segments(self):
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(
            df, id_col='id_contrato', time_col='data_ref',
            bucket_col='rating', segment_col='segmento'
        )
        
        assert len(learner.segmented_matrices_) > 0
        for seg, mat in learner.segmented_matrices_.items():
            assert list(mat.index) == scale
            assert list(mat.columns) == scale
    
    def test_fit_with_weights(self):
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(
            df, id_col='id_contrato', time_col='data_ref',
            bucket_col='rating', weight_col='ead'
        )
        
        assert learner.is_fitted_
        row_sums = learner.transition_matrix_.sum(axis=1)
        for rating in scale:
            if row_sums[rating] > 0:
                np.testing.assert_almost_equal(row_sums[rating], 1.0, decimal=6)


class TestRatingScaleValidation:
    """Test validation of rating column."""
    
    def test_unknown_rating_raises(self):
        df = _make_rating_data(n_contracts=50, n_periods=6)
        # Inject an unknown rating
        df.loc[0, 'rating'] = '99'
        scale = ['7', '6', '5', '4', '3', '2', '1']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        
        with pytest.raises(ValueError, match="unknown values"):
            learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
    
    def test_null_rating_raises(self):
        df = _make_rating_data(n_contracts=50, n_periods=6)
        df.loc[0, 'rating'] = None
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        
        with pytest.raises(ValueError, match="null values"):
            learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')


class TestRatingScaleDownstream:
    """Test downstream features (PD, stress testing, ECL) with rating scale."""
    
    def test_pd_term_structure(self):
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
        
        pd_ts = learner.get_pd_term_structure(
            n_periods=12, default_states=['1']
        )
        assert len(pd_ts) == 12
    
    def test_stress_testing(self):
        from credit_risk_transition_matrix import apply_scalar_stress
        
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
        
        stressed = apply_scalar_stress(learner.transition_matrix_, stress_factor=1.5)
        assert stressed.shape == learner.transition_matrix_.shape
        row_sums = stressed.sum(axis=1)
        for s in row_sums:
            np.testing.assert_almost_equal(s, 1.0, decimal=6)
    
    def test_ecl_by_stage(self):
        from credit_risk_transition_matrix import classify_stage, ecl_by_stage
        
        df = _make_rating_data(n_contracts=200, n_periods=12)
        scale = ['1', '2', '3', '4', '5', '6', '7']
        learner = TransitionMatrixLearner(rating_scale=scale, min_observations=10)
        learner.fit(df, id_col='id_contrato', time_col='data_ref', bucket_col='rating')
        
        # Create a snapshot for staging
        snapshot = df.groupby('id_contrato').last().reset_index()
        snapshot = snapshot.rename(columns={'rating': 'bucket'})
        staged = classify_stage(
            snapshot, bucket_col='bucket',
            bucket_labels=scale,
            default_buckets=['1']
        )
        
        # Calculate ECL
        result = ecl_by_stage(
            learner.transition_matrix_,
            staged,
            bucket_col='bucket',
            stage_col='stage',
            ead_col='ead',
            lgd=0.45,
            default_states=['1']
        )
        
        assert 'ecl' in result.columns
        assert (result['ecl'] >= 0).all()
