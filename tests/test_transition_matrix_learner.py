"""
Test the TransitionMatrixLearner class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from credit_risk_transition_matrix import TransitionMatrixLearner


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create 100 contracts with monthly observations over 2 years
    contracts = [f"CONT_{i:03d}" for i in range(100)]
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
    
    data = []
    for contract in contracts:
        # Each contract starts in a random bucket and evolves
        current_bucket = np.random.choice([0, 15, 30, 60])
        
        for date in dates:
            # Add some randomness to bucket evolution
            if np.random.random() < 0.1:  # 10% chance to worsen
                current_bucket = min(current_bucket + 30, 360)
            elif np.random.random() < 0.05:  # 5% chance to improve
                current_bucket = max(current_bucket - 15, 0)
            
            data.append({
                'id_contrato': contract,
                'data_ref': date,
                'dias_atraso': current_bucket,
                'segmento': 'Premium' if contract.endswith(('0', '1', '2')) else 'Standard'
            })
    
    return pd.DataFrame(data)


class TestTransitionMatrixLearner:
    """Test cases for TransitionMatrixLearner class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        learner = TransitionMatrixLearner()
        
        assert learner.time_horizon == 12
        assert learner.min_observations == 100
        assert len(learner.buckets_) == 9
        assert not learner.is_fitted_
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        custom_buckets = [0, 30, 60, 90, 180, 360]
        learner = TransitionMatrixLearner(
            buckets=custom_buckets,
            time_horizon=6,
            min_observations=50
        )
        
        assert learner.buckets_ == custom_buckets
        assert learner.time_horizon == 6
        assert learner.min_observations == 50
        assert len(learner.bucket_labels_) == len(custom_buckets)
    
    def test_fit_basic(self, sample_data):
        """Test basic fitting functionality."""
        learner = TransitionMatrixLearner(min_observations=50)
        
        # Should not raise any exceptions
        learner.fit(
            sample_data,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='dias_atraso'
        )
        
        assert learner.is_fitted_
        assert learner.transition_matrix_ is not None
        assert learner.transition_matrix_.shape[0] == learner.transition_matrix_.shape[1]
    
    def test_fit_with_segments(self, sample_data):
        """Test fitting with segment analysis."""
        learner = TransitionMatrixLearner(min_observations=10)
        
        learner.fit(
            sample_data,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='dias_atraso',
            segment_col='segmento'
        )
        
        assert learner.is_fitted_
        assert len(learner.segmented_matrices_) > 0
        assert 'Premium' in learner.segmented_matrices_
        assert 'Standard' in learner.segmented_matrices_
    
    def test_transform_global(self, sample_data):
        """Test transform method for global matrices."""
        learner = TransitionMatrixLearner(min_observations=50)
        learner.fit(
            sample_data,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='dias_atraso'
        )
        
        result = learner.transform(modes=['global'])
        
        assert 'global' in result
        assert isinstance(result['global'], pd.DataFrame)
        
        # Check matrix properties
        matrix = result['global']
        assert matrix.shape[0] == matrix.shape[1]  # Square matrix
        
        # Check row sums are approximately 1 (probabilities)
        row_sums = matrix.sum(axis=1)
        assert all(abs(row_sum - 1.0) < 0.01 for row_sum in row_sums if row_sum > 0)
    
    def test_predict_transitions(self, sample_data):
        """Test prediction functionality."""
        learner = TransitionMatrixLearner(min_observations=50)
        learner.fit(
            sample_data,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='dias_atraso'
        )
        
        # Test with bucket name
        result = learner.predict_transitions('0-14', n_periods=1)
        
        assert isinstance(result, list)
        assert len(result) == len(learner.bucket_labels_)
        assert abs(sum(result) - 1.0) < 0.01  # Should sum to 1
        
        # Test with probability vector
        initial_state = [1.0] + [0.0] * (len(learner.bucket_labels_) - 1)
        result2 = learner.predict_transitions(initial_state, n_periods=1)
        
        assert isinstance(result2, list)
        assert len(result2) == len(learner.bucket_labels_)
    
    def test_calculate_pd(self, sample_data):
        """Test PD calculation."""
        learner = TransitionMatrixLearner(min_observations=50)
        learner.fit(
            sample_data,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='dias_atraso'
        )
        
        pd_results = learner.calculate_pd(time_horizon=12)
        
        assert 'pd_by_bucket' in pd_results
        assert 'time_horizon' in pd_results
        assert 'default_buckets' in pd_results
        
        # Check that PD values are between 0 and 1
        for bucket, pd_value in pd_results['pd_by_bucket'].items():
            assert 0 <= pd_value <= 1
    
    def test_error_handling(self):
        """Test error handling for various edge cases."""
        learner = TransitionMatrixLearner()
        
        # Test fitting with empty dataframe
        with pytest.raises(ValueError):
            learner.fit(pd.DataFrame(), 'id', 'date', 'bucket')
        
        # Test operations before fitting
        with pytest.raises(ValueError):
            learner.transform()
        
        with pytest.raises(ValueError):
            learner.predict_transitions('0-14')
        
        with pytest.raises(ValueError):
            learner.calculate_pd()
    
    def test_validation_errors(self, sample_data):
        """Test validation errors for bad input."""
        learner = TransitionMatrixLearner()
        
        # Test with missing columns
        with pytest.raises(ValueError):
            learner.fit(sample_data, 'missing_col', 'data_ref', 'dias_atraso')
        
        # Test with non-DataFrame input
        with pytest.raises(TypeError):
            learner.fit("not a dataframe", 'id', 'date', 'bucket')
    
    def test_repr(self, sample_data):
        """Test string representation."""
        learner = TransitionMatrixLearner()
        
        # Before fitting
        repr_str = repr(learner)
        assert 'not fitted' in repr_str
        assert 'buckets=9' in repr_str
        
        # After fitting
        learner.fit(
            sample_data,
            id_col='id_contrato',
            time_col='data_ref',
            bucket_col='dias_atraso',
            segment_col='segmento'
        )
        
        repr_str = repr(learner)
        assert 'fitted' in repr_str
        assert 'segments=2' in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
