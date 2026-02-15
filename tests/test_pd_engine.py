"""
Tests for the PD Engine module.
"""

import pytest
import pandas as pd
import numpy as np

from credit_risk_transition_matrix.core.pd_engine import (
    cumulative_pd,
    marginal_pd,
    pd_term_structure,
    survival_curve,
    cure_rates,
    absorbing_analysis,
    weighted_matrix
)


@pytest.fixture
def simple_matrix():
    """A simple 3x3 transition matrix for testing.
    
    States: Good, Bad, Default
    Good -> Good: 0.8, Good -> Bad: 0.15, Good -> Default: 0.05
    Bad -> Good: 0.10, Bad -> Bad: 0.60, Bad -> Default: 0.30
    Default -> Default: 1.0  (absorbing)
    """
    labels = ['Good', 'Bad', 'Default']
    data = [
        [0.80, 0.15, 0.05],
        [0.10, 0.60, 0.30],
        [0.00, 0.00, 1.00]
    ]
    return pd.DataFrame(data, index=labels, columns=labels)


@pytest.fixture
def learner_matrix():
    """Realistic 4-bucket matrix similar to what TransitionMatrixLearner produces."""
    labels = ['0-14', '15-29', '30-59', '60+']
    data = [
        [0.85, 0.10, 0.03, 0.02],
        [0.20, 0.55, 0.15, 0.10],
        [0.05, 0.10, 0.50, 0.35],
        [0.00, 0.00, 0.05, 0.95]
    ]
    return pd.DataFrame(data, index=labels, columns=labels)


class TestCumulativePD:
    """Tests for cumulative_pd function."""

    def test_basic_output_shape(self, simple_matrix):
        result = cumulative_pd(simple_matrix, n_periods=6)
        assert result.shape == (6, 3)
        assert list(result.index) == list(range(1, 7))
        assert list(result.columns) == ['Good', 'Bad', 'Default']

    def test_default_uses_last_bucket(self, simple_matrix):
        result = cumulative_pd(simple_matrix, n_periods=1)
        # Period 1: Good -> Default = 0.05
        assert abs(result.loc[1, 'Good'] - 0.05) < 1e-10

    def test_pd_is_monotonically_increasing(self, simple_matrix):
        result = cumulative_pd(simple_matrix, n_periods=12, default_states=['Default'])
        for col in result.columns:
            values = result[col].values
            for i in range(len(values) - 1):
                assert values[i + 1] >= values[i] - 1e-10

    def test_absorbing_state_reaches_1(self, simple_matrix):
        result = cumulative_pd(simple_matrix, n_periods=1, default_states=['Default'])
        # Default -> Default = 1.0, so PD from Default state should be 1.0
        assert abs(result.loc[1, 'Default'] - 1.0) < 1e-10

    def test_custom_default_states(self, simple_matrix):
        # Both Bad and Default as default states
        result = cumulative_pd(simple_matrix, n_periods=1, default_states=['Bad', 'Default'])
        # Good -> Bad + Good -> Default = 0.15 + 0.05 = 0.20
        assert abs(result.loc[1, 'Good'] - 0.20) < 1e-10

    def test_values_between_0_and_1(self, learner_matrix):
        result = cumulative_pd(learner_matrix, n_periods=24)
        assert (result.values >= -1e-10).all()
        assert (result.values <= 1.0 + 1e-10).all()


class TestMarginalPD:
    """Tests for marginal_pd function."""

    def test_basic_output(self, simple_matrix):
        result = marginal_pd(simple_matrix, n_periods=6)
        assert result.shape == (6, 3)

    def test_first_period_equals_cumulative(self, simple_matrix):
        mpd = marginal_pd(simple_matrix, n_periods=6)
        cpd = cumulative_pd(simple_matrix, n_periods=6)
        # First period marginal should equal cumulative
        pd.testing.assert_series_equal(mpd.iloc[0], cpd.iloc[0])

    def test_marginal_sums_to_cumulative(self, simple_matrix):
        mpd = marginal_pd(simple_matrix, n_periods=12)
        cpd = cumulative_pd(simple_matrix, n_periods=12)
        # Sum of marginals up to period t should equal cumulative at period t
        for col in mpd.columns:
            cumsum = mpd[col].cumsum()
            np.testing.assert_allclose(cumsum.values, cpd[col].values, atol=1e-10)


class TestSurvivalCurve:
    """Tests for survival_curve function."""

    def test_survival_plus_pd_equals_1(self, simple_matrix):
        surv = survival_curve(simple_matrix, n_periods=12)
        cpd = cumulative_pd(simple_matrix, n_periods=12)
        total = surv + cpd
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_monotonically_decreasing(self, simple_matrix):
        surv = survival_curve(simple_matrix, n_periods=12, default_states=['Default'])
        for col in surv.columns:
            values = surv[col].values
            for i in range(len(values) - 1):
                assert values[i + 1] <= values[i] + 1e-10


class TestPDTermStructure:
    """Tests for pd_term_structure function."""

    def test_has_all_metrics(self, simple_matrix):
        result = pd_term_structure(simple_matrix, n_periods=6)
        assert isinstance(result.columns, pd.MultiIndex)
        # Should have cumulative_pd, marginal_pd, survival for each bucket
        for bucket in simple_matrix.index:
            assert (bucket, 'cumulative_pd') in result.columns
            assert (bucket, 'marginal_pd') in result.columns
            assert (bucket, 'survival') in result.columns


class TestCureRates:
    """Tests for cure_rates function."""

    def test_basic_output(self, simple_matrix):
        result = cure_rates(simple_matrix)
        assert isinstance(result, pd.DataFrame)
        assert 'from_bucket' in result.columns
        assert 'to_bucket' in result.columns
        assert 'cure_probability' in result.columns

    def test_cure_rate_values(self, simple_matrix):
        result = cure_rates(simple_matrix)
        # Bad -> Good cure rate = 0.10
        bad_to_good = result[
            (result['from_bucket'] == 'Bad') & (result['to_bucket'] == 'Good')
        ]
        assert len(bad_to_good) == 1
        assert abs(bad_to_good.iloc[0]['cure_probability'] - 0.10) < 1e-10

    def test_total_cure_rate(self, simple_matrix):
        result = cure_rates(simple_matrix)
        # Bad total cure rate = probability of going to Good = 0.10
        bad_total = result[
            (result['from_bucket'] == 'Bad') & (result['to_bucket'] == '_TOTAL_CURE_')
        ]
        assert abs(bad_total.iloc[0]['cure_probability'] - 0.10) < 1e-10

    def test_no_cure_from_first_bucket(self, simple_matrix):
        result = cure_rates(simple_matrix)
        # First bucket has no cure (no better bucket to go to)
        assert 'Good' not in result['from_bucket'].values


class TestAbsorbingAnalysis:
    """Tests for absorbing_analysis function."""

    def test_expected_steps_output(self, simple_matrix):
        result = absorbing_analysis(simple_matrix, absorbing_states=['Default'])
        assert 'expected_steps' in result
        assert 'absorption_probabilities' in result
        
        expected = result['expected_steps']
        assert 'Good' in expected.index
        assert 'Bad' in expected.index
        assert 'Default' not in expected.index  # Absorbing state excluded

    def test_expected_steps_positive(self, simple_matrix):
        result = absorbing_analysis(simple_matrix, absorbing_states=['Default'])
        assert (result['expected_steps'] > 0).all()

    def test_absorption_probabilities_sum_to_1(self, simple_matrix):
        result = absorbing_analysis(simple_matrix, absorbing_states=['Default'])
        probs = result['absorption_probabilities']
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_worse_state_faster_absorption(self, simple_matrix):
        result = absorbing_analysis(simple_matrix, absorbing_states=['Default'])
        # Bad state should reach default faster than Good state
        assert result['expected_steps']['Bad'] < result['expected_steps']['Good']

    def test_default_absorbing_state(self, simple_matrix):
        # Default behavior: uses last bucket as absorbing
        result = absorbing_analysis(simple_matrix)
        assert 'Default' not in result['expected_steps'].index

    def test_all_absorbing_returns_empty(self, simple_matrix):
        # All states absorbing — edge case
        result = absorbing_analysis(
            simple_matrix, 
            absorbing_states=['Good', 'Bad', 'Default']
        )
        assert len(result['expected_steps']) == 0


class TestWeightedMatrix:
    """Tests for weighted_matrix function."""

    def test_basic_output(self):
        labels = ['A', 'B']
        transitions = pd.DataFrame({
            'from_bucket': ['A', 'A', 'A', 'B', 'B'],
            'to_bucket':   ['A', 'B', 'A', 'A', 'B'],
            'weight':      [100, 200, 50, 30, 70]
        })
        result = weighted_matrix(transitions, labels)
        
        assert result.shape == (2, 2)
        # A→A: (100+50)/(100+200+50) = 150/350 ≈ 0.4286
        assert abs(result.loc['A', 'A'] - 150/350) < 1e-10
        # A→B: 200/350 ≈ 0.5714
        assert abs(result.loc['A', 'B'] - 200/350) < 1e-10

    def test_rows_sum_to_1(self):
        labels = ['X', 'Y', 'Z']
        transitions = pd.DataFrame({
            'from_bucket': ['X', 'X', 'Y', 'Y', 'Z'],
            'to_bucket':   ['X', 'Y', 'Y', 'Z', 'Z'],
            'weight':      [10, 20, 30, 40, 50]
        })
        result = weighted_matrix(transitions, labels)
        for label in labels:
            row_sum = result.loc[label].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
