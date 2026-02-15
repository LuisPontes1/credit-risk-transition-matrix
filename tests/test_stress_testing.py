"""Tests for the Stress Testing module."""

import pytest
import pandas as pd
import numpy as np

from credit_risk_transition_matrix.core.stress_testing import (
    apply_scalar_stress,
    apply_z_shift,
    generate_scenarios,
    weighted_scenario_matrix
)


@pytest.fixture
def simple_matrix():
    labels = ['Good', 'Bad', 'Default']
    data = [
        [0.80, 0.15, 0.05],
        [0.10, 0.60, 0.30],
        [0.00, 0.00, 1.00]
    ]
    return pd.DataFrame(data, index=labels, columns=labels)


class TestApplyScalarStress:

    def test_rows_sum_to_1(self, simple_matrix):
        result = apply_scalar_stress(simple_matrix, stress_factor=2.0)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_worsen_increases_downward_migration(self, simple_matrix):
        result = apply_scalar_stress(simple_matrix, stress_factor=2.0, direction='worsen')
        # Good -> Default should increase relative to original
        # But diagonal should decrease since off-diagonal increased
        assert result.loc['Good', 'Default'] > simple_matrix.loc['Good', 'Default']

    def test_improve_increases_upward_migration(self, simple_matrix):
        result = apply_scalar_stress(simple_matrix, stress_factor=2.0, direction='improve')
        # Bad -> Good should increase
        assert result.loc['Bad', 'Good'] > simple_matrix.loc['Bad', 'Good']

    def test_factor_1_returns_same_matrix(self, simple_matrix):
        result = apply_scalar_stress(simple_matrix, stress_factor=1.0)
        pd.testing.assert_frame_equal(result, simple_matrix, atol=1e-10)

    def test_negative_factor_raises(self, simple_matrix):
        with pytest.raises(ValueError):
            apply_scalar_stress(simple_matrix, stress_factor=-1.0)

    def test_invalid_direction_raises(self, simple_matrix):
        with pytest.raises(ValueError):
            apply_scalar_stress(simple_matrix, direction='invalid')

    def test_absorbing_state_preserved(self, simple_matrix):
        result = apply_scalar_stress(simple_matrix, stress_factor=3.0)
        # Default is absorbing (1.0 on diagonal, 0 elsewhere) — should stay same
        assert abs(result.loc['Default', 'Default'] - 1.0) < 1e-10


class TestApplyZShift:

    def test_rows_sum_to_1(self, simple_matrix):
        result = apply_z_shift(simple_matrix, z_shift=0.5)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_zero_shift_preserves_matrix(self, simple_matrix):
        result = apply_z_shift(simple_matrix, z_shift=0.0)
        pd.testing.assert_frame_equal(result, simple_matrix, atol=1e-10)

    def test_positive_shift_moves_mass_right(self, simple_matrix):
        result = apply_z_shift(simple_matrix, z_shift=1.0)
        # Probability mass should shift toward worse buckets
        # Good row: the mass at Good should be less than original
        assert result.loc['Good', 'Good'] < simple_matrix.loc['Good', 'Good']


class TestGenerateScenarios:

    def test_default_scenarios(self, simple_matrix):
        result = generate_scenarios(simple_matrix)
        assert 'base' in result
        assert 'optimistic' in result
        assert 'pessimistic' in result
        assert '_weights' in result

    def test_custom_scenarios(self, simple_matrix):
        config = {
            'mild': {'factor': 1.2, 'weight': 0.5},
            'severe': {'factor': 2.5, 'weight': 0.5}
        }
        result = generate_scenarios(simple_matrix, scenarios_config=config)
        assert 'mild' in result
        assert 'severe' in result

    def test_scenario_matrices_valid(self, simple_matrix):
        result = generate_scenarios(simple_matrix)
        for name, matrix in result.items():
            if name == '_weights':
                continue
            row_sums = matrix.sum(axis=1)
            np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_weights_sum_to_1(self, simple_matrix):
        result = generate_scenarios(simple_matrix)
        weights = result['_weights']
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestWeightedScenarioMatrix:

    def test_weighted_rows_sum_to_1(self, simple_matrix):
        scenarios = generate_scenarios(simple_matrix)
        result = weighted_scenario_matrix(scenarios)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_equal_weights_returns_average(self, simple_matrix):
        matrices = {
            'a': simple_matrix,
            'b': simple_matrix
        }
        result = weighted_scenario_matrix(matrices, weights={'a': 0.5, 'b': 0.5})
        pd.testing.assert_frame_equal(result, simple_matrix, atol=1e-10)

    def test_single_scenario_returns_itself(self, simple_matrix):
        result = weighted_scenario_matrix(
            {'only': simple_matrix},
            weights={'only': 1.0}
        )
        pd.testing.assert_frame_equal(result, simple_matrix, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
