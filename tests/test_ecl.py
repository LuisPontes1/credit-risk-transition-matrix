"""Tests for the ECL Calculator module."""

import pytest
import pandas as pd
import numpy as np

from credit_risk_transition_matrix.core.ecl_calculator import (
    calculate_ecl,
    ecl_by_stage,
    portfolio_ecl
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


@pytest.fixture
def staged_portfolio():
    """Portfolio with stage classification."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'id': [f'C{i}' for i in range(n)],
        'bucket': np.random.choice(
            ['Good', 'Bad', 'Default'], size=n, p=[0.6, 0.3, 0.1]
        ),
        'stage': np.nan,  # Will be set below
        'ead': np.random.uniform(5000, 50000, size=n).round(2)
    }).assign(
        stage=lambda df: df['bucket'].map({'Good': 1, 'Bad': 2, 'Default': 3})
    )


class TestCalculateECL:

    def test_basic_ecl(self):
        pd_vals = pd.Series([0.05, 0.20, 1.0])
        ead_vals = pd.Series([10000, 20000, 5000])
        ecl = calculate_ecl(pd_vals, ead_vals, lgd=0.45)
        
        expected = pd_vals * 0.45 * ead_vals
        pd.testing.assert_series_equal(ecl, expected)

    def test_ecl_non_negative(self):
        pd_vals = pd.Series([0.0, 0.5, 1.0])
        ead_vals = pd.Series([10000, 20000, 5000])
        ecl = calculate_ecl(pd_vals, ead_vals, lgd=0.45)
        assert (ecl >= 0).all()

    def test_discount_reduces_ecl(self):
        pd_vals = pd.Series([0.10])
        ead_vals = pd.Series([10000])
        
        ecl_no_discount = calculate_ecl(pd_vals, ead_vals, lgd=0.45, discount_rate=0.0)
        ecl_discounted = calculate_ecl(pd_vals, ead_vals, lgd=0.45, discount_rate=0.10)
        
        assert ecl_discounted.iloc[0] < ecl_no_discount.iloc[0]

    def test_zero_pd_gives_zero_ecl(self):
        pd_vals = pd.Series([0.0])
        ead_vals = pd.Series([10000])
        ecl = calculate_ecl(pd_vals, ead_vals, lgd=0.45)
        assert ecl.iloc[0] == 0.0


class TestECLByStage:

    def test_output_has_ecl_column(self, simple_matrix, staged_portfolio):
        result = ecl_by_stage(
            simple_matrix,
            staged_portfolio,
            bucket_col='bucket',
            stage_col='stage',
            ead_col='ead'
        )
        assert 'ecl' in result.columns
        assert 'pd_used' in result.columns
        assert 'lgd_used' in result.columns

    def test_stage3_pd_is_1(self, simple_matrix, staged_portfolio):
        result = ecl_by_stage(
            simple_matrix,
            staged_portfolio,
            bucket_col='bucket',
            stage_col='stage',
            ead_col='ead'
        )
        stage3 = result[result['stage'] == 3]
        if len(stage3) > 0:
            np.testing.assert_allclose(stage3['pd_used'].values, 1.0)

    def test_ecl_non_negative(self, simple_matrix, staged_portfolio):
        result = ecl_by_stage(
            simple_matrix,
            staged_portfolio,
            bucket_col='bucket',
            stage_col='stage',
            ead_col='ead'
        )
        assert (result['ecl'] >= 0).all()

    def test_no_rows_lost(self, simple_matrix, staged_portfolio):
        result = ecl_by_stage(
            simple_matrix,
            staged_portfolio,
            bucket_col='bucket',
            stage_col='stage',
            ead_col='ead'
        )
        assert len(result) == len(staged_portfolio)

    def test_custom_lgd(self, simple_matrix, staged_portfolio):
        result_low = ecl_by_stage(
            simple_matrix, staged_portfolio,
            bucket_col='bucket', stage_col='stage', ead_col='ead',
            lgd=0.20
        )
        result_high = ecl_by_stage(
            simple_matrix, staged_portfolio,
            bucket_col='bucket', stage_col='stage', ead_col='ead',
            lgd=0.80
        )
        assert result_high['ecl'].sum() > result_low['ecl'].sum()


class TestPortfolioECL:

    def test_summary_structure(self, simple_matrix, staged_portfolio):
        ecl_df = ecl_by_stage(
            simple_matrix, staged_portfolio,
            bucket_col='bucket', stage_col='stage', ead_col='ead'
        )
        result = portfolio_ecl(ecl_df)
        
        assert 'count' in result.columns
        assert 'ead_total' in result.columns
        assert 'ecl_total' in result.columns
        assert 'coverage_ratio' in result.columns

    def test_total_row_exists(self, simple_matrix, staged_portfolio):
        ecl_df = ecl_by_stage(
            simple_matrix, staged_portfolio,
            bucket_col='bucket', stage_col='stage', ead_col='ead'
        )
        result = portfolio_ecl(ecl_df)
        assert 'Total' in result['stage'].values

    def test_coverage_between_0_and_1(self, simple_matrix, staged_portfolio):
        ecl_df = ecl_by_stage(
            simple_matrix, staged_portfolio,
            bucket_col='bucket', stage_col='stage', ead_col='ead'
        )
        result = portfolio_ecl(ecl_df)
        total_row = result[result['stage'] == 'Total']
        coverage = total_row['coverage_ratio'].iloc[0]
        assert 0 <= coverage <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
