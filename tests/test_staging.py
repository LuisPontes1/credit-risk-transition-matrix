"""Tests for the Staging module (IFRS 9)."""

import pytest
import pandas as pd
import numpy as np

from credit_risk_transition_matrix.core.staging import (
    classify_stage,
    stage_migration_matrix,
    stage_summary
)


@pytest.fixture
def sample_portfolio():
    """Portfolio with known bucket distribution."""
    labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
    np.random.seed(42)
    
    n = 200
    data = {
        'id_contrato': [f'C{i:03d}' for i in range(n)],
        'data_ref': pd.Timestamp('2024-01-31'),
        'bucket': np.random.choice(labels, size=n, p=[
            0.50, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03
        ]),
        'ead': np.random.uniform(1000, 100000, size=n).round(2)
    }
    return pd.DataFrame(data)


@pytest.fixture
def panel_data():
    """Panel data with multiple periods for migration analysis."""
    labels = ['0-14', '15-29', '30-59', '60-89', '360+']
    
    records = []
    for cid in range(50):
        buckets = np.random.choice(labels, size=6, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        for t, bucket in enumerate(buckets):
            records.append({
                'id_contrato': f'C{cid:03d}',
                'data_ref': pd.Timestamp('2024-01-31') + pd.DateOffset(months=t),
                'bucket': bucket,
                'ead': 10000.0
            })
    
    return pd.DataFrame(records)


class TestClassifyStage:

    def test_basic_classification(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
        result = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels,
            stage1_max_bucket='15-29',
            default_buckets=['360+']
        )
        
        assert 'stage' in result.columns
        assert set(result['stage'].unique()).issubset({1, 2, 3})

    def test_stage3_is_default(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
        result = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels,
            default_buckets=['360+']
        )
        
        stage3 = result[result['stage'] == 3]
        assert all(stage3['bucket'] == '360+')

    def test_stage1_within_threshold(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
        result = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels,
            stage1_max_bucket='15-29',
            default_buckets=['360+']
        )
        
        stage1 = result[result['stage'] == 1]
        assert all(stage1['bucket'].isin(['0-14', '15-29']))

    def test_all_stages_present(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
        result = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels,
            stage1_max_bucket='15-29',
            default_buckets=['360+']
        )
        assert 1 in result['stage'].values
        assert 2 in result['stage'].values
        assert 3 in result['stage'].values

    def test_no_rows_lost(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '360+']
        result = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels
        )
        assert len(result) == len(sample_portfolio)


class TestStageMigrationMatrix:

    def test_output_shape(self, panel_data):
        labels = ['0-14', '15-29', '30-59', '60-89', '360+']
        staged = classify_stage(
            panel_data,
            bucket_col='bucket',
            bucket_labels=labels,
            default_buckets=['360+']
        )
        result = stage_migration_matrix(staged)
        assert result.shape == (3, 3)

    def test_rows_sum_to_1(self, panel_data):
        labels = ['0-14', '15-29', '30-59', '60-89', '360+']
        staged = classify_stage(
            panel_data,
            bucket_col='bucket',
            bucket_labels=labels,
            default_buckets=['360+']
        )
        result = stage_migration_matrix(staged)
        row_sums = result.sum(axis=1)
        for val in row_sums:
            if val > 0:
                assert abs(val - 1.0) < 0.01


class TestStageSummary:

    def test_basic_summary(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
        staged = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels,
            default_buckets=['360+']
        )
        result = stage_summary(staged, ead_col='ead')
        
        assert len(result) == 3  # Stage 1, 2, 3
        assert 'count' in result.columns
        assert 'ead_total' in result.columns
        assert 'pct_count' in result.columns

    def test_counts_match(self, sample_portfolio):
        labels = ['0-14', '15-29', '30-59', '60-89', '90-119', '120-179', '180-239', '240-359', '360+']
        staged = classify_stage(
            sample_portfolio,
            bucket_col='bucket',
            bucket_labels=labels,
            default_buckets=['360+']
        )
        result = stage_summary(staged)
        assert result['count'].sum() == len(sample_portfolio)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
