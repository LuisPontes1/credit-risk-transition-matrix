"""
Tests for the EAD Engine — CCF estimation and EAD computation.
"""

import pytest
import numpy as np
import pandas as pd

from credit_risk_transition_matrix.core.ead_engine import (
    estimate_ccf,
    ccf_by_segment,
    compute_ead,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_revolving_panel(n_contracts=50, n_periods=14, seed=42):
    """
    Simulate revolving panel data where some contracts draw down
    more limit before defaulting (high CCF pattern).
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='MS')
    
    records = []
    for cid in range(1, n_contracts + 1):
        limit = rng.uniform(5000, 30000)
        balance = limit * rng.uniform(0.2, 0.6)
        dpd = 0
        
        for t, date in enumerate(dates):
            bucket = (
                '0-14' if dpd < 15 else
                '15-29' if dpd < 30 else
                '30-59' if dpd < 60 else
                '60-89' if dpd < 90 else
                '90+'
            )
            
            records.append({
                'id': f'REV_{cid:03d}',
                'data_ref': date,
                'saldo': round(balance, 2),
                'limite': round(limit, 2),
                'bucket': bucket,
                'produto': 'cartao'
            })
            
            # Transition
            if dpd == 0:
                if rng.random() < 0.08:
                    dpd = 15
                    balance = min(balance * 1.1, limit)  # draws more
            elif dpd < 30:
                if rng.random() < 0.3:
                    dpd = 0
                    balance *= 0.9
                else:
                    dpd = 30
                    balance = min(balance * 1.15, limit)  # draws even more
            elif dpd < 60:
                if rng.random() < 0.1:
                    dpd = 0
                    balance *= 0.8
                else:
                    dpd = 60
                    balance = min(balance * 1.05, limit)
            elif dpd < 90:
                dpd = 90
            else:
                dpd = min(dpd + 30, 360)
    
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — CCF estimation
# ──────────────────────────────────────────────────────────────────────────────

class TestEstimateCCF:
    """Test empirical CCF estimation."""
    
    def test_ccf_basic(self):
        df = _make_revolving_panel()
        ccf_df = estimate_ccf(
            df, id_col='id', time_col='data_ref',
            balance_col='saldo', limit_col='limite',
            bucket_col='bucket', default_states=['90+']
        )
        assert isinstance(ccf_df, pd.DataFrame)
        assert 'ccf' in ccf_df.columns
    
    def test_ccf_between_0_and_1(self):
        df = _make_revolving_panel()
        ccf_df = estimate_ccf(
            df, id_col='id', time_col='data_ref',
            balance_col='saldo', limit_col='limite',
            bucket_col='bucket', default_states=['90+']
        )
        if not ccf_df.empty:
            assert (ccf_df['ccf'] >= 0).all()
            assert (ccf_df['ccf'] <= 1).all()
    
    def test_ccf_has_default_info(self):
        df = _make_revolving_panel()
        ccf_df = estimate_ccf(
            df, id_col='id', time_col='data_ref',
            balance_col='saldo', limit_col='limite',
            bucket_col='bucket', default_states=['90+']
        )
        if not ccf_df.empty:
            assert 'balance_at_default' in ccf_df.columns
            assert 'balance_pre_default' in ccf_df.columns
            assert 'contract_id' in ccf_df.columns
    
    def test_no_defaults_returns_empty(self):
        """If no contracts default, return empty DataFrame."""
        df = pd.DataFrame({
            'id': ['A'] * 5,
            'data_ref': pd.date_range('2023-01-01', periods=5, freq='MS'),
            'saldo': [1000] * 5,
            'limite': [5000] * 5,
            'bucket': ['0-14'] * 5
        })
        ccf_df = estimate_ccf(
            df, id_col='id', time_col='data_ref',
            balance_col='saldo', limit_col='limite',
            bucket_col='bucket', default_states=['90+']
        )
        assert len(ccf_df) == 0


class TestCCFBySegment:
    """Test CCF aggregation."""
    
    def test_overall_aggregation(self):
        ccf_data = pd.DataFrame({
            'ccf': [0.5, 0.7, 0.3, 0.8],
            'segment': ['cartao', 'cartao', 'cheque', 'cheque']
        })
        result = ccf_by_segment(ccf_data)
        assert len(result) == 1
        assert result.iloc[0]['segment'] == 'ALL'
    
    def test_by_segment(self):
        ccf_data = pd.DataFrame({
            'ccf': [0.5, 0.7, 0.3, 0.8],
            'segment': ['cartao', 'cartao', 'cheque', 'cheque']
        })
        result = ccf_by_segment(ccf_data, segment_col='segment')
        assert len(result) == 2
    
    def test_empty_input(self):
        result = ccf_by_segment(pd.DataFrame(columns=['ccf']))
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests — EAD computation
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeEAD:
    """Test EAD calculation."""
    
    def test_term_ead_equals_balance(self):
        """Term loan EAD = outstanding balance."""
        snapshot = pd.DataFrame({
            'saldo': [10000, 5000, 2000],
            'limite': [10000, 10000, 10000]
        })
        ead = compute_ead(snapshot, balance_col='saldo', product_type='term')
        pd.testing.assert_series_equal(ead, snapshot['saldo'], check_names=False)
    
    def test_revolving_ead_greater_than_balance(self):
        """Revolving EAD should be >= balance (adds CCF × undrawn)."""
        snapshot = pd.DataFrame({
            'saldo': [3000, 7000, 10000],
            'limite': [10000, 10000, 10000]
        })
        ead = compute_ead(
            snapshot, balance_col='saldo', limit_col='limite',
            ccf=0.75, product_type='revolving'
        )
        assert (ead >= snapshot['saldo']).all()
    
    def test_revolving_fully_drawn_equals_balance(self):
        """If fully drawn (balance=limit), EAD = balance."""
        snapshot = pd.DataFrame({
            'saldo': [10000],
            'limite': [10000]
        })
        ead = compute_ead(
            snapshot, balance_col='saldo', limit_col='limite',
            ccf=0.75, product_type='revolving'
        )
        assert ead.iloc[0] == 10000
    
    def test_revolving_ccf_0_equals_balance(self):
        """CCF=0 means no additional drawdown expected."""
        snapshot = pd.DataFrame({
            'saldo': [3000],
            'limite': [10000]
        })
        ead = compute_ead(
            snapshot, balance_col='saldo', limit_col='limite',
            ccf=0.0, product_type='revolving'
        )
        assert ead.iloc[0] == 3000
    
    def test_revolving_ccf_1_equals_limit(self):
        """CCF=1 means full drawdown expected → EAD = limit."""
        snapshot = pd.DataFrame({
            'saldo': [3000],
            'limite': [10000]
        })
        ead = compute_ead(
            snapshot, balance_col='saldo', limit_col='limite',
            ccf=1.0, product_type='revolving'
        )
        assert ead.iloc[0] == 10000
    
    def test_revolving_per_contract_ccf(self):
        """Support per-contract CCF from a column."""
        snapshot = pd.DataFrame({
            'saldo': [3000, 5000],
            'limite': [10000, 10000],
            'ccf_col': [0.5, 0.8]
        })
        ead = compute_ead(
            snapshot, balance_col='saldo', limit_col='limite',
            ccf='ccf_col', product_type='revolving'
        )
        # Contract 1: 3000 + 0.5 * 7000 = 6500
        assert abs(ead.iloc[0] - 6500) < 1
        # Contract 2: 5000 + 0.8 * 5000 = 9000
        assert abs(ead.iloc[1] - 9000) < 1
    
    def test_negative_balance_clipped(self):
        """Negative balances should be clipped to 0."""
        snapshot = pd.DataFrame({'saldo': [-100, 0, 500]})
        ead = compute_ead(snapshot, balance_col='saldo', product_type='term')
        assert (ead >= 0).all()
    
    def test_invalid_product_type_raises(self):
        snapshot = pd.DataFrame({'saldo': [1000]})
        with pytest.raises(ValueError, match="Unknown product_type"):
            compute_ead(snapshot, balance_col='saldo', product_type='invalid')
