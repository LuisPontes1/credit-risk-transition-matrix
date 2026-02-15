"""
Integration tests for ContractRiskEngine — full ECL pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from credit_risk_transition_matrix.core.contract_risk_engine import ContractRiskEngine
from credit_risk_transition_matrix.core.forward_looking import MacroScenario
from credit_risk_transition_matrix.core.product_schema import (
    REVOLVING, GIRO_TPV, GIRO_TRADITIONAL
)


def _make_giro_panel(n_contracts=60, n_periods=12, seed=42):
    """Simulate giro panel data for fitting."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='MS')
    records = []
    for cid in range(1, n_contracts + 1):
        dpd = 0
        ead = rng.uniform(5000, 50000)
        for t, date in enumerate(dates):
            records.append({
                'id': f'GIRO_{cid:03d}', 'data_ref': date,
                'dpd': dpd, 'saldo': round(ead, 2),
            })
            if dpd == 0:
                if rng.random() < 0.08: dpd = 30
            elif dpd == 30:
                r = rng.random()
                if r < 0.35: dpd = 0; ead *= 0.95
                elif r < 0.65: dpd = 60
            elif dpd == 60:
                r = rng.random()
                if r < 0.15: dpd = 0; ead *= 0.90
                elif r < 0.55: dpd = 90
            elif dpd >= 90:
                if dpd < 360: dpd = min(dpd + 30, 360)
    return pd.DataFrame(records)


def _make_snapshot(n=50, seed=99):
    """Create a realistic portfolio snapshot for rating."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'id': [f'C{i:03d}' for i in range(n)],
        'dpd': rng.choice([0, 0, 0, 0, 15, 30, 60, 90, 120, 360], n),
        'saldo': rng.uniform(1000, 50000, n).round(2),
        'pd_app': rng.uniform(0.005, 0.05, n).round(4),  # origination PD
        'pd_bhv': np.nan,  # will be computed from dpd
        'pct_clean': rng.uniform(0.2, 1.0, n).round(2),
        'garantia': rng.choice([None, 'tpv', 'imovel'], n),
    })


def _enrich_snapshot(snapshot):
    """Add behavioral PD based on DPD (simplistic but realistic)."""
    df = snapshot.copy()
    dpd = df['dpd'].astype(float)
    # PD increases with DPD
    base_pd = df['pd_app']
    factor = 1 + dpd / 30  # DPD 0→1x, 30→2x, 90→4x
    df['pd_bhv'] = (base_pd * factor).clip(upper=1.0).round(6)
    return df


class TestContractRiskEngineInit:
    def test_init(self):
        engine = ContractRiskEngine(GIRO_TPV)
        assert engine.schema.name == 'giro_tpv'
        assert not engine._is_fitted
    
    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            ContractRiskEngine("not_a_schema")
    
    def test_repr_unfitted(self):
        engine = ContractRiskEngine(GIRO_TPV)
        r = repr(engine)
        assert 'giro_tpv' in r
        assert 'not fitted' in r


class TestContractRiskEngineFit:
    @pytest.fixture
    def fitted_engine(self):
        df = _make_giro_panel()
        engine = ContractRiskEngine(GIRO_TPV)
        engine.fit(df, id_col='id', time_col='data_ref',
                   bucket_col='dpd', ead_col='saldo')
        return engine
    
    def test_fit_marks_as_fitted(self, fitted_engine):
        assert fitted_engine._is_fitted
    
    def test_lgd_engine_fitted(self, fitted_engine):
        assert fitted_engine.lgd_engine._is_fitted
    
    def test_repr_fitted(self, fitted_engine):
        r = repr(fitted_engine)
        assert 'fitted' in r
        assert 'not fitted' not in r


class TestComputeECL:
    @pytest.fixture
    def fitted_engine(self):
        df = _make_giro_panel()
        engine = ContractRiskEngine(GIRO_TPV)
        engine.fit(df, id_col='id', time_col='data_ref',
                   bucket_col='dpd', ead_col='saldo')
        return engine
    
    def test_ecl_pipeline_adds_all_columns(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot())
        result = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
            pct_clean_col='pct_clean',
            collateral_type_col='garantia',
        )
        expected_cols = [
            'stage', 'sicr_trigger', 'lgd_value', 'lgd_rating',
            'ead_computed', 'maturity_months', 'pd_adjusted',
            'lgd_adjusted', 'ecl', 'ecl_coverage'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_ecl_non_negative(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot())
        result = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
        )
        assert (result['ecl'] >= 0).all()
    
    def test_stage3_pd_is_1(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot())
        result = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
        )
        stage3 = result[result['stage'] == 3]
        if len(stage3) > 0:
            assert (stage3['pd_for_ecl'] == 1.0).all()
    
    def test_stage2_has_sicr_trigger(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot())
        result = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
        )
        stage2 = result[result['stage'] == 2]
        if len(stage2) > 0:
            assert stage2['sicr_trigger'].notna().all()
    
    def test_coverage_between_0_and_1(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot())
        result = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
        )
        coverage = result['ecl_coverage']
        assert (coverage >= 0).all()
        assert (coverage <= 1.01).all()
    
    def test_worse_dpd_higher_ecl(self, fitted_engine):
        """Contracts with higher DPD should generally have higher ECL."""
        snapshot = pd.DataFrame({
            'id': ['good', 'bad'],
            'dpd': [0, 120],
            'saldo': [10000, 10000],
            'pd_app': [0.02, 0.02],
            'pd_bhv': [0.02, 0.20],
        })
        result = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
        )
        ecl_good = result.loc[result['id'] == 'good', 'ecl'].iloc[0]
        ecl_bad = result.loc[result['id'] == 'bad', 'ecl'].iloc[0]
        assert ecl_bad > ecl_good
    
    def test_forward_looking_disabled(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot(n=10))
        result_with = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            dpd_col='dpd', bucket_col='dpd',
            apply_forward_looking=True,
        )
        result_without = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            dpd_col='dpd', bucket_col='dpd',
            apply_forward_looking=False,
        )
        # Forward-looking applies ~1.05x adjustment, so ECL should differ
        assert not np.allclose(result_with['ecl'], result_without['ecl'])
    
    def test_discount_reduces_ecl(self, fitted_engine):
        snapshot = _enrich_snapshot(_make_snapshot(n=10))
        result_no_disc = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            dpd_col='dpd', bucket_col='dpd',
            discount_rate=0.0,
        )
        result_disc = fitted_engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            dpd_col='dpd', bucket_col='dpd',
            discount_rate=0.10,
        )
        assert result_disc['ecl'].sum() <= result_no_disc['ecl'].sum()
    
    def test_unfitted_raises(self):
        engine = ContractRiskEngine(GIRO_TPV)
        snapshot = pd.DataFrame({
            'pd_bhv': [0.05], 'saldo': [10000], 'dpd': [0]
        })
        with pytest.raises(RuntimeError, match="must be fit"):
            engine.compute_ecl(snapshot, pd_col='pd_bhv', ead_col='saldo')


class TestECLSummary:
    def test_summary_structure(self):
        df = _make_giro_panel()
        engine = ContractRiskEngine(GIRO_TPV)
        engine.fit(df, id_col='id', time_col='data_ref',
                   bucket_col='dpd', ead_col='saldo')
        
        snapshot = _enrich_snapshot(_make_snapshot())
        ecl_df = engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            pd_origination_col='pd_app', dpd_col='dpd',
            bucket_col='dpd',
        )
        
        summary = engine.ecl_summary(ecl_df)
        assert len(summary) == 4  # S1, S2, S3, Total
        assert summary.iloc[-1]['stage'] == 'Total'
        assert summary.iloc[-1]['ecl_total'] > 0


class TestCustomScenarios:
    def test_custom_scenarios(self):
        df = _make_giro_panel()
        
        custom = [
            MacroScenario('boom', 0.30, 0.6, 0.7),
            MacroScenario('base', 0.40, 1.0, 1.0),
            MacroScenario('crisis', 0.30, 2.0, 1.5),
        ]
        
        engine = ContractRiskEngine(GIRO_TPV, macro_scenarios=custom)
        engine.fit(df, id_col='id', time_col='data_ref',
                   bucket_col='dpd', ead_col='saldo')
        
        snapshot = _enrich_snapshot(_make_snapshot(n=20))
        result = engine.compute_ecl(
            snapshot, pd_col='pd_bhv', ead_col='saldo',
            dpd_col='dpd', bucket_col='dpd',
        )
        assert (result['ecl'] >= 0).all()
