"""
Tests for SICR staging, forward-looking, maturity engine, and backtesting.
"""

import pytest
import numpy as np
import pandas as pd

from credit_risk_transition_matrix.core.staging import classify_stage_sicr
from credit_risk_transition_matrix.core.forward_looking import (
    MacroScenario,
    DEFAULT_SCENARIOS,
    validate_scenarios,
    apply_macro_overlay,
    scenario_ecl_breakdown,
)
from credit_risk_transition_matrix.core.maturity_engine import (
    contractual_maturity,
    behavioral_maturity,
    effective_maturity,
)
from credit_risk_transition_matrix.core.backtesting import (
    backtest_pd,
    backtest_lgd,
    backtest_ecl,
    discrimination_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SICR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyStageSICR:
    """Test PD-based SICR staging."""
    
    def _base_df(self):
        return pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'pd_app': [0.02, 0.02, 0.01, 0.005, 0.001, 0.03],
            'pd_bhv': [0.02, 0.05, 0.05, 0.005, 0.002, 0.10],
            'dpd':    [0,    0,    0,    0,     0,     120],
        })
    
    def test_no_sicr_stays_stage1(self):
        df = self._base_df()
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd'
        )
        # Contract A: PD unchanged → Stage 1
        assert result.loc[0, 'stage'] == 1
    
    def test_relative_trigger(self):
        df = self._base_df()
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd', sicr_relative_threshold=2.0
        )
        # Contract B: PD went from 0.02 → 0.05 (2.5x), triggers relative SICR
        assert result.loc[1, 'stage'] == 2
        assert result.loc[1, 'sicr_trigger'] == 'relative'
    
    def test_absolute_trigger(self):
        df = self._base_df()
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd', sicr_absolute_threshold=0.03
        )
        # Contract C: PD went from 0.01 → 0.05, delta=0.04 >= 0.03
        assert result.loc[2, 'stage'] == 2
    
    def test_dpd_backstop(self):
        df = pd.DataFrame({
            'pd_app': [0.02], 'pd_bhv': [0.02], 'dpd': [45]
        })
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd'
        )
        assert result.iloc[0]['stage'] == 2
        assert result.iloc[0]['sicr_trigger'] == 'dpd_backstop'
    
    def test_stage3_for_default(self):
        df = self._base_df()
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd'
        )
        # Contract F: DPD=120 → Stage 3
        assert result.loc[5, 'stage'] == 3
    
    def test_low_risk_exemption(self):
        df = self._base_df()
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd', low_risk_exemption=0.003
        )
        # Contract E: PD=0.002 < 0.003 → low-risk exempt, stays Stage 1
        # even though PD doubled (0.001 → 0.002)
        assert result.loc[4, 'stage'] == 1
    
    def test_low_risk_exemption_disabled(self):
        df = pd.DataFrame({
            'pd_app': [0.001], 'pd_bhv': [0.003], 'dpd': [0]
        })
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd', low_risk_exemption=0.0,  # disabled
            sicr_relative_threshold=2.0
        )
        # PD tripled → should trigger without exemption
        assert result.iloc[0]['stage'] == 2
    
    def test_default_flag_overrides_dpd(self):
        df = pd.DataFrame({
            'pd_app': [0.02], 'pd_bhv': [0.02],
            'dpd': [0], 'is_default': [True]
        })
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
            dpd_col='dpd', default_flag_col='is_default'
        )
        assert result.iloc[0]['stage'] == 3
    
    def test_output_has_sicr_trigger(self):
        df = self._base_df()
        result = classify_stage_sicr(
            df, pd_current_col='pd_bhv', pd_origination_col='pd_app',
        )
        assert 'sicr_trigger' in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD-LOOKING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMacroScenario:
    """Test MacroScenario dataclass."""
    
    def test_valid_creation(self):
        s = MacroScenario('test', 0.5, 1.2, 1.1)
        assert s.name == 'test'
        assert s.weight == 0.5
    
    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="Weight"):
            MacroScenario('bad', 1.5, 1.0, 1.0)
    
    def test_negative_pd_adjustment_raises(self):
        with pytest.raises(ValueError, match="pd_adjustment"):
            MacroScenario('bad', 0.5, -0.5, 1.0)


class TestValidateScenarios:
    def test_default_scenarios_valid(self):
        validate_scenarios(DEFAULT_SCENARIOS)
    
    def test_weights_must_sum_to_1(self):
        scenarios = [
            MacroScenario('a', 0.3, 1.0, 1.0),
            MacroScenario('b', 0.3, 1.0, 1.0),
        ]
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_scenarios(scenarios)
    
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            validate_scenarios([])


class TestApplyMacroOverlay:
    def test_neutral_scenarios_no_change(self):
        """Scenarios with all adj=1.0 should return original values."""
        neutral = [MacroScenario('neutral', 1.0, 1.0, 1.0)]
        pd_vals = pd.Series([0.05, 0.10, 0.20])
        lgd_vals = pd.Series([0.40, 0.50, 0.60])
        
        pd_adj, lgd_adj = apply_macro_overlay(pd_vals, lgd_vals, neutral)
        pd.testing.assert_series_equal(pd_adj, pd_vals, check_names=False)
        pd.testing.assert_series_equal(lgd_adj, lgd_vals, check_names=False)
    
    def test_pessimistic_increases_pd(self):
        pessimistic = [MacroScenario('worst', 1.0, 1.5, 1.3)]
        pd_vals = pd.Series([0.10])
        lgd_vals = pd.Series([0.40])
        
        pd_adj, lgd_adj = apply_macro_overlay(pd_vals, lgd_vals, pessimistic)
        assert pd_adj.iloc[0] > pd_vals.iloc[0]
        assert lgd_adj.iloc[0] > lgd_vals.iloc[0]
    
    def test_pd_capped_at_1(self):
        extreme = [MacroScenario('extreme', 1.0, 10.0, 1.0)]
        pd_vals = pd.Series([0.50])
        lgd_vals = pd.Series([0.40])
        
        pd_adj, _ = apply_macro_overlay(pd_vals, lgd_vals, extreme)
        assert pd_adj.iloc[0] <= 1.0
    
    def test_default_scenarios_produce_reasonable_change(self):
        pd_vals = pd.Series([0.05, 0.10])
        lgd_vals = pd.Series([0.45, 0.45])
        
        pd_adj, lgd_adj = apply_macro_overlay(pd_vals, lgd_vals)
        # With default scenarios (opt 0.8, base 1.0, pess 1.4),
        # weighted avg adjustment = 0.25*0.8 + 0.50*1.0 + 0.25*1.4 = 1.05
        assert (pd_adj > pd_vals * 0.99).all()
        assert (pd_adj < pd_vals * 1.10).all()
    
    def test_scalar_lgd(self):
        pd_vals = pd.Series([0.05, 0.10])
        pd_adj, lgd_adj = apply_macro_overlay(pd_vals, 0.45)
        assert len(lgd_adj) == 2


class TestScenarioECLBreakdown:
    def test_breakdown_structure(self):
        pd_vals = pd.Series([0.05, 0.10])
        lgd_vals = pd.Series([0.45, 0.45])
        ead_vals = pd.Series([10000, 20000])
        
        result = scenario_ecl_breakdown(pd_vals, lgd_vals, ead_vals)
        assert 'TOTAL_WEIGHTED' in result['scenario'].values
        assert len(result) == len(DEFAULT_SCENARIOS) + 1
        assert result.iloc[-1]['ecl_weighted'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# MATURITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContractualMaturity:
    def test_basic(self):
        df = pd.DataFrame({
            'dt_vencimento': ['2025-12-31', '2026-06-30'],
            'data_ref': ['2025-01-01', '2025-01-01']
        })
        mat = contractual_maturity(df, 'dt_vencimento', 'data_ref')
        assert mat.iloc[0] == pytest.approx(12.0, abs=1.0)
        assert mat.iloc[1] == pytest.approx(18.0, abs=1.0)
    
    def test_expired_contract_zero(self):
        df = pd.DataFrame({
            'dt_vencimento': ['2024-01-01'],
            'data_ref': ['2025-01-01']
        })
        mat = contractual_maturity(df, 'dt_vencimento', 'data_ref')
        assert mat.iloc[0] == 0.0


class TestBehavioralMaturity:
    def test_balance_halflife(self):
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        df = pd.DataFrame({
            'id': ['A'] * 12,
            'data_ref': dates,
            'saldo': [1000 - i * 80 for i in range(12)]  # steady decrease
        })
        result = behavioral_maturity(df, 'id', 'data_ref', 'saldo',
                                     method='balance_halflife')
        assert len(result) == 1
        assert result.iloc[0]['behavioral_maturity'] > 0
    
    def test_median_tenure(self):
        dates = pd.date_range('2023-01-01', periods=6, freq='MS')
        df = pd.DataFrame({
            'id': ['A'] * 6 + ['B'] * 3,
            'data_ref': list(dates) + list(dates[:3]),
            'saldo': [1000] * 6 + [500] * 3,
        })
        result = behavioral_maturity(df, 'id', 'data_ref', 'saldo',
                                     method='median_tenure')
        assert len(result) == 2
    
    def test_invalid_method_raises(self):
        df = pd.DataFrame({
            'id': ['A'], 'data_ref': ['2023-01-01'], 'saldo': [1000]
        })
        with pytest.raises(ValueError, match="Unknown method"):
            behavioral_maturity(df, 'id', 'data_ref', 'saldo', method='invalid')


class TestEffectiveMaturity:
    def test_applies_floor(self):
        df = pd.DataFrame({'mat': [3.0, 6.0, 24.0]})
        result = effective_maturity(df, contractual_col='mat', floor_months=12)
        assert (result >= 12.0).all()
    
    def test_applies_cap(self):
        df = pd.DataFrame({'mat': [100.0, 200.0]})
        result = effective_maturity(df, contractual_col='mat', cap_months=60)
        assert (result <= 60.0).all()
    
    def test_no_column_uses_floor(self):
        df = pd.DataFrame({'other': [1, 2, 3]})
        result = effective_maturity(df, floor_months=12)
        assert (result == 12.0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestPD:
    def test_output_structure(self):
        rng = np.random.RandomState(42)
        pd_vals = pd.Series(rng.uniform(0, 0.3, 200))
        defaults = pd.Series((rng.random(200) < pd_vals).astype(int))
        
        result = backtest_pd(pd_vals, defaults)
        assert 'predicted_dr' in result.columns
        assert 'observed_dr' in result.columns
        assert 'ratio' in result.columns
    
    def test_perfect_model_ratio_near_1(self):
        rng = np.random.RandomState(42)
        pd_vals = pd.Series(rng.uniform(0.01, 0.3, 1000))
        defaults = pd.Series((rng.random(1000) < pd_vals).astype(int))
        
        result = backtest_pd(pd_vals, defaults, n_bins=5)
        # Ratio should be reasonably close to 1 for a well-calibrated model
        assert result['ratio'].dropna().mean() == pytest.approx(1.0, abs=0.5)


class TestBacktestLGD:
    def test_output(self):
        pred = pd.Series([0.40, 0.50, 0.60, 0.70, 0.45])
        actual = pd.Series([0.35, 0.55, 0.58, 0.72, 0.40])
        result = backtest_lgd(pred, actual)
        assert 'bias' in result.columns
    
    def test_empty_input(self):
        result = backtest_lgd(pd.Series(dtype=float), pd.Series(dtype=float))
        assert len(result) == 0


class TestBacktestECL:
    def test_basic(self):
        pred = pd.Series([100, 200, 50])
        actual = pd.Series([90, 220, 60])
        result = backtest_ecl(pred, actual)
        assert result.iloc[0]['group'] == 'ALL'
        assert result.iloc[0]['ecl_predicted'] > 0
    
    def test_by_stage(self):
        pred = pd.Series([100, 200, 50])
        actual = pd.Series([90, 220, 60])
        stages = pd.Series([1, 2, 3])
        result = backtest_ecl(pred, actual, stage_col=stages)
        assert len(result) == 4  # ALL + 3 stages


class TestDiscriminationMetrics:
    def test_good_model_high_auc(self):
        rng = np.random.RandomState(42)
        n = 500
        defaults = pd.Series(rng.binomial(1, 0.1, n))
        pd_vals = pd.Series(
            np.where(defaults == 1, rng.uniform(0.3, 0.9, n),
                     rng.uniform(0.01, 0.2, n))
        )
        
        metrics = discrimination_metrics(pd_vals, defaults)
        assert metrics['auc'] > 0.7
        assert metrics['gini'] > 0.4
        assert metrics['ks'] > 0.3
    
    def test_random_model_auc_near_05(self):
        rng = np.random.RandomState(42)
        n = 500
        defaults = pd.Series(rng.binomial(1, 0.1, n))
        pd_vals = pd.Series(rng.uniform(0, 1, n))
        
        metrics = discrimination_metrics(pd_vals, defaults)
        assert 0.3 < metrics['auc'] < 0.7
    
    def test_no_defaults_returns_nan(self):
        pd_vals = pd.Series([0.1, 0.2, 0.3])
        defaults = pd.Series([0, 0, 0])
        metrics = discrimination_metrics(pd_vals, defaults)
        assert np.isnan(metrics['auc'])
