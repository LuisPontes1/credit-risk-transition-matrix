"""
Tests for the Product Schema module.
"""

import pytest
from credit_risk_transition_matrix.core.product_schema import (
    ProductSchema,
    CollateralConfig,
    REVOLVING,
    GIRO_TPV,
    GIRO_TRADITIONAL,
)


class TestCollateralConfig:
    """Test CollateralConfig dataclass."""
    
    def test_valid_creation(self):
        cc = CollateralConfig(
            name='tpv', ephemerality=0.8,
            base_lgd=0.20, stressed_lgd=0.90
        )
        assert cc.name == 'tpv'
        assert cc.ephemerality == 0.8
    
    def test_invalid_ephemerality(self):
        with pytest.raises(ValueError, match="Ephemerality"):
            CollateralConfig(name='x', ephemerality=1.5, base_lgd=0.2, stressed_lgd=0.9)
    
    def test_invalid_base_lgd(self):
        with pytest.raises(ValueError, match="base_lgd"):
            CollateralConfig(name='x', ephemerality=0.5, base_lgd=-0.1, stressed_lgd=0.9)


class TestProductSchema:
    """Test ProductSchema dataclass."""
    
    def test_revolving_schema(self):
        assert REVOLVING.name == 'revolving'
        assert REVOLVING.default_threshold == 90
        assert 'dpd' in REVOLVING.state_dimensions
        assert 'iu' in REVOLVING.state_dimensions
    
    def test_giro_tpv_schema(self):
        assert GIRO_TPV.name == 'giro_tpv'
        assert 'tpv' in GIRO_TPV.collateral_types
        assert 'imovel' in GIRO_TPV.collateral_types
    
    def test_bucket_labels(self):
        labels = REVOLVING.bucket_labels
        assert labels[0] == '0-14'
        assert labels[-1] == '360+'
    
    def test_default_buckets(self):
        defaults = REVOLVING.default_buckets
        assert '0-14' not in defaults
        assert '90-119' in defaults
        assert '360+' in defaults
    
    def test_performing_buckets(self):
        performing = REVOLVING.performing_buckets
        assert '0-14' in performing
        assert '360+' not in performing
    
    def test_invalid_default_threshold(self):
        with pytest.raises(ValueError, match="default_threshold"):
            ProductSchema(
                name='bad', buckets=[0, 30, 60, 90],
                default_threshold=45  # not in buckets
            )
    
    def test_empty_buckets_raises(self):
        with pytest.raises(ValueError, match="buckets must not be empty"):
            ProductSchema(name='bad', buckets=[], default_threshold=90)


class TestBlendedLGD:
    """Test blended LGD calculation."""
    
    def test_fully_clean(self):
        lgd = GIRO_TPV.compute_contract_lgd(pct_clean=1.0)
        assert lgd == GIRO_TPV.clean_lgd
    
    def test_fully_collateralized_tpv(self):
        lgd = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.0, collateral_type='tpv'
        )
        assert lgd == GIRO_TPV.collateral_types['tpv'].base_lgd
    
    def test_blended_lgd(self):
        """50% clean + 50% TPV should be between clean and TPV LGDs."""
        lgd = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.5, collateral_type='tpv'
        )
        tpv_lgd = GIRO_TPV.collateral_types['tpv'].base_lgd
        expected = 0.5 * GIRO_TPV.clean_lgd + 0.5 * tpv_lgd
        assert abs(lgd - expected) < 1e-4
    
    def test_stressed_lgd_higher(self):
        """Stressed (churn) should produce higher LGD."""
        normal = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.3, collateral_type='tpv', collateral_stressed=False
        )
        stressed = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.3, collateral_type='tpv', collateral_stressed=True
        )
        assert stressed > normal
    
    def test_imovel_more_stable_than_tpv(self):
        """Imóvel should have lower base LGD than TPV when stressed."""
        lgd_tpv = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.0, collateral_type='tpv', collateral_stressed=True
        )
        lgd_imovel = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.0, collateral_type='imovel', collateral_stressed=True
        )
        assert lgd_imovel < lgd_tpv
    
    def test_unknown_collateral_treated_as_clean(self):
        lgd = GIRO_TPV.compute_contract_lgd(
            pct_clean=0.5, collateral_type='unknown_type'
        )
        assert lgd == GIRO_TPV.clean_lgd


class TestIUBands:
    """Test utilization band assignment."""
    
    def test_low_utilization(self):
        assert REVOLVING.assign_iu_band(0.20) == 'low'
    
    def test_medium_utilization(self):
        assert REVOLVING.assign_iu_band(0.45) == 'medium'
    
    def test_high_utilization(self):
        assert REVOLVING.assign_iu_band(0.75) == 'high'
    
    def test_very_high_utilization(self):
        assert REVOLVING.assign_iu_band(0.95) == 'very_high'
    
    def test_over_limit(self):
        assert REVOLVING.assign_iu_band(1.15) == 'over_limit'
    
    def test_no_iu_bands_returns_na(self):
        assert GIRO_TPV.assign_iu_band(0.5) == 'n/a'
