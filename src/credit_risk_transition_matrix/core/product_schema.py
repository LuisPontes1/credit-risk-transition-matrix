"""
Product Schema — Define product-specific configurations for LGD analysis.

Each product type (revolving credit, term loans) has different:
- DPD bucket boundaries
- Default threshold
- Collateral types and their ephemerality
- State dimensions (DPD only, or DPD × utilization)

Provides predefined schemas for common products and a framework
for creating custom product configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollateralConfig:
    """
    Configuration for a collateral type.
    
    Parameters
    ----------
    name : str
        Collateral type name (e.g., 'tpv', 'imovel', 'recebiveis').
    ephemerality : float
        Index 0.0–1.0 indicating how volatile/ephemeral the collateral is.
        0.0 = very stable (real estate), 1.0 = fully ephemeral (can vanish).
    base_lgd : float
        Base LGD when this collateral is present and active.
    stressed_lgd : float
        LGD when collateral is compromised (e.g., churn for TPV).
    description : str
        Human-readable description.
    """
    name: str
    ephemerality: float
    base_lgd: float
    stressed_lgd: float
    description: str = ""
    
    def __post_init__(self):
        if not 0.0 <= self.ephemerality <= 1.0:
            raise ValueError(
                f"Ephemerality must be between 0.0 and 1.0, got {self.ephemerality}"
            )
        if not 0.0 <= self.base_lgd <= 1.0:
            raise ValueError(
                f"base_lgd must be between 0.0 and 1.0, got {self.base_lgd}"
            )
        if not 0.0 <= self.stressed_lgd <= 1.0:
            raise ValueError(
                f"stressed_lgd must be between 0.0 and 1.0, got {self.stressed_lgd}"
            )


@dataclass
class ProductSchema:
    """
    Define how to process and analyze a specific product type for LGD.
    
    Parameters
    ----------
    name : str
        Product identifier (e.g., 'revolving', 'giro_tpv').
    buckets : List[int]
        DPD bucket boundaries (days past due thresholds).
    default_threshold : int
        DPD value at which a contract is considered in default.
    state_dimensions : List[str]
        Columns that define the "state" for transition analysis.
        Default: ['dpd']. For revolving: ['dpd', 'iu'].
    collateral_types : Dict[str, CollateralConfig]
        Available collateral types and their configurations.
    clean_lgd : float
        LGD for the unsecured (clean) portion of exposure.
    iu_bands : List[Tuple[float, str]], optional
        Utilization bands for revolving products.
        Format: [(threshold, label), ...].
        Example: [(0.3, 'low'), (0.6, 'medium'), (0.8, 'high'), (1.0, 'very_high'), (float('inf'), 'over')]
    description : str
        Human-readable description of the product.
    """
    name: str
    buckets: List[int]
    default_threshold: int
    state_dimensions: List[str] = field(default_factory=lambda: ['dpd'])
    collateral_types: Dict[str, CollateralConfig] = field(default_factory=dict)
    clean_lgd: float = 0.85
    iu_bands: Optional[List[Tuple[float, str]]] = None
    description: str = ""
    
    def __post_init__(self):
        if not self.buckets:
            raise ValueError("buckets must not be empty")
        if self.default_threshold not in self.buckets:
            raise ValueError(
                f"default_threshold ({self.default_threshold}) must be "
                f"one of the bucket boundaries: {self.buckets}"
            )
    
    @property
    def bucket_labels(self) -> List[str]:
        """Generate human-readable labels from bucket boundaries."""
        labels = []
        for i in range(len(self.buckets)):
            low = self.buckets[i]
            if i < len(self.buckets) - 1:
                high = self.buckets[i + 1] - 1
                labels.append(f"{low}-{high}")
            else:
                labels.append(f"{low}+")
        return labels
    
    @property
    def default_buckets(self) -> List[str]:
        """Bucket labels at or above default threshold."""
        labels = self.bucket_labels
        default_idx = self.buckets.index(self.default_threshold)
        return labels[default_idx:]
    
    @property
    def performing_buckets(self) -> List[str]:
        """Bucket labels below default threshold."""
        labels = self.bucket_labels
        default_idx = self.buckets.index(self.default_threshold)
        return labels[:default_idx]
    
    def compute_contract_lgd(
        self,
        pct_clean: float,
        collateral_type: Optional[str] = None,
        collateral_stressed: bool = False
    ) -> float:
        """
        Compute blended LGD for a single contract based on collateral composition.
        
        LGD = pct_clean × clean_lgd + (1 - pct_clean) × collateral_lgd
        
        Parameters
        ----------
        pct_clean : float
            Fraction of exposure without collateral (0.0–1.0).
        collateral_type : str, optional
            Type of collateral for the secured portion.
        collateral_stressed : bool, default=False
            If True, uses stressed LGD (e.g., churn scenario for TPV).
            
        Returns
        -------
        float
            Blended LGD value (0.0–1.0).
        """
        pct_clean = max(0.0, min(1.0, pct_clean))
        
        if pct_clean >= 1.0 or collateral_type is None:
            return self.clean_lgd
        
        if collateral_type not in self.collateral_types:
            logger.warning(
                "Unknown collateral type '%s', treating as clean",
                collateral_type
            )
            return self.clean_lgd
        
        config = self.collateral_types[collateral_type]
        
        if collateral_stressed:
            collateral_lgd = config.stressed_lgd
        else:
            collateral_lgd = config.base_lgd
        
        blended = pct_clean * self.clean_lgd + (1.0 - pct_clean) * collateral_lgd
        
        return round(min(1.0, max(0.0, blended)), 6)
    
    def assign_iu_band(self, iu: float) -> str:
        """
        Assign utilization band label for revolving products.
        
        Parameters
        ----------
        iu : float
            Utilization index (saldo / limite).
            
        Returns
        -------
        str
            Band label (e.g., 'low', 'medium', 'high').
        """
        if self.iu_bands is None:
            return 'n/a'
        
        for threshold, label in self.iu_bands:
            if iu <= threshold:
                return label
        
        return self.iu_bands[-1][1]  # fallback to last band


# ──────────────────────────────────────────────────────────────────────────────
# Predefined Schemas
# ──────────────────────────────────────────────────────────────────────────────

REVOLVING = ProductSchema(
    name='revolving',
    description='Cartão de crédito / Cheque especial — cohort fatura × IU',
    buckets=[0, 15, 30, 60, 90, 120, 180, 360],
    default_threshold=90,
    state_dimensions=['dpd', 'iu'],
    collateral_types={},  # revolving is typically unsecured
    clean_lgd=0.85,
    iu_bands=[
        (0.30, 'low'),
        (0.60, 'medium'),
        (0.80, 'high'),
        (1.00, 'very_high'),
        (float('inf'), 'over_limit')
    ]
)


GIRO_TPV = ProductSchema(
    name='giro_tpv',
    description='Capital de giro com garantia de TPV (maquininha) — Stone',
    buckets=[0, 30, 60, 90, 120, 180, 360],
    default_threshold=90,
    state_dimensions=['dpd'],
    collateral_types={
        'tpv': CollateralConfig(
            name='tpv',
            ephemerality=0.8,
            base_lgd=0.20,
            stressed_lgd=0.90,
            description='Retenção de transações da maquininha — evapora com churn'
        ),
        'recebiveis': CollateralConfig(
            name='recebiveis',
            ephemerality=0.5,
            base_lgd=0.30,
            stressed_lgd=0.70,
            description='Cessão fiduciária de recebíveis — risco de inadimplência dos sacados'
        ),
        'imovel': CollateralConfig(
            name='imovel',
            ephemerality=0.1,
            base_lgd=0.25,
            stressed_lgd=0.40,
            description='Alienação fiduciária de imóvel — estável, execução lenta'
        )
    },
    clean_lgd=0.85
)


GIRO_TRADITIONAL = ProductSchema(
    name='giro_traditional',
    description='Capital de giro com garantias tradicionais (imóvel, recebíveis)',
    buckets=[0, 30, 60, 90, 120, 180, 360],
    default_threshold=90,
    state_dimensions=['dpd'],
    collateral_types={
        'imovel': CollateralConfig(
            name='imovel',
            ephemerality=0.1,
            base_lgd=0.25,
            stressed_lgd=0.40,
            description='Alienação fiduciária de imóvel'
        ),
        'recebiveis': CollateralConfig(
            name='recebiveis',
            ephemerality=0.5,
            base_lgd=0.30,
            stressed_lgd=0.70,
            description='Cessão fiduciária de recebíveis'
        ),
        'equipamentos': CollateralConfig(
            name='equipamentos',
            ephemerality=0.3,
            base_lgd=0.35,
            stressed_lgd=0.60,
            description='Alienação de equipamentos / máquinas'
        )
    },
    clean_lgd=0.85
)
