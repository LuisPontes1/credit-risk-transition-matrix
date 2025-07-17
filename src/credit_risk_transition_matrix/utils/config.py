"""
Configuration settings and default values for the credit risk transition matrix library.

This module contains default bucket definitions, labels, and other configuration
settings that are used throughout the library.
"""

from typing import List

# Default risk bucket definitions (days past due)
DEFAULT_BUCKETS: List[int] = [0, 15, 30, 60, 90, 120, 180, 240, 360]

# Default bucket labels for visualization and reports
DEFAULT_BUCKET_LABELS: List[str] = [
    "0-14",      # 0 to 14 days
    "15-29",     # 15 to 29 days  
    "30-59",     # 30 to 59 days
    "60-89",     # 60 to 89 days
    "90-119",    # 90 to 119 days
    "120-179",   # 120 to 179 days
    "180-239",   # 180 to 239 days
    "240-359",   # 240 to 359 days
    "360+"       # 360+ days
]

# Visualization settings
DEFAULT_HEATMAP_CONFIG = {
    'figsize': (12, 10),
    'cmap': 'Blues',
    'annot': True,
    'fmt': '.2%',
    'square': True,
    'linewidths': 0.5
}

# Analysis settings
DEFAULT_ANALYSIS_CONFIG = {
    'min_observations': 100,
    'max_time_gap_months': 6,
    'min_time_gap_months': 0.5,
    'default_time_horizon': 12
}

# Export settings
DEFAULT_EXPORT_CONFIG = {
    'dpi': 300,
    'format': 'png',
    'bbox_inches': 'tight',
    'facecolor': 'white'
}

# Validation settings
VALIDATION_CONFIG = {
    'min_records': 10,
    'min_unique_contracts': 5,
    'max_segments': 20,
    'min_segments': 2
}
