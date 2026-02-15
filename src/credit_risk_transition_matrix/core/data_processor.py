"""
Data processing utilities for transition matrix calculations.

This module handles data preprocessing, bucket assignment, transition tracking,
and other data manipulation tasks required for transition matrix analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from .validators import (
    validate_id_column, 
    validate_time_column, 
    validate_bucket_column,
    validate_rating_column,
    validate_segment_column
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor for transition matrix calculations.
    
    This class handles all data preprocessing steps including bucket assignment,
    transition tracking, and data quality checks.
    
    Parameters
    ----------
    buckets : List[int]
        Risk bucket definitions in days past due
    bucket_labels : List[str]
        Human-readable labels for buckets
    """
    
    def __init__(self, buckets: List[int], bucket_labels: List[str], pre_assigned: bool = False):
        self.buckets = buckets
        self.bucket_labels = bucket_labels
        self.pre_assigned = pre_assigned
        
    def process_data(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        bucket_col: str,
        segment_col: Optional[str] = None,
        weight_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process input data for transition matrix calculation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset. Also accepts PySpark DataFrames (converted via .toPandas()).
        id_col : str
            Column name containing contract IDs
        time_col : str
            Column name containing reference dates
        bucket_col : str
            Column name containing bucket values
        segment_col : str, optional
            Column name containing segment information
        weight_col : str, optional
            Column name containing exposure weights (e.g. EAD) for weighted matrices
            
        Returns
        -------
        Dict[str, Any]
            Processed data including transitions and metadata
        """
        
        # Auto-convert Spark DataFrame to pandas
        df = self._ensure_pandas(df)
        
        # Validate columns
        validate_id_column(df, id_col)
        validate_time_column(df, time_col)
        
        if self.pre_assigned:
            validate_rating_column(df, bucket_col, self.bucket_labels)
        else:
            validate_bucket_column(df, bucket_col, self.buckets)
        
        if segment_col is not None:
            validate_segment_column(df, segment_col)
        
        # Build list of columns to select
        cols = [id_col, time_col, bucket_col]
        if segment_col is not None:
            cols.append(segment_col)
        if weight_col is not None:
            cols.append(weight_col)
        
        # Create working copy
        data = df[cols].copy()
        
        if self.pre_assigned:
            # Rating scale mode: use values directly as bucket labels
            rename_map = {
                id_col: 'contract_id',
                time_col: 'reference_date',
                bucket_col: 'bucket'
            }
            if segment_col is not None:
                rename_map[segment_col] = 'segment'
            if weight_col is not None:
                rename_map[weight_col] = 'weight'
            data = data.rename(columns=rename_map)
            data['reference_date'] = pd.to_datetime(data['reference_date'])
            data['bucket'] = data['bucket'].astype(str)
        else:
            # DPD mode: assign buckets from numeric values
            rename_map = {
                id_col: 'contract_id',
                time_col: 'reference_date', 
                bucket_col: 'days_past_due'
            }
            if segment_col is not None:
                rename_map[segment_col] = 'segment'
            if weight_col is not None:
                rename_map[weight_col] = 'weight'
            data = data.rename(columns=rename_map)
            data['reference_date'] = pd.to_datetime(data['reference_date'])
            data['bucket'] = self._assign_buckets(data['days_past_due'])
        
        # Sort data
        data = data.sort_values(['contract_id', 'reference_date']).reset_index(drop=True)
        
        # Calculate transitions (vectorized)
        transitions = self._calculate_transitions(data)
        
        logger.info(
            "Processed %d observations → %d transitions from %d contracts",
            len(data), len(transitions), data['contract_id'].nunique()
        )
        
        # Prepare results
        results = {
            'processed_data': data,
            'transitions': transitions,
            'segments': data['segment'].unique().tolist() if segment_col else None,
            'date_range': {
                'min_date': data['reference_date'].min(),
                'max_date': data['reference_date'].max()
            },
            'summary': self._create_summary(data, transitions)
        }
        
        return results
    
    @staticmethod
    def _ensure_pandas(df: Any) -> pd.DataFrame:
        """Convert Spark DataFrame to pandas if necessary."""
        if isinstance(df, pd.DataFrame):
            return df
        # Check for PySpark DataFrame without hard dependency
        type_name = type(df).__name__
        if type_name == 'DataFrame' and hasattr(df, 'toPandas'):
            logger.info("Converting Spark DataFrame to pandas (%s)", type_name)
            return df.toPandas()
        return df
    
    def _assign_buckets(self, days_past_due: pd.Series) -> pd.Series:
        """
        Assign bucket labels based on days past due using vectorized pd.cut().
        
        Parameters
        ----------
        days_past_due : pd.Series
            Series with days past due values
            
        Returns
        -------
        pd.Series
            Series with bucket labels (as string, not Categorical)
        """
        bins = self.buckets + [np.inf]
        result = pd.cut(
            days_past_due,
            bins=bins,
            labels=self.bucket_labels,
            right=False,
            include_lowest=True
        )
        return result.astype(str)
    
    def _calculate_transitions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transitions between consecutive observations using vectorized
        groupby + shift operations.
        
        Parameters
        ----------
        data : pd.DataFrame
            Processed data with contract_id, reference_date, bucket columns.
            Must be sorted by [contract_id, reference_date].
            
        Returns
        -------
        pd.DataFrame
            DataFrame with transition records
        """
        
        # Vectorized: shift within each contract group
        grouped = data.groupby('contract_id', sort=False)
        
        data = data.copy()
        data['to_bucket'] = grouped['bucket'].shift(-1)
        data['to_date'] = grouped['reference_date'].shift(-1)
        if 'days_past_due' in data.columns:
            data['to_days_past_due'] = grouped['days_past_due'].shift(-1)
        
        if 'weight' in data.columns:
            data['to_weight'] = grouped['weight'].shift(-1)
        
        # Drop rows without a next observation (last row per contract)
        transitions = data.dropna(subset=['to_bucket']).copy()
        
        # Calculate time difference in months
        transitions['time_diff_months'] = (
            (transitions['to_date'] - transitions['reference_date']).dt.days / 30.44
        )
        
        # Filter to reasonable time gaps (0.5 to 6 months)
        transitions = transitions[
            transitions['time_diff_months'].between(0.5, 6.0)
        ].copy()
        
        if transitions.empty:
            raise ValueError("No valid transitions found in the data")
        
        rename_cols = {
            'reference_date': 'from_date',
            'bucket': 'from_bucket'
        }
        if 'days_past_due' in transitions.columns:
            rename_cols['days_past_due'] = 'from_days_past_due'
        transitions = transitions.rename(columns=rename_cols)
        
        # Select output columns
        output_cols = [
            'contract_id', 'from_date', 'to_date',
            'from_bucket', 'to_bucket',
            'time_diff_months'
        ]
        if 'from_days_past_due' in transitions.columns:
            output_cols.extend(['from_days_past_due', 'to_days_past_due'])
        if 'segment' in transitions.columns:
            output_cols.append('segment')
        if 'weight' in transitions.columns:
            output_cols.append('weight')
        
        return transitions[output_cols].reset_index(drop=True)
    
    def _create_summary(self, data: pd.DataFrame, transitions: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for the processed data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Processed data
        transitions : pd.DataFrame
            Calculated transitions
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        
        n_contracts = data['contract_id'].nunique()
        
        summary = {
            'total_observations': len(data),
            'unique_contracts': n_contracts,
            'total_transitions': len(transitions),
            'date_range_months': (
                data['reference_date'].max() - data['reference_date'].min()
            ).days / 30.44,
            'avg_observations_per_contract': len(data) / n_contracts if n_contracts > 0 else 0,
            'bucket_distribution': data['bucket'].value_counts().to_dict(),
            'transition_distribution': transitions.groupby(['from_bucket', 'to_bucket']).size().to_dict()
        }
        
        # Add segment summary if available
        if 'segment' in data.columns:
            summary['segment_distribution'] = data['segment'].value_counts().to_dict()
            summary['segments'] = data['segment'].unique().tolist()
        
        return summary
