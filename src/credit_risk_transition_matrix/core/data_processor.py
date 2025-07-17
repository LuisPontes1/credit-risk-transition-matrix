"""
Data processing utilities for transition matrix calculations.

This module handles data preprocessing, bucket assignment, transition tracking,
and other data manipulation tasks required for transition matrix analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from .validators import (
    validate_id_column, 
    validate_time_column, 
    validate_bucket_column,
    validate_segment_column
)


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
    
    def __init__(self, buckets: List[int], bucket_labels: List[str]):
        self.buckets = buckets
        self.bucket_labels = bucket_labels
        
    def process_data(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        bucket_col: str,
        segment_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process input data for transition matrix calculation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset
        id_col : str
            Column name containing contract IDs
        time_col : str
            Column name containing reference dates
        bucket_col : str
            Column name containing bucket values
        segment_col : str, optional
            Column name containing segment information
            
        Returns
        -------
        Dict[str, Any]
            Processed data including transitions and metadata
        """
        
        # Validate columns
        validate_id_column(df, id_col)
        validate_time_column(df, time_col)
        validate_bucket_column(df, bucket_col, self.buckets)
        
        if segment_col is not None:
            validate_segment_column(df, segment_col)
        
        # Create working copy
        data = df[[id_col, time_col, bucket_col] + 
                 ([segment_col] if segment_col else [])].copy()
        
        # Standardize column names
        data = data.rename(columns={
            id_col: 'contract_id',
            time_col: 'reference_date', 
            bucket_col: 'days_past_due'
        })
        
        if segment_col is not None:
            data = data.rename(columns={segment_col: 'segment'})
        
        # Convert date column
        data['reference_date'] = pd.to_datetime(data['reference_date'])
        
        # Assign buckets
        data['bucket'] = self._assign_buckets(data['days_past_due'])
        
        # Sort data
        data = data.sort_values(['contract_id', 'reference_date']).reset_index(drop=True)
        
        # Calculate transitions
        transitions = self._calculate_transitions(data)
        
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
    
    def _assign_buckets(self, days_past_due: pd.Series) -> pd.Series:
        """
        Assign bucket labels based on days past due.
        
        Parameters
        ----------
        days_past_due : pd.Series
            Series with days past due values
            
        Returns
        -------
        pd.Series
            Series with bucket labels
        """
        
        bucket_assignments = pd.Series(index=days_past_due.index, dtype='object')
        
        for i, days in enumerate(days_past_due):
            # Find appropriate bucket
            assigned_bucket = None
            
            for j, bucket_threshold in enumerate(self.buckets):
                if j == len(self.buckets) - 1:
                    # Last bucket (360+)
                    if days >= bucket_threshold:
                        assigned_bucket = self.bucket_labels[j]
                        break
                else:
                    # Regular buckets
                    next_threshold = self.buckets[j + 1]
                    if bucket_threshold <= days < next_threshold:
                        assigned_bucket = self.bucket_labels[j]
                        break
            
            # If no bucket found, assign to first bucket (should not happen with validation)
            if assigned_bucket is None:
                assigned_bucket = self.bucket_labels[0]
                
            bucket_assignments.iloc[i] = assigned_bucket
        
        return bucket_assignments
    
    def _calculate_transitions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transitions between consecutive observations.
        
        Parameters
        ----------
        data : pd.DataFrame
            Processed data with contract_id, reference_date, bucket columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with transition records
        """
        
        transitions = []
        
        # Group by contract and calculate transitions
        for contract_id, group in data.groupby('contract_id'):
            group = group.sort_values('reference_date').reset_index(drop=True)
            
            # Calculate consecutive transitions
            for i in range(len(group) - 1):
                current_row = group.iloc[i]
                next_row = group.iloc[i + 1]
                
                # Calculate time difference in months
                time_diff = (next_row['reference_date'] - current_row['reference_date']).days / 30.44
                
                # Only include transitions with reasonable time gaps (1-6 months)
                if 0.5 <= time_diff <= 6:
                    transition = {
                        'contract_id': contract_id,
                        'from_date': current_row['reference_date'],
                        'to_date': next_row['reference_date'],
                        'from_bucket': current_row['bucket'],
                        'to_bucket': next_row['bucket'],
                        'time_diff_months': time_diff,
                        'from_days_past_due': current_row['days_past_due'],
                        'to_days_past_due': next_row['days_past_due']
                    }
                    
                    # Add segment if available
                    if 'segment' in current_row:
                        transition['segment'] = current_row['segment']
                    
                    transitions.append(transition)
        
        if not transitions:
            raise ValueError("No valid transitions found in the data")
        
        return pd.DataFrame(transitions)
    
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
        
        summary = {
            'total_observations': len(data),
            'unique_contracts': data['contract_id'].nunique(),
            'total_transitions': len(transitions),
            'date_range_months': (
                data['reference_date'].max() - data['reference_date'].min()
            ).days / 30.44,
            'avg_observations_per_contract': len(data) / data['contract_id'].nunique(),
            'bucket_distribution': data['bucket'].value_counts().to_dict(),
            'transition_distribution': transitions.groupby(['from_bucket', 'to_bucket']).size().to_dict()
        }
        
        # Add segment summary if available
        if 'segment' in data.columns:
            summary['segment_distribution'] = data['segment'].value_counts().to_dict()
            summary['segments'] = data['segment'].unique().tolist()
        
        return summary
