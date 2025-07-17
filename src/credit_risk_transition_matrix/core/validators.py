"""
Data validation utilities for credit risk transition matrix analysis.

This module provides functions to validate input data format, column names,
and data quality for transition matrix calculations.
"""

from typing import List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validate basic input data requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate
        
    Raises
    ------
    ValueError
        If validation fails
    TypeError
        If input is not a DataFrame
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if len(df) < 10:
        raise ValueError(f"Input DataFrame must have at least 10 rows, got {len(df)}")


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that required columns exist in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    required_columns : List[str]
        List of required column names (excluding None values)
        
    Raises
    ------
    ValueError
        If required columns are missing
    """
    
    # Filter out None values from required columns
    required_columns = [col for col in required_columns if col is not None]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_id_column(df: pd.DataFrame, id_col: str) -> None:
    """
    Validate ID column requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    id_col : str
        Name of the ID column
        
    Raises
    ------
    ValueError
        If ID column validation fails
    """
    
    if df[id_col].isnull().any():
        raise ValueError(f"ID column '{id_col}' cannot contain null values")
    
    if df[id_col].dtype == 'object':
        # Check for empty strings
        if (df[id_col] == '').any():
            raise ValueError(f"ID column '{id_col}' cannot contain empty strings")


def validate_time_column(df: pd.DataFrame, time_col: str) -> None:
    """
    Validate time column requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Name of the time column
        
    Raises
    ------
    ValueError
        If time column validation fails
    """
    
    if df[time_col].isnull().any():
        raise ValueError(f"Time column '{time_col}' cannot contain null values")
    
    # Try to convert to datetime if not already
    try:
        pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(
            f"Time column '{time_col}' cannot be converted to datetime: {e}"
        )


def validate_bucket_column(df: pd.DataFrame, bucket_col: str, buckets: List[int]) -> None:
    """
    Validate bucket column requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    bucket_col : str
        Name of the bucket column
    buckets : List[int]
        Expected bucket values
        
    Raises
    ------
    ValueError
        If bucket column validation fails
    """
    
    if df[bucket_col].isnull().any():
        raise ValueError(f"Bucket column '{bucket_col}' cannot contain null values")
    
    # Check if values are numeric
    if not pd.api.types.is_numeric_dtype(df[bucket_col]):
        raise ValueError(f"Bucket column '{bucket_col}' must contain numeric values")
    
    # Check for negative values
    if (df[bucket_col] < 0).any():
        raise ValueError(f"Bucket column '{bucket_col}' cannot contain negative values")


def validate_segment_column(df: pd.DataFrame, segment_col: str) -> None:
    """
    Validate segment column requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    segment_col : str
        Name of the segment column
        
    Raises
    ------
    ValueError
        If segment column validation fails
    """
    
    if df[segment_col].isnull().any():
        raise ValueError(f"Segment column '{segment_col}' cannot contain null values")
    
    # Check number of unique segments
    n_segments = df[segment_col].nunique()
    if n_segments < 2:
        raise ValueError(
            f"Segment column '{segment_col}' must have at least 2 unique values, "
            f"got {n_segments}"
        )
    
    if n_segments > 20:
        raise ValueError(
            f"Segment column '{segment_col}' has too many unique values ({n_segments}). "
            "Maximum recommended is 20 for practical analysis."
        )


def validate_data_quality(df: pd.DataFrame, id_col: str, time_col: str) -> dict:
    """
    Perform comprehensive data quality checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    id_col : str
        Name of the ID column
    time_col : str
        Name of the time column
        
    Returns
    -------
    dict
        Dictionary with data quality metrics and warnings
    """
    
    quality_report = {
        'total_records': len(df),
        'unique_contracts': df[id_col].nunique(),
        'date_range': {
            'min_date': df[time_col].min(),
            'max_date': df[time_col].max()
        },
        'warnings': [],
        'recommendations': []
    }
    
    # Check for duplicate records
    duplicates = df.duplicated(subset=[id_col, time_col]).sum()
    if duplicates > 0:
        quality_report['warnings'].append(
            f"Found {duplicates} duplicate records (same ID and date)"
        )
    
    # Check time series completeness
    avg_obs_per_contract = len(df) / df[id_col].nunique()
    if avg_obs_per_contract < 6:
        quality_report['warnings'].append(
            f"Low average observations per contract: {avg_obs_per_contract:.1f}. "
            "Recommended minimum is 6 for reliable transition analysis."
        )
    
    # Check date gaps
    time_series = pd.to_datetime(df[time_col])
    date_range_months = (time_series.max() - time_series.min()).days / 30.44
    if date_range_months < 12:
        quality_report['recommendations'].append(
            f"Data spans only {date_range_months:.1f} months. "
            "Recommended minimum is 12 months for stable transition matrices."
        )
    
    return quality_report
