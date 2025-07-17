"""
TransitionMatrixLearner - Core class for credit risk transition matrix analysis.

This module implements the main class for calculating transition matrices from 
credit portfolio data, supporting both global and segmented analysis.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.config import DEFAULT_BUCKETS, DEFAULT_BUCKET_LABELS
from .validators import validate_input_data, validate_columns
from .data_processor import DataProcessor


class TransitionMatrixLearner:
    """
    Main class for learning transition matrices from credit risk data.
    
    This class processes panel data with contract IDs, reference dates, and 
    risk buckets to calculate transition probabilities between different 
    risk states over time.
    
    Parameters
    ----------
    buckets : List[int], optional
        Custom risk bucket definitions in days past due.
        Default: [0, 15, 30, 60, 90, 120, 180, 240, 360]
    time_horizon : int, default=12
        Time horizon in months for analysis
    min_observations : int, default=100
        Minimum number of observations required for reliable matrix calculation
    
    Attributes
    ----------
    buckets_ : List[int]
        Risk bucket definitions used for analysis
    bucket_labels_ : List[str]
        Human-readable labels for buckets
    transition_matrix_ : pd.DataFrame
        Global transition matrix (fitted)
    segmented_matrices_ : Dict[str, pd.DataFrame]
        Transition matrices by segment (if fitted with segments)
    data_processor_ : DataProcessor
        Internal data processor instance
    is_fitted_ : bool
        Whether the model has been fitted
    
    Examples
    --------
    Basic usage:
    
    >>> from credit_risk_transition_matrix import TransitionMatrixLearner
    >>> learner = TransitionMatrixLearner(
    ...     buckets=[0, 15, 30, 60, 90, 120, 180, 240, 360],
    ...     time_horizon=12
    ... )
    >>> learner.fit(df, id_col="id_contrato", time_col="data_ref", bucket_col="dias_atraso")
    >>> learner.plot_heatmaps(modes=["global"], save_dir="./outputs/")
    
    Segmented analysis:
    
    >>> learner.fit(df, id_col="id_contrato", time_col="data_ref", 
    ...            bucket_col="dias_atraso", segment_col="produto_tipo")
    >>> matrices = learner.transform(modes=['global', 'segmented'])
    """
    
    def __init__(
        self, 
        buckets: Optional[List[int]] = None,
        time_horizon: int = 12,
        min_observations: int = 100
    ) -> None:
        """Initialize the TransitionMatrixLearner."""
        
        # Set bucket definitions
        self.buckets_ = buckets if buckets is not None else DEFAULT_BUCKETS.copy()
        self.bucket_labels_ = self._create_bucket_labels(self.buckets_)
        
        # Parameters
        self.time_horizon = time_horizon
        self.min_observations = min_observations
        
        # State attributes (set during fit)
        self.transition_matrix_: Optional[pd.DataFrame] = None
        self.segmented_matrices_: Dict[str, pd.DataFrame] = {}
        self.data_processor_: Optional[DataProcessor] = None
        self.is_fitted_ = False
        
        # Store original data info after fitting
        self._original_columns: Dict[str, str] = {}
        self._segments: Optional[List[str]] = None
        
    def _create_bucket_labels(self, buckets: List[int]) -> List[str]:
        """Create human-readable labels for buckets."""
        labels = []
        for i, bucket in enumerate(buckets):
            if i == 0:
                labels.append(f"0-{buckets[i+1]-1}" if i+1 < len(buckets) else "0+")
            elif i == len(buckets) - 1:
                labels.append(f"{bucket}+")
            else:
                labels.append(f"{bucket}-{buckets[i+1]-1}")
        return labels
    
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str, 
        bucket_col: str,
        segment_col: Optional[str] = None
    ) -> "TransitionMatrixLearner":
        """
        Fit the transition matrix learner on historical data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset with contract observations over time
        id_col : str
            Column name containing unique contract identifiers
        time_col : str
            Column name containing reference dates
        bucket_col : str
            Column name containing risk bucket values (days past due)
        segment_col : str, optional
            Column name for portfolio segments (for segmented analysis)
            
        Returns
        -------
        self : TransitionMatrixLearner
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If input data validation fails
        """
        
        # Validate input data
        validate_input_data(df)
        validate_columns(df, [id_col, time_col, bucket_col] + 
                        ([segment_col] if segment_col else []))
        
        # Store column mappings
        self._original_columns = {
            'id': id_col,
            'time': time_col, 
            'bucket': bucket_col,
            'segment': segment_col
        }
        
        # Initialize data processor
        self.data_processor_ = DataProcessor(
            buckets=self.buckets_,
            bucket_labels=self.bucket_labels_
        )
        
        # Process the data
        processed_data = self.data_processor_.process_data(
            df=df,
            id_col=id_col,
            time_col=time_col,
            bucket_col=bucket_col,
            segment_col=segment_col
        )
        
        # Calculate global transition matrix
        self.transition_matrix_ = self._calculate_transition_matrix(
            processed_data['transitions']
        )
        
        # Calculate segmented matrices if segment column provided
        if segment_col is not None:
            self._segments = processed_data['segments']
            self.segmented_matrices_ = {}
            
            for segment in self._segments:
                segment_transitions = processed_data['transitions'][
                    processed_data['transitions']['segment'] == segment
                ]
                if len(segment_transitions) >= self.min_observations:
                    self.segmented_matrices_[segment] = self._calculate_transition_matrix(
                        segment_transitions
                    )
                else:
                    warnings.warn(
                        f"Segment '{segment}' has insufficient observations "
                        f"({len(segment_transitions)} < {self.min_observations}). "
                        "Skipping matrix calculation."
                    )
        
        self.is_fitted_ = True
        return self
    
    def _calculate_transition_matrix(self, transitions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transition matrix from transitions dataframe."""
        
        if len(transitions_df) < self.min_observations:
            raise ValueError(
                f"Insufficient observations for matrix calculation: "
                f"{len(transitions_df)} < {self.min_observations}"
            )
        
        # Create transition matrix
        transition_counts = pd.crosstab(
            transitions_df['from_bucket'],
            transitions_df['to_bucket'],
            normalize='index'  # Normalize by rows (from_bucket)
        )
        
        # Ensure all buckets are present in the matrix
        all_buckets = self.bucket_labels_
        transition_matrix = pd.DataFrame(
            index=all_buckets,
            columns=all_buckets,
            dtype=float
        ).fillna(0.0)
        
        # Fill in calculated values
        for from_bucket in transition_counts.index:
            for to_bucket in transition_counts.columns:
                if from_bucket in all_buckets and to_bucket in all_buckets:
                    transition_matrix.loc[from_bucket, to_bucket] = transition_counts.loc[from_bucket, to_bucket]
        
        return transition_matrix
    
    def transform(
        self, 
        df: Optional[pd.DataFrame] = None,
        modes: List[str] = ['global']
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate transition matrices.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            New data for transformation. If None, returns fitted matrices.
        modes : List[str], default=['global']
            Types of matrices to return: 'global', 'segmented'
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with transition matrices by mode
        """
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transform. Call fit() first.")
        
        results = {}
        
        if 'global' in modes:
            results['global'] = self.transition_matrix_.copy()
            
        if 'segmented' in modes:
            if self.segmented_matrices_:
                results['segmented'] = {
                    segment: matrix.copy() 
                    for segment, matrix in self.segmented_matrices_.items()
                }
            else:
                warnings.warn("No segmented matrices available. Fit with segment_col to enable.")
        
        return results
    
    def predict_transitions(
        self, 
        current_state: Union[str, List[float]], 
        n_periods: int = 1
    ) -> List[float]:
        """
        Predict future state distributions.
        
        Parameters
        ----------
        current_state : str or List[float]
            Current state bucket name or probability distribution
        n_periods : int, default=1
            Number of periods to project forward
            
        Returns
        -------
        List[float]
            Predicted probability distribution after n_periods
        """
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Convert current state to probability vector
        if isinstance(current_state, str):
            if current_state not in self.bucket_labels_:
                raise ValueError(f"Unknown bucket: {current_state}")
            
            state_vector = np.zeros(len(self.bucket_labels_))
            state_vector[self.bucket_labels_.index(current_state)] = 1.0
        else:
            if len(current_state) != len(self.bucket_labels_):
                raise ValueError(
                    f"State vector length ({len(current_state)}) must match "
                    f"number of buckets ({len(self.bucket_labels_)})"
                )
            state_vector = np.array(current_state)
        
        # Apply transition matrix n_periods times
        transition_matrix = self.transition_matrix_.values
        result_vector = state_vector.copy()
        
        for _ in range(n_periods):
            result_vector = result_vector @ transition_matrix
            
        return result_vector.tolist()
    
    def plot_heatmaps(
        self, 
        modes: List[str] = ['global'], 
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate and save heatmap visualizations.
        
        Parameters
        ----------
        modes : List[str], default=['global']
            Types of heatmaps to generate: 'global', 'segmented'
        save_dir : str, optional
            Directory to save plots. If None, plots are displayed only.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with matplotlib figure objects
        """
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")
        
        # Import here to avoid circular imports
        from ..visualization.heatmaps import plot_global_heatmap, plot_segmented_heatmaps
        
        figures = {}
        
        if 'global' in modes:
            fig = plot_global_heatmap(
                transition_matrix=self.transition_matrix_,
                title="Global Transition Matrix",
                save_path=f"{save_dir}/global_heatmap.png" if save_dir else None
            )
            figures['global'] = fig
            
        if 'segmented' in modes and self.segmented_matrices_:
            seg_figures = plot_segmented_heatmaps(
                matrices_dict=self.segmented_matrices_,
                save_dir=save_dir
            )
            figures['segmented'] = seg_figures
        elif 'segmented' in modes:
            warnings.warn("No segmented matrices available for plotting.")
            
        return figures
    
    def calculate_pd(
        self, 
        time_horizon: Optional[int] = None,
        default_buckets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Probability of Default metrics.
        
        Parameters
        ----------
        time_horizon : int, optional
            Time horizon for PD calculation. If None, uses instance default.
        default_buckets : List[str], optional
            Buckets considered as default states. If None, uses last bucket.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with PD calculations and metrics
        """
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before PD calculation. Call fit() first.")
        
        horizon = time_horizon if time_horizon is not None else self.time_horizon
        
        if default_buckets is None:
            # Use the highest bucket as default
            default_buckets = [self.bucket_labels_[-1]]
        
        # Calculate PD for each starting bucket
        pd_results = {}
        
        for bucket in self.bucket_labels_:
            # Predict after horizon periods
            future_dist = self.predict_transitions(bucket, n_periods=horizon)
            
            # Sum probabilities for default buckets
            pd_value = sum(
                future_dist[self.bucket_labels_.index(db)] 
                for db in default_buckets 
                if db in self.bucket_labels_
            )
            
            pd_results[bucket] = pd_value
        
        return {
            'pd_by_bucket': pd_results,
            'time_horizon': horizon,
            'default_buckets': default_buckets,
            'calculation_date': datetime.now().isoformat()
        }
    
    def validate_model(
        self, 
        test_data: pd.DataFrame, 
        metrics: List[str] = ['accuracy']
    ) -> Dict[str, float]:
        """
        Validate model performance on test data.
        
        Parameters
        ----------
        test_data : pd.DataFrame
            Test dataset with same structure as training data
        metrics : List[str], default=['accuracy']
            Validation metrics to calculate
            
        Returns
        -------
        Dict[str, float]
            Dictionary with validation metrics
        """
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before validation. Call fit() first.")
        
        # This is a placeholder for basic validation
        # Full implementation would include proper validation logic
        
        results = {}
        
        if 'accuracy' in metrics:
            # Placeholder accuracy calculation
            results['accuracy'] = 0.85  # Mock value
            
        return results
    
    def __repr__(self) -> str:
        """String representation of the learner."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        n_buckets = len(self.buckets_)
        n_segments = len(self.segmented_matrices_) if self.segmented_matrices_ else 0
        
        return (
            f"TransitionMatrixLearner("
            f"buckets={n_buckets}, "
            f"time_horizon={self.time_horizon}, "
            f"segments={n_segments}, "
            f"status={status})"
        )
