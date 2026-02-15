"""
TransitionMatrixLearner - Core class for credit risk transition matrix analysis.

This module implements the main class for calculating transition matrices from 
credit portfolio data, supporting both global and segmented analysis.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.config import DEFAULT_BUCKETS, DEFAULT_BUCKET_LABELS
from .validators import validate_input_data, validate_columns
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


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
        rating_scale: Optional[List[str]] = None,
        time_horizon: int = 12,
        min_observations: int = 100
    ) -> None:
        """Initialize the TransitionMatrixLearner.
        
        Parameters
        ----------
        buckets : List[int], optional
            DPD boundaries for bucket creation (default: [0,15,30,60,90,120,180,240,360])
        rating_scale : List[str], optional
            Pre-defined rating labels ordered best → worst 
            (e.g., ['1','2','3','4','5','6','7'] or ['AAA','AA','A','BBB','BB','B','D']).
            When provided, ``bucket_col`` in ``fit()`` is expected to contain these
            labels directly — no numeric bucketing is performed.
        time_horizon : int
            Default time horizon in months (default: 12)
        min_observations : int
            Minimum transitions required per matrix (default: 100)
        """
        
        if rating_scale is not None:
            # Rating scale mode: labels provided directly
            self.buckets_ = None
            self.bucket_labels_ = [str(r) for r in rating_scale]
            self._pre_assigned = True
        else:
            # DPD mode: numeric bucket boundaries
            self.buckets_ = buckets if buckets is not None else DEFAULT_BUCKETS.copy()
            self.bucket_labels_ = self._create_bucket_labels(self.buckets_)
            self._pre_assigned = False
        
        # Parameters
        self.time_horizon = time_horizon
        self.min_observations = min_observations
        
        # State attributes (set during fit)
        self.transition_matrix_: Optional[pd.DataFrame] = None
        self.segmented_matrices_: Dict[str, pd.DataFrame] = {}
        self.data_processor_: Optional[DataProcessor] = None
        self.is_fitted_ = False
        
        # Store original data info after fitting
        self._original_columns: Dict[str, Optional[str]] = {}
        self._segments: Optional[List[str]] = None
        self._transitions: Optional[pd.DataFrame] = None
        
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
        segment_col: Optional[str] = None,
        weight_col: Optional[str] = None
    ) -> "TransitionMatrixLearner":
        """
        Fit the transition matrix learner on historical data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset with contract observations over time.
            Also accepts PySpark DataFrames (auto-converted via .toPandas()).
        id_col : str
            Column name containing unique contract identifiers
        time_col : str
            Column name containing reference dates
        bucket_col : str
            Column name containing risk bucket values (days past due) or
            pre-assigned rating labels when ``rating_scale`` was provided
        segment_col : str, optional
            Column name for portfolio segments (for segmented analysis)
        weight_col : str, optional
            Column name with exposure weights (EAD) for weighted transition matrices
            
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
        required_cols = [id_col, time_col, bucket_col]
        if segment_col:
            required_cols.append(segment_col)
        if weight_col:
            required_cols.append(weight_col)
        validate_columns(df, required_cols)
        
        # Store column mappings
        self._original_columns = {
            'id': id_col,
            'time': time_col, 
            'bucket': bucket_col,
            'segment': segment_col,
            'weight': weight_col
        }
        
        # Initialize data processor
        self.data_processor_ = DataProcessor(
            buckets=self.buckets_ or [],
            bucket_labels=self.bucket_labels_,
            pre_assigned=self._pre_assigned
        )
        
        # Process the data
        processed_data = self.data_processor_.process_data(
            df=df,
            id_col=id_col,
            time_col=time_col,
            bucket_col=bucket_col,
            segment_col=segment_col,
            weight_col=weight_col
        )
        
        # Store transitions for later use (validate_model, etc.)
        self._transitions = processed_data['transitions']
        
        # Calculate global transition matrix
        self.transition_matrix_ = self._calculate_transition_matrix(
            processed_data['transitions'],
            weighted=weight_col is not None
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
                        segment_transitions,
                        weighted=weight_col is not None
                    )
                else:
                    warnings.warn(
                        f"Segment '{segment}' has insufficient observations "
                        f"({len(segment_transitions)} < {self.min_observations}). "
                        "Skipping matrix calculation."
                    )
        
        self.is_fitted_ = True
        logger.info(
            "Model fitted: %d buckets, %d transitions, %d segments",
            len(self.bucket_labels_),
            len(processed_data['transitions']),
            len(self.segmented_matrices_)
        )
        return self
    
    def _calculate_transition_matrix(
        self, 
        transitions_df: pd.DataFrame,
        weighted: bool = False
    ) -> pd.DataFrame:
        """Calculate transition matrix from transitions dataframe."""
        
        if len(transitions_df) < self.min_observations:
            raise ValueError(
                f"Insufficient observations for matrix calculation: "
                f"{len(transitions_df)} < {self.min_observations}"
            )
        
        all_buckets = self.bucket_labels_
        
        if weighted and 'weight' in transitions_df.columns:
            # Weighted transition matrix (by EAD)
            transition_matrix = pd.DataFrame(
                0.0, index=all_buckets, columns=all_buckets
            )
            for from_bucket in all_buckets:
                mask = transitions_df['from_bucket'] == from_bucket
                subset = transitions_df[mask]
                if len(subset) > 0:
                    total_weight = subset['weight'].sum()
                    if total_weight > 0:
                        for to_bucket in all_buckets:
                            to_mask = subset['to_bucket'] == to_bucket
                            transition_matrix.loc[from_bucket, to_bucket] = (
                                subset.loc[to_mask, 'weight'].sum() / total_weight
                            )
        else:
            # Count-based transition matrix
            transition_counts = pd.crosstab(
                transitions_df['from_bucket'],
                transitions_df['to_bucket'],
                normalize='index'
            )
            
            # Ensure all buckets are present
            transition_matrix = pd.DataFrame(
                0.0, index=all_buckets, columns=all_buckets
            )
            
            for from_bucket in transition_counts.index:
                for to_bucket in transition_counts.columns:
                    if from_bucket in all_buckets and to_bucket in all_buckets:
                        transition_matrix.loc[from_bucket, to_bucket] = (
                            transition_counts.loc[from_bucket, to_bucket]
                        )
        
        return transition_matrix
    
    def transform(
        self, 
        df: Optional[pd.DataFrame] = None,
        modes: List[str] = ['global']
    ) -> Dict[str, Any]:
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
        Dict[str, Any]
            Dictionary with transition matrices by mode
        """
        
        self._check_is_fitted("transform")
        
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
        
        self._check_is_fitted("predict_transitions")
        
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
        
        self._check_is_fitted("plot_heatmaps")
        
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
        
        self._check_is_fitted("calculate_pd")
        
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
            Validation metrics to calculate: 'accuracy', 'stability', 'coverage'
            
        Returns
        -------
        Dict[str, float]
            Dictionary with validation metrics
        """
        
        self._check_is_fitted("validate_model")
        
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = self._calculate_accuracy(test_data)
        
        if 'stability' in metrics:
            # Diagonal dominance: higher = more stable matrix
            diagonal_sum = np.trace(self.transition_matrix_.values)
            n_buckets = len(self.bucket_labels_)
            results['stability'] = diagonal_sum / n_buckets
        
        if 'coverage' in metrics:
            # Fraction of non-zero cells in the matrix
            non_zero_cells = (self.transition_matrix_.values > 0).sum()
            total_cells = self.transition_matrix_.shape[0] * self.transition_matrix_.shape[1]
            results['coverage'] = non_zero_cells / total_cells
        
        return results
    
    def _calculate_accuracy(self, test_data: pd.DataFrame) -> float:
        """
        Calculate prediction accuracy on test data.
        
        For each observed transition in the test data, predicts the most
        likely next bucket using the fitted transition matrix and compares
        with the actual observed next bucket.
        """
        
        # Process test data to get transitions
        processed_test = self.data_processor_.process_data(
            df=test_data,
            id_col=self._original_columns['id'],
            time_col=self._original_columns['time'],
            bucket_col=self._original_columns['bucket'],
            segment_col=self._original_columns.get('segment'),
            weight_col=self._original_columns.get('weight')
        )
        
        test_transitions = processed_test['transitions']
        
        if test_transitions.empty:
            logger.warning("No transitions found in test data")
            return 0.0
        
        # For each transition, get predicted most likely next bucket
        matrix = self.transition_matrix_
        
        correct = 0
        total = 0
        
        for from_bucket in test_transitions['from_bucket'].unique():
            if from_bucket not in matrix.index:
                continue
            
            # Most likely next bucket according to the model
            predicted_bucket = matrix.loc[from_bucket].idxmax()
            
            # Actual next buckets for this from_bucket
            mask = test_transitions['from_bucket'] == from_bucket
            actual_buckets = test_transitions.loc[mask, 'to_bucket']
            
            correct += (actual_buckets == predicted_bucket).sum()
            total += len(actual_buckets)
        
        return correct / total if total > 0 else 0.0
    
    def _check_is_fitted(self, method_name: str) -> None:
        """Check if the model has been fitted, raise ValueError if not."""
        if not self.is_fitted_:
            raise ValueError(
                f"Model must be fitted before {method_name}. Call fit() first."
            )
    
    def get_pd_term_structure(
        self,
        n_periods: Optional[int] = None,
        default_states: Optional[List[str]] = None
    ) -> "pd.DataFrame":
        """
        Convenience method: complete PD term structure from the fitted matrix.
        
        Parameters
        ----------
        n_periods : int, optional
            Number of periods. Defaults to self.time_horizon.
        default_states : List[str], optional
            Bucket labels considered as default. Defaults to last bucket.
            
        Returns
        -------
        pd.DataFrame
            MultiIndex columns with cumulative_pd, marginal_pd, survival per bucket.
        """
        self._check_is_fitted("get_pd_term_structure")
        from .pd_engine import pd_term_structure as _pd_term_structure
        
        return _pd_term_structure(
            self.transition_matrix_,
            n_periods=n_periods or self.time_horizon,
            default_states=default_states
        )
    
    def get_cure_rates(self) -> "pd.DataFrame":
        """
        Convenience method: extract cure rates from the fitted matrix.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with from_bucket, to_bucket, cure_probability.
        """
        self._check_is_fitted("get_cure_rates")
        from .pd_engine import cure_rates as _cure_rates
        
        return _cure_rates(self.transition_matrix_)
    
    def get_absorbing_analysis(
        self,
        absorbing_states: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method: absorbing state analysis from the fitted matrix.
        
        Parameters
        ----------
        absorbing_states : List[str], optional
            States considered absorbing. Defaults to last bucket.
            
        Returns
        -------
        Dict[str, Any]
            'expected_steps' and 'absorption_probabilities'.
        """
        self._check_is_fitted("get_absorbing_analysis")
        from .pd_engine import absorbing_analysis as _absorbing_analysis
        
        return _absorbing_analysis(self.transition_matrix_, absorbing_states)
    
    def get_lgd_implied(
        self,
        default_states: Optional[List[str]] = None,
        recovery_haircut: float = 0.0
    ) -> "pd.DataFrame":
        """
        Convenience method: LGD implied by cure rates from the fitted matrix.
        
        Parameters
        ----------
        default_states : List[str], optional
            Bucket labels considered as default. Defaults to last bucket.
        recovery_haircut : float, default=0.0
            Haircut on recovery estimate (0.0–1.0).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with default_state, cure_rate, recovery_rate, lgd_implied.
        """
        self._check_is_fitted("get_lgd_implied")
        from .lgd_engine import lgd_from_cure_rates as _lgd_from_cure_rates
        
        return _lgd_from_cure_rates(
            self.transition_matrix_,
            default_states=default_states,
            recovery_haircut=recovery_haircut
        )
    
    def log_to_mlflow(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log model parameters, metrics, and transition matrix to MLflow.
        
        Requires mlflow to be installed. This method does NOT create a hard
        dependency on mlflow — it will raise ImportError if not available.
        
        Parameters
        ----------
        experiment_name : str, optional
            MLflow experiment name. If None, uses the active experiment.
        run_name : str, optional
            Name for this MLflow run.
        extra_params : Dict[str, Any], optional
            Additional parameters to log.
        extra_metrics : Dict[str, float], optional
            Additional metrics to log.
        """
        
        self._check_is_fitted("log_to_mlflow")
        
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for log_to_mlflow(). "
                "Install with: pip install mlflow"
            )
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("n_buckets", len(self.bucket_labels_))
            mlflow.log_param("time_horizon", self.time_horizon)
            mlflow.log_param("min_observations", self.min_observations)
            mlflow.log_param("bucket_labels", str(self.bucket_labels_))
            
            if extra_params:
                mlflow.log_params(extra_params)
            
            # Log metrics (matrix properties)
            matrix = self.transition_matrix_.values
            mlflow.log_metric("diagonal_dominance", float(np.trace(matrix) / len(self.bucket_labels_)))
            mlflow.log_metric("non_zero_cells", int((matrix > 0).sum()))
            
            if self._transitions is not None:
                mlflow.log_metric("n_transitions", len(self._transitions))
            
            if extra_metrics:
                mlflow.log_metrics(extra_metrics)
            
            # Log transition matrix as CSV artifact
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                matrix_path = os.path.join(tmpdir, "transition_matrix.csv")
                self.transition_matrix_.to_csv(matrix_path)
                mlflow.log_artifact(matrix_path)
                
                # Log segmented matrices
                for seg_name, seg_matrix in self.segmented_matrices_.items():
                    seg_path = os.path.join(tmpdir, f"transition_matrix_{seg_name}.csv")
                    seg_matrix.to_csv(seg_path)
                    mlflow.log_artifact(seg_path)
        
        logger.info("Logged model to MLflow (experiment=%s, run=%s)", experiment_name, run_name)
    
    def save(self, path: str) -> None:
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        path : str
            File path to save the model (recommended: .pkl extension).
        """
        self._check_is_fitted("save")
        
        import pickle
        from pathlib import Path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'buckets_': self.buckets_,
            'bucket_labels_': self.bucket_labels_,
            'time_horizon': self.time_horizon,
            'min_observations': self.min_observations,
            'transition_matrix_': self.transition_matrix_,
            'segmented_matrices_': self.segmented_matrices_,
            'is_fitted_': self.is_fitted_,
            '_original_columns': self._original_columns,
            '_segments': self._segments,
            '_transitions': self._transitions,
            '_pre_assigned': self._pre_assigned,
            '_version': '0.3.0'
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info("Model saved to %s", path)
    
    @classmethod
    def load(cls, path: str) -> "TransitionMatrixLearner":
        """
        Load a saved model from disk.
        
        Parameters
        ----------
        path : str
            File path to load the model from.
            
        Returns
        -------
        TransitionMatrixLearner
            Loaded, fitted model instance.
        """
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        pre_assigned = state.get('_pre_assigned', False)
        
        if pre_assigned:
            instance = cls(
                rating_scale=state['bucket_labels_'],
                time_horizon=state['time_horizon'],
                min_observations=state['min_observations']
            )
        else:
            instance = cls(
                buckets=state['buckets_'],
                time_horizon=state['time_horizon'],
                min_observations=state['min_observations']
            )
        
        instance.transition_matrix_ = state['transition_matrix_']
        instance.segmented_matrices_ = state['segmented_matrices_']
        instance.is_fitted_ = state['is_fitted_']
        instance._original_columns = state['_original_columns']
        instance._segments = state['_segments']
        instance._transitions = state.get('_transitions')
        
        # Re-initialize DataProcessor for validation/re-fitting
        instance.data_processor_ = DataProcessor(
            buckets=instance.buckets_ or [],
            bucket_labels=instance.bucket_labels_,
            pre_assigned=instance._pre_assigned
        )
        
        logger.info("Model loaded from %s", path)
        return instance

    def __repr__(self) -> str:
        """String representation of the learner."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        n_buckets = len(self.bucket_labels_)
        n_segments = len(self.segmented_matrices_) if self.segmented_matrices_ else 0
        mode = "rating_scale" if self._pre_assigned else "dpd_buckets"
        
        return (
            f"TransitionMatrixLearner("
            f"mode={mode}, "
            f"buckets={n_buckets}, "
            f"time_horizon={self.time_horizon}, "
            f"segments={n_segments}, "
            f"status={status})"
        )

