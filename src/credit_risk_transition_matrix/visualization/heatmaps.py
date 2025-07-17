"""
Professional heatmap visualizations for transition matrices.

This module provides functions to create publication-ready heatmaps showing
transition probabilities between risk buckets, with support for both global
and segmented analysis.
"""

from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings


def plot_global_heatmap(
    transition_matrix: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    annot: bool = True,
    fmt: str = '.2%',
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Create a professional heatmap visualization of a transition matrix.
    
    This function generates a publication-ready heatmap showing transition
    probabilities between risk buckets, with proper formatting and styling
    suitable for reports and presentations.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Square transition matrix with bucket labels as index and columns
    title : str, optional
        Custom title for the plot. If None, uses default title.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(12, 10)
        Figure size in inches (width, height)
    annot : bool, default=True
        Whether to annotate cells with transition probabilities
    fmt : str, default='.2%'
        Format string for annotations (percentage format)
    cmap : str, default='Blues'
        Colormap for the heatmap
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
        
    Examples
    --------
    >>> fig = plot_global_heatmap(
    ...     transition_matrix,
    ...     title="Portfolio Transition Matrix - 2023",
    ...     save_path="./reports/transition_heatmap.png"
    ... )
    """
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    heatmap = sns.heatmap(
        transition_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Transition Probability'},
        ax=ax
    )
    
    # Customize the plot
    if title is None:
        title = "Global Transition Matrix"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Bucket Atraso - Próxima Safra', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bucket Atraso - Safra Atual', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add a subtle border around the heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('gray')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add metadata text
    metadata_text = f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    fig.text(0.99, 0.01, metadata_text, fontsize=8, ha='right', va='bottom', 
             alpha=0.7, style='italic')
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in high resolution
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Heatmap saved to: {save_path}")
    
    return fig


def plot_segmented_heatmaps(
    matrices_dict: Dict[str, pd.DataFrame],
    save_dir: Optional[str] = None,
    figsize: tuple = (10, 8),
    ncols: int = 2
) -> Dict[str, plt.Figure]:
    """
    Create heatmaps for multiple segmented transition matrices.
    
    Parameters
    ----------
    matrices_dict : Dict[str, pd.DataFrame]
        Dictionary with segment names as keys and transition matrices as values
    save_dir : str, optional
        Directory to save individual segment heatmaps
    figsize : tuple, default=(10, 8)
        Figure size for each individual heatmap
    ncols : int, default=2
        Number of columns for subplot layout (if creating combined plot)
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with segment names as keys and figure objects as values
    """
    
    figures = {}
    
    for segment, matrix in matrices_dict.items():
        # Create individual heatmap for each segment
        fig = plot_global_heatmap(
            transition_matrix=matrix,
            title=f"Transition Matrix - {segment}",
            save_path=f"{save_dir}/heatmap_{segment}.png" if save_dir else None,
            figsize=figsize
        )
        
        figures[segment] = fig
    
    return figures


def plot_comparison_heatmaps(
    matrices_dict: Dict[str, pd.DataFrame],
    layout: str = 'grid',
    figsize: tuple = (20, 15),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create side-by-side comparison of multiple transition matrices.
    
    Parameters
    ----------
    matrices_dict : Dict[str, pd.DataFrame]
        Dictionary with matrix names and transition matrices
    layout : str, default='grid'
        Layout type: 'grid' or 'horizontal'
    figsize : tuple, default=(20, 15)
        Overall figure size
    save_path : str, optional
        Path to save the comparison plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with comparison heatmaps
    """
    
    n_matrices = len(matrices_dict)
    
    if layout == 'grid':
        ncols = min(2, n_matrices)
        nrows = (n_matrices + ncols - 1) // ncols
    else:  # horizontal
        ncols = n_matrices
        nrows = 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle single subplot case
    if n_matrices == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if n_matrices > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, (name, matrix) in enumerate(matrices_dict.items()):
        ax = axes[i] if i < len(axes) else None
        
        if ax is not None:
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                square=True,
                linewidths=0.5,
                ax=ax
            )
            
            ax.set_title(f"{name}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Next Period Bucket', fontsize=10)
            ax.set_ylabel('Current Period Bucket', fontsize=10)
            
            # Rotate labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Hide unused subplots
    for i in range(n_matrices, len(axes)):
        axes[i].set_visible(False)
    
    # Main title
    fig.suptitle(
        'Transition Matrix Comparison',
        fontsize=18,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_transition_flow(
    transition_matrix: pd.DataFrame,
    threshold: float = 0.05,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create a flow diagram showing significant transitions between buckets.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Transition matrix with bucket labels
    threshold : float, default=0.05
        Minimum transition probability to display (5%)
    save_path : str, optional
        Path to save the flow diagram
    figsize : tuple, default=(14, 10)
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with flow diagram
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # This is a simplified flow diagram
    # A full implementation would use networkx or similar for proper flow layout
    
    buckets = transition_matrix.index.tolist()
    n_buckets = len(buckets)
    
    # Create positions for buckets (circular layout)
    angles = np.linspace(0, 2*np.pi, n_buckets, endpoint=False)
    positions = {
        bucket: (np.cos(angle), np.sin(angle)) 
        for bucket, angle in zip(buckets, angles)
    }
    
    # Draw buckets as circles
    for bucket, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.15, color='lightblue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, bucket, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw significant transitions as arrows
    for from_bucket in buckets:
        for to_bucket in buckets:
            prob = transition_matrix.loc[from_bucket, to_bucket]
            
            if prob >= threshold and from_bucket != to_bucket:
                from_pos = positions[from_bucket]
                to_pos = positions[to_bucket]
                
                # Draw arrow
                ax.annotate(
                    '',
                    xy=to_pos,
                    xytext=from_pos,
                    arrowprops=dict(
                        arrowstyle='->',
                        lw=prob * 10,  # Line width proportional to probability
                        alpha=0.6,
                        color='red' if prob > 0.2 else 'blue'
                    )
                )
                
                # Add probability label
                mid_x = (from_pos[0] + to_pos[0]) / 2
                mid_y = (from_pos[1] + to_pos[1]) / 2
                ax.text(mid_x, mid_y, f'{prob:.1%}', fontsize=6, ha='center')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(
        f'Transition Flow Diagram\n(Showing transitions ≥ {threshold:.1%})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add legend
    ax.text(
        -1.4, -1.3,
        'Arrow thickness indicates\ntransition probability',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7)
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Flow diagram saved to: {save_path}")
    
    return fig
