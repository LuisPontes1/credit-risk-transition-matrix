"""
Basic usage example of the Credit Risk Transition Matrix Library.

This example demonstrates how to:
1. Create sample credit portfolio data
2. Initialize and fit the TransitionMatrixLearner
3. Generate transition matrices and visualizations
4. Calculate probability of default (PD) metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import our library
from credit_risk_transition_matrix import TransitionMatrixLearner


def create_sample_data(n_contracts=500, n_months=24):
    """
    Create sample credit portfolio data for demonstration.
    
    This function generates realistic-looking credit portfolio data with:
    - Contract IDs
    - Monthly reference dates
    - Days past due (with realistic evolution patterns)
    - Portfolio segments
    
    Parameters
    ----------
    n_contracts : int
        Number of contracts to generate
    n_months : int
        Number of months of history
        
    Returns
    -------
    pd.DataFrame
        Sample portfolio data
    """
    
    print(f"Creating sample data: {n_contracts} contracts, {n_months} months")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate contract IDs
    contracts = [f"LOAN_{i:05d}" for i in range(1, n_contracts + 1)]
    
    # Generate monthly dates
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(n_months)]
    
    # Portfolio segments
    segments = ['Premium', 'Standard', 'Subprime']
    
    data = []
    
    for contract in contracts:
        # Assign segment based on contract characteristics
        segment = np.random.choice(segments, p=[0.3, 0.5, 0.2])
        
        # Initialize starting risk profile based on segment
        if segment == 'Premium':
            current_dpd = np.random.choice([0, 15], p=[0.9, 0.1])
        elif segment == 'Standard':
            current_dpd = np.random.choice([0, 15, 30], p=[0.7, 0.2, 0.1])
        else:  # Subprime
            current_dpd = np.random.choice([0, 15, 30, 60], p=[0.5, 0.2, 0.2, 0.1])
        
        for date in dates:
            # Simulate risk evolution over time
            transition_prob = np.random.random()
            
            if segment == 'Premium':
                # Premium: very stable, low deterioration
                if transition_prob < 0.05:  # 5% chance to worsen
                    current_dpd = min(current_dpd + 15, 360)
                elif transition_prob < 0.15 and current_dpd > 0:  # 10% chance to improve
                    current_dpd = max(current_dpd - 15, 0)
                    
            elif segment == 'Standard':
                # Standard: moderate risk evolution
                if transition_prob < 0.08:  # 8% chance to worsen
                    current_dpd = min(current_dpd + 30, 360)
                elif transition_prob < 0.18 and current_dpd > 0:  # 10% chance to improve
                    current_dpd = max(current_dpd - 15, 0)
                    
            else:  # Subprime
                # Subprime: higher volatility
                if transition_prob < 0.12:  # 12% chance to worsen
                    current_dpd = min(current_dpd + 45, 360)
                elif transition_prob < 0.20 and current_dpd > 0:  # 8% chance to improve
                    current_dpd = max(current_dpd - 30, 0)
            
            # Add some exposure amount for potential weighted analysis
            exposure = np.random.uniform(10000, 500000)
            
            data.append({
                'id_contrato': contract,
                'data_ref': date,
                'dias_atraso': current_dpd,
                'segmento': segment,
                'valor_exposicao': round(exposure, 2)
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df):,} observations")
    print(f"Date range: {df['data_ref'].min()} to {df['data_ref'].max()}")
    print(f"Segments: {df['segmento'].value_counts().to_dict()}")
    
    return df


def main():
    """Main example workflow."""
    
    print("=" * 60)
    print("CREDIT RISK TRANSITION MATRIX LIBRARY - BASIC EXAMPLE")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n1. Creating Sample Portfolio Data")
    print("-" * 40)
    df = create_sample_data(n_contracts=300, n_months=18)
    
    # Display sample of the data
    print("\nSample of generated data:")
    print(df.head(10))
    
    # Step 2: Initialize the learner
    print("\n\n2. Initializing TransitionMatrixLearner")
    print("-" * 40)
    
    learner = TransitionMatrixLearner(
        buckets=[0, 15, 30, 60, 90, 120, 180, 240, 360],
        time_horizon=12,
        min_observations=50
    )
    
    print(f"Learner initialized: {learner}")
    print(f"Bucket definitions: {learner.buckets_}")
    print(f"Bucket labels: {learner.bucket_labels_}")
    
    # Step 3: Fit the model (global analysis)
    print("\n\n3. Fitting Global Transition Matrix")
    print("-" * 40)
    
    learner.fit(
        df,
        id_col='id_contrato',
        time_col='data_ref',
        bucket_col='dias_atraso'
    )
    
    print("✅ Model fitted successfully!")
    print(f"Transition matrix shape: {learner.transition_matrix_.shape}")
    
    # Display the global transition matrix
    print("\nGlobal Transition Matrix (probabilities):")
    print(learner.transition_matrix_.round(3))
    
    # Step 4: Fit with segmented analysis
    print("\n\n4. Fitting Segmented Analysis")
    print("-" * 40)
    
    learner_segmented = TransitionMatrixLearner(
        buckets=[0, 15, 30, 60, 90, 120, 180, 240, 360],
        time_horizon=12,
        min_observations=30  # Lower threshold for segments
    )
    
    learner_segmented.fit(
        df,
        id_col='id_contrato',
        time_col='data_ref',
        bucket_col='dias_atraso',
        segment_col='segmento'
    )
    
    print("✅ Segmented model fitted successfully!")
    print(f"Number of segments: {len(learner_segmented.segmented_matrices_)}")
    print(f"Segments: {list(learner_segmented.segmented_matrices_.keys())}")
    
    # Step 5: Generate visualizations
    print("\n\n5. Generating Visualizations")
    print("-" * 40)
    
    # Create output directory
    import os
    os.makedirs("./output", exist_ok=True)
    
    # Generate global heatmap
    print("Generating global heatmap...")
    fig_global = learner.plot_heatmaps(
        modes=['global'],
        save_dir='./output'
    )
    
    # Generate segmented heatmaps
    print("Generating segmented heatmaps...")
    fig_segments = learner_segmented.plot_heatmaps(
        modes=['global', 'segmented'],
        save_dir='./output'
    )
    
    print("✅ Visualizations saved to ./output/")
    
    # Step 6: Calculate PD metrics
    print("\n\n6. Calculating Probability of Default (PD)")
    print("-" * 40)
    
    pd_results = learner.calculate_pd(
        time_horizon=12,
        default_buckets=['240-359', '360+']  # Consider these as default
    )
    
    print("PD Results (12-month horizon):")
    for bucket, pd_value in pd_results['pd_by_bucket'].items():
        print(f"  {bucket:>10}: {pd_value:>6.2%}")
    
    # Step 7: Predictive analysis
    print("\n\n7. Predictive Analysis Examples")
    print("-" * 40)
    
    # Example 1: Predict from current state
    current_bucket = '30-59'
    future_dist = learner.predict_transitions(current_bucket, n_periods=6)
    
    print(f"\nIf a loan is currently in bucket '{current_bucket}',")
    print("after 6 months the probability distribution will be:")
    
    for i, (bucket, prob) in enumerate(zip(learner.bucket_labels_, future_dist)):
        print(f"  {bucket:>10}: {prob:>6.2%}")
    
    # Example 2: Portfolio-level prediction
    print(f"\nPortfolio stress test - what if all loans start in '60-89' bucket:")
    stressed_dist = learner.predict_transitions('60-89', n_periods=12)
    default_prob = sum(stressed_dist[-2:])  # Last two buckets
    print(f"Default probability after 12 months: {default_prob:.2%}")
    
    # Step 8: Model validation (basic)
    print("\n\n8. Model Summary and Validation")
    print("-" * 40)
    
    # Get transformation results
    matrices = learner_segmented.transform(modes=['global', 'segmented'])
    
    print(f"Global matrix diagonal (stability): ")
    global_diag = np.diag(matrices['global'].values)
    for bucket, stability in zip(learner.bucket_labels_, global_diag):
        print(f"  {bucket:>10}: {stability:>6.2%}")
    
    print(f"\nSegment comparison (Premium vs Subprime stability in '0-14' bucket):")
    if 'Premium' in matrices['segmented'] and 'Subprime' in matrices['segmented']:
        premium_stability = matrices['segmented']['Premium'].loc['0-14', '0-14']
        subprime_stability = matrices['segmented']['Subprime'].loc['0-14', '0-14']
        print(f"  Premium:  {premium_stability:>6.2%}")
        print(f"  Subprime: {subprime_stability:>6.2%}")
        print(f"  Difference: {premium_stability - subprime_stability:>+6.2%}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"✅ Processed {len(df):,} observations")
    print(f"✅ Analyzed {df['id_contrato'].nunique():,} contracts")
    print(f"✅ Generated transition matrices for {len(learner_segmented.segmented_matrices_)} segments")
    print(f"✅ Calculated PD metrics for {len(pd_results['pd_by_bucket'])} risk buckets")
    print(f"✅ Saved visualizations to ./output/")
    
    print("\nFiles generated:")
    print("  - ./output/global_heatmap.png")
    for segment in learner_segmented.segmented_matrices_.keys():
        print(f"  - ./output/heatmap_{segment}.png")
    
    print("\nNext steps:")
    print("  - Review the generated heatmaps")
    print("  - Analyze PD metrics for portfolio management")
    print("  - Use transition matrices for stress testing")
    print("  - Compare segment performance")


if __name__ == "__main__":
    main()
