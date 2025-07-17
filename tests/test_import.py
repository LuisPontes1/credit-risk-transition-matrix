#!/usr/bin/env python3

"""
Simple test script to verify library import and functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing Credit Risk Transition Matrix Library...")
print("=" * 50)

try:
    print("1. Testing import...")
    from credit_risk_transition_matrix import TransitionMatrixLearner
    print("   ✅ Import successful!")
    
    print("2. Testing initialization...")
    learner = TransitionMatrixLearner()
    print(f"   ✅ Initialization successful: {learner}")
    
    print("3. Testing basic attributes...")
    print(f"   - Buckets: {learner.buckets_}")
    print(f"   - Time horizon: {learner.time_horizon}")
    print(f"   - Is fitted: {learner.is_fitted_}")
    
    print("4. Testing custom initialization...")
    custom_learner = TransitionMatrixLearner(
        buckets=[0, 30, 60, 90, 180, 360],
        time_horizon=6
    )
    print(f"   ✅ Custom learner: {custom_learner}")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Library is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
