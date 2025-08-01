#!/usr/bin/env python3
"""
Test script to verify consistency of area calculations for the baseline case.
"""

import numpy as np
from my_function import CellDynamicsAnalyzer
import pandas as pd

def test_area_calculation_consistency():
    """Test if repeated area calculations give consistent results."""
    print("Testing area calculation consistency...")
    
    # Load the base separatrix
    data = pd.read_csv('separatrix.csv')
    base_separatrix = data[['mF', 'M']].values
    base_separatrix = base_separatrix[np.argsort(base_separatrix[:, 0])]
    
    # Set up the HealingAreaAnalyzer
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location('treatment_simulation', 'treatment simulation.py')
    treatment_simulation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treatment_simulation)
    HealingAreaAnalyzer = treatment_simulation.HealingAreaAnalyzer
    
    area_analyzer = HealingAreaAnalyzer()
    
    # Set up grid (same as in the main analysis)
    x_vals = np.linspace(-3, 8, 200)
    y_vals = np.linspace(-3, 8, 200)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    
    # Transform base separatrix to log scale
    base_sep_log = area_analyzer.safe_log10_transform(base_separatrix)
    
    print(f"Base separatrix shape: {base_separatrix.shape}")
    print(f"Base separatrix log shape: {base_sep_log.shape}")
    
    # Test 1: Repeated calculation of the same base area
    print("\n=== TEST 1: Base area consistency ===")
    base_areas = []
    for i in range(5):
        area = area_analyzer.compute_healing_area(base_sep_log, grid_x, grid_y)
        base_areas.append(area)
        print(f"Run {i+1}: {area}")
    
    print(f"Mean: {np.mean(base_areas)}")
    print(f"Std: {np.std(base_areas)}")
    print(f"All identical: {len(set(base_areas)) == 1}")
    
    # Test 2: Baseline case (fold = 1.0 for both parameters)
    print("\n=== TEST 2: Baseline case (alpha2=1.0, beta3=1.0) ===")
    
    # Get default parameters
    from my_function import params
    baseline_params = params.copy()
    # No scaling needed for fold = 1.0
    
    baseline_areas = []
    baseline_ratios = []
    
    for i in range(5):
        print(f"\nRun {i+1}:")
        
        # Create analyzer with baseline parameters
        analyzer = CellDynamicsAnalyzer(baseline_params)
        analysis = analyzer.analyze_system()
        
        print(f"  Found {len(analysis['separatrices'])} separatrices")
        
        if analysis['separatrices'] and len(analysis['separatrices']) > 0:
            try:
                sep_all = np.vstack(analysis['separatrices'])
                print(f"  Combined separatrix shape: {sep_all.shape}")
                
                if sep_all.shape[0] > 0:
                    sep_log = area_analyzer.safe_log10_transform(sep_all)
                    area = area_analyzer.compute_healing_area(sep_log, grid_x, grid_y)
                    ratio = area / base_areas[0]  # Use first base area as reference
                    
                    baseline_areas.append(area)
                    baseline_ratios.append(ratio)
                    
                    print(f"  Area: {area}")
                    print(f"  Ratio: {ratio}")
                else:
                    print("  Empty separatrix!")
                    baseline_areas.append(0.0)
                    baseline_ratios.append(0.0)
            except Exception as e:
                print(f"  Error: {e}")
                baseline_areas.append(np.nan)
                baseline_ratios.append(np.nan)
        else:
            print("  No separatrices found!")
            baseline_areas.append(0.0)
            baseline_ratios.append(0.0)
    
    print(f"\nBaseline areas: {baseline_areas}")
    print(f"Baseline ratios: {baseline_ratios}")
    print(f"Ratio mean: {np.nanmean(baseline_ratios)}")
    print(f"Ratio std: {np.nanstd(baseline_ratios)}")
    print(f"Expected ratio: 1.0")
    print(f"All ratios close to 1.0: {np.allclose(baseline_ratios, 1.0, rtol=1e-6, atol=1e-6)}")
    
    # Test 3: Compare base separatrix vs computed separatrix
    print("\n=== TEST 3: Base vs Computed separatrix comparison ===")
    
    analyzer = CellDynamicsAnalyzer(baseline_params)
    analysis = analyzer.analyze_system()
    
    if analysis['separatrices'] and len(analysis['separatrices']) > 0:
        computed_sep = np.vstack(analysis['separatrices'])
        
        print(f"Base separatrix shape: {base_separatrix.shape}")
        print(f"Computed separatrix shape: {computed_sep.shape}")
        
        # Compare ranges
        print(f"Base range - mF: [{base_separatrix[:, 0].min():.2f}, {base_separatrix[:, 0].max():.2f}]")
        print(f"Base range - M: [{base_separatrix[:, 1].min():.2f}, {base_separatrix[:, 1].max():.2f}]")
        print(f"Computed range - mF: [{computed_sep[:, 0].min():.2f}, {computed_sep[:, 0].max():.2f}]")
        print(f"Computed range - M: [{computed_sep[:, 1].min():.2f}, {computed_sep[:, 1].max():.2f}]")
        
        # Check if they're similar
        if computed_sep.shape[0] > 0 and base_separatrix.shape[0] > 0:
            base_log = area_analyzer.safe_log10_transform(base_separatrix)
            computed_log = area_analyzer.safe_log10_transform(computed_sep)
            
            base_area = area_analyzer.compute_healing_area(base_log, grid_x, grid_y)
            computed_area = area_analyzer.compute_healing_area(computed_log, grid_x, grid_y)
            
            print(f"Base area: {base_area}")
            print(f"Computed area: {computed_area}")
            print(f"Ratio: {computed_area / base_area if base_area > 0 else 'undefined'}")

if __name__ == "__main__":
    # Set deterministic behavior
    np.random.seed(42)
    test_area_calculation_consistency()
