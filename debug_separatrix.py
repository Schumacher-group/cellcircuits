#!/usr/bin/env python3
"""
Debug script to investigate separatrix finding inconsistencies.
"""

import numpy as np
import pandas as pd
from my_function import *

# Set deterministic behavior
np.random.seed(42)

def debug_single_parameter_case(param_x, param_y, fold1, fold2, runs=5):
    """Run the same parameter case multiple times to check consistency."""
    
    print("=" * 50)
    print("DEBUGGING: {}={:.2f}, {}={:.2f}".format(param_x, fold1, param_y, fold2))
    print("=" * 50)
    
    results = []
    
    for run in range(runs):
        print("\n--- RUN {} ---".format(run + 1))
        
        # Reset seed for each run to check determinism
        np.random.seed(42)
        
        params_new = params.copy()
        params_new[param_x] *= fold1
        params_new[param_y] *= fold2
        
        try:
            analyzer = CellDynamicsAnalyzer(params_new)
            analysis = analyzer.analyze_system()
            
            result = {
                'run': run + 1,
                'fixed_points': len(analysis['fixed_points']),
                'cf_points': len(analysis['cold_fibrosis_points']),
                'saddles': len(analysis['saddle_points']),
                'separatrices_count': len(analysis['separatrices']),
                'fp_coords': [tuple(fp) for fp in analysis['fixed_points']],
                'cf_coords': [tuple(fp) for fp in analysis['cold_fibrosis_points']],
                'saddle_coords': [tuple(fp) for fp in analysis['saddle_points']],
            }
            
            if analysis['separatrices']:
                result['sep_shapes'] = [sep.shape for sep in analysis['separatrices']]
                result['sep_finite'] = [np.all(np.isfinite(sep)) for sep in analysis['separatrices']]
            else:
                result['sep_shapes'] = []
                result['sep_finite'] = []
                
            results.append(result)
            
            print("Fixed points: {} at {}".format(result['fixed_points'], result['fp_coords']))
            print("CF points: {} at {}".format(result['cf_points'], result['cf_coords']))
            print("Saddles: {} at {}".format(result['saddles'], result['saddle_coords']))
            print("Separatrices: {} with shapes {}".format(result['separatrices_count'], result['sep_shapes']))
            
        except Exception as e:
            print("ERROR in run {}: {}".format(run + 1, str(e)))
            results.append({
                'run': run + 1,
                'error': str(e)
            })
    
    # Check consistency
    print("\n" + "=" * 30)
    print("CONSISTENCY CHECK")
    print("=" * 30)
    
    if len(results) > 1:
        first_result = results[0]
        all_consistent = True
        
        for i, result in enumerate(results[1:], 2):
            if 'error' in result or 'error' in first_result:
                continue
                
            for key in ['fixed_points', 'cf_points', 'saddles', 'separatrices_count']:
                if result[key] != first_result[key]:
                    print("INCONSISTENCY in {}: run 1={}, run {}={}".format(
                        key, first_result[key], i, result[key]))
                    all_consistent = False
                    
            # Check coordinate consistency (with tolerance for numerical differences)
            for coord_key in ['fp_coords', 'cf_coords', 'saddle_coords']:
                if coord_key in result and coord_key in first_result:
                    if len(result[coord_key]) == len(first_result[coord_key]):
                        for j, (coord1, coord2) in enumerate(zip(first_result[coord_key], result[coord_key])):
                            if not np.allclose(coord1, coord2, rtol=1e-10, atol=1e-12):
                                print("COORDINATE INCONSISTENCY in {} index {}: run 1={}, run {}={}".format(
                                    coord_key, j, coord1, i, coord2))
                                all_consistent = False
        
        if all_consistent:
            print("ALL RUNS CONSISTENT!")
        else:
            print("INCONSISTENCIES DETECTED!")
    
    return results

def debug_problematic_cases():
    """Debug the specific cases that show inconsistencies."""
    
    # Test some problematic parameter combinations
    test_cases = [
        ('alpha2', 'beta3', 0.5, 1.25),
        ('alpha2', 'beta3', 1.0, 1.0),  # baseline
        ('alpha2', 'beta3', 2.0, 2.0),  # extreme case
    ]
    
    all_results = {}
    
    for param_x, param_y, fold1, fold2 in test_cases:
        case_key = "{}_{:.2f}_{}_{:.2f}".format(param_x, fold1, param_y, fold2)
        all_results[case_key] = debug_single_parameter_case(param_x, param_y, fold1, fold2, runs=3)
    
    return all_results

if __name__ == "__main__":
    print("Starting separatrix debugging...")
    results = debug_problematic_cases()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for case, case_results in results.items():
        print("\nCase {}: {} runs".format(case, len(case_results)))
        errors = [r for r in case_results if 'error' in r]
        if errors:
            print("  {} runs had errors".format(len(errors)))
        else:
            print("  All runs completed successfully")
