#!/usr/bin/env python3
"""
Debug script to investigate why separatrix area computation fails for certain parameter combinations.
"""

import numpy as np
from my_function import CellDynamicsAnalyzer

def analyze_separatrix_values(alpha2, beta3):
    """Analyze separatrix values in detail."""
    print(f"\n=== ANALYZING alpha2={alpha2}, beta3={beta3} ===")
    
    # Set parameters using the default values from my_function.py
    base_params = {
        'lambda1': 0.9, 'lambda2': 0.8, 'mu1': 0.3, 'mu2': 0.3,
        'K': 1e6, 'k1': 1e9, 'k2': 1e9, 'beta1': 470*60*24,
        'beta2': 70*60*24, 'beta3': 240*60*24, 'alpha1': 940*60*24,
        'alpha2': 510*60*24, 'gamma': 2
    }
    
    # Scale alpha2 and beta3 by the requested factors
    params = base_params.copy()
    params['alpha2'] = base_params['alpha2'] * alpha2
    params['beta3'] = base_params['beta3'] * beta3
    
    analyzer = CellDynamicsAnalyzer(params)
    
    try:
        # First, let's debug the separatrix computation step by step
        fixed_points = analyzer.compute_fixed_points()
        cf_points = analyzer.cold_fibrosis_point()
        all_fixed_points = analyzer.combine_fixed_points(fixed_points, cf_points)
        saddle_points = analyzer.find_saddle_points(all_fixed_points)
        
        print(f"Found {len(saddle_points)} saddle points:")
        for i, sp in enumerate(saddle_points):
            print(f"  Saddle {i+1}: ({sp[0]:.6f}, {sp[1]:.6f})")
        
        # Now debug the separatrix computation with detailed output
        print(f"\nComputing separatrices...")
        debug_separatrices = []
        
        from scipy.integrate import odeint
        t_span = np.linspace(0, 200, 2000)
        eps = 1e-3
        
        for i, sp in enumerate(saddle_points):
            print(f"\n--- Saddle point {i+1}: ({sp[0]:.6f}, {sp[1]:.6f}) ---")
            
            # Compute raw trajectories
            left = odeint(analyzer.reversed_ode, [sp[0] - eps, sp[1] + eps], t_span, 
                         rtol=1e-10, atol=1e-12)
            right = odeint(analyzer.reversed_ode, [sp[0] + eps, sp[1] - eps], t_span,
                          rtol=1e-10, atol=1e-12)
            raw_separatrix = np.vstack([left[::-1], right])
            
            print(f"Raw separatrix shape: {raw_separatrix.shape}")
            print(f"Raw min values: mF={raw_separatrix[:, 0].min():.6f}, M={raw_separatrix[:, 1].min():.6f}")
            print(f"Raw max values: mF={raw_separatrix[:, 0].max():.6f}, M={raw_separatrix[:, 1].max():.6f}")
            print(f"Raw negative count: {np.sum(raw_separatrix < 0)}/{raw_separatrix.size}")
            
            # Filter out negative values
            #positive_mask = np.all(raw_separatrix > 0, axis=1)
            #positive_separatrix = raw_separatrix[positive_mask]
            #print(f"After removing negatives: {positive_separatrix.shape}")
            
            # Filter out values < 1.0 (matching the actual implementation)
            cutoff_mask = np.all(raw_separatrix >= 1.0, axis=1)
            final_separatrix = raw_separatrix[cutoff_mask]
            print(f"After cutoff at 1.0: {final_separatrix.shape}")
            
            if final_separatrix.shape[0] > 0:
                print(f"Final min values: mF={final_separatrix[:, 0].min():.6f}, M={final_separatrix[:, 1].min():.6f}")
                print(f"Final max values: mF={final_separatrix[:, 0].max():.6f}, M={final_separatrix[:, 1].max():.6f}")
            else:
                print("*** EMPTY after filtering ***")
            
            debug_separatrices.append(final_separatrix)
        
        # Analyze the system
        result = analyzer.analyze_system()
        
        print(f"\nFound {len(result['separatrices'])} separatrices")
        
        for i, sep in enumerate(result['separatrices']):
            print(f"\nSeparatrix {i+1}:")
            print(f"  Shape: {sep.shape}")
            
            if sep.shape[0] == 0:
                print(f"  *** EMPTY SEPARATRIX - all points filtered out ***")
                continue
                
            print(f"  Min values: mF={sep[:, 0].min():.6f}, M={sep[:, 1].min():.6f}")
            print(f"  Max values: mF={sep[:, 0].max():.6f}, M={sep[:, 1].max():.6f}")
            # Check for problematic values
            print(f"  Contains zeros: {np.any(sep == 0)}")
            print(f"  Contains negatives: {np.any(sep < 0)}")
            print(f"  Contains infs: {np.any(~np.isfinite(sep))}")
            
            # Count problematic values
            zero_count = np.sum(sep == 0)
            neg_count = np.sum(sep < 0)
            print(f"  Zero count: {zero_count}/{sep.size}")
            print(f"  Negative count: {neg_count}/{sep.size}")
            
            # Check first few and last few points
            print(f"  First 3 points:")
            for j in range(min(3, len(sep))):
                print(f"    [{j}]: ({sep[j, 0]:.6f}, {sep[j, 1]:.6f})")
            
            print(f"  Last 3 points:")
            for j in range(max(0, len(sep)-3), len(sep)):
                print(f"    [{j}]: ({sep[j, 0]:.6f}, {sep[j, 1]:.6f})")
            
            # Check if logarithmic transformation would work
            try:
                log_sep = np.log10(sep)
                print(f"  Log10 transformation: SUCCESS")
                print(f"    Log min: ({log_sep[:, 0].min():.3f}, {log_sep[:, 1].min():.3f})")
                print(f"    Log max: ({log_sep[:, 0].max():.3f}, {log_sep[:, 1].max():.3f})")
            except Exception as e:
                print(f"  Log10 transformation: FAILED - {e}")
        
        return result
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

def main():
    """Main debugging function."""
    np.random.seed(42)  # For reproducibility
    
    # Test the problematic parameter combination
    test_cases = [
        (1.25, 0.5),  # The problematic case
        (1.25, 1.25),   # A working case
        (2.0, 0.5),   # Another problematic case
    ]
    
    for alpha2, beta3 in test_cases:
        analyze_separatrix_values(alpha2, beta3)

if __name__ == "__main__":
    main()
