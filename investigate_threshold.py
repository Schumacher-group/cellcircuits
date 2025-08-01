#!/usr/bin/env python3
"""
Script to investigate why saddle points disappear above a certain alpha2 threshold.
Visualizes nullclines and separatrices as alpha2 increases.
"""

import numpy as np
import matplotlib.pyplot as plt
from my_function import CellDynamicsAnalyzer, params
import matplotlib.patches as patches

def plot_nullclines_and_phase_portrait(analyzer, title_suffix="", ax=None):
    """Plot nullclines, fixed points, and separatrices."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create grid for plotting
    mF_range = np.logspace(0, 6, 100)  # 1 to 1e6
    M_range = np.logspace(0, 6, 100)   # 1 to 1e6
    
    # Compute nullclines
    mF_nullcline_M = []
    M_nullcline_mF = []
    
    # mF nullcline: dmF/dt = 0
    for mF in mF_range:
        try:
            # From the ODE: dmF/dt = mF * ((lambda1 * PDGF / (k1 + PDGF)) * (1 - mF/K) - mu1) = 0
            # This gives us PDGF as a function of mF, then we can find M
            p = analyzer.params
            if mF < p['K']:
                PDGF_needed = (p['mu1'] * p['k1']) / (p['lambda1'] * (1 - mF/p['K']) - p['mu1'])
                if PDGF_needed > 0:
                    # From PDGF steady state: PDGF = (alpha2 * mF * M) / (gamma + M)
                    # Solve for M: M = (gamma * PDGF_needed) / (alpha2 * mF - PDGF_needed)
                    denominator = p['alpha2'] * mF - PDGF_needed
                    if denominator > 0:
                        M = (p['gamma'] * PDGF_needed) / denominator
                        if M > 0:
                            mF_nullcline_M.append(M)
                        else:
                            mF_nullcline_M.append(np.nan)
                    else:
                        mF_nullcline_M.append(np.nan)
                else:
                    mF_nullcline_M.append(np.nan)
            else:
                mF_nullcline_M.append(np.nan)
        except:
            mF_nullcline_M.append(np.nan)
    
    # M nullcline: dM/dt = 0
    for M in M_range:
        try:
            # From the ODE: dM/dt = M * ((lambda2 * CSF / (k2 + CSF)) - mu2) = 0
            # This gives us CSF as a function of M, then we can find mF
            p = analyzer.params
            CSF_needed = (p['mu2'] * p['k2']) / (p['lambda2'] - p['mu2'])
            if CSF_needed > 0:
                # From CSF steady state: CSF = (beta3 * mF) / (gamma + M)
                # Solve for mF: mF = (CSF_needed * (gamma + M)) / beta3
                mF = (CSF_needed * (p['gamma'] + M)) / p['beta3']
                if mF > 0:
                    M_nullcline_mF.append(mF)
                else:
                    M_nullcline_mF.append(np.nan)
            else:
                M_nullcline_mF.append(np.nan)
        except:
            M_nullcline_mF.append(np.nan)
    
    # Plot nullclines
    ax.loglog(mF_range, mF_nullcline_M, 'r-', linewidth=2, label='mF nullcline (dmF/dt=0)')
    ax.loglog(M_nullcline_mF, M_range, 'b-', linewidth=2, label='M nullcline (dM/dt=0)')
    
    # Analyze the system
    try:
        analysis = analyzer.analyze_system()
        
        # Plot fixed points
        if analysis['fixed_points']:
            for i, fp in enumerate(analysis['fixed_points']):
                ax.loglog(fp[0], fp[1], 'go', markersize=8, label=f'Fixed point {i+1}' if i == 0 else "")
        
        # Plot cold fibrosis points
        if analysis['cold_fibrosis_points']:
            for i, cf in enumerate(analysis['cold_fibrosis_points']):
                ax.loglog(cf[0], cf[1], 'bs', markersize=8, label=f'CF point {i+1}' if i == 0 else "")
        
        # Plot saddle points
        if analysis['saddle_points']:
            for i, sp in enumerate(analysis['saddle_points']):
                ax.loglog(sp[0], sp[1], 'ro', markersize=10, markerfacecolor='none', 
                         markeredgewidth=2, label=f'Saddle {i+1}' if i == 0 else "")
        
        # Plot separatrices
        if analysis['separatrices']:
            for i, sep in enumerate(analysis['separatrices']):
                if len(sep) > 0:
                    ax.loglog(sep[:, 0], sep[:, 1], 'k--', linewidth=1.5, alpha=0.7,
                             label=f'Separatrix {i+1}' if i == 0 else "")
        
        # Add info to title
        n_fp = len(analysis['fixed_points'])
        n_cf = len(analysis['cold_fibrosis_points'])
        n_saddles = len(analysis['saddle_points'])
        n_sep = len([s for s in analysis['separatrices'] if len(s) > 0])
        
        info = f" (FP:{n_fp}, CF:{n_cf}, Saddles:{n_saddles}, Sep:{n_sep})"
        ax.set_title(f'Phase Portrait{title_suffix}{info}')
        
    except Exception as e:
        ax.set_title(f'Phase Portrait{title_suffix} (Analysis failed: {str(e)})')
    
    ax.set_xlabel('mF (log scale)')
    ax.set_ylabel('M (log scale)')
    ax.set_xlim(1, 1e6)
    ax.set_ylim(1, 1e6)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

def investigate_alpha2_threshold(beta3_value=1.0, alpha2_values=None):
    """Investigate how the system changes as alpha2 increases."""
    
    if alpha2_values is None:
        # Start from baseline and increase alpha2
        base_alpha2 = params['alpha2']
        alpha2_values = np.linspace(0.5, 3.0, 8) * base_alpha2
    
    print(f"Investigating alpha2 threshold for beta3 = {beta3_value}")
    alpha2_folds = [a/params['alpha2'] for a in alpha2_values]
    print(f"Testing alpha2 values: {alpha2_folds} × baseline")
    
    # Create subplots
    n_plots = len(alpha2_values)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Store results for summary
    results = []
    
    for i, alpha2_val in enumerate(alpha2_values):
        # Create modified parameters
        test_params = params.copy()
        test_params['alpha2'] = alpha2_val
        test_params['beta3'] = params['beta3'] * beta3_value
        
        # Create analyzer
        analyzer = CellDynamicsAnalyzer(test_params)
        
        # Plot
        ax = axes[i] if i < len(axes) else None
        if ax is not None:
            alpha2_fold = alpha2_val / params['alpha2']
            plot_nullclines_and_phase_portrait(analyzer, f' α2={alpha2_fold:.2f}×', ax)
        
        # Collect results
        try:
            analysis = analyzer.analyze_system()
            result = {
                'alpha2_fold': alpha2_val / params['alpha2'],
                'alpha2_abs': alpha2_val,
                'n_fixed_points': len(analysis['fixed_points']),
                'n_cf_points': len(analysis['cold_fibrosis_points']),
                'n_saddles': len(analysis['saddle_points']),
                'n_separatrices': len([s for s in analysis['separatrices'] if len(s) > 0]),
                'has_saddles': len(analysis['saddle_points']) > 0
            }
        except Exception as e:
            result = {
                'alpha2_fold': alpha2_val / params['alpha2'],
                'alpha2_abs': alpha2_val,
                'error': str(e),
                'has_saddles': False
            }
        
        results.append(result)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'α2 fold':<10} {'α2 abs':<12} {'FP':<4} {'CF':<4} {'Saddles':<8} {'Sep':<4} {'Has Saddles'}")
    print("-"*80)
    
    threshold_alpha2 = None
    for result in results:
        if 'error' in result:
            print(f"{result['alpha2_fold']:<10.2f} {result['alpha2_abs']:<12.0f} ERROR: {result['error']}")
        else:
            print(f"{result['alpha2_fold']:<10.2f} {result['alpha2_abs']:<12.0f} "
                  f"{result['n_fixed_points']:<4} {result['n_cf_points']:<4} "
                  f"{result['n_saddles']:<8} {result['n_separatrices']:<4} {result['has_saddles']}")
            
            # Find threshold
            if not result['has_saddles'] and threshold_alpha2 is None:
                threshold_alpha2 = result['alpha2_fold']
    
    if threshold_alpha2:
        print(f"\nThreshold detected: Saddle points disappear above α2 ≈ {threshold_alpha2:.2f} × baseline")
    else:
        print("\nNo clear threshold detected in this range.")
    
    return results

def zoom_around_threshold(beta3_value=1.0, threshold_estimate=2.0, n_points=6):
    """Zoom in around the estimated threshold to understand the transition better."""
    
    print(f"\nZooming around threshold α2 ≈ {threshold_estimate:.2f} × baseline")
    
    # Create fine-grained alpha2 values around the threshold
    base_alpha2 = params['alpha2']
    alpha2_range = np.linspace(threshold_estimate - 0.5, threshold_estimate + 0.5, n_points)
    alpha2_values = alpha2_range * base_alpha2
    
    results = investigate_alpha2_threshold(beta3_value, alpha2_values)
    
    return results

def main():
    """Main analysis function."""
    print("Investigating α2 threshold behavior...")
    print("="*60)
    
    # Set deterministic behavior
    np.random.seed(42)
    
    # Test different beta3 values - expanded range for robust fitting
    beta3_values = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]
    
    for beta3 in beta3_values:
        print(f"\n{'='*60}")
        print(f"TESTING β3 = {beta3} × baseline")
        print(f"{'='*60}")
        
        # Broad range first
        results = investigate_alpha2_threshold(beta3_value=beta3)
        
        # Find approximate threshold
        threshold = None
        for i, result in enumerate(results):
            if not result.get('has_saddles', False):
                if i > 0:
                    threshold = (results[i-1]['alpha2_fold'] + result['alpha2_fold']) / 2
                else:
                    threshold = result['alpha2_fold']
                break
        
        # Zoom in if threshold found
        if threshold:
            print(f"\nZooming around estimated threshold: {threshold:.2f}")
            zoom_results = zoom_around_threshold(beta3, threshold, n_points=8)

if __name__ == "__main__":
    main()
