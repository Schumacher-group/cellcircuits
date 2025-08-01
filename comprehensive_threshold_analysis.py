"""
Comprehensive analysis of threshold behavior and trajectory dynamics
when saddle points disappear.
"""

from my_function import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def investigate_trajectory_behavior_no_saddles(params, beta3_fold, alpha2_fold, n_trajectories=50):
    """
    Investigate where trajectories end up when there are no saddle points.
    Test multiple initial conditions to see if they all flow to healing state.
    """
    params_new = params.copy()
    params_new['alpha2'] *= alpha2_fold
    params_new['beta3'] *= beta3_fold
    
    # Create analyzer
    analyzer = CellDynamicsAnalyzer(params_new)
    analysis = analyzer.analyze_system()
    
    # Time span for integration
    t_span = np.linspace(0, 100, 1000)
    
    # Generate random initial conditions across the phase space
    np.random.seed(42)  # For reproducibility
    mF_init = np.logspace(-1, 3, n_trajectories)
    M_init = np.logspace(-1, 3, n_trajectories)
    
    # Track trajectory endpoints
    endpoints = []
    trajectory_data = []
    
    for i in range(n_trajectories):
        y0 = [mF_init[i], M_init[i]]
        
        try:
            # Integrate trajectory
            sol = odeint(lambda y, t: analyzer.mf_m_ode_system(y[0], y[1]), y0, t_span)
            
            # Store trajectory and endpoint
            trajectory_data.append(sol)
            endpoints.append(sol[-1])
            
        except Exception as e:
            print(f"Integration failed for IC ({mF_init[i]:.2f}, {M_init[i]:.2f}): {e}")
            continue
    
    endpoints = np.array(endpoints)
    
    # Analyze endpoints
    result = {
        'n_saddles': len(analysis['saddle_points']),
        'n_fixed_points': len(analysis['fixed_points']),
        'endpoints': endpoints,
        'trajectories': trajectory_data,
        'initial_conditions': np.column_stack([mF_init, M_init]),
        'params': params_new,
        'analysis': analysis
    }
    
    return result

def plot_trajectory_analysis(result, title_suffix=""):
    """Plot trajectory analysis results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Phase portrait with trajectories
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot trajectories
    for i, traj in enumerate(result['trajectories'][:20]):  # Show first 20 trajectories
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)
        # Mark initial condition
        ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=4, alpha=0.7)
        # Mark endpoint
        ax1.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=4, alpha=0.7)
    
    # Plot fixed points if they exist
    if result['analysis']['fixed_points']:
        fps = np.array(result['analysis']['fixed_points'])
        ax1.scatter(fps[:, 0], fps[:, 1], c='purple', s=100, marker='*', 
                   label=f'Fixed Points ({len(fps)})', zorder=5)
    
    ax1.set_xlabel('mF')
    ax1.set_ylabel('M')
    ax1.set_title(f'Trajectories {title_suffix}\n(Saddles: {result["n_saddles"]}, FPs: {result["n_fixed_points"]})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.1, 1000)
    ax1.set_ylim(0.1, 1000)
    
    # Plot 2: Endpoint distribution
    if len(result['endpoints']) > 0:
        endpoints = result['endpoints']
        
        # Log-scale scatter plot of endpoints
        ax2.scatter(endpoints[:, 0], endpoints[:, 1], alpha=0.6, s=30)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('mF (endpoint)')
        ax2.set_ylabel('M (endpoint)')
        ax2.set_title(f'Trajectory Endpoints Distribution\n({len(endpoints)} trajectories)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mF_mean = np.mean(endpoints[:, 0])
        M_mean = np.mean(endpoints[:, 1])
        mF_std = np.std(endpoints[:, 0])
        M_std = np.std(endpoints[:, 1])
        
        ax2.text(0.05, 0.95, f'mF: {mF_mean:.2e} ± {mF_std:.2e}\nM: {M_mean:.2e} ± {M_std:.2e}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def comprehensive_threshold_analysis():
    """Run comprehensive threshold analysis with more data points."""
    
    # Expanded beta3 range for robust fitting
    beta3_values = np.array([0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0])
    alpha2_thresholds = []
    
    print("Running comprehensive threshold analysis...")
    
    for beta3_fold in beta3_values:
        print(f"\nTesting β3 = {beta3_fold}...")
        
        # Test alpha2 range
        alpha2_test = np.linspace(0.5, 3.0, 15)
        
        threshold_found = None
        
        for alpha2_fold in alpha2_test:
            params_new = params.copy()
            params_new['alpha2'] *= alpha2_fold
            params_new['beta3'] *= beta3_fold
            
            try:
                analyzer = CellDynamicsAnalyzer(params_new)
                analysis = analyzer.analyze_system()
                
                n_saddles = len(analysis['saddle_points'])
                
                # If no saddles found, this is above threshold
                if n_saddles == 0:
                    threshold_found = alpha2_fold
                    break
                    
            except Exception as e:
                # If analysis fails, consider this above threshold
                threshold_found = alpha2_fold
                break
        
        if threshold_found is not None:
            # Refine threshold with binary search
            alpha2_low = max(0.5, threshold_found - 0.2)
            alpha2_high = threshold_found
            
            for _ in range(10):  # 10 iterations of binary search
                alpha2_mid = (alpha2_low + alpha2_high) / 2
                
                params_new = params.copy()
                params_new['alpha2'] *= alpha2_mid
                params_new['beta3'] *= beta3_fold
                
                try:
                    analyzer = CellDynamicsAnalyzer(params_new)
                    analysis = analyzer.analyze_system()
                    n_saddles = len(analysis['saddle_points'])
                    
                    if n_saddles > 0:
                        alpha2_low = alpha2_mid
                    else:
                        alpha2_high = alpha2_mid
                        
                except:
                    alpha2_high = alpha2_mid
            
            threshold_found = (alpha2_low + alpha2_high) / 2
        
        alpha2_thresholds.append(threshold_found if threshold_found else np.nan)
        print(f"  Threshold: {threshold_found:.3f}" if threshold_found else "  No threshold found")
    
    return beta3_values, np.array(alpha2_thresholds)

def plot_enhanced_threshold_summary(beta3_values, alpha2_thresholds):
    """Plot enhanced threshold summary with robust fitting."""
    
    # Remove NaN values for fitting
    valid_mask = ~np.isnan(alpha2_thresholds)
    beta3_fit = beta3_values[valid_mask]
    alpha2_fit = alpha2_thresholds[valid_mask]
    
    plt.figure(figsize=(10, 7))
    
    # Plot data points
    plt.plot(beta3_fit, alpha2_fit, 'bo', markersize=8, label='Threshold Data')
    
    # Add data point labels
    for i, (b3, a2) in enumerate(zip(beta3_fit, alpha2_fit)):
        plt.annotate(f'({b3:.1f}, {a2:.2f})', 
                    (b3, a2), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Fit linear relationship
    if len(beta3_fit) >= 2:
        z = np.polyfit(beta3_fit, alpha2_fit, 1)
        p = np.poly1d(z)
        
        # Calculate R-squared
        alpha2_pred = p(beta3_fit)
        ss_res = np.sum((alpha2_fit - alpha2_pred) ** 2)
        ss_tot = np.sum((alpha2_fit - np.mean(alpha2_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Plot fit line
        x_fit = np.linspace(min(beta3_fit), max(beta3_fit), 100)
        plt.plot(x_fit, p(x_fit), 'r--', alpha=0.8, linewidth=2,
                label=f'Linear fit: α2_thresh = {z[0]:.3f}β3 + {z[1]:.3f}\nR² = {r_squared:.4f}')
        
        # Add confidence interval (rough estimate)
        residuals = alpha2_fit - alpha2_pred
        std_error = np.std(residuals)
        
        plt.fill_between(x_fit, p(x_fit) - 2*std_error, p(x_fit) + 2*std_error, 
                        alpha=0.2, color='red', label='±2σ confidence band')
    
    plt.xlabel('β3 fold change')
    plt.ylabel('α2 threshold (fold change)')
    plt.title('Enhanced Saddle Point Disappearance Threshold\n(More data points for robust fitting)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf(), z if len(beta3_fit) >= 2 else None

def investigate_no_saddle_scenarios():
    """Investigate specific cases where no saddle points exist."""
    
    print("\n" + "="*60)
    print("INVESTIGATING TRAJECTORY BEHAVIOR WITH NO SADDLE POINTS")
    print("="*60)
    
    # Test cases where we know saddle points disappear
    test_cases = [
        {'beta3': 0.5, 'alpha2': 1.5, 'name': 'β3=0.5, α2=1.5 (above threshold)'},
        {'beta3': 1.0, 'alpha2': 2.0, 'name': 'β3=1.0, α2=2.0 (above threshold)'},
        {'beta3': 1.5, 'alpha2': 3.0, 'name': 'β3=1.5, α2=3.0 (above threshold)'},
        {'beta3': 1.0, 'alpha2': 1.5, 'name': 'β3=1.0, α2=1.5 (below threshold - control)'}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nAnalyzing: {case['name']}")
        
        result = investigate_trajectory_behavior_no_saddles(
            params, case['beta3'], case['alpha2'], n_trajectories=100)
        
        results.append((case, result))
        
        # Print summary
        endpoints = result['endpoints']
        if len(endpoints) > 0:
            # Check if trajectories converge to a common region
            mF_range = np.log10(np.max(endpoints[:, 0])) - np.log10(np.min(endpoints[:, 0]))
            M_range = np.log10(np.max(endpoints[:, 1])) - np.log10(np.min(endpoints[:, 1]))
            
            print(f"  Saddle points: {result['n_saddles']}")
            print(f"  Fixed points: {result['n_fixed_points']}")
            print(f"  Trajectories analyzed: {len(endpoints)}")
            print(f"  Endpoint spread (log10): mF={mF_range:.2f}, M={M_range:.2f}")
            
            # Check if endpoints cluster (indicating convergence)
            if mF_range < 1.0 and M_range < 1.0:  # Within 1 order of magnitude
                print("  → Trajectories CONVERGE to common region")
                print(f"  → Mean endpoint: mF={np.mean(endpoints[:, 0]):.2e}, M={np.mean(endpoints[:, 1]):.2e}")
            else:
                print("  → Trajectories DIVERGE or show complex behavior")
    
    return results

# Run comprehensive analysis
if __name__ == "__main__":
    print("Starting comprehensive threshold and trajectory analysis...")
    
    # 1. Enhanced threshold analysis with more data points
    print("\n1. Running enhanced threshold analysis...")
    beta3_vals, alpha2_thresh = comprehensive_threshold_analysis()
    
    # 2. Plot enhanced threshold summary
    print("\n2. Creating enhanced threshold plot...")
    fig1, fit_params = plot_enhanced_threshold_summary(beta3_vals, alpha2_thresh)
    plt.show()
    
    # 3. Investigate trajectory behavior when no saddle points
    print("\n3. Investigating trajectory behavior without saddle points...")
    trajectory_results = investigate_no_saddle_scenarios()
    
    # 4. Plot trajectory analyses
    print("\n4. Creating trajectory analysis plots...")
    for i, (case, result) in enumerate(trajectory_results):
        fig = plot_trajectory_analysis(result, f"- {case['name']}")
        plt.show()
    
    # 5. Summary of findings
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    
    if fit_params is not None:
        print(f"Enhanced threshold relationship: α2_thresh = {fit_params[0]:.3f}β3 + {fit_params[1]:.3f}")
    
    print("\nTrajectory behavior summary:")
    for case, result in trajectory_results:
        endpoints = result['endpoints']
        if len(endpoints) > 0:
            mF_range = np.log10(np.max(endpoints[:, 0])) - np.log10(np.min(endpoints[:, 0]))
            M_range = np.log10(np.max(endpoints[:, 1])) - np.log10(np.min(endpoints[:, 1]))
            converges = mF_range < 1.0 and M_range < 1.0
            print(f"  {case['name']}: {'Converges' if converges else 'Diverges'} (saddles: {result['n_saddles']})")
    
    print("\nAnalysis complete!")
