"""
Focused analysis of trajectory behavior when no saddle points exist.
Specifically checking if trajectories go to healing state (origin) or infinity.
"""

from my_function import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def analyze_trajectory_fate_detailed(params, beta3_fold, alpha2_fold, n_trajectories=20):
    """
    Detailed analysis of trajectory fate when no saddle points exist.
    Check if trajectories converge to origin (healing) or diverge to infinity.
    """
    params_new = params.copy()
    params_new['alpha2'] *= alpha2_fold
    params_new['beta3'] *= beta3_fold
    
    # Create analyzer
    analyzer = CellDynamicsAnalyzer(params_new)
    analysis = analyzer.analyze_system()
    
    # Time span for integration - longer to see asymptotic behavior
    t_span = np.linspace(0, 200, 2000)
    
    # Generate initial conditions in a systematic grid
    mF_init = np.logspace(0, 2, int(np.sqrt(n_trajectories)))  # 1 to 100
    M_init = np.logspace(0, 2, int(np.sqrt(n_trajectories)))   # 1 to 100
    
    mF_grid, M_grid = np.meshgrid(mF_init, M_init)
    initial_conditions = np.column_stack([mF_grid.ravel(), M_grid.ravel()])
    
    # Track trajectory behavior
    results = {
        'converge_to_origin': 0,
        'diverge_to_infinity': 0,
        'bounded_oscillation': 0,
        'integration_failed': 0,
        'trajectories': [],
        'endpoints': [],
        'initial_conditions': initial_conditions,
        'analysis': analysis
    }
    
    healing_threshold = 0.1  # Below this is considered "healed"
    divergence_threshold = 1e4  # Above this is considered "diverged"
    
    for i, (mF0, M0) in enumerate(initial_conditions):
        try:
            # Integrate trajectory
            sol = odeint(lambda y, t: analyzer.mf_m_ode_system(y[0], y[1]), 
                        [mF0, M0], t_span, rtol=1e-10, atol=1e-12)
            
            # Check trajectory fate
            final_mF, final_M = sol[-1]
            
            if final_mF < healing_threshold and final_M < healing_threshold:
                results['converge_to_origin'] += 1
                fate = 'healing'
            elif final_mF > divergence_threshold or final_M > divergence_threshold:
                results['diverge_to_infinity'] += 1
                fate = 'diverge'
            elif np.isnan(final_mF) or np.isnan(final_M) or np.isinf(final_mF) or np.isinf(final_M):
                results['integration_failed'] += 1
                fate = 'failed'
            else:
                results['bounded_oscillation'] += 1
                fate = 'bounded'
            
            results['trajectories'].append({
                'solution': sol,
                'initial': [mF0, M0],
                'final': [final_mF, final_M],
                'fate': fate
            })
            
            results['endpoints'].append([final_mF, final_M])
            
        except Exception as e:
            results['integration_failed'] += 1
            print(f"Integration failed for IC ({mF0:.2f}, {M0:.2f}): {e}")
    
    return results

def plot_trajectory_fate_analysis(results, title_suffix=""):
    """Plot detailed trajectory fate analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Phase portrait with colored trajectories by fate
    ax1 = axes[0, 0]
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    color_map = {'healing': 'green', 'diverge': 'red', 'bounded': 'blue', 'failed': 'gray'}
    
    for traj_data in results['trajectories'][:16]:  # Show first 16 trajectories
        sol = traj_data['solution']
        fate = traj_data['fate']
        color = color_map.get(fate, 'black')
        
        # Plot trajectory
        ax1.plot(sol[:, 0], sol[:, 1], color=color, alpha=0.7, linewidth=1.5)
        # Mark initial condition
        ax1.plot(sol[0, 0], sol[0, 1], 'o', color=color, markersize=6)
        # Mark endpoint (if finite)
        if np.isfinite(sol[-1, 0]) and np.isfinite(sol[-1, 1]):
            ax1.plot(sol[-1, 0], sol[-1, 1], 's', color=color, markersize=4)
    
    # Add healing region
    ax1.axhspan(0.01, 0.1, alpha=0.2, color='green', label='Healing region')
    ax1.axvspan(0.01, 0.1, alpha=0.2, color='green')
    
    # Plot fixed points if they exist
    if results['analysis']['fixed_points']:
        fps = np.array(results['analysis']['fixed_points'])
        ax1.scatter(fps[:, 0], fps[:, 1], c='purple', s=100, marker='*', 
                   label=f'Fixed Points ({len(fps)})', zorder=5)
    
    ax1.set_xlabel('mF')
    ax1.set_ylabel('M')
    ax1.set_title(f'Trajectory Fates {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.01, 1000)
    ax1.set_ylim(0.01, 1000)
    
    # Create custom legend for trajectory fates
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='green', lw=2, label='→ Healing'),
                      Line2D([0], [0], color='red', lw=2, label='→ Diverge'),
                      Line2D([0], [0], color='blue', lw=2, label='→ Bounded'),
                      Line2D([0], [0], color='gray', lw=2, label='→ Failed')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Bar chart of trajectory fates
    ax2 = axes[0, 1]
    fates = ['Healing', 'Diverge', 'Bounded', 'Failed']
    counts = [results['converge_to_origin'], results['diverge_to_infinity'], 
              results['bounded_oscillation'], results['integration_failed']]
    colors = ['green', 'red', 'blue', 'gray']
    
    bars = ax2.bar(fates, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of trajectories')
    ax2.set_title(f'Trajectory Fate Distribution\n{title_suffix}')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
    
    # Plot 3: Initial conditions colored by fate
    ax3 = axes[1, 0]
    
    for traj_data in results['trajectories']:
        initial = traj_data['initial']
        fate = traj_data['fate']
        color = color_map.get(fate, 'black')
        ax3.scatter(initial[0], initial[1], c=color, s=50, alpha=0.7)
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Initial mF')
    ax3.set_ylabel('Initial M')
    ax3.set_title('Initial Conditions by Fate')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: System information
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info_text = f"""System Analysis {title_suffix}
    
Fixed Points: {len(results['analysis']['fixed_points'])}
Saddle Points: {len(results['analysis']['saddle_points'])}
Cold Fibrosis Points: {len(results['analysis']['cold_fibrosis_points'])}

Trajectory Analysis:
→ Healing: {results['converge_to_origin']} ({100*results['converge_to_origin']/len(results['trajectories']):.1f}%)
→ Diverge: {results['diverge_to_infinity']} ({100*results['diverge_to_infinity']/len(results['trajectories']):.1f}%)
→ Bounded: {results['bounded_oscillation']} ({100*results['bounded_oscillation']/len(results['trajectories']):.1f}%)
→ Failed: {results['integration_failed']} ({100*results['integration_failed']/len(results['trajectories']):.1f}%)

Total trajectories: {len(results['trajectories'])}"""
    
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, verticalalignment='top',
            fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main_trajectory_analysis():
    """Main function for trajectory analysis."""
    
    print("DETAILED TRAJECTORY FATE ANALYSIS")
    print("="*50)
    
    # Test cases: compare with/without saddle points
    test_cases = [
        {'beta3': 1.0, 'alpha2': 1.5, 'name': 'β3=1.0, α2=1.5 (HAS saddles)', 'expected_saddles': True},
        {'beta3': 1.0, 'alpha2': 2.2, 'name': 'β3=1.0, α2=2.2 (NO saddles)', 'expected_saddles': False},
        {'beta3': 0.5, 'alpha2': 1.5, 'name': 'β3=0.5, α2=1.5 (NO saddles)', 'expected_saddles': False},
        {'beta3': 1.5, 'alpha2': 2.0, 'name': 'β3=1.5, α2=2.0 (HAS saddles)', 'expected_saddles': True},
    ]
    
    results_all = []
    
    for case in test_cases:
        print(f"\nAnalyzing: {case['name']}")
        
        result = analyze_trajectory_fate_detailed(
            params, case['beta3'], case['alpha2'], n_trajectories=25)
        
        results_all.append((case, result))
        
        # Print summary
        total = len(result['trajectories'])
        healing_pct = 100 * result['converge_to_origin'] / total if total > 0 else 0
        diverge_pct = 100 * result['diverge_to_infinity'] / total if total > 0 else 0
        
        print(f"  Saddle points: {len(result['analysis']['saddle_points'])}")
        print(f"  Healing: {result['converge_to_origin']}/{total} ({healing_pct:.1f}%)")
        print(f"  Diverging: {result['diverge_to_infinity']}/{total} ({diverge_pct:.1f}%)")
        
        # Show plots
        fig = plot_trajectory_fate_analysis(result, f"- {case['name']}")
        plt.show()
    
    # Summary analysis
    print("\n" + "="*50)
    print("SUMMARY: HEALING AREA IMPLICATIONS")
    print("="*50)
    
    for case, result in results_all:
        total = len(result['trajectories'])
        healing_pct = 100 * result['converge_to_origin'] / total if total > 0 else 0
        saddles = len(result['analysis']['saddle_points'])
        
        if saddles == 0 and healing_pct > 50:
            print(f"✓ {case['name']}: NO saddles, but {healing_pct:.1f}% → healing")
            print(f"  → Healing area could be considered LARGER (no separatrix boundary)")
        elif saddles == 0 and healing_pct < 50:
            print(f"✗ {case['name']}: NO saddles, only {healing_pct:.1f}% → healing")
            print(f"  → Healing area is REDUCED or complex")
        else:
            print(f"• {case['name']}: HAS saddles, {healing_pct:.1f}% → healing")
            print(f"  → Normal separatrix-bounded healing area")

if __name__ == "__main__":
    main_trajectory_analysis()
