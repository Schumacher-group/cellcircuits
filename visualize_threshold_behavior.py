"""
Visualize how phase space topology changes as parameters approach the threshold
where saddle points disappear.
"""

from my_function import *
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_portrait_comparison(beta3_fold, alpha2_values, params, title_suffix=""):
    """Plot phase portraits for different alpha2 values at fixed beta3."""
    
    fig, axes = plt.subplots(1, len(alpha2_values), figsize=(5*len(alpha2_values), 5))
    if len(alpha2_values) == 1:
        axes = [axes]
    
    # Create phase space grid
    mF_range = np.logspace(-1, 3, 20)
    M_range = np.logspace(-1, 3, 20)
    mF_grid, M_grid = np.meshgrid(mF_range, M_range)
    
    for i, alpha2_fold in enumerate(alpha2_values):
        ax = axes[i]
        
        # Set parameters
        params_new = params.copy()
        params_new['alpha2'] *= alpha2_fold
        params_new['beta3'] *= beta3_fold
        
        # Create analyzer
        analyzer = CellDynamicsAnalyzer(params_new)
        analysis = analyzer.analyze_system()
        
        # Compute vector field
        U = np.zeros_like(mF_grid)
        V = np.zeros_like(M_grid)
        
        for j in range(mF_grid.shape[0]):
            for k in range(mF_grid.shape[1]):
                mF, M = mF_grid[j, k], M_grid[j, k]
                dmF_dt, dM_dt = analyzer.mf_m_ode_system(mF, M)
                U[j, k] = dmF_dt
                V[j, k] = dM_dt
        
        # Plot vector field
        ax.quiver(mF_grid, M_grid, U, V, alpha=0.5, width=0.002)
        
        # Plot fixed points
        if analysis['fixed_points']:
            fps = np.array(analysis['fixed_points'])
            ax.scatter(fps[:, 0], fps[:, 1], c='red', s=100, marker='o', 
                      label='Fixed Points ({})'.format(len(fps)), zorder=5)
        
        # Plot saddle points specifically
        if analysis['saddle_points']:
            saddles = np.array(analysis['saddle_points'])
            ax.scatter(saddles[:, 0], saddles[:, 1], c='blue', s=150, marker='s', 
                      label='Saddles ({})'.format(len(saddles)), zorder=6)
        
        # Plot separatrices if they exist
        if analysis['separatrices'] and len(analysis['separatrices']) > 0:
            for sep in analysis['separatrices']:
                if len(sep) > 0:
                    ax.plot(sep[:, 0], sep[:, 1], 'k-', linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(0.1, 1000)
        ax.set_xlabel('mF')
        ax.set_ylabel('M')
        ax.set_title('α2 = {:.2f}×\n(FP:{}, Saddles:{})'.format(
            alpha2_fold, len(analysis['fixed_points']), len(analysis['saddle_points'])))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Phase Portraits: β3 = {:.1f}× {}'.format(beta3_fold, title_suffix), fontsize=14)
    plt.tight_layout()
    return fig

def plot_threshold_summary():
    """Plot summary of threshold behavior with enhanced data."""
    
    # Enhanced data from investigation - more data points for robust fitting
    beta3_values = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]
    # These are approximate values - will be refined by comprehensive analysis
    alpha2_thresholds = [0.85, 1.11, 1.45, 1.82, 2.15, 2.54, 2.88, 3.20]
    
    plt.figure(figsize=(10, 7))
    plt.plot(beta3_values, alpha2_thresholds, 'bo-', linewidth=2, markersize=8)
    
    # Add data points labels
    for i, (b3, a2) in enumerate(zip(beta3_values, alpha2_thresholds)):
        plt.annotate('({:.1f}, {:.2f})'.format(b3, a2), 
                    (b3, a2), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('β3 fold change')
    plt.ylabel('α2 threshold (fold change)')
    plt.title('Enhanced Saddle Point Disappearance Threshold\n(More data points for robust fitting)')
    plt.grid(True, alpha=0.3)
    
    # Fit linear relationship with enhanced data
    z = np.polyfit(beta3_values, alpha2_thresholds, 1)
    p = np.poly1d(z)
    
    # Calculate R-squared
    alpha2_pred = p(beta3_values)
    ss_res = np.sum((np.array(alpha2_thresholds) - alpha2_pred) ** 2)
    ss_tot = np.sum((np.array(alpha2_thresholds) - np.mean(alpha2_thresholds)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    x_fit = np.linspace(min(beta3_values), max(beta3_values), 100)
    plt.plot(x_fit, p(x_fit), 'r--', alpha=0.8, linewidth=2,
             label='Linear fit: α2_thresh = {:.3f}β3 + {:.3f}\nR² = {:.4f}'.format(z[0], z[1], r_squared))
    
    # Add confidence band
    residuals = np.array(alpha2_thresholds) - alpha2_pred
    std_error = np.std(residuals)
    plt.fill_between(x_fit, p(x_fit) - 2*std_error, p(x_fit) + 2*std_error, 
                    alpha=0.2, color='red', label='±2σ confidence band')
    
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def investigate_nullclines_at_threshold():
    """Investigate how nullclines change near the threshold."""
    
    # Parameters
    beta3_fold = 1.0
    alpha2_values = [1.5, 1.7, 1.82, 1.9, 2.0]  # Around the threshold at 1.82
    
    fig, axes = plt.subplots(1, len(alpha2_values), figsize=(4*len(alpha2_values), 4))
    if len(alpha2_values) == 1:
        axes = [axes]
    
    mF_range = np.logspace(-1, 3, 1000)
    
    for i, alpha2_fold in enumerate(alpha2_values):
        ax = axes[i]
        
        # Set parameters
        params_new = params.copy()
        params_new['alpha2'] *= alpha2_fold
        params_new['beta3'] *= beta3_fold
        
        # Create analyzer
        analyzer = CellDynamicsAnalyzer(params_new)
        analysis = analyzer.analyze_system()
        
        # Compute nullclines
        mF_nullcline = []
        M_nullcline = []
        
        for mF in mF_range:
            # mF nullcline: dmF/dt = 0
            # Solve for M when dmF/dt = 0
            try:
                M_nc = analyzer.nullclines_mF(mF)
                if M_nc is not None and M_nc > 0:
                    mF_nullcline.append(mF)
                    M_nullcline.append(M_nc)
            except:
                continue
        
        if mF_nullcline:
            ax.plot(mF_nullcline, M_nullcline, 'b-', linewidth=2, label='mF nullcline')
        
        # M nullcline
        M_range = np.logspace(-1, 3, 1000)
        mF_nullcline_M = []
        M_nullcline_M = []
        
        for mF in mF_range:
            # M nullcline: dM/dt = 0  
            try:
                M_nc = analyzer.nullclines_M(mF)
                if M_nc is not None and M_nc > 0:
                    mF_nullcline_M.append(mF)
                    M_nullcline_M.append(M_nc)
            except:
                continue
        
        if mF_nullcline_M:
            ax.plot(mF_nullcline_M, M_nullcline_M, 'r-', linewidth=2, label='M nullcline')
        
        # Plot fixed points
        if analysis['fixed_points']:
            fps = np.array(analysis['fixed_points'])
            ax.scatter(fps[:, 0], fps[:, 1], c='green', s=100, marker='o', 
                      label='Fixed Points', zorder=5)
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(0.1, 1000)
        ax.set_xlabel('mF')
        ax.set_ylabel('M')
        ax.set_title('α2 = {:.2f}×\n({} FPs)'.format(alpha2_fold, len(analysis['fixed_points'])))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Nullclines Near Threshold: β3 = {:.1f}×'.format(beta3_fold), fontsize=14)
    plt.tight_layout()
    return fig

# Run the analysis
if __name__ == "__main__":
    print("Generating threshold behavior visualizations...")
    
    # 1. Plot threshold summary
    print("1. Creating threshold summary plot...")
    fig1 = plot_threshold_summary()
    plt.show()
    
    # 2. Phase portraits around threshold for β3 = 1.0
    print("2. Creating phase portraits around threshold...")
    alpha2_values = [1.5, 1.7, 1.82, 1.9, 2.0]  # Around threshold at 1.82
    fig2 = plot_phase_portrait_comparison(1.0, alpha2_values, params, "(around threshold)")
    plt.show()
    
    # 3. Nullclines investigation
    print("3. Investigating nullclines near threshold...")
    fig3 = investigate_nullclines_at_threshold()
    plt.show()
    
    # 4. Compare different β3 values
    print("4. Comparing phase portraits for different β3 values...")
    
    # Below threshold
    fig4a = plot_phase_portrait_comparison(0.5, [1.0], params, "(β3=0.5, below threshold)")
    plt.show()
    
    # At threshold  
    fig4b = plot_phase_portrait_comparison(1.0, [1.8], params, "(β3=1.0, near threshold)")
    plt.show()
    
    # Above threshold
    fig4c = plot_phase_portrait_comparison(1.5, [2.5], params, "(β3=1.5, below threshold)")
    plt.show()
    
    print("Visualization complete!")
