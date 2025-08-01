from my_function import *
from signal_simulation import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path

# Set deterministic behavior
np.random.seed(42)

# Initialize analyzer
base_analyzer = CellDynamicsAnalyzer(params)

# Load the previously saved separatrix data
data = pd.read_csv('separatrix.csv')
separatrix = data[['mF', 'M']].values
separatrix = separatrix[np.argsort(separatrix[:, 0])]

# Function to determine if a point lies in the healing basin
def below_separatrix(mF, M, separatrix):
    if mF <= 0 or M <= 0:
        return True
    
    # Use safe log transformation
    min_val = 1
    safe_sep = np.maximum(separatrix, min_val)
    safe_mF = max(mF, min_val)
    safe_M = max(M, min_val)
    
    sep_log_mF = np.log10(safe_sep[:, 0])
    sep_log_M = np.log10(safe_sep[:, 1])
    
    log_mF = np.log10(safe_mF)

    # If the point outside the healing basin: fibrosis
    if log_mF > sep_log_mF[-1]:
        return False
    
    # Interpolate separatrix M at the current mF value
    log_M_interp = np.interp(log_mF, sep_log_mF, sep_log_M)
    return np.log10(safe_M) < log_M_interp

class TreatmentAnalyzer:
    """Class for treatment simulation and parameter sensitivity analysis."""
    
    def __init__(self, base_params, base_separatrix):
        self.base_params = base_params
        self.base_separatrix = base_separatrix
        
    def single_parameter_sensitivity(self, param_names, folds, t_span=(0, 80, 500), 
                                   initial_condition=[1, 1], pulse_type='prolonged'):
        """Perform single parameter sensitivity analysis."""
        t = np.linspace(t_span[0], t_span[1], t_span[2])
        y0 = initial_condition
        
        results = {}
        
        for param in param_names:
            trajectories = []
            end_points = []
            status = []
            
            for fold in folds:
                new_params = self.base_params.copy()
                new_params[param] *= fold
                
                # Simulate the system
                sol = odeint(modified_mf_m_ode, y0, t, args=(new_params, pulse_type))
                trajectories.append(sol)
                end_points.append(sol[-1])
                status.append(below_separatrix(sol[-1, 0], sol[-1, 1], self.base_separatrix))
            
            results[param] = {
                'trajectories': trajectories,
                'end_points': end_points,
                'status': status,
                'folds': folds
            }
        
        return results
    
    def safe_plot_log10(self, arr, min_val=1):
        """Safely compute log10 for plotting, handling zeros."""
        safe_arr = np.maximum(arr, min_val)
        return np.log10(safe_arr)

    def plot_results(self, results):
        """Plot treatment analysis results."""
        num_params = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, param in enumerate(results.keys()):
            if idx >= len(axes):
                break
                
            data = results[param]
            trajectories = data['trajectories']
            status = data['status']
            folds = data['folds']
            
            ax = axes[idx]
            for i, sol in enumerate(trajectories):
                label = "{:.2f}×{}".format(folds[i], param)
                color = 'green' if status[i] else 'red'
                ax.plot(self.safe_plot_log10(sol[:, 0]), self.safe_plot_log10(sol[:, 1]), label=label, color=color)
            
            # Add separatrix
            ax.plot(self.safe_plot_log10(self.base_separatrix[:, 0]), self.safe_plot_log10(self.base_separatrix[:, 1]), 
                   'k--', label='Separatrix')
            ax.set_title('Trajectories with {}'.format(param))
            ax.set_xlabel('log10(mF)')
            ax.set_ylabel('log10(M)')
            ax.set_xlim(0, 7)
            ax.set_ylim(0, 7)
    
    def two_parameter_sweep(self, param_pairs, folds, t_span=(0, 80, 500), 
                           initial_condition=[1, 1], pulse_type='prolonged'):
        """Perform 2D parameter sweep analysis."""
        t = np.linspace(t_span[0], t_span[1], t_span[2])
        y0 = initial_condition
        
        fate_maps = []
        
        for param_x, param_y in param_pairs:
            fate_map = np.zeros((len(folds), len(folds)), dtype=bool)
            
            for i, x in enumerate(folds):
                for j, y in enumerate(folds):
                    new_p = self.base_params.copy()
                    new_p[param_x] *= x
                    new_p[param_y] *= y
                    
                    sol = odeint(modified_mf_m_ode, y0, t, args=(new_p, pulse_type))
                    mF_final, M_final = sol[-1]
                    fate_map[i, j] = below_separatrix(mF_final, M_final, self.base_separatrix)
            
            fate_maps.append(fate_map)
        
        return fate_maps
    
    def plot_two_parameter_sweep(self, param_pairs, fate_maps, folds, figsize=(18, 10)):
        """Plot 2D parameter sweep results."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        X, Y = np.meshgrid(folds, folds)
        cmap = plt.get_cmap('Greens')
        
        for idx, ((param_x, param_y), fate_map) in enumerate(zip(param_pairs, fate_maps)):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            im = ax.pcolormesh(X, Y, fate_map, cmap=cmap, shading='auto', vmin=0, vmax=1)
            ax.set_xlabel('{} fold'.format(param_x))
            ax.set_ylabel('{} fold'.format(param_y))
            ax.set_title('{} vs {}'.format(param_x, param_y))
        
        plt.colorbar(im, label='Healing (1=True, 0=False)')
        plt.tight_layout(pad=2.5)
        return fig

# Initialize treatment analyzer
treatment_analyzer = TreatmentAnalyzer(params, separatrix)

# =======================================
# Single parameter sensitivity analysis
# =======================================
folds = [0.5, 1, 2]  # Define fold change to scan
param_names = ['alpha1', 'alpha2', 'beta1', 'beta3', 'lambda1', 'mu1']

print("Running single parameter sensitivity analysis...")
sensitivity_results = treatment_analyzer.single_parameter_sensitivity(param_names, folds)
treatment_analyzer.plot_results(sensitivity_results)
plt.show()

# =====================================================
# 2D parameter sweep of alpha2, beta3, lambda1 and mu1
# =====================================================
# Define all possible parameter pairs to scan
param_pairs = [
    ('alpha2', 'beta3'),
    ('alpha2', 'lambda1'),
    ('alpha2', 'mu1'),
    ('beta3', 'lambda1'),
    ('beta3', 'mu1'),
    ('lambda1', 'mu1')
]

# # Define fold change to scan
# sweep_folds = np.linspace(0.5, 2.0, 20)

# print("Running 2D parameter sweep analysis...")
# fate_maps = treatment_analyzer.two_parameter_sweep(param_pairs, sweep_folds)
# treatment_analyzer.plot_two_parameter_sweep(param_pairs, fate_maps, sweep_folds)
# plt.show()


# =======================================
# Compute area under the separatrix
# =======================================
class HealingAreaAnalyzer:
    """Class for computing and analyzing healing area changes."""
    
    @staticmethod
    def validate_separatrix(separatrix):
        """Validate separatrix geometry for area computation."""
        if separatrix is None or len(separatrix) < 3:
            return False, "Too few points"
        
        if not np.all(np.isfinite(separatrix)):
            return False, "Contains non-finite values"
        
        # Check for negative values (shouldn't happen in biological context)
        if np.any(separatrix < 0):
            return False, "Contains negative values"
        
        # Check for self-intersections (basic check)
        if len(separatrix) > 1000:  # Subsample for efficiency
            indices = np.linspace(0, len(separatrix)-1, 100, dtype=int)
            check_sep = separatrix[indices]
        else:
            check_sep = separatrix
            
        # Check if points form a reasonable boundary
        x_range = np.ptp(check_sep[:, 0])  # peak-to-peak range
        y_range = np.ptp(check_sep[:, 1])
        
        if x_range < 1e-10 or y_range < 1e-10:
            return False, "Degenerate boundary"
            
        return True, "Valid"

    @staticmethod
    def compute_healing_area(separatrix, grid_x, grid_y):
        """Compute the area under the separatrix boundary."""
        # Validate separatrix first
        is_valid, reason = HealingAreaAnalyzer.validate_separatrix(separatrix)
        if not is_valid:
            print("Invalid separatrix: {}".format(reason))
            return 0.0
            
        try:
            # Create polygon path from the separatrix boundary
            boundary_path = Path(separatrix)
            # Flatten the grid into 2D 
            points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            # Check if the grid lie inside the separatrix region
            inside = boundary_path.contains_points(points)
            inside_mask = inside.reshape(grid_x.shape)

            # Compute the area of each grid
            dx = np.abs(grid_x[0,1] - grid_x[0,0])
            dy = np.abs(grid_y[1,0] - grid_y[0,0])
            # Compute unit area
            unit_area = dx * dy
            # Total area: number of points inside * unit area
            healing_area = np.sum(inside_mask) * unit_area

            return healing_area
        except Exception as e:
            print("Area computation failed: {}".format(str(e)))
            return 0.0
    
    def safe_log10_transform(self, separatrix, min_val=1):
        """Safely transform separatrix to log10 scale, handling zeros."""
        # Replace any zeros or very small values with min_val
        safe_sep = np.maximum(separatrix, min_val)
        return np.log10(safe_sep)

    def analyze_parameter_effect_on_area(self, param_x, param_y, folds, base_separatrix):
        """Analyze how parameter changes affect healing basin area."""
        x_vals = np.linspace(-3, 8, 200)
        y_vals = np.linspace(-3, 8, 200)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        # Use safe log transformation
        sep_log = self.safe_log10_transform(base_separatrix)

        # Compute the original healing area
        original_area = self.compute_healing_area(sep_log, grid_x, grid_y)
        print("Original healing area: {}".format(original_area))

        # Calculate maximum possible area (entire grid)
        dx = np.abs(grid_x[0,1] - grid_x[0,0])
        dy = np.abs(grid_y[1,0] - grid_y[0,0])
        max_area = grid_x.shape[0] * grid_x.shape[1] * dx * dy
        print("Maximum possible area: {}".format(max_area))

        # Threshold relationship: α2_thresh = 1.338*β3 + 0.424
        def is_above_saddle_threshold(alpha2_fold, beta3_fold):
            threshold = 1.338 * beta3_fold + 0.424
            return alpha2_fold > threshold

        # Initialize area matrix
        area_matrix = np.full((len(folds), len(folds)), np.nan)
        t_span = np.linspace(0, 200, 2000)
        eps = 1e-3
        debug_log = []

        for i, fold1 in enumerate(folds):
            for j, fold2 in enumerate(folds):
                params_new = params.copy()
                params_new[param_x] *= fold1
                params_new[param_y] *= fold2
                
                # Check if we're above the saddle point threshold
                if param_x == 'alpha2' and param_y == 'beta3':
                    above_threshold = is_above_saddle_threshold(fold1, fold2)
                elif param_x == 'beta3' and param_y == 'alpha2':
                    above_threshold = is_above_saddle_threshold(fold2, fold1)
                else:
                    # For other parameter combinations, use the current method
                    above_threshold = False
                
                # Debug logging
                debug_info = {
                    'fold1': fold1, 'fold2': fold2,
                    'above_threshold': above_threshold,
                    'fixed_points': None, 'cf_points': None, 'saddles': None,
                    'separatrices_count': 0, 'area': None, 'status': 'unknown'
                }
                
                try:
                    # Create analyzer with new parameters
                    analyzer = CellDynamicsAnalyzer(params_new)
                    analysis = analyzer.analyze_system()
                    
                    # Log intermediate results
                    debug_info['fixed_points'] = len(analysis['fixed_points'])
                    debug_info['cf_points'] = len(analysis['cold_fibrosis_points'])
                    debug_info['saddles'] = len(analysis['saddle_points'])
                    debug_info['separatrices_count'] = len(analysis['separatrices'])
                    
                    # If above threshold (no saddle points), healing area = max area
                    if above_threshold or len(analysis['saddle_points']) == 0:
                        area_matrix[i, j] = max_area
                        debug_info['area'] = max_area
                        debug_info['status'] = 'max_area_no_saddles'
                        print("Max area (no saddles) for {}={:.2f}, {}={:.2f}".format(param_x, fold1, param_y, fold2))
                    
                    elif analysis['separatrices'] and len(analysis['separatrices']) > 0:
                        try:
                            sep_all = np.vstack(analysis['separatrices'])
                            debug_info['sep_shape'] = sep_all.shape
                            debug_info['sep_finite'] = np.all(np.isfinite(sep_all))
                            
                            if sep_all.shape[0] > 0 and np.all(np.isfinite(sep_all)):
                                sep_log = self.safe_log10_transform(sep_all)
                                area = self.compute_healing_area(sep_log, grid_x, grid_y)
                                debug_info['area'] = area
                                
                                if np.isfinite(area) and area > 0:
                                    area_matrix[i, j] = area
                                    debug_info['status'] = 'success'
                                elif area == 0:
                                    area_matrix[i, j] = area
                                    debug_info['status'] = 'zero_area'
                                    print("Zero area for {}={:.2f}, {}={:.2f}".format(param_x, fold1, param_y, fold2))
                                elif np.isnan(area):
                                    debug_info['status'] = 'nan_area'
                                    print("Nan area for {}={:.2f}, {}={:.2f}".format(param_x, fold1, param_y, fold2))
                                elif not np.isfinite(area):
                                    debug_info['status'] = 'infinite_area'
                                    print("Infinite area for {}={:.2f}, {}={:.2f}".format(param_x, fold1, param_y, fold2))
                            elif not np.all(np.isfinite(sep_all)):
                                debug_info['status'] = 'infinite_separatrices'
                                print("Infinite spx for {}={:.2f}, {}={:.2f}?".format(param_x, fold1, param_y, fold2))
                            elif sep_all.shape[0] == 0:
                                debug_info['status'] = 'empty_separatrices'
                                print("Empty spx for {}={:.2f}, {}={:.2f}?".format(param_x, fold1, param_y, fold2))
                        except (ValueError, RuntimeError) as e:
                            debug_info['status'] = 'vstack_error'
                            debug_info['error'] = str(e)
                            continue
                    else:
                        debug_info['status'] = 'no_separatrices'
                        print("No separatrices found for {}={:.2f}, {}={:.2f}".format(param_x, fold1, param_y, fold2))
                        
                except Exception as e:
                    debug_info['status'] = 'general_error'
                    debug_info['error'] = str(e)
                    print("Error for {}={:.2f}, {}={:.2f}: {}".format(param_x, fold1, param_y, fold2, str(e)))
                    continue
                
                debug_log.append(debug_info)

        # Print debug summary
        print("\n=== DEBUG SUMMARY ===")
        statuses = [info.get('status', 'unknown') for info in debug_log if 'status' in info]
        from collections import Counter
        status_counts = Counter(statuses)
        
        for status, count in status_counts.items():
            print("{}: {} cases".format(status, count))

        ratio_matrix = area_matrix / original_area
        return area_matrix, ratio_matrix, original_area

# Initialize healing area analyzer
area_analyzer = HealingAreaAnalyzer()

# Analyze alpha2 vs beta3 effect on healing area
print("Computing healing area analysis for alpha2 vs beta3...")
folds_area = np.linspace(0.5, 2.0, 10)
area_matrix, ratio_matrix, original_area = area_analyzer.analyze_parameter_effect_on_area(
    'alpha2', 'beta3', folds_area, separatrix)

plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(folds_area, folds_area)

# Create diverging colormap centered at 1.0 with potentially asymmetric limits
min_val = np.nanmin(ratio_matrix)
max_val = np.nanmax(ratio_matrix)

# Use the actual data range, but ensure the center (1.0) maps to neutral color
center = 1.0
vmin = min_val
vmax = max_val

# Create a custom normalization that centers the colormap at 1.0
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

figure = plt.pcolormesh(X, Y, ratio_matrix, cmap='RdBu_r', norm=norm)
plt.colorbar(figure, label='Healing Area Ratio (new / original)')

# Add contour line at ratio = 1.0 for reference
contour = plt.contour(X, Y, ratio_matrix, levels=[1.0], colors='black', linewidths=2, linestyles='--')
plt.clabel(contour, inline=True, fontsize=10, fmt='%.1f')

plt.xlabel('beta3 fold')
plt.ylabel('alpha2 fold')
plt.title('Healing Basin Area Ratio Heatmap')
plt.tight_layout()
plt.show()

# # Test case
# print("Running test case...")
# original_sep = pd.read_csv('separatrix.csv')[['mF', 'M']].values
# params_test = params.copy()
# params_test['alpha2'] *= 0.5
# params_test['beta3'] *= 1.0

# test_analyzer = CellDynamicsAnalyzer(params_test)
# test_separatrices = test_analyzer.generate_separatrix_plot(original_sep)
# test_analysis = test_analyzer.analyze_system()
# print("Number of fixed points in test case: {}".format(len(test_analysis['fixed_points'])))
# plt.show()