import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
from numpy.linalg import eigvals
import pandas as pd
from functools import lru_cache

# Default parameters
params = {
    'lambda1': 0.9,
    'lambda2': 0.8,
    'mu1': 0.3,
    'mu2': 0.3,
    'K': 1e6,
    'k1': 1e9,
    'k2': 1e9,
    'beta1': 470*60*24,
    'beta2': 70*60*24,
    'beta3': 240*60*24,
    'alpha1': 940*60*24,
    'alpha2': 510*60*24,
    'gamma': 2 }

class CellDynamicsAnalyzer:
    """
    A class to analyze cell dynamics with fixed points, separatrices, and phase portraits.
    """
    
    def __init__(self, params=None):
        self.params = params if params is not None else globals()['params']
        self._cached_vector_field = None
        self._cached_grid = None

    
    @staticmethod
    def full_ode(y, p=None):
        """Full ODE system including CSF and PDGF dynamics."""
        if p is None:
            p = globals()['params']
        mF, M, CSF, PDGF = y
        
        dmF = mF*( (p['lambda1']*PDGF/(p['k1'] + PDGF))*(1 - mF/p['K']) - p['mu1'] )
        dM = M*( p['lambda2']*CSF / (p['k2'] + CSF) - p['mu2'] )
        dcsf = p['beta1']*mF - p['alpha1']*M*(CSF/(p['k2']+CSF)) - p['gamma']*CSF
        dpdgf = p['beta2']*M + p['beta3']*mF - p['alpha2']*mF*(PDGF/(p['k1']+PDGF)) - p['gamma']*PDGF

        return [dmF, dM, dcsf, dpdgf]

    @staticmethod
    def _solve_quadratic_steady_state(a, b, c):
        """Solve quadratic equation and return positive real roots."""
        roots = np.roots([a, b, c])
        return [r.real for r in roots if np.isreal(r) and r.real >= 0]

    @staticmethod
    def csf_steady(mF, M, p=None):
        """Compute steady-state CSF concentration."""
        if p is None:
            p = globals()['params']
        # -gamma*CSF^2 + (beta1*mF-alpha1*M-gamma*k2)*CSF + beta1*mF*k2 = 0
        a = -p['gamma']
        b = p['beta1']*mF - p['alpha1']*M - p['gamma']*p['k2']
        c = p['beta1']*mF*p['k2']
        return CellDynamicsAnalyzer._solve_quadratic_steady_state(a, b, c)

    @staticmethod
    def pdgf_steady(mF, M, p=None):
        """Compute steady-state PDGF concentration."""
        if p is None:
            p = globals()['params']
        # -gamma*PDGF^2 + (beta2*M+beta3*mF-alpha2*mF-gamma*k1)*PDGF + k1*(beta2*M+beta3*mF) = 0
        a = -p['gamma']
        b = p['beta2']*M + p['beta3']*mF - p['alpha2']*mF - p['gamma']*p['k1']
        c = p['k1']*(p['beta2']*M + p['beta3']*mF)
        return CellDynamicsAnalyzer._solve_quadratic_steady_state(a, b, c)

    @staticmethod
    def mf_m_ode_system(mF, M, p=None):
        """Reduced ODE system using steady-state approximations for CSF and PDGF."""
        if p is None:
            p = globals()['params']
        csf = CellDynamicsAnalyzer.csf_steady(mF, M, p)
        pdgf = CellDynamicsAnalyzer.pdgf_steady(mF, M, p)
        if csf and pdgf:
            dmF = mF * ((p['lambda1'] * pdgf[0] / (p['k1'] + pdgf[0])) * (1 - mF / p['K']) - p['mu1'])
            dM = M * (p['lambda2'] * csf[0] / (p['k2'] + csf[0]) - p['mu2'])
            return [dmF, dM]
        else:
            return [0, 0]
    
    def compute_vector_field(self, x_log_range=(0, 8), y_log_range=(0, 8), grid_size=100):
        """Compute and cache the vector field for streamplot visualization."""
        cache_key = (x_log_range, y_log_range, grid_size)
        if self._cached_vector_field is not None and self._cached_grid == cache_key:
            return self._cached_vector_field
            
        x_log = np.linspace(x_log_range[0], x_log_range[1], grid_size)
        y_log = np.linspace(y_log_range[0], y_log_range[1], grid_size)
        X_log, Y_log = np.meshgrid(x_log, y_log)
        
        # Convert to linear scale
        X = 10 ** X_log
        Y = 10 ** Y_log
        
        # Vectorized computation using broadcasting
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Flatten for vectorized computation
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        # Compute vector field values
        for i, (mF, M) in enumerate(zip(X_flat, Y_flat)):
            u, v = self.mf_m_ode_system(mF, M, self.params)
            U.flat[i] = u
            V.flat[i] = v
        
        # Normalize
        N = np.sqrt(U**2 + V**2)
        U = np.divide(U, N + 1e-7, out=np.zeros_like(U), where=(N + 1e-7) != 0)
        V = np.divide(V, N + 1e-7, out=np.zeros_like(V), where=(N + 1e-7) != 0)
        
        result = (X_log, Y_log, U, V)
        self._cached_vector_field = result
        self._cached_grid = cache_key
        return result

    def plot_streamplot(self, figsize=(8, 6)):
        """Plot streamplot of the mF-M dynamic system."""
        X_log, Y_log, U, V = self.compute_vector_field()
        
        plt.figure(figsize=figsize)
        plt.streamplot(X_log, Y_log, U, V, color='gray', linewidth=1, density=1.2)
        plt.xlabel('log10(mF)')
        plt.ylabel('log10(M)')
        plt.title('Streamplot of mF-M Dynamic System (in log space)')
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()

    
    def nullclines_mF(self, mF):
        """Compute M value on mF-nullcline (dmF/dt = 0)."""
        p = self.params
        pdgf = (p['mu1']*p['k1']*p['K'])/(p['lambda1']*p['K'] - p['mu1']*p['K'] - p['lambda1']*mF)
        if pdgf <= 0:
            return None
        M = (1/p['beta2'])*(p['alpha2']*mF*pdgf/(p['k1']+pdgf) + p['gamma']*pdgf - p['beta3']*mF)
        return M if M >= 0 else None

    def nullclines_M(self, mF):
        """Compute M value on M-nullcline (dM/dt = 0)."""
        p = self.params
        csf = p['k2']*p['mu2'] / (p['lambda2']-p['mu2'])
        if csf <= 0:
            return None
        M = ((p['k2']+csf) / (p['alpha1']*csf)) * (p['beta1']*mF-p['gamma']*csf)
        return M if M >=0 else None

    def compute_fixed_points(self):
        """Find fixed points by computing intersections of nullclines."""
        def nullclines_diff(mF):
            m1 = self.nullclines_mF(mF)
            m2 = self.nullclines_M(mF)
            if m1 is not None and m2 is not None:
                return m2 - m1
            else:
                return np.nan

        mF_space = np.logspace(0, 7, 50)
        diff_vals = np.array([nullclines_diff(mF) for mF in mF_space])

        # Find sign changes
        guess_list = []
        for i in range(len(diff_vals) - 1):
            if not (np.isnan(diff_vals[i]) or np.isnan(diff_vals[i + 1])):
                if diff_vals[i] * diff_vals[i + 1] < 0:
                    guess_list.append(mF_space[i])

        # Compute fixed points
        fixed_points = []
        for guess_point in guess_list:
            try:
                mF_sol = fsolve(nullclines_diff, guess_point)[0]
                M_sol = self.nullclines_mF(mF_sol)
                if mF_sol > 0 and M_sol is not None and M_sol > 0:
                    fixed_points.append((mF_sol, M_sol))
            except:
                continue

        return fixed_points

    def cold_fibrosis_point(self):
        """Find cold fibrosis fixed points (M=0 case)."""
        p = self.params
        # Solve cubic equation for PDGF when M=0
        a = -p['gamma']
        b = ((p['K']/p['lambda1']) * (p['lambda1'] - p['mu1']) * (p['beta3'] - p['alpha2'])) - p['gamma']*p['k1']
        c = (p['K'] * p['k1'] / p['lambda1']) * (p['beta3']*p['lambda1'] - 2*p['mu1']*p['beta3'] + p['mu1']*p['alpha2'])
        d = -p['k1']**2 * p['mu1'] * p['K'] * p['beta3'] / p['lambda1']

        pdgf_roots = np.roots([a, b, c, d])
        
        cf_fixed_points = []
        for pdgf in pdgf_roots:
            if np.isreal(pdgf) and pdgf >= 0:
                pdgf = pdgf.real
                mF = p['K'] * ((p['lambda1'] - p['mu1']) / p['lambda1'] - (p['mu1'] * p['k1']) / (p['lambda1'] * pdgf))
                if mF >= 0:
                    cf_fixed_points.append((mF, 1)) # cut off at cell count = 1

        return cf_fixed_points

    @staticmethod
    def combine_fixed_points(fp_list1, fp_list2, tol=1e-3):
        """Combine fixed point lists, avoiding duplicates."""
        combined = fp_list1.copy()
        for p2 in fp_list2:
            if all(np.linalg.norm(np.array(p2) - np.array(p1)) > tol for p1 in combined):
                combined.append(p2)
        return combined

    def plot_nullclines_and_fixed_points(self, figsize=(8, 6)):
        """Plot nullclines and fixed points."""
        fixed_points = self.compute_fixed_points()
        
        mF_vals = np.logspace(0, 8, 500)
        mF_null = [self.nullclines_mF(mf) for mf in mF_vals]
        M_null = [self.nullclines_M(mf) for mf in mF_vals]

        plt.figure(figsize=figsize)
        plt.plot(mF_vals, mF_null, label='mF-nullcline (dmF/dt=0)')
        plt.plot(mF_vals, M_null, label='M-nullcline (dM/dt=0)')
        if fixed_points:
            fp_arr = np.array(fixed_points)
            plt.scatter(fp_arr[:, 0], fp_arr[:, 1], color='red', label='Fixed Points')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('mF')
        plt.ylabel('M')
        plt.legend()
        plt.title('Nullclines and Fixed Points')
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()

    
    @staticmethod
    def jacobian(function, point, eps=1e-6):
        """Compute Jacobian matrix of a 2D vector field."""
        mF, M = point
        
        f0 = np.array(function([mF, M]))
        f1 = np.array(function([mF + eps, M]))
        df_dmF = (f1 - f0) / eps

        f2 = np.array(function([mF, M + eps]))
        df_dM = (f2 - f0) / eps
        J = np.column_stack([df_dmF, df_dM])
        return J

    def find_saddle_points(self, fixed_points):
        """Identify saddle points from a list of fixed points."""
        saddle_points = []
        for fp in fixed_points:
            J = self.jacobian(lambda state: self.mf_m_ode_system(state[0], state[1], self.params), fp)
            eigvals = np.linalg.eigvals(J)
            if np.any(eigvals.real > 0) and np.any(eigvals.real < 0):
                saddle_points.append(fp)
        return saddle_points

    def mf_m_ode_for_odeint(self, state, t=0):
        """ODE wrapper for use with odeint (time parameter required)."""
        mF, M = state
        return self.mf_m_ode_system(mF, M, self.params)

    def reversed_ode(self, state, t):
        """Reversed-time ODE for separatrix computation."""
        dmF, dM = self.mf_m_ode_for_odeint(state, t)
        return [-dmF, -dM]

    def compute_separatrices(self, saddle_points, t_span=None, eps=1e-3):
        """Compute separatrix trajectories from saddle points."""
        if t_span is None:
            t_span = np.linspace(0, 200, 2000)
            
        separatrices = []
        for sp in saddle_points:
            # Use stricter tolerances for deterministic results
            left = odeint(self.reversed_ode, [sp[0] - eps, sp[1] + eps], t_span, 
                         rtol=1e-10, atol=1e-12)
            right = odeint(self.reversed_ode, [sp[0] + eps, sp[1] - eps], t_span,
                          rtol=1e-10, atol=1e-12)
            separatrix = np.vstack([left[::-1], right])
            # Filter out negative values and cut off at cell count = 1
            # separatrix = separatrix[np.all(separatrix > 0, axis=1)]
            separatrix = separatrix[np.all(separatrix >= 1.0, axis=1)]
            separatrices.append(separatrix)
        
        return separatrices

    def analyze_system(self):
        """Complete analysis of the dynamical system."""
        # Find all fixed points
        fixed_points = self.compute_fixed_points()
        cf_points = self.cold_fibrosis_point()
        all_fixed_points = self.combine_fixed_points(fixed_points, cf_points)
        
        # Find saddle points
        saddle_points = self.find_saddle_points(all_fixed_points)
        
        # Compute separatrices
        separatrices = self.compute_separatrices(saddle_points)
        
        return {
            'fixed_points': fixed_points,
            'cold_fibrosis_points': cf_points,
            'all_fixed_points': all_fixed_points,
            'saddle_points': saddle_points,
            'separatrices': separatrices
        }

    def save_separatrix_data(self, separatrices, filename="separatrix.csv"):
        """Save separatrix data to CSV file."""
        if separatrices:
            data = pd.DataFrame(np.vstack(separatrices), columns=['mF', 'M'])
            data.to_csv(filename, index=False)
            return filename
        return None

    def plot_complete_analysis(self, figsize=(8, 6)):
        """Plot complete phase portrait with streamlines and separatrices."""
        # Get vector field
        X_log, Y_log, U, V = self.compute_vector_field()
        
        # Analyze system
        analysis = self.analyze_system()
        fixed_points = analysis['fixed_points']
        cf_points = analysis['cold_fibrosis_points']
        separatrices = analysis['separatrices']
        
        # Save separatrix data
        self.save_separatrix_data(separatrices)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.streamplot(X_log, Y_log, U, V, color='gray', linewidth=1, density=1.2)

        # Plot separatrices
        for sep in separatrices:
            if len(sep) > 0:
                plt.plot(np.log10(sep[:, 0]), np.log10(sep[:, 1]), 'r--')

        # Plot fixed points
        if len(fixed_points) >= 2:
            plt.plot(np.log10(fixed_points[0][0]), np.log10(fixed_points[0][1]), 'go', label='Unstable Fixed Point')
            plt.plot(np.log10(fixed_points[1][0]), np.log10(fixed_points[1][1]), 'ko', label='Hot Fibrosis')

        # Plot cold fibrosis points
        for i, fp in enumerate(cf_points):
            plt.plot(np.log10(fp[0]), np.log10(fp[1] + 1e-3), 'bo', label='Cold Fibrosis' if i == 0 else "")

        plt.xlim(0, 8)
        plt.ylim(0, 8)
        plt.xlabel('log10(mF)')
        plt.ylabel('log10(M)')
        plt.title('Streamplot with Separatrix')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

    def generate_separatrix_plot(self, original_separatrix=None, figsize=(8, 6)):
        """Generate separatrix plot with optional comparison to original."""
        analysis = self.analyze_system()
        fixed_points = analysis['fixed_points']
        separatrices = analysis['separatrices']
        
        plt.figure(figsize=figsize)
        
        # Plot new separatrices
        for sep in separatrices:
            if len(sep) > 0:
                plt.plot(np.log10(sep[:, 0]), np.log10(sep[:, 1]), 'r--')
        
        # Plot fixed points
        for fp in fixed_points:
            plt.plot(np.log10(fp[0]), np.log10(fp[1]), 'ko')
        
        # Plot original separatrix if provided
        if original_separatrix is not None:
            plt.plot(np.log10(original_separatrix[:, 0]), np.log10(original_separatrix[:, 1]),
                     'b--', label='Original Separatrix')
        
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        plt.xlabel('log10(mF)')
        plt.ylabel('log10(M)')
        plt.title('Separatrix at alpha2={:.1f}, beta3={:.1f}'.format(
            self.params["alpha2"], self.params["beta3"]))
        plt.legend()
        plt.tight_layout()
        
        return separatrices


# ==============================================
# Convenience functions for backward compatibility
# ==============================================

def csf_steady(mF, M, p=None):
    """Backward compatibility wrapper."""
    if p is None:
        p = params
    return CellDynamicsAnalyzer.csf_steady(mF, M, p)

def pdgf_steady(mF, M, p=None):
    """Backward compatibility wrapper."""
    if p is None:
        p = params
    return CellDynamicsAnalyzer.pdgf_steady(mF, M, p)

def mf_m_ode_system(mF, M, p=None):
    """Backward compatibility wrapper."""
    if p is None:
        p = params
    return CellDynamicsAnalyzer.mf_m_ode_system(mF, M, p)

def compute_fixed_points(p=None):
    """Backward compatibility wrapper."""
    if p is None:
        p = params
    analyzer = CellDynamicsAnalyzer(p)
    return analyzer.compute_fixed_points()

def cold_fibrosis_point(p=None):
    """Backward compatibility wrapper."""
    if p is None:
        p = params
    analyzer = CellDynamicsAnalyzer(p)
    return analyzer.cold_fibrosis_point()

def combine_fixed_points(fp_list1, fp_list2, tol=1e-3):
    """Backward compatibility wrapper."""
    return CellDynamicsAnalyzer.combine_fixed_points(fp_list1, fp_list2, tol)

def mf_m_ode_wrapper(state, p=None):
    """Backward compatibility wrapper."""
    if p is None:
        p = params
    mF, M = state
    return CellDynamicsAnalyzer.mf_m_ode_system(mF, M, p)

def jacobian(function, point, eps=1e-6):
    """Backward compatibility wrapper."""
    return CellDynamicsAnalyzer.jacobian(function, point, eps)

def reversed_ode(state, t, p):
    """Backward compatibility wrapper."""
    analyzer = CellDynamicsAnalyzer(p)
    return analyzer.reversed_ode(state, t)

def generate_separatrix_plot(params_new, original_separatrix, eps=1e-3):
    """Backward compatibility wrapper."""
    analyzer = CellDynamicsAnalyzer(params_new)
    return analyzer.generate_separatrix_plot(original_separatrix)


# ==============================================
# Initialize default analyzer and run analysis
# ==============================================

# Create default analyzer instance
default_analyzer = CellDynamicsAnalyzer(params)

# Show initial plots
if __name__ == "__main__":
    print("Generating streamplot...")
    default_analyzer.plot_streamplot()
    plt.show()
    
    print("Generating nullclines plot...")
    default_analyzer.plot_nullclines_and_fixed_points()
    plt.show()
    
    print("Running complete analysis...")
    analysis_results = default_analyzer.analyze_system()
    
    print("Cold fibrosis fixed points:", analysis_results['cold_fibrosis_points'])
    print("Total fixed points found: {}".format(len(analysis_results['all_fixed_points'])))
    print("Number of regular fixed points: {}".format(len(analysis_results['fixed_points'])))
    
    # Print eigenvalues for verification
    if analysis_results['fixed_points']:
        fixed_points_sorted = sorted(analysis_results['fixed_points'], key=lambda x: x[0])
        if len(fixed_points_sorted) >= 1:
            unstable_fixed_point = fixed_points_sorted[0]
            J = default_analyzer.jacobian(lambda state: default_analyzer.mf_m_ode_system(state[0], state[1], params), unstable_fixed_point)
            eigvals = np.linalg.eigvals(J)
            print("Eigenvalues of first fixed point:", eigvals)
    
    # Analyze cold fibrosis points
    for fp in analysis_results['cold_fibrosis_points']:
        J_cf = default_analyzer.jacobian(lambda state: default_analyzer.mf_m_ode_system(state[0], state[1], params), fp)
        eigvals_cf = np.linalg.eigvals(J_cf)
        print("Cold fibrosis eigenvalues:", eigvals_cf)
    
    print("Generating complete analysis plot...")
    default_analyzer.plot_complete_analysis()
    plt.show()