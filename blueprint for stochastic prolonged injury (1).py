import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Define constants
lambda1 = 0.9  # proliferation rate of mF
lambda2 = 0.8  # proliferation rate of M
mu1 = 0.3  # removal rate of mF
mu2 = 0.3  # removal rate of M
K = 10**6  # carrying capacity of mF
k1 = 10**9  # binding affinity of CSF
k2 = 10**9  # binding affinity of PDGF
beta1 = 470 * 60 * 24  # max secretion rate of CSF by mF
beta2 = 70 * 60 * 24  # max secretion rate of PDGF by M
beta3 = 240 * 60 * 24  # max secretion rate of PDGF by mF
alpha1 = 940 * 60 * 24  # max endocytosis rate of CSF by M
alpha2 = 510 * 60 * 24  # max endocytosis rate of PDGF by mF
gamma = 2  # degradation rate of growth factors
A_0 = 10**6

# Define the mean and standard deviation for the injury signal
I_mean = A_0  # Mean of the prolonged injury signal
sigma = 0.9 * A_0 # change this value to change the noise value

# Define the time step and total number of steps
dt = 0.1
t = np.linspace(0, 80, 1000)
num_steps = int(t[-1] / dt)

# Function to find steady CSF and PDGF levels for given mF and M levels
def CSF_PDGF_steady(x):
    mF = x[0]
    M = x[1]
    c_CSF = np.array([-1 * gamma, beta1 * mF - alpha1 * M - k2 * gamma, beta1 * k2 * mF])
    c_PDGF = np.array([-1 * gamma, beta2 * M + beta3 * mF - alpha2 * mF - gamma * k1, k1 * (beta2 * M + beta3 * mF)])
    CSF_roots = np.roots(c_CSF)
    PDGF_roots = np.roots(c_PDGF)
    root_pairs = []
    for CSF_root in CSF_roots:
        for PDGF_root in PDGF_roots:
            if np.isreal(CSF_root) and np.isreal(PDGF_root) and PDGF_root >= 0 and CSF_root >= 0:
                root_pairs.append((np.real(CSF_root), np.real(PDGF_root)))
    if len(root_pairs) == 0:
        return [np.nan, np.nan]
    return root_pairs[0]

# Function to calculate derivatives for mF and M
def mF_M_rates(x, t):
    mF = x[0]
    M = x[1]
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    if np.isnan(CSF) or np.isnan(PDGF):
        return [0, 0]
    dmFdt = mF * (lambda1 * (PDGF / (k1 + PDGF)) * (1 - mF / K) - mu1)
    dMdt = M * (lambda2 * (CSF / (k2 + CSF)) - mu2)
    return [dmFdt, dMdt]

# Function to calculate derivatives for prolonged injury
def transient_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    if t <= 2:
        new_derivatives[1] += A_0  # Add prolonged injury signal
    return new_derivatives

# Function to calculate derivatives for stochastic prolonged injury
def transient_stochastic_derivatives(x, t, dt):
    regular_derivatives = mF_M_rates(x, t)
    if t <= 2:
        stochastic_term = I_mean + sigma * np.random.normal(0, np.sqrt(dt))
    else:
        stochastic_term = 0  # Signal turned off after 4 days
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + stochastic_term
    return np.array(new_derivatives)

# Function to calculate the separatrix
def myofib_macro_ODE_reverse(x, t):
    mF = x[0]
    M = x[1]
    if 0 <= mF <= 10**6 and 0 <= M <= 10**7:
        return [-i for i in mF_M_rates(x, t)]
    else:
        return [0, 0]

# Function to find nullclines
def nullcline_mF(mF):
    smF_PDGF = (mu1 * k1 * K) / (lambda1 * K - mu1 * K - mF * lambda1)
    smF_M = -1 / beta2 * (beta3 * mF - alpha2 * mF * smF_PDGF / (k1 + smF_PDGF) - gamma * smF_PDGF)
    return [mF, smF_M]

def nullcline_M(M):
    sM_CSF = (k2 * mu2) / (lambda2 - mu2)
    sM_mF = ((k2 + sM_CSF) / (alpha1 * sM_CSF)) * (beta1 * M - gamma * sM_CSF)
    return [sM_mF, M]

# Calculate the separatrix
def calculate_separatrix(uns_soln2):
    eps = 1e-6
    t_sep = np.linspace(0, 800, 1000)
    separatrix_left = odeint(myofib_macro_ODE_reverse, [uns_soln2[0] - eps, uns_soln2[1] + eps], t_sep)
    separatrix_right = odeint(myofib_macro_ODE_reverse, [uns_soln2[0] + eps, uns_soln2[1] - eps], t_sep)
    return separatrix_left, separatrix_right

# Initial conditions for the separatrix
uns_soln2 = uns_soln2 = nullcline_mF(uns_soln)  
separatrix_left, separatrix_right = calculate_separatrix(uns_soln2)

# Initial conditions
x_in = [1, 1]  # Initial [mF, M] values


# Calculate deterministic prolonged injury trajectory
x_transient = odeint(transient_derivatives, x_in, t)
mF_transient = x_transient[:, 0]
M_transient = x_transient[:, 1]

# Start and state variables for the stochastic prolonged injury
x_stochastic = np.zeros((num_steps, len(x_in)))
x_stochastic[0] = x_in

# Iterate using the Euler-Maruyama method
for i in range(1, num_steps):
    t_step = i * dt
    x_stochastic[i] = x_stochastic[i-1] + transient_stochastic_derivatives(x_stochastic[i-1], t_step, dt) * dt

# Extract results for plotting
mF_stochastic = x_stochastic[:, 0]
M_stochastic = x_stochastic[:, 1]

# Plot the trajectories and the separatrix
plt.figure()
plt.plot(mF_stochastic, M_stochastic, label='Stochastic Transient Trajectory', color='purple')
plt.scatter(mF_stochastic[-1], M_stochastic[-1], color='red', zorder=5, label='Stochastic End Point')

plt.plot(mF_transient, M_transient, label='Deterministic Transient Trajectory', color='green')
plt.scatter(mF_transient[-1], M_transient[-1], color='blue', zorder=5, label='Deterministic End Point')

plt.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black', label='Separatrix')
plt.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')

plt.xlabel('Myofibroblast Concentration')
plt.ylabel('Macrophage Concentration')
plt.title('Trajectory and Stop Points for Transient Injuries')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 10**7)
plt.ylim(1, 10**7)
plt.legend()
plt.show()
