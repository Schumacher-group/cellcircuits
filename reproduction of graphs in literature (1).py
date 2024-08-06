import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
lambda1 = 0.9  # proliferation rate of mF
lambda2 = 0.8  # proliferation rate of M
mu1 = 0.3  # removal rate of mF
mu2 = 0.3  # removal rate of M
K = 10 ** 6  # carrying capacity of mF
k1 = 10 ** 9  # binding affinity of CSF
k2 = 10 ** 9  # binding affinity of PDGF
# converted from paper to match units min -> day
beta1 = 470 * 60 * 24 # max secretion rate of CSF by mF
beta2 = 70 * 60 * 24  # max secretion rate of PDGF by M
beta3 = 240 * 60 * 24  # max secretion rate of PDGF by mF
alpha1 = 940 *60 * 24 # max endocytosis rate of CSF by M
alpha2 = 510 * 60 * 24   # max endocytosis rate of PDGF by mF
gamma = 2  # degradation rate of growth factors
A_0 = 10**6

def myofib_macro(x, t): # outputs list of gradients
    # variables
    mF = x[0]
    M = x[1]
    CSF = x[2]
    PDGF = x[3]
    # diff eqns
    dmFdt= mF*(lambda1 * ((PDGF)/(k1 + PDGF))*(1-mF/K) -mu1)
    dMdt = M*(lambda2*(CSF/(k2+CSF))-mu2)
    dCSFdt = beta1*mF-alpha1*M*(CSF/(k2+CSF))-gamma*CSF
    dPDGFdt= beta2*M+beta3*mF-alpha2*mF*(PDGF)/(k1+PDGF)-gamma*PDGF
    return [dmFdt, dMdt, dCSFdt, dPDGFdt]

# playing around with trajetories
x0 = [6*10**3, 7*10**3] # initial point for trajectory
x1 = [10**3, 10**3, 0, 0]
t = np.linspace(0, 80, 1000)
def derivatives(mF_M):
    mF = mF_M[0]
    M = mF_M[1]
    CSF = mF_M[2]
    PDGF = mF_M[3]
    # diff eqns
    dmFdt= mF*(lambda1 * ((PDGF)/(k1 + PDGF))*(1-mF/K) -mu1)
    dMdt = M*(lambda2*(CSF/(k2+CSF))-mu2)
    dCSFdt = beta1*mF-alpha1*M*(CSF/(k2+CSF))-gamma*CSF
    dPDGFdt = beta2*M+beta3*mF-alpha2*mF*(PDGF)/(k1+PDGF)-gamma*PDGF
    return (dmFdt, dMdt, dCSFdt,dPDGFdt)

def CSF_PDGF_steady(x): # finds steady CSF and PDGF levels for given mF and M levels
    mF = x[0]
    M = x[1]
    # equation for steady CSF is -gamma*(CSF)**2 + CSF*(beta1*mF-alpha1*M-k2*gamma) + beta1*k2*mF
    # equation for steady PDGF is  -gamma*(PDGF)**2 + PDGF * (beta2*M + beta3*mF -alpha2 * mF- gamma * k1) +k1*(beta2*M+beta3*mF)
    c_CSF = np.array([-1*gamma, beta1*mF-alpha1*M-k2*gamma, beta1*k2*mF])
    c_PDGF = np.array([-1*gamma, beta2*M + beta3*mF -alpha2 * mF - gamma * k1, k1*(beta2*M+beta3*mF)])
    CSF_roots = np.roots(c_CSF)
    PDGF_roots = np.roots(c_PDGF)
    root_pairs = []
    for CSF_root in CSF_roots:
        for PDGF_root in PDGF_roots:
                if np.isreal(CSF_root) and np.isreal(PDGF_root) and PDGF_root >= 0 and CSF_root >= 0:
                    root_pairs.append(CSF_root)
                    root_pairs.append(PDGF_root)
    return(root_pairs)

def mF_M_rates(x, t):
    mF = x[0]
    M = x[1]
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    dmFdt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    dMdt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return [dmFdt, dMdt]
x_ = odeint(mF_M_rates, x0, t)
mF = x_[:, 0]
M = x_[:, 1]
mF_log = np.logspace(0, 7, 10)
M_log = np.logspace(1, 7, 10)
def nullcline_mF(x): # finds nullcline of mF
    mF = x
    smF_PDGF =  (mu1 * k1 * K)/(lambda1 * K - mu1*K - mF*lambda1) # after rearranging eqn 1 equated to 0
    smF_M = -1/beta2 *( beta3*mF-alpha2*mF*smF_PDGF/(k1+smF_PDGF)-gamma*smF_PDGF) # rearranging eqn 4 equated to 0
    return [mF, smF_M]
def nullcline_M(x): # finds nullcline of M
    sM_mF = x
    sM_CSF = (k2 * mu2) / (lambda2 - mu2) # equate eqn 2 to 0
    M = ((k2 +sM_CSF)/(alpha1 * sM_CSF) ) * (beta1*sM_mF-gamma*sM_CSF) # equate eqn 3 to 0
    return [sM_mF, M]


#reproduce graphs in in paper - step wise signal graphs and trajectory graphs for each injury

def theta(t): # heaviside step function
    if t >= 0:
        return 1
    else:
        return 0
# define transient, repetitive and prolonged signals
def transient(t):
    return A_0*(theta(t) - theta(t-2))
#def repetitive(t):
    return transient(t) + transient(t-4)
def prolonged(t):
    return A_0 *(theta(t)- theta(t-4))
# defining new derivative values for transient, repetitive and prolonged signals
def transient_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + transient(t)
    return new_derivatives
def repetitive_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + repetitive(t)
    return new_derivatives
def prolonged_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + prolonged(t)
    return new_derivatives
x_in = [1, 1] # initial [mF, M] values
print(repetitive_derivatives(x_in, 4))

x_transient = odeint(transient_derivatives, x_in, t)
x_repetitive = odeint(repetitive_derivatives, x_in, t)
x_prolonged = odeint(prolonged_derivatives, x_in, t)

def time_taken(traj):
    end_point = [0, 0]
    for fixed_pt in [hotfibr2, uns_soln2]:
        if (traj[-1][0] - fixed_pt[0])**2 + (traj[-1][1] - fixed_pt[1])**2 < (traj[-1][0] - end_point[0])**2 + (traj[-1][1] - end_point[1])**2:
            end_point = fixed_pt # find correct end point
            for i in range(len(traj)):
                if (traj[i][0] - end_point[0])**2 + (traj[i][1] - end_point[1])**2 < 1e-3 *((end_point[0])**2 + (end_point[1])**2) :
                    return t[i] # finds time at which trajectory reaches end point if end point is not 0
    for i in range(len(traj)):
        if ((traj[i][0]) ** 2 + (traj[i][1]) ** 2) < 1e-3:
            return t[i] # finds time at which trajectory reaches end point if end point is 0
        
def time_taken_rd(traj): # so that time_taken can be rounded and included in figures, print time taken
    return round(time_taken(traj), 2)
print(time_taken(x_repetitive))
print(time_taken(x_prolonged))
print(time_taken(x_transient))

#plot all injury trajectories and the injusry signals
fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)
fig.subplots_adjust(hspace= 0.5)
# setting up trajectory plots
for ax2 in (ax21, ax22, ax23):
    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
    ax2.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')
    ax2.plot(coldfibr2[0], coldfibr2[1], marker='o', color="black")
# setting up injury plots
for ax1 in (ax11, ax12, ax13):
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    ax1.set_xticks([0, 2, 4, 6, 8])
    ax1.set_yticks([0, 9], [0, "A0".translate(SUB)])
# plotting separate signals
ax21.plot(x_transient[:, 0], x_transient[:, 1], 'deepskyblue')
ax21.set_title("time taken: " + str(time_taken_rd(x_transient)) + " days")
ax22.plot(x_repetitive[:, 0], x_repetitive[:, 1], 'orange')
ax22.set_title("time taken: " + str(time_taken_rd(x_repetitive)) + " days")
ax23.plot(x_prolonged[:, 0], x_prolonged[:, 1], 'green')
ax23.set_title("time taken: " + str(time_taken_rd(x_prolonged)) + " days")

# plotting injury signal graphs
#transient signal
tr_x = [0, 2, 3]
tr_y = [0, 9, 0]
ax11.step(tr_x, tr_y, color = 'deepskyblue')
ax11.set_title('Transient signal')
#repetitive signal
re_x = [0, 2, 4, 6, 7]
re_y = [0, 9, 0, 9, 0]
ax12.step(re_x, re_y, color = 'orange')
ax12.set_title('Repetitive signal')
# prolonged signal
pr_x = [0, 4, 5]
pr_y = [0, 9, 0]
ax13.step(pr_x, pr_y, color = 'green')
ax13.set_title('Prolonged signal')

# setting up plots for figs 3G-F in paper
for ax1 in (ax11, ax12, ax13):
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('Myofibroblasts')
    ax1.set_xticks([0, 2, 4, 6, 8])
    ax1.set_yticks([0, 9], [0, "A0".translate(SUB)])
    
    # Define the heaviside step function
def theta(t): 
    return 1 if t >= 0 else 0

# Define transient, repetitive, and prolonged signals
def transient(t):
    return A_0 * (theta(t) - theta(t - 2))

def repetitive(t):
    return transient(t) + transient(t - 4)

def prolonged(t):
    return A_0 * (theta(t) - theta(t - 4))

# Define new derivative values for transient, repetitive, and prolonged signals
def transient_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + transient(t)
    return new_derivatives

def repetitive_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + repetitive(t)
    return new_derivatives

def prolonged_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + prolonged(t)
    return new_derivatives

# Initial conditions
x_in = [1, 1] # initial [mF, M] values

# Calculate trajectories
x_transient = odeint(transient_derivatives, x_in, t)
x_repetitive = odeint(repetitive_derivatives, x_in, t)
x_prolonged = odeint(prolonged_derivatives, x_in, t)

# Extract myofibroblast concentrations
mF_transient = x_transient[:, 0]
mF_repetitive = x_repetitive[:, 0]
mF_prolonged = x_prolonged[:, 0]

# Plot myofibroblast concentration vs time for each injury type
plt.figure()
plt.plot(t, mF_transient, label='Transient', color='deepskyblue')
plt.plot(t, mF_repetitive, label='Repetitive', color='orange')
plt.plot(t, mF_prolonged, label='Prolonged', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Myofibroblast Concentration')
plt.title('Myofibroblast Concentration vs Time for Different Injury Types')
plt.legend()
plt.show()

# Define the Heaviside step function
def theta(t): 
    return 1 if t >= 0 else 0

# Define transient, repetitive, and prolonged signals
def transient(t):
    return A_0 * (theta(t) - theta(t - 2))

def repetitive(t):
    return transient(t) + transient(t - 4)

def prolonged(t):
    return A_0 * (theta(t) - theta(t - 4))

# Define new derivative functions for transient, repetitive, and prolonged signals
def transient_derivatives(x, t):
    dmFdt, dMdt = mF_M_rates(x, t)
    dMdt += transient(t)
    return [dmFdt, dMdt]

def repetitive_derivatives(x, t):
    dmFdt, dMdt = mF_M_rates(x, t)
    dMdt += repetitive(t)
    return [dmFdt, dMdt]

def prolonged_derivatives(x, t):
    dmFdt, dMdt = mF_M_rates(x, t)
    dMdt += prolonged(t)
    return [dmFdt, dMdt]

# Initial conditions
x_in = [1, 1] # initial [mF, M] values

# Calculate trajectories
t = np.linspace(0, 80, 1000)
x_transient = odeint(transient_derivatives, x_in, t)
x_repetitive = odeint(repetitive_derivatives, x_in, t)
x_prolonged = odeint(prolonged_derivatives, x_in, t)

# Extract macrophage concentrations
M_transient = x_transient[:, 1]
M_repetitive = x_repetitive[:, 1]
M_prolonged = x_prolonged[:, 1]

# Plot macrophage concentration vs time for each injury type
plt.figure()
plt.plot(t, M_transient, label='Transient', color='deepskyblue')
plt.plot(t, M_repetitive, label='Repetitive', color='orange')
plt.plot(t, M_prolonged, label='Prolonged', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Macrophage Concentration')
plt.title('Macrophage Concentration vs Time for Different')