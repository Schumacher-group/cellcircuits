#import numpy as np is contained in parameters
from parameters import *
from scipy.optimize import fsolve
from scipy.integrate import odeint


#Outputs list of gradients, state encompasses concentrations for mF, M, CSF and PDGF as cells per ml
def myofib_macro(state): # outputs list of gradients
    
    mF, M, CSF, PDGF = state

    d_mF_dt = mF * (lambda1 * (PDGF / (k1 + PDGF)) * (1 - mF / K) - mu1)
    d_M_dt = M * (lambda2 * (CSF / (k2+CSF)) - mu2)
    d_CSF_dt = beta1 * mF - alpha1 * M * (CSF / (k2 + CSF)) - gamma * CSF
    d_PDGF_dt = beta2 * M + beta3 * mF - alpha2 * mF * (PDGF / (k1 + PDGF))- gamma * PDGF

    return [d_mF_dt, d_M_dt, d_CSF_dt, d_PDGF_dt]




#find steady state for CSF and PDGF given mF and M levels using the fast timescale 
def CSF_PDGF_steady(x):
    mF = x[0]
    M = x[1]
    
    #In case the number of macrophages drops below 0 (e.g. in the numerical integration) interpret it as 0
    if M < 0:
        M = 0
    if mF < 0:
        mF = 0

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

#t parameter is necessary for using the odeint function from scipy.integrate
def mF_M_rates(state, t):
    mF, M = state
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    d_mF_dt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    d_M_dt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return [d_mF_dt, d_M_dt]

#outputs reverse derivative, state encompasses mF, M concentrations 
def myofib_macro_ODE_reverse(state, t):
    
    #checking if concentrations are in a reasonable range
    return [-d for d in mF_M_rates(state, t)] if all(0 <= x <= 10**7 for x in state) else [0, 0]

def rev_mf_M_rates(state, t):
    derivatives = mF_M_rates(state, t)
    return [-d for d in derivatives]


# finds steady CSF and PDGF levels for given mF and M levels numerically from the steady state equations
def CSF_PDGF_steady_array(x):
    mF = x[0]
    M = x[1]
    # equation for steady CSF is -gamma*(CSF)**2 + CSF*(beta1*mF-alpha1*M-k2*gamma) + beta1*k2*mF
    # equation for steady PDGF is  -gamma*(PDGF)**2 + PDGF * (beta2*M + beta3*mF -alpha2 * mF- gamma * k1) +k1*(beta2*M+beta3*mF)

    c_CSF_array = np.array([-1*gamma*np.ones(np.shape(mF)), beta1*mF-alpha1*M-k2*gamma, beta1*k2*mF])
    c_PDGF_array = np.array([-1*gamma*np.ones(np.shape(mF)), beta2*M + beta3*mF -alpha2 * mF - gamma * k1, k1*(beta2*M+beta3*mF)])
    # define empty arrays fo CSF and PDGF
    CSF_array = np.zeros(np.shape(mF)) #is an array of the form [[][]]
    PDGF_array = np.zeros(np.shape(mF))
    for i in range(0, np.shape(mF)[0]):
        for j in range(0, np.shape(mF)[1]):
            # get 1d arrays of CSF and PDGF coefficients for each grid value
            c_CSF = np.array([c_CSF_array[0][i][j], c_CSF_array[1][i][j], c_CSF_array[2][i][j]])
            c_PDGF = np.array([c_PDGF_array[0][i][j], c_PDGF_array[1][i][j], c_PDGF_array[2][i][j]])
            CSF_roots = np.roots(c_CSF)
            PDGF_roots = np.roots(c_PDGF)
            for CSF_root in CSF_roots:
                for PDGF_root in PDGF_roots:
                    if np.isreal(CSF_root) and np.isreal(PDGF_root) and PDGF_root >= 0 and CSF_root >= 0:
                        CSF_array[i][j] = CSF_root
                        PDGF_array[i][j] = PDGF_root
    return [CSF_array, PDGF_array] 


def mF_M_rates_array(exp_mF, exp_M):
    # we need dmFdt and dMdt to be plotted at different values as streamplot can only take in linearly spaced values,
    # so we take in the exponents of mF and M values to get logarithmically spaced
    mF = 10**exp_mF
    M = 10**exp_M
    CSF, PDGF = CSF_PDGF_steady_array([mF, M])
    d_mF_dt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    d_M_dt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return d_mF_dt, d_M_dt


def nullcline_mF(mF):
    smF_PDGF = (mu1 * k1 *K) / (lambda1 * K - mu1 *K - mF *lambda1)
    smF_M = -1 / beta2 * (beta3 * mF - alpha2 * mF * smF_PDGF / (k1 + smF_PDGF) - gamma * smF_PDGF)
    return [mF, smF_M]

def nullcline_M(sM_mF):
    sM_CSF = (k2 * mu2) / (lambda2 - mu2)
    M = ((k2 + sM_CSF) / (alpha1 * sM_CSF)) * (beta1 * sM_mF - gamma * sM_CSF)
    return [sM_mF, M]


def calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix):
    eps = 1e-6
    separatrix_left = odeint(myofib_macro_ODE_reverse, [unstable_fixed_point_mF_M[0] - eps,
                                                        unstable_fixed_point_mF_M[1] + eps], t_separatrix)
    separatrix_right = odeint(myofib_macro_ODE_reverse, [unstable_fixed_point_mF_M[0] + eps,
                                                         unstable_fixed_point_mF_M[1] - eps], t_separatrix)
    return np.array(separatrix_left), np.array(separatrix_right)

def nulldiff(x):
    return nullcline_M(x)[1] - nullcline_mF(x)[1]

# finds intersection of nullclines, bad first estimate
def intersectionNull_bad(mFM_space):
    mF_list = []
    for i in range(len(mFM_space) - 1):
        if nulldiff(mFM_space[i]) * nulldiff(mFM_space[i + 1]) < 0 or nulldiff(mFM_space[i]) == 0:
            mF_list.append(mFM_space[i])
    return mF_list


def unstable_fixed_point_hotfibrosis_mF_M(mFM_space):
    #use intersection_Null_bad to make a first rough approximation of the fixed points
    fixed_point_mF_bad = intersectionNull_bad(mFM_space)
    fixed_point_M_bad = [nullcline_M(i)[1] for i in fixed_point_mF_bad]

    unstable_guess = fixed_point_mF_bad[0]
    hotfibrosis_guess = fixed_point_mF_bad[1]

    #Now use fsolve function to get a more precise solution and make it a floating number (instead of array)
    unstable_fixed_point_mF = fsolve(nulldiff, unstable_guess)[0]

    hotfibrosis_mF = fsolve(nulldiff, hotfibrosis_guess)[0]

    #find mF_M concentration at unstable fixed point
    unstable_fixed_point_mF_M = nullcline_mF(unstable_fixed_point_mF)

    #find mF_M concentrations at hotfibrosis point
    hotfibrosis_mF_M = nullcline_mF(hotfibrosis_mF)
    

    return (unstable_fixed_point_mF_M, hotfibrosis_mF_M)


def cold_fibr():
    # Set M = 0 in eqn 4, use eqn 1. solve system for PDGF, get a cubic

    PDGF_coeff = np.array([-gamma,
                           (K / lambda1) * (lambda1 - mu1) * (beta3 - alpha2) - gamma * k1,
                           (K * k1 / lambda1) * (beta3 * lambda1 - 2 * mu1 * beta3 + mu1 *alpha2),
                           -k1**2 * mu1 * K * beta3 / lambda1])
    # rearranged from eqns in transparent methods
    coldPDGF = np.roots(PDGF_coeff)
    coldmF = []
    
    for coldroot in coldPDGF:
        if np.isreal(coldroot) and coldroot >= 0:
            coldmF.append(K * ((lambda1 - mu1) / (lambda1) - (mu1 * k1) / (lambda1 * np.real(coldroot)))) # finds mF value given PDGF value
    
    return coldmF

#Returns true if trajectory crosses seperatrix and will then move to the hot fibrosis state
def check_hot_fibrosis(end_point, separatrix_left, separatrix_right):
    interpolation = np.interp(end_point[0], np.concatenate( (separatrix_left[:, 0], separatrix_right[:, 0]) ),
                              np.concatenate( (separatrix_left[:, 1], separatrix_right[:, 1]) ))
    if end_point[1] > interpolation:
        return True
    else: False


#Determine first crossing indices for integrated trajectory
def find_first_crossing_index(x, interpolation):
    
    interpolation_values = interpolation(x[:, 0])

    #check if the trajectory crosses the separatrix
    crosses = (x[:, 1] > interpolation_values)

    #find first index where crossing occours
    crossing_indices = np.where(crosses)[0]

    return crossing_indices[0] if crossing_indices.size > 0 else -1


def time_taken(traj, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M):
    end_point = [0, 0]
    for fixed_pt in [hotfibrosis_mF_M, unstable_fixed_point_mF_M]:
        if (traj[-1][0] - fixed_pt[0])**2 + (traj[-1][1] - fixed_pt[1])**2 < (traj[-1][0] - end_point[0])**2 + (traj[-1][1] - end_point[1])**2:
            end_point = fixed_pt 
            for i in range(len(traj)):
                #end point length included to take scale into account
                if (traj[i][0] - end_point[0])**2 + (traj[i][1] - end_point[1])**2 < 1e-3 *((end_point[0])**2 + (end_point[1])**2) :
                    return t[i]
                
            print("\033[91mWarning: The calculated trajectory time is higher than calculated. Try to adjust the time intervall.\033[0m")
            return t[-1] 
    for i in range(len(traj)):
        if ((traj[i][0]) ** 2 + (traj[i][1]) ** 2) < 1e-3:
            return t[i]
        
    print("\033[91mWarning: The calculated trajectory time is higher than calculated. Try to adjust the time intervall.\033[0m")
    return t[-1]

def time_taken_rd(traj, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M): 
    return round(time_taken(traj, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M), 2)

def array_statistics(array, unit = ''):
    print(f"Count: {len(array)}")
    print(f"Mean: {np.mean(array):.2f} {unit}")
    print(f"StdDev: {np.std(array):.2f} {unit}")
    print(f"Minimum: {np.min(array)} {unit}")
    print(f"Maximum: {np.max(array)} {unit}")
    print(f"Median: {np.median(array)} {unit}")