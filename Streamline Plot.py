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