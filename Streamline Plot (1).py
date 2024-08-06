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
# to plot the streamlines, we need to redefine functions to find steady CSF and PDGF levels that work for arrays
def CSF_PDGF_steady_array(x): # finds steady CSF and PDGF levels for given mF and M levels
    mF = x[0]
    M = x[1]
    # equation for steady CSF is -gamma*(CSF)**2 + CSF*(beta1*mF-alpha1*M-k2*gamma) + beta1*k2*mF
    # equation for steady PDGF is  -gamma*(PDGF)**2 + PDGF * (beta2*M + beta3*mF -alpha2 * mF- gamma * k1) +k1*(beta2*M+beta3*mF)
    c_CSF_array = np.array([-1*gamma*np.ones(np.shape(mF)), beta1*mF-alpha1*M-k2*gamma, beta1*k2*mF])
    c_PDGF_array = np.array([-1*gamma*np.ones(np.shape(mF)), beta2*M + beta3*mF -alpha2 * mF - gamma * k1, k1*(beta2*M+beta3*mF)])
    # define empty arrays fo CSF and PDGF
    CSF_array = np.zeros(np.shape(mF))
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

def mF_M_rates_array(exp_mF, exp_M, t):
    # we need dmFdt and dMdt to be plotted at different values as streamplot can only take in linearly spaced values,
    # so we take in the exponents of mF and M values to get logarithmically spaced
    mF = 10**exp_mF
    M = 10**exp_M
    CSF, PDGF = CSF_PDGF_steady_array([mF, M])
    dmFdt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    dMdt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return dmFdt, dMdt

mFM_space = np.logspace(0, 7, 10**4)
# mFnull1, mFnull2, mFnull3 are intervals that do not contain any poles
# smoothmF1 and smoothmF2 are intervals that contain poles
mFnull1 = np.logspace(0, 5.7, 10**3)
smoothmF1 = np.logspace(5.7, 5.85, 10**3)
mFnull2 = np.logspace(5.85, 5.95, 10**3)
smoothmF2 = np.logspace(5.95, 6.05, 10**3)
mFnull3 = np.logspace(6.05, 7, 10**3)
def intersectionNull_bad(): # finds intersection of nullclines, bad estimate
    mF_list = []
    def diff(x):
        return nullcline_M(x)[1] - nullcline_mF(x)[1]
    for i in range(len(mFM_space)-1):
        if diff(mFM_space[i]) == 0 or diff(mFM_space[i]) * diff(mFM_space[i+1]) < 0:
            mF_list.append(mFM_space[i])
    return mF_list
fpt_mF_bad = intersectionNull_bad()
fpt_M_bad = [nullcline_M(i)[1] for i in fpt_mF_bad]
print(intersectionNull_bad(), fpt_M_bad, "fixed point approximations where mF and M are nonzero")
def nulldiff(x):
    return nullcline_M(x)[1] - nullcline_mF(x)[1]
# use initial fixed point approximations to find solutions for the unstable point and the hot fibrosis point
uns_guess = fpt_mF_bad[0]
uns_guess = fpt_mF_bad[0]
second_guess = fpt_mF_bad[1]
uns_soln = fsolve(nulldiff, uns_guess)
uns_soln = uns_soln[0] # changes array to float
hotfibr = fsolve(nulldiff, second_guess)
hotfibr = hotfibr[0]  # changes array to float
print(uns_soln, "mF value at unstable fix pt")
print(hotfibr, "mF value at hot fibrosis")
uns_soln2 = nullcline_mF(uns_soln) # gives [mF, M] at unstable fixed point from mF value
uns_CSFPDGF = CSF_PDGF_steady(uns_soln2) # gives [CSF, PDGF] at unstable fixed point from mF value
nullsoln4 = uns_soln2 + uns_CSFPDGF # get [mF, M, CSF, PDGF] at unstable fixed point
hotfibr2 = nullcline_mF(hotfibr)
hot_CSF_PDGF = CSF_PDGF_steady(hotfibr2)
hotfibr4 = hotfibr2 + hot_CSF_PDGF

print(nullsoln4)
print(myofib_macro(nullsoln4, t), "rates at unstable fixed pt")
def cold_fibr(): # finds the cold fibrosis point
    # Set M = 0 in eqn 4, use eqn 1. solve system for PDGF, get a cubic
    PDGF_coeff = np.array([-gamma, (K/lambda1)*(lambda1-mu1)*(beta3-alpha2)-gamma*k1,
                           (K/lambda1)*(lambda1-mu1+beta3-alpha2), -mu1* k1**2 * beta3 * K/lambda1])
                            # rearranged from eqns in transparent methods
    coldPDGF = np.roots(PDGF_coeff)
    coldmF = []
    for coldroot in coldPDGF:
        if np.isreal(coldroot):
            coldmF.append(K * ((lambda1-mu1)/(lambda1)-(mu1*k1)/(lambda1*np.real(coldroot)))) # finds mF value given PDGF value
    return coldmF[0]
coldfibr2 = [cold_fibr(), 1]
def rev_mF_M(x,t):
    mF = x[0]
    M = x[1]
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    dmFdt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    dMdt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return [-dmFdt, -dMdt]

def myofib_macro_ODE_reverse(x, t):    # outputs reverse derivative
    # variables
    mF = x[0]
    M = x[1]
    if 0 <= mF <= 10**6 and 0 <= M <= 10**7:
        return [-i for i in mF_M_rates(x, t)]
    else:
        return [0, 0]
eps = 1e-6
t_sep = np.linspace(0, 800, 1000)
separatrix_left = odeint(myofib_macro_ODE_reverse, [uns_soln2[0]-eps, uns_soln2[1]+eps], t_sep)
separatrix_right = odeint(myofib_macro_ODE_reverse, [uns_soln2[0]+eps, uns_soln2[1]-eps], t_sep)

fig = plt.figure()
mF_mesh = np.linspace(0, 6, 25)
M_mesh = np.linspace(0, 6, 25)
mF_stream, M_stream = np.meshgrid(mF_mesh, M_mesh) # returns
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
mF_rate, M_rate = mF_M_rates_array(mF_stream, M_stream, t)
# we need to adjust for the exponential values
mF_rate_scaled = mF_rate/(10**mF_stream)
M_rate_scaled = M_rate/(10**M_stream)
strm = ax.streamplot(mF_stream, M_stream, mF_rate_scaled, M_rate_scaled,
                     color = (np.sqrt((mF_rate_scaled)**2 + (M_rate_scaled)**2)) , cmap = 'autumn')
ax.set_xlim(0, 6)
ax.set_ylim(0,6)
ax.set_xticks([])
ax.set_yticks([])
ax2.set_xlabel('myofibroblasts')
ax2.set_ylabel('macrophages')
ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1, 10**6)
ax2.set_ylim(1, 10**6)
ax2.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
ax2.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')
ax2.plot(coldfibr2[0], coldfibr2[1], marker = 'o', color = "black")
print(mF_M_rates([10,5], t))