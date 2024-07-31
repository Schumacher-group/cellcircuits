from parameters import *
import numpy as np
from scipy.optimize import fsolve

#find steady state for CSF and PDGF given mF and M levels using the fast timescale 
def CSF_PDGF_steady(x):
    mF, M = x
    
    # equation for steady CSF is   0 = -gamma*(CSF)**2 + CSF*(beta1*mF-alpha1*M-k2*gamma) + beta1*k2*mF
    # equation for steady PDGF is  0 = -gamma*(PDGF)**2 + PDGF * (beta2*M + beta3*mF -alpha2 * mF- gamma * k1) +k1*(beta2*M+beta3*mF)


    c_CSF = np.array([-1 * gamma, beta1 * mF - alpha1 * M - k2 * gamma, beta1 * k2 * mF])
    c_PDGF = np.array([-1 * gamma, beta2 * M + beta3 * mF -alpha2 * mF - gamma * k1, k1 * (beta2 * M + beta3 * mF)])
    CSF_roots = np.roots(c_CSF)
    PDGF_roots = np.roots(c_PDGF)

    root_pairs = [(CSF_root, PDGF_root) for CSF_root in CSF_roots for PDGF_root in PDGF_roots
                  if np.isreal(CSF_root) and np.isreal(PDGF_root) and CSF_root >= 0 and PDGF_root >= 0]

    return root_pairs

def mF_M_rates(exp_mF, exp_M, t):
    # we need dmFdt and dMdt to be plotted at different values as streamplot can only take in linearly spaced values,
    # so we take in the exponents of mF and M values to get logarithmically spaced
    mF = 10**exp_mF
    M = 10**exp_M
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    d_mF_dt = mF * (lambda1 * (PDGF / (k1 + PDGF)) * (1 - mF / K) - mu1)
    d_M_dt = M * (lambda2 * (CSF / (k2 + CSF)) - mu2)
    return d_mF_dt, d_M_dt

def nullcline_mF(mF):
    smF_PDGF = (mu1 * k1 *K) / (lambda1 * K - mu1 *K - mF *lambda1)
    smF_M = -1 / beta2 * (beta3 * mF - alpha2 * mF * smF_PDGF / (k1 + smF_PDGF) - gamma * smF_PDGF)
    return [mF, smF_M]

def nullcline_M(sM_mF):
    sM_CSF = (k2 * mu2) / (lambda2 - mu2)
    M = ((k2 + sM_CSF) / (alpha1 * sM_CSF)) * (beta1 * sM_mF - gamma * sM_CSF)
    return [sM_mF, M]

# finds intersection of nullclines, bad estimate
def intersectionNull_bad(mFM_space):
    def diff(x):
        return nullcline_M(x)[1] - nullcline_mF(x)[1]
    mF_list = []
    for i in range(len(mFM_space) - 1):
        if diff(mFM_space[i]) * diff(mFM_space[i + 1]) < 0:
            mF_list.append(mFM_space[i])
    return mF_list

def cold_fibr():
    # Set M = 0 in eqn 4, use eqn 1. solve system for PDGF, get a cubic
    PDGF_coeff = np.array([-gamma, 
                           (K / lambda1) * (lambda1 - mu1) * (beta3 - alpha2) - gamma * k1,
                           (K / lambda1) * (lambda1 - mu1 + beta3 - alpha2),
                           - mu1 * k1**2 * beta3 * K / lambda1])
    # rearranged from eqns in transparent methods
    coldPDGF = np.roots(PDGF_coeff)
    coldmF = []
    for coldroot in coldPDGF:
        if np.isreal(coldroot):
            coldmF.append(K * ((lambda1-mu1)/(lambda1)-(mu1*k1)/(lambda1*np.real(coldroot)))) # finds mF value given PDGF value
    return coldmF[0]

coldfibr2 = [cold_fibr(), 1]
