from parameters import *
import numpy as np

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