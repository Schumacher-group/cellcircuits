import numpy as np

#Define Constants
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

#time vector
t = np.linspace(0, 80, 1000)

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")