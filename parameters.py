import numpy as np

#Define Constants
lambda1 = 0.9  # proliferation rate of mF in 1/day
lambda2 = 0.8  #  proliferation rate of M in 1/day
mu1 = 0.3      #       removal rate of mF in 1/day
mu2 = 0.3      #        removal rate of M in 1/day
K = 10 ** 6    #  carrying capacity of mF in cells/ml
k1 = 10 ** 9   #  binding affinity of CSF in molecules/ml
k2 = 10 ** 9   # binding affinity of PDGF in molecules/ml


# converted from paper to match units min -> day
beta1 = 470 * 60 * 24  #    max secretion rate of CSF by mF in molecules/(cell * day) 
beta2 = 70 * 60 * 24   #    max secretion rate of PDGF by M in molecules/(cell * day)
beta3 = 240 * 60 * 24  #   max secretion rate of PDGF by mF in molecules/(cell * day)
alpha1 = 940 *60 * 24  #   max endocytosis rate of CSF by M in molecules/(cell * day)
alpha2 = 510 * 60 * 24 # max endocytosis rate of PDGF by mF in molecules/(cell * day)
gamma = 2              # degradation rate of growth factors in 1/day
A_0 = 10**6