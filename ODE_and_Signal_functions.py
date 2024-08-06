import numpy as np
from parameters import *
from analysis import CSF_PDGF_steady

#Outpust list of gradients, state encompasses concentrations for mF, M, CSF and PDGF as cells per ml
def myofib_macro(state, t): # outputs list of gradients
    
    mF, M, CSF, PDGF = state

    d_mF_dt = mF * (lambda1 * (PDGF / (k1 + PDGF)) * (1 - mF / K) - mu1)
    d_M_dt = M * (lambda2 * (CSF / (k2+CSF)) - mu2)
    d_CSF_dt = beta1 * mF - alpha1 * M * (CSF / (k2 + CSF)) - gamma * CSF
    d_PDGF_dt = beta2 * M + beta3 * mF - alpha2 * mF * (PDGF / (k1 + PDGF))- gamma * PDGF

    return [d_mF_dt, d_M_dt, d_CSF_dt, d_PDGF_dt]


def mF_M_rates(state, t):
    mF, M = state
    
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    d_mF_dt = mF * (lambda1 * (PDGF / (k1 + PDGF)) * (1 - mF / K) - mu1)
    d_M_dt = M * (lambda2 * (CSF / (k2 + CSF)) - mu2)
    return [d_mF_dt, d_M_dt]

def rev_mf_M_rates(state, t):
    derivatives = mF_M_rates(state, t)
    return [-d for d in derivatives]

#outputs reverse derivative, state encompasses mF, M concentrations 
def myofib_macro_ODE_reverse(state, t):
    derivatives = myofib_macro(state, t)
    
    #checking if concentrations are in a reasonable range
    return [-d for d in derivatives] if all(0 <= x <= 10**7 for x in state) else [0, 0]


#Step function
def theta(t):
    return np.heaviside(t, 1)

#Signal functions
#A generalized signal function
def signal(t, start = 0, duration = 1, amplitude = 1):
    return amplitude * (theta(t) - theta(t - (start + duration)))



def transient(t):
    return signal(t, start = 0, duration = 2, amplitude = A_0)

def repetetive(t):
    return transient(t) + transient(t - 4)

def prolonged(t):
    return A_0 * (theta(t) - theta(t - 4))

def sharp_signal(t):
    return (A_0 /0.2) * (theta(t) - theta(t - 0.2))

def one_signal(t, delta_t):
    return (A_0 * 4/ delta_t) * (theta(t) - theta(t - delta_t))



#Generalised block signal function
def block(t, amplitudes, signal_starting_points, signal_durations):
    total_signal = np.zeros_like(t)
    for amplitude, start, duration in zip(amplitudes, signal_starting_points, signal_durations):
        total_signal += signal(t, start=start, duration=duration, amplitude=amplitude)
    return total_signal

def blocks1(t):
    return 0.25 * one_signal(t, 8) + 0.35 * one_signal(t - 8, 2) + 0.4 * one_signal(t - 10,2 )

def blocks2(t):
    return 0.6 * one_signal(t, 20) + 0.3 * one_signal(t - 20, 1) + 0.1 * one_signal(t - 21, 3)

def blocks3(t):
    return 0.5 * one_signal(t, 2) + 0.3 * one_signal(t - 2, 2) + 0.2 * one_signal(t - 4, 2)


#Generalized signal/block derivative function, added signal increaces macrophage rate
def adjusted_derivatives_with_signal(state, t, signal_func):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += signal_func(t)
    return derivatives



def transient_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += transient(t)
    return derivatives

def repetitive_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += repetetive(t)
    return derivatives

def prolonged_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += prolonged(t)
    return derivatives

def sharp_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += sharp_signal(t)
    return derivatives

#Block derivatives
def blocks1_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += blocks1(t)
    return derivatives

def blocks2_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += blocks2(t)
    return derivatives

def blocks3_derivatives(state, t):
    derivatives = mF_M_rates(state, t)
    derivatives[1] += blocks3(t)
    return derivatives