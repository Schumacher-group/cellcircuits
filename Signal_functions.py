import numpy as np
from parameters import *
from analysis import mF_M_rates


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


def sharp_signal(t, A_0, delta_t):
    return (A_0/delta_t)*(theta(t) - theta(t - delta_t))

def inter_signal(t):
    return one_signal(t, 8)

def slow_signal(t):
    return one_signal(t, 20)

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

def inter_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + inter_signal(t)
    return new_derivatives

def slow_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + slow_signal(t)
    return new_derivatives


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