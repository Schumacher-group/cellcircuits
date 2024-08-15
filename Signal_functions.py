#import numpy as np is contained in parameters
from parameters import *
from analysis import mF_M_rates


class Signal:
    def __init__(self, name: str, start_points = [0], durations = [1], amplitudes = [1], standard_deviations = [0]):
        # convert all arrays to numpy arrays
        self.name = name
        self.start_points = np.array(start_points)
        self.durations = np.array(durations)
        self.amplitudes = np.array(amplitudes)
        if isinstance(standard_deviations, list) and standard_deviations == [0]:
            self.standard_deviations = np.zeros_like(amplitudes)
        else:
            self.standard_deviations = np.array(standard_deviations)

    def __repr__(self):
        return f'{self.name}'
    
    def __add__(self, other):
        if isinstance(other, Signal):
            return Signal(
                name = f'{self.name} + {other.name}',
                start_points = np.concatenate((self.start_points, other.start_points)),
                durations = np.concatenate((self.durations, other.durations)),
                amplitudes = np.concatenate((self.amplitudes, other.amplitudes)),
                standard_deviations = np.concatenate((self.standard_deviations, other.standard_deviations))
                )
        else:
            return NotImplemented
    
    def endpoint_of_signal(self):
        return np.max(self.start_points + self.durations)
    
    def theta(self, t):
        return np.heaviside(t, 1)

    def basic_signal(self, start, duration, amplitude, t, std = 0, dt = 0):
        is_scalar = np.isscalar(t)
        t = np.array(t) #ensure t is a numpy array for computational purposes
        stochastic_term = std * np.random.normal(0, np.sqrt(dt), t.size)
        signal = (amplitude + stochastic_term) * (self.theta(t - start) - self.theta(t - (start + duration)))
        
        #Make sure that for a scalar input we have a scalar output
        #This is necessary to also have scalar output for signal_function for scalar input
        #as this is needed for the 'odeint' function from scipy
        if is_scalar:
            return signal.item()
        return signal
    
    #final signal made up as a combination of the basic_signal
    def signal_function(self,t, dt = 0):
        total_signal = np.zeros_like(t)
        for start, duration, amplitude, std in zip(self.start_points, self.durations, self.amplitudes, self.standard_deviations):
            total_signal += self.basic_signal(start, duration, amplitude, t, std, dt)
        return total_signal



#Generalized signal/block RHS of differential equation, derivative of M, added signal increaces macrophage rate
def adjusted_derivatives_with_signal(signal_function):
    def derivative_function(state, t):
        derivatives = mF_M_rates(state, t)
        derivatives[1] += signal_function(t) #Update the rate for macrophages
        return derivatives
    return derivative_function

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