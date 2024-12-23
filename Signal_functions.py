#import numpy as np is contained in parameters
from parameters import *
from analysis import mF_M_rates

'''
Parameter units:
Time: day
Amplitude: cells/day
'''
class Signal:
    def __init__(self, name = 'Input Signal', start_points = [0], durations = [1], amplitudes = [1],
                normal_standard_deviations = [0], poisson_lams = [0], gamma_alphas = [0], gamma_betas = [0]):
        # convert all arrays to numpy arrays
        self.name = name
        self.start_points = np.array(start_points)
        self.durations = np.array(durations)
        self.amplitudes = np.array(amplitudes)

        #for default value create a zero std array to enable proper addition of signal
        if isinstance(normal_standard_deviations, list) and np.array_equal(normal_standard_deviations, [0]):
            self.standard_deviations = np.zeros_like(amplitudes)
        else:
            self.standard_deviations = np.array(normal_standard_deviations)

        if isinstance(poisson_lams, list) and np.array_equal(poisson_lams, [0]):
            self.poisson_lams = np.zeros_like(amplitudes)
        else:
            self.poisson_lams = np.array(poisson_lams)

        if isinstance(gamma_alphas, list) and np.array_equal(gamma_alphas, [0]):
            self.gamma_alphas = np.zeros_like(amplitudes)
        else:
            self.gamma_alphas = np.array(gamma_alphas)

        if isinstance(gamma_betas, list) and np.array_equal(gamma_betas, [0]):
            self.gamma_betas = np.zeros_like(amplitudes)
        else:
            self.gamma_betas = np.array(gamma_betas)

    def __repr__(self):
        return f'{self.name}'
    
    def __add__(self, other):
        if isinstance(other, Signal):
            return Signal(
                name = f'{self.name} + {other.name}',
                start_points = np.concatenate((self.start_points, other.start_points)),
                durations = np.concatenate((self.durations, other.durations)),
                amplitudes = np.concatenate((self.amplitudes, other.amplitudes)),
                normal_standard_deviations = np.concatenate((self.standard_deviations, other.standard_deviations)),
                poisson_lams = np.concatenate((self.poisson_lams, other.poisson_lams))
                )
        else:
            return NotImplemented
    
    def endpoint_of_signal(self):
        return np.max(self.start_points + self.durations)
    
    def theta(self, t):
        return np.heaviside(t, 1)

    def basic_signal(self, start, duration, amplitude, t):
        is_scalar = np.isscalar(t)
        t = np.array(t) #ensure t is a numpy array for computational purposes
        signal = amplitude * (self.theta(t - start) - self.theta(t - (start + duration)))
        
        #Make sure that for a scalar input we have a scalar output
        #This is necessary to also have scalar output for signal_function for scalar input
        #as this is needed for the 'odeint' function from scipy
        if is_scalar:
            return signal.item()
        return signal
    
    #signal made up as a combination of the basic_signal without any noise
    def signal_function(self,t):
        total_signal = np.zeros_like(t)
        for start, duration, amplitude in zip(self.start_points, self.durations, self.amplitudes):
            total_signal += self.basic_signal(start, duration, amplitude, t)
        return total_signal

    #Gaussian noise 
    def gaussian_signal(self, start, duration, std, t, dt):
        is_scalar = np.isscalar(t)
        t = np.array(t)
        noise = (std * np.sqrt(dt) * np.random.normal(0, 1, t.size)) * (self.theta(t - start) - self.theta(t - (start + duration)))

        if is_scalar:
            return noise.item()
        return noise
    
    #Overlay the noise functions
    def gaussian_noise_function(self, t, dt):
        total_noise = np.zeros_like(t)
        for start, duration, std in zip(self.start_points, self.durations, self.standard_deviations):
            total_noise += self.gaussian_signal(start, duration, std, t, dt)
        return total_noise
    
    #Poisson noise
    def poisson_signal(self, start, duration, lam, t, dt):
        is_scalar = np.isscalar(t)
        t = np.array(t)
        noise = np.random.poisson(dt*lam, t.size) * (self.theta(t - start) - self.theta(t - (start + duration)))
        
        if is_scalar:
            return noise.item()
        return noise
    
    def poisson_noise_function(self, t, dt):
        total_noise = np.zeros_like(t)
        for start, duration, lam in zip(self.start_points, self.durations, self.poisson_lams):
            total_noise += self.poisson_signal(start, duration, lam, t, dt)
        return total_noise
    
    #Gamma noise
    def gamma_signal(self, start, duration, alpha, beta, t, dt):
        is_scalar = np.isscalar(t)
        t = np.array(t)
        #beta is intended to be rate parameter but np implementation takes scale parameter theta = 1/beta
        noise = np.random.gamma(dt * alpha, 1/beta, t.size) * (self.theta(t - start) - self.theta(t - (start + duration)))
        
        if is_scalar:
            return noise.item()
        return noise
    
    def gamma_noise_function(self, t, dt):
        total_noise = np.zeros_like(t)
        for start, duration, alpha, beta in zip(self.start_points, self.durations, self.gamma_alphas, self.gamma_betas):
            total_noise += self.gamma_signal(start, duration, alpha, beta, t, dt)
        return total_noise




#Generalized signal/block RHS of differential equation, derivative of M, added signal increaces macrophage rate
def adjusted_derivatives_with_signal(signal_function):
    def derivative_function(state, t):
        derivatives = mF_M_rates(state, t)
        derivatives[1] += signal_function(t) #Update the rate for macrophages
        return derivatives
    return derivative_function