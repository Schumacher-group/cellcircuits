#import numpy as np is contained in parameters
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Signal_functions import Signal
from analysis import mF_M_rates, nullcline_mF
import plotting as plotting
from parameters import *


def main():
    #time vector 
    t = np.linspace(0, 100, 1000)
    t_separatrix = np.linspace(0, 800, 1000)


    #Nullcline parameters for plotting
    mFM_space = np.logspace(0, 7, 10**4)

    #Between those intervalls the nullcline for mF has poles, so we smooth those out  
    mFnull1 = np.logspace(0, 5.7, 10**3)
    mFnull2 = np.logspace(5.85, 5.95, 10**3)
    mFnull3 = np.logspace(6.05, 7, 10**3)


    # straight lines to replace/ignore the sharp increase near the poles
    xsmooth = [10**5.7, 10**5.85]
    ysmooth = [nullcline_mF(pt)[1] for pt in xsmooth]

    #plotting.plot_nullclines_fixed_points_separatrix(mFM_space, mFnull1, mFnull2, mFnull3, xsmooth, ysmooth, t_separatrix = t_separatrix)

    #plotting.plot_streamlines(mFM_space, t, t_separatrix)


    transient_signal = Signal(name = 'Transient signal', start_points = [0], durations = [2], amplitudes = [1.2*A_0])
    repetitive_signal = Signal(name = 'Repetitive signal', start_points = [0,4], durations = [2,2], amplitudes = [A_0, A_0])
    random_transient_signal = Signal(name = 'Random transient', start_points = [0], durations = [2], amplitudes = [A_0], standard_deviations = [A_0])
    combined_signal = transient_signal + repetitive_signal
    
    #Use different plot functions for determinstic and random signals as one implements the stochastic euler method
    #plotting.plot_signals_and_trajectories(mFM_space, t, t_separatrix, signal = transient_signal)

    num_sim = 500
    #plotting.plot_random_signal_trajectory_fibrosis_count(mFM_space, t, t_separatrix, signal = random_transient_signal, num_sim = num_sim)

    #print(plotting.get_fibrosis_ratio(mFM_space, t, t_separatrix, start_point = 0, duration = 2, amplitude = A_0, standard_deviation = A_0, num_sim = num_sim))

    standard_deviations = [i*A_0 for i in np.arange(0, 5, 0.5)]

    plotting.plot_fibrosis_ratios(mFM_space, t, t_separatrix, start_point = 0, duration = 2, amplitude = A_0,
                                  standard_deviations = standard_deviations, num_sim = num_sim,)
    plt.show()


if __name__ == '__main__':
    main()
