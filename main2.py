#import numpy as np is contained in parameters
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Signal_functions import Signal
from analysis import mF_M_rates, nullcline_mF
import plotting as plotting
from parameters import *


def main():
    #time vector 
    t = np.linspace(0, 80, 1000)
    t_separatrix = np.linspace(0, 800, 1000)

    #initial starting points
    x0 = [6*10**3, 7*10**3]

    x_ = odeint(mF_M_rates, x0, t)
    mF = x_[:,0]
    M = x_[:,1]

    #Nullcline parameters for plotting
    mFM_space = np.logspace(0, 7, 10**4)

    mFnull1 = np.logspace(0, 5.7, 10**3)
    mFnull2 = np.logspace(5.85, 5.95, 10**3)
    mFnull3 = np.logspace(6.05, 7, 10**3)


    # straight lines to replace/ignore the sharp increase near the poles
    xsmooth = [10**5.7, 10**5.85]
    ysmooth = [nullcline_mF(pt)[1] for pt in xsmooth]

    #plotting.plot_nullclines_fixed_points_separatrix(mFM_space, mFnull1, mFnull2, mFnull3, xsmooth, ysmooth, t_separatrix = t)

    #plotting.plot_streamlines(mFM_space, t, t_separatrix)

    #plotting.plot_signals_and_trajectories(mFM_space, signal=transient, signal_derivative=transient_derivatives, t=t, t_separatrix=t_separatrix)

    transient_signal = Signal(name = 'Transient signal', start_points = [0], durations = [2], amplitudes = [A_0])
    repetitive_signal = Signal(name = 'Repetitive signal', start_points = [0,4], durations = [2,2], amplitudes = [A_0, A_0])
    random_transient_signal = Signal(name = 'Random transient', start_points = [0], durations = [2], amplitudes = [A_0], standard_deviations = [A_0/2], dt = 0.1)

    
    plotting.plot_signals_and_trajectories2(mFM_space, t, t_separatrix, signal = transient_signal)

    plt.show()


if __name__ == '__main__':
    main()
