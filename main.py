#import numpy as np is contained in parameters
from matplotlib.pyplot import show
from signal_functions import Signal
from analysis import nullcline_mF
import plotting as plotting
from parameters import *


def main():
    #time vectors 
    t_trajectory = np.linspace(0, 100, 10001)
    t_separatrix = np.linspace(0, 800, 1000)

    #Initial mF and M concentration
    x_initial = [1,1]

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

    #plotting.plot_streamlines(mFM_space, t_separatrix)

    amplitudes = [i*A_0 for i in np.arange(0.01, 10.01, 0.01)]
    plotting.amplitude_duration_dependence_for_hot_fibrosis(mFM_space, t_trajectory, t_separatrix, x_initial, amplitudes)


    normal_std = 0.2*A_0

    poisson_lam = A_0

    gamma_mean = 0.1*A_0
    gamma_std = 0.01*A_0

    #Gamma parameters in terms of mena and variance are (alpha, beta) = (mu**2/sigma**2, mu/sigma**2)
    transient_signal = Signal(name = 'Transient signal', start_points = [0], durations = [2], amplitudes = [10*A_0])
    prolonged_signal = Signal(name = 'Prolonged signal', start_points = [0], durations = [4], amplitudes = [A_0])
    repetitive_signal = Signal(name = 'Repetitive signal', start_points = [0,30], durations = [2,2], amplitudes = [A_0, 0.1*A_0])
    random_transient_signal1 = Signal(name = 'Random signal', start_points = [0], durations = [8], amplitudes = [0],
                                     normal_standard_deviations = [normal_std], poisson_lams = [poisson_lam],
                                     gamma_alphas = [gamma_mean**2/gamma_std**2], gamma_betas = [gamma_mean/gamma_std**2])
    

    gamma_mean = 0.1*A_0
    gamma_std = 0.03*A_0
    random_transient_signal2 = Signal(name = 'Random signal', start_points = [0], durations = [8], amplitudes = [0],
                                     normal_standard_deviations = [normal_std], poisson_lams = [poisson_lam],
                                     gamma_alphas = [gamma_mean**2/gamma_std**2], gamma_betas = [gamma_mean/gamma_std**2])
    
    gamma_mean = 0.1*A_0
    gamma_std = 0.05*A_0
    random_transient_signal3 = Signal(name = 'Random signal', start_points = [0], durations = [8], amplitudes = [0],
                                     normal_standard_deviations = [normal_std], poisson_lams = [poisson_lam],
                                     gamma_alphas = [gamma_mean**2/gamma_std**2], gamma_betas = [gamma_mean/gamma_std**2])

    #plotting.signals_and_trajectories(mFM_space, t_trajectory, t_separatrix, x_initial, signal = transient_signal)
    #plotting.signals_and_trajectories(mFM_space, t_trajectory, t_separatrix, x_initial, signal = prolonged_signal)
    #plotting.signals_and_trajectories(mFM_space, t_trajectory, t_separatrix, x_initial, signal = repetitive_signal)

    noise_type = 'gamma'

    num_sim = 10
    #plotting.plot_random_signal_trajectory_fibrosis_count(mFM_space, t_trajectory, t_separatrix, x_initial, signal = random_transient_signal1, num_sim = num_sim, noise_type=noise_type)
    #plotting.plot_random_signal_trajectory_fibrosis_count(mFM_space, t_trajectory, t_separatrix, x_initial, signal = random_transient_signal2, num_sim = num_sim, noise_type=noise_type)
    #plotting.plot_random_signal_trajectory_fibrosis_count(mFM_space, t_trajectory, t_separatrix, x_initial, signal = random_transient_signal3, num_sim = num_sim, noise_type=noise_type)

    gaussian_standard_deviations = [i*A_0 for i in np.arange(0, 2, 1)]
    
    poisson_lams = [i*A_0 for i in np.arange(0,2, 1)]
    
    gamma_means = [0.1*A_0]
    gamma_standard_deviations = [i*A_0 for i in np.arange(0.01, 0.1, 0.01)]

    '''
    plotting.plot_fibrosis_ratios(mFM_space, t_trajectory, t_separatrix, x_initial, start_point = 0, duration = 8, amplitude = 0, num_sim = num_sim, noise_type = noise_type,
                                standard_deviations = gaussian_standard_deviations, poisson_lams = poisson_lams, gamma_means = gamma_means,
                                  gamma_standard_deviations = gamma_standard_deviations)
    '''
    #Depict the plots
    show()


if __name__ == '__main__':
    main()
