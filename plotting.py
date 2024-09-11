#import numpy as np is contained in parameters
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from parameters import *
from analysis import (
    nullcline_mF, nullcline_M, unstable_fixed_point_hotfibrosis_mF_M, calculate_separatrix ,
    cold_fibr, mF_M_rates_array, check_hot_fibrosis, find_first_crossing_index, time_taken_rd, array_statistics)
from Signal_functions import Signal, adjusted_derivatives_with_signal
from euler_maruyama_method import simulate_euler_maruyama, single_euler_maruyama_simulation

from joblib import Parallel, delayed


def plot_nullclines_fixed_points_separatrix(mFM_space, mFnull1, mFnull2, mFnull3, xsmooth, ysmooth, t_separatrix):
    plt.figure()
    plt.plot(nullcline_M(mFM_space)[0], nullcline_M(mFM_space)[1], 'r', label = 'Macrophage nullcline')

    plt.plot(nullcline_mF(mFnull1)[0], nullcline_mF(mFnull1)[1], 'b', label = 'Myofibroblast nullcline')
    plt.plot(nullcline_mF(mFnull2)[0], nullcline_mF(mFnull2)[1], 'b')
    plt.plot(nullcline_mF(mFnull3)[0], nullcline_mF(mFnull3)[1], 'b')
    plt.plot(xsmooth, ysmooth, 'b')


    plt.xlabel('Myofibroblasts')
    plt.ylabel('Macrophages')
    plt.xlim(1, 10**7)
    plt.ylim(1, 10**7)
    plt.xscale('log')
    plt.yscale('log')

    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    fixed_point_end_of_separatrix = [cold_fibr()[1], 1]

    plt.annotate('unstable fixed point', unstable_fixed_point_mF_M)
    plt.annotate('hot fibrosis fixed point', hotfibrosis_mF_M)
    plt.annotate('cold fibrosis fixed point', coldfibrosis_mF_M)

    plt.plot(unstable_fixed_point_mF_M[0], unstable_fixed_point_mF_M[1], marker = 'o', color = 'black')
    plt.plot(hotfibrosis_mF_M[0], hotfibrosis_mF_M[1], marker = 'o', color = 'black')
    plt.plot(coldfibrosis_mF_M[0], coldfibrosis_mF_M[1], marker = 'o', color = "black")
    plt.plot(fixed_point_end_of_separatrix[0], fixed_point_end_of_separatrix[1], marker = 'o', color = 'black')


    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)
    plt.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black', label = 'Separatrix')
    plt.plot(separatrix_right[:, 0],separatrix_right[:, 1], 'black')

    plt.legend()


def plot_streamlines(mFM_space, t_separatrix):
    fig = plt.figure()
    mF_mesh = np.linspace(0, 7, 30)
    M_mesh = np.linspace(0, 7, 30)
    mF_stream, M_stream = np.meshgrid(mF_mesh, M_mesh)

    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    mF_rate, M_rate = mF_M_rates_array(mF_stream, M_stream)

    #scale the rates to appropriate size
    mF_rate_scaled = mF_rate/(10**mF_stream)
    M_rate_scaled = M_rate/(10**M_stream)

    strm = ax.streamplot(mF_stream, M_stream, mF_rate_scaled, M_rate_scaled,
                     color = (np.sqrt((mF_rate_scaled)**2 + (M_rate_scaled)**2)) , cmap = 'autumn')

    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    
    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)
    

    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    fixed_point_end_of_separatrix = [cold_fibr()[1], 1]

    ax.set_xlim(0, 7)
    ax.set_ylim(0,7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')


    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(unstable_fixed_point_mF_M[0], unstable_fixed_point_mF_M[1], marker = 'o', color = 'black')
    ax2.plot(hotfibrosis_mF_M[0], hotfibrosis_mF_M[1], marker = 'o', color = 'black')
    ax2.plot(coldfibrosis_mF_M[0], coldfibrosis_mF_M[1], marker = 'o', color = "black")
    ax2.plot(fixed_point_end_of_separatrix[0], fixed_point_end_of_separatrix[1], marker = 'o', color = 'black')


#if create_plots is set to False the function only returns the fibrosis status
def signals_and_trajectories(mFM_space, t_trajectory, t_separatrix, x_initial, signal: Signal, create_plots = True):
    signal_function = signal.signal_function
    signal_derivative = adjusted_derivatives_with_signal(signal_function)
    endpoint_of_signal = signal.endpoint_of_signal()
    
    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)
    
    x = odeint(signal_derivative, x_initial, t_trajectory)

    if not create_plots:
        end_point = x[-1]
        separatrix_left_reverse = separatrix_left[::-1]
        if check_hot_fibrosis(end_point, separatrix_left_reverse, separatrix_right):
            return True
        else:
            False


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace= 0.5)

    # setting up injury plots
    ax1.set_xlim(0,endpoint_of_signal + 1)
    ax1.set_ylim(-2,5)
    #ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    y_ticks = np.arange(-2, 5)
    y_tick_labels = [f'{i}*$A_0$'if i != 0 else '0' for i in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)

    '''
    For visualization of the behaviour on the x/y_axis one can change the 'log' to symlog for the x and yscale, add linthresh
    and change the lower x/y_lim to 0
    '''
    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(unstable_fixed_point_mF_M[0], unstable_fixed_point_mF_M[1], marker = 'o', color = 'black')
    ax2.plot(hotfibrosis_mF_M[0], hotfibrosis_mF_M[1], marker = 'o', color = 'black')
    ax2.annotate('hot fibrosis', hotfibrosis_mF_M)
    ax2.plot(coldfibrosis_mF_M[0], coldfibrosis_mF_M[1], marker='o', color="black")
    ax2.annotate('cold fibrosis', coldfibrosis_mF_M)
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')


    
    t_signal = np.linspace(0, endpoint_of_signal + 1, 1000)
    ax1.plot(t_signal, signal_function(t_signal)/A_0, color = 'red')
    ax1.set_title(signal.name)

    ax2.plot(x[:,0], x[:,1], 'red')
    ax2.yaxis.set_label_position("right")
    ax2.set_title("time taken: " + str(time_taken_rd(x, t_trajectory, hotfibrosis_mF_M, unstable_fixed_point_mF_M)) + " days")


def amplitude_duration_dependence_for_hot_fibrosis(mFM_space, t_trajectory, t_separatrix, x_initial, amplitudes):
    plt.figure()
    crossing_times = np.array([])
    
    unstable_fixed_point_mF_M, _ = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)
    separatrix_left_reverse = separatrix_left[::-1]

    separatrix = np.concatenate( (separatrix_left_reverse, separatrix_right) )

    #CubicSpline needs strictly increasing x-values, so we remove potential duplicates
    separatrix_x_unique, indices = np.unique(separatrix[:, 0], return_index = True)
    separatrix_y_unique = separatrix[indices, 1]

    separatrix_interp = CubicSpline(separatrix_x_unique, separatrix_y_unique, extrapolate = True)

    t_end = t_trajectory[-1]

    for amplitude in amplitudes:
        signal = Signal(durations = [t_end], amplitudes = [amplitude])
    
        signal_function = signal.signal_function
        signal_derivative = adjusted_derivatives_with_signal(signal_function)
        
        x = odeint(signal_derivative, x_initial, t_trajectory)

        #find crossing index and append the corresponding time value
        first_crossing_index = find_first_crossing_index(x, separatrix_interp)
        first_crossing_time = t_trajectory[first_crossing_index]
        crossing_times = np.append(crossing_times, first_crossing_time)

    plt.xlabel("Amplitudes (cell/day)")
    plt.ylabel("Time (days)")
    plt.scatter(amplitudes, crossing_times, color = 'blue')
    plt.title("Time to reach fibrosis basin under constant injury signal")

    plt.xticks(np.arange(0, 2*A_0, 0.1*A_0))
    plt.yticks(np.arange(0, 9, 0.5))
    plt.grid(True)
    plt.plot(amplitudes, crossing_times, color = 'red')

    plt.figure()
    plt.axis('off')

    amplitudes_scaled = [amp/A_0 for amp in amplitudes]
    data = np.vstack([amplitudes_scaled, crossing_times]).T  # Stack x and y as two columns
    table_data = [[f"{xi:.2f}", f"{yi:.2f}"] for xi, yi in data]  # Format values

    # Add a table
    table = plt.table(cellText=table_data,
                    colLabels=["Amplitudes in 10^6 cell/day", "Time in days"],
                    cellLoc="center",
                    loc="center",  # Position table in the center
                    )  

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1) #width and height of table

def plot_random_signal_trajectory_fibrosis_count(mFM_space, t_trajectory, t_separatrix, x_initial, signal: Signal, num_sim, noise_type = 'gaussian'):
    signal_function = signal.signal_function
    deterministic_derivative = adjusted_derivatives_with_signal(signal_function)
    endpoint_of_signal = signal.endpoint_of_signal()


    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace= 0.5)

    '''
    Setting up the plot for the signal function
    '''
    ax1.set_xlim(0,endpoint_of_signal + 1)
    ax1.set_ylim(-2,10)
    #ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    y_ticks = np.arange(-2, 5)
    y_tick_labels = [f'{i}*$A_0$'if i != 0 else '0' for i in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)

    t_signal = np.linspace(0, endpoint_of_signal + 1, 100)
    ax1.plot(t_signal, signal_function(t_signal)/A_0, color = 'red', label = 'Deterministic signal', linestyle = '--')
    

    if noise_type == 'poisson':
        #for plotting an example for the incoming noise
        signal_representation = signal.poisson_noise_function(t_signal, 1)
        noise_function = signal.poisson_noise_function
        ax1.plot(t_signal, signal_function(t_signal)/A_0 + signal_representation/A_0, color = 'orange', label = 'Noise')
        ax1.set_title(f'{noise_type} noise (representation) \nlambda = {signal.poisson_lams}')
    elif noise_type == 'gamma':
        #for plotting an example of the incoming noise
        signal_representation = signal.gamma_noise_function(t_signal, 1)
        noise_function = signal.gamma_noise_function
        ax1.plot(t_signal, signal_function(t_signal)/A_0 + signal_representation/A_0, color = 'orange', label = 'Noise')
        ax1.set_title(f'{noise_type.title()} noise (representation) \nmean = {np.round(signal.gamma_alphas/signal.gamma_betas, 3)/A_0}E+6, \n'
                      f'std = {np.round(np.sqrt(signal.gamma_alphas)/signal.gamma_betas,2)/A_0}E+6')
    else:
        #for plotting an example for the incoming noise
        signal_representation = signal.gaussian_noise_function(t_signal, 1)
        noise_function = signal.gaussian_noise_function
        ax1.plot(t_signal, signal_function(t_signal)/A_0 + signal_representation/A_0, color = 'orange', label = 'Noise')
        ax1.set_title(f'{noise_type} noise (reprasentation) \nstd = {signal.standard_deviations}')

    ax1.legend()

    '''
    Using Euler-Maruyama method to solve the stochastic differential equation
    '''
    #non parallelized version of the Euler-Maruyama method
    #end_points = simulate_euler_maruyama(deterministic_derivative, noise_function, t_trajectory, x0, num_sim = num_sim, axis = ax2)


    #Parallelized version for Euler Maruyama method
    def run_parallel_simulation(num):
        return single_euler_maruyama_simulation(deterministic_derivative, noise_function,
                                                t_trajectory, x_initial)
    
    #-1 means we use all cores on our device
    results = Parallel(n_jobs = -1)(delayed(run_parallel_simulation)(num) for num in range(num_sim))


    end_points = [result[0] for result in results]
    trajectories = [result[1] for result in results]

    '''
    Setting up the plot for inflammation trajectories
    '''
    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(unstable_fixed_point_mF_M[0], unstable_fixed_point_mF_M[1], marker = 'o', color = 'black')
    ax2.plot(hotfibrosis_mF_M[0], hotfibrosis_mF_M[1], marker = 'o', color = 'black')
    ax2.annotate('hot fibrosis', hotfibrosis_mF_M)
    ax2.plot(coldfibrosis_mF_M[0], coldfibrosis_mF_M[1], marker='o', color="black")
    ax2.annotate('cold fibrosis', coldfibrosis_mF_M)
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')
    ax2.yaxis.set_label_position("right")

    #deterministic trajectory
    if noise_type == 'gaussian':
        x = odeint(deterministic_derivative, x_initial, t_trajectory)
        ax2.plot(x[:,0], x[:,1], color = 'purple', label = 'Deterministic trajectory', linewidth = 1.5)


    healing_count = 0
    fibrosis_count = 0


    #left separatrix branch is build from right to left, so we need to revere the array
    separatrix_left_reverse = separatrix_left[::-1]
    times_to_fibrosis = []

    #make an interpolation to check if the end point of trajectory lies in the basin of healing or fibrosis point
    #The plus operator '+' concatenates usual Python arrays
    for trajectory, end_point in zip(trajectories, end_points):
        if check_hot_fibrosis(end_point, separatrix_left_reverse, separatrix_right):
            fibrosis_count += 1
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha = 0.1, color ='red')
            times_to_fibrosis.append(time_taken_rd(trajectory, t_trajectory, hotfibrosis_mF_M, unstable_fixed_point_mF_M))
        else:
            healing_count += 1
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha = 0.1, color = 'green')
    
    ax2.legend(loc = 'lower right')    

    print('Healing count', healing_count)
    print('Fibrosis count', fibrosis_count)

    
    plt.figure()

    
    labels = ['Healing count', 'Fibrosis count']
    values = [healing_count, fibrosis_count]

    plt.subplot(1, 2, 1)
    plt.bar(labels, values, color = ['green', 'red'])

    plt.title(f'Fibrosis ratio {fibrosis_count/num_sim} (n = {num_sim})')

    plt.subplot(1, 2, 2)
    
    #empty sequences like [] return false
    if not times_to_fibrosis:
        plt.title(f'No trajectories ended in fibrosis')
    else:
        sns.violinplot(x= times_to_fibrosis) 
        plt.title(f'Time to fibrosis (n={fibrosis_count})')
        plt.xlabel('Time (day)')

        print()
        print("Statistics for fibrosis times:")
        array_statistics(times_to_fibrosis, 'days')
    



def get_fibrosis_ratio(mFM_space, t_trajectory, t_separatrix, x_initial, start_point, duration, num_sim, noise_type, amplitude = 0, standard_deviation = 0,
                       lam = 0, alpha = 0, beta = 0):
    signal = Signal(start_points= [start_point], durations= [duration], amplitudes = [amplitude], normal_standard_deviations= [standard_deviation],
                    poisson_lams = [lam], gamma_alphas = [alpha], gamma_betas = [beta])

    signal_function = signal.signal_function
    deterministic_derivative = adjusted_derivatives_with_signal(signal_function)
    
    if noise_type == 'poisson':
        noise_function = signal.poisson_noise_function
    elif noise_type == 'gamma':
        noise_function = signal.gamma_noise_function
    else:
        noise_function = signal.gaussian_noise_function
    
    unstable_fixed_point_mF_M, _ = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    #Parallelized version for Euler Maruyama method
    def run_parallel_simulation(num):
        return single_euler_maruyama_simulation(deterministic_derivative, noise_function,
                                                t_trajectory, x_initial)
    
    #-1 means we use all cores on our device
    results = Parallel(n_jobs = -1)(delayed(run_parallel_simulation)(num) for num in range(num_sim))


    end_points = [result[0] for result in results]
    fibrosis_count = 0


    #left separatrix branch is build here from right to left (from unstable fixed point), so we need to reverse the array
    separatrix_left_reverse = separatrix_left[::-1]

    for end_point in end_points:
        if check_hot_fibrosis(end_point, separatrix_left_reverse, separatrix_right):
            fibrosis_count += 1

    return fibrosis_count/num_sim        

def plot_fibrosis_ratios(mFM_space, t_trajectory, t_separatrix, x_initial, start_point, duration, amplitude, num_sim, noise_type, standard_deviations = [0],
                         poisson_lams = [0], gamma_means = [1], gamma_standard_deviations = [1]):
    
    fibrosis_counts = np.array([])
    _, ax = plt.subplots()
    ax.set_title(f'Fibrosis ratio (n = {num_sim}) for \nsignal length = {duration} days')
    ax.set_ylim([0,1])

    if noise_type == 'poisson':
        poisson_lams = np.array(poisson_lams)
        for lam in poisson_lams:
            fibrosis_counts = np.append(fibrosis_counts, 
                                        get_fibrosis_ratio(mFM_space, t_trajectory, t_separatrix, x_initial,
                                                           start_point, duration, num_sim, noise_type, amplitude, lam = lam))
        ax.set_xlabel('lambda in $A_0')
        ax.plot(poisson_lams/A_0, fibrosis_counts)
        ax.scatter(poisson_lams/A_0, fibrosis_counts, color = 'red')
    elif noise_type == 'gamma':
        gamma_means_scaled = [mean/A_0 for mean in gamma_means]
        gamma_standard_deviations_scaled = [std/A_0 for std in gamma_standard_deviations]

        fibrosis_count_grid = np.zeros((len(gamma_means), len(gamma_standard_deviations))) 
        for i, mean in enumerate(gamma_means):
            fibrosis_counts = np.array([])
            for j, std in enumerate(gamma_standard_deviations):
                #transform means, variances to parameters for the gamma distribution which has mean = alpha/beta and sigma**2 = alpha/beta**2
                alpha = mean**2/std**2
                beta = mean/std**2

                fibrosis_count = get_fibrosis_ratio(mFM_space, t_trajectory, t_separatrix, x_initial,
                                                    start_point, duration, num_sim, noise_type, amplitude, alpha = alpha, beta = beta)
                fibrosis_counts = np.append(fibrosis_counts, fibrosis_count)
                fibrosis_count_grid[i, j] = fibrosis_count
            ax.set_xlabel('std in $A_0$')
            ax.plot(gamma_standard_deviations_scaled, fibrosis_counts, label = f'mean = {mean/A_0}')
            ax.scatter(gamma_standard_deviations_scaled, fibrosis_counts, color = 'red')
            ax.legend(loc = 'upper right')

        
        _, ax2 = plt.subplots()
        sns.heatmap(fibrosis_count_grid, xticklabels = np.round(gamma_standard_deviations_scaled, 2), yticklabels = np.round(gamma_means_scaled, 2),
                    annot = True, cmap = 'Reds', ax = ax2)
        ax2.set_title(f'Fibrosis ratio (n = {num_sim}) for \nsignal length = {duration} days')
        ax2.set_xlabel('std in $A_0$')
        ax2.set_ylabel('mean in $A_0$')
        ax2.set_aspect('equal')

    else:
        standard_deviations = np.array(standard_deviations)
        for standard_deviation in standard_deviations:
            fibrosis_counts = np.append(fibrosis_counts, 
                                        get_fibrosis_ratio(mFM_space, t_trajectory, t_separatrix, x_initial,
                                                        start_point, duration, num_sim, noise_type, amplitude, standard_deviation = standard_deviation))
        ax.set_xlabel('std in $A_0')
        ax.plot(standard_deviations/A_0, fibrosis_counts)
        ax.scatter(standard_deviations/A_0, fibrosis_counts,color = 'red')