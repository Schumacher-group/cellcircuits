#import numpy as np is contained in parameters
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from parameters import *
from analysis import (
    nullcline_mF, nullcline_M, unstable_fixed_point_hotfibrosis_mF_M, calculate_separatrix ,
    cold_fibr, mF_M_rates_array, time_taken_rd, array_statistics)
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


def plot_streamlines(mFM_space, t, t_separatrix):
    fig = plt.figure()
    mF_mesh = np.linspace(0, 7, 30)
    M_mesh = np.linspace(0, 7, 30)
    mF_stream, M_stream = np.meshgrid(mF_mesh, M_mesh)

    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    mF_rate, M_rate = mF_M_rates_array(mF_stream, M_stream, t)

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



def plot_signals_and_trajectories(mFM_space, t, t_separatrix, signal: Signal):
    signal_function = signal.signal_function
    signal_derivative = adjusted_derivatives_with_signal(signal_function)
    endpoint_of_signal = signal.endpoint_of_signal()
    
    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace= 0.5)

    # setting up injury plots
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    ax1.set_yticks([1],['A0'.translate(SUB)])

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


    x_initial = [1, 1] #mF, M
    
    x = odeint(signal_derivative, x_initial, t)

    
    t_signal = np.linspace(0, endpoint_of_signal + 1, 1000)
    ax1.plot(t_signal, signal_function(t_signal)/A_0, color = 'red')
    ax1.set_title(signal.name)

    ax2.plot(x[:,0], x[:,1], 'red')
    ax2.yaxis.set_label_position("right")
    ax2.set_title("time taken: " + str(time_taken_rd(x, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M)) + " days")



def plot_random_signal_trajectory_fibrosis_count(mFM_space, t_trajectory, t_separatrix, signal: Signal, num_sim):
    signal_function = signal.signal_function
    deterministic_derivative = adjusted_derivatives_with_signal(signal_function)
    noise_function = signal.noise_function
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
    ax1.plot(t_signal, signal_function(t_signal)/A_0, color = 'red', label = 'deterministic signal', linestyle = '--')
    ax1.plot(t_signal, signal_function(t_signal)/A_0 + noise_function(t_signal)/A_0, color = 'orange', label = 'noisy signal')
    ax1.set_title(f'{signal.name} (Reprasentation)')
    ax1.legend()

    '''
    Using Euler-Maruyama method to solve the stochastic differential equation
    '''
    x0 = [1,1] #initial mF and M point

    #end_points = simulate_euler_maruyama(deterministic_derivative, noise_function, t_trajectory, x0, num_sim = num_sim, axis = ax2)


    #Parallelized version for Euler Maruyama method
    def run_parallel_simulation(num):
        return single_euler_maruyama_simulation(deterministic_derivative, noise_function,
                                                t_trajectory, x0)
    
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


    healing_count = 0
    fibrosis_count = 0


    #left separatrix branch is build from right to left, so we need to revere the array
    separatrix_left_reverse = separatrix_left[::-1]
    times_to_fibrosis = []

    #make an interpolation to check if the end point of trajectory lies in the basin of healing or fibrosis point
    #The plus operator '+' concatenates usual Python arrays
    for trajectory, end_point in zip(trajectories, end_points):
        interpolation = np.interp(end_point[0], separatrix_left_reverse[:, 0] + separatrix_right[:, 0],
                                  separatrix_left_reverse[:, 1] + separatrix_right[:, 1])
        if end_point[1] < interpolation:
            healing_count += 1
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha = 0.1, color = 'green')
        else:
            fibrosis_count += 1
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha = 0.1, color ='red')
            times_to_fibrosis.append(time_taken_rd(trajectory, t_trajectory, hotfibrosis_mF_M, unstable_fixed_point_mF_M))
        

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
    
    if not fibrosis_count:
        print()
        print("Statistics for fibrosis times:")
        array_statistics(times_to_fibrosis, 'days')



def get_fibrosis_ratio(mFM_space, t_trajectory, t_separatrix, start_point, duration, amplitude, standard_deviation, num_sim):
    signal = Signal(start_points= [start_point], durations= [duration], amplitudes = [amplitude], standard_deviations= [standard_deviation])

    signal_function = signal.signal_function
    deterministic_derivative = adjusted_derivatives_with_signal(signal_function)
    noise_function = signal.noise_function
    
    unstable_fixed_point_mF_M, _ = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    x0 = [1, 1] #mF and M

    #Parallelized version for Euler Maruyama method
    def run_parallel_simulation(num):
        return single_euler_maruyama_simulation(deterministic_derivative, noise_function,
                                                t_trajectory, x0)
    
    #-1 means we use all cores on our device
    results = Parallel(n_jobs = -1)(delayed(run_parallel_simulation)(num) for num in range(num_sim))


    end_points = [result[0] for result in results]
    fibrosis_count = 0


    #left separatrix branch is build from right to left, so we need to revere the array
    separatrix_left_reverse = separatrix_left[::-1]

    #make an interpolation to check if the end point of trajectory lies in the basin of healing or fibrosis point
    #The plus operation '+' concatenates uual Python arrays
    for end_point in end_points:
        interpolation = np.interp(end_point[0], separatrix_left_reverse[:, 0] + separatrix_right[:, 0],
                                  separatrix_left_reverse[:, 1] + separatrix_right[:, 1])
        if end_point[1] > interpolation:
            fibrosis_count += 1

    return fibrosis_count/num_sim        

def plot_fibrosis_ratios(mFM_space, t_trajectory, t_separatrix, start_point, duration, amplitude, standard_deviations, num_sim):
    standard_deviations = np.array(standard_deviations)
    fibrosis_counts = np.array([])
    
    for standard_deviation in standard_deviations:
        fibrosis_counts = np.append(fibrosis_counts, 
                                    get_fibrosis_ratio(mFM_space, t_trajectory, t_separatrix,
                                                       start_point, duration, amplitude, standard_deviation, num_sim))

    _, ax = plt.subplots()

    ax.set_xlabel("std")
    ax.set_title(f'Fibrosis ratio for n = {num_sim}')
    ax.set_ylim([0,1])

    ax.plot(standard_deviations/A_0, fibrosis_counts)
    ax.scatter(standard_deviations/A_0, fibrosis_counts,color = 'red')