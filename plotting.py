#import numpy as np is contained in parameters
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from parameters import *
from analysis import (
    nullcline_mF, nullcline_M, unstable_fixed_point_hotfibrosis_mF_M, calculate_separatrix ,
    cold_fibr, mF_M_rates_array, time_taken_rd)
from Signal_functions import Signal, adjusted_derivatives_with_signal
from euler_maruyama_method import simulate_euler_maruyama


def plot_nullclines_fixed_points_separatrix(mFM_space, mFnull1, mFnull2, mFnull3, xsmooth, ysmooth, t_separatrix):
    plt.figure()
    plt.plot(nullcline_M(mFM_space)[0], nullcline_M(mFM_space)[1], 'r', label = 'Macrophage nullcline')

    plt.plot(nullcline_mF(mFnull1)[0], nullcline_mF(mFnull1)[1], 'b', label = 'Myofibroblast nullcline')
    plt.plot(nullcline_mF(mFnull2)[0], nullcline_mF(mFnull2)[1], 'b')
    plt.plot(nullcline_mF(mFnull3)[0], nullcline_mF(mFnull3)[1], 'b')
    plt.plot(xsmooth, ysmooth, 'b')

    #print(nullcline_mF(10**5.7), nullcline_mF(10**5.85), nullcline_mF(10**5.95), nullcline_mF(10**6.05))

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


#Plot the trajectory for one signal function
def plot_signals_and_trajectories(mFM_space, signal_function, signal_derivative, t, t_separatrix):
    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace= 0.5)

    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(unstable_fixed_point_mF_M[0], unstable_fixed_point_mF_M[1], marker = 'o', color = 'black')
    ax2.plot(hotfibrosis_mF_M[0], hotfibrosis_mF_M[1], marker = 'o', color = 'black')
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')
    ax2.plot(coldfibrosis_mF_M[0], coldfibrosis_mF_M[1], marker='o', color="black")

    # setting up injury plots
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')

    x_initial = [1, 1] #mF, M
    
    x = odeint(signal_derivative, x_initial, t)

    #ax1 plot need to be adjusted
    ax1.plot()

    ax2.plot(x[:,0], x[:,1], 'red')
    ax2.set_title("time taken: " + str(time_taken_rd(x, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M)) + " days")


def plot_signals_and_trajectories2(mFM_space, t, t_separatrix, signal: Signal):
    signal_function = signal.signal_function
    signal_derivative = adjusted_derivatives_with_signal(signal_function)
    endpoint_of_signal = signal.endpoint_of_signal()
    
    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace= 0.5)

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

    # setting up injury plots
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    #ax1.set_xticks(np.concatenate(signal.start_points, signal.start_points + signal.durations))
    ax1.set_yticks([1],['A0'.translate(SUB)])

    x_initial = [1, 1] #mF, M
    
    x = odeint(signal_derivative, x_initial, t)

    
    t_signal = np.linspace(0, endpoint_of_signal + 1, 1000)
    ax1.plot(t_signal, signal_function(t_signal)/A_0, color = 'red')
    ax1.set_title(signal.name)

    ax2.plot(x[:,0], x[:,1], 'red')
    ax2.yaxis.set_label_position("right")
    ax2.set_title("time taken: " + str(time_taken_rd(x, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M)) + " days")

def plot_random_signal_and_trajectory(mFM_space, t_trajectory, t_separatrix, signal: Signal):
    signal_function = signal.signal_function
    deterministic_derivative = adjusted_derivatives_with_signal(signal_function)
    noise_function = signal.noise_function
    endpoint_of_signal = signal.endpoint_of_signal()
    
    coldfibrosis_mF_M = [cold_fibr()[0], 1]
    unstable_fixed_point_mF_M, hotfibrosis_mF_M = unstable_fixed_point_hotfibrosis_mF_M(mFM_space)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace= 0.5)

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

    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    #ax1.set_xticks(np.concatenate(signal.start_points, signal.start_points + signal.durations))
    ax1.set_yticks([1],['A0'.translate(SUB)])

    #Using Euler Maruyama method
    x0 = [1,1]
    num_steps = 100
    end_points = simulate_euler_maruyama(deterministic_derivative, noise_function, t_trajectory, x0, num_steps = num_steps, axis = ax2)

    '''
    t0 = t_trajectory[0]
    dt = (t_trajectory[-1] - t0)/(t_trajectory.size)
    x_initial = [1, 1] #mF, M
    x = np.zeros((t_trajectory.size, 2))
    x[0] = x_initial

    for k in range(1,t_trajectory.size):
        t_step = t0 + k * dt
        x[k] = x[k-1] + dt * np.array(deterministic_derivative(x[k-1], t_step)) + np.sqrt(dt) * np.array([0, noise(t_step)])
    '''

    
    t_signal = np.linspace(0, endpoint_of_signal + 1, 1000)
    ax1.plot(t_signal, signal_function(t_signal)/A_0, color = 'red')
    ax1.set_title(signal.name)


    #ax2.plot(x[:,0], x[:,1], 'red')
    ax2.yaxis.set_label_position("right")
    #ax2.set_title("time taken: " + str(time_taken_rd(x, t, hotfibrosis_mF_M, unstable_fixed_point_mF_M)) + " days")




    healing_count = 0
    fibrosis_count = 0


    #left separatrix branch is build from right to left, so we need to revere the array
    separatrix_left_reverse = separatrix_left[::-1]

    #make an interpolation to check if the end point of trajectory lies in the basin of healing or fibrosis point (left separatrix already sufficient)
    for point in end_points:
        interpolation = np.interp(point[0], separatrix_left_reverse[:,0] + separatrix_right[:,0],
                                  separatrix_left_reverse[:,1] + separatrix_right[:,1])
        if point[1] < interpolation:
            healing_count += 1
            print('End point', point)
            print('Interpolation point', interpolation)
        else:
            fibrosis_count += 1

    print('Healing count', healing_count)
    print('Fibrosis count', fibrosis_count)
    
    plt.figure()

    labels = ['Healing count', 'Fibrosis count']
    values = [healing_count, fibrosis_count]

    plt.bar(labels, values, color = ['green', 'red'])

    plt.title(f'Fibrosis ratio {fibrosis_count/num_steps} and Healing ratio {healing_count/num_steps}')
