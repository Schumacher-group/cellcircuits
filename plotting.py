import matplotlib.pyplot as plt
from parameters import *
from analysis import nullcline_mF, nullcline_M, unstable_fixed_point_hotfibrosis_mF_M, calculate_separatrix ,coldfibr2



def plot_nullclines_fixed_points_separatrix(mFM_space, mFnull1, mFnull2, mFnull3, xsmooth, ysmooth):
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

    coldfibrosis_mF_M = coldfibr2

    plt.annotate('unstable fixed point', unstable_fixed_point_mF_M)
    plt.annotate('hot fibrosis fixed point', hotfibrosis_mF_M)
    plt.annotate('cold fibrosis fixed point', coldfibr2)

    plt.plot(unstable_fixed_point_mF_M[0], unstable_fixed_point_mF_M[1], marker = 'o', color = 'black')
    plt.plot(hotfibrosis_mF_M[0], hotfibrosis_mF_M[1], marker = 'o', color = 'black')
    plt.plot(coldfibr2[0], coldfibr2[1], marker = 'o', color = "black")

    
    t_separatrix = np.linspace(0, 800, 1000)

    separatrix_left, separatrix_right = calculate_separatrix(unstable_fixed_point_mF_M, t_separatrix)
    plt.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black', label = 'Separatrix')
    plt.plot(separatrix_right[:, 0],separatrix_right[:, 1], 'black')

    plt.legend()
    plt.show()

