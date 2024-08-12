import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from analysis import *
from Signal_functions import *
from plotting import *
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

    plot_nullclines_fixed_points_separatrix(mFM_space, mFnull1, mFnull2, mFnull3, xsmooth, ysmooth, t_separatrix = t)

    plot_streamlines(mFM_space, t, t_separatrix)

    plt.show()


if __name__ == '__main__':
    main()
