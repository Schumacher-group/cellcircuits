import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
lambda1 = 0.9  # proliferation rate of mF
lambda2 = 0.8  # proliferation rate of M
mu1 = 0.3  # removal rate of mF
mu2 = 0.3  # removal rate of M
K = 10 ** 6  # carrying capacity of mF
k1 = 10 ** 9  # binding affinity of CSF
k2 = 10 ** 9  # binding affinity of PDGF
# converted from paper to match units min -> day
beta1 = 470 * 60 * 24 # max secretion rate of CSF by mF
beta2 = 70 * 60 * 24  # max secretion rate of PDGF by M
beta3 = 240 * 60 * 24  # max secretion rate of PDGF by mF
alpha1 = 940 *60 * 24 # max endocytosis rate of CSF by M
alpha2 = 510 * 60 * 24   # max endocytosis rate of PDGF by mF
gamma = 2  # degradation rate of growth factors
A_0 = 10**6

def myofib_macro(x, t): # outputs list of gradients
    # variables
    mF = x[0]
    M = x[1]
    CSF = x[2]
    PDGF = x[3]
    # diff eqns
    dmFdt= mF*(lambda1 * ((PDGF)/(k1 + PDGF))*(1-mF/K) -mu1)
    dMdt = M*(lambda2*(CSF/(k2+CSF))-mu2)
    dCSFdt = beta1*mF-alpha1*M*(CSF/(k2+CSF))-gamma*CSF
    dPDGFdt= beta2*M+beta3*mF-alpha2*mF*(PDGF)/(k1+PDGF)-gamma*PDGF

    return [dmFdt, dMdt, dCSFdt, dPDGFdt]

# playing around with trajetories
x0 = [6*10**3, 7*10**3] # initial point for trajectory
x1 = [10**3, 10**3, 0, 0]

t = np.linspace(0, 80, 1000)

def derivatives(mF_M):
    mF = mF_M[0]
    M = mF_M[1]
    CSF = mF_M[2]
    PDGF = mF_M[3]
    # diff eqns
    dmFdt= mF*(lambda1 * ((PDGF)/(k1 + PDGF))*(1-mF/K) -mu1)
    dMdt = M*(lambda2*(CSF/(k2+CSF))-mu2)
    dCSFdt = beta1*mF-alpha1*M*(CSF/(k2+CSF))-gamma*CSF
    dPDGFdt = beta2*M+beta3*mF-alpha2*mF*(PDGF)/(k1+PDGF)-gamma*PDGF
    return (dmFdt, dMdt, dCSFdt,dPDGFdt)


def CSF_PDGF_steady(x): # finds steady CSF and PDGF levels for given mF and M levels
    mF = x[0]
    M = x[1]
    # equation for steady CSF is -gamma*(CSF)**2 + CSF*(beta1*mF-alpha1*M-k2*gamma) + beta1*k2*mF
    # equation for steady PDGF is  -gamma*(PDGF)**2 + PDGF * (beta2*M + beta3*mF -alpha2 * mF- gamma * k1) +k1*(beta2*M+beta3*mF)
    c_CSF = np.array([-1*gamma, beta1*mF-alpha1*M-k2*gamma, beta1*k2*mF])
    c_PDGF = np.array([-1*gamma, beta2*M + beta3*mF -alpha2 * mF - gamma * k1, k1*(beta2*M+beta3*mF)])
    CSF_roots = np.roots(c_CSF)
    PDGF_roots = np.roots(c_PDGF)
    root_pairs = []
    for CSF_root in CSF_roots:
        for PDGF_root in PDGF_roots:
                if np.isreal(CSF_root) and np.isreal(PDGF_root) and PDGF_root >= 0 and CSF_root >= 0:
                    root_pairs.append(CSF_root)
                    root_pairs.append(PDGF_root)
    return(root_pairs)



def mF_M_rates(x, t):
    mF = x[0]
    M = x[1]
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    dmFdt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    dMdt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return [dmFdt, dMdt]


x_ = odeint(mF_M_rates, x0, t)

mF = x_[:, 0]
M = x_[:, 1]


mF_log = np.logspace(0, 7, 10)
M_log = np.logspace(1, 7, 10)



def nullcline_mF(x): # finds nullcline of mF
    mF = x
    smF_PDGF =  (mu1 * k1 * K)/(lambda1 * K - mu1*K - mF*lambda1) # after rearranging eqn 1 equated to 0
    smF_M = -1/beta2 *( beta3*mF-alpha2*mF*smF_PDGF/(k1+smF_PDGF)-gamma*smF_PDGF) # rearranging eqn 4 equated to 0
    return [mF, smF_M]

def nullcline_M(x): # finds nullcline of M
    sM_mF = x
    sM_CSF = (k2 * mu2) / (lambda2 - mu2) # equate eqn 2 to 0
    M = ((k2 +sM_CSF)/(alpha1 * sM_CSF) ) * (beta1*sM_mF-gamma*sM_CSF) # equate eqn 3 to 0
    return [sM_mF, M]

mFM_space = np.logspace(0, 7, 10**4)

# mFnull1, mFnull2, mFnull3 are intervals that do not contain any poles
# smoothmF1 and smoothmF2 are intervals that contain poles
mFnull1 = np.logspace(0, 5.7, 10**3)
smoothmF1 = np.logspace(5.7, 5.85, 10**3)
mFnull2 = np.logspace(5.85, 5.95, 10**3)
smoothmF2 = np.logspace(5.95, 6.05, 10**3)
mFnull3 = np.logspace(6.05, 7, 10**3)

# straight lines to replace/ignore the sharp increase near the poles
xsmooth1 = [10**5.7, 10**5.85]
ysmooth1 = [nullcline_mF(pt)[1] for pt in xsmooth1]
xsmooth2 = [10**5.95, 10**6.1]
ysmooth2 = [nullcline_mF(pt)[1] for pt in xsmooth2]




plt.figure()
plt.plot(nullcline_M(mFM_space)[0], nullcline_M(mFM_space)[1], 'r', label = 'Macrophage nullcline')
#plt.plot(nullcline_mF(mFM_space)[0], nullcline_mF(mFM_space)[1], 'b')
plt.plot(nullcline_mF(mFnull1)[0], nullcline_mF(mFnull1)[1], 'b', label = 'Myofibroblasts nullcline')
plt.plot(nullcline_mF(mFnull2)[0], nullcline_mF(mFnull2)[1], 'b')
plt.plot(nullcline_mF(mFnull3)[0], nullcline_mF(mFnull3)[1], 'b')
plt.plot(xsmooth1, ysmooth1, 'b')
print(nullcline_mF(10**5.7), nullcline_mF(10**5.85), nullcline_mF(10**5.95), nullcline_mF(10**6.05))
plt.xlabel("myofibroblasts")
plt.ylabel("macrophages")
plt.xlim((1, 10**7))
plt.ylim((1, 10**7))
plt.xscale("log")
plt.yscale("log")



def intersectionNull_bad(): # finds intersection of nullclines, bad estimate
    mF_list = []
    def diff(x):
        return nullcline_M(x)[1] - nullcline_mF(x)[1]
    for i in range(len(mFM_space)-1):
        if diff(mFM_space[i]) == 0 or diff(mFM_space[i]) * diff(mFM_space[i+1]) < 0:
            mF_list.append(mFM_space[i])
    return mF_list


fpt_mF_bad = intersectionNull_bad()
fpt_M_bad = [nullcline_M(i)[1] for i in fpt_mF_bad]
print(intersectionNull_bad(), fpt_M_bad, "fixed point approximations where mF and M are nonzero")


def nulldiff(x):
    return nullcline_M(x)[1] - nullcline_mF(x)[1]

# use initial fixed point approximations to find solutions for the unstable point and the hot fibrosis point
uns_guess = fpt_mF_bad[0]
second_guess = fpt_mF_bad[1]


uns_soln = fsolve(nulldiff, uns_guess)
uns_soln = uns_soln[0] # changes array to float

hotfibr = fsolve(nulldiff, second_guess)
hotfibr = hotfibr[0]  # changes array to float

print(uns_soln, "mF value at unstable fix pt")
print(hotfibr, "mF value at hot fibrosis")

uns_soln2 = nullcline_mF(uns_soln) # gives [mF, M] at unstable fixed point from mF value
uns_CSFPDGF = CSF_PDGF_steady(uns_soln2) # gives [CSF, PDGF] at unstable fixed point from mF value
nullsoln4 = uns_soln2 + uns_CSFPDGF # get [mF, M, CSF, PDGF] at unstable fixed point

hotfibr2 = nullcline_mF(hotfibr)
hot_CSF_PDGF = CSF_PDGF_steady(hotfibr2)
hotfibr4 = hotfibr2 + hot_CSF_PDGF

plt.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
plt.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')

print(nullsoln4)

print(myofib_macro(nullsoln4, t), "rates at unstable fixed pt")


def cold_fibr(): # finds the cold fibrosis point
    # Set M = 0 in eqn 4, use eqn 1. solve system for PDGF, get a cubic
    PDGF_coeff = np.array([-gamma, (K/lambda1)*(lambda1-mu1)*(beta3-alpha2)-gamma*k1,
                           (K/lambda1)*(lambda1-mu1+beta3-alpha2), -mu1* k1**2 * beta3 * K/lambda1])
                            # rearranged from eqns in transparent methods
    coldPDGF = np.roots(PDGF_coeff)
    coldmF = []
    for coldroot in coldPDGF:
        if np.isreal(coldroot):
            coldmF.append(K * ((lambda1-mu1)/(lambda1)-(mu1*k1)/(lambda1*np.real(coldroot)))) # finds mF value given PDGF value
    return coldmF[0]

coldfibr2 = [cold_fibr(), 1]

def rev_mF_M(x,t):
    mF = x[0]
    M = x[1]
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    dmFdt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    dMdt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return [-dmFdt, -dMdt]

def myofib_macro_ODE_reverse(x, t):    # outputs reverse derivative
    # variables
    mF = x[0]
    M = x[1]
    if 0 <= mF <= 10**6 and 0 <= M <= 10**7:
        return [-i for i in mF_M_rates(x, t)]
    else:
        return [0, 0]

eps = 1e-6
t_sep = np.linspace(0, 800, 1000)

separatrix_left = odeint(myofib_macro_ODE_reverse, [uns_soln2[0]-eps, uns_soln2[1]+eps], t_sep)

plt.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black', label = 'Separatrix')

separatrix_right = odeint(myofib_macro_ODE_reverse, [uns_soln2[0]+eps, uns_soln2[1]-eps], t_sep)

plt.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
plt.annotate('unstable fixed point', uns_soln2)
plt.annotate('hot fibrosis fixed point', hotfibr2)
plt.plot(coldfibr2[0], coldfibr2[1], marker = 'o', color = "black")
plt.annotate('cold fibrosis fixed point', coldfibr2)

plt.legend()

# to plot the streamlines, we need to redefine functions to find steady CSF and PDGF levels that work for arrays


def CSF_PDGF_steady_array(x): # finds steady CSF and PDGF levels for given mF and M levels
    mF = x[0]
    M = x[1]
    # equation for steady CSF is -gamma*(CSF)**2 + CSF*(beta1*mF-alpha1*M-k2*gamma) + beta1*k2*mF
    # equation for steady PDGF is  -gamma*(PDGF)**2 + PDGF * (beta2*M + beta3*mF -alpha2 * mF- gamma * k1) +k1*(beta2*M+beta3*mF)
    c_CSF_array = np.array([-1*gamma*np.ones(np.shape(mF)), beta1*mF-alpha1*M-k2*gamma, beta1*k2*mF])
    c_PDGF_array = np.array([-1*gamma*np.ones(np.shape(mF)), beta2*M + beta3*mF -alpha2 * mF - gamma * k1, k1*(beta2*M+beta3*mF)])
    # define empty arrays fo CSF and PDGF
    CSF_array = np.zeros(np.shape(mF))
    PDGF_array = np.zeros(np.shape(mF))
    for i in range(0, np.shape(mF)[0]):
        for j in range(0, np.shape(mF)[1]):
            # get 1d arrays of CSF and PDGF coefficients for each grid value
            c_CSF = np.array([c_CSF_array[0][i][j], c_CSF_array[1][i][j], c_CSF_array[2][i][j]])
            c_PDGF = np.array([c_PDGF_array[0][i][j], c_PDGF_array[1][i][j], c_PDGF_array[2][i][j]])
            CSF_roots = np.roots(c_CSF)
            PDGF_roots = np.roots(c_PDGF)
            for CSF_root in CSF_roots:
                for PDGF_root in PDGF_roots:
                    if np.isreal(CSF_root) and np.isreal(PDGF_root) and PDGF_root >= 0 and CSF_root >= 0:
                        CSF_array[i][j] = CSF_root
                        PDGF_array[i][j] = PDGF_root
    return [CSF_array, PDGF_array]



def mF_M_rates_array(exp_mF, exp_M, t):
    # we need dmFdt and dMdt to be plotted at different values as streamplot can only take in linearly spaced values,
    # so we take in the exponents of mF and M values to get logarithmically spaced
    mF = 10**exp_mF
    M = 10**exp_M
    CSF, PDGF = CSF_PDGF_steady_array([mF, M])
    dmFdt = mF * (lambda1 * ((PDGF)/(k1+PDGF))*(1-mF/K)-mu1)
    dMdt = M*(lambda2*(CSF/(k2 + CSF))- mu2)
    return dmFdt, dMdt

# plotting streamlines

fig = plt.figure()

mF_mesh = np.linspace(0, 6, 25)
M_mesh = np.linspace(0, 6, 25)
mF_stream, M_stream = np.meshgrid(mF_mesh, M_mesh) # returns

ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)



mF_rate, M_rate = mF_M_rates_array(mF_stream, M_stream, t)
# we need to adjust for the exponential values
mF_rate_scaled = mF_rate/(10**mF_stream)
M_rate_scaled = M_rate/(10**M_stream)

strm = ax.streamplot(mF_stream, M_stream, mF_rate_scaled, M_rate_scaled,
                     color = (np.sqrt((mF_rate_scaled)**2 + (M_rate_scaled)**2)) , cmap = 'autumn')


ax.set_xlim(0, 6)
ax.set_ylim(0,6)
ax.set_xticks([])
ax.set_yticks([])
ax2.set_xlabel('myofibroblasts')
ax2.set_ylabel('macrophages')



ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1, 10**6)
ax2.set_ylim(1, 10**6)
ax2.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
ax2.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')
ax2.plot(coldfibr2[0], coldfibr2[1], marker = 'o', color = "black")

print(mF_M_rates(10))

def theta(t): # heaviside step function
    if t >= 0:
        return 1
    else:
        return 0


# define transient, repetitive and prolonged signals
def transient(t):
    return A_0*(theta(t) - theta(t-2))

def repetitive(t):
    return transient(t) + transient(t-4)

def prolonged(t):
    return A_0 *(theta(t)- theta(t-4))

# defining new derivative values for transient, repetitive and prolonged signals

def transient_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + transient(t)
    return new_derivatives

def repetitive_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + repetitive(t)
    return new_derivatives

def prolonged_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + prolonged(t)
    return new_derivatives



x_in = [1, 1] # initial [mF, M] values
print(repetitive_derivatives(x_in, 4))



x_transient = odeint(transient_derivatives, x_in, t)
x_repetitive = odeint(repetitive_derivatives, x_in, t)
x_prolonged = odeint(prolonged_derivatives, x_in, t)



def time_taken(traj):
    end_point = [0, 0]
    for fixed_pt in [hotfibr2, uns_soln2]:
        if (traj[-1][0] - fixed_pt[0])**2 + (traj[-1][1] - fixed_pt[1])**2 < (traj[-1][0] - end_point[0])**2 + (traj[-1][1] - end_point[1])**2:
            end_point = fixed_pt # find correct end point
            for i in range(len(traj)):
                if (traj[i][0] - end_point[0])**2 + (traj[i][1] - end_point[1])**2 < 1e-3 *((end_point[0])**2 + (end_point[1])**2) :
                    return t[i] # finds time at which trajectory reaches end point if end point is not 0
    for i in range(len(traj)):
        if ((traj[i][0]) ** 2 + (traj[i][1]) ** 2) < 1e-3:
            return t[i] # finds time at which trajectory reaches end point if end point is 0


def time_taken_rd(traj): # so that time_taken can be rounded and included in figures
    return round(time_taken(traj), 2)

print(time_taken(x_repetitive))
print(time_taken(x_prolonged))
print(time_taken(x_transient))


# plotting transient, repetitive and prolonged signals

fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)
fig.subplots_adjust(hspace= 0.5)

# setting up trajectory plots
for ax2 in (ax21, ax22, ax23):
    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
    ax2.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')
    ax2.plot(coldfibr2[0], coldfibr2[1], marker='o', color="black")

# setting up injury plots
for ax1 in (ax11, ax12, ax13):
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')
    ax1.set_xticks([0, 2, 4, 6, 8])
    ax1.set_yticks([0, 9], [0, "A0".translate(SUB)])

# plotting separate signals
ax21.plot(x_transient[:, 0], x_transient[:, 1], 'deepskyblue')
ax21.set_title("time taken: " + str(time_taken_rd(x_transient)) + " days")
ax22.plot(x_repetitive[:, 0], x_repetitive[:, 1], 'orange')
ax22.set_title("time taken: " + str(time_taken_rd(x_repetitive)) + " days")
ax23.plot(x_prolonged[:, 0], x_prolonged[:, 1], 'green')
ax23.set_title("time taken: " + str(time_taken_rd(x_prolonged)) + " days")

# plotting injury signal graphs
#transient signal
tr_x = [0, 2, 3]
tr_y = [0, 9, 0]
ax11.step(tr_x, tr_y, color = 'deepskyblue')
ax11.set_title('Transient signal')
#repetitive signal
re_x = [0, 2, 4, 6, 7]
re_y = [0, 9, 0, 9, 0]
ax12.step(re_x, re_y, color = 'orange')
ax12.set_title('Repetitive signal')
# prolonged signal
pr_x = [0, 4, 5]
pr_y = [0, 9, 0]

ax13.step(pr_x, pr_y, color = 'green')
ax13.set_title('Prolonged signal')


# plotting signals for one constant signal delivered over different lengths of time
fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)

fig.subplots_adjust(hspace= 0.5)

# setting up plots for trajectories
for ax2 in (ax21, ax22, ax23):
    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
    ax2.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel('myofibroblasts')
    ax2.set_ylabel('macrophages')
    ax2.plot(coldfibr2[0], coldfibr2[1], marker='o', color="black")

# setting up plots for injury signals
for ax1 in (ax11, ax12, ax13):
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('I(t)')


# defining area under injury signal graph for figures 4 & 5
total_signal1  = 2 * A_0
total_signal = 4 * A_0


# defines one constant injury signal with fixed injury amount,
# taking in t_del = time during which the signal is delivered
def one_signal(t, t_del):
    return (total_signal1/t_del)*(theta(t) - theta(t - t_del))

small_t = 0.2

def sharp_signal(t):
    return (total_signal1/small_t)*(theta(t) - theta(t - small_t))

def inter_signal(t):
    return one_signal(t, 8)

def slow_signal(t):
    return one_signal(t, 20)

# defining new derivatives
def sharp_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + sharp_signal(t)
    return new_derivatives

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


x_sharp = odeint(sharp_derivatives, x_in, t)
x_inter = odeint(inter_derivatives, x_in, t)
x_slow = odeint(slow_derivatives, x_in, t)


# plotting step function/injury signals

sharp_x = [0, 1, 10]
sharp_y = [0, 9, 0]

ax21.plot(x_sharp[:,0], x_sharp[:,1], color = 'm')
ax21.set_title("time taken:" + str(round(time_taken(x_sharp), 2)) + " days")
ax11.step(sharp_x, sharp_y, color = 'm')
ax11.set_xticks([0, 1], [0, 0.1])
ax11.set_yticks([0, 9], [0, "20"+"A0".translate(SUB)])
ax11.set_title('Sharp signal')

ax22.plot(x_inter[:, 0], x_inter[:, 1], color = 'pink')
ax22.set_title("time taken: " + str(time_taken_rd(x_inter)) + " days")



# plotting step function/injury signals
inter_x = [0,8, 10]
inter_y = [0, 1.125, 0] # find magnitude of signal by dividing total_signal/t_del

ax12.step(inter_x, inter_y, color = 'pink')
ax12.set_xticks([0, 8])
ax12.set_yticks([0, 1.125], [0, "0.25" + "A0".translate(SUB)])
ax12.set_title('Low signal')

slow_x = [0, 10, 11]
slow_y = [0, 0.45, 0]

ax23.plot(x_slow[:,0], x_slow[:, 1], color ='crimson')
ax23.set_title("time taken: " + str(time_taken_rd(x_slow)) + " days")
ax13.step(slow_x, slow_y, color = 'crimson')
ax13.set_xticks([0, 10], [0, 20])
ax13.set_yticks([0, 0.45], [0, "0.1" + "A0".translate(SUB)])
ax13.set_title('Minimal signal')


# plotting signals of different amplitudes at different times with constant area

fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)
fig.subplots_adjust(hspace = 0.5)

# setting up figure
for ax2 in (ax21, ax22, ax23):
    ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
    ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 10**7)
    ax2.set_ylim(1, 10**7)
    ax2.plot(uns_soln2[0], uns_soln2[1], marker = 'o', color = 'black')
    ax2.plot(hotfibr2[0], hotfibr2[1], marker = 'o', color = 'black')
    ax2.set_aspect('equal')
    ax2.set_xticks([10**i for i in range(8)])
    ax2.set_xlabel("myofibroblasts")
    ax2.set_ylabel("macrophages")
    ax2.plot(coldfibr2[0], coldfibr2[1], marker='o', color="black")

for ax1 in (ax11, ax12, ax13):
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,24)
    ax1.set_ylim(0,24)
    ax1.set_aspect('equal')
    ax1.set_xlabel('time taken (days)')
    ax1.set_ylabel('I(t)')

# defining blocky signals
def blocks1(t): # increasing steps
    return 0.25*one_signal(t, 8) + 0.35*one_signal(t-8, 2) + 0.4*one_signal(t-10, 2)

def blocks2(t): # increasing then decreasing step
    return 0.6*one_signal(t, 20) + 0.3*one_signal(t-20, 1) + 0.1*one_signal(t-21, 3)

def blocks3(t): # decreasing steps
    return 0.5*one_signal(t, 2) + 0.3*one_signal(t-2, 2) + 0.2*one_signal(t-4, 2)

# defining new derivatives
def blocks1_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + blocks1(t)
    return new_derivatives

def blocks2_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + blocks2(t)
    return new_derivatives

def blocks3_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + blocks3(t)
    return new_derivatives

x_blocks1 = odeint(blocks1_derivatives, x_in, t)
x_blocks2 = odeint(blocks2_derivatives, x_in, t)
x_blocks3 = odeint(blocks3_derivatives, x_in, t)



# plotting injury step function
blocks1_x = [0, 8, 10, 12, 13]
blocks1_y = [50*i for i in [0, 0.25/8, 0.35/2, 0.4/2, 0]] # adjusting to fit into frame

ax21.plot(x_blocks1[:,0], x_blocks1[:, 1], color = 'tomato')
ax21.set_title("time taken: " + str(time_taken_rd(x_blocks1)) + " days")


ax11.set_yticks([50*i for i in [0.25/8, 0.35/2, 0.4/2]], [str(0.25/8) + "A0".translate(SUB),
                                                          str(0.35/2)+ "A0".translate(SUB),
                                                          str(0.2) + "A0".translate(SUB)])
ax11.step(blocks1_x, blocks1_y, color = 'tomato')
ax11.set_xticks([0, 8, 10, 12])
ax11.set_title('Increasing signal')

blocks2_x = [0, 20, 21, 24, 25]
blocks2_y = [50*i for i in [0, 0.6/20, 0.3/1, 0.1/3, 0]] # adjusting to fit into frame

ax22.plot(x_blocks2[:,0], x_blocks2[:, 1], color = 'darkorange')
ax22.set_title("time taken: " + str(time_taken_rd(x_blocks2)) + " days")

ax12.set_title('Spike in signal')
ax12.step(blocks2_x, blocks2_y, color = 'darkorange')
ax12.set_yticks([50*i for i in [0.6/20, 0.3]], [j +
                "A0".translate(SUB) for j in [str(i) for i in [0.6/20, 0.3]]])
ax12.set_xticks([0, 20, 21, 24])

blocks3_x = [0,2,4,6,7]
blocks3_y = [50*i for i in [0, 0.5/2, 0.3/2, 0.2/2, 0]] # adjusting to fit into frame

ax23.plot(x_blocks3[:,0], x_blocks3[:, 1], color = 'gold')
ax23.set_title("time taken: " + str(time_taken_rd(x_blocks3)) + " days")
ax13.step(blocks3_x, blocks3_y, color = 'gold')
ax13.set_yticks([50*i for i in [0.5/2, 0.3/2, 0.2/2]], [j + "A0".translate(SUB) for j in
                                                        [str(i) for i in [0.5/2, 0.3/2, 0.2/2]]])
ax13.set_xticks([0, 2, 4, 6])
ax13.set_title("Decreasing steps")

plt.show()