# Add stocastic signal. 
#Injury Signal:

# I_mean = A_0: Sets the mean level of the prolonged injury signal to a constant A_0.
# sigma = 0.1 * A_0: Sets the standard deviation of the noise component to 10% of A_0. This controls the amplitude of the added noise.
#Time Step Initialization:

# dt = 0.1: Defines a small time step ((\Delta t)) for the Euler-Maruyama method, which is used for numerical integration.
# num_steps = int(t[-1] / dt): Calculates the total number of integration steps by dividing the total time (t[-1] = 80 days) by the time step ((\Delta t = 0.1) days).
# Define the mean and standard deviation for the injury signal
I_mean = A_0  # Mean of the prolonged injury signal
sigma = 0.1 * A_0  # initial guess, can adjust this to alter noise value

# Define the time step and total number of steps
dt = 0.1
num_steps = int(t[-1] / dt)
#Function Definition:
# prolonged stochastic derivatives: A function that computes the derivatives incorporating stochastic noise.

#Calculate Regular Derivatives:
# regular derivatives = mF_M_rates(x, t): Calls another function (mF_M_rates) that returns the deterministic part of the derivative of the state variables ((mF) and (M)) based on the current state x and time t.

#Add Stochastic Term:
# stochastic_term = I_mean + sigma * np.random.normal(0, np.sqrt(dt)): Generates a stochastic term. Here, np.random.normal(0, np.sqrt(dt)) generates a Gaussian noise term with mean 0 and standard deviation sqrt(dt), which scales the noise appropriately for time step dt.

#Update Derivatives:
# new_derivatives = regular_derivatives: Initializes the new derivatives with the deterministic components.
# new_derivatives[1] = new_derivatives[1] + stochastic_term: Adds the stochastic term to the derivative of the second state variable (M in this context).

#Return Updated Derivatives:
# New prolonged injury derivative function to include stochasticity and create new array

def prolonged_stochastic_derivatives(x, t, dt):
    regular_derivatives = mF_M_rates(x, t)
    if t <= 4:
        stochastic_term = I_mean + sigma * np.random.normal(0, np.sqrt(dt))
    else:
        stochastic_term = 0  # Signal turned off after 4 days
    new_derivatives = regular_derivatives
    new_derivatives[1] = new_derivatives[1] + stochastic_term
    return np.array(new_derivatives)
# State Initialization:
# x_stochastic = np.zeros((num_steps, len(x_in))): Initializes an array to store the state variables (mF and M) for each time step. x_in is assumed to contain initial values for both state variables ([mF_0, M_0]).

# Initial Condition:
# x_stochastic[0] = x_in: Sets the initial state of the system to the provided starting values.

# Euler-Maruyama Integration:
# A for loop iterates over each time step to update the state variables using the Euler-Maruyama method:

# Time Calculation: 
# t_step = i * dt calculates the current time.

# Update State: 
#x_stochastic[i] = x_stochastic[i-1] + prolonged_stochastic_derivatives(x_stochastic[i-1], t_step, dt) * dt 
# updates the state variables using the Euler-Maruyama rule, which includes both deterministic and stochastic components.

# mF_stochastic = x_stochastic[:, 0]: Extracts the simulation results for the myofibroblast concentration.
# M_stochastic = x_stochastic[:, 1]: Extracts the simulation results for the macrophage concentration.

# start and state variables
x_stochastic = np.zeros((num_steps, len(x_in)))
x_stochastic[0] = x_in

# Iterate using the Euler-Maruyama method
for i in range(1, num_steps):
    t_step = i * dt
    x_stochastic[i] = x_stochastic[i-1] + prolonged_stochastic_derivatives(x_stochastic[i-1], t_step, dt) * dt

# Extract results for plotting
mF_stochastic = x_stochastic[:, 0]
M_stochastic = x_stochastic[:, 1]

# Plot myofibroblast concentration vs time for both stochastic and non-stochastic prolonged injury
plt.figure()
plt.plot(np.linspace(0, 80, num_steps), mF_stochastic, label='Stochastic Prolonged', color='purple')
plt.plot(t, mF_prolonged, label='Prolonged', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Myofibroblast Concentration')
plt.title('Myofibroblast Concentration vs Time for Prolonged Injury')
plt.legend()
plt.show()

# Plot macrophage concentration vs time for both stochastic and non-stochastic prolonged injury
plt.figure()
plt.plot(np.linspace(0, 80, num_steps), M_stochastic, label='Stochastic Prolonged', color='purple')
plt.plot(t, M_prolonged, label='Prolonged', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Macrophage Concentration')
plt.title('Macrophage Concentration vs Time for Prolonged Injury')
plt.legend()
plt.show()

#plot prolonged signal and trajectory with added noise 


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.subplots_adjust(wspace=0.5)
# Plotting the trajectories
ax2.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black')
ax2.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1, 10**7)
ax2.set_ylim(1, 10**7)
ax2.plot(uns_soln2[0], uns_soln2[1], marker='o', color='black')
ax2.plot(hotfibr2[0], hotfibr2[1], marker='o', color='black')
ax2.plot(coldfibr2[0], coldfibr2[1], marker='o', color='black')
ax2.plot(x_prolonged[:, 0], x_prolonged[:, 1], 'green', label='Prolonged')
ax2.plot(mF_stochastic, M_stochastic, 'purple', label='Stochastic Prolonged')
ax2.set_xlabel('myofibroblasts')
ax2.set_ylabel('macrophages')
ax2.set_title('Prolonged and Stochastic Trajectories')
ax2.legend()



# Define constants
A_0 = 10**6  # Mean of the prolonged injury signal
sigma = 0.1 * A_0  # Standard deviation of the noise

# Define time
t_max = 10  # Maximum time (days)
dt = 0.1
t = np.arange(0, t_max + dt, dt)
num_steps = len(t)

# Deterministic prolonged signal
prolonged_signal = np.piecewise(t, [t < 4, t >= 4], [A_0, 0])

# Stochastic prolonged signal
stochastic_signal = np.zeros(num_steps)
for i in range(num_steps):
    if t[i] < 4:
        stochastic_signal[i] = A_0 + sigma * np.random.normal(0, np.sqrt(dt))
    else:
        stochastic_signal[i] = 0



# Plot deterministic prolonged signal
ax1.step(t, prolonged_signal, where='post', label='Prolonged Signal (Deterministic)', color='green')

# Plot stochastic prolonged signal
ax1.plot(t, stochastic_signal, label='Prolonged Signal (Stochastic)', color='purple', alpha=0.7)

# Styling the plot
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1.5 * A_0)
ax1.set_xlabel('time (days)')
ax1.set_ylabel('I(t)')
ax1.set_xticks([0, 2, 4, 6, 8, 10])
ax1.set_yticks([0, A_0])
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val/A_0)}A_0' if val != 0 else '0'))
ax1.legend()
ax1.set_title('Prolonged signal')

plt.show()
