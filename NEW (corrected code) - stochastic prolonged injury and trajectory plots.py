import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define constants
lambda1 = 0.9  # proliferation rate of mF
lambda2 = 0.8  # proliferation rate of M
mu1 = 0.3  # removal rate of mF
mu2 = 0.3  # removal rate of M
K = 10 ** 6  # carrying capacity of mF
k1 = 10 ** 9  # binding affinity of CSF
k2 = 10 ** 9  # binding affinity of PDGF
beta1 = 470 * 60 * 24  # max secretion rate of CSF by mF
beta2 = 70 * 60 * 24  # max secretion rate of PDGF by M
beta3 = 240 * 60 * 24  # max secretion rate of PDGF by mF
alpha1 = 940 * 60 * 24  # max endocytosis rate of CSF by M
alpha2 = 510 * 60 * 24  # max endocytosis rate of PDGF by mF
gamma = 2  # degradation rate of growth factors
A_0 = 10 ** 6

# Define the mean and standard deviation for the injury signal
I_mean = A_0  # Mean of the transient injury signal
sigma =  0.1 * A_0  # Initial guess for noise strength

# Time step
dt = 0.1
t = np.linspace(0, 80, 1000)  # time vector
num_steps = len(t)

# Function to find steady CSF and PDGF levels for given mF and M levels
def CSF_PDGF_steady(x):
    mF = x[0]
    M = x[1]
    
    # Coefficients for CSF and PDGF cubic equations
    c_CSF = np.array([-1 * gamma, beta1 * mF - alpha1 * M - k2 * gamma, beta1 * k2 * mF])
    c_PDGF = np.array([-1 * gamma, beta2 * M + beta3 * mF - alpha2 * mF - gamma * k1, k1 * (beta2 * M + beta3 * mF)])
    
    # Find roots of CSF and PDGF
    CSF_roots = np.roots(c_CSF)
    PDGF_roots = np.roots(c_PDGF)
    
    # Initialize root pairs list
    root_pairs = []
    
    # Find valid real, non-negative roots
    for CSF_root in CSF_roots:
        for PDGF_root in PDGF_roots:
            if np.isreal(CSF_root) and np.isreal(PDGF_root) and PDGF_root >= 0 and CSF_root >= 0:
                root_pairs.append((np.real(CSF_root), np.real(PDGF_root)))  # Append real part as tuple
    
    # Check if we have found any valid pairs, if not return fallback values
    if len(root_pairs) > 0:
        return root_pairs[0]  # Return the first valid pair
    else:
        # Return fallback values if no valid roots are found
        return (0, 0)

# Function to calculate deterministic derivatives for mF and M
def mF_M_rates(x, t):
    mF = x[0]
    M = x[1]
    CSF, PDGF = CSF_PDGF_steady([mF, M])
    dmFdt = mF * (lambda1 * (PDGF / (k1 + PDGF)) * (1 - mF / K) - mu1)
    dMdt = M * (lambda2 * (CSF / (k2 + CSF)) - mu2)
    return [dmFdt, dMdt]

# Function to calculate deterministic derivatives for transient injury
def prolonged_derivatives(x, t):
    regular_derivatives = mF_M_rates(x, t)
    new_derivatives = regular_derivatives
    if t <= 4:
        new_derivatives[1] += A_0  # Add transient injury signal
    return new_derivatives

# Euler-Maruyama method for stochastic injury
def stochastic_prolonged_injury(x, t, dt):
    regular_derivatives = mF_M_rates(x, t)
    noise_term = np.random.normal(0, 1) * np.sqrt(dt) * sigma if t <= 4 else 0
    dmFdt = regular_derivatives[0] * dt  # deterministic update for mF
    dMdt = (regular_derivatives[1] + (I_mean if t <= 4 else 0)) * dt + noise_term

    return np.array([dmFdt, dMdt])


# Function to calculate the separatrix
def myofib_macro_ODE_reverse(x, t):
    mF = x[0]
    M = x[1]
    if 0 <= mF <= 10**6 and 0 <= M <= 10**7:
        return [-i for i in mF_M_rates(x, t)]
    else:
        return [0, 0]

# Function to find nullclines
def nullcline_mF(x):
    mF = x
    smF_PDGF = (mu1 * k1 * K) / (lambda1 * K - mu1 * K - mF * lambda1)
    smF_M = -1 / beta2 * (beta3 * mF - alpha2 * mF * smF_PDGF / (k1 + smF_PDGF) - gamma * smF_PDGF)
    return [mF, smF_M]

def nullcline_M(x):
    sM_mF = x
    sM_CSF = (k2 * mu2) / (lambda2 - mu2)
    M = ((k2 + sM_CSF) / (alpha1 * sM_CSF)) * (beta1 * sM_mF - gamma * sM_CSF)
    return [sM_mF, M]

# Initial conditions for separatrix
uns_soln2 = nullcline_mF(5328.844696280525)  # Example point

# Separatrix calculation
def calculate_separatrix(uns_soln2):
    eps = 1e-6
    t_sep = np.linspace(0, 800, 1000)
    separatrix_left = odeint(myofib_macro_ODE_reverse, [uns_soln2[0] - eps, uns_soln2[1] + eps], t_sep)
    separatrix_right = odeint(myofib_macro_ODE_reverse, [uns_soln2[0] + eps, uns_soln2[1] - eps], t_sep)
    return separatrix_left, separatrix_right

separatrix_left, separatrix_right = calculate_separatrix(uns_soln2)

# Initial conditions for trajectories
x_in = [1, 1]  # Initial [mF, M]

# Deterministic transient injury trajectory
x_transient = odeint(prolonged_derivatives, x_in, t)
mF_transient = x_prolonged[:, 0]
M_transient = x_prolonged[:, 1]

# Stochastic transient injury trajectory using Euler-Maruyama
x_stochastic = np.zeros((num_steps, 2))
x_stochastic[0] = x_in

for i in range(1, num_steps):
    x_stochastic[i] = x_stochastic[i - 1] + stochastic_prolonged_injury(x_stochastic[i - 1], t[i - 1], dt)

# Extract results for plotting
mF_stochastic = x_stochastic[:, 0]
M_stochastic = x_stochastic[:, 1]

# Plot the trajectories and the separatrix
plt.figure()
plt.plot(mF_stochastic, M_stochastic, label='Stochastic Prolonged Trajectory', color='purple')
plt.scatter(mF_stochastic[-1], M_stochastic[-1], color='red', zorder=5, label='Stochastic End Point')

plt.plot(mF_prolonged, M_prolonged, label='Deterministic Prolonged Trajectory', color='green')
plt.scatter(mF_prolonged[-1], M_prolonged[-1], color='blue', zorder=5, label='Deterministic End Point')

plt.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black', label='Separatrix')
plt.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')

plt.xlabel('Myofibroblast Concentration')
plt.ylabel('Macrophage Concentration')
plt.title('Trajectory and Stop Points for Prolonged Injuries')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 10**7)
plt.ylim(1, 10**7)
plt.legend()
plt.show()


# time taken to reach steady state 
# Tolerance for detecting steady state (small changes)
tolerance = 1e-3

# Function to detect the time at which the system reaches a steady state
def detect_end_time(trajectory, time_vector, tolerance=1e-3):
    num_steps = len(trajectory)
    for i in range(1, num_steps):
        diff = np.abs(trajectory[i] - trajectory[i-1])
        if np.all(diff < tolerance):  # Check if both mF and M changes are small
            return time_vector[i]
    return time_vector[-1]  # Return final time if steady state is not reached

# Find the end time for deterministic trajectory
end_time_deterministic = detect_end_time(x_transient, t, tolerance)

# Find the end time for stochastic trajectory
end_time_stochastic = detect_end_time(x_stochastic, t, tolerance)

# Print the results
print(f"Time to reach steady state for deterministic trajectory: {end_time_deterministic:.2f} units")
print(f"Time to reach steady state for stochastic trajectory: {end_time_stochastic:.2f} units")

# run simaulation 1000 times
# Number of simulations
num_simulations = 1000
final_end_points = []

plt.figure()

# Run the stochastic simulation multiple times and collect final endpoints
for _ in range(num_simulations):
    x_stochastic = np.zeros((num_steps, len(x_in)))
    x_stochastic[0] = x_in

    # Iterate using the Euler-Maruyama method
    for i in range(1, num_steps):
        t_step = t[i-1]  # Use the time vector t
        x_stochastic[i] = x_stochastic[i-1] + stochastic_prolonged_injury(x_stochastic[i-1], t_step, dt)

    mF_stochastic = x_stochastic[:, 0]
    M_stochastic = x_stochastic[:, 1]
    
    # Plot each stochastic trajectory
    plt.plot(mF_stochastic, M_stochastic, color='purple', alpha=0.1)
    plt.scatter(mF_stochastic[-1], M_stochastic[-1], color='red', s=5, zorder=5)  # Final points in red
    
    # Append final points to the list
    final_end_points.append((mF_stochastic[-1], M_stochastic[-1]))

# Plot the deterministic trajectory
plt.plot(mF_prolonged, M_prolonged, label='Deterministic Prolonged Trajectory', color='green')
plt.scatter(mF_prolonged[-1], M_prolonged[-1], color='blue', zorder=5, label='Deterministic End Point')

# Plot the separatrix
plt.plot(separatrix_left[:, 0], separatrix_left[:, 1], 'black', label='Separatrix')
plt.plot(separatrix_right[:, 0], separatrix_right[:, 1], 'black')

# Set plot attributes
plt.xlabel('Myofibroblast Concentration')
plt.ylabel('Macrophage Concentration')
plt.title('Stochastic and Deterministic Trajectories for Prolonged Injuries')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 10**7)
plt.ylim(1, 10**7)
plt.legend()
plt.show()


# find average times to healing and fibrosis for stocastic and deterministic trajectories 
import numpy as np
import matplotlib.pyplot as plt

# Reverse the left separatrix (as it's calculated from right to left)
rev_left_separatrix = np.flipud(separatrix_left)

# Concatenate the reversed left separatrix with the right separatrix to form the complete separatrix
separatrix = np.vstack((rev_left_separatrix, separatrix_right))

# Initialize time trackers
times_det_healing = []
times_stoch_healing = []
times_stoch_fibrosis_det_healing = []
times_det_fibrosis = []
times_stoch_fibrosis = []

# Function to detect when the trajectory reaches a steady state
def detect_end_time(trajectory, time_vector):
    tolerance = 1e-3  # Small threshold to detect changes
    for i in range(1, len(trajectory)):
        if np.all(np.abs(trajectory[i] - trajectory[i-1]) < tolerance):
            return time_vector[i]
    return time_vector[-1]  # If no change detected, return the last time step

# Determine the time for the deterministic trajectory to reach its end point
deterministic_end_time = detect_end_time(x_transient, t)

# Run stochastic simulation multiple times
for _ in range(num_simulations):
    # Initialize the stochastic trajectory
    x_stochastic = np.zeros((num_steps, len(x_in)))
    x_stochastic[0] = x_in

    # Iterate using the Euler-Maruyama method
    for i in range(1, num_steps):
        t_step = t[i-1]
        x_stochastic[i] = x_stochastic[i-1] + stochastic_transient_injury(x_stochastic[i-1], t_step, dt)

    mF_stochastic = x_stochastic[:, 0]
    M_stochastic = x_stochastic[:, 1]

    # Interpolate using the full separatrix
    interpolated_M = np.interp(mF_stochastic[-1], separatrix[:, 0], separatrix[:, 1])

    # Detect the time for the stochastic trajectory to reach its end point
    stochastic_end_time = detect_end_time(x_stochastic, t)

    # Classify the stochastic outcome as healing or fibrosis
    if M_stochastic[-1] < interpolated_M:
        # If both deterministic and stochastic tend to healing
        times_det_healing.append(deterministic_end_time)
        times_stoch_healing.append(stochastic_end_time)
    else:
        # If deterministic tends to healing but stochastic tends to fibrosis
        times_stoch_fibrosis_det_healing.append(stochastic_end_time)

# Calculate average times
average_time_det_healing = np.mean(times_det_healing) if times_det_healing else 0
average_time_stoch_healing = np.mean(times_stoch_healing) if times_stoch_healing else 0
average_time_stoch_fibrosis_det_healing = np.mean(times_stoch_fibrosis_det_healing) if times_stoch_fibrosis_det_healing else 0

# Print results
print(f"Average deterministic time to healing: {average_time_det_healing:.2f}")
print(f"Average stochastic time to healing: {average_time_stoch_healing:.2f}")
print(f"Average stochastic time to fibrosis (while deterministic heals): {average_time_stoch_fibrosis_det_healing:.2f}")

# Plot the results
outcomes = ['Deterministic Healing', 'Stochastic Healing', 'Stochastic Fibrosis (Det. Heals)']
avg_times = [average_time_det_healing, average_time_stoch_healing, average_time_stoch_fibrosis_det_healing]

plt.figure()
plt.bar(outcomes, avg_times, color=['blue', 'green', 'red'])
plt.xlabel('Trajectory Outcome')
plt.ylabel('Average Time to End Point')
plt.title('Average Time to Reach Healing/Fibrosis')
plt.show()


# graph results of healing/ fibrosis 
 Reverse the left separatrix (as it's calculated from right to left)
rev_left_separatrix = np.flipud(separatrix_left)

# Concatenate the reversed left separatrix with the right separatrix to form the complete separatrix
separatrix = np.vstack((rev_left_separatrix, separatrix_right))

# Initialize the counts
healing_count = 0
fibrosis_count = 0

# Categorize the final end points as healing or fibrosis using the complete separatrix
for end_point in final_end_points:
    # Interpolate using the full separatrix
    interpolated_M = np.interp(end_point[0], separatrix[:, 0], separatrix[:, 1])
    
    # Compare the macrophage concentration (M) to the interpolated value to classify as healing or fibrosis
    if end_point[1] < interpolated_M:
        healing_count += 1
    else:
        fibrosis_count += 1

# Print the counts
print(f"Healing count (0): {healing_count}")
print(f"Fibrosis count (1): {fibrosis_count}")

# Plot the bar graph to show the results
outcomes = ['Healing (0)', 'Fibrosis (1)']
counts = [healing_count, fibrosis_count]

plt.figure()
plt.bar(outcomes, counts, color=['blue', 'red'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Number of Healing and Fibrosis Outcomes')
plt.show()
                              

# healing/fibrosis results for differing sigma levels
# Reverse the left separatrix
rev_left_separatrix = np.flipud(separatrix_left)

# Concatenate the reversed left separatrix with the right separatrix
separatrix = np.vstack((rev_left_separatrix, separatrix_right))

# Initialize different sigma values for the stochastic noise
sigma_values = [0.1 * A_0, 0.5 * A_0, 0.9 * A_0]
healing_counts = []
fibrosis_counts = []

# Run simulations for each sigma value
for sigma in sigma_values:
    # Initialize the counts
    healing_count = 0
    fibrosis_count = 0
    final_end_points = []

    # Run the stochastic simulation multiple times and collect final end points
    for _ in range(num_simulations):
        x_stochastic = np.zeros((num_steps, len(x_in)))
        x_stochastic[0] = x_in

        # Iterate using the Euler-Maruyama method
        for i in range(1, num_steps):
            t_step = t[i-1]  # Adjusted to use the time vector t
            x_stochastic[i] = x_stochastic[i-1] + stochastic_prolonged_injury(x_stochastic[i-1], t_step, dt)

        mF_stochastic = x_stochastic[:, 0]
        M_stochastic = x_stochastic[:, 1]
        final_end_points.append((mF_stochastic[-1], M_stochastic[-1]))

    # Categorize the final end points as healing or fibrosis
    for end_point in final_end_points:
        # Use np.interp with the combined separatrix to classify the point
        interpolated_M = np.interp(end_point[0], separatrix[:, 0], separatrix[:, 1])
        if end_point[1] < interpolated_M:
            healing_count += 1
        else:
            fibrosis_count += 1

    # Store the counts for the current sigma value
    healing_counts.append(healing_count)
    fibrosis_counts.append(fibrosis_count)

# Print the counts for each sigma
for idx, sigma in enumerate(sigma_values):
    print(f"Sigma: {sigma}")
    print(f"Healing count (0): {healing_counts[idx]}")
    print(f"Fibrosis count (1): {fibrosis_counts[idx]}")
    print('-' * 30)

# Plot the bar graph to show the results
plt.figure(figsize=(10, 6))

# Convert sigma values to clean numeric representation (0.1, 0.5, 0.9)
clean_sigma_values = [0.1, 0.5, 0.9]

# Plot for each sigma value
for i, sigma in enumerate(clean_sigma_values):
    plt.bar([f'Healing (0), σ={sigma}', f'Fibrosis (1), σ={sigma}'], 
            [healing_counts[i], fibrosis_counts[i]], color=['blue', 'red'], alpha=0.7)

plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Number of Healing and Fibrosis Outcomes for Different Sigma Levels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()