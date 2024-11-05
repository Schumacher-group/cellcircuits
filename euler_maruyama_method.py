'''
Implement a numerical solution to stochastic differential equation
dX_t = mu(X_t,t) dt + sigma(X_t,t) dW_t where dW_t is a Wiener process (Brownian motion) using Euler Maruyama method
Our signal function is given as [dmF, dM + signal + noise], so we need to apply the following recursion
 [mF_n+1, dM_n+1] = [mF_n, M_n] + [mF_n * dt, (dM_n + signal)*dt  + sqrt(dt) * noise] 
'''
 
import numpy as np

def euler_maruyama(deterministic_derivative, noise_function ,t_steps, x0, dt):

    x = np.zeros((t_steps.size, 2))
    x[0] = x0

    noise_terms = noise_function(t_steps, dt)

    for k in range(1, t_steps.size):
        deterministic_term = np.array(deterministic_derivative(x[k-1], t_steps[k-1]))
        #noise affects only the macrophages influx (second entry)
        x[k] = x[k-1] + dt * deterministic_term + np.array([0, noise_terms[k-1]])
    
    return x[-1], x


#Can be used if you want to only use non-parallelized code
def simulate_euler_maruyama(deterministic_derivative, noise_function, t_trajectory, x0, num_sim):
    t0 = t_trajectory[0]
    dt = (t_trajectory[-1] - t0)/(t_trajectory.size)

    t_steps = np.linspace(t0, t_trajectory[-1], t_trajectory.size)

    end_points = np.zeros((num_sim, 2))
    trajectories = []

    for num in range(num_sim):
        end_point, x = euler_maruyama(deterministic_derivative, noise_function, t_steps, x0, dt)

        end_points[num] = end_point
        trajectories.append(x)

    trajectories = np.array(trajectories)
    return end_points, trajectories


def single_euler_maruyama_simulation(deterministic_derivative, noise_function, t_trajectory, x0):
    t0 = t_trajectory[0]
    dt = t_trajectory[1] - t_trajectory[0]

    t_steps = np.linspace(t0, t_trajectory[-1], t_trajectory.size)

    end_point, x = euler_maruyama(deterministic_derivative, noise_function, t_steps, x0, dt)

    return end_point, x