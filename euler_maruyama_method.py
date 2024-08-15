#Implement a numerical solution to stochastic differential equation
#dX_t = mu(X_t,t) dt + sigma(X_t,t) dW_t where dW_t is a Wiener process (Brownian motion) using Euler Maruyama method
#Our signal function is given as [dmF, dM + signal + noise], so we need to apply the following recursion
# [mF_n+1, dM_n+1] = [mF_n, M_n] + [mF_n * dt, (dM_n + signal)*dt  + sqrt(dt) * noise] 

import numpy as np

def euler_maruyama(deterministic_derivative, noise_function ,t_steps, x0, dt, sqrt_dt):

    x = np.zeros((t_steps.size, 2))
    x[0] = x0

    noise_terms = np.array([noise_function(t) for t in t_steps])

    for k in range(1, t_steps.size):
        deterministic_term = np.array(deterministic_derivative(x[k-1], t_steps[k-1]))
        x[k] = x[k-1] + dt * deterministic_term + sqrt_dt * np.array([0, noise_terms[k-1]])
    
    return x

def simulate_euler_maruyama(deterministic_derivative, noise_function, t_trajectory, x0, num_steps, axis):
    t0 = t_trajectory[0]
    dt = (t_trajectory[-1] - t0)/(t_trajectory.size)
    sqrt_dt = np.sqrt(dt)

    t_steps = np.linspace(t0, t_trajectory[-1], t_trajectory.size)

    for _ in range(num_steps):
        x = euler_maruyama(deterministic_derivative, noise_function, t_steps, x0, dt, sqrt_dt)
        axis.plot(x[:,0], x[:,1])