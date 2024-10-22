import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data
data = np.genfromtxt('data/pathdiv.csv', delimiter=',', skip_header=1)
time = np.array(range(2756))

# Parameters
v = 0.1  # Velocity
predictionhorizon = 5
theta_p = np.random.randn(predictionhorizon - 1) * 0.1  # Random small initial guess

# Initialize predicted state
path_traversed = np.zeros((len(time), 2))
path_traversed[0] = data[0]

def cost(theta_p, state_p_init, time, data, k):
    # Create a local copy of state_p to avoid modifying the original array
    state_p = state_p_init.copy()
    cost = 0  # Initialize cost

    for i in range(1, predictionhorizon):
        deltat = time[k + i] - time[k + i - 1]
        
        # Guard against deltat being zero or too small
        if deltat <= 0:
            return np.inf  # Penalize if the time difference is invalid

        # Update predicted state
        state_p[i] = state_p[i - 1] + np.array([v * deltat * np.cos(theta_p[i - 1]), 
                                                v * deltat * np.sin(theta_p[i - 1])])

        # Check for NaN or inf in state_p (to catch invalid operations early)
        if np.any(np.isnan(state_p[i])) or np.any(np.isinf(state_p[i])):
            return np.inf  # Return a large cost if state becomes invalid
        
        # Calculate the cost as the squared distance from the real data point
        cost += np.sum((data[k + i] - state_p[i]) ** 2)

    return cost

# def numerical_gradient(cost_function, theta_p, *args, epsilon=1e-5):
#     """
#     Compute numerical gradient of the cost function w.r.t. theta_p
#     """
#     grad = np.zeros_like(theta_p)
#     for i in range(len(theta_p)):
#         theta_p_plus = theta_p.copy()
#         theta_p_minus = theta_p.copy()
        
#         theta_p_plus[i] += epsilon
#         theta_p_minus[i] -= epsilon
        
#         cost_plus = cost_function(theta_p_plus, *args)
#         cost_minus = cost_function(theta_p_minus, *args)
        
#         grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
    
#     return grad

# Iterate over each time step to optimize the trajectory
for k in range(len(time) - predictionhorizon):
    # Initialize predicted state for each step
    state_p_init = np.zeros((predictionhorizon, 2), dtype=np.float32)
    state_p_init[0] = path_traversed[k]
    
    # Perform the optimization with bounds to prevent extreme values
    bounds = [(-np.pi, np.pi)] * (predictionhorizon - 1)  # Bound theta_p to reasonable values for angles
    
    result = minimize(cost, theta_p, args=(state_p_init, time, data, k), method='Nelder-Mead', 
                  options={'disp': True})

    # Check if the optimization was successful
    if result.success:
        theta_p = result.x
    else:
        print(f"Optimization failed at step {k}: {result.message}")
    
    # Debugging: print optimized theta_p and cost at each step
    #print(f"Step {k}, Optimized theta_p: {theta_p}, Cost: {result.fun}")
    
    # Compute and print numerical gradient for debugging
    #grad = numerical_gradient(cost, theta_p, state_p_init, time, data, k)
    #print(f"Step {k}, Gradient: {grad}")
    
    # Update the traversed path using the first action (theta_p[0])
    deltat = time[k + 1] - time[k]
    path_traversed[k + 1] = path_traversed[k] + np.array([v * deltat * np.cos(theta_p[0]), 
                                                           v * deltat * np.sin(theta_p[0])])

# Plot the paths
plt.plot(data[:, 0], data[:, 1], label='Desired Path')
plt.plot(path_traversed[:, 0], path_traversed[:, 1], label='Traversed Path')
plt.legend()
plt.show()
