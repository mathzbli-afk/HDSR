import os, sys, subprocess
import numpy as np
import math
import time
import matlab.engine
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import emcee

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Set console encoding (Windows compatibility)
if os.name == "nt":
    subprocess.call("chcp 65001 > nul", shell=True)
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# MCMC sample generator
def generate_mcmc_samples(num_samples, dim, bounds):
    """
    Parameters:
        num_samples : number of samples to generate
        dim : parameter dimension (number of decision variables)
        bounds : parameter boundary list, e.g. [(min,max)]*dim
    """
    # Define log-probability function (MCMC core)
    def log_prob(params):
        # Check if all parameters are within bounds
        for i in range(dim):
            if not bounds[i][0] < params[i] < bounds[i][1]:
                return -np.inf  # Out of bounds returns -inf (unacceptable)
        return 0.0  # Within bounds returns uniform log-probability (constant)

    # MCMC parameter settings
    num_walkers = dim * 2  # Recommended walkers = 2*dim, ensures sampling efficiency
    # Generate initial positions: each walker randomly initialized within parameter bounds
    initial_positions = [
        np.random.uniform(low=[b[0] for b in bounds],
                          high=[b[1] for b in bounds])
        for _ in range(num_walkers)
    ]
    # Create EnsembleSampler instance
    sampler = emcee.EnsembleSampler(num_walkers, dim, log_prob)  # walker number, dimension, probability function
    sampler.run_mcmc(initial_positions, 100, progress=False)  # Run MCMC chain: 100 steps, no progress bar
    flat_samples = sampler.get_chain(discard=50, flat=True)  # Get processed chain: discard first 50 (burn-in), flatten
    return flat_samples[:num_samples]  # Return required number of samples


def objective_function(X):
    """ Objective function: compute SNRI value of SR-4D system via MATLAB """
    # Convert input to 1×N matrix (avoid batch-processing error)
    X = X.reshape(1, -1)  # Prevent batch being treated as multiple rows
    a = X[:, 0]
    b = X[:, 1]
    c = X[:, 2]
    d = X[:, 3]
    e = X[:, 4]
    f = X[:, 5]
    g = X[:, 6]
    h = X[:, 7]
    # Build MATLAB-compatible input (1×8 array)
    input_array = [[a, b, c, d, e, f, g, h]]
    # Call MATLAB function sr4d_eval to compute system performance
    # nargout=1 means only return 1 output value
    result = eng.sr4d_eval(matlab.double(input_array),False, nargout=1)
    fr = float(result)  # Convert MATLAB output to Python float
    return -fr  # Return negative because Bayesian optimization minimizes (maximize SNRI)


# Parameter ranges
param_bounds = [
    (18, 22),    # a
    (-10, -8),   # b
    (2, 8),      # c
    (-5, 0),     # d
    (2, 8),      # e
    (-8, -2),    # f
    (78, 82),    # g
    (5, 15)      # h
]
dim = len(param_bounds)
initial_sample_num = 50  # Number of initial samples

## Build initial sample set: combine empirical parameters + MCMC sampling
manual_X = np.array([
    [18, -10, 5, -3, 3, -5, 80, 15],
    # adding High-quality empirical parameters without theoretical support may guide BOA to obtain better SNRI
])
manual_num = manual_X.shape[0]
mcmc_num = initial_sample_num - manual_num  # Number of MCMC samples
# Call MCMC function to generate MCMC samples (uniformly distributed over parameter space)
mcmc_X = generate_mcmc_samples(mcmc_num, dim, param_bounds)
initial_X = np.vstack([manual_X, mcmc_X])  # Merge all initial samples


# GPyOpt formatted bounds # domain is wrapped in dict form for GPy
bounds_gpyopt = [
    {'name': f'param_{i}', 'type': 'continuous', 'domain': b}
    for i, b in enumerate(param_bounds)
]

print("Number of initial sample points:", initial_X.shape[0])
print("Initial sample results:")
# Bayesian optimizer initialization (LCB strategy)
optimizer = BayesianOptimization(
    f = objective_function,  # Optimization objective function
    domain = bounds_gpyopt,  # Parameter domain definition
    model_type = 'GP',       # Use Gaussian Process model
    acquisition_type = 'LCB',  # Acquisition function: Lower Confidence Bound
    acquisition_weight = 0.35,  # Conservative # LCB balance parameter (exploration/exploitation trade-off)
    X = initial_X,           # Initial sample points
    normalize_Y = False,     # Do not normalize outputs (keep original scale)
    de_duplication = True,   # Enable duplicate detection
    initial_design_type = None  # Disable built-in initialization (samples already provided)
)
# Log
log_path = "boa_log_sr4d.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("Bayesian Optimization Log for SR-4D\n")
    f.write("Iteration\tParam a\tParam b\tParam c\tParam d\tParam e\tParam f\tParam g\tParam h\tSNRI Value\n")

# Run optimization
max_iterations = 200  # Maximum iterations
start_time = time.time()  # Record start time

for i in range(max_iterations):
    optimizer.run_optimization(1, verbosity=False)  # Run 1 step of Bayesian optimization (no verbose output)
    X_history, Y_history = optimizer.get_evaluations()  # Get all evaluation history
    X_last = X_history[-1]
    Y_last = -Y_history[-1]  # Latest output (negated back to original SNRI)

    print(f"\n===== Iteration {i+1} =====")
    print(f"Current evaluation point X = {X_last}")
    print(f"Current output Y:SNRI(dB)  = {Y_last[0]:.6f}")
    print("----------------------------------------")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{i+1}\t" + "\t".join([f"{val:.6f}" for val in X_last]) + f"\t{Y_last[0]:.6f}\n")


# Optimal parameter performance evaluation
best_params = optimizer.x_opt  # Get global optimal parameters from optimizer

print("\n==== Global optimal parameter result output ====")
print("Optimal parameters X:")
for j, name in enumerate(['a','b','c','d','e','f','g','h']):
    print(f"  {name} = {best_params[j]:.6f}")
print(f"Optimal objective function value Y:SNRI(dB) = {-optimizer.fx_opt:.6f}")
print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

# Save final results to log
with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n==== Global optimal parameters ====\n")
    for j, name in enumerate(['a','b','c','d','e','f','g','h']):
        f.write(f"{name} = {best_params[j]:.6f}\n")
    f.write(f"Optimal SNRI = {-optimizer.fx_opt:.6f}\n")
    f.write(f"Total time: {time.time() - start_time:.2f} seconds\n")

# Keep Python running so MATLAB figures remain open
input("\nPress Enter to close after viewing MATLAB figures...")

# Close all MATLAB figures after receiving your instruction
eng.sr4d_eval("close all", nargout=0)
