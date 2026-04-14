# Big success!!! Rule: When setting BOA, switch the kernel function to LCB (more exploratory than EI),
# set acquisition_weight=0.5 + diverse initial points
import os, sys, subprocess
import numpy as np
import math
import time
import matlab.engine
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import emcee

# Set console encoding
if os.name == "nt":
    subprocess.call("chcp 65001 > nul", shell=True)
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# MCMC sample generation
def generate_mcmc_samples(num_samples, dim, bounds):
    def log_prob(params):
        for i in range(dim):
            if not bounds[i][0] < params[i] < bounds[i][1]:
                return -np.inf
        return 0.0

    num_walkers = dim * 2
    initial_positions = [
        np.random.uniform(low=[b[0] for b in bounds],
                          high=[b[1] for b in bounds])
        for _ in range(num_walkers)
    ]

    sampler = emcee.EnsembleSampler(num_walkers, dim, log_prob)
    sampler.run_mcmc(initial_positions, 100, progress=False)
    flat_samples = sampler.get_chain(discard=50, flat=True)
    return flat_samples[:num_samples]

# Objective function (with outlier filtering)
def objective_function(X):
    a = X[:, 0]
    b = X[:, 1]
    input_array = [[a, b]]
    result = eng.sr1d_eval(matlab.double(input_array), False, nargout=1)
    fr = float(result)
    return -fr

# Parameter space
param_bounds = [(1e-6, 4), (1e-6, 4)]
initial_sample_num = 200

# Initial points (manual + MCMC)
manual_X = np.array([
    [1.0, 1.0],
    # adding High-quality empirical parameters without theoretical support may guide BOA to obtain better SNRI
])
manual_num = manual_X.shape[0]
mcmc_num = initial_sample_num - manual_num
mcmc_X = generate_mcmc_samples(mcmc_num, 2, param_bounds)
initial_X = np.vstack([manual_X, mcmc_X])

# GPyOpt format
gpy_bounds = [
    {'name': 'a', 'type': 'continuous', 'domain': param_bounds[0]},
    {'name': 'b', 'type': 'continuous', 'domain': param_bounds[1]}
]

print("Number of initial sample points:", initial_X.shape[0])

# Optimizer initialization (LCB strategy)
optimizer = BayesianOptimization(
    f=objective_function, # Optimization objective function
    domain=gpy_bounds,    # Parameter domain definition
    model_type='GP',      # Use Gaussian Process model
    acquisition_type='LCB', # Acquisition function: Lower Confidence Bound
    acquisition_weight=0.35, # Conservative # Balance parameter of LCB (exploration/exploitation tradeoff)
    X=initial_X,          # Initial sample points
    normalize_Y=False,    # Do not normalize output values (keep original scale)
    de_duplication=True,  # Enable duplicate detection
    initial_design_type=None # Disable built-in initialization (since samples are provided)
)

# Log settings
log_path = "boa_log_sr1d.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("Bayesian Optimization Log for SR-1D\n")
    f.write("Iteration\tParameter a\tParameter b\tSNRI Value\n")

# Run optimization
max_iterations = 10
start_time = time.time()

for i in range(max_iterations):
    optimizer.run_optimization(1, verbosity=False)
    X_history, Y_history = optimizer.get_evaluations()
    X_last = X_history[-1]
    Y_last = -Y_history[-1]
    print(f"\n===== Iteration {i+1} =====")
    print(f"Current evaluation point X = {X_last}")
    print(f"Current output Y:SNRI(dB)  = {Y_last[0]:.6f}")
    print("----------------------------------------")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{i+1}\t{X_last[0]:.6f}\t{X_last[1]:.6f}\t{Y_last[0]:.6f}\n")

best_a, best_b = optimizer.x_opt[0], optimizer.x_opt[1]
print("\n==== Global optimal parameter result output ====")
print(f"Optimal parameters X: a = {best_a:.6f}, b = {best_b:.6f}")
print(f"Optimal objective function value Y:SNRI(dB) = {-optimizer.fx_opt:.6f}")
print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

# Save final results to log
with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n==== Global optimal parameters ====\n")
    f.write(f"Optimal parameter a = {best_a:.6f}, b = {best_b:.6f}\n")
    f.write(f"Optimal SNRI(dB)= {-optimizer.fx_opt:.6f}\n")
    f.write(f"Total time: {time.time() - start_time:.2f} seconds\n")

# Keep Python running so MATLAB figures remain open
input("\nPress Enter to close after viewing MATLAB figures...")

# Close all MATLAB figures after receiving your instruction
eng.sr1d_eval("close all", nargout=0)
