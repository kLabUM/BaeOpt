import os
import argparse
import numpy as np
from pystorms.scenarios import epsilon
from GPyOpt.methods import BayesianOptimization


# Objective function for scenaio epsilon
def f_loss(x):
    # Flatten the actions for the pystorms
    x = x.flatten()

    # Inialize the scenario
    env = epsilon()
    done = False

    # Simulate
    while not done:
        done = env.step(x)

    # Query the estimated loss
    loss = env.performance()

    return loss


# Argument parser for batach run
parser = argparse.ArgumentParser(
    description="Generalizability Test Bayesian Optimization Epslion"
)
parser.add_argument("-seed", nargs=1, type=int, help="Random seed for the solver")
parser.add_argument("-iters", nargs=1, type=int, help="Number of acquisitions")
args = parser.parse_args()

# Read the parsed args
random_seed = args.seed[0]
number_iter = args.iters[0]

# Set the random seed
np.random.seed(random_seed)

# Set path to save model runs
save_path = (
    "./" + str(random_seed) + "_" + str(number_iter) + "_GeneralizabilityEpsilon"
)
os.mkdir(save_path)

# Create the domain
domain = []
for i in range(1, 12):
    domain.append({"name": "var_" + str(i), "type": "continuous", "domain": (0.0, 1.0)})


myBopt = BayesianOptimization(
    f=f_loss, domain=domain, model_type="GP", acquisition_type="EI",
)

myBopt.run_optimization(
    save_path,
    report_file=save_path + "_report.txt",
    max_iter=number_iter,
    save_inter_models=True,
    intervals=25,
    verbosity=True,
    eps=0.005,
)
