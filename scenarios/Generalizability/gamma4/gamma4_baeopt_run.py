import os
import argparse
import numpy as np
import pandas as pd
from pystorms.scenarios import gamma
from GPyOpt.methods import BayesianOptimization


# Gamma Scenario for the basins in series
def GammaData(actions, CTRL_ASSETS=4):

    # Initialize the scenario
    env = gamma()
    done = False

    # Modify the logger function to store depths
    env.data_log["depthN"] = {}
    for i in np.linspace(1, CTRL_ASSETS, CTRL_ASSETS, dtype=int):
        env.data_log["depthN"][str(i)] = []

    # Dont log the params from upstream, we donot care about them
    for i in np.linspace(CTRL_ASSETS + 1, 11, 11 - CTRL_ASSETS, dtype=int):
        del env.data_log["flow"]['O' + str(i)]
        del env.data_log["flooding"][str(i)]

    # Simulate controlled response
    while not done:
        done = env.step(actions)

    # Return the logged params
    return env.data_log


# Objective Function
def f_loss(x):
    # GypOpt uses 2d array
    # pystorms requies 1d array
    x = x.flatten()

    # Simulate the response of control actions
    data = GammaData(x)

    # Convert to pandas dataframes
    depths = pd.DataFrame.from_dict(data["depthN"])
    flows = pd.DataFrame.from_dict(data["flow"])
    flooding = pd.DataFrame.from_dict(data["flooding"])

    # Compute loss - check the performance metric equation in Readme
    loss = 0.0

    # Flooding loss
    for node in flooding.keys():
        if flooding[node].sum() > 0.0:
            loss += 10 ** 4 * flooding[node].sum()

    # Flow deviation
    flows = flows.sub(4.0)
    flows[flows < 0.0] = 0.0
    loss += flows.sum().sum()

    # Prevent basins from storing water at the end.
    for i in depths.values[-1]:
        if i > 0.1:
            loss += i * 10 ** 3

    return loss


# Argument parser for batach run
parser = argparse.ArgumentParser(
    description="Generalizability Test Bayesian Optimization Gamma4"
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
save_path = "./" + str(random_seed) + "_" + str(number_iter) + "_GeneralizabilityGamma4"
os.mkdir(save_path)


# Create the domain
domain = []
for i in range(1, 5):
    domain.append({"name": "var_" + str(i), "type": "continuous", "domain": (0.0, 1.0)})


myBopt = BayesianOptimization(
    f=f_loss, domain=domain,
    model_type="GP",
    acquisition_type="EI",
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
