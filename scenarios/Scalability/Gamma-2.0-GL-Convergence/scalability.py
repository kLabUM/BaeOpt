import numpy as np
import pandas as pd
from GPyOpt.methods import BayesianOptimization
from pystorms.scenarios import gamma
import os
import argparse


# Create gamma scenario
def GammaData(actions):
    env = gamma()
    done = False
    # Modify the logger function to store depths
    env.data_log["depthN"] = {}
    for i in np.linspace(1, len(actions), len(actions), dtype=int):
        env.data_log["depthN"][str(i)] = []
    for i in np.linspace(len(actions) + 1, 11, 11 - len(actions), dtype=int):
        del env.data_log["flow"]["O" + str(i)]
        del env.data_log["flooding"][str(i)]
    while not done:
        done = env.step(actions)
    return env.data_log


def f_loss(x):
    data = GammaData(x[0])

    # Convert to pandas dataframes
    depths = pd.DataFrame.from_dict(data["depthN"])
    flows = pd.DataFrame.from_dict(data["flow"])
    flooding = pd.DataFrame.from_dict(data["flooding"])

    loss = 0.0
    flooding = flooding.gt(0.0)
    flooding = flooding.any()
    if flooding.any():
        loss += 10 ** 5

    flows = flows.sub(4.0)
    flows[flows < 0.0] = 0.0
    loss += flows.sum().sum()

    for i in depths.values[-1]:
        if i > 0.1:
            loss += i * 10 ** 4
    return loss


# Argument parser for batach run
parser = argparse.ArgumentParser(description="Scalability Test Bayesian Optimization")
parser.add_argument("-seed", nargs=1, type=int, help="Random seed for the solver")
parser.add_argument("-ctrl", nargs=1, type=int, help="Number of control points")
parser.add_argument("-iters", nargs=1, type=int, help="Number of acquisitions")
args = parser.parse_args()

random_seed = args.seed[0]
ctrl_elements = args.ctrl[0]
number_iter = args.iters[0]

# Set the random seed
np.random.seed(random_seed)
save_path = "./" + str(random_seed) + "_" + str(ctrl_elements) + "_Scalability"
os.mkdir(save_path)
# Create the domain
domain = []
for i in range(1, ctrl_elements + 1):
    domain.append({"name": "var_" + str(i), "type": "continuous", "domain": (0.0, 1.0)})

# Solve the bayesian optimization
myBopt = BayesianOptimization(
    f=f_loss, domain=domain, model_type="GP", acquisition_type="EI"
)

myBopt.run_optimization(
    save_path,
    report_file=save_path + "_report.txt",
    max_iter=number_iter,
    save_inter_models=True,
    intervals=200,
    verbosity=True,
    eps=10**-3,
)
