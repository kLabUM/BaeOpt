import numpy as np
import pandas as pd
from pystorms.scenarios import gamma
from GPyOpt.methods import BayesianOptimization


def objective_function(x):
    env = gamma()
    done = False

    valves = np.ones(11)
    for i in range(0, 4):
        valves[0] = x[0][i]

    env.data_log["depthN"] = {}
    nodes = np.linspace(1, 11, 11, dtype=int)
    for i in nodes:
        env.data_log["depthN"][str(i)] = []

    while not done:
        done = env.step(valves)

    data = env.data_log

    depths = pd.DataFrame.from_dict(data["depthN"])
    flows = pd.DataFrame.from_dict(data["flow"])
    flooding = pd.DataFrame.from_dict(data["flooding"])

    loss = 0.0

    # drop all the flows, depths and flooding from upstream nodes
    for i in nodes[4:]:
        del depths[str(i)]
        del flows["O" + str(i)]
        del flooding[str(i)]

    # Check if there is flooding in the assets
    flooding = flooding.gt(0.0)
    flooding = flooding.any()
    if flooding.any():
        loss += np.inf

    # Check if flows are exceeding threshold
    flows = flows.sub(4.0)
    flows[flows < 0.0] = 0.0
    loss += flows.sum(axis=1).sum(axis=0)

    # go though all the depths
    for i in depths.keys():
        if depths[i].values[-1] > 0.10:
            loss += depths[i].values[-1] * 10000.0
        else:
            loss += 0.0

    return loss


# Set the random seed
np.random.seed(42)


# Define gpopt
domain = [
    {"name": "var_1", "type": "continuous", "domain": (0.0, 1.0)},
    {"name": "var_2", "type": "continuous", "domain": (0.0, 1.0)},
    {"name": "var_3", "type": "continuous", "domain": (0.0, 1.0)},
    {"name": "var_4", "type": "continuous", "domain": (0.0, 1.0)},
]

myBopt = BayesianOptimization(
    f=objective_function,
    domain=domain,
    model_type="GP",
    acquisition_type="EI",
    acquisition_weight=4,
    initial_design_numdata=4,
)

log = myBopt.run_optimization(
    "./ckpts2-test/",
    max_iter=250,
    save_inter_models=True,
    intervals=25,
    eps=0.10
)
