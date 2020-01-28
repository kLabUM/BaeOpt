import numpy as np
import pandas as pd
from baestorm.scenarios import GammaSinglePond
from GPyOpt.methods import BayesianOptimization


# Define the performance
def f_loss(x):
    """
    Objective to be optimized by the bayesian optimizer

    Params:
    ---------------------------------------------------
    x     : (ndarray) valve positions

    Returns:
    ---------------------------------------------------
    loss  : (float) performance metric
    """
    data = GammaSinglePond(x[0][0])

    # Covert to pandas dataframes
    depth = pd.Series(data["depthN"]["1"])
    flows = pd.DataFrame.from_dict(data["flow"]["O1"])
    flooding = pd.DataFrame.from_dict(data["flooding"]["1"])

    loss = 0.0
    # Check if there is flooding in the assets
    flooding = flooding.gt(0.0)
    flooding = flooding.any()
    if flooding.any():
        loss += np.inf

    flows = flows.sub(4.0)
    flows[flows < 0.0] = 0.0
    loss += flows.sum(axis=0)
    if depth.values[-1] > 0.10:
        loss += depth.values[-1] * 10000.0
    else:
        loss += 0.0
    print(loss)
    return loss


# Set the random seed
np.random.seed(42)


# Define gpopt
domain = [{"name": "var_1", "type": "continuous", "domain": (0.0, 1.0)}]

myBopt = BayesianOptimization(
    f=f_loss,
    domain=domain,
    model_type="GP",
    acquisition_type="EI",
    acquisition_weight=1,
    initial_design_numdata=1,
)

log = myBopt.run_optimization(
    "./ckpts1/", max_iter=50, save_inter_models=True, intervals=1
)
