import numpy as np
import sys
sys.path.append("../common")
from GPyOpt.methods import BayesianOptimization
from scenarios import AASinglePond


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
    outflow, overflow, depth = AASinglePond(x[0][0])
    loss = 0.0
    for i, j in zip(outflow, depth):
        if i > 4:
            loss += 10
        else:
            loss += 0
        loss += j
    loss += (overflow) * 100.0
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
    initial_design_numdata=1
)

log = myBopt.run_optimization(
    "./ckpts1/",
    max_iter=100,
    save_inter_models=True,
    intervals=1,
    eps=0.01
    )