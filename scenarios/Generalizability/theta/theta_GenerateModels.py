import pystorms
import numpy as np
from GPyOpt.methods import BayesianOptimization


# Define the performance
def f_loss(x):
    env = pystorms.scenarios.theta()
    done = False

    actions = x.flatten()
    print(actions)
    while not done:
        done = env.step(actions)

    loss = env.performance()
    return loss


# Set the random seed
np.random.seed(1)


# Define gpopt
domain = [
    {"name": "var_1", "type": "continuous", "domain": (0.0, 0.10)},
    {"name": "var_2", "type": "continuous", "domain": (0.0, 0.10)},
]

myBopt = BayesianOptimization(
    f=f_loss,
    domain=domain,
    model_type="GP",
    acquisition_type="EI"
)

log = myBopt.run_optimization(
    "./ckpts1/",
    max_iter=50,
    save_inter_models=True,
    intervals=10
)
