import numpy as np
from pystorms.scenarios import epsilon
from GPyOpt.methods import BayesianOptimization


def objective_function(x):
    env = epsilon()
    done = False
    # Update simulation time
    while not done:
        done = env.step(x.flatten())
    return env.performance()


# Set the random seed
np.random.seed(30)

# Create the domain
domain = []
for i in range(1, 12):
    domain.append({"name": "var_" + str(i), "type": "continuous", "domain": (0.0, 1.0)})


myBopt = BayesianOptimization(
    f=objective_function, domain=domain, model_type="GP", acquisition_type="EI"
)

log = myBopt.run_optimization(
    "./ckpts/", max_iter=100, save_inter_models=True, intervals=5, eps=0.10
)
