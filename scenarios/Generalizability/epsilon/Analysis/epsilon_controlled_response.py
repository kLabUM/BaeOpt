import numpy as np
import pickle
from pystorms.scenarios import epsilon
from GPyOpt.methods import BayesianOptimization


def f_loss(x):
    env = epsilon()
    done = False

    while not done:
        done = env.step(x.flatten())

    return env.performance()


# Create the domain
domain = []
for i in range(1, 12):
    domain.append({"name": "var_" + str(i), "type": "continuous", "domain": (0.0, 1.0)})

myBopt = BayesianOptimization(
    f=f_loss,
    domain=domain,
    model_type="GP",
    acquisition_type="EI",
    initial_design_numdata=0,
)

myBopt.__dict__.update(pickle.load(open("./151_100_GeneralizabilityEpsilon/Bayopt100", "rb")))
myBopt._compute_results()

valve_positions = myBopt.x_opt

env = epsilon()
done = False
while not done:
    done = env.step(valve_positions)

np.save("./controlled_performance", env.performance())
np.save("./actions", valve_positions)
np.save("./controlled_response.npy", env.data_log)
