import pickle
import pystorms
import numpy as np
from GPyOpt.methods import BayesianOptimization


# Define the performance
def f_loss(x):
    env = pystorms.scenarios.theta()
    done = False

    actions = x.flatten()
    while not done:
        done = env.step(actions)

    loss = env.performance()
    return loss


# Set the random seed
np.random.seed(40)


# Define gpopt
domain = [
    {"name": "var_1", "type": "continuous", "domain": (0.0, 0.10)},
    {"name": "var_2", "type": "continuous", "domain": (0.0, 0.10)},
]

myBopt = BayesianOptimization(
    f=f_loss,
    domain=domain,
    model_type="GP",
    acquisition_type="EI",
    initial_design_numdata=0,
)

myBopt.__dict__.update(pickle.load(open("./ckpts1/Bayopt50", "rb")))
myBopt._compute_results()

valve_positions = myBopt.x_opt

env = pystorms.scenarios.theta()
done = False

env.data_log["depthN"] = {}
for i in ["P1", "P2"]:
    env.data_log["depthN"][i] = []

while not done:
    done = env.step(valve_positions)

np.save("./controlled_performance", env.performance())
np.save("./actions", valve_positions)
np.save("./controlled_response.npy", env.data_log)
