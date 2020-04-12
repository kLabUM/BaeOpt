import baestorm
import numpy as np
import pandas as pd
from pyswmm_lite import environment


# Generate Gaussian Flows
def gaussian_flows(x, mu=0.0, sigma=1.0):
    y = (1.0 / (sigma * (np.sqrt(2.0 * np.pi)))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
    return y


def single_basin(actions, flows):
    env = environment(baestorm.load_networks("parallel"), False)
    data = {}
    data["outflow"] = []
    data["overflow"] = []
    data["depth"] = []
    # pad inflows, so that the systems receds
    flows = np.append(flows, np.zeros(300))
    for time in range(0, len(flows)):
        # set the gate_position
        env._setValvePosition("1", actions)
        # set inflow
        env.sim._model.setNodeInflow("P1", 3 * flows[time])
        env.sim._model.setNodeInflow("P2", 0.0)
        # record_data
        data["outflow"].append(env._getLinkFlow("1"))
        data["overflow"].append(env.sim._model.getNodeResult("P1", 4))
        data["depth"].append(env._getNodeDepth("P1"))
        # step through simulation
        _ = env.step()
    env._terminate()
    return data["outflow"], sum(data["overflow"]), data["depth"]


# Objective function
def objective_function(x, inflows):
    valves = x[0]
    # Sample a random inflow
    temp_1 = np.random.choice(np.linspace(0, 9, 10, dtype=int))
    # Simulate and generate flow transformations
    flows, overflows, depth = single_basin(valves, inflows[str(temp_1)].values)
    # If completely closed, then penalize on depth
    threshold = 0.50
    # IF flows exceed threhold, penalize based on the diviation
    flows = pd.Series(flows)
    flows_sub = flows.sub(threshold)
    flows_sub[flows_sub < 0.0] = 0.0
    if flows.sum() == 0.0:
        flow_ration = 0.0
    else:
        flow_ration = flows_sub.sum() / flows.sum()
    reward = (
        np.exp(overflows / inflows[str(temp_1)].sum() + flow_ration + depth[-1] / 2.0)
        - 1.0
    )
    return reward


# Create an objective function
class Objective:
    def __init__(self):
        means = [0.0]
        sigma = np.linspace(2.0, 5.0, 10)
        scale = 5.0

        inflows = {}
        count = 0
        for mu in means:
            for sig in sigma:
                inflows[str(count)] = scale * gaussian_flows(
                    np.linspace(-10.0, 10.0, 100), mu, sig
                )
                count += 1
        self.inflows = pd.DataFrame.from_dict(inflows)

    def f(self, x):
        x = x.flatten()
        reward = objective_function(x, self.inflows)
        return reward


def plot_acquisition(axis, model):
    bounds = model.acquisition.space.get_bounds()

    x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
    x_grid = x_grid.reshape(len(x_grid), 1)
    # acqu = model.acquisition.acquisition_function(x_grid)
    # acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
    m, v = model.model.predict(x_grid)
    factor = max(m + 1.96 * np.sqrt(v)) - min(m - 1.96 * np.sqrt(v))
    axis.plot(x_grid, m, color="#D5313E", lw=2.0)
    axis.plot(x_grid, m - 1.96 * np.sqrt(v), color="#445792")
    axis.plot(x_grid, m + 1.96 * np.sqrt(v), color="#445792")
    axis.fill_between(
        np.ravel(x_grid),
        np.ravel(m - 1.96 * np.sqrt(v)),
        np.ravel(m + 1.96 * np.sqrt(v)),
        color="#445792",
        alpha=0.5,
    )
    y = model.Y - model.Y.mean()
    y = y / model.Y.std()
    axis.scatter(model.X[:-1], y[:-1], color="#9F383E", marker="o")

    axis.set_xlabel("Valve Setting")
    axis.set_ylabel("Objective")
    axis.set_ylim(
        min(m - 1.96 * np.sqrt(v)) - 0.25 * factor,
        max(m + 1.96 * np.sqrt(v)) + 0.05 * factor,
    )
