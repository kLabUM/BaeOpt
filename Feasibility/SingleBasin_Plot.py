import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../common")
from scenarios import AASinglePond
from GPyOpt.methods import BayesianOptimization


# Plot the convergence based on gpyopt
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
    outflow, overflow1, depth = AASinglePond(x[0][0])
    loss = 0.0
    for i, j in zip(outflow, depth):
        if i > 4:
            loss += 10
        else:
            loss += 0
        loss += j
    loss += (overflow1) * 100.0
    return loss


# Set the random seed
np.random.seed(42)

iteration_numbers = [2, 5, 10, 25, 100]
domain = [{"name": "var_1", "type": "continuous", "domain": (0.0, 1.0)}]
# Store the instances at the iter for all the recon !
bay_opt_container = {}
for i in iteration_numbers:
    temp_myBopt = BayesianOptimization(
        f=f_loss,
        domain=domain,
        model_type="GP",
        acquisition_type="EI",
        initial_design_numdata=0
    )

    temp_myBopt.__dict__.update(
        pickle.load(open("./ckpts/Bayopt" + str(i), "rb"))
    )
    temp_myBopt._compute_results()
    bay_opt_container[str(i)] = temp_myBopt

# Generate the responses for the stormwater network performance
iter_loss = [2, 10, 25, 100]
uc_loss = f_loss([[1.0]])
_temp = [bay_opt_container[str(i)].x_opt for i in iter_loss]
loss_perfomance = [f_loss(x.reshape(1, 1)) for x in _temp]
# Generate the stormwater flow responses
flows_perf = []
for i in iter_loss:
    outflow, _, _ = AASinglePond(bay_opt_container[str(i)].x_opt[0])
    flows_perf.append(outflow)

loss_perfomance.insert(0, uc_loss)
iter_loss.insert(0, 0)

# Uncontrolled response
uc_flow, _, _ = AASinglePond(1.0)

fig = plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
ax3 = plt.subplot2grid((3, 3), (2, 0))
ax4 = plt.subplot2grid((3, 3), (2, 1))
ax5 = plt.subplot2grid((3, 3), (2, 2))

ax1.plot(uc_flow, label="Uncontrolled")
ax1.plot(flows_perf[0], label="1")
ax1.plot(flows_perf[1], label="10")
ax1.plot(flows_perf[2], label="25")
ax1.plot(flows_perf[3], label="100")
ax1.axhline(4.0)
ax1.legend()

ax2.scatter(iter_loss, loss_perfomance)

plot_acquisition(ax3, bay_opt_container["5"])
plot_acquisition(ax4, bay_opt_container["25"])
plot_acquisition(ax5, bay_opt_container["100"])

plt.tight_layout()
plt.show()
