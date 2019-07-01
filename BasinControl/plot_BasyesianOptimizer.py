import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from GPyOpt.methods import BayesianOptimization
import pickle


# Plot the convergence based on gpyopt
def plot_acquisition(axis, model):
    bounds = model.acquisition.space.get_bounds()

    x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
    x_grid = x_grid.reshape(len(x_grid), 1)
    acqu = model.acquisition.acquisition_function(x_grid)
    acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
    m, v = model.model.predict(x_grid)

    axis.plot(x_grid, m, "k-", lw=1, alpha=0.6)
    axis.plot(x_grid, m - 1.96 * np.sqrt(v), "k-", alpha=0.2)
    axis.plot(x_grid, m + 1.96 * np.sqrt(v), "k-", alpha=0.2)

    axis.plot(model.X, model.Y, "g.", markersize=10)

    factor = max(m + 1.96 * np.sqrt(v)) - min(m - 1.96 * np.sqrt(v))
    axis.plot(
        x_grid,
        0.2 * factor * acqu_normalized
        - abs(min(m - 1.96 * np.sqrt(v)))
        - 0.25 * factor,
        "r-",
        lw=2,
        label="Acquisition (arbitrary units)",
    )

    axis.set_xlabel("Valve Setting")
    axis.set_ylabel("Objective")
    axis.set_ylim(
        min(m - 1.96 * np.sqrt(v)) - 0.25 * factor,
        max(m + 1.96 * np.sqrt(v)) + 0.05 * factor,
    )


iteration_numbers = [1, 5, 10, 25]
domain = [{"name": "var_1", "type": "continuous", "domain": (0.0, 1.0)}]
# Store the instances at the iter for all the recon !
f_loss = lambda x: x + 0.0

bay_opt_container = {}
for i in iteration_numbers:
    temp_myBopt = BayesianOptimization(
        f=f_loss,
        domain=domain,
        model_type="GP",
        acquisition_type="EI",
        acquisition_weight=1,
    )

    temp_myBopt.__dict__.update(
        pickle.load(open("./model_checkpoints/Bayopt" + str(i), "rb"))
    )
    temp_myBopt._compute_results()
    bay_opt_container[str(i)] = temp_myBopt

sns.set_style("whitegrid")
fig = plt.figure()
counter = 1
for i in iteration_numbers:
    ax = fig.add_subplot(2, 2, counter)
    plot_acquisition(ax, bay_opt_container[str(i)])
    ax.set_title(str(i))
    counter += 1

plt.show()
