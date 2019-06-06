# Plotting Libraries and default stuff
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("whitegrid")

# Bayesian Optimization
from GPyOpt.methods import BayesianOptimization
import tslearn.metrics as ts

from scenarios import AASinglePond
import pickle

# Set the random seed 
np.random.seed(42)

# Deep copy
import copy

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
    outflow, overflow1, depth= AASinglePond(x[0][0])
    loss = 0.0
    for i,j in zip(outflow, depth):
        if i > 4:
            loss += 10
        else:
            loss += 0
        loss += j
    loss += (overflow1)*100.0
    return loss

# Plot the convergence based on gpyopt 
def plot_acquisition(axis, model):
    bounds = model.acquisition.space.get_bounds()

    x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
    x_grid = x_grid.reshape(len(x_grid),1)
    acqu = model.acquisition.acquisition_function(x_grid)
    acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
    m, v = model.model.predict(x_grid)


    #model.plot_density(bounds[0], alpha=.5)

    axis.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
    axis.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
    axis.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

    axis.plot(model.X, model.Y, 'b.', markersize=10)
    
    factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))
    axis.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor,
            'r-',lw=2,label ='Acquisition (arbitrary units)')

    axis.set_xlabel('Valve Setting')
    axis.set_ylabel('Objective')
    axis.set_ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
    #axis.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
    #axis.legend(loc='upper left')


# Define gpopt
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.0, 1.0)}]

visual_iteration = [10, 30, 60, 90]

log = {}


myBopt = BayesianOptimization(f=f_loss,
    domain=domain,
    model_type = 'GP',
    acquisition_type='EI',
    acquisition_weight = 1)

myBopt.run_optimization(max_iter=100, logger=log, save_inter_models=True, intervals=1, verbosity=True)
# Store the instances at the iter for all the recon ! 


bay_opt_container = {}
for i in log.keys():
    l = []
    temp_myBopt = BayesianOptimization(f=f_loss,
            domain=domain,
            model_type = 'GP',
            acquisition_type='EI',
            acquisition_weight = 1)

    temp_myBopt.__dict__.update(pickle.load(open(log[i], 'rb')))
    temp_myBopt._compute_results()
    bay_opt_container[i] = temp_myBopt


fig = plt.figure()
# Subplot 1 - Performance 
ax1 = fig.add_subplot(3,2,1)

un_flow,_,_ = AASinglePond(1.0)
ax1.plot(un_flow)
label = ["Uncontrolled"]
f = []
for key in visual_iteration:
    flow, _, _ = AASinglePond(bay_opt_container[str(key)].x_opt[0])
    f.append(flow)
    ax1.plot(f[-1])
    label.append(str(key))
ax1.legend(label)
ax1.set_ylabel("Outflow")

# Subplot 2 - Performance with keys
ax2 = fig.add_subplot(3,2,2)
loss = []
loss_index = []
for key in log.keys():
    loss.append(f_loss([bay_opt_container[str(key)].x_opt]))
    loss_index.append(float(key))
ax2.plot(loss_index, loss)
ax2.set_ylabel("Performance")
ax2.set_xlabel("Iteration")


counter = 3
for i in visual_iteration:
    ax = fig.add_subplot(3,2,counter) 
    plot_acquisition(ax,bay_opt_container[str(i)])
    ax.set_title(str(i))
    counter+=1

plt.show()
