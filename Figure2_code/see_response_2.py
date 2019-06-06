import numpy as np
import pandas as pd
from pyswmm_lite import Env
import tslearn.metrics as ts
import scipy.signal as spy
import scipy
import pickle
from automate_objective import generate_targets, generate_weights
from utilities_baeopt import swmm_execute, performance_metric
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

NODES_LIS = {'93-49743' : 'OR39',
             '93-49868' : 'OR34',
             '93-49919' : 'OR44',
             '93-49921' : 'OR45',
             '93-50074' : 'OR38',
             '93-50076' : 'OR46',
             '93-50077' : 'OR48',
             '93-50081' : 'OR47',
             '93-50225' : 'OR36',
             '93-90357' : 'OR43',
             '93-90358' : 'OR35'}
ctrl_elements = ["93-50077", "93-50076"]


# Get the uncontrolled data
data_uncontrolled = np.load("./uncontrolled_flow_AA.npy").item()
flows_uncontrolled = data_uncontrolled["outflows"]
inflows_uncontrolled = data_uncontrolled["inflows"]

# Set the threshold
THRESHOLD = 4.5 

# Control targets
temp_targets = generate_targets(flows_uncontrolled, THRESHOLD)
ctrl_targets = {}
for key in ctrl_elements:
    ctrl_targets[key] = temp_targets[key]

# Control weights 
ctrl_weights = generate_weights(inflows_uncontrolled)

def objective_function(x):
    # Run SWMM and get the trajectories 
    data = swmm_execute(x, ctrl_elements)

    # Decimate the points under consideration and compute the DTW distance
    obj_value = 0.0
    for key in ctrl_elements:
        temp_response = spy.resample_poly(data["outflows"][key], 4, 1500)
        path, dist =  ts.dtw_path(temp_response, ctrl_targets[key][:len(temp_response)],
                global_constraint="sakoe_chiba", sakoe_chiba_radius=1)

        obj_value += ctrl_weights[key] * dist
        # check for flooding
        flood = data["flooding"][key]
        flood = flood.gt(0.0)
        if flood.any():
            obj_value += 1000
    return obj_value

# Define gpopt
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_2', 'type': 'continuous', 'domain': (0.0, 1.0)}]


temp_myBopt = BayesianOptimization(f=objective_function,
        domain=domain,
        model_type = 'GP',
        acquisition_type='EI',
        acquisition_weight = 1)

index = np.linspace(2, 100, 50, dtype=int)
pre = []
actions = []
for ind in index:
    temp_myBopt.__dict__.update(pickle.load(open("./2_pond_fig2/Bayopt"+str(ind)+".pickle", 'rb')))
    temp_myBopt._compute_results()
    data_controlled = swmm_execute(temp_myBopt.x_opt.reshape(1,2))
    perf = performance_metric(data_controlled, THRESHOLD)
    print(perf)
    pre.append(perf)
    actions.append(temp_myBopt.x_opt.reshape(1,2))

# Save the info
data_trained = {"iter":index, "performance":pre, "actions":actions}
np.save("./2_pond_fig2/performance_bayopt_2", data_trained)
