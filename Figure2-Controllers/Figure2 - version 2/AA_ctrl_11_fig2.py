import numpy as np
import pandas as pd
from pyswmm_lite import Env
import tslearn.metrics as ts
import scipy.signal as spy
import scipy
import pickle
from automate_objective import generate_targets, generate_weights
from utilities_baeopt import swmm_execute
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import requests

def notify_me(notification):
    headers = {
        'Content-type': 'application/json',
    }

    data = {"text":notification}

    response = requests.post('https://hooks.slack.com/services/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', headers=headers, data=str(data))
    return response.content

# Metadata for the nodes and orifices 
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
ctrl_elements = NODES_LIS.keys()

data_uncontrolled = np.load("./uncontrolled_flow_AA.npy").item()
inflows_uncontrolled = data_uncontrolled["inflows"]
flows_uncontrolled = data_uncontrolled["outflows"]


# Set the threshold
THRESHOLD = 5.0 

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
    print(x)
    # Decimate the points under consideration and compute the DTW distance
    obj_value = 0.0
    for key in ctrl_elements:
        temp_response = spy.resample_poly(data["outflows"][key], 2, 1500)
        path, dist =  ts.dtw_path(temp_response, ctrl_targets[key][:len(temp_response)],
                global_constraint="sakoe_chiba", sakoe_chiba_radius=1)

        obj_value += ctrl_weights[key] * dist
        # check for flooding
        flood = data["flooding"][key]
        flood = flood.gt(0.0)
        if flood.any():
            obj_value += 1000
    print(obj_value)
    return obj_value

# Define gpopt
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_2', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_3', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_4', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_5', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_6', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_7', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_8', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_9', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_10', 'type': 'continuous', 'domain': (0.0, 1.0)},
          {'name': 'var_11', 'type': 'continuous', 'domain': (0.0, 1.0)}]


log = {}
# --- Solve your problem
myBopt = BayesianOptimization(f=objective_function,
        domain=domain,
        model_type = 'GP',
        acquisition_type='EI',
        acquisition_weight = 11)

myBopt.run_optimization(max_iter=100, logger=log, save_inter_models=True, intervals=2, verbosity=True)
notify_me("5 Ponds Done")
