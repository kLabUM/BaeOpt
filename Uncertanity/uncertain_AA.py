import numpy as np
import scipy as spy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyswmm_lite import Env
import pickle
from automate_objective import generate_targets, generate_weights
from utilities_baeopt import swmm_execute, uncontrolled_response
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
import requests
np.random.seed(42)
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

# Test the response of the ensemble. 
#uncontrolled_responses = {}
#ensemble = np.linspace(1, 20, 20, dtype=int)
#for i in ensemble:
#    env = Env("./raindata/"+str(i)+"runfile.inp")
#    uncontrolled_responses[str(i)] = uncontrolled_response(env, NODES_LIS, save=False, destiation = "./"+str(i)+".npy")

def performance_metric(data, ctrl_weights, threshold=5.0, verbose=False):
    """
    Computes performance metrics for the optimization 
    based on the threshold. 

    Parameters:
    --------------------------------------------------
    data: dict of pandas dataframes with outflows and flooding

    Returns:
    --------------------------------------------------
    performance_metric : pefromance of the objetive 
    """
    outflows = data["outflows"]
    flooding = data["flooding"]
    # Get the performance metric 
    perforamce  = 0.0
    # Check if there is flooding in the assets 
    flooding = flooding.gt(0.0)
    flooding = flooding.any()
    if flooding.any(): 
        if verbose:
            # TODO print the quantity of flooding
            print("Nodes with flooding: {}".format(flooding[flooding == True].index))
        perforamce += 10**8
    # Estimate the perfomance from the flows
    outflows = outflows.sub(threshold)
    outflows[outflows < 0.0] = 0.0
    outflows = outflows.sum()
    for i in outflows.keys():
        perforamce += outflows[i] * ctrl_weights[i]
    return perforamce

# Set up the objective function
data_uncontrolled = np.load("./uncontrolled_flow_AA.npy").item()
inflows_uncontrolled = data_uncontrolled["inflows"]
# Control weights 
ctrl_weights = generate_weights(inflows_uncontrolled)

# Set the threshold
THRESHOLD = 5.0 
ctrl_elements = NODES_LIS.keys()
number_of_ctrl_elements = 11 
save_path = "./uncertain/"
def objective_function(x):
    # Run SWMM and get the trajectories 
    # Pick a random snapshot 
    i = np.random.choice(np.linspace(1, 20, 20, dtype=int), 1)
    data = swmm_execute(x, ctrl_elements, network="./raindata/"+str(i[0])+"runfile.inp")
    obj_value = performance_metric(data, ctrl_weights, THRESHOLD)
    return obj_value

# Create the domain
domain = []
for i in range(1, number_of_ctrl_elements+1):
    domain.append({'name': 'var_'+str(i), 'type': 'continuous', 'domain': (0.0, 1.0)})

# Solve the bayesian optimization 
myBopt = BayesianOptimization(f=objective_function, domain=domain, model_type = 'GP', acquisition_type='EI')
verbo = True
myBopt.run_optimization(save_path, max_iter=500, save_inter_models=True, intervals=10, verbosity=verbo)

if verbo:
    print("Optimial Solution :", myBopt.x_opt)

