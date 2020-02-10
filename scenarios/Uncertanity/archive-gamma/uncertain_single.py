import numpy as np
import scipy as spy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyswmm_lite import Env
import pickle
from automate_objective import generate_targets, generate_weights
from utilities_baeopt import swmm_execute
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
NODES_LIS = {"P1":"1", "P2":"2"}

# Generate waves 
def generate_waves(window_len, cycles, magnitude = 1.0):
    half_wave = np.hstack((np.sin(np.linspace(0, 1.0, window_len)*np.pi),np.zeros([window_len]))) * magnitude
    flow = np.zeros([2*window_len*cycles])
    counter = 0
    for i in range(0, cycles):
        start_dim = counter
        end_dim = counter+(2*window_len)
        flow[start_dim:end_dim] = half_wave
        counter = counter+(2*window_len)
    return flow

def swmm_execute(x,
        inflow,
        ctrl_elemnts = ["P1"],
        network="./networks/parallel.inp",
        meta_data=NODES_LIS):
    """
    Run the controlled simuation

    Parameters:
    ------------------------------------------------
    x : control actions  
    network : path to network 
    meta_data : nodes list 

    Returns:
    ------------------------------------------------
    data: Dict of pandas dataframe with keys as outflow, depth, flooding
    and inflows 

    """
    env = Env(network)

    data = {}
    data["outflows"] = {}
    data["inflows"] = {}
    data["flooding"] = {}
    data["depth"] = {}

    for attribute in data.keys():
        for node in meta_data.keys():
            data[attribute][node] = []
    
    # Set the valve positions 
    valve_positions = []
    for i in range(0, x.shape[1]):
        valve_positions.append(x[0][i])


    done = False
    while not done:
        done = env.step()

        for valve,valve_position in zip(ctrl_elemnts, valve_positions):
            env.set_gate(NODES_LIS[valve], valve_position)

        # record the data
        for node in NODES_LIS.keys():
            data["outflows"][node].append(env.flow(NODES_LIS[node]))
            data["flooding"][node].append(env.get_flooding(node))
            data["depth"][node].append(env.depthN(node))
            data["inflows"][node].append(env.get_inflow(node))

    data["error"] = env.sim._model.runoff_routing_stats()
    env.terminate()
    
    # Convert to data frames 
    data["outflows"] = pd.DataFrame.from_dict(data["outflows"])
    data["inflows"]  = pd.DataFrame.from_dict(data["inflows"])
    data["flooding"] = pd.DataFrame.from_dict(data["flooding"])
    data["depth"] = pd.DataFrame.from_dict(data["depth"])

    return data

def performance_metric(data):
    # check for the performance 
    
