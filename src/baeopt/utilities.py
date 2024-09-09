from pystorms import environment
import numpy as np
import pandas as pd
import os
import re
import glob


def load_networks(network):
    path = os.path.abspath(os.path.dirname(__file__)) + "/networks/" + network + ".inp"
    return path


def meta_data_aa():
    """
    Returns the meta data for the stormwater network
    """
    NODES_LIS = {
        "93-49743": "OR39",
        "93-49868": "OR34",
        "93-49919": "OR44",
        "93-49921": "OR45",
        "93-50074": "OR38",
        "93-50076": "OR46",
        "93-50077": "OR48",
        "93-50081": "OR47",
        "93-50225": "OR36",
        "93-90357": "OR43",
        "93-90358": "OR35",
    }
    return NODES_LIS


def uncontrolled_response(
    env, NODES_LIS, save=True, destination="./uncontrolled_flow_AA.npy"
):
    """
    Run the uncontrolled simuation and present the
    save the run data as dict.

    Parameters:
    ------------------------------------------------
    env : pyswmm environment class
    NODES_LIS : Dictionary of nodes and corresponding valves
    Save : True or False

    Returns:
    ------------------------------------------------
    data: Dict of pandas dataframe with keys as outflow, depth, flooding
    and inflows

    """
    data = {}
    data["outflows"] = {}
    data["inflows"] = {}
    data["flooding"] = {}
    data["depth"] = {}

    for attribute in data.keys():
        for node in NODES_LIS.keys():
            data[attribute][node] = []

    done = False
    while not done:
        done = env.step()

        # record the data
        for node in NODES_LIS.keys():
            data["outflows"][node].append(env.methods["flow"](NODES_LIS[node]))
            data["flooding"][node].append(env.methods["flooding"](node))
            data["depth"][node].append(env.methods["depthN"](node))
            data["inflows"][node].append(env.methods["inflow"](node))

    data["error"] = env.sim._model.runoff_routing_stats()
    env._terminate()

    # Convert to data frames
    data["outflows"] = pd.DataFrame.from_dict(data["outflows"])
    data["inflows"] = pd.DataFrame.from_dict(data["inflows"])
    data["flooding"] = pd.DataFrame.from_dict(data["flooding"])
    data["depth"] = pd.DataFrame.from_dict(data["depth"])

    # Save
    if save:
        # TODO Update the storage with h5py
        np.save(destination, data)
    else:
        return data


def performance_metric(data, threshold=4.5, verbose=True):
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
    perforamce = 0.0
    # Check if there is flooding in the assets
    flooding = flooding.gt(0.0)
    flooding = flooding.any()
    if flooding.any():
        if verbose:
            # TODO print the quantity of flooding
            _idx = flooding[flooding is True].index
            print("Nodes with flooding: {}".format(_idx))
        perforamce += np.inf
    # Estimate the perfomance from the flows
    outflows = outflows.sub(threshold)
    outflows[outflows < 0.0] = 0.0
    perforamce += outflows.sum(axis=1).sum(axis=0)
    return perforamce


# Create the objective function
def swmm_execute(x, ctrl_elemnts, network, meta_data):
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
    env = environment(network, False)

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

        for valve, valve_position in zip(ctrl_elemnts, valve_positions):
            env._setValvePosition(meta_data[valve], valve_position)

        # record the data
        for node in meta_data.keys():
            data["outflows"][node].append(env.methods["flow"](meta_data[node]))
            data["flooding"][node].append(env.methods["flooding"](node))
            data["depth"][node].append(env.methods["depthN"](node))
            data["inflows"][node].append(env.methods["inflow"](node))

    data["error"] = env.sim._model.runoff_routing_stats()
    env._terminate()

    # Convert to data frames
    data["outflows"] = pd.DataFrame.from_dict(data["outflows"])
    data["inflows"] = pd.DataFrame.from_dict(data["inflows"])
    data["flooding"] = pd.DataFrame.from_dict(data["flooding"])
    data["depth"] = pd.DataFrame.from_dict(data["depth"])

    return data


def generate_waves(window_length, cycles, magnitude=1.0):
    """
    Generate Synthetic Influents

    Parameters:
    ------------------------------------------------
    window_lengthgth : number of time steps for generating waves
    cycles : number of cycles
    magnitude : amplitude of the wave

    Returns:
    ------------------------------------------------
    numpy array : timeseries of flows
    """
    half_wave = (
        np.hstack(
            (
                np.sin(np.linspace(0, 1.0, window_length) * np.pi),
                np.zeros([window_length]),
            )
        )
        * magnitude
    )
    flow = np.zeros([2 * window_length * cycles])
    counter = 0
    for i in range(0, cycles):
        start_dim = counter
        end_dim = counter + (2 * window_length)
        flow[start_dim:end_dim] = half_wave
        counter = counter + (2 * window_length)
    return flow


def extract_performance(path=".",
                        seed=None,
                        controller=None):
    metrics = {"time": [],
               "performance": [],
               "control_points": [],
               "iterations": [],
               'completed': [],
               'seed': []}

    if controller is None:
        if seed is None:
            reports = glob.glob(path + "/*_Scalability_report.txt")
        else:
            reports = glob.glob(path + str(seed) + "_*_Scalability_report.txt")
    else:
        if seed is None:
            reports = glob.glob(path + "/*?[_" + str(controller) + "_]_Scalability_report.txt")
        else:
            reports = glob.glob(path + str(seed) + "_" + str(controller) + "_Scalability_report.txt")

    # parse them to find the compute time and performance for each controller
    for file in reports:
        if seed is None:
            metrics["seed"].append(int(re.search(r"\d+\.?\d*", file).__getitem__(0)))

        metrics["control_points"].append(int(re.findall(r"\_(\d+)\_", file)[0]))

        # open the file
        with open(file, "r") as f:
            content = f.readlines()
        for line in content:
            # if line has optimization time, extract time and append it to array
            if bool(re.match(r"Optimization time:", line)):
                metrics["time"].append(
                    float(re.search(r"\d+\.?\d*", line).__getitem__(0)) / 60.0
                )
            # if line has performance
            if bool(re.match(r"Value at minimum:", line)):
                metrics["performance"].append(
                    float(re.search(r"\d+\.?\d*", line).__getitem__(0))
                )
            # if line has performance
            if bool(re.match(r"Optimization completed:", line)):
                metrics["iterations"].append(
                    float(re.search(r"\d+\.?\d*", line).__getitem__(0))
                )
                idic_yes = re.search(r'\bYES\b', line)
                if (idic_yes is not None):
                    metrics["completed"].append(1)
                else:
                    metrics["completed"].append(0)

    return metrics
