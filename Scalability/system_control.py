import pickle
import sys
import os
import requests
import numpy as np

sys.path.append("../common")
from automate_objective import generate_weights
from utilities import swmm_execute, meta_data_aa
from gpyopt.methods import BayesianOptimization


def notify_me(notification):
    headers = {"content-type": "application/json"}
    data = {"text": notification}
    response = requests.post(
        "https://hooks.slack.com/services/t0fhy378u/bgtdw34rm/pjkiek6nhglol4ymeoghjvtn",
        headers=headers,
        data=str(data),
    )
    return response.content


def performance_metric(data, ctrl_weights, threshold=5.0, verbose=False):
    """
    computes performance metrics for the optimization
    based on the threshold.

    parameters:
    --------------------------------------------------
    data: dict of pandas dataframes with outflows and flooding

    returns:
    --------------------------------------------------
    performance_metric : pefromance of the objetive
    """
    outflows = data["outflows"]
    flooding = data["flooding"]
    # get the performance metric
    perforamce = 0.0
    # check if there is flooding in the assets
    flooding = flooding.gt(0.0)
    flooding = flooding.any()
    if flooding.any():
        if verbose:
            # todo print the quantity of flooding
            print("nodes with flooding: {}".format(flooding[flooding == True].index))
        perforamce += 10 ** 8
    # estimate the perfomance from the flows
    outflows = outflows.sub(threshold)
    outflows[outflows < 0.0] = 0.0
    outflows = outflows.sum()
    for i in outflows.keys():
        perforamce += outflows[i] * ctrl_weights[i]
    return perforamce


def aa_scalability(
    number_of_ctrl_elements,
    random_seed,
    iterations,
    model_save_interval,
    notification=True,
    verbo=True,
):
    # make it a global function
    number_of_ctrl_elements = int(sys.argv[1])
    random_seed = int(sys.argv[2])
    iterations = int(sys.argv[3])
    model_save_interval = int(sys.argv[4])
    notification = True
    verbo = True

    save_path = (
        "./" + str(random_seed) + "_" + str(number_of_ctrl_elements) + "_Scalability"
    )
    os.mkdir(save_path)

    """
    Run the system optimization on the network A and save the model at intervals
    """
    # Metadata for the nodes and orifices
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
    # Choose the number of control assets
    if number_of_ctrl_elements == 11:
        ctrl_elements = NODES_LIS.keys()
    elif number_of_ctrl_elements == 10:
        ctrl_elements = [
            "93-50077",
            "93-50076",
            "93-50081",
            "93-49921",
            "93-50074",
            "93-49868",
            "93-49919",
            "93-90357",
            "93-90358",
            "93-50225",
        ]
    elif number_of_ctrl_elements == 9:
        ctrl_elements = [
            "93-50077",
            "93-50076",
            "93-50081",
            "93-49921",
            "93-50074",
            "93-49868",
            "93-49919",
            "93-90357",
            "93-90358",
        ]
    elif number_of_ctrl_elements == 8:
        ctrl_elements = [
            "93-50077",
            "93-50076",
            "93-50081",
            "93-49921",
            "93-50074",
            "93-49868",
            "93-49919",
            "93-90357",
        ]
    elif number_of_ctrl_elements == 7:
        ctrl_elements = [
            "93-50077",
            "93-50076",
            "93-50081",
            "93-49921",
            "93-50074",
            "93-49868",
            "93-49919",
        ]
    elif number_of_ctrl_elements == 6:
        ctrl_elements = [
            "93-50077",
            "93-50076",
            "93-50081",
            "93-49921",
            "93-50074",
            "93-49868",
        ]
    elif number_of_ctrl_elements == 5:
        ctrl_elements = ["93-50077", "93-50076", "93-50081", "93-49921", "93-50074"]
    elif number_of_ctrl_elements == 4:
        ctrl_elements = ["93-50077", "93-50076", "93-50081", "93-49921"]
    elif number_of_ctrl_elements == 3:
        ctrl_elements = ["93-50077", "93-50076", "93-50081"]
    elif number_of_ctrl_elements == 2:
        ctrl_elements = ["93-50077", "93-50076"]
    elif number_of_ctrl_elements == 1:
        ctrl_elements = ["93-50077"]
    else:
        raise ValueError("Invalid Control Asset Count")

    # Set the random seed
    np.random.seed(random_seed)

    # Set up the objective function
    data_uncontrolled = np.load("./uncontrolled_flow_AA.npy").item()
    inflows_uncontrolled = data_uncontrolled["inflows"]
    # Control weights
    ctrl_weights = generate_weights(inflows_uncontrolled)

    # Set the threshold
    THRESHOLD = 4.0

    def objective_function(x):
        # Run SWMM and get the trajectories
        meta = meta_data_aa()
        data = swmm_execute(x, ctrl_elements, "../networks/aa_0360min_025yr.inp", meta)
        obj_value = performance_metric(data, ctrl_weights, THRESHOLD)
        return obj_value

    # Create the domain
    domain = []
    for i in range(1, number_of_ctrl_elements + 1):
        domain.append(
            {"name": "var_" + str(i), "type": "continuous", "domain": (0.0, 1.0)}
        )

    # Solve the bayesian optimization
    myBopt = BayesianOptimization(
        f=objective_function, domain=domain, model_type="GP", acquisition_type="EI"
    )

    myBopt.run_optimization(
        save_path,
        max_iter=iterations,
        save_inter_models=True,
        intervals=model_save_interval,
        verbosity=verbo,
    )

    if verbo:
        print("Optimial Solution :", myBopt.x_opt)

    # Identify the best solution for the saved bayesian optimization models
    temp_myBopt = BayesianOptimization(
        f=objective_function, domain=domain, model_type="GP", acquisition_type="EI"
    )

    index = np.linspace(
        model_save_interval,
        iterations,
        int(iterations / model_save_interval),
        dtype=int,
    )
    pre = []
    actions = []

    meta = meta_data_aa()
    for ind in index:
        f_open = open(save_path + "/Bayopt" + str(ind), "rb")
        temp_myBopt.__dict__.update(pickle.load(f_open))
        temp_myBopt._compute_results()
        data_controlled = swmm_execute(
            temp_myBopt.x_opt.reshape(1, number_of_ctrl_elements),
            ctrl_elements,
            "../networks/aa_0360min_025yr.inp",
            meta,
        )
        perf = objective_function(temp_myBopt.x_opt.reshape(1, number_of_ctrl_elements))
        pre.append(perf)
        actions.append(temp_myBopt.x_opt.reshape(1, number_of_ctrl_elements))
        print(perf)

    # Save the info
    data_trained = {"iterations": index, "performance": pre, "actions": actions}
    np.save(
        save_path
        + "/performance_bayopt_"
        + str(number_of_ctrl_elements)
        + "_RandomSeed_"
        + str(random_seed),
        data_trained,
    )

    if notification:
        noti = str(number_of_ctrl_elements) + " Ponds are done."
        notify_me(noti)
    return 0
