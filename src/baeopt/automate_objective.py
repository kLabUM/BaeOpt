import numpy as np


def generate_targets(flows, threshold, split_metric="median"):
    """
    Generate target signals for dynamic time warping
    based on the median uncontrolled flows

    Parameters:
    --------------------------------------------------------------
    flows = Outflows for the nodes as pandas dataframe
    threshold = floating point value as threshold

    Returns:
    --------------------------------------------------------------
    Dict with target signals

    """
    # Get a set of classifications based on the lenght they see water.
    nodes = flows[flows > 0.010].count()  # Length of non zero metrics
    if split_metric == "median":
        # Sort them based on median
        median = nodes.median()
        nodes_lower = nodes[nodes >= median]
        nodes_higher = nodes[nodes < median]
    elif split_metric == "mean":
        # Sort them based on median
        mean = nodes.mean()
        nodes_lower = nodes[nodes >= mean]
        nodes_higher = nodes[nodes < mean]
    else:
        raise ValueError("Split metric key error: Only median and mean are supported")

    # Assign the warping to the lower node
    ctrl_objective = {}
    # Indicate 1 for longer timeseries
    # Indicate 0 for timer timeseries
    for nodes in nodes_lower.index:
        ctrl_objective[nodes] = 0
    for nodes in nodes_higher.index:
        ctrl_objective[nodes] = 1

    # Generate a numpy array for DTW as the target signal
    for key in ctrl_objective.keys():
        if ctrl_objective[key] == 0:
            ctrl_objective[key] = np.ones(flows[key].shape[0]) * threshold
        else:

            def f(x):
                if x > threshold:
                    return threshold
                else:
                    return x

            ctrl_objective[key] = flows[key].apply(f)
    return ctrl_objective


def generate_weights(inflows):
    """
    Generate weights based on the amount of flows encounted by the
    nodes

    Parameters:
    --------------------------------------------------------------
    flows = Inflows as pandas dataframe

    Returns:
    --------------------------------------------------------------
    Dict with target weights normalized with max as 1

    """
    inflows_sum = inflows.sum()
    weights = inflows_sum.div(inflows_sum.max())
    return weights.to_dict()
