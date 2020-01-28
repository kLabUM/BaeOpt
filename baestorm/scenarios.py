from pyswmm_lite import environment
import pystorms
import numpy as np
import os


def ParallelNetwork(valve_set1, valve_set2):
    # Local Path
    path = os.path.abspath(os.path.dirname(__file__)) + "/networks/parallel.inp"
    env = environment(path, False)

    flow = np.sin(np.linspace(0.0, 1.0, 100) * np.pi) * 2.0  # Half sine wave
    flow = np.hstack((np.zeros(10), flow, np.zeros(200)))

    data = {}
    data["outflow"] = []
    data["overflow1"] = []
    data["overflow2"] = []

    for time in range(0, len(flow)):
        # set the gate_position
        env.set_gate("1", valve_set1)
        env.set_gate("2", valve_set2)

        env.sim._model.setNodeInflow("P1", 3 * flow[time])
        env.sim._model.setNodeInflow("P2", 3 * flow[time])

        # record_data
        data["outflow"].append(env.flow("8"))
        data["overflow1"].append(env.sim._model.getNodeResult("P1", 4))
        data["overflow2"].append(env.sim._model.getNodeResult("P2", 4))

        # step through simulation
        _ = env.step()

    return data["outflow"], sum(data["overflow1"]), sum(data["overflow2"])


def SinglePond(valve_set1):
    path = os.path.abspath(os.path.dirname(__file__)) + "/networks/parallel.inp"
    env = environment(path, False)

    flow = np.sin(np.linspace(0.0, 1.0, 100) * np.pi) * 3.0  # Half sine wave
    flow = np.hstack((np.zeros(10), flow, np.zeros(200)))

    data = {}
    data["outflow"] = []
    data["overflow1"] = []

    for time in range(0, len(flow)):
        # set the gate_position
        env.set_gate("1", valve_set1)

        env.sim._model.setNodeInflow("P1", 3 * flow[time])
        env.sim._model.setNodeInflow("P2", 0.0)

        # record_data
        data["outflow"].append(env.flow("8"))
        data["overflow1"].append(env.sim._model.getNodeResult("P1", 4))

        # step through simulation
        _ = env.step()

    return data["outflow"], sum(data["overflow1"])


def GammaSinglePond(valve_setting):
    env = pystorms.scenarios.gamma()
    done = False

    # Update the datalog to add depth
    env.data_log["depthN"] = {"1": []}

    # Set actions
    actions = np.ones(11)
    actions[0] = valve_setting

    while not done:
        # step through simulation
        done = env.step(actions)

    return env.data_log


def test_ParallelNetwork():
    # TODO Make more defined test for the uncontrolled scenario.
    f, o1, o2 = ParallelNetwork(1.0, 1.0)
    assert len(f) >= 300
