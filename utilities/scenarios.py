# Pyswmm engine with ICC
from pyswmm_lite import environment
import numpy as np


def ParallelNetwork(valve_set1, valve_set2):
    env = environment("./networks/parallel.inp")

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
    env = environment("./networks/parallel.inp")

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


def AASinglePond(valve_set1):
    env = environment("../networks/aa_0360min_025yr.inp", False)
    data = {}
    data["outflow"] = []
    data["overflow1"] = []
    data["depth"] = []

    for time in range(0, 35000):
        # set the gate_position
        env._setValvePosition("OR48", valve_set1)
        # record_data
        data["outflow"].append(env.methods["flow"]("OR48"))
        data["overflow1"].append(env.methods["flooding"]("93-50077"))
        data["depth"].append(env.methods["depthN"]("93-50077"))
        # step through simulation
        env.step()

    env._terminate()

    return data["outflow"], sum(data["overflow1"]), data["depth"]


def test_ParallelNetwork():
    # TODO Make more defined test for the uncontrolled scenario.
    f, o1, o2 = ParallelNetwork(1.0, 1.0)
    assert len(f) >= 300
