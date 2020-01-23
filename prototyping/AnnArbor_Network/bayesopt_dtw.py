from pyswmm_lite import Env
import pyswmm
import numpy as np


# Generate sine waves
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

def run_swmm(valve_set1, valve_set2, flow):
    env = Env("./parallel.inp")

    data = {}
    data["endflow"] = []
    data["overflow1"]  = []
    data["overflow2"]  = []
    done = False
    for time in range(0, len(flow)):
        # set the gate_position 
        env.set_gate("1", valve_set1)
        env.set_gate("2", valve_set2)

        env.sim._model.setNodeInflow("P1", 3*flow[time])
        env.sim._model.setNodeInflow("P2", 3*flow[time])


        # record_data
        data["endflow"].append(env.flow("8"))
        data["overflow1"].append(env.sim._model.getNodeResult("P1", 4))
        data["overflow2"].append(env.sim._model.getNodeResult("P2", 4))

        # step through simulation
        done = env.step()

    return data["endflow"], sum(data["overflow1"]), sum(data["overflow2"])
