import pyswmm
import numpy as np
import matplotlib.pyplot as plt
from environment_mbc_wq import Env
import seaborn as sns
sns.set_style("whitegrid")

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

env = Env("./parallel.inp")

data = {}
data["endflow"] = []

flow = generate_waves(100, 10, 2.0)
for time in range(0, len(flow)):

    # set inflow
    env.sim._model.setNodeInflow("P1", flow[time])
    env.sim._model.setNodeInflow("P2", flow[time])
    
    # record_data
    data["endflow"].append(env.flow("8"))

    # step through simulation 
    env.step()

# terminate swmm 
env.sim._model.swmm_end()
env.sim._model.swmm_close()

plt.plot(data["endflow"])
plt.show()
