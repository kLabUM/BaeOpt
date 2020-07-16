import pystorms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a function for running evaluvating actions
def swmm_execute(actions=np.ones(11)):
    env = pystorms.scenarios.gamma()
    done = False
    # Append the states you might be intrested!
    nodes = [1, 2, 3, 4]
    env.data_log["depthN"] = {}
    for i in nodes:
        env.data_log["depthN"][str(i)] = []

    time = []
    while not done:
        time.append(env.env.sim.current_time)
        done = env.step(actions)
    return env.data_log, time


def resample(data, attr=["flow", "depthN"]):
    for i in attr:
        for node in data[i].keys():
            data[i][node] = pd.Series(data[i][node], data["time"])
            data[i][node] = data[i][node].resample("60S").max()
    return data


uc_data, time = swmm_execute()
uc_data["time"] = time

actions = [0.34895609, 1.0, 0.21987084, 0.20640974]
c_data, time = swmm_execute(actions)
c_data["time"] = time

uc_data = resample(uc_data)
c_data = resample(c_data)

for i in [4, 3, 2, 1]:
    plt.subplot(2, 4, i)
    if i == 1:
        plt.ylabel("Depth")
    plt.plot(c_data["depthN"][str(i)] * 0.3048)
    plt.plot(uc_data["depthN"][str(i)] * 0.3048)
    plt.ylim([0, 2.5])

    plt.subplot(2, 4, i + 4)

    if i == 1:
        plt.ylabel("Outflow")
    plt.plot(c_data["flow"]["O" + str(i)] * 0.03)
    plt.plot(uc_data["flow"]["O" + str(i)] * 0.03)
    plt.axhline(0.11)
    plt.ylim([0, 0.40])
plt.show()
