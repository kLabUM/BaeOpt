import numpy as np
import matplotlib.pyplot as plt


data_uncontrolled = np.load("./uncontrolled_response.npy", allow_pickle=True).item()
data_controlled = np.load("./controlled_response.npy", allow_pickle=True).item()

performance_uncontrolled = np.load("./uncontrolled_performance.npy")
performance_controlled = np.load("./controlled_performance.npy")

valves = np.load("./actions.npy")

print(
    "Valve Positions : {} \n ----- Performance ----- \n Uncontrolled = {} \n Controlled = {}".format(
        valves, performance_uncontrolled, performance_controlled
    )
)

plt.figure(1)
plt.subplot(1, 3, 1)
plt.plot(data_controlled["depthN"]["P1"])
plt.plot(data_uncontrolled["depthN"]["P1"])

plt.subplot(1, 3, 2)
plt.plot(data_controlled["depthN"]["P2"])
plt.plot(data_uncontrolled["depthN"]["P2"])

plt.subplot(1, 3, 3)
plt.plot(data_controlled["flow"]["8"])
plt.plot(data_uncontrolled["flow"]["8"])

plt.legend()
plt.show()
