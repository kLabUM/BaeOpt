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
plt.plot(data_controlled["loading"]["001"], label="Controlled")
plt.plot(
    data_uncontrolled["loading"]["001"],
    color="black",
    linestyle="--",
    label="Uncontrolled",
)
plt.axhline(1.075, color="Red")
plt.title("Scenario Epsilon")
plt.ylabel("Loading")
plt.legend()

plt.show()
