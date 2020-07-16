import pystorms
import numpy as np

env = pystorms.scenarios.theta()
done = False

env.data_log["depthN"] = {}
for i in ["P1", "P2"]:
    env.data_log["depthN"][i] = []

while not done:
    done = env.step()

np.save("./uncontrolled_performance", env.performance())
np.save("./uncontrolled_response.npy", env.data_log)
