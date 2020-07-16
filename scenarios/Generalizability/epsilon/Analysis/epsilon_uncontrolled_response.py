import pystorms
import numpy as np

env = pystorms.scenarios.epsilon()
done = False

while not done:
    done = env.step()

np.save("./uncontrolled_response", env.data_log)
np.save("./uncontrolled_performance", env.performance())
