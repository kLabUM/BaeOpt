import os
import numpy as np

# Meta data for the analysis
np.random.seed(42)
sds = np.random.choice(1000, 5)
itr_counter = 100
interval = 50

for controller in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    for i in sds:
        os.system(
            "python3 scalability.py {} {} {} {}".format(
                controller, i, itr_counter, interval
            )
        )
