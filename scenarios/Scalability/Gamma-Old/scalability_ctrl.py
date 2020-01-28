import os
import numpy as np
import sys

# Meta data for the analysis
np.random.seed(42)
controller = int(sys.argv[1])
sds = np.random.choice(1000, 42)
itr_counter = 200
interval = 50

for i in sds:
    os.system(
        "python3 scalability.py {} {} {} {}".format(
            controller,
            i,
            itr_counter,
            interval
        )
    )
