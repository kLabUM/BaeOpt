import os
import numpy as np
import argparse


# Argument parser for batach run
parser = argparse.ArgumentParser(description="Scalability Test Bayesian Optimization - Iteration Time")
parser.add_argument("-ctrl", nargs=1, type=int, help="Number of control points")
args = parser.parse_args()
ctrl_elements = args.ctrl[0]

# Meta data for the analysis
np.random.seed(42)
sds = np.random.choice(1000, 15)
itr_numbers = [500]

for itr_counter in itr_numbers:
    for i in sds:
        os.system(
            "python scalability.py -seed {} -ctrl {} -iter {}".format(
                i, ctrl_elements, itr_counter
            )
        )
