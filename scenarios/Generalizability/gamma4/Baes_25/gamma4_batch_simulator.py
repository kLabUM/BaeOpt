import os
import numpy as np
import argparse


# Argument parser for batach run
parser = argparse.ArgumentParser(
    description="Generalizability-Gamma4: GA performance Comparision"
)
parser.add_argument("-seed", nargs=1, type=int, help="Random Seed")
parser.add_argument("-iters", nargs=1, type=int, help="Number of acquisitions")
args = parser.parse_args()

# Random seed for random seeds
seed = args.seed[0]
iters = args.iters[0]

# Meta data for the analysis
np.random.seed(seed)
sds = np.random.choice(1000, 100)
itr_numbers = []

for i in sds:
    os.system(
        "python gamma4_baeopt_run.py -seed {} -iters {}".format(i, iters)
        )