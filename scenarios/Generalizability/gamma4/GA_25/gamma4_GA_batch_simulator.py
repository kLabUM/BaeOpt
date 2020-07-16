import os
import numpy as np
import argparse


# Argument parser for batach run
parser = argparse.ArgumentParser(
    description="Generalizability-Gamma4: GA performance Comparision"
)
parser.add_argument("-seed", nargs=1, type=int, help="Random Seed")
parser.add_argument("-rang", nargs=1, type=int, help="Random sample grab range")
args = parser.parse_args()

# Random seed for random seeds
seed = args.seed[0]
sample_range = args.rang[0]
# Meta data for the analysis
np.random.seed(seed)
grab_range = np.linspace(100*(sample_range-1), 100*(sample_range), 101, dtype=int)
sds = np.random.choice(grab_range, 25)
itr_numbers = []

for i in sds:
    os.system(
        "python basicGA_gamma.py -seed {}".format(i)
        )
