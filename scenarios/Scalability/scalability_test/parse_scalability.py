import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", palette="pastel")


def extract_performance(controller, path="."):
    metrics = {"time": [], "performance": [], "control_points": []}
    # query all the files with controller names
    reports = glob.glob(path + "/*?[_" + str(controller) + "_]_Scalability_report.txt")
    # parse them to find the compute time and performance for each controller
    for file in reports:
        # open the file
        with open(file, "r") as f:
            content = f.readlines()
        for line in content:
            # if line has optimization time, extract time and append it to array
            if bool(re.match(r"Optimization time:", line)):
                metrics["time"].append(
                    float(re.search(r"\d+\.?\d*", line).__getitem__(0)) / 60.0
                )
            # if line has performance
            if bool(re.match(r"Value at minimum:", line)):
                metrics["performance"].append(
                    float(re.search(r"\d+\.?\d*", line).__getitem__(0))
                )
        metrics["control_points"].append(controller)
    return metrics


# Get the performamce metrics for all the data
controllers = np.linspace(2, 11, 10, dtype=int)
performance = {}
for i in controllers:
    performance[str(i)] = extract_performance(i)

# Create them into a dataframe
simdata = pd.concat(
    [pd.DataFrame.from_dict(performance[i]) for i in performance.keys()],
    ignore_index=True,
)

sns.catplot(
    x="control_points",
    y="time",
    data=simdata,
    capsize=0.2,
    height=6,
    aspect=0.75,
    kind="point",
)
plt.show()
