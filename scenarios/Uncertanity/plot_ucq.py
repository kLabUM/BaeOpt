import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load("./Prototyping-Notebooks/data_ucq1.npy", allow_pickle=True).item()

sns.set_style("white")

plt.subplot(3, 1, 1)
# EMP vs VGP
plt.plot(data["vgp_x"], data["vgp_y_mean"], color="#142850")
plt.fill_between(
    data["vgp_x"], data["vgp_y_up"], data["vgp_y_dwm"], color="#142850", alpha=0.3
)
plt.ylim([-0.5, 1.5])


plt.plot(data["emp_x"], data["emp_y_mean"], color="#e74c3c")
plt.fill_between(
    data["emp_x"], data["emp_y_up"], data["emp_y_dwm"], color="#e74c3c", alpha=0.5
)
plt.ylim([-0.5, 1.5])
plt.plot(data["sample_x"], data["sample_y"], "x", color="#383e56")

plt.subplot(3, 1, 2)
# EMP vs HGP
plt.plot(data["emp_x"], data["emp_y_mean"], color="#e74c3c", label="Empirical")
plt.fill_between(
    data["emp_x"], data["emp_y_up"], data["emp_y_dwm"], color="#e74c3c", alpha=0.5
)
plt.ylim([-0.5, 1.5])

plt.plot(data["hgp_x"], data["hgp_y_mean"], color="#3498db")
plt.fill_between(
    data["hgp_x"], data["hgp_y_up"], data["hgp_y_dwm"], color="#3498db", alpha=0.5
)
plt.ylim([-0.5, 1.5])

plt.plot(data["sample_x"], data["sample_y"], "x", color="#383e56")

plt.subplot(3, 1, 3)

plt.plot(data["vgp_x"], data["vgp_y_mean"], color="#142850")
plt.fill_between(
    data["vgp_x"], data["vgp_y_up"], data["vgp_y_dwm"], color="#142850", alpha=0.3
)
plt.ylim([-0.5, 1.5])

plt.plot(data["hgp_x"], data["hgp_y_mean"], color="#3498db")
plt.fill_between(
    data["hgp_x"], data["hgp_y_up"], data["hgp_y_dwm"], color="#3498db", alpha=0.5
)
plt.ylim([-0.5, 1.5])

plt.plot(data["sample_x"], data["sample_y"], "x", color="#383e56")

plt.show()
