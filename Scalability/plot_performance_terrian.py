import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plot_matrix = np.zeros((10, 20))
for ponds in range(2, 12):
    path = "./20_"+str(ponds)+"_Ponds_Fig2/performance_bayopt_"+str(ponds)+"_RandomSeed_20.npy"
    print(path)
    temp = np.load(path).item()
    plot_matrix[ponds-2:] = temp["performance"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, 10, 10, dtype=int)
y = np.linspace(10, 200, 20, dtype=int)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, plot_matrix.T, rstride=10, cstride=1, cmap=plt.cm.viridis, linewidth=0, shade=False)
ax.set_title('surface')
ax.set_ylabel("Iterations")
ax.set_xlabel("Controlled Assets")
ax.set_zlabel("Performance")
plt.show()
