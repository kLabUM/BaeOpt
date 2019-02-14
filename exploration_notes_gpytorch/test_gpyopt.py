# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np
np.random.seed(42)
# Create the function f
def f(X, noise=0.5):
    return np.sin(np.pi*X) - X**2 + 0.7*X + noise * np.random.rand()

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,2)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f,
        domain=domain,
        model_type = 'GP',
        acquisition_type='EI')

myBopt.run_optimization(max_iter=100)
myBopt.plot_acquisition()
