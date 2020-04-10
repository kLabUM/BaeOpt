import GPyOpt
import GPy
import uncertaninity_baestorms as ubay
import numpy as np


# Create an objective function
class Objective:
    def f(self, x):
        # Convert 2d array to 1d
        print(x.shape)
        x = x.flatten()
        reward = ubay.ObjectiveFunction(x)
        return reward


# Initalize the objective
Obj_temp = Objective()
Obj_temp = GPyOpt.core.task.SingleObjective(Obj_temp.f)

# Create Search Space
Search_Space = GPyOpt.Design_space(
    space=[
        {"name": "var_1", "type": "continuous", "domain": (0.0, 1.0)},
        {"name": "var_2", "type": "continuous", "domain": (0.0, 1.0)}
    ]
)

# Pick a kernel
kernel = GPy.kern.RBF(input_dim=1) + GPy.kern.White(input_dim=1)
# Set up the model
model = GPyOpt.models.GPModel(kernel=kernel,
                              noise_var=0.0,
                              optimize_restarts=5,
                              verbose=False)

# How do you sample the region
initial_design = GPyOpt.experiment_design.initial_design("random", Search_Space, 5)

# Choose the acquision function
acq_optimizer = GPyOpt.optimization.AcquisitionOptimizer(Search_Space)
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, Search_Space, acq_optimizer)

evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(
    model, Search_Space, Obj_temp, acquisition, evaluator, initial_design
)

max_iter = 100
bo.run_optimization(path_to_save="./", max_iter=max_iter)
print(bo.x_opt)
