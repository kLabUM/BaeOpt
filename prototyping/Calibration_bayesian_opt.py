import pyswmm
import numpy as np
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
from environment_mbc_wq import Env
import seaborn as sns
sns.set_style("whitegrid")
# Generate sine waves
def generate_waves(window_len, cycles, magnitude = 1.0):
    half_wave = np.hstack((np.sin(np.linspace(0, 1.0, window_len)*np.pi),np.zeros([window_len]))) * magnitude
    flow = np.zeros([2*window_len*cycles])
    counter = 0
    for i in range(0, cycles):
        start_dim = counter
        end_dim = counter+(2*window_len)
        flow[start_dim:end_dim] = half_wave
        counter = counter+(2*window_len)
    return flow

env = Env("./parallel.inp")

data = {}
data["endflow"] = []
done = False
flow = generate_waves(100, 20, 1.0)
# target values 

valve_set1 = 0.20
valve_set2 = 0.20

for time in range(0, len(flow)):
    # set the gate_position 
    env.set_gate("1", valve_set1)
    env.set_gate("2", valve_set2)

    env.sim._model.setNodeInflow("P1", flow[time])
    env.sim._model.setNodeInflow("P2", flow[time])


    # record_data
    data["endflow"].append(env.flow("8"))

    # step through simulation
    done = env.step()

# terminate swmm
env.sim._model.swmm_end()
env.sim._model.swmm_close()

np.save("./true_response.npy", data["endflow"])

# Set up basesian opt 

def run_swmm(valve_set1, valve_set2):
    env = Env("./parallel.inp")

    data = {}
    data["endflow"] = []
    done = False
    flow = generate_waves(100, 20, 1.0)
    for time in range(0, len(flow)):
        # set the gate_position 
        env.set_gate("1", valve_set1)
        env.set_gate("2", valve_set2)

        env.sim._model.setNodeInflow("P1", flow[time])
        env.sim._model.setNodeInflow("P2", flow[time])


        # record_data
        data["endflow"].append(env.flow("8"))

        # step through simulation
        done = env.step()

    return data["endflow"]

# define swmm fun
def f(x):
    x1 = x[0][0]
    x2 = x[0][1]
    true_flow = np.asarray(np.load("true_response.npy"))
    pred_flow = np.asarray(run_swmm(x1, x2))
    loss = np.abs(true_flow.sum() - pred_flow.sum())
    return loss

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 0.5)},
          {'name': 'var_2', 'type': 'continuous', 'domain': (0, 0.5)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f,
        domain=domain,
        model_type = 'GP',
        acquisition_type='EI',
        exact_feval = True,
        acquisition_weight = 2)

myBopt.run_optimization(max_iter=100)
print(myBopt.x_opt)
calb = myBopt.x_opt
myBopt.plot_acquisition()
myBopt.plot_convergence()
clalib_flow = run_swmm(myBopt.x_opt[0], myBopt.x_opt[1])
plt.plot(np.load("./true_response.npy"), label="True Respose")
plt.plot(clalib_flow, label = "Calib Flow")
plt.legend()
plt.show()



