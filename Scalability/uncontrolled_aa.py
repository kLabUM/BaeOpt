import numpy as np
import pandas as pd
from pyswmm_lite import Env
import tslearn.metrics as ts
import scipy.signal as spy
import scipy
import pickle
from automate_objective import generate_targets, generate_weights
from utilities_baeopt import swmm_execute, performance_metric, uncontrolled_response
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization

NODES_LIS = {'93-49743' : 'OR39',
             '93-49868' : 'OR34',
             '93-49919' : 'OR44',
             '93-49921' : 'OR45',
             '93-50074' : 'OR38',
             '93-50076' : 'OR46',
             '93-50077' : 'OR48',
             '93-50081' : 'OR47',
             '93-50225' : 'OR36',
             '93-90357' : 'OR43',
             '93-90358' : 'OR35'}

ctrl_elements = ["93-50077", "93-50076", "93-50081", "93-49921"]


env = Env("./networks/aa_0360min_025yr.inp")
data = uncontrolled_response(env, NODES_LIS, False)
