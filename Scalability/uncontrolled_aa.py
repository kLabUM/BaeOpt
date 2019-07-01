from pyswmm_lite import environment
import sys
sys.path.append("../common")
from utilities import uncontrolled_response, meta_data_aa


env = environment("../networks/aa_0360min_025yr.inp", False)
_temp = meta_data_aa()
uncontrolled_response(env, _temp, True)
