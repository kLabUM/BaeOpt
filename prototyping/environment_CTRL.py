from pyswmm import Simulation
import numpy as np

class Env:
    """
    Open AI style API to interact with EPA SWMM.
    Depends on pyswmm for interacting with SWMMS computational engine.
    ...

    Attributes
    ----------
    input_file: string
        EPA SWMM's input file

    state_space: dictionary 
        keys of the differet attributes "depth","flow"

    control_points: list
        list of orifice ids

    Methods
    -------
    start_state()
        gives the inital state of the system
        
    step()
        steps the simulation by one time step. 
        TODO: Length of the step can be configured through pyswmm.
    
    reset()
        resets the simulation to initial state to repeat the simulation.

    reward()
        set the reward function --> User Supplied

    """
    def __init__(self,
            input_file,
            state_space,
            control_points):

        self.sim = Simulation(input_file)
        self.sim._model.swmm_start()
        self.state_space = state_space
        self.control_points = control_points

    def step(self, actions):
        # Set the valve positions
        for orifice, valve_postions in zip(self.control_points, actions):
            self.sim._model.setLinkSetting(orifice, valve_postions)

        # Take a step
        time = self.sim._model.swmm_step()

        done = True if time <= 0 else False

        # 
        reward = self.reward()
        # Read the new states
        new_states = self.states()

        return new_states, reward , done

    def start_state(self):
        state = self.states()
        return state 

    def reset(self):
        # Close the curren simulation
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()
        # Start the next simulation
        self.sim._model.swmm_open()
        self.sim._model.swmm_start()

    def states(self):
        new_states = []
        for d in self.state_space["depths"]:
            new_states.append(self.sim._model.getNodeResult(d, 5))
        
        for f in self.state_space["flows"]:
            new_states.append(self.sim._model.getLinkResult(f, 0))
        return np.asarray([new_states])


    def reward(self):
        return 0.0