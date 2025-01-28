# define config variables for all states here

# imports
import numpy as np

# general params class to add attributes to for each state
class FSMparams:
    pass

fsm_params = FSMparams()

# control values for open and close
fsm_params.ctrl_open = np.array([250])
fsm_params.ctrl_grasp = np.array([10])

# wrist pose default
fsm_params.base_pos_default = np.array([0.0, -0.5, 0.2]) #np.array([0.0, 0.0, 0.05])
fsm_params.base_R_default = np.eye(3)

# wrist pose for holding
fsm_params.base_pos_hold = np.array([0.0, -0.1, 0.2]) #np.array([0.0, 0.0, 0.25])
fsm_params.base_R_hold = np.eye(3)

# object pose default??
fsm_params.obj_pos_default = np.array([0.0, 0.0, 0.05])
fsm_params.obj_R_default = np.eye(3)

# dict of trajectory times for each state
fsm_params.times = {'wait':     1.0,
                    'plan':     0.5,
                    'approach': 0.5,
                    'pregrasp': 0.5,
                    'grasp':    1.0,
                    'lift':     1.0,
                    'hold':     1.0,
                    'release':  1.0}
