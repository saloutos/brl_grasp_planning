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
fsm_params.base_pos_default = np.array([0.0, -0.3, 0.2]) #np.array([0.0, 0.0, 0.05])
fsm_params.base_R_default = np.eye(3)

# # wrist pose for approach
fsm_params.base_pos_approach = np.array([0.0, -0.3, 0.2]) #np.array([0.0, -0.2, 0.04]) #np.array([0.0, 0.0, 0.05])
fsm_params.base_R_approach = np.eye(3)
# wrist pose for grasp
fsm_params.base_pos_grasp = np.array([0.0, -0.3, 0.2]) #np.array([0.0, -0.1, 0.04]) #np.array([0.0, 0.0, 0.05])
fsm_params.base_R_grasp = np.eye(3)


# wrist pose for holding
fsm_params.base_pos_hold = np.array([0.0, -0.1, 0.2]) #np.array([0.0, 0.0, 0.25])
fsm_params.base_R_hold = np.eye(3)

# object pose default??
fsm_params.obj_pos_default = np.array([0.0, 0.0, 0.05])
fsm_params.obj_R_default = np.eye(3)

# dict of trajectory times for each state
fsm_params.times = {'plan':     0.5,
                    'approach': 0.25,
                    'pregrasp': 0.5,
                    'grasp':    1.0,
                    'lift':     1.0,
                    'hold':     1.0,
                    'release':  1.0}
