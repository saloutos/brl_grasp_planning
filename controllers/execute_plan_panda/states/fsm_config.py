# define config variables for all states here

# imports
import numpy as np

# general params class to add attributes to for each state
class FSMparams:
    pass

fsm_params = FSMparams()

# # control values for open and close (with position control panda)
# fsm_params.ctrl_open = np.array([250])
# fsm_params.ctrl_grasp = np.array([10])

# control values for open and close (with force control panda)
fsm_params.ctrl_open = np.array([0])
fsm_params.ctrl_grasp = np.array([10])

# wrist pose default
fsm_params.base_pos_default = np.array([0.0, -0.5, 0.2])
fsm_params.base_R_default = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, -1.0, 0.0]])

# wrist pose offset for holding
fsm_params.base_pos_hold_offset = np.array([0.0, 0.0, 0.2])

# offset distance for approach pose (along -z of grasp pose)
fsm_params.approach_offset = 0.1


# dict of trajectory times for each state
fsm_params.times = {'wait':     0.5,
                    'plan':     0.5,
                    'approach': 1.0,
                    'pregrasp': 1.0,
                    'grasp':    1.0,
                    'lift':     1.0,
                    'hold':     1.0,
                    'release':  1.0}
