# imports
import numpy as np
import mujoco as mj
from controllers.base.baseState import *
from .fsm_config import *

# define state
class Planning(BaseState):
    def __init__(self):
        self.name = "Planning"
        self.enabled = 0

    def enter(self, GP):
        print(self.name)
        self.enabled = 1
        # get initial time
        self.start_time = GP.time()

    def exit(self, GP):
        self.enabled = 0

    def execute(self, GP):

        # stay in this state
        next_state = self.name

        # get current time
        cur_time = GP.time()

        # set new "control" value
        GP.gr_data.set_ctrl(GP.gr_data.all_idxs, fsm_params.ctrl_open)

        # set wrist cartesian position
        GP.gr_data.kinematics['base_des']['p'] = fsm_params.base_pos_default
        GP.gr_data.kinematics['base_des']['R'] = fsm_params.base_R_default

        # TODO: add planning code here!
        # and then make state transition dependent on planning code output

        # declare grasp pose and approach pose?
        # TODO: not sure if this is the best way to do this?
        # wrist pose for approach
        fsm_params.base_pos_approach = np.array([0.0, -0.2, 0.04]) #np.array([0.0, 0.0, 0.05])
        fsm_params.base_R_approach = np.eye(3)
        # wrist pose for grasp
        fsm_params.base_pos_grasp = np.array([0.0, -0.1, 0.04]) #np.array([0.0, 0.0, 0.05])
        fsm_params.base_R_grasp = np.eye(3)

        # state transition to execute grasp
        if (cur_time-self.start_time) > fsm_params.times['plan']:
            next_state = "MoveToApproach"

        # check for manual state transition to grasp or reset
        if GP.char_in=='G' or GP.char_in=='g':
            next_state = "MoveToApproach"
        elif GP.char_in=='R' or GP.char_in=='r':
            next_state = "Reset"

        return next_state