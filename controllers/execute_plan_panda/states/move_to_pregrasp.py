# imports
import numpy as np
from controllers.base.baseState import *
from .fsm_config import *

# define state
class MoveToPreGrasp(BaseState):
    def __init__(self):
        self.name = "MoveToPreGrasp"
        self.enabled = 0
        self.start_time = 0

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
        # TODO: move from default to approach pose, then from approach pose to grasp pose here?
        ratio = (cur_time - self.start_time)/fsm_params.times['pregrasp']
        GP.gr_data.kinematics['base_des']['p'] = ratio*fsm_params.base_pos_grasp + (1-ratio)*fsm_params.base_pos_approach
        # TODO: interpolate rotations?
        GP.gr_data.kinematics['base_des']['R'] = fsm_params.base_R_grasp #ratio*fsm_params.base_R_default + (1-ratio)*fsm_params.base_R_default

        # state transition to holding object
        if (cur_time-self.start_time) > fsm_params.times['pregrasp']:
            next_state = "ExecuteGrasp"

        # check for manual state transition to reset
        if GP.char_in=='R' or GP.char_in=='r':
            next_state = "Reset"

        return next_state