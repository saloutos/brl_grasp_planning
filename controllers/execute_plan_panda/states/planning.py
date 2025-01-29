# imports
import numpy as np
import mujoco as mj
import copy
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
        # set flag to indicate that controller is ready for a plan
        GP.ready_to_plan = True

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

        # state transition to execute grasp
        # wait for complete plan, and some minimum time
        if GP.plan_complete and ((cur_time-self.start_time) > fsm_params.times['plan']):
            GP.plan_complete = False
            # grasp pose has been populated, so calculate approach pose and execute plan
            approach_pose = copy.deepcopy(GP.planned_poses['grasp_pose'])
            approach_pose[:3,3] = approach_pose[:3,3] - fsm_params.approach_offset*approach_pose[:3,2]
            GP.planned_poses['approach_pose'] = copy.deepcopy(approach_pose)
            next_state = "MoveToApproach"

        # check for manual state transition to grasp or reset
        if GP.char_in=='G' or GP.char_in=='g':
            next_state = "MoveToApproach"
        elif GP.char_in=='R' or GP.char_in=='r':
            next_state = "Reset"

        return next_state