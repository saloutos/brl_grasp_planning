# imports
import numpy as np
import mujoco as mj
from controllers.base.baseState import *
from .fsm_config import *

# define state
class Reset(BaseState):
    def __init__(self):
        self.name = "Reset"
        self.enabled = 0

    def enter(self, GP):
        print(self.name)
        self.enabled = 1

    def exit(self, GP):
        self.enabled = 0

    def execute(self, GP):

        # increment cycle counter here
        GP.num_cycles += 1
        print("Controller cycle:", GP.num_cycles)

        # TODO: update mj_model settings based on number of cycles here?

        # TODO: reset using keyframe of model?

        # set new "control" value
        GP.gr_data.set_ctrl(GP.gr_data.all_idxs, fsm_params.ctrl_open)

        # directly set wrist position
        GP.mj_data.mocap_pos = fsm_params.base_pos_default
        base_quat_default = np.zeros((4,))
        mj.mju_mat2Quat(base_quat_default, fsm_params.base_R_default.flatten())
        GP.mj_data.mocap_quat = base_quat_default

        # TODO: directly set object position?
        # now, it goes 7 for wrist pose, 2 for finger joints, 7 for object pose
        # object pos is 9,10,11, and object quat is 12,13,14,15
        GP.mj_data.qpos[9:12] = fsm_params.obj_pos_default
        obj_quat_default = np.zeros((4,))
        mj.mju_mat2Quat(obj_quat_default, fsm_params.obj_R_default.flatten())
        GP.mj_data.qpos[12:16] = obj_quat_default

        # go to waiting state after reset
        next_state = "Waiting"

        return next_state