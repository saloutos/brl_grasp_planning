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

        # reset to keyframe 0 (don't SET keyframe 0, as this will overwrite it!)
        mj.mj_resetDataKeyframe(GP.mj_model, GP.mj_data, 0)

        # go to waiting state after reset
        next_state = "Waiting"

        return next_state