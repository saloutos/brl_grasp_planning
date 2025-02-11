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
        # set flag to indicate that controller is ready to reset
        GP.ready_to_reset = True

    def exit(self, GP):
        self.enabled = 0

    def execute(self, GP):

        # stay in this state
        next_state = self.name

        if GP.allow_reset:
            # reset both flags to be safe
            GP.ready_to_reset = False
            GP.allow_reset = False
            # increment cycle counter here
            GP.num_cycles += 1
            print("Controller cycle:", GP.num_cycles)

            # reset to keyframe 0 (don't SET keyframe 0, as this will overwrite it!)
            mj.mj_resetDataKeyframe(GP.mj_model, GP.mj_data, 0)

            # reset user scn ngeom
            GP.mj_viewer.user_scn.ngeom = 0

            # go to waiting state after reset
            next_state = "Waiting"

        return next_state