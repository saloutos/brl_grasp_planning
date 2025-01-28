# imports
import mujoco as mj
import numpy as np
import atexit
import tty
import termios
import sys
import os
import time
import csv
from datetime import datetime as dt

from utils import *
from panda_gripper.PandaGripperPlatform import PandaGripperPlatform

# initialization
print("Starting init.")
init_settings = termios.tcgetattr(sys.stdin)

# load model here
scene_path = os.path.join(get_base_path(), "scene", "scene_with_panda_hand.xml")
spec = mj.MjSpec.from_file(scene_path)

# load a bunch of objects?
# load_random_grid_fixed_primitives(spec, 4)

# single object to grasp
load_objects_from_yaml(spec, "primitives/single_objects/fixed/cylinder_1.yaml", pos=[0,0,0.05], rpy=[0,0,0])

mj_model = spec.compile()

# platform
PandaGP = PandaGripperPlatform(mj_model, viewer_enable=True, log_path=None)

# controller
from controllers.single_object_panda.PandaGrabLiftFSM import PandaGrabLiftFSM
controller = PandaGrabLiftFSM()

# add variables to GP
PandaGP.num_cycles = 0
atexit.register(PandaGP.shutdown)
print("Finished init.")

# start experiment
try:
    tty.setcbreak(sys.stdin.fileno())
    PandaGP.initialize()
    controller.begin(PandaGP)
    PandaGP.apply_control()
    PandaGP.sync_viewer()
    print("Starting main loop.")
    real_start_time = time.time()
    log_start = PandaGP.time()
    while PandaGP.mj_viewer.is_running():
        if not PandaGP.paused:
            # step in time to update data from hardware or sim
            PandaGP.step()
            # run controller and update commands
            PandaGP.dt_comp = 0.0 # for real-time simulation
            if PandaGP.run_control:
                control_start_time = PandaGP.time()
                PandaGP.run_control = False
                PandaGP.sync_data()
                controller.update(PandaGP)
                PandaGP.apply_control()
                PandaGP.log_data()
                PandaGP.dt_comp += PandaGP.time() - control_start_time
            # sync viewer
            if PandaGP.run_viewer_sync:
                viewer_sync_start_time = PandaGP.time()
                PandaGP.run_viewer_sync = False
                PandaGP.sync_viewer()
                PandaGP.dt_comp += PandaGP.time() - viewer_sync_start_time

# end experiment
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, init_settings)