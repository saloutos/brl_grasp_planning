# imports
import mujoco as mj
import numpy as np
import atexit
import tty
import termios
import sys
import os
import time
import glfw
import copy
import dill

# import experiment function
from experiments import single_object_experiment

obj_name = 'box_7'
obj_nominal_pose = ([0, 0, 0.08], [0, 0, 30])
planner_name = 'CGN'
num_plans = 10

data, success = single_object_experiment(obj_name, obj_nominal_pose, planner_name, num_plans)