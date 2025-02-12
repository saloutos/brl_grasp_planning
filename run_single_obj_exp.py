# imports
import mujoco as mj
import numpy as np
import os
import time
import copy
import dill
import glfw
import datetime as dt

# import experiment function
from experiments import single_object_experiment_panda

# Initialize GLFW here?
glfw.init()

obj_names = [
    'box_7', 'box_7', 'box_9', 'box_9',
    'cylinder_3', 'cylinder_3', 'cylinder_6', 'cylinder_6',
    'sphere_4', 'sphere_5'
    ]
obj_poses = [
    ([0, 0, 0.08], [0, 0, 0]),  # box 7
    ([0, 0, 0.08], [90, 0, 0]),  # box 7B
    ([0, 0, 0.08], [0, 0, 0]),  # box 9
    ([0, 0, 0.08], [90, 0, 0]),  # box 9B
    ([0, 0, 0.08], [0, 0, 0]),  # cylinder 3
    ([0, 0, 0.08], [90, 0, 0]),  # cylinder 3B
    ([0, 0, 0.08], [0, 0, 0]),  # cylinder 6
    ([0, 0, 0.08], [90, 0, 0]),  # cylinder 6B
    ([0, 0, 0.08], [0, 0, 0]),  # sphere 4
    ([0, 0, 0.08], [0, 0, 0]),  # sphere 5
]

planners = ['CGN', 'EDGE', 'GSN', 'GIGA']

num_plans = 3

outputs = []

# set up logging folder
if not os.path.exists('logs/'):
    os.makedirs('logs/')
log_folder = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"/"
log_folder = os.path.join('logs/', log_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

output_filename = "/soep_summary.txt"

for i, obj_name in enumerate(obj_names):
    obj_nominal_pose = obj_poses[i]

    for planner_name in planners:

        data = single_object_experiment_panda(obj_name, obj_nominal_pose, planner_name, num_plans)
        output = [obj_name, obj_nominal_pose, planner_name, num_plans, data['success'], data['exp_time']]
        outputs.append(output)
        # save data
        print("Saving...")
        save_file = "/soep_full_{}_{}_{}.dill".format(i, obj_name, planner_name)
        with open(log_folder+save_file, 'wb') as f:
            dill.dump(data, f)
        # append outputs to summary file?
        with open(log_folder+output_filename, 'a') as f:
            f.write("{}\n".format(output))

for line in outputs:
    print(line)