# load objects into mujoco scene

import mujoco
import mujoco.viewer
import os
import glfw
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import open3d as o3d
import yaml
import copy
import math

from utils import *

# Load the MuJoCo model
# TODO: make sure best MuJoCo options are set in this file?
scene_path = os.path.join(get_base_path(), "scene", "scene_with_panda_hand.xml")
spec = mujoco.MjSpec.from_file(scene_path)

# np.random.seed(0)
# Load objects
# load_single_primitive(spec, 'test_obj', [0.0, 0.0, 0.2],
#                     obj_type=mujoco.mjtGeom.mjGEOM_SPHERE,
#                     size=[0.05, 0, 0],
#                     mass=0.5,
#                     rgba=[0.1, 0.9, 0.1, 1.0])
# load_random_grid_primitives(spec, 4)
# load_from_file_primitives(spec, file)

# can load 2 yamls, but must check that names are unique
# load_objects_from_yaml(spec, 'object_test.yaml')
# load_objects_from_yaml(spec, 'object_test_2.yaml')

# load_random_grid_fixed_primitives(spec, 4)

load_objects_from_yaml(spec, 'primitives/collections/scene_9.yaml')

model = spec.compile()
data = mujoco.MjData(model)

# init mujoco viewer
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:

    # Main simulation loop
    sim_i = 0

    while viewer.is_running():

        # Step the simulation
        mujoco.mj_step(model, data)
        time.sleep(0.001)
        # update gripper pose and finger width
        sim_t = data.time
        base_pos_des = np.array([0.0, 0.2, 0.8])
        data.mocap_pos = base_pos_des
        base_quat_des = np.zeros((4,))
        mujoco.mju_mat2Quat(base_quat_des, np.eye(3).flatten())
        data.mocap_quat = base_quat_des
        data.ctrl[0] = 200 # not real units, goes from 0 to 255

        # if sim_i==2000:
        #     index = viewer.user_scn.ngeom
        #     # add line?
        #     mujoco.mjv_initGeom(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, [0]*3, [0]*3, [0]*9, [1]*4)
        #     mujoco.mjv_connector(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, 5, [1,1,0], [0,0,2.0])
        #     viewer.user_scn.geoms[index].rgba = np.array([1.0, 0.0, 0.0, 0.9])
        #     viewer.user_scn.geoms[index].label = ''
        #     index+=1
        #     # update number of geoms and sync
        #     viewer.user_scn.ngeom = index
        #     print("Added line!")

        # then sync viewer
        viewer.sync()
        sim_i += 1


