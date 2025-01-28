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
import glfw
import copy
import open3d as o3d

from utils import *
from panda_gripper.PandaGripperPlatform import PandaGripperPlatform

# load planners
from planners.contact_graspnet.cgn_model import ContactGraspNet
from planners.edge_grasp.edge_grasp_model import EdgeGraspNet
from planners.graspness.graspness_model import GraspnessNet
from planners.giga.giga_model import GIGANet


# initialization
print("Starting init.")
init_settings = termios.tcgetattr(sys.stdin)

# Initialize GLFW
glfw.init()

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
# from controllers.single_object_panda.PandaGrabLiftFSM import PandaGrabLiftFSM
# controller = PandaGrabLiftFSM()
from controllers.execute_plan_panda.PandaGrabLiftFSM import PandaGrabLiftFSM
controller = PandaGrabLiftFSM()

# planner(s)
# Contact Grasp Net
with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
    cgn_config = yaml.safe_load(f)
cgn_config['OPTIMIZER']['batch_size'] = int(1)
cgn_config['DATA']['checkpoint_path'] = 'planners/contact_graspnet/checkpoints/model.pt'
CGN = ContactGraspNet(cgn_config)
# # Edge Grasp
# with open('planners/edge_grasp/edge_grasp_config.yaml', 'r') as f:
#     edge_grasp_config = yaml.safe_load(f)
# EDGE = EdgeGraspNet(edge_grasp_config)
# # GS Net
# with open('planners/graspness/graspness_config.yaml', 'r') as f:
#     graspness_config = yaml.safe_load(f)
# GSN = GraspnessNet(graspness_config)
# # GIGA
# with open('planners/giga/giga_config.yaml', 'r') as f:
#     giga_config = yaml.safe_load(f)
# GIGA = GIGANet(giga_config)

# add variables to GP
PandaGP.num_cycles = 0
# for grasp planning trigger
PandaGP.ready_to_plan = False
PandaGP.plan_complete = False
PandaGP.planned_poses = {'approach_pose': np.eye(4), 'grasp_pose': np.eye(4)}

atexit.register(PandaGP.shutdown)
print("Finished init.")

# start experiment
try:
    tty.setcbreak(sys.stdin.fileno())
    PandaGP.initialize()
    PandaGP.initialize_rendering() # need this for planning
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
            # check for planning flag
            if PandaGP.ready_to_plan:
                PandaGP.ready_to_plan = False

                # save image
                pcd_cam, pcd_world, cam_extrinsics = PandaGP.capture_scene()

                # Show point cloud
                print(pcd_world)

                # run planning
                plan_start = time.time()
                # TODO: remove some of these inputs? probably don't need them
                # TODO: stop returning dicts for each, just return the final grasp poses
                grasp_poses_cam, scores, contact_pts, widths = CGN.predict_scene_grasps(pcd_cam,
                                                                                    pc_segments={},
                                                                                    local_regions=False,
                                                                                    filter_grasps=True,
                                                                                    forward_passes=1)
                # grasp_poses_world, grasp_scores, grasp_widths = EDGE.predict_scene_grasps(pcd_world)
                # grasp_poses_cam, grasp_scores, grasp_widths = GSN.predict_scene_grasps(pcd_cam)
                # TODO: pull this out of the capture scene function!
                # grasp_poses_world, grasp_scores, grasp_widths = GIGA.predict_scene_grasps(depth_array, k_d405_640x480, cam_extrinsics)
                plan_time = time.time() - plan_start

                # put grasps in world frame
                grasp_poses_world_array = np.zeros_like(grasp_poses_cam[-1])
                for i,g in enumerate(grasp_poses_cam[-1]):
                    grasp_poses_world_array[i,:4,:4] = np.matmul(cam_extrinsics, g)
                grasp_poses_world = {-1: grasp_poses_world_array}

                best_grasp = np.argmax(scores[-1])
                best_score = scores[-1][best_grasp]
                best_width = widths[-1][best_grasp]
                best_pose = grasp_poses_world_array[best_grasp,:,:]

                print("Best grasp: ", best_grasp)
                print("Score: ", best_score)
                print("Width: ", best_width)
                print("Pose: ", best_pose)

                # calculate approach pose from grasp pose
                # TODO: check this
                approach_pose = copy.deepcopy(best_pose)
                approach_pose[:3,3] = approach_pose[:3,3] - 0.1*approach_pose[:3,2]

                # save grasp pose and approach pose
                PandaGP.planned_poses['approach_pose'] = copy.deepcopy(approach_pose)
                PandaGP.planned_poses['grasp_pose'] = copy.deepcopy(best_pose)
                # PandaGP.planned_poses['approach_pose'][:3,3] = np.array([0.0, -0.2, 0.04])
                # PandaGP.planned_poses['grasp_pose'][:3,3] = np.array([0.0, -0.1, 0.04])

                # finally, visualize the grasps
                # TODO: add a grasp at approach pose too?
                PandaGP.mj_viewer.user_scn.ngeom = 0
                new_rgb = np.random.rand(3)
                mjv_draw_grasps(PandaGP.mj_viewer, grasp_poses_world[-1], scores=scores[-1], linewidth=3)
                # mjv_draw_grasps(PandaGP.mj_viewer, grasp_poses_world[-1], rgba=[new_rgb[0], new_rgb[1], new_rgb[2], 0.25])

                # # visualize grasps in separate window
                # visualize_grasps(pcd_world, grasp_poses_world, scores,
                #                 window_name = 'ContactGraspNet',
                #                 plot_origin=True,
                #                 gripper_openings=None)

                PandaGP.plan_complete = True


# end experiment
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, init_settings)