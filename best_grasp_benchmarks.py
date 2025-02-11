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

from planners.contact_graspnet.cgn_model import ContactGraspNet
from planners.edge_grasp.edge_grasp_model import EdgeGraspNet
from planners.graspness.graspness_model import GraspnessNet
from planners.giga.giga_model import GIGANet
from utils import *

# TODO: load some global config stuff here? scene, table boundaries, etc


# Initialize GLFW
glfw.init()

# Load the MuJoCo model
# TODO: make sure best MuJoCo options are set in this file?
scene_path = os.path.join(get_base_path(), "scene", "scene_with_panda_hand.xml")
spec = mujoco.MjSpec.from_file(scene_path)

# Load some objects
# load_objects_from_yaml(spec, "primitives/single_objects/fixed/box_7.yaml", pos=[0,0,0.05], rpy=[90,30,30])
load_objects_from_yaml(spec, "primitives/single_objects/fixed/cylinder_3.yaml", pos=[0,0,0.05], rpy=[0,0,0])
# load_objects_from_yaml(spec, 'primitives/collections/panda_graspable/scene_0.yaml')

model = spec.compile()
data = mujoco.MjData(model)

# Setup for rendering (need to declare these first!)
cam_width = 640
cam_height = 480
gl_context = mujoco.GLContext(cam_width, cam_height)
gl_context.make_current()
renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Function for rendering from a camera
def capture_scene(viewer, model, data, cam_name):

    # need to make current every time
    gl_context.make_current()
    viewer.sync()

    # Get the camera ID for cam_name
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    # set up camera intrinsics
    cam_fovy = np.deg2rad(model.cam_fovy[cam_id])
    cam_cx = cam_width/2
    cam_cy = cam_height/2
    cam_f = cam_height / (2 * math.tan(cam_fovy / 2))
    cam_K = np.array([[cam_f, 0.0, cam_cx], [0.0, cam_f, cam_cy], [0.0, 0.0, 1.0]])
    # TODO: could return this tuple!
    cam_intrinsics = (cam_width, cam_height, cam_cx, cam_cy, cam_f)

    # Get z-buffer properties
    # https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L817
    z_extent = model.stat.extent
    z_near = model.vis.map.znear * z_extent # 0.005
    z_far = model.vis.map.zfar * z_extent # 30

    # Define the camera parameters (you can modify these based on your need)
    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
    cam = mujoco.MjvCamera()
    cam.fixedcamid = cam_id  # Use the camera ID obtained earlier
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # Fixed camera

    # Prepare to render
    scene = mujoco.MjvScene(model, maxgeom=20000)
    viewport = mujoco.MjrRect(0, 0, cam_width, cam_height)

    # update camera stuff
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

    # Render the scene to an offscreen buffer
    mujoco.mjr_render(viewport, scene, renderer)

    # Read pixels from the OpenGL buffer (MuJoCo renders in RGB format)
    rgb_array = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)  # Image size: height=480, width=640
    depth_array = np.zeros((cam_height, cam_width), dtype=np.float32)  # Depth array
    mujoco.mjr_readPixels(rgb_array, depth_array, viewport, renderer)

    # Flip the image vertically (because OpenGL origin is bottom-left)
    rgb_array = np.flipud(rgb_array)
    depth_array = np.flipud(depth_array)

    # convert from raw depth to metric depth
    depth_array = z_near / (1 - depth_array * (1 - z_near / z_far))

    ### ADD NOISE TO DEPTH IMAGE ###
    # TODO: make this more realistic?
    depth_noise = np.random.normal(loc=0.0, scale=0.001, size=depth_array.shape)
    # depth_array += depth_noise

    # --- Process image --- #
    cam_xpos = data.cam_xpos[cam_id]
    cam_xmat = data.cam_xmat[cam_id]
    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, :3] = cam_xmat.reshape(3, 3)
    cam_extrinsics[:3, 3] = cam_xpos
    # -z --> +z
    # -y --> +y
    # +x --> +x
    T_camzforward_cam = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    cam_extrinsics = cam_extrinsics @ T_camzforward_cam

    # get point cloud
    mask = np.where((depth_array > 0.0) & (depth_array < 2.0))
    x,y = mask[1], mask[0]
    pc_rgb = rgb_array[y,x,:]
    normalized_x = (x.astype(np.float32) - cam_K[0,2])
    normalized_y = (y.astype(np.float32) - cam_K[1,2])
    world_x = normalized_x * depth_array[y, x] / cam_K[0,0]
    world_y = normalized_y * depth_array[y, x] / cam_K[1,1]
    world_z = depth_array[y, x]
    pc_xyz = np.vstack((world_x, world_y, world_z)).T

    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(pc_xyz)
    if rgb_array is not None:
        pcd_cam.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)

    # convert PC to world frame
    pcd_world = copy.deepcopy(pcd_cam).transform(cam_extrinsics)

    return pcd_cam, pcd_world, cam_extrinsics, cam_intrinsics, rgb_array, depth_array



# load a grasp generation model

# with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
#     cgn_config = yaml.safe_load(f)
# cgn = ContactGraspNet(cgn_config)

# with open('planners/edge_grasp/edge_grasp_config.yaml', 'r') as f:
#     edge_grasp_config = yaml.safe_load(f)
# edge_grasp = EdgeGraspNet(edge_grasp_config)

# with open('planners/graspness/graspness_config.yaml', 'r') as f:
#     graspness_config = yaml.safe_load(f)
# gsnet = GraspnessNet(graspness_config)

with open('planners/giga/giga_config.yaml', 'r') as f:
    giga_config = yaml.safe_load(f)
giganet = GIGANet(giga_config)

# Main simulation loop
sim_i = 0
n_grasps  = 0
n_frame_geoms = 0
cam_vis_idx = None

# init mujoco viewer
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:

    # Step the simulation several times
    mujoco.mj_step(model, data)
    # update gripper pose and finger width
    sim_t = data.time
    base_pos_des = np.array([0.0, 0.2, 0.8])
    data.mocap_pos = base_pos_des
    base_quat_des = np.zeros((4,))
    mujoco.mju_mat2Quat(base_quat_des, np.eye(3).flatten())
    data.mocap_quat = base_quat_des
    data.ctrl[0] = 200 # not real units, goes from 0 to 255
    for _ in range(1000):
        mujoco.mj_step(model, data)

    # then sync viewer
    viewer.sync()

    num_iters = 20

    plot_vals = np.zeros((num_iters, 4))

    for iter in range(num_iters):

        # capture frame from camera
        render_tic = time.time()
        pcd_cam, pcd_world, cam_extrinsics, cam_intrinsics, rgb_array, depth_array = capture_scene(viewer, model, data, "overhead_cam")
        render_toc = time.time() - render_tic

        # generate grasp candidates
        plan_tic = time.time()
        # grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = cgn.predict_scene_grasps(pcd_world, cam_extrinsics)
        # grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = edge_grasp.predict_scene_grasps(pcd_world)
        # grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = gsnet.predict_scene_grasps(pcd_world, cam_extrinsics)
        grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = giganet.predict_scene_grasps(depth_array, cam_intrinsics, cam_extrinsics, pcd_world)

        plan_toc = time.time() - plan_tic

        # get best grasp, score, width
        best_idx = np.argmax(grasp_scores)
        best_grasp = grasp_poses_world[best_idx:best_idx+1]
        best_score = grasp_scores[best_idx]
        best_width = grasp_widths[best_idx]
        # TODO: get top ten?

        # visualize grasps in mujoco
        mjv_draw_grasps(viewer, best_grasp, rgba=[0,0.7,1, 0.25], plot_best=False)

        ### PRINT SOME PLANNER OUTPUTS ###
        out = [iter, render_toc, plan_toc, best_score]
        plot_vals[iter,:] = np.array(out)
        print(out)

    # # plot rendering time
    # plt.figure()
    # plt.hist(plot_vals[:,1], bins=20)
    # plt.xlabel('Rendering Time (s)')
    # plt.ylabel('Frequency')
    # plt.title('Rendering Time Histogram')

    # # plot grasp generation time
    # plt.figure()
    # plt.hist(plot_vals[:,2], bins=20)
    # plt.xlabel('Grasp Generation Time (s)')
    # plt.ylabel('Frequency')
    # plt.title('Grasp Generation Time Histogram')

    # # plot best grasp scores
    # plt.figure()
    # plt.hist(plot_vals[:,3], bins=np.arange(11)/10.0)
    # plt.xlabel('Best Grasp Score')
    # plt.ylabel('Frequency')
    # plt.title('Best Grasp Score Histogram')

    # plt.show()

    while viewer.is_running(): pass

