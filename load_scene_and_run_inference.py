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

from planners.contact_graspnet.cgn_model import ContactGraspNet
from planners.edge_grasp.edge_grasp_model import EdgeGraspNet
from planners.graspness.graspness_model import GraspnessNet
from planners.giga.giga_model import GIGANet
from utils import *

# TODO: load some global config stuff here? scene, table boundaries, etc


# Initialize GLFW and OpenCV window
glfw.init()
cv2.namedWindow("MuJoCo Camera", cv2.WINDOW_AUTOSIZE)

# Load the MuJoCo model
scene_path = os.path.join(get_base_path(), "scene", "scene_with_hand.xml")
spec = mujoco.MjSpec.from_file(scene_path)

# np.random.seed(0)
load_random_grid_ycb(spec, 4)

model = spec.compile()
data = mujoco.MjData(model)

# Get the camera ID for 'overhead_cam'
cam_name = "overhead_cam"
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

# Create an OpenGL context and GLFW window (needed for offscreen rendering in MuJoCo)
glfw_window = glfw.create_window(640, 480, "mj_dummy", None, None)
glfw.make_context_current(glfw_window)

# Setup MuJoCo rendering context
gl_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Get z-buffer properties
# https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L817
extent = model.stat.extent
near = model.vis.map.znear * extent # 0.005
far = model.vis.map.zfar * extent # 30

# init mujoco viewer
# with mujoco.viewer.launch_passive(model, data) as viewer:


# Define the camera parameters (you can modify these based on your need)
# https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
cam = mujoco.MjvCamera()
cam.fixedcamid = cam_id  # Use the camera ID obtained earlier
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # Fixed camera

# Prepare to render
scene = mujoco.MjvScene(model, maxgeom=1000)

# Main simulation loop
sim_i = 0
n_grasps  = 0
n_frame_geoms = 0
cam_vis_idx = None

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
for _ in range(100):
    mujoco.mj_step(model, data)

# then sync viewer
# viewer.sync()

# update camera stuff
mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

# Render the scene to an offscreen buffer
viewport = mujoco.MjrRect(0, 0, 640, 480)
mujoco.mjr_render(viewport, scene, gl_context)

# Read pixels from the OpenGL buffer (MuJoCo renders in RGB format)
rgb_array = np.zeros((480, 640, 3), dtype=np.uint8)  # Image size: height=480, width=640
depth_array = np.zeros((480, 640), dtype=np.float32)  # Depth array
mujoco.mjr_readPixels(rgb_array, depth_array, viewport, gl_context)

# Flip the image vertically (because OpenGL origin is bottom-left)
rgb_array = np.flipud(rgb_array)
depth_array = np.flipud(depth_array)
depth_array = raw_to_metric_depth(depth_array, near, far)
depth_array_clipped = np.clip(depth_array, 0.4, 1.2)

# --- Process image --- #
image = rgb_array

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
k_d405_640x480 = CameraIntrinsic(width=640, height=480, fx=382, fy=382, cx=320, cy=240)
pc_xyz, pc_rgb = depth2pc(depth_array, k_d405_640x480, rgb_array)
pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(pc_xyz)
if rgb_array is not None:
    pcd_cam.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)

# convert PC to world frame
pcd_world = copy.deepcopy(pcd_cam).transform(cam_extrinsics)
# crop PC based on bounding box in world frame
workspace_bb = o3d.geometry.OrientedBoundingBox(np.array([0.0, 0.0, 0.225]), np.eye(3), np.array([1.2, 0.8, 0.4]))
pcd_world_crop = pcd_world.crop(workspace_bb)
# get cropped point cloud in camera frame as well
pcd_cam_crop = copy.deepcopy(pcd_world_crop).transform(np.linalg.inv(cam_extrinsics))

# TODO: create TSDF here instead of in GIGA model?



# # Convert RGB to BGR for OpenCV
# bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

# # Display the image in the OpenCV window
# cv2.imshow("MuJoCo Camera", bgr_image)

# # Show depth image
# depth_array_colored = get_depth_display(depth_array_clipped)
# cv2.imshow("Depth Map", depth_array_colored)

# # wait until button press while focused on opencv window
# cv2.waitKey(0)
# # Clean up OpenCV and GLFW
# cv2.destroyAllWindows()
# glfw.terminate()



# Show point cloud
# o3d.visualization.draw_geometries([pcd_world])



# ### CONTACT GRASPNET ###
# print('')
# print('')
# print('EVALUATING CONTACT GRASPNET')
# # load grasp generation model
# # TODO: clean up this yaml file
# with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
#     cgn_config = yaml.safe_load(f)
# cgn_config['OPTIMIZER']['batch_size'] = int(1)
# cgn_config['DATA']['checkpoint_path'] = 'planners/contact_graspnet/checkpoints/model.pt'
# cgn = ContactGraspNet(cgn_config)
# # generate grasp candidates
# # TODO: pass in grasp success threshold? take threshold from config file?
# print('Generating Grasps...')
# # TODO: is dict with key -1 best way to return these values??
# # TODO: do we ned to keep other function inputs?
# cgn_tic = time.time()
# cgn_grasp_poses_cam, cgn_grasp_scores, cgn_contact_pts, cgn_grasp_widths = cgn.predict_scene_grasps(pcd_cam_crop,
#                                                                     pc_segments={},
#                                                                     local_regions=False,
#                                                                     filter_grasps=True,
#                                                                     forward_passes=1)
# cgn_toc = time.time() - cgn_tic
# # put grasps in world frame
# cgn_grasp_poses_world_array = np.zeros_like(cgn_grasp_poses_cam[-1])
# for i,g in enumerate(cgn_grasp_poses_cam[-1]):
#     cgn_grasp_poses_world_array[i,:4,:4] = np.matmul(cam_extrinsics, g)
# cgn_grasp_poses_world = {-1: cgn_grasp_poses_world_array}
# # visualize grasps
# visualize_grasps(pcd_world_crop, cgn_grasp_poses_world, cgn_grasp_scores,
#                 window_name = 'ContactGraspNet',
#                 plot_origin=True,
#                 gripper_openings=None)


# ### EDGE GRASP ###
# print('')
# print('')
# print('EVALUATING EDGE GRASP')
# # load model
# with open('planners/edge_grasp/edge_grasp_config.yaml', 'r') as f:
#     edge_grasp_config = yaml.safe_load(f)
# edge_grasp = EdgeGraspNet(edge_grasp_config)
# # generate grasp candidates
# edge_tic = time.time()
# edge_grasp_poses_world, edge_grasp_scores, edge_grasp_widths = edge_grasp.predict_scene_grasps(pcd_world_crop)
# edge_toc = time.time() - edge_tic
# # visualize grasps
# visualize_grasps(pcd_world_crop, edge_grasp_poses_world, edge_grasp_scores,
#                 window_name = 'EdgeGrasp',
#                 plot_origin=True,
#                 gripper_openings=None)

# ### VN-EDGE GRASP ###
# # TODO: implement this


# ### GRASPNESS ###
# print('')
# print('')
# print('EVALUATING GRASPNESS')
# # load model
# with open('planners/graspness/graspness_config.yaml', 'r') as f:
#     graspness_config = yaml.safe_load(f)
# gsnet = GraspnessNet(graspness_config)
# # generate grasp candidates
# gsnet_tic = time.time()
# gsnet_grasp_poses_cam, gsnet_grasp_scores, gsnet_grasp_widths = gsnet.predict_scene_grasps(pcd_cam_crop)
# gsnet_toc = time.time() - gsnet_tic
# # put grasps in world frame
# gsnet_grasp_poses_world_array = np.zeros_like(gsnet_grasp_poses_cam[-1])
# for i,g in enumerate(gsnet_grasp_poses_cam[-1]):
#     gsnet_grasp_poses_world_array[i,:4,:4] = np.matmul(cam_extrinsics, g)
# gsnet_grasp_poses_world = {-1: gsnet_grasp_poses_world_array}
# # visualize grasps
# visualize_grasps(pcd_world_crop, gsnet_grasp_poses_world, gsnet_grasp_scores,
#                 window_name = 'Graspness',
#                 plot_origin=True,
#                 gripper_openings=None)


### GIGA PACKED ###
### GIGA PILE? ###
print('')
print('')
print('EVALUATING GIGA')
# load model
with open('planners/giga/giga_config.yaml', 'r') as f:
    giga_config = yaml.safe_load(f)
giganet = GIGANet(giga_config)
# generate grasp candidates
giga_tic = time.time()
giga_grasp_poses, giga_grasp_scores, giga_grasp_widths = giganet.predict_scene_grasps(depth_array, k_d405_640x480, cam_extrinsics)
giga_toc = time.time() - giga_tic

print(giga_grasp_poses)
print(giga_grasp_scores)
print(giga_grasp_widths)


# visualize grasps
# visualize_grasps(pcd_world_crop, giga_grasp_poses, giga_grasp_scores,
#                 window_name = 'GIGA',
#                 plot_origin=True,
#                 gripper_openings=None)
# TODO: visualize the estimated meshes too?



# ### ALL PLANNER OUTPUTS ###
# # plot world point cloud and grasps from each planner in their own color
# print('')
# print('')
# print('PLOTTING ALL PLANNER OUTPUTS')
# # print all timings
# print('')
print('Evaluation times:')
# print('ContactGraspNet: ', cgn_toc, ' seconds.')
# print('EdgeGraspNet: ', edge_toc, ' seconds.')
# print('Graspness: ', gsnet_toc, ' seconds.')
print('GIGA: ', giga_toc, ' seconds.')
# # create plotting window
# vis_grasps_many_planners(pcd_world,
#                         [cgn_grasp_poses_world, edge_grasp_poses_world, gsnet_grasp_poses_world],
#                         [(1,0,0), (0,1,0), (0,0,1)])




