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

from planners.contact_graspnet.cgn_model import ContactGraspNet
from utils import *

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

# TODO: where is the best place to put this?
Z_RANGE = [0.4, 1.2]

depth_array_clipped = np.clip(depth_array, Z_RANGE[0], Z_RANGE[1])

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
k_d405_640x480 = np.array([[382.418, 0, 320], [0, 382.418, 240], [0, 0, 1]])
pc_xyz, pc_rgb = depth2pc(depth_array, k_d405_640x480, rgb_array)

# filter pc based on z-range
if rgb_array is not None:
    pc_rgb = pc_rgb[(pc_xyz[:,2] < Z_RANGE[1]) & (pc_xyz[:,2] > Z_RANGE[0])]
pc_xyz = pc_xyz[(pc_xyz[:,2] < Z_RANGE[1]) & (pc_xyz[:,2] > Z_RANGE[0])]

# for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_xyz)
if rgb_array is not None:
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)

# Convert RGB to BGR for OpenCV
bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

# Display the image in the OpenCV window
cv2.imshow("MuJoCo Camera", bgr_image)

# Show depth image
depth_array_colored = get_depth_display(depth_array_clipped)
cv2.imshow("Depth Map", depth_array_colored)

# wait until button press while focused on opencv window
cv2.waitKey(0)
# Clean up OpenCV and GLFW
cv2.destroyAllWindows()
glfw.terminate()

# Show point cloud
o3d.visualization.draw_geometries([pcd])

# load grasp generation model
# TODO: clean up this yaml file
with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
    global_config = yaml.safe_load(f)
# set forward passes to 1 TODO: is this necessary?
global_config['OPTIMIZER']['batch_size'] = int(1)
# set model path here just in case
global_config['DATA']['checkpoint_path'] = 'planners/contact_graspnet/checkpoints/model.pt'
# create model (weights are loaded in init)
cgn = ContactGraspNet(global_config)

# generate grasp candidates
# TODO: pass in grasp success threshold?
print('Generating Grasps...')
pred_grasps_cam, scores, contact_pts, pred_grasp_widths = cgn.predict_scene_grasps(pc_xyz,
                                                                    pc_segments={},
                                                                    local_regions=False,
                                                                    filter_grasps=True,
                                                                    forward_passes=1)

# TODO: any post-processing? putting grasps back in world frame? and point cloud?

# visualize grasps
visualize_grasps(pc_xyz, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_rgb, gripper_openings=None)