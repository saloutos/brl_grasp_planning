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

from planners.contact_graspnet.cgn_model import ContactGraspNet
import yaml

from utils import load_random_grid_ycb, get_base_path, depth2pc

def get_depth_display(depth_array):
    depth_display = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
    depth_display = (depth_display * 255).astype(np.uint8)  # Scale to 0-255 for display
    depth_display_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)  # Apply a color map
    return depth_display_colored

def raw_to_metric_depth(raw_depth, near, far):
    """
    Convert raw depth values to metric depth values
    https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L817
    """
    return near / (1 - raw_depth * (1 - near / far))

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
k_d405_640x480 = np.array([[382.418, 0, 320], [0, 382.418, 240], [0, 0, 1]])
pc = depth2pc(depth_array, k_d405_640x480, rgb_array)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[0])
if pc[1] is not None:
    pcd.colors = o3d.utility.Vector3dVector(pc[1] / 255.0)

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
# o3d.visualization.draw_geometries([pcd])




# load grasp generation model
with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
    global_config = yaml.safe_load(f)
# set forward passes to 1
global_config['OPTIMIZER']['batch_size'] = int(1)
cgn = ContactGraspNet(global_config)

# TODO: Load the weights


# TODO: process point cloud?
# # We set the camera intrinsic matrix to be the same as the D405 when we
# # create the camera in the aloha.xml
# pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
#     depth_array,
#     # depth_inferred,
#     k_d405_640x480,
#     segmap=None,
#     rgb=image,
#     skip_border_objects=False,
#     z_range=[0.05, 1.0],
# )

# generate grasp candidates
print('Generating Grasps...')
pred_grasps_cam, scores, contact_pts, _ = cgn.predict_scene_grasps(pc[0],
                                                                    pc_segments={},
                                                                    local_regions=False,
                                                                    filter_grasps=True,
                                                                    forward_passes=1)


# any post-processing?
# putting grasps back in world frame

# # visualize grasps

# TODO: put this function in high-level utils
# visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)


# # -- Plot grasps -- #
# cm2 = plt.get_cmap('viridis')
# n_grasps = 0
# n_frame_geoms = 0
# for i, k in enumerate(pred_grasps_cam):
#     if np.any(pred_grasps_cam[k]):
#         grasps = pred_grasps_cam[k]
#         # grasps are T_cam_final_grasp
#         # cam_extrinsics is T_world_cam
#         # I want T_world_finalgrasp
#         for i in range(len(grasps)):
#             grasps[i] = cam_extrinsics @ grasps[i]
#         # grasps_world = np.einsum("ijk, kl->ijl", grasps, cam_extrinsics)
#         # grasps_world = np.einsum("ijk, lj->ilk", grasps, np.linalg.inv(cam_extrinsics))
#         grasps_world = grasps
#         max_score = np.max(scores[k])
#         min_score = np.min(scores[k])
#         topk_idx = np.argsort(scores[k])
#         # max_idx = np.argmax(scores[k])
#         # best_grasp = grasps_world[max_idx]
#         colors = [cm2((score - min_score) / (max_score - min_score))[:3] for score in scores[k]]
#         for grasp, color in zip(grasps_world, colors):
#             rgba = list(color) + [1]
#             draw_grasp(viewer.user_scn, grasp, 0.1, rgba)
#             n_grasps += 1











