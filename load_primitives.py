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
scene_path = os.path.join(get_base_path(), "scene", "scene.xml")
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
load_objects_from_yaml(spec, 'object_test.yaml')
load_objects_from_yaml(spec, 'object_test_2.yaml')

model = spec.compile()
data = mujoco.MjData(model)

# Get the camera ID for 'overhead_cam'
cam_name = "overhead_cam"
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

# set up camera intrinsics
cam_fovy = np.deg2rad(model.cam_fovy[cam_id])
cam_width = 640
cam_height = 480
cam_cx = cam_width/2
cam_cy = cam_height/2
cam_f = cam_height / (2 * math.tan(cam_fovy / 2))
k_d405_640x480 = CameraIntrinsic(cam_width, cam_height, cam_f, cam_f, cam_cx, cam_cy)

# Setup MuJoCo rendering context
gl_context = mujoco.GLContext(cam_width, cam_height)
gl_context.make_current()
renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Get z-buffer properties
# https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L817
extent = model.stat.extent
near = model.vis.map.znear * extent # 0.005
far = model.vis.map.zfar * extent # 30

# init mujoco viewer
with mujoco.viewer.launch_passive(model, data) as viewer:


    # Define the camera parameters (you can modify these based on your need)
    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
    cam = mujoco.MjvCamera()
    cam.fixedcamid = cam_id  # Use the camera ID obtained earlier
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # Fixed camera

    # Prepare to render
    scene = mujoco.MjvScene(model, maxgeom=20000)

    # Main simulation loop
    sim_i = 0
    n_grasps  = 0
    n_frame_geoms = 0
    cam_vis_idx = None

    while viewer.is_running():

        # Step the simulation
        mujoco.mj_step(model, data)
        time.sleep(0.001)
        # update gripper pose and finger width
        sim_t = data.time
        # base_pos_des = np.array([0.0, 0.2, 0.8])
        # data.mocap_pos = base_pos_des
        # base_quat_des = np.zeros((4,))
        # mujoco.mju_mat2Quat(base_quat_des, np.eye(3).flatten())
        # data.mocap_quat = base_quat_des
        # data.ctrl[0] = 200 # not real units, goes from 0 to 255

        if sim_i==2000:
            index = viewer.user_scn.ngeom
            # add line?
            mujoco.mjv_initGeom(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, [0]*3, [0]*3, [0]*9, [1]*4)
            mujoco.mjv_connector(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, 5, [1,1,0], [0,0,2.0])
            viewer.user_scn.geoms[index].rgba = np.array([1.0, 0.0, 0.0, 0.9])
            viewer.user_scn.geoms[index].label = ''
            index+=1
            # update number of geoms and sync
            viewer.user_scn.ngeom = index
            print("Added line!")

        # then sync viewer
        viewer.sync()


        # # update camera stuff less often
        # if sim_i % 100 == 0:
        #     mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        #     # Render the scene to an offscreen buffer
        #     viewport = mujoco.MjrRect(0, 0, cam_width, cam_height)
        #     mujoco.mjr_render(viewport, scene, renderer)

        #     # Read pixels from the OpenGL buffer (MuJoCo renders in RGB format)
        #     rgb_array = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)  # Image size: height=480, width=640
        #     depth_array = np.zeros((cam_height, cam_width), dtype=np.float32)  # Depth array
        #     mujoco.mjr_readPixels(rgb_array, depth_array, viewport, renderer)

        #     # Flip the image vertically (because OpenGL origin is bottom-left)
        #     rgb_array = np.flipud(rgb_array)
        #     depth_array = np.flipud(depth_array)
        #     depth_array = raw_to_metric_depth(depth_array, near, far)
        #     depth_array_clipped = np.clip(depth_array, 0.4, 1.2)

        #     # --- Process image --- #
        #     image = rgb_array

        #     cam_xpos = data.cam_xpos[cam_id]
        #     cam_xmat = data.cam_xmat[cam_id]

        #     cam_extrinsics = np.eye(4)
        #     cam_extrinsics[:3, :3] = cam_xmat.reshape(3, 3)
        #     cam_extrinsics[:3, 3] = cam_xpos

        #     # -z --> +z
        #     # -y --> +y
        #     # +x --> +x
        #     T_camzforward_cam = np.array([[1, 0, 0, 0],
        #                                 [0, -1, 0, 0],
        #                                 [0, 0, -1, 0],
        #                                 [0, 0, 0, 1]])
        #     cam_extrinsics = cam_extrinsics @ T_camzforward_cam

        #     # get point cloud
        #     pc_xyz, pc_rgb = depth2pc(depth_array, k_d405_640x480, rgb_array)
        #     pcd_cam = o3d.geometry.PointCloud()
        #     pcd_cam.points = o3d.utility.Vector3dVector(pc_xyz)
        #     if rgb_array is not None:
        #         pcd_cam.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)

        #     # convert PC to world frame
        #     pcd_world = copy.deepcopy(pcd_cam).transform(cam_extrinsics)
        #     # crop PC based on bounding box in world frame
        #     workspace_bb = o3d.geometry.OrientedBoundingBox(np.array([0.0, 0.0, 0.225]), np.eye(3), np.array([1.2, 0.8, 0.4]))
        #     pcd_world_crop = pcd_world.crop(workspace_bb)
        #     # get cropped point cloud in camera frame as well
        #     pcd_cam_crop = copy.deepcopy(pcd_world_crop).transform(np.linalg.inv(cam_extrinsics))

        #     # # Convert RGB to BGR for OpenCV
        #     # bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        #     # # Display the image in the OpenCV window
        #     # cv2.imshow("MuJoCo Camera", bgr_image)
        #     # # Show depth image
        #     # depth_array_colored = get_depth_display(depth_array_clipped)
        #     # cv2.imshow("Depth Map", depth_array_colored)

        # # Check if 'q' is pressed to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        sim_i += 1


# Clean up OpenCV and GLFW
cv2.destroyAllWindows()
glfw.terminate()

