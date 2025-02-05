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

# np.random.seed(0)
# CHOOSE TO LOAD YCB MESHES OR RANDOM PRIMITIVES?
# load_random_grid_ycb(spec, 4)
# load_random_grid_fixed_primitives(spec, 4)
# single object to grasp
# load_objects_from_yaml(spec, "primitives/single_objects/fixed/box_7.yaml", pos=[0,0,0.05], rpy=[90,30,30])
# load_objects_from_yaml(spec, "primitives/single_objects/fixed/cylinder_3.yaml", pos=[0,0,0.05], rpy=[0,0,0])
load_objects_from_yaml(spec, 'primitives/collections/panda_graspable/scene_2.yaml')

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

    # capture frame from camera
    render_tic = time.time()
    pcd_cam, pcd_world, cam_extrinsics, cam_intrinsics, rgb_array, depth_array = capture_scene(viewer, model, data, "overhead_cam")
    pcd_cam2, pcd_world2, cam_extrinsics2, cam_intrinsics2, rgb_array2, depth_array2 = capture_scene(viewer, model, data, "overhead_cam2")
    pcd_cam3, pcd_world3, cam_extrinsics3, cam_intrinsics3, rgb_array3, depth_array3 = capture_scene(viewer, model, data, "overhead_cam3")
    pcd_cam4, pcd_world4, cam_extrinsics4, cam_intrinsics4, rgb_array4, depth_array4 = capture_scene(viewer, model, data, "overhead_cam4")
    # merge world point clouds
    full_pcd_world = pcd_world + pcd_world2 + pcd_world3 + pcd_world4
    full_pcd_world = full_pcd_world.voxel_down_sample(voxel_size=0.002) # TODO: tune this downsample parameter?
    # could crop and then convert full world pc to any camera frame here
    render_toc = time.time() - render_tic

    # ### CONTACT GRASPNET ###
    print('')
    print('')
    print('EVALUATING CONTACT GRASPNET')
    # load grasp generation model
    with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
        cgn_config = yaml.safe_load(f)
    cgn = ContactGraspNet(cgn_config)
    # generate grasp candidates
    # TODO: pass in grasp success threshold? take threshold from config file?
    print('Generating Grasps...')
    cgn_tic = time.time()
    cgn_grasp_poses_world1, cgn_grasp_scores1, cgn_contact_pts1, cgn_grasp_widths1, cgn_pcd_world1 = cgn.predict_scene_grasps(pcd_world, cam_extrinsics)
    # cgn_grasp_poses_world2, cgn_grasp_scores2, cgn_contact_pts2, cgn_grasp_widths2, cgn_pcd_world2 = cgn.predict_scene_grasps(pcd_world2, cam_extrinsics2)
    # cgn_grasp_poses_world3, cgn_grasp_scores3, cgn_contact_pts3, cgn_grasp_widths3, cgn_pcd_world3 = cgn.predict_scene_grasps(pcd_world3, cam_extrinsics3)
    # cgn_grasp_poses_world4, cgn_grasp_scores4, cgn_contact_pts4, cgn_grasp_widths4, cgn_pcd_world4 = cgn.predict_scene_grasps(pcd_world4, cam_extrinsics4)
    cgn_toc = time.time() - cgn_tic

    # can also add contact points to visualzation
    # cgn_contact_pts1 += 0.0005 # small displacement to avoid overlap with object
    # cgn_contact_pts1_pcd = o3d.geometry.PointCloud()
    # cgn_contact_pts1_pcd.points = o3d.utility.Vector3dVector(cgn_contact_pts1)
    # cgn_contact_pts1_pcd.colors = o3d.utility.Vector3dVector(np.array([[0,0,0]]*len(cgn_contact_pts1)))
    # pcd_world_with_pts = pcd_world_crop + cgn_contact_pts1_pcd

    # visualize grasps
    # visualize_grasps(cgn_pcd_world1, cgn_grasp_poses_world1, cgn_grasp_scores1,
    #                 window_name = 'ContactGraspNet',
    #                 plot_origin=True,
    #                 gripper_openings=cgn_grasp_widths1)
    # visualize_grasps(cgn_pcd_world2, cgn_grasp_poses_world2, cgn_grasp_scores2,
    #                 window_name = 'ContactGraspNet2',
    #                 plot_origin=True,
    #                 gripper_openings=cgn_grasp_widths2)
    # visualize_grasps(cgn_pcd_world3, cgn_grasp_poses_world3, cgn_grasp_scores3,
    #                 window_name = 'ContactGraspNet3',
    #                 plot_origin=True,
    #                 gripper_openings=cgn_grasp_widths3)
    # visualize_grasps(cgn_pcd_world4, cgn_grasp_poses_world4, cgn_grasp_scores4,
    #                 window_name = 'ContactGraspNet4',
    #                 plot_origin=True,
    #                 gripper_openings=cgn_grasp_widths4)

    # also visualize grasps in mujoco
    mjv_draw_grasps(viewer, cgn_grasp_poses_world1, rgba=[1,0,0, 0.25])
    # mjv_draw_grasps(viewer, cgn_grasp_poses_world2, rgba=[0,1,0, 0.25])
    # mjv_draw_grasps(viewer, cgn_grasp_poses_world3, rgba=[0,0,1, 0.25])
    # mjv_draw_grasps(viewer, cgn_grasp_poses_world4, rgba=[0,0.7,1, 0.25])

    # vis_grasps_many_planners(full_pcd_world,
    #                         [cgn_grasp_poses_world1, cgn_grasp_poses_world2, cgn_grasp_poses_world3, cgn_grasp_poses_world4],
    #                         [(1,0,0), (1, 0.6, 0.1), (0,1,0), (0,0.7,1)])


    # ### EDGE GRASP ###
    print('')
    print('')
    print('EVALUATING EDGE GRASP')
    # load model
    with open('planners/edge_grasp/edge_grasp_config.yaml', 'r') as f:
        edge_grasp_config = yaml.safe_load(f)
    edge_grasp = EdgeGraspNet(edge_grasp_config)
    # generate grasp candidates
    edge_tic = time.time()
    edge_grasp_poses_world1, edge_grasp_scores1, edge_grasp_widths1, edge_pcd_world1 = edge_grasp.predict_scene_grasps(pcd_world)
    # edge_grasp_poses_world2, edge_grasp_scores2, edge_grasp_widths2, edge_pcd_world2 = edge_grasp.predict_scene_grasps(pcd_world2)
    # edge_grasp_poses_world3, edge_grasp_scores3, edge_grasp_widths3, edge_pcd_world3 = edge_grasp.predict_scene_grasps(pcd_world3)
    # edge_grasp_poses_world4, edge_grasp_scores4, edge_grasp_widths4, edge_pcd_world4 = edge_grasp.predict_scene_grasps(pcd_world4)
    edge_toc = time.time() - edge_tic

    # visualize grasps
    # visualize_grasps(edge_pcd_world1, edge_grasp_poses_world1, edge_grasp_scores1,
    #                 window_name = 'EdgeGrasp',
    #                 plot_origin=True,
    #                 gripper_openings=edge_grasp_widths1)
    # visualize_grasps(edge_pcd_world2, edge_grasp_poses_world2, edge_grasp_scores2,
    #                 window_name = 'EdgeGrasp2',
    #                 plot_origin=True,
    #                 gripper_openings=edge_grasp_widths2)
    # visualize_grasps(edge_pcd_world3, edge_grasp_poses_world3, edge_grasp_scores3,
    #                 window_name = 'EdgeGrasp3',
    #                 plot_origin=True,
    #                 gripper_openings=edge_grasp_widths3)
    # visualize_grasps(edge_pcd_world4, edge_grasp_poses_world4, edge_grasp_scores4,
    #                 window_name = 'EdgeGrasp4',
    #                 plot_origin=True,
    #                 gripper_openings=edge_grasp_widths4)

    # also visualize grasps in mujoco
    mjv_draw_grasps(viewer, edge_grasp_poses_world1, rgba=[1, 0.6, 0.1, 0.25])
    # mjv_draw_grasps(viewer, edge_grasp_poses_world2, rgba=[0,1,0, 0.25])
    # mjv_draw_grasps(viewer, edge_grasp_poses_world3, rgba=[0,0,1, 0.25])
    # mjv_draw_grasps(viewer, edge_grasp_poses_world4, rgba=[0,0.7,1, 0.25])

    # vis_grasps_many_planners(full_pcd_world,
    #                         [edge_grasp_poses_world1, edge_grasp_poses_world2, edge_grasp_poses_world3, edge_grasp_poses_world4],
    #                         [(1,0,0), (1, 0.6, 0.1), (0,1,0), (0,0.7,1)])




    # ### VN-EDGE GRASP ###
    # # TODO: implement this


    ### GRASPNESS ###
    print('')
    print('')
    print('EVALUATING GRASPNESS')
    # load model
    with open('planners/graspness/graspness_config.yaml', 'r') as f:
        graspness_config = yaml.safe_load(f)
    gsnet = GraspnessNet(graspness_config)
    # generate grasp candidates
    gsnet_tic = time.time()
    gsnet_grasp_poses_world1, gsnet_grasp_scores1, gsnet_grasp_widths1, gsnet_pcd_world1 = gsnet.predict_scene_grasps(pcd_world, cam_extrinsics)
    # gsnet_grasp_poses_world2, gsnet_grasp_scores2, gsnet_grasp_widths2, gsnet_pcd_world2 = gsnet.predict_scene_grasps(pcd_world2, cam_extrinsics2)
    # gsnet_grasp_poses_world3, gsnet_grasp_scores3, gsnet_grasp_widths3, gsnet_pcd_world3 = gsnet.predict_scene_grasps(pcd_world3, cam_extrinsics3)
    # gsnet_grasp_poses_world4, gsnet_grasp_scores4, gsnet_grasp_widths4, gsnet_pcd_world4 = gsnet.predict_scene_grasps(pcd_world4, cam_extrinsics4)
    gsnet_toc = time.time() - gsnet_tic
    # visualize grasps
    # visualize_grasps(gsnet_pcd_world1, gsnet_grasp_poses_world1, gsnet_grasp_scores1,
    #                 window_name = 'Graspness',
    #                 plot_origin=True,
    #                 gripper_openings=gsnet_grasp_widths1)
    # visualize_grasps(gsnet_pcd_world2, gsnet_grasp_poses_world2, gsnet_grasp_scores2,
    #                 window_name = 'Graspness2',
    #                 plot_origin=True,
    #                 gripper_openings=gsnet_grasp_widths2)
    # visualize_grasps(gsnet_pcd_world3, gsnet_grasp_poses_world3, gsnet_grasp_scores3,
    #                 window_name = 'Graspness3',
    #                 plot_origin=True,
    #                 gripper_openings=gsnet_grasp_widths3)
    # visualize_grasps(gsnet_pcd_world4, gsnet_grasp_poses_world4, gsnet_grasp_scores4,
    #                 window_name = 'Graspness4',
    #                 plot_origin=True,
    #                 gripper_openings=gsnet_grasp_widths4)
    # also visualize grasps in mujoco
    mjv_draw_grasps(viewer, gsnet_grasp_poses_world1, rgba=[0.0, 1.0, 0.0, 0.25])
    # mjv_draw_grasps(viewer, gsnet_grasp_poses_world2, rgba=[0,1,1, 0.25])
    # mjv_draw_grasps(viewer, gsnet_grasp_poses_world3, rgba=[0,0,1, 0.25])
    # mjv_draw_grasps(viewer, gsnet_grasp_poses_world4, rgba=[0,0.7,1, 0.25])

    ### GIGA PACKED? ###
    ### GIGA PILE? ###
    # print('')
    # print('')
    # print('EVALUATING GIGA')
    # # load model
    # with open('planners/giga/giga_config.yaml', 'r') as f:
    #     giga_config = yaml.safe_load(f)
    # giganet = GIGANet(giga_config)
    # # generate grasp candidates
    # giga_tic = time.time()
    # # choose how many cameras to use here
    # depths = [depth_array, depth_array2, depth_array3, depth_array4]
    # cam_poses = [cam_extrinsics, cam_extrinsics2, cam_extrinsics3, cam_extrinsics4]
    # giga_grasp_poses_world, giga_grasp_scores, giga_grasp_widths = giganet.predict_scene_grasps(depths, cam_intrinsics, cam_poses)
    # giga_toc = time.time() - giga_tic
    # # visualize grasps
    # # visualize_grasps(pcd_world_crop, giga_grasp_poses_world, giga_grasp_scores,
    # #                 window_name = 'GIGA',
    # #                 plot_origin=True,
    # #                 gripper_openings=None)
    # # TODO: visualize the estimated meshes too?
    # # also visualize grasps in mujoco
    # mjv_draw_grasps(viewer, giga_grasp_poses_world, rgba=[0,0.7,1, 0.5])


    ### ALL PLANNER OUTPUTS ###
    # plot world point cloud and grasps from each planner in their own color
    print('')
    print('')
    print('PLOTTING ALL PLANNER OUTPUTS')
    # print all timings
    print('')
    print('Evaluation times:')
    print('Rendering: {:.3f} seconds.'.format(render_toc))
    print('ContactGraspNet: {:.3f} seconds.'.format(cgn_toc))
    print('EdgeGraspNet: {:.3f} seconds.'.format(edge_toc))
    print('Graspness: {:.3f} seconds.'.format(gsnet_toc))
    # print('GIGA: {:.3f} seconds.'.format(giga_toc))
    # create plotting window
    # vis_grasps_many_planners(full_pcd_world,
    #                         [cgn_grasp_poses_world1, edge_grasp_poses_world, gsnet_grasp_poses_world, giga_grasp_poses_world],
    #                         [(1,0,0), (1, 0.6, 0.1), (0,1,0), (0,0.7,1)])

    while viewer.is_running(): pass

