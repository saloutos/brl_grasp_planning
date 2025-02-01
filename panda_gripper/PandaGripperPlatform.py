# imports
import time
import sys
import yaml
import numpy as np
import csv
import math
from datetime import datetime as dt
import select
import os
from enum import Enum
import copy
import open3d as o3d

import mujoco as mj
import mujoco.viewer as mjv

from .PandaGripperData import *

# define possible platform modes
class PlatformMode(Enum):
    # TODO: change order here, default to 0 for simulation with vis
    SIM_WITH_VIS    = 2
    SIM_NO_VIS      = 3


# define the PANDA Gripper Platform class
class PandaGripperPlatform:
    def __init__(self, mj_model, viewer_enable=True, setup_rendering=False, log_path=None):
        # based on enable flags, set platform mode
        # TODO: might not need to save flags as class variables, should use mode for everything from here onwards?
        # TODO: pass modes as arguments instead of flags, check modes everywhere? then can re-set mode bewteen init and initialize()
        self.viewer_enable = viewer_enable
        if self.viewer_enable:
            self.mode = PlatformMode.SIM_WITH_VIS
        else:
            self.mode = PlatformMode.SIM_NO_VIS

        # load mujoco model and data
        self.mj_model = mj_model
        self.mj_data = mj.MjData(self.mj_model)

        # gripper data init
        # TODO: if GripperData() takes list of joints and sensors as arguments, then pass them here
        self.gr_data = PandaGripperData()

        # general init for platform
        self.paused = False
        self.current_t = 0.0
        self.last_view_t = 0.0
        self.last_control_t = 0.0
        # self.sim_dt = 0.001
        # self.mj_model.opt.timestep = self.sim_dt # re-assign in initialize for safety
        self.sim_dt = self.mj_model.opt.timestep
        self.view_dt = 0.0333 # 30fps default
        self.control_dt = 0.002 # 500Hz default
        self.enforce_real_time_sim = True # default to real-time sim
        self.dt_comp = 0.0
        self.sim_steps_per_control = math.floor(self.control_dt/self.sim_dt) # re-calculate in initialize for safety
        self.run_viewer_sync = False
        self.run_control = False
        self.char_in = None
        self.new_char = False

        # setup for rendering
        # NOTE: rendering is for grasp planning, needs to be set up here before simulation is started
        self.setup_rendering = setup_rendering
        if setup_rendering:
            # Get the camera ID for 'overhead_cam'
            self.cam_name = "overhead_cam"
            self.cam_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, self.cam_name)

            # set up camera intrinsics
            self.cam_fovy = np.deg2rad(self.mj_model.cam_fovy[self.cam_id])
            self.cam_width = 640
            self.cam_height = 480
            self.cam_cx = self.cam_width/2
            self.cam_cy = self.cam_height/2
            self.cam_f = self.cam_height / (2 * math.tan(self.cam_fovy / 2))
            self.cam_K = np.array([[self.cam_f, 0.0, self.cam_cx], [0.0, self.cam_f, self.cam_cy], [0.0, 0.0, 1.0]])
            # k_d405_640x480 = CameraIntrinsic(cam_width, cam_height, cam_f, cam_f, cam_cx, cam_cy)

            # Setup MuJoCo rendering context
            self.gl_context = mj.GLContext(self.cam_width, self.cam_height)
            self.gl_context.make_current()
            self.renderer = mj.MjrContext(self.mj_model, mj.mjtFontScale.mjFONTSCALE_150)

            # Get z-buffer properties
            # https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L817
            z_extent = self.mj_model.stat.extent
            self.z_near = self.mj_model.vis.map.znear * z_extent # 0.005
            self.z_far = self.mj_model.vis.map.zfar * z_extent # 30

            # Define the camera parameters (you can modify these based on your need)
            # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
            self.cam = mj.MjvCamera()
            self.cam.fixedcamid = self.cam_id  # Use the camera ID obtained earlier
            self.cam.type = mj.mjtCamera.mjCAMERA_FIXED  # Fixed camera

            # Prepare to render
            self.scene = mj.MjvScene(self.mj_model, maxgeom=20000)
            self.viewport = mj.MjrRect(0, 0, self.cam_width, self.cam_height)


        # general init for logging
        self.log_enable = (log_path is not None)
        if self.log_enable:
            # TODO: better log name convention here?
            self.log_name = log_path+"log_"+str(dt.now()).replace(" ", "_")+".csv"
            self.log_header = ['t']+self.gr_data.log_header()
            self.log = [ [0]+self.gr_data.log_data() ]
            self.log_file = open(self.log_name, mode='w')
            self.log_writer = csv.writer(self.log_file, delimiter=',')
            self.log_writer.writerows([self.log_header])
            self.log_start = 0.0 # will udpate this in initialize later


    def initialize(self):

        # prepare mujoco model
        mj.mj_forward(self.mj_model, self.mj_data)

        # start viewer
        if self.mode==PlatformMode.SIM_WITH_VIS:
            self.mj_viewer = mjv.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False, key_callback=self.key_callback)
            with self.mj_viewer.lock():
                # set viewer options here
                self.mj_viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
                # can also tweak visualization elements here
                # TODO
                self.mj_viewer.cam.distance = 2.0
                self.mj_viewer.cam.elevation = -20
                self.mj_viewer.cam.azimuth = 135
                self.mj_viewer.cam.lookat = np.array([-0.1, 0.1, 0.15])
                if self.mode==PlatformMode.SIM_WITH_VIS:
                    self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                    self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                    self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTSPLIT] = True
                    self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_SELECT] = True
                    self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = False
                    # enable viewing groups (0,1,2 are enabled by default)
                    # self.mj_viewer.opt.geomgroup[3] = True

                mj.mj_forward(self.mj_model, self.mj_data)
            print("Viewer started.")

        # TODO: step simulation to get initial state? or does mj_forward take care of this?
        # mj.step(self.mj_model, self.mj_data, nstep=1)

        # just in case, re-calculate sim steps per control period and update model timestep
        # this should not change after this point
        if self.sim_dt != self.mj_model.opt.timestep:
            self.mj_model.opt.timestep = self.sim_dt
        self.sim_steps_per_control = math.floor(self.control_dt/self.sim_dt)

        # save log start time
        if self.log_enable:
            self.log_start = self.time()

    # def initialize_rendering(self):
    #     # set up camera stuff for planning here

    #     # TODO: add an "enable rendering" flag here?



    def shutdown(self):
        # close viewer
        if self.mode==PlatformMode.SIM_WITH_VIS:
            self.mj_viewer.close()
            print("Viewer closed.")
        # log recorded data
        if self.log_enable:
            print("Logging data.")
            self.log_writer.writerows(self.log)
            self.log_file.close()
            print("Log saved.")
        print("Shutdown.")

    def log_data(self, extra_data=None):
        if self.log_enable:
            t_log = self.time() - self.log_start
            log_line = [t_log] + self.gr_data.log_data()
            # TODO: log some sim data?
            if extra_data is not None:
                log_line = log_line + extra_data
            self.log.append(log_line)

    def check_user_input(self):
        # with no viewer, need to check terminal input
        if self.mode==PlatformMode.SIM_NO_VIS:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                self.char_in = sys.stdin.read(1)
            else:
                self.char_in = None
        else: # make sure we only receive mujoco callback input once
            if self.new_char==True:
                self.new_char = False
            else:
                self.char_in = None
    def key_callback(self, keycode):
        # handle keyboard inputs
        self.char_in = chr(keycode)
        self.new_char = True
        if self.char_in==' ' and self.mode==PlatformMode.SIM_WITH_VIS: # can only pause in sim with viewer
            self.paused = not self.paused # toggle pause
            print(f"Paused: {self.paused}")

    def time(self):
        if (self.mode==PlatformMode.SIM_WITH_VIS or self.mode==PlatformMode.SIM_NO_VIS) and not self.enforce_real_time_sim:
            return self.mj_data.time
        else:
            return time.time()

    def sync_viewer(self):
        # TODO: is this check even necessary? might end up redundant
        if (self.mode==PlatformMode.SIM_WITH_VIS) and self.mj_viewer.is_running():
            # TODO: update any other visual elements here
            # self.mj_viewer.user_scn.ngeom = 0
            # sync sensor visualizations?
            # sync mujoco viewer
            self.mj_viewer.sync()

    def step(self):
        # get current time
        self.previous_t = self.current_t
        self.current_t = self.time()
        # print(self.current_t-self.previous_t)

        # if mode has viewer
        if self.mode==PlatformMode.SIM_WITH_VIS:
            # check if viewer flag needs to be set
            if self.current_t - self.last_view_t > self.view_dt:
                self.last_view_t = self.current_t
                self.run_viewer_sync = True


        # if simulation mode
        if self.mode==PlatformMode.SIM_WITH_VIS or self.mode==PlatformMode.SIM_NO_VIS:
            # holding controller inputs constant for control dt, so simulate all steps in one go
            # simulate for self.sim_steps_per_control
            mj.mj_step(self.mj_model, self.mj_data, nstep=self.sim_steps_per_control)
            # for real-time sim, wait for control_dt since start of step()
            if self.enforce_real_time_sim:
                while (self.time()-self.current_t < self.control_dt-self.dt_comp): continue
            # update timing and set control flag
            self.last_control_t = self.current_t
            self.run_control = True

    def sync_data(self):
        # check for user input
        self.check_user_input()

        # TODO: store some data from Panda gripper here

        # always update some kinematic data
        # TODO: update names here
        wrist_p = self.mj_data.body('hand').xpos
        wrist_R = self.mj_data.body('hand').xmat.reshape((3,3))
        l_dip_tip_p = self.mj_data.body('left_finger').xpos
        l_dip_tip_R = self.mj_data.body('left_finger').xmat.reshape((3,3))
        r_dip_tip_p = self.mj_data.body('right_finger').xpos
        r_dip_tip_R = self.mj_data.body('right_finger').xmat.reshape((3,3))
        self.gr_data.kinematics['base']['p'] = wrist_p
        self.gr_data.kinematics['base']['R'] = wrist_R
        self.gr_data.kinematics['left_finger']['p'] = l_dip_tip_p
        self.gr_data.kinematics['left_finger']['R'] = l_dip_tip_R
        self.gr_data.kinematics['right_finger']['p'] = r_dip_tip_p
        self.gr_data.kinematics['right_finger']['R'] = r_dip_tip_R

        # update sensor kinematics
        # TODO: better way to do this? initialize with corresponding site name for kinematics?
        # TODO: re-implement this if we add sensors to panda

        # fill gr_data from mj_data
        # start with just joint info (q, qd, tau), access by name
        # TODO: should this iterate through model joints first? then gr_data joint keys?
        mj_joints = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        for key in self.gr_data.joints.keys():
            self.gr_data.joints[key].q = self.mj_data.joint(key).qpos
            self.gr_data.joints[key].qd = self.mj_data.joint(key).qvel
            self.gr_data.joints[key].tau = self.mj_data.joint(key).qfrc_actuator
        # get contact location data for fingertips
        # TODO: re-implement this if we add sensors to panda

        # update sensor data (i.e. apply filters, etc.)
        # TODO: uncomment this when we have sensors
        # self.gr_data.process_all_sensor_data()

    def apply_control(self):
        # update internal value of tau_command for each joint
        base_quat_des = np.zeros((4,))

        # TODO: apply control using GR data here

        # # update position and orientation of wrist mocap body
        # # TODO: should this be for simulation only? once arm is added, mocap pose will be set from arm forward kinematics?
        self.mj_data.mocap_pos = self.gr_data.kinematics['base_des']['p']
        mj.mju_mat2Quat(base_quat_des, self.gr_data.kinematics['base_des']['R'].flatten())
        self.mj_data.mocap_quat = base_quat_des

        # # if simulation mode
        # update actuator commands based on gr_data tau_command, will be applied during next mj_step call
        # TODO: update this
        self.mj_data.ctrl = self.gr_data.get_ctrl(self.gr_data.all_idxs)

    def capture_scene(self):

        # need to make current every time
        self.gl_context.make_current()
        self.mj_viewer.sync()

        # update camera stuff
        mj.mjv_updateScene(self.mj_model, self.mj_data, mj.MjvOption(), None, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scene)

        # Render the scene to an offscreen buffer
        mj.mjr_render(self.viewport, self.scene, self.renderer)

        # Read pixels from the OpenGL buffer (MuJoCo renders in RGB format)
        rgb_array = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)  # Image size: height=480, width=640
        depth_array = np.zeros((self.cam_height, self.cam_width), dtype=np.float32)  # Depth array
        mj.mjr_readPixels(rgb_array, depth_array, self.viewport, self.renderer)

        # Flip the image vertically (because OpenGL origin is bottom-left)
        rgb_array = np.flipud(rgb_array)
        depth_array = np.flipud(depth_array)

        # convert from raw depth to metric depth
        # depth_array = raw_to_metric_depth(depth_array, self.z_near, self.z_far)
        depth_array = self.z_near / (1 - depth_array * (1 - self.z_near / self.z_far))

        # --- Process image --- #
        cam_xpos = self.mj_data.cam_xpos[self.cam_id]
        cam_xmat = self.mj_data.cam_xmat[self.cam_id]
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
        normalized_x = (x.astype(np.float32) - self.cam_K[0,2])
        normalized_y = (y.astype(np.float32) - self.cam_K[1,2])
        world_x = normalized_x * depth_array[y, x] / self.cam_K[0,0]
        world_y = normalized_y * depth_array[y, x] / self.cam_K[1,1]
        world_z = depth_array[y, x]
        pc_xyz = np.vstack((world_x, world_y, world_z)).T

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

        return pcd_cam_crop, pcd_world_crop, cam_extrinsics, rgb_array, depth_array