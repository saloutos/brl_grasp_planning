# imports
import mujoco as mj
import numpy as np
import atexit
import tty
import termios
import sys
import os
import time
import glfw
import copy
import dill

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

# single object to grasp
# TODO: randomize (x,y) position, rpy of this object?
obj_name = "box_7"
obj_file = "primitives/single_objects/fixed/"+obj_name+".yaml"
obj_pos = [0,0,0.08]
obj_rpy = [0,0,30]
load_objects_from_yaml(spec, obj_file, pos=obj_pos, rpy=obj_rpy)

mj_model = spec.compile()

# platform
PandaGP = PandaGripperPlatform(mj_model, viewer_enable=True, setup_rendering=True, log_path=None)
PandaGP.experiment_data = {}

# controller
from controllers.execute_plan_panda.PandaGrabLiftFSM import PandaGrabLiftFSM
controller = PandaGrabLiftFSM()

# Contact Grasp Net
with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
    cgn_config = yaml.safe_load(f)
CGN = ContactGraspNet(cgn_config)
PandaGP.experiment_data['planner'] = 'CGN'

# add variables to GP
PandaGP.num_cycles = 0
# for grasp planning trigger
PandaGP.ready_to_plan = False
PandaGP.plan_complete = False
PandaGP.planned_poses = {'approach_pose': np.eye(4), 'grasp_pose': np.eye(4)}
# for saving experiment results before resetting
PandaGP.ready_to_reset = False
PandaGP.allow_reset = True # allow reset on startup

# experiment parameters
save_file = "single_object_experiment.dill"
num_plans = 3
num_perturbations = 0
total_cycles = num_plans*(1+num_perturbations)
print("Number of grasps: {}, number of perturbations: {}, number of cycles: {}".format(num_plans, num_perturbations, total_cycles))

PandaGP.experiment_data['obj_name'] = obj_name
PandaGP.experiment_data['obj_pos'] = obj_pos
PandaGP.experiment_data['obj_rpy'] = obj_rpy
for i in range(total_cycles):
    PandaGP.experiment_data[i+1] = {}


atexit.register(PandaGP.shutdown)
print("Finished init.")

# start experiment
try:
    tty.setcbreak(sys.stdin.fileno())
    PandaGP.initialize()
    controller.begin(PandaGP)
    PandaGP.apply_control()
    PandaGP.sync_viewer()
    print("Starting main loop.")
    real_start_time = time.time()
    log_start = PandaGP.time()
    while PandaGP.mj_viewer.is_running() and PandaGP.num_cycles <= total_cycles:
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
                # capture image of scene and create point cloud
                render_start = time.time()
                pcd_cam, pcd_world, cam_extrinsics, cam_intrinsics, rgb_array, depth_array = PandaGP.capture_scene("overhead_cam")
                render_time = time.time() - render_start
                print("Rendering completed in {:.2f} seconds.".format(render_time))
                # run planning
                plan_start = time.time()
                grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = CGN.predict_scene_grasps(pcd_world, cam_extrinsics)
                plan_time = time.time() - plan_start
                # grasps are sorted by default, so can probably remove this argmax
                best_grasp = np.argmax(grasp_scores)
                best_score = grasp_scores[best_grasp]
                best_width = grasp_widths[best_grasp]
                best_pose = grasp_poses_world[best_grasp,:,:]
                # save grasp pose
                PandaGP.planned_poses['grasp_pose'] = copy.deepcopy(best_pose)
                PandaGP.experiment_data[PandaGP.num_cycles].update({
                    'grasp_pose': best_pose,
                    'grasp_score': best_score,
                    'grasp_width': best_width,
                })
                # finally, visualize the grasps
                PandaGP.mj_viewer.user_scn.ngeom = 0
                new_rgb = np.random.rand(3)
                mjv_draw_grasps(PandaGP.mj_viewer, grasp_poses_world, plot_best=True, rgba=[new_rgb[0], new_rgb[1], new_rgb[2], 0.25])
                print("Planning completed in {:.2f} seconds.".format(plan_time))
                # set flag to true
                PandaGP.plan_complete = True
            # check for reset flag
            if PandaGP.ready_to_reset:
                PandaGP.ready_to_reset = False
                print("Saving data.")
                print("")
                # save experiment data
                # have to define success here

                # check object height
                obj_height = PandaGP.mj_data.body(obj_name).xpos[2]
                if obj_height > 0.05:
                    success = 1
                else:
                    success = 0
                print("Success: ", success)

                PandaGP.experiment_data[PandaGP.num_cycles].update({
                    'success': success,
                    'obj_height': obj_height,
                })

                # set flag to true
                PandaGP.allow_reset = True

    # save data
    with open(save_file, 'wb') as f:
        dill.dump(PandaGP.experiment_data, f)
    # # test by loading data
    # with open('data.dill', 'rb') as f:
    #     data = dill.load(f)
    # print(PandaGP.experiment_data)
    # print(data)

# end experiment
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, init_settings)