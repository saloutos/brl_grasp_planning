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


def single_object_experiment(obj_name, obj_nominal_pose, planner_name, num_plans):

    # initialization
    print("Starting init.")
    init_settings = termios.tcgetattr(sys.stdin)

    exp_tic = time.time()

    # Initialize GLFW
    glfw.init()

    # load model here
    scene_path = os.path.join(get_base_path(), "scene", "scene_with_panda_hand.xml")
    spec = mj.MjSpec.from_file(scene_path)

    # single object to grasp
    # TODO: randomize (x,y) position, rpy of this object?
    obj_file = "primitives/single_objects/fixed/"+obj_name+".yaml"
    obj_pos = obj_nominal_pose[0]
    obj_rpy = obj_nominal_pose[1]
    load_objects_from_yaml(spec, obj_file, pos=obj_pos, rpy=obj_rpy)

    mj_model = spec.compile()

    # platform
    PandaGP = PandaGripperPlatform(mj_model, viewer_enable=True, setup_rendering=True, log_path=None)
    PandaGP.experiment_data = {}

    # controller
    from controllers.execute_plan_panda.PandaGrabLiftFSM import PandaGrabLiftFSM
    controller = PandaGrabLiftFSM()

    # planners
    PandaGP.experiment_data['planner'] = planner_name
    if planner_name == 'CGN':
        # Contact Grasp Net
        with open('planners/contact_graspnet/cgn_config.yaml','r') as f:
            cgn_config = yaml.safe_load(f)
        planner = ContactGraspNet(cgn_config)
    elif planner_name == 'EDGE':
        # Edge Grasp
        with open('planners/edge_grasp/edge_grasp_config.yaml', 'r') as f:
            edge_grasp_config = yaml.safe_load(f)
        planner = EdgeGraspNet(edge_grasp_config)
    elif planner_name == 'GSN':
        # GS Net
        with open('planners/graspness/graspness_config.yaml', 'r') as f:
            graspness_config = yaml.safe_load(f)
        planner = GraspnessNet(graspness_config)
    elif planner_name == 'GIGA':
        # GIGA
        with open('planners/giga/giga_config.yaml', 'r') as f:
            giga_config = yaml.safe_load(f)
        planner = GIGANet(giga_config)
    else:
        raise ValueError("Invalid planner name: {}".format(planner_name))
    
    # add variables to GP
    PandaGP.num_cycles = 0
    # for grasp planning trigger
    PandaGP.ready_to_plan = False
    PandaGP.plan_complete = False
    PandaGP.planned_poses = {'approach_pose': np.eye(4), 'grasp_pose': np.eye(4)}
    # for saving experiment results before resetting
    PandaGP.ready_to_reset = False
    PandaGP.allow_reset = True # allow reset on startup

    # experiment data
    # perturbations, to be applied in grasp pose frame
    perturbations = [
        create_SE3([-0.01, 0.0, 0.0]),
        create_SE3([0.01, 0.0, 0.0]),
        create_SE3([0.0, -0.01, 0.0]),
        create_SE3([0.0, 0.01, 0.0]),
        create_SE3([0.0, 0.0, -0.01]),
        create_SE3([0.0, 0.0, 0.01]),
        create_SE3([0.0, 0.0, 0.0], [-10, 0, 0]),
        create_SE3([0.0, 0.0, 0.0], [10, 0, 0]),
        create_SE3([0.0, 0.0, 0.0], [0, -10, 0]),
        create_SE3([0.0, 0.0, 0.0], [0, 10, 0]),
        create_SE3([0.0, 0.0, 0.0], [0, 0, -10]),
        create_SE3([0.0, 0.0, 0.0], [0, 0, 10]),
    ]
    num_perturbations = len(perturbations)

    total_cycles = num_plans*(1+num_perturbations)
    print("Number of grasps: {}, number of perturbations: {}, number of cycles: {}".format(num_plans, num_perturbations, total_cycles))

    PandaGP.plan_perturbation = 0
    PandaGP.experiment_data['obj_name'] = obj_name
    PandaGP.experiment_data['obj_pos'] = obj_pos
    PandaGP.experiment_data['obj_rpy'] = obj_rpy
    for i in range(total_cycles):
        PandaGP.experiment_data[i+1] = {}

    base_successes = []
    perturb_successes = []

    # turn off real time sim
    PandaGP.enforce_real_time_sim = False

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
                    # check perturbation counter
                    if PandaGP.plan_perturbation == 0:
                        # capture image of scene and create point cloud
                        render_start = time.time()
                        pcd_cam, pcd_world, cam_extrinsics, cam_intrinsics, rgb_array, depth_array = PandaGP.capture_scene("overhead_cam")
                        render_time = time.time() - render_start
                        print("Rendering completed in {:.2f} seconds.".format(render_time))
                        # run planning
                        plan_start = time.time()
                        if planner_name == 'CGN':
                            grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = planner.predict_scene_grasps(pcd_world, cam_extrinsics)
                        elif planner_name == 'EDGE':
                            grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = planner.predict_scene_grasps(pcd_world)
                        elif planner_name == 'GSN':
                            grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = planner.predict_scene_grasps(pcd_world, cam_extrinsics)
                        elif planner_name == 'GIGA':
                            grasp_poses_world, grasp_scores, grasp_widths, pcd_world_out = planner.predict_scene_grasps(depth_array, cam_intrinsics, cam_extrinsics, pcd_world)
                        plan_time = time.time() - plan_start
                        # grasps are sorted by default, so can probably remove this argmax
                        best_grasp = np.argmax(grasp_scores)
                        best_score = grasp_scores[best_grasp]
                        best_width = grasp_widths[best_grasp]
                        best_pose = grasp_poses_world[best_grasp,:,:]
                        # save grasp pose
                        PandaGP.planned_poses['grasp_pose'] = copy.deepcopy(best_pose)
                        PandaGP.experiment_data[PandaGP.num_cycles].update({
                            'perturbation': False,
                            'grasp_pose': copy.deepcopy(best_pose),
                            'grasp_score': copy.deepcopy(best_score),
                            'grasp_width': copy.deepcopy(best_width),
                        })
                        # finally, visualize the grasps
                        PandaGP.mj_viewer.user_scn.ngeom = 0
                        new_rgb = np.random.rand(3)
                        mjv_draw_grasps(PandaGP.mj_viewer, grasp_poses_world, plot_best=True, rgba=[new_rgb[0], new_rgb[1], new_rgb[2], 0.25])
                        print("Planning completed in {:.2f} seconds.".format(plan_time))
                        # incremement perturbation counter
                        PandaGP.plan_perturbation += 1
                    else:
                        print("Applying perturbation: ", PandaGP.plan_perturbation-1)
                        # get best pose again
                        new_pose = copy.deepcopy(best_pose)
                        # TODO: make perturbations be SE(3) transforms
                        new_pose =  np.matmul(new_pose, perturbations[PandaGP.plan_perturbation-1])

                        # save grasp pose
                        PandaGP.planned_poses['grasp_pose'] = copy.deepcopy(new_pose)
                        PandaGP.experiment_data[PandaGP.num_cycles].update({
                            'perturbation': True,
                            'grasp_pose': copy.deepcopy(new_pose),
                            'grasp_score': copy.deepcopy(best_score),
                            'grasp_width': copy.deepcopy(best_width),
                        })
                        # finally, visualize the grasps
                        PandaGP.mj_viewer.user_scn.ngeom = 0
                        mjv_draw_grasps(PandaGP.mj_viewer, [new_pose, best_pose], rgba=[new_rgb[0], new_rgb[1], new_rgb[2], 0.25])
                        # increment perturbation counter, check for roll over
                        PandaGP.plan_perturbation += 1
                        if PandaGP.plan_perturbation > num_perturbations:
                            PandaGP.plan_perturbation = 0
                    # set flag to true
                    PandaGP.plan_complete = True
                # check for reset flag
                if PandaGP.ready_to_reset:
                    PandaGP.ready_to_reset = False
                    print("Saving data.")
                    # save experiment data
                    # have to define success here

                    # check object height
                    obj_height = PandaGP.mj_data.body(obj_name).xpos[2]
                    if obj_height > 0.1:
                        success = 1
                    else:
                        success = 0
                    print("Success: ", success)

                    PandaGP.experiment_data[PandaGP.num_cycles].update({
                        'success': success,
                        'obj_height': obj_height,
                    })

                    if PandaGP.plan_perturbation == 1:
                        base_successes.append(success)
                    else:
                        perturb_successes.append(success)

                    print("")
                    # set flag to true
                    PandaGP.allow_reset = True

        # save data
        # with open(save_file, 'wb') as f:
        #     dill.dump(PandaGP.experiment_data, f)
        # # test by loading data
        # with open('data.dill', 'rb') as f:
        #     data = dill.load(f)
        # print(PandaGP.experiment_data)
        # print(data)

        print("Base Successes: ", sum(base_successes), "/", len(base_successes))
        print("Perturbation Successes: ", sum(perturb_successes), "/", len(perturb_successes))
        print("All successes: ", sum(base_successes) + sum(perturb_successes), "/", len(base_successes) + len(perturb_successes))

        successes = [sum(base_successes)/len(base_successes), sum(perturb_successes)/len(perturb_successes),
                    (sum(base_successes)+sum(perturb_successes))/(len(base_successes)+len(perturb_successes))]

        exp_toc = time.time() - exp_tic
        print("Experiment completed in {:.2f} seconds.".format(exp_toc))

    # end experiment
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, init_settings)

    # return values
    return PandaGP.experiment_data, successes