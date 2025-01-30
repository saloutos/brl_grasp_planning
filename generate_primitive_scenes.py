import numpy as np
import yaml
import os

# will need mujoco
import mujoco
import mujoco.viewer
from utils import *

# some setup
num_scenes = 10 # generate this many scenes
ke_thresh = 1e-6 # threshold for kinetic energy to determine stability

fnames = os.listdir('primitives/single_objects/fixed') # get list of all individual primitives
scene_path = os.path.join(get_base_path(), "scene", "scene_with_enclosure.xml")

for i in range(num_scenes): # TODO: this needs to wrap entire thing!!
    # create model
    spec = mujoco.MjSpec.from_file(scene_path)

    # pull in objects
    num_objects = np.random.choice([7,8,9,10]) # choose between 6 and 10 objects

    # sample with replacement? then can iterate through and check for panda compatibility if necessary
    chosen_object_files = np.random.choice(fnames, num_objects, replace=False)

    # print scene info header
    print('')
    print('Scene ' + str(i+1) + ':')
    print(chosen_object_files)
    print('')
    # dict for full scene
    scene = {}

    # for each object, load it into the scene
    for j, obj in enumerate(chosen_object_files):
        with open('primitives/single_objects/fixed/'+obj, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            obj_name = list(data.keys())[0]
        # get primitive type
        if data[obj_name]['type']==0:
            obj_type = mujoco.mjtGeom.mjGEOM_BOX
        elif data[obj_name]['type']==1:
            obj_type = mujoco.mjtGeom.mjGEOM_CYLINDER
        elif data[obj_name]['type']==2:
            obj_type = mujoco.mjtGeom.mjGEOM_SPHERE
        # what to do with position and orientation????
        new_x = np.random.uniform(-0.1, 0.1)
        new_y = np.random.uniform(-0.1, 0.1)
        new_z = 0.1*(j+1)
        new_roll = np.random.uniform(-180, 180)
        new_pitch = np.random.uniform(-180, 180)
        new_yaw = np.random.uniform(-180, 180)
        data[obj_name]['pos'] = [new_x, new_y, new_z]
        data[obj_name]['rpy'] = [new_roll, new_pitch, new_yaw]
        # load object into mujoco scene
        conv_quat = Rotation.from_euler('zyx', data[obj_name]['rpy'], degrees=True).as_quat()
        load_single_primitive(spec,
                            obj_name,
                            data[obj_name]['pos'],
                            obj_type=obj_type,
                            size=data[obj_name]['size'],
                            mass=data[obj_name]['mass'],
                            rgba=data[obj_name]['rgba'],
                            quat=conv_quat)
        scene.update(data)

    # print final dict
    chosen_object_names = list(scene.keys())
    print(chosen_object_names)
    # print(scene)

    # simulate until objects reach a steady state
    # to limit velocities, turn down gravity?
    # spec.option.gravity = np.array([0, 0, -0.01])

    model = spec.compile()
    data = mujoco.MjData(model)
    # init mujoco viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_i = 0
        mujoco.mj_step(model, data)
        while viewer.is_running():
            # step simulation
            mujoco.mj_step(model, data)
            # time.sleep(0.001)
            sim_t = data.time
            sim_i += 1
            viewer.sync()

            # monitor energy and time
            if sim_i % 100 == 0:
                print('Time: {:.3f}, Kinetic Energy: {:6f}'.format(data.time, data.energy[1]))
            # check if objects are stable?
            if data.energy[1] < ke_thresh:
                break

    # update positions and orientations in scene dict and save to scene yaml
    print('')
    print('{} Objects:'.format(num_objects))
    for obj_name in chosen_object_names:
        # get positions and orientations of each object
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        pos = data.xpos[obj_id]
        quat = data.xquat[obj_id]
        print(obj_id, obj_name, pos, quat)

        # update scene dict
        # add a small z-height increase to avoid initializing scene with contacts
        scene[obj_name]['pos'] = [float(pos[0]), float(pos[1]), float(1.01*pos[2])]
        scene[obj_name]['quat'] = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        # remove rpy from scene keys since we are using quat for orientation now
        scene[obj_name].pop('rpy', None)

    # write to yaml
    with open('primitives/collections/scene_'+str(i)+'.yaml', 'a') as file:
        yaml.dump(scene, file)
    # test
    with open('primitives/collections/scene_'+str(i)+'.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        # print(data)
