# load each primtive into mujoco in a large grid
import numpy as np
import yaml
import os
import mujoco
import mujoco.viewer
from utils import *

fnames = os.listdir('primitives/single_objects/fixed') # get list of all individual primitives
scene_path = os.path.join(get_base_path(), "scene", "scene.xml")

# create model
spec = mujoco.MjSpec.from_file(scene_path)

# for each object, load it into the scene
for j, obj in enumerate(fnames):
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
    new_x = -1.0 + 0.2*(j%10)
    new_y = -0.5 + 0.2*(j//10)
    new_z = 0.2
    data[obj_name]['pos'] = [new_x, new_y, new_z]
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

# simulate until objects reach a steady state
model = spec.compile()
data = mujoco.MjData(model)
# init mujoco viewer
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:

    # other viewer options?
    # with viewer.lock():
    #     viewer.opt.flags[mujoco.mjtVisFlag.mjVISFLAG_ACTUATORS] = False
    #     viewer.opt.flags[mujoco.mjtVisFlag.mjVISFLAG_CONTACTPOINTS] = False

    sim_i = 0
    mujoco.mj_step(model, data)
    viewer.sync()
    while viewer.is_running():
        pass
        # step simulation
        # mujoco.mj_step(model, data)
        # time.sleep(0.001)
        # sim_t = data.time
        # sim_i += 1
        # viewer.sync()

    # update positions and orientations in scene dict and save to scene yaml

