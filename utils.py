import numpy as np
import mujoco
import time
import os
import trimesh
import trimesh.transformations as tra
import open3d as o3d

# TODO: get rid of this function
def get_base_path():
    file_path = os.path.abspath(__file__)
    base_dir_path = os.path.dirname(file_path)
    return base_dir_path

def load_ycb_obj(spec: mujoco.MjSpec, obj_name, initial_pos=[0.0, 0.0, 1.0]):
    """
    Load an object from the YCB dataset into the scene.

    Args:
        spec (mujoco.MjSpec): The specification of the scene.
        scene_path (str): The path to the scene file.
        obj_name (str): The name of the object to load.

    Returns:
        None
    """

    # Get absolute paths
    absolute_object_path = os.path.join(get_base_path(), "ycb", obj_name)
    data_type = "processed"
    obj_path = os.path.join(absolute_object_path, data_type, "textured.obj")
    texture_path = os.path.join(absolute_object_path, data_type, "textured.png")

    # Add mesh
    mesh = spec.add_mesh()
    mesh.name = obj_name
    mesh.file = obj_path

    # Add texture
    texture = spec.add_texture()
    texture.nchannel = 3
    texture.name = obj_name + "_texture"
    texture.file = texture_path
    texture.type = mujoco._enums.mjtTexture.mjTEXTURE_2D

    # Add material
    material = spec.add_material()
    material.name = obj_name + "_material"
    textures = [""] * 10
    textures[1] = obj_name + "_texture"
    material.textures = textures

    # Add geom
    body = spec.worldbody.add_body()
    joint = body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = obj_name
    geom.material = obj_name + "_material"
    geom.pos = initial_pos


    # Append to keyframe
    # if len(spec.keys) != 0:
    #     current_qpos = spec.keys[0].qpos.tolist()
    #     obj_qpos = initial_pos + [0.0, 0.0, 0.0, 1.0]
    #     new_qpos = current_qpos + obj_qpos
    #     spec.keys[0].qpos = new_qpos

    return None

def get_all_ycb_obj_fnames():
    obj_path = os.path.join(get_base_path(), "ycb")
    fnames = os.listdir(obj_path)
    with open(os.path.join(obj_path, "blacklist.txt"), "r") as f:
        blacklist = f.read().split("\n")
    blacklist.append("blacklist.txt")
    good_fnames = [f for f in fnames if f not in blacklist]
    return good_fnames

def load_random_grid_ycb(spec: mujoco.MjSpec, grid_side_length=4):
    total_length = 0.4

    edge = np.linspace(-total_length / 2, total_length / 2, grid_side_length)
    coordinate_grid_x, coordinate_grid_y = np.meshgrid(edge, edge)
    coordinate_grid_x = coordinate_grid_x.flatten()
    coordinate_grid_y = coordinate_grid_y.flatten()

    all_objects = get_all_ycb_obj_fnames()
    selected_objects = np.random.choice(all_objects, grid_side_length ** 2, replace=False)

    for i in range(len(coordinate_grid_x)):
        xi = coordinate_grid_x[i]
        yi = coordinate_grid_y[i]
        obj_name = selected_objects[i]
        load_ycb_obj(spec, obj_name, initial_pos=[xi, yi, 0.05 + np.random.rand() * 0.1])

    return True


def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where((depth > 0.0) & (depth < 2.0))
    x,y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]

    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)


