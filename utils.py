import numpy as np
import mujoco
import time
import os
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

# TODO: get rid of this function
def get_base_path():
    file_path = os.path.abspath(__file__)
    base_dir_path = os.path.dirname(file_path)
    return base_dir_path

### YCB OBJECT STUFF ###
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

### DEPTH CAMERA STUFF ###

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
    return pc, rgb




### GRASP PLOTTING STUFF ###

def visualize_grasps(pcd_cam, pred_grasps_cam, scores, window_name='Open3D', plot_origin=False, gripper_openings=None, gripper_width=0.08,
                        T_world_cam=np.eye(4), plot_others=[]):
    """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions.
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
    """

    print('Visualizing...')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd_cam)

    if plot_origin:
        plot_coordinates(vis, np.zeros(3,),np.eye(3,3), central_color=(0.5, 0.5, 0.5))
        # This is world in cam frame
        T_cam_world = np.linalg.inv(T_world_cam)  # We plot everything in the camera frame
        t = T_cam_world[:3,3]
        r = T_cam_world[:3,:3]
        plot_coordinates(vis, t, r)

    for t in plot_others:
        plot_coordinates(vis, t[:3, 3], t[:3,:3])

    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('viridis')

    colors = [cm(1. * i/len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k:cm2(0.5*np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}

    for i,k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            # Set gripper openings
            if gripper_openings is None:
                gripper_openings_k = np.ones(len(pred_grasps_cam[k]))*gripper_width
            else:
                gripper_openings_k = gripper_openings[k]

            if len(pred_grasps_cam) > 1:
                draw_grasps(vis, pred_grasps_cam[k], np.eye(4), colors=[colors[i]], gripper_openings=gripper_openings_k)
                draw_grasps(vis, [pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), colors=[colors2[k]],
                            gripper_openings=[gripper_openings_k[np.argmax(scores[k])]], tube_radius=0.0025)
            else:
                max_score = np.max(scores[k])
                min_score = np.min(scores[k])

                colors3 = [cm2((score - min_score) / (max_score - min_score))[:3] for score in scores[k]]
                draw_grasps(vis, pred_grasps_cam[k], np.eye(4), colors=colors3, gripper_openings=gripper_openings_k)
                best_grasp_idx = np.argmax(scores[k])
                draw_grasps(vis, [pred_grasps_cam[k][best_grasp_idx]], np.eye(4), colors=[(1, 0, 0)], gripper_openings=gripper_openings_k)

    vis.run()
    vis.destroy_window()
    return

def plot_coordinates(vis, t, r, tube_radius=0.005, central_color=None):
    """
    Plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """

    # Create a line for each axis of the coordinate frame
    lines = []
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue

    if central_color is not None:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        ball.paint_uniform_color(np.array(central_color))
        vis.add_geometry(ball)

    for i in range(3):
        line_points = [[t[0], t[1], t[2]],
                       [t[0] + 0.2 * r[0, i], t[1] + 0.2 * r[1, i], t[2] + 0.2 * r[2, i]]]

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(line_points)
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        line.colors = o3d.utility.Vector3dVector(np.array([colors[i]]))

        line.paint_uniform_color(colors[i])  # Set line color
        lines.append(line)

    # Visualize the lines in the Open3D visualizer
    for line in lines:
        vis.add_geometry(line)

def draw_grasps(vis, grasps, cam_pose, gripper_openings, colors=[(0, 1., 0)]):
    """
    Draws wireframe grasps from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    # TODO: could clean up a little bit?
    # TODO: correct file path?
    control_points = np.load('panda.npy')[:, :3]
    control_points = [[0, 0, 0], control_points[0, :], control_points[1, :], control_points[-2, :], control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points[1:3, 2] = 0.0584
    control_points = np.tile(np.expand_dims(control_points, 0), [1, 1, 1])
    gripper_control_points = control_points.squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    all_pts = []
    connections = []
    index = 0
    N = 7
    for i, (g,g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * g_opening/2

        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]

        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5), np.arange(index + 1, index + N - .5)]).T)
        index += N

    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_pts)
    line_set.lines = o3d.utility.Vector2iVector(connections)

    if len(colors) == 1:
        colors = np.vstack(colors).astype(np.float64)
        colors = np.repeat(colors, len(grasps), axis=0)
    elif len(colors) == len(grasps):
        colors = np.vstack(colors).astype(np.float64)
    else:
        raise ValueError('Number of colors must be 1 or equal to number of grasps')
    colors = np.repeat(colors, N - 1, axis=0)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)
