import numpy as np
import mujoco
import time
import os
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import mujoco.viewer

# TODO: get rid of this function
def get_base_path():
    file_path = os.path.abspath(__file__)
    base_dir_path = os.path.dirname(file_path)
    return base_dir_path

### PRIMITIVE OBJECT STUFF ###

# load random primitive objects (in grid?) (random orientations?)
# TODO: could also randomly load from single object yaml files?
def load_random_grid_primitives(spec: mujoco.MjSpec, grid_side_length=4):
    total_length = 0.4
    obj_mass_range = [0.1, 1.0]
    obj_size_0_range = [0.02, 0.04] # radius or cube side, nominal is 0.03
    obj_size_1_range = [0.04, 0.08] # height, nominal is 0.06

    edge = np.linspace(-total_length / 2, total_length / 2, grid_side_length)
    coordinate_grid_x, coordinate_grid_y = np.meshgrid(edge, edge)
    coordinate_grid_x = coordinate_grid_x.flatten()
    coordinate_grid_y = coordinate_grid_y.flatten()

    # iterate through grid
    # randomly sample object type, size, mass
    for i in range(len(coordinate_grid_x)):
        # sample object type (0: box, 1: cylinder, 2: sphere)
        temp_type = np.random.randint(0, 3)
        obj_mass = np.random.uniform(obj_mass_range[0], obj_mass_range[1]) # default: 0.2
        if (temp_type==0):
            # box
            obj_type = mujoco.mjtGeom.mjGEOM_BOX
            size_0 = np.random.uniform(obj_size_0_range[0], obj_size_0_range[1])
            size_1 = np.random.uniform(obj_size_0_range[0], obj_size_0_range[1])
            size_2 = np.random.uniform(obj_size_0_range[0], obj_size_0_range[1])
            obj_size = [size_0, size_1, size_2]
        elif (temp_type==1):
            # cylinder
            obj_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            size_0 = np.random.uniform(obj_size_0_range[0], obj_size_0_range[1])
            size_1 = np.random.uniform(obj_size_1_range[0], obj_size_1_range[1])
            obj_size = [size_0, size_1, 0]
        elif (temp_type==2):
            # sphere
            obj_type = mujoco.mjtGeom.mjGEOM_SPHERE
            size_0 = np.random.uniform(obj_size_0_range[0], obj_size_0_range[1])
            obj_size = [size_0, size_0, size_0]
        # actually load object
        obj_name = 'test_obj_'+str(i+1)
        obj_pos = [coordinate_grid_x[i], coordinate_grid_y[i], 0.2] # TODO: add some random noise to z coord?
        obj_rgba = list(np.random.rand(3)) + [1.0] # can also randomize color
        obj_quat = [0, 0, 0, 1] # TODO: also randomize orientation!!!
        load_single_primitive(spec, obj_name, obj_pos, obj_type=obj_type, size=obj_size, mass=obj_mass, rgba=obj_rgba, quat=obj_quat)


# load objects from yaml description
def load_objects_from_yaml(spec: mujoco.MjSpec, yaml_file_path):
    # TODO: take yaml path or contents?
    # can define multiple objects per yaml, or load single object from many yamls? or both
    a = 1




# load primitive object with type, size, mass, friction, name, color, at pos, quat and attach to world spec
def load_single_primitive(spec: mujoco.MjSpec,
                        name,
                        pos,
                        obj_type=mujoco.mjtGeom.mjGEOM_SPHERE, # only allow sphere, cylinder, box
                        size=[0.1, 0.1, 0.1], # always needs 3 elements, but unused elements can be 0
                        mass=0.1,
                        friction=[1.0, 0.02, 0.0005], # sliding, torsion, rolling
                        rgba=[0.9, 0.1, 0.1, 1.0],
                        quat=[0.0, 0.0, 0.0, 1.0]):

    # Add body, free joint, and new geom
    body = spec.worldbody.add_body()
    joint = body.add_freejoint()
    geom = body.add_geom()

    # Set some default params of geom, body, and joint
    geom.group = 1
    geom.contype = 1
    geom.conaffinity = 1
    geom.condim = 6
    geom.priority = 2
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2]
    geom.solref = [0.005, 1]

    # NOTE: remember that these properties exist, but we won't mess with them for now
    # geom.pos = [0, 0, 0]
    # geom.quat = [0, 0, 0, 1]
    # geom.margin = 0
    # geom.gap = 0
    # NOTE: for free joint, won't have pos and axis, joint limits, or friction
    joint.group = 2
    joint.stiffness = 0
    joint.damping = 0
    joint.frictionloss = 0
    joint.armature = 0

    # TODO: check that inputs are valid?

    # Set specificed params
    geom.name = name
    geom.rgba = rgba
    geom.type = obj_type
    geom.size = size
    # TODO: should friction be a default?
    geom.friction = friction
    body.name = name
    body.mass = mass
    # based on type, calculate inertias
    # NOTE: can also set position and orientation of inertial frame
    # body.ipos = [0, 0, 0]
    # body.iquat = [0, 0, 0, 1]
    if obj_type==mujoco.mjtGeom.mjGEOM_SPHERE:
        i = (2/5)*mass*(size[0]**2)
        body.inertia = [i, i, i]
    elif obj_type==mujoco.mjtGeom.mjGEOM_CYLINDER:
        i_xy = (1/4)*mass*size[0]**2 + 1/12*mass*size[1]**2
        i_z = (1/2)*mass*size[0]**2
        body.inertia = [i_xy, i_xy, i_z]
    elif obj_type==mujoco.mjtGeom.mjGEOM_BOX:
        i_x = (1/12)*mass*((2*size[1])**2 + (2*size[2])**2)
        i_y = (1/12)*mass*((2*size[0])**2 + (2*size[2])**2)
        i_z = (1/12)*mass*((2*size[0])**2 + (2*size[1])**2)
        body.inertia = [i_x, i_y, i_z]
    else:
        print("Not a valid type of object.")
        # set inertias anyways?
        body.inertia = [0.1, 0.1, 0,1]

    # finally, set position and orientation of object
    # TODO: should this be body pos?
    body.pos = pos
    body.quat = quat

    # TODO: Append to keyframe?
    # if len(spec.keys) != 0:
    #     current_qpos = spec.keys[0].qpos.tolist()
    #     obj_qpos = initial_pos + initial_quat
    #     new_qpos = current_qpos + obj_qpos
    #     spec.keys[0].qpos = new_qpos


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

    # TODO: should this be body pos?
    body.pos = initial_pos


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

def depth2pc(depth, intrinsic, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param intrinsic: Camera intrinsic class, intrinsic.K is 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where((depth > 0.0) & (depth < 2.0))
    x,y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - intrinsic.K[0,2])
    normalized_y = (y.astype(np.float32) - intrinsic.K[1,2])

    world_x = normalized_x * depth[y, x] / intrinsic.K[0,0]
    world_y = normalized_y * depth[y, x] / intrinsic.K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]

    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb

class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


### GRASP PLOTTING STUFF ###

def vis_grasps_many_planners(pcd_input, pred_grasps=[], pred_grasp_colors=[], other_frames=[]):
    """
    Visualizes point cloud and grasps from many planners.
    Generalization of visualize_grasps function.
    """
    gripper_width=0.08

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Many Planners")
    vis.add_geometry(pcd_input)

    # plot the origin?
    # plot_coordinates(vis, np.zeros(3,),np.eye(3,3), central_color=(0.5, 0.5, 0.5))

    # plot other frames
    for f in other_frames:
        plot_coordinates(vis, f[:3, 3], f[:3,:3])

    # for each set of grasps, plot with corresponding color
    # only works for non-segmented point cloud right now
    for i in range(len(pred_grasps)):
        pred_grasps_indiv = pred_grasps[i]
        color_indiv = pred_grasp_colors[i]
        for i,k in enumerate(pred_grasps_indiv):
            gripper_openings_k = np.ones(len(pred_grasps_indiv[k]))*gripper_width

            draw_grasps(vis,
                        pred_grasps_indiv[k],
                        np.eye(4),
                        colors=[color_indiv]*len(pred_grasps_indiv[k]),
                        gripper_openings=gripper_openings_k)

    vis.run()
    vis.destroy_window()
    return



def visualize_grasps(pcd_cam, pred_grasps_cam, scores, window_name='Open3D', plot_origin=False, gripper_openings=None, gripper_width=0.08, plot_others=[]):
    """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions.
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot origin coordinate frame (default: {False})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
    """

    print('Visualizing...')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd_cam)

    if plot_origin:
        plot_coordinates(vis, np.zeros(3,),np.eye(3,3), central_color=(0.5, 0.5, 0.5))

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

                # # plot grasp frames for debugging
                # for g in pred_grasps_cam[k]:
                #     plot_coordinates(vis, g[:3,3], g[:3,:3])

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

def mjv_draw_grasps(viewer, grasps, cam_pose=np.eye(4), rgba=[0.0, 1.0, 0.0, 0.1]):
    # TODO: add gripper openings, multiple colors back in?
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

    index = viewer.user_scn.ngeom
    # for i, (g,g_opening) in enumerate(zip(grasps, gripper_openings)):
    for i, g in enumerate(grasps):
        gripper_control_points_closed = grasp_line_plot.copy()
        g_opening = 0.08
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * g_opening/2

        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]

        # line 1
        mujoco.mjv_connector(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, 2, pts[0,:], pts[1,:])
        viewer.user_scn.geoms[index].rgba = np.array(rgba)
        viewer.user_scn.geoms[index].label = ''
        index+=1
        # line 2
        mujoco.mjv_connector(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, 2, pts[2,:], pts[3,:])
        viewer.user_scn.geoms[index].rgba = np.array(rgba)
        viewer.user_scn.geoms[index].label = ''
        index+=1
        # line 3
        mujoco.mjv_connector(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, 2, pts[2,:], pts[5,:])
        viewer.user_scn.geoms[index].rgba = np.array(rgba)
        viewer.user_scn.geoms[index].label = ''
        index+=1
        # line 4
        mujoco.mjv_connector(viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_LINE, 2, pts[5,:], pts[6,:])
        viewer.user_scn.geoms[index].rgba = np.array(rgba)
        viewer.user_scn.geoms[index].label = ''
        index+=1

    # update number of geoms and sync
    viewer.user_scn.ngeom = index
    viewer.sync()





