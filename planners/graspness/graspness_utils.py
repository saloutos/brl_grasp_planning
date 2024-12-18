import torch
from torch.autograd import Function
import torch.nn as nn
import sys
import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
import copy

# TODO: remove this dependency?
from grasp_nms import nms_grasp



# TODO: move these to config file
GRASP_MAX_WIDTH = 0.1
GRASPNESS_THRESHOLD = 0.1
NUM_VIEW = 300
NUM_ANGLE = 12
NUM_DEPTH = 4
M_POINT = 1024


# NOTE: GraspGroup, utility functions from GraspNetAPI seem pretty nice, could try to use more of them!


# TODO: try to remove this function as a dependency?
import collections.abc as container_abcs
import MinkowskiEngine as ME

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch = np.ascontiguousarray(coordinates_batch, dtype=np.int32)
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res



def pred_decode(end_points):

    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds


def generate_grasp_views(N=300, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi ** 2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi ** 2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch

        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y, 1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.linalg.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']  # (B, M_point, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        poses = end_points['object_poses_list'][i]  # [(3, 4),]

        # get merged grasp points for label computation
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]  # (Np, 3)
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]  # (Np, V, A, D)
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]  # (Np, V, A, D)
            _, V, A, D = grasp_scores.size()
            num_grasp_points = grasp_points.size(0)
            # generate and transform template grasp views
            grasp_views = generate_grasp_views(V).to(pose.device)  # (V, 3)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

            # assign views
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # (V, 3, 3)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)  # (Np, V, 3, 3)
            grasp_scores = torch.index_select(grasp_scores, 1, view_inds)  # (Np, V, A, D)
            grasp_widths = torch.index_select(grasp_widths, 1, view_inds)  # (Np, V, A, D)
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_views_rot_merged.append(grasp_views_rot_trans)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # (Np', V, 3, 3)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # (Np', V, A, D)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # (Np', V, A, D)

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Ns)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Np')
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1  # (Ns)

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)  # (Ns, 3)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)  # (Ns, V, 3, 3)
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)  # (Ns, V, A, D)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)  # (Ns, V, A, D)

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(grasp_views_rot_merged)
        batch_grasp_scores.append(grasp_scores_merged)
        batch_grasp_widths.append(grasp_widths_merged)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)  # (B, Ns, V, 3, 3)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, V, A, D)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, V, A, D)

    # compute view graspness
    view_u_threshold = 0.6
    view_grasp_num = 48
    batch_grasp_view_valid_mask = (batch_grasp_scores <= view_u_threshold) & (batch_grasp_scores > 0) # (B, Ns, V, A, D)
    batch_grasp_view_valid = batch_grasp_view_valid_mask.float()
    batch_grasp_view_graspness = torch.sum(torch.sum(batch_grasp_view_valid, dim=-1), dim=-1) / view_grasp_num  # (B, Ns, V)
    view_graspness_min, _ = torch.min(batch_grasp_view_graspness, dim=-1)  # (B, Ns)
    view_graspness_max, _ = torch.max(batch_grasp_view_graspness, dim=-1)
    view_graspness_max = view_graspness_max.unsqueeze(-1).expand(-1, -1, 300)  # (B, Ns, V)
    view_graspness_min = view_graspness_min.unsqueeze(-1).expand(-1, -1, 300)  # same shape as batch_grasp_view_graspness
    batch_grasp_view_graspness = (batch_grasp_view_graspness - view_graspness_min) / (view_graspness_max - view_graspness_min + 1e-5)

    # process scores
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)  # (B, Ns, V, A, D)
    batch_grasp_scores[~label_mask] = 0

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_view_rot'] = batch_grasp_views_rot
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_grasp_view_graspness

    return end_points


def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [torch.FloatTensor, (N,3)]
                points in original coordinates
            transform: [torch.FloatTensor, (3,3)/(3,4)/(4,4)]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [torch.FloatTensor, (N,3)]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = torch.matmul(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = cloud.new_ones(cloud.size(0), device=cloud.device).unsqueeze(-1)
        cloud_ = torch.cat([cloud, ones], dim=1)
        cloud_transformed = torch.matmul(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def knn(ref, query, k=1):
    """ K-nearest neighbor search.
    Args:
        ref: reference points (1, N, C)
        query: query points (1, M, C)
        k: number of neighbors
    """

    kd_ref = torch.squeeze( ref, dim=0).cpu().T
    kd_query = torch.squeeze( query, dim=0).cpu().T
    kd_tree = KDTree(kd_ref, leaf_size=100)
    _, kd_inds = kd_tree.query(kd_query, k=k)

    return torch.from_numpy( kd_inds.squeeze() + 1 ).long().cuda()


def match_grasp_view_and_label(end_points):
    """ Slice grasp labels according to predicted views. """
    top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
    template_views_rot = end_points['batch_grasp_view_rot']  # (B, Ns, V, 3, 3)
    grasp_scores = end_points['batch_grasp_score']  # (B, Ns, V, A, D)
    grasp_widths = end_points['batch_grasp_width']  # (B, Ns, V, A, D, 3)

    B, Ns, V, A, D = grasp_scores.size()
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_view_grasp_scores = torch.gather(grasp_scores, 2, top_view_inds_).squeeze(2)
    top_view_grasp_widths = torch.gather(grasp_widths, 2, top_view_inds_).squeeze(2)

    u_max = top_view_grasp_scores.max()
    po_mask = top_view_grasp_scores > 0
    po_mask_num = torch.sum(po_mask)
    if po_mask_num > 0:
        u_min = top_view_grasp_scores[po_mask].min()
        top_view_grasp_scores[po_mask] = torch.log(u_max / top_view_grasp_scores[po_mask]) / (torch.log(u_max / u_min) + 1e-6)

    end_points['batch_grasp_score'] = top_view_grasp_scores  # (B, Ns, A, D)
    end_points['batch_grasp_width'] = top_view_grasp_widths  # (B, Ns, A, D)

    return top_template_views_rot, end_points



GRASP_ARRAY_LEN = 17
EPS = 1e-8

class Grasp():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the score, width, height, depth, rotation_matrix, translation, object_id

        - the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        - the length of the numpy array is 17.
        '''
        if len(args) == 0:
            self.grasp_array = np.array([0, 0.02, 0.02, 0.02, 1, 0, 0, 0, 1 ,0 , 0, 0, 1, 0, 0, 0, -1], dtype = np.float64)
        elif len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == 7:
            score, width, height, depth, rotation_matrix, translation, object_id = args
            self.grasp_array = np.concatenate([np.array((score, width, height, depth)),rotation_matrix.reshape(-1), translation, np.array((object_id)).reshape(-1)]).astype(np.float64)
        else:
            raise ValueError('only 1 or 7 arguments are accepted')

    def __repr__(self):
        return 'Grasp: score:{}, width:{}, height:{}, depth:{}, translation:{}\nrotation:\n{}\nobject id:{}'.format(self.score, self.width, self.height, self.depth, self.translation, self.rotation_matrix, self.object_id)

    @property
    def score(self):
        '''
        **Output:**

        - float of the score.
        '''
        return float(self.grasp_array[0])

    @score.setter
    def score(self, score):
        '''
        **input:**

        - float of the score.
        '''
        self.grasp_array[0] = score

    @property
    def width(self):
        '''
        **Output:**

        - float of the width.
        '''
        return float(self.grasp_array[1])

    @width.setter
    def width(self, width):
        '''
        **input:**

        - float of the width.
        '''
        self.grasp_array[1] = width

    @property
    def height(self):
        '''
        **Output:**

        - float of the height.
        '''
        return float(self.grasp_array[2])

    @height.setter
    def height(self, height):
        '''
        **input:**

        - float of the height.
        '''
        self.grasp_array[2] = height

    @property
    def depth(self):
        '''
        **Output:**

        - float of the depth.
        '''
        return float(self.grasp_array[3])

    @depth.setter
    def depth(self, depth):
        '''
        **input:**

        - float of the depth.
        '''
        self.grasp_array[3] = depth

    @property
    def rotation_matrix(self):
        '''
        **Output:**

        - np.array of shape (3, 3) of the rotation matrix.
        '''
        return self.grasp_array[4:13].reshape((3,3))

    @rotation_matrix.setter
    def rotation_matrix(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of matrix

        - len(args) == 9: float of matrix
        '''
        if len(args) == 1:
            self.grasp_array[4:13] = np.array(args[0],dtype = np.float64).reshape(9)
        elif len(args) == 9:
            self.grasp_array[4:13] = np.array(args,dtype = np.float64)

    @property
    def translation(self):
        '''
        **Output:**

        - np.array of shape (3,) of the translation.
        '''
        return self.grasp_array[13:16]

    @translation.setter
    def translation(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of x, y, z

        - len(args) == 3: float of x, y, z
        '''
        if len(args) == 1:
            self.grasp_array[13:16] = np.array(args[0],dtype = np.float64)
        elif len(args) == 3:
            self.grasp_array[13:16] = np.array(args,dtype = np.float64)

    @property
    def object_id(self):
        '''
        **Output:**

        - int of the object id that this grasp grasps
        '''
        return int(self.grasp_array[16])

    @object_id.setter
    def object_id(self, object_id):
        '''
        **Input:**

        - int of the object_id.
        '''
        self.grasp_array[16] = object_id

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)

        **Output:**

        - Grasp instance after transformation, the original Grasp will also be changed.
        '''
        rotation = T[:3,:3]
        translation = T[:3,3]
        self.translation = np.dot(rotation, self.translation.reshape((3,1))).reshape(-1) + translation
        self.rotation_matrix = np.dot(rotation, self.rotation_matrix)
        return self

    def to_open3d_geometry(self, color=None):
        '''
        **Input:**

        - color: optional, tuple of shape (3) denotes (r, g, b), e.g., (1,0,0) for red

        **Ouput:**

        - list of open3d.geometry.Geometry of the gripper.
        '''
        return plot_gripper_pro_max(self.translation, self.rotation_matrix, self.width, self.depth, score = self.score, color = color)

class GraspGroup():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be (1) nothing (2) numpy array of grasp group array (3) str of the npy file.
        '''
        if len(args) == 0:
            self.grasp_group_array = np.zeros((0, GRASP_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.grasp_group_array = args[0]
            elif isinstance(args[0], str):
                self.grasp_group_array = np.load(args[0])
            else:
                raise ValueError('args must be nothing, numpy array or string.')
        else:
            raise ValueError('args must be nothing, numpy array or string.')

    def __len__(self):
        '''
        **Output:**

        - int of the length.
        '''
        return len(self.grasp_group_array)

    def __repr__(self):
        repr = '----------\nGrasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for grasp_array in self.grasp_group_array:
                repr += Grasp(grasp_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += Grasp(self.grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += Grasp(self.grasp_group_array[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        **Input:**

        - index: int, slice, list or np.ndarray.

        **Output:**

        - if index is int, return Grasp instance.

        - if index is slice, np.ndarray or list, return GraspGroup instance.
        '''
        if type(index) == int:
            return Grasp(self.grasp_group_array[index])
        elif type(index) == slice:
            graspgroup = GraspGroup()
            graspgroup.grasp_group_array = copy.deepcopy(self.grasp_group_array[index])
            return graspgroup
        elif type(index) == np.ndarray:
            return GraspGroup(self.grasp_group_array[index])
        elif type(index) == list:
            return GraspGroup(self.grasp_group_array[index])
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for GraspGroup'.format(type(index)))

    @property
    def scores(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the scores.
        '''
        return self.grasp_group_array[:,0]

    @scores.setter
    def scores(self, scores):
        '''
        **Input:**

        - scores: numpy array of shape (-1, ) of the scores.
        '''
        assert scores.size == len(self)
        self.grasp_group_array[:,0] = copy.deepcopy(scores)

    @property
    def widths(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the widths.
        '''
        return self.grasp_group_array[:,1]

    @widths.setter
    def widths(self, widths):
        '''
        **Input:**

        - widths: numpy array of shape (-1, ) of the widths.
        '''
        assert widths.size == len(self)
        self.grasp_group_array[:,1] = copy.deepcopy(widths)

    @property
    def heights(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the heights.
        '''
        return self.grasp_group_array[:,2]

    @heights.setter
    def heights(self, heights):
        '''
        **Input:**

        - heights: numpy array of shape (-1, ) of the heights.
        '''
        assert heights.size == len(self)
        self.grasp_group_array[:,2] = copy.deepcopy(heights)

    @property
    def depths(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the depths.
        '''
        return self.grasp_group_array[:,3]

    @depths.setter
    def depths(self, depths):
        '''
        **Input:**

        - depths: numpy array of shape (-1, ) of the depths.
        '''
        assert depths.size == len(self)
        self.grasp_group_array[:,3] = copy.deepcopy(depths)

    @property
    def rotation_matrices(self):
        '''
        **Output:**

        - np.array of shape (-1, 3, 3) of the rotation matrices.
        '''
        return self.grasp_group_array[:, 4:13].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices):
        '''
        **Input:**

        - rotation_matrices: numpy array of shape (-1, 3, 3) of the rotation_matrices.
        '''
        assert rotation_matrices.shape == (len(self), 3, 3)
        self.grasp_group_array[:,4:13] = copy.deepcopy(rotation_matrices.reshape((-1, 9)))

    @property
    def translations(self):
        '''
        **Output:**

        - np.array of shape (-1, 3) of the translations.
        '''
        return self.grasp_group_array[:, 13:16]

    @translations.setter
    def translations(self, translations):
        '''
        **Input:**

        - translations: numpy array of shape (-1, 3) of the translations.
        '''
        assert translations.shape == (len(self), 3)
        self.grasp_group_array[:,13:16] = copy.deepcopy(translations)

    @property
    def object_ids(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the object ids.
        '''
        return self.grasp_group_array[:,16]

    @object_ids.setter
    def object_ids(self, object_ids):
        '''
        **Input:**

        - object_ids: numpy array of shape (-1, ) of the object_ids.
        '''
        assert object_ids.size == len(self)
        self.grasp_group_array[:,16] = copy.deepcopy(object_ids)

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)

        **Output:**

        - GraspGroup instance after transformation, the original GraspGroup will also be changed.
        '''
        rotation = T[:3,:3]
        translation = T[:3,3]
        self.translations = np.dot(rotation, self.translations.T).T + translation # (-1, 3)
        self.rotation_matrices = np.matmul(rotation, self.rotation_matrices).reshape((-1, 3, 3)) # (-1, 9)
        return self

    def add(self, element):
        '''
        **Input:**

        - element: Grasp instance or GraspGroup instance.
        '''
        if isinstance(element, Grasp):
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_array.reshape((-1, GRASP_ARRAY_LEN))))
        elif isinstance(element, GraspGroup):
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_group_array))
        else:
            raise TypeError('Unknown type:{}'.format(element))
        return self

    def remove(self, index):
        '''
        **Input:**

        - index: list of the index of grasp
        '''
        self.grasp_group_array = np.delete(self.grasp_group_array, index, axis = 0)
        return self

    def from_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        self.grasp_group_array = np.load(npy_file_path)
        return self

    def save_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        np.save(npy_file_path, self.grasp_group_array)

    def to_open3d_geometry_list(self):
        '''
        **Output:**

        - list of open3d.geometry.Geometry of the grippers.
        '''
        geometry = []
        for i in range(len(self.grasp_group_array)):
            g = Grasp(self.grasp_group_array[i])
            geometry.append(g.to_open3d_geometry())
        return geometry

    def sort_by_score(self, reverse = False):
        '''
        **Input:**

        - reverse: bool of order, if False, from high to low, if True, from low to high.

        **Output:**

        - no output but sort the grasp group.
        '''
        score = self.grasp_group_array[:,0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.grasp_group_array = self.grasp_group_array[index]
        return self

    def random_sample(self, numGrasp = 20):
        '''
        **Input:**

        - numGrasp: int of the number of sampled grasps.

        **Output:**

        - GraspGroup instance of sample grasps.
        '''
        if numGrasp > self.__len__():
            raise ValueError('Number of sampled grasp should be no more than the total number of grasps in the group')
        shuffled_grasp_group_array = copy.deepcopy(self.grasp_group_array)
        np.random.shuffle(shuffled_grasp_group_array)
        shuffled_grasp_group = GraspGroup()
        shuffled_grasp_group.grasp_group_array = copy.deepcopy(shuffled_grasp_group_array[:numGrasp])
        return shuffled_grasp_group


    def nms(self, translation_thresh = 0.03, rotation_thresh = 30.0 / 180.0 * np.pi):
        '''
        **Input:**

        - translation_thresh: float of the translation threshold.

        - rotation_thresh: float of the rotation threshold.

        **Output:**

        - GraspGroup instance after nms.
        '''
        return GraspGroup(nms_grasp(self.grasp_group_array, translation_thresh, rotation_thresh))


def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    '''
    Author: chenxi-wang

    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score

    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper


def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                        [width,0,0],
                        [0,0,depth],
                        [width,0,depth],
                        [0,height,0],
                        [width,height,0],
                        [0,height,depth],
                        [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                            [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                            [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box



class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """
    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious

            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask3 = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask5 = (targets[:,:,1] < (widths/2 + self.finger_width))
        mask6 = (targets[:,:,1] > widths/2)
        # bottom mask
        mask7 = ((targets[:,:,0] <= depths - self.finger_length)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:,:,0] <= depths - self.finger_length - self.finger_width)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size**3)).reshape(-1)
        bottom_volume = (heights * (widths+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).reshape(-1)
        shifting_volume = (heights * (widths+2*self.finger_width) * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume+1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1)/inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume+1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume+1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume+1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume+1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value