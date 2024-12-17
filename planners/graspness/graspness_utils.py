import torch
from torch.autograd import Function
import torch.nn as nn
import sys
import numpy as np
from sklearn.neighbors import KDTree

# TODO: move these to config file
GRASP_MAX_WIDTH = 0.1
GRASPNESS_THRESHOLD = 0.1
NUM_VIEW = 300
NUM_ANGLE = 12
NUM_DEPTH = 4
M_POINT = 1024








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
    axis_z = torch.cross(axis_x, axis_y)
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