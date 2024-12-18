import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import open3d as o3d

# TODO: remove this as a dependency???
import MinkowskiEngine as ME
from .minkowski_resnets import *
# from .resnets import *


from .graspness_utils import *


### CLASS TO EVALUATE EDGE GRASP, PERFORM PRE AND POST PROCESSING OF DATA ###

class GraspnessNet:
    def __init__(self, cfg):
        self._graspnet_cfg = cfg

        # instantiate model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = GraspNetModel(seed_feat_dim=512, is_training=False)
        self.model.to(self.device)

        # load weights
        checkpoint_path = 'planners/graspness/checkpoints/minkuresunet_realsense.tar' # TODO: put this in config yaml
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Loaded checkpoint %s (epoch: %d)." % (checkpoint_path, start_epoch))

        # ready to evaluate
        self.model.eval()

    def predict_scene_grasps(self, pcd_cam):

        # some pre-processing
        pc = np.asarray(pcd_cam.points)
        data_dict = self.sample_points(pc)

        # batch and process the data
        batch_data = minkowski_collate_fn([data_dict])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        # Forward pass
        with torch.no_grad():
            end_points = self.model(batch_data)
            grasp_preds = pred_decode(end_points)
        # return the predictions
        preds = grasp_preds[0].detach().cpu().numpy()

        # Filtering grasp poses for real-world execution.
        # The first mask preserves the grasp poses that are within a 30-degree angle with the vertical pose and have a width of less than 9cm.
        # mask = (preds[:,10] > 0.9) & (preds[:,1] < 0.09)
        # The second mask preserves the grasp poses within the workspace of the robot.
        # workspace_mask = (preds[:,13] > -0.20) & (preds[:,13] < 0.21) & (preds[:,14] > -0.06) & (preds[:,14] < 0.18) & (preds[:,15] > 0.63) 
        # preds = preds[mask & workspace_mask]

        # if len(preds) == 0:
        #         print('No grasp detected after masking')
        #         return

        gg = GraspGroup(preds)
        # collision detection
        collision_thresh = 0 # TODO: put this in config yaml
        voxel_size_cd = 0.01 # TODO: config yaml
        # TODO: will need to do this in world frame?
        if collision_thresh > 0:
            cloud = data_dict['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        # TODO: what is this?
        gg = gg.nms()
        gg = gg.sort_by_score()
        # trim list?
        # TODO: put this limit in config file
        num_grasp_limit = 100
        if gg.__len__() > num_grasp_limit:
            print('Trimming final grasp list to %d.' % num_grasp_limit)
            gg = gg[:num_grasp_limit]

        # from grippers, get grasp poses and gripper widths
        # TODO: need to swap some axes here, since grasp frame is different than our baseline CGN parameterization
        pred_grasp_array = np.zeros((len(gg),4,4))
        for i in range(len(gg)):
            g = gg[i]
            new_pose = np.eye(4)
            # new_pose[:3,:3] = g.rotation_matrix
            new_pose[:3,0] = g.rotation_matrix[:3,1]
            new_pose[:3,1] = g.rotation_matrix[:3,2]
            new_pose[:3,2] = g.rotation_matrix[:3,0]
            offset_dist = g.depth + 0.05
            new_pose[:3,3] = g.translation - offset_dist*g.rotation_matrix[:3,0]
            pred_grasp_array[i,:4,:4] = new_pose

        pred_grasps = {-1: pred_grasp_array}
        # also return scores
        grasp_scores = {-1 : gg.scores}
        gripper_widths = {-1:  gg.widths}

        return pred_grasps, grasp_scores, gripper_widths

    def sample_points(self, masked_pc):
        # masked pc is numpy array
        # sample masked_pc random

        num_points = 15000 # TODO: put this in config yaml file
        voxel_size = 0.005 # TODO: this should also be in config yaml file

        if len(masked_pc) >= num_points:
            idxs = np.random.choice(len(masked_pc), num_points, replace=False)
        else:
            idxs1 = np.arange(len(masked_pc))
            idxs2 = np.random.choice(len(masked_pc), num_points - len(masked_pc), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = masked_pc[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict



### GRASPNESS MODEL ITSELF ###

class GraspNetModel(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=False):
        super().__init__()
        self.is_training = is_training

        # TODO: put a lot of this in config file!
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW

        # TODO: replace this
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # self.backbone = ResUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)

        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size

        # TODO: what are the shapes of these?
        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']

        # TODO: replace this
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        # res_input = []
        # seed_features = self.backbone(res_input).F

        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.swad(group_features, end_points)

        return end_points


### INDIVIDUAL MODULES

class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample, use_xyz=True, normalize_xyz=True)
        self.mlps = SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot, seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points


### POINTNET2 AND PYTORCH UTILS STUFF

# TODO: check these implementations (how will I debug?)
def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # NOTE: taken from pointnet2 utils in Ethan's CGN repo
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def gather_operation(features, idx):
    """

    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor

    idx : torch.Tensor
        (B, npoint) tensor of the features to gather

    Returns
    -------
    torch.Tensor
        (B, C, npoint) tensor
    """
    # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
    device = features.device
    B, C, N = features.shape
    all_idx = idx.unsqueeze(1).repeat(1, C, 1).long()
    return features.gather(2, all_idx)


def grouping_operation(features, idx):
    """
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1).long()
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


def cylinder_query(radius, hmin, hmax, nsample, xyz, new_xyz, rot):
    """

    Parameters
    ----------
    radius : float
        radius of the cylinders
    hmin, hmax : float
        endpoints of cylinder height in x-rotation axis
    nsample : int
        maximum number of features in the cylinders
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the cylinder query
    rot: torch.Tensor
        (B, npoint, 9) flatten rotation matrices from
                        cylinder frame to world frame

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    # type: (Any, float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    device = xyz.device
    B, N, _ = xyz.shape
    _, npoint, _ = new_xyz.shape
    r2 = radius**2

    # calculate distances from cylinder centers, in cylinder frames
    dists = new_xyz[:,:,None,:] - xyz[:,None,:,:] # B, npoint, N, 3
    dists = dists.unsqueeze(4) # B, npoint, N, 3, 1
    rots = rot.unsqueeze(2).view(B, npoint, 1, 3, 3) # B, npoint, 1, 3, 3
    rot_dists = torch.matmul(rots, dists).squeeze(4) # B, npoint, N, 3
    # filters
    radius_mask = ( (torch.square(rot_dists[:,:,:,1])+torch.square(rot_dists[:,:,:,2])) < r2 ) # B, npoint, N
    height_mask = torch.logical_and((rot_dists[:,:,:,0]<hmax), (rot_dists[:,:,:,0]>hmin) )# B, npoint, N
    # get valid points/indices
    all_valid_pts = torch.logical_and(radius_mask, height_mask) # B, npoint, N
    all_valid_idxs = torch.nonzero(all_valid_pts, as_tuple=False).int() # num_valid, 3
    all_valid_sample_point_idxs = all_valid_idxs[:,1]
    # just take first nsample points from all valid pts
    unique_values, inverse_indices, counts = torch.unique_consecutive(all_valid_sample_point_idxs, return_inverse=True, return_counts=True)
    query_features = torch.zeros((B, npoint, nsample), dtype=torch.int).to(device)

    for i, count in enumerate(counts):
        start_idx = (inverse_indices == i).nonzero()[0]
        cnt = min(count, nsample)
        end_idx = start_idx + cnt
        query_features[all_valid_idxs[start_idx:end_idx,0],
                        all_valid_idxs[start_idx:end_idx,1],
                        torch.arange(cnt)] = all_valid_idxs[start_idx:end_idx,2]

    return query_features


class CylinderQueryAndGroup(nn.Module):
    """
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, rot, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


# TODO: not sure why these are necessary?
class SharedMLP(nn.Sequential):
    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )


class _ConvBase(nn.Sequential):
    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )