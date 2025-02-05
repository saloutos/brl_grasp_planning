import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import copy

from .cgn_utils import *


### CLASS TO EVALUATE CGN, PERFORM PRE AND POST PROCESSING OF DATA ###

class ContactGraspNet:
    """
    Class for running inference on Contact-GraspNet (CGN)

    :param cfg: config dict
    """
    def __init__(self, cfg):
        self._contact_grasp_cfg = cfg

        # TODO: move this to where it is actually used?
        self._num_input_points = self._contact_grasp_cfg['DATA']['num_input_points']

        # instantiate model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ContactGraspnetModel(cfg, self.device)
        self.model.to(self.device)
        # load weights
        checkpoint_state_dict = torch.load(self._contact_grasp_cfg['DATA']['checkpoint_path'], weights_only=False)
        model_state_dict = self.model.state_dict()
        for k, v in model_state_dict.items():
            if k in checkpoint_state_dict['model'].keys():
                model_state_dict[k] = checkpoint_state_dict['model'][k]
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        # TODO: not sure what the scalars are used for?
        scalars = {}
        for k, v in checkpoint_state_dict.items():
            if k != 'model':
                scalars[k] = v
        # put loaded weights in model
        self.model.load_state_dict(model_state_dict)

    # run model for entire scene with segmented point clouds
    def predict_scene_grasps(self, pcd_world, cam_extrinsics):
        """
        Predict num_point grasps on a full point cloud or in local box regions around point cloud segments.

        Arguments:
            pcd_cam {o3d.geometry.PointCloud} -- input point cloud in camera frame
            cam_extrinsics {np.ndarray} -- 4x4 camera extrinsics matrix

        Keyword Arguments:
            return_world_frame {bool} -- return grasps in world frame (default: {True})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- pred_grasps_cam, scores, contact_pts, gripper_openings
        """

        # crop PC based on bounding box in world frame
        # TODO: put this box size in config?
        workspace_bb = o3d.geometry.OrientedBoundingBox(np.array([0.0, 0.0, 0.2]), np.eye(3), np.array([0.7, 0.6, 0.41]))
        pcd_world = pcd_world.crop(workspace_bb)
        # get cropped point cloud in camera frame as well
        pcd_cam = copy.deepcopy(pcd_world).transform(np.linalg.inv(cam_extrinsics))
        # TODO: downsample point cloud? put this voxel size in config?
        # print('Cropped point cloud to {} points'.format(len(pcd_cam.points)))
        pcd_cam = pcd_cam.voxel_down_sample(voxel_size=0.004)
        # print('Downsampled point cloud to {} points'.format(len(pcd_cam.points)))

        pc_full = np.asarray(pcd_cam.points)
        pred_grasps_cam, scores, contact_pts, gripper_openings = self.predict_grasps(pc_full)
        print('Generated {} grasps'.format(len(pred_grasps_cam)))
        # TODO: filter grasps to only lie within object segments?

        # convert to world frame, and return grasps in world frame
        # put grasps in world frame
        pred_grasps_world = np.zeros_like(pred_grasps_cam)
        contact_pts_world = np.zeros_like(contact_pts)
        for i,g in enumerate(pred_grasps_cam):
            pred_grasps_world[i,:4,:4] = np.matmul(cam_extrinsics, g)
            contact_pts_world[i] = np.matmul(cam_extrinsics[:3,:3], contact_pts[i]) + cam_extrinsics[:3,3]
        # return grasps, scores, contact points, and grasp widths
        return pred_grasps_world, scores, contact_pts_world, gripper_openings, pcd_world


    # main function for evaluating model
    def predict_grasps(self, pc, convert_cam_coords=True, forward_passes=1):
        """
        Predict raw grasps on point cloud

        :param sess: tf.Session
        :param pc: Nx3 point cloud in camera coordinates
        :param convert_cam_coords: Convert from OpenCV to internal training camera coordinates (x left, y up, z front) and converts grasps back to openCV coordinates
        :param constant_offset: do not predict offset and place gripper at constant `extra_opening` distance from contact point
        :param forward_passes: Number of forward passes to run on each point cloud. default: 1
        :returns: (pred_grasps_cam, pred_scores, pred_points, gripper_openings) Predicted grasps/scores/contact-points/gripper-openings
        """

        # regularize number of points in point cloud
        # Convert point cloud coordinates from OpenCV to internal coordinates (x left, y up, z front)
        # print('Total points:', len(pc))
        # pc, pc_mean = preprocess_pc_for_inference(pc.squeeze(),
        #                                         self._contact_grasp_cfg['DATA']['num_input_points'],
        #                                         return_mean=True,
        #                                         use_farthest_point=False, # TODO:  farthest point sampling shouldn't take this long!! if we use tensors, can copy function here
        #                                         convert_to_internal_coords=convert_cam_coords)

        # PULLING OUT PREPROCESS FUNCTION HERE
        # regularize number of points in point cloud
        use_farthest_point = False
        if pc.shape[0] > self._contact_grasp_cfg['DATA']['num_input_points']:
            if use_farthest_point:
                # TODO: this is still quite slow!
                # create tensor from pc
                pc_tensor = torch.from_numpy(pc).type(torch.float32).to(self.device)
                pc_tensor = pc_tensor.unsqueeze(0)
                # get farthest points
                center_indexes = farthest_point_sample(pc_tensor, self._contact_grasp_cfg['DATA']['num_input_points']).squeeze().to('cpu').numpy()
                # _, center_indexes = farthest_points(pc, self._contact_grasp_cfg['DATA']['num_input_points'], distance_by_translation_point, return_center_indexes=True)
            else:
                center_indexes = np.random.choice(range(pc.shape[0]), size=self._contact_grasp_cfg['DATA']['num_input_points'], replace=False)
            pc = pc[center_indexes, :]
        else:
            required = self._contact_grasp_cfg['DATA']['num_input_points'] - pc.shape[0]
            if required > 0:
                index = np.random.choice(range(pc.shape[0]), size=required)
                pc = np.concatenate((pc, pc[index, :]), axis=0)

        # Convert point cloud coordinates from OpenCV to internal coordinates (x left, y up, z front)
        pc[:,:2] *= -1
        # calculate mean of point cloud
        pc_mean = np.mean(pc, 0)
        # subtract mean
        pc -= np.expand_dims(pc_mean, 0)



        if len(pc.shape) == 2:
            pc_batch = pc[np.newaxis,:,:]
        if forward_passes > 1:
            pc_batch = np.tile(pc_batch, (forward_passes,1,1))
        pc_batch = torch.from_numpy(pc_batch).type(torch.float32).to(self.device)
        # Run model inference
        pred = self.model(pc_batch)
        pred_grasps_cam = pred['pred_grasps_cam']
        pred_scores = pred['pred_scores']
        pred_points = pred['pred_points']
        offset_pred = pred['offset_pred']

        pred_grasps_cam = pred_grasps_cam.detach().cpu().numpy()
        pred_scores = pred_scores.detach().cpu().numpy()
        pred_points = pred_points.detach().cpu().numpy()
        offset_pred = offset_pred.detach().cpu().numpy()

        pred_grasps_cam = pred_grasps_cam.reshape(-1, *pred_grasps_cam.shape[-2:])
        pred_points = pred_points.reshape(-1, pred_points.shape[-1])
        pred_scores = pred_scores.reshape(-1)
        offset_pred = offset_pred.reshape(-1)

        # uncenter grasps
        pred_grasps_cam[:,:3, 3] += pc_mean.reshape(-1,3)
        pred_points[:,:3] += pc_mean.reshape(-1,3)
        # limit gripper openings
        gripper_openings = np.minimum(offset_pred + self._contact_grasp_cfg['TEST']['extra_opening'], self._contact_grasp_cfg['DATA']['gripper_width'])

        selection_idcs = self.select_grasps(pred_points[:,:3], pred_scores,
                                            self._contact_grasp_cfg['TEST']['max_farthest_points'],
                                            self._contact_grasp_cfg['TEST']['num_samples'],
                                            self._contact_grasp_cfg['TEST']['first_thres'],
                                            self._contact_grasp_cfg['TEST']['second_thres'],
                                            with_replacement=self._contact_grasp_cfg['TEST']['with_replacement'])

        # if no grasps are selected, just select the one with the highest score
        if not np.any(selection_idcs):
            # selection_idcs=np.array([], dtype=np.int32)
            selection_idcs = np.array([np.argmax(pred_scores)])

        # convert back to opencv coordinates
        if convert_cam_coords:
            pred_grasps_cam[:,:2, :] *= -1
            pred_points[:,:2] *= -1

        # return grasps, scores, points, and widths, based on grasp selection
        return pred_grasps_cam[selection_idcs], pred_scores[selection_idcs], pred_points[selection_idcs], gripper_openings[selection_idcs].squeeze()

    # helper functions
    def select_grasps(self, contact_pts, contact_conf, max_farthest_points=150, num_grasps=200, first_thres=0.25, second_thres=0.2, with_replacement=False):
        """
        Select subset of num_grasps by contact confidence thresholds and farthest contact point sampling. 

        1.) Samples max_farthest_points among grasp contacts with conf > first_thres
        2.) Fills up remaining grasp contacts to a maximum of num_grasps with highest confidence contacts with conf > second_thres

        Arguments:
            contact_pts {np.ndarray} -- num_point x 3 subset of input point cloud for which we have predictions 
            contact_conf {[type]} -- num_point x 1 confidence of the points being a stable grasp contact

        Keyword Arguments:
            max_farthest_points {int} -- Maximum amount from num_grasps sampled with farthest point sampling (default: {150})
            num_grasps {int} -- Maximum number of grasp proposals to select (default: {200})
            first_thres {float} -- first confidence threshold for farthest point sampling (default: {0.6})
            second_thres {float} -- second confidence threshold for filling up grasp proposals (default: {0.6})
            with_replacement {bool} -- Return fixed number of num_grasps with conf > first_thres and repeat if there are not enough (default: {False})

        Returns:
            [np.ndarray] -- Indices of selected contact_pts
        """

        grasp_conf = contact_conf.squeeze()
        contact_pts = contact_pts.squeeze()

        conf_idcs_greater_than = np.nonzero(grasp_conf > first_thres)[0]
        print("Number of grasps with confidence > first_thres: ", len(conf_idcs_greater_than))
        _, center_indexes = farthest_points(contact_pts[conf_idcs_greater_than,:3],
                                            np.minimum(max_farthest_points, len(conf_idcs_greater_than)),
                                            distance_by_translation_point,
                                            return_center_indexes=True)

        remaining_confidences = np.setdiff1d(np.arange(len(grasp_conf)), conf_idcs_greater_than[center_indexes])
        sorted_confidences = np.argsort(grasp_conf)[::-1]
        mask = np.in1d(sorted_confidences, remaining_confidences)
        sorted_remaining_confidence_idcs = sorted_confidences[mask]

        if with_replacement:
            selection_idcs = list(conf_idcs_greater_than[center_indexes])
            j=len(selection_idcs)
            while j < num_grasps and conf_idcs_greater_than.shape[0] > 0:
                selection_idcs.append(conf_idcs_greater_than[j%len(conf_idcs_greater_than)])
                j+=1
            selection_idcs = np.array(selection_idcs)
        else:
            remaining_idcs = sorted_remaining_confidence_idcs[:num_grasps-len(conf_idcs_greater_than[center_indexes])]
            remaining_conf_idcs_greater_than = np.nonzero(grasp_conf[remaining_idcs] > second_thres)[0]
            selection_idcs = np.union1d(conf_idcs_greater_than[center_indexes], remaining_idcs[remaining_conf_idcs_greater_than])

        return selection_idcs



### CONTACT GRASPNET MODEL ITSELF ###

class ContactGraspnetModel(nn.Module):
    def __init__(self, global_config, device):
        super(ContactGraspnetModel, self).__init__()

        self.device = device

        # -- Extract config -- #
        self.global_config = global_config
        self.model_config = global_config['MODEL']
        self.data_config = global_config['DATA']

        radius_list_0 = self.model_config['pointnet_sa_modules_msg'][0]['radius_list']
        radius_list_1 = self.model_config['pointnet_sa_modules_msg'][1]['radius_list']
        radius_list_2 = self.model_config['pointnet_sa_modules_msg'][2]['radius_list']

        nsample_list_0 = self.model_config['pointnet_sa_modules_msg'][0]['nsample_list']
        nsample_list_1 = self.model_config['pointnet_sa_modules_msg'][1]['nsample_list']
        nsample_list_2 = self.model_config['pointnet_sa_modules_msg'][2]['nsample_list']

        mlp_list_0 = self.model_config['pointnet_sa_modules_msg'][0]['mlp_list'] # list of lists
        mlp_list_1 = self.model_config['pointnet_sa_modules_msg'][1]['mlp_list']
        mlp_list_2 = self.model_config['pointnet_sa_modules_msg'][2]['mlp_list']

        npoint_0 = self.model_config['pointnet_sa_modules_msg'][0]['npoint']
        npoint_1 = self.model_config['pointnet_sa_modules_msg'][1]['npoint']
        npoint_2 = self.model_config['pointnet_sa_modules_msg'][2]['npoint']

        fp_mlp_0 = self.model_config['pointnet_fp_modules'][0]['mlp']
        fp_mlp_1 = self.model_config['pointnet_fp_modules'][1]['mlp']
        fp_mlp_2 = self.model_config['pointnet_fp_modules'][2]['mlp']

        sa_mlp = self.model_config['pointnet_sa_module']['mlp']
        sa_group_all = self.model_config['pointnet_sa_module']['group_all']

        self.input_normals = self.data_config['input_normals']
        self.offset_bins = self.data_config['labels']['offset_bins']
        self.joint_heads = self.model_config['joint_heads']

        # For adding additional features
        additional_channel = 3 if self.input_normals else 0

        if 'asymmetric_model' in self.model_config and self.model_config['asymmetric_model']:
            # It looks like there are xyz_points and "points".  The points are normals I think?
            self.sa1 = PointNetSetAbstractionMsg(npoint=npoint_0,
                                                radius_list=radius_list_0,
                                                nsample_list=nsample_list_0,
                                                in_channel = 3 + additional_channel,
                                                mlp_list=mlp_list_0)

            # Sum the size of last layer of each mlp in mlp_list
            sa1_out_channels = sum([mlp_list_0[i][-1] for i in range(len(mlp_list_0))])
            self.sa2 = PointNetSetAbstractionMsg(npoint=npoint_1,
                                                radius_list=radius_list_1,
                                                nsample_list=nsample_list_1,
                                                in_channel=sa1_out_channels,
                                                mlp_list=mlp_list_1)

            sa2_out_channels = sum([mlp_list_1[i][-1] for i in range(len(mlp_list_1))])
            self.sa3 = PointNetSetAbstractionMsg(npoint=npoint_2,
                                                radius_list=radius_list_2,
                                                nsample_list=nsample_list_2,
                                                in_channel=sa2_out_channels,
                                                mlp_list=mlp_list_2)

            sa3_out_channels = sum([mlp_list_2[i][-1] for i in range(len(mlp_list_2))])
            self.sa4 = PointNetSetAbstraction(npoint=None,
                                                radius=None,
                                                nsample=None,
                                                in_channel=3 + sa3_out_channels,
                                                mlp=sa_mlp,
                                                group_all=sa_group_all)

            self.fp3 = PointNetFeaturePropagation(in_channel=sa_mlp[-1] + sa3_out_channels,
                                                mlp=fp_mlp_2)

            self.fp2 = PointNetFeaturePropagation(in_channel=fp_mlp_2[-1] + sa2_out_channels,
                                                mlp=fp_mlp_1)

            self.fp1 = PointNetFeaturePropagation(in_channel=fp_mlp_1[-1] + sa1_out_channels,
                                                mlp=fp_mlp_0)
        else:
            raise NotImplementedError

        if self.joint_heads:
            raise NotImplementedError
        else:
            # Theres prob some bugs here TODO?

            # Head for grasp direction
            self.grasp_dir_head = nn.Sequential(
                nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3), # p = 1 - keep_prob  (tf is inverse of torch)
                nn.Conv1d(128, 3, 1, padding=0)
            )

            # Remember to normalize the output of this head

            # Head for grasp approach
            self.grasp_approach_head = nn.Sequential(
                nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(128, 3, 1, padding=0)
            )

            # Head for grasp width
            if self.model_config['dir_vec_length_offset']:
                raise NotImplementedError
            elif self.model_config['bin_offsets']:
                self.grasp_offset_head = nn.Sequential(
                    nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, len(self.offset_bins) - 1, 1, padding=0)
                )
            else:
                raise NotImplementedError

            # Head for contact points
            self.binary_seg_head = nn.Sequential(
                nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),  # 0.5 in original code
                nn.Conv1d(128, 1, 1, padding=0)
            )

    def forward(self, point_cloud):

        # Convert from tf to torch ordering
        point_cloud = torch.transpose(point_cloud, 1, 2) # Now we have batch x channels (3 or 6) x num_points

        l0_xyz = point_cloud[:, :3, :]
        l0_points = point_cloud[:, 3:6, :] if self.input_normals else l0_xyz.clone()

        # -- PointNet Backbone -- #
        # Set Abstraction Layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation Layers
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = l1_points
        pred_points = l1_xyz

        # -- Heads -- #
        # Grasp Direction Head
        grasp_dir_head = self.grasp_dir_head(l0_points)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)  # normalize along channels

        # Grasp Approach Head
        approach_dir_head = self.grasp_approach_head(l0_points)

        # compute the gram schmidt orthonormalization
        dot_product = torch.sum(grasp_dir_head_normed * approach_dir_head, dim=1, keepdim=True)
        projection = dot_product * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_head - projection, p=2, dim=1)

        # Grasp Width Head
        # TODO: remove this config option?
        if self.model_config['dir_vec_length_offset']:
            raise NotImplementedError
            grasp_offset_head = F.normalize(grasp_dir_head, p=2, dim=1)
        elif self.model_config['bin_offsets']:
            grasp_offset_head = self.grasp_offset_head(l0_points)

        # Binary Segmentation Head
        binary_seg_head = self.binary_seg_head(l0_points)

        # -- Construct Output -- #
        # Get 6 DoF grasp pose
        torch_bin_vals = self.get_bin_vals()

        # PyTorch equivalent of tf.gather_nd with conditional, the output should be B x N
        # TODO: check this?
        if self.model_config['bin_offsets']:
            argmax_indices = torch.argmax(grasp_offset_head, dim=1, keepdim=True)
            offset_bin_pred_vals = torch_bin_vals[argmax_indices]  # kinda sketch but works?
            # expand_dims_indices = argmax_indices.unsqueeze(1)
            # offset_bin_pred_vals = torch.gather(torch_bin_vals, 1, argmax_indices)
        else:
            offset_bin_pred_vals = grasp_offset_head[:, 0, :]

        # get predicted grasp poses
        pred_grasps_cam = self.build_grasp_poses(approach_dir_head_orthog.permute(0, 2, 1),
                                                grasp_dir_head_normed.permute(0, 2, 1),
                                                pred_points.permute(0, 2, 1),
                                                offset_bin_pred_vals.permute(0, 2, 1),
                                                use_torch=True)  # B x N x 4 x 4

        # Get pred scores
        pred_scores = torch.sigmoid(binary_seg_head).permute(0, 2, 1)

        # Get pred points
        pred_points = pred_points.permute(0, 2, 1)

        # Get pred offsets
        # TODO: are these offsets the gripper widths?
        offset_pred = offset_bin_pred_vals

        # return pred_grasps_cam, pred_scores, pred_points, offset_pred, intermediates
        pred = dict(
            pred_grasps_cam=pred_grasps_cam,
            pred_scores=pred_scores,
            pred_points=pred_points,
            offset_pred=offset_pred,
            grasp_offset_head=grasp_offset_head)
        return pred


    # TODO: should this be evaluated once on init?
    def get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            torch.tensor -- bin value tensor
        """
        bins_bounds = np.array(self.data_config['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width'] \
                                    -self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32).to(self.device)
        return bin_vals


    def build_grasp_poses(self, approach_dirs, base_dirs, contact_pts, thickness, use_torch=False, gripper_depth = 0.1034):
        """
        Build 6-DoF grasps + width from point-wise network predictions

        Arguments:
            approach_dirs {np.ndarray/torch.tensor} -- Nx3 approach direction vectors
            base_dirs {np.ndarray/torch.tensor} -- Nx3 base direction vectors
            contact_pts {np.ndarray/torch.tensor} -- Nx3 contact points
            thickness {np.ndarray/torch.tensor} -- Nx1 grasp width

        Keyword Arguments:
            use_torch {bool} -- whether inputs and outputs are torch tensors (default: {False})
            gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

        Returns:
            np.ndarray -- Nx4x4 grasp poses in camera coordinates
        """
        # We are trying to build a stack of 4x4 homogeneous transform matricies of size B x N x 4 x 4.
        # To do so, we calculate the rotation and translation portions according to the paper.
        # This gives us positions as shown:
        # [ R R R T ]
        # [ R R R T ]
        # [ R R R T ]
        # [ 0 0 0 1 ]                    Note that the ^ dim is 2 and the --> dim is 3
        # We need to pad with zeros and ones to get the final shape so we generate
        # ones and zeros and stack them.
        if use_torch:
            grasp_R = torch.stack([base_dirs, torch.linalg.cross(approach_dirs,base_dirs),approach_dirs], dim=3)  # B x N x 3 x 3
            grasp_t = contact_pts + (thickness / 2) * base_dirs - gripper_depth * approach_dirs  # B x N x 3
            grasp_t = grasp_t.unsqueeze(3)  # B x N x 3 x 1
            ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32).to(self.device)  # B x N x 1 x 1
            zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32).to(self.device)  # B x N x 1 x 3
            homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
            grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4

        else:
            grasps = []
            for i in range(len(contact_pts)):
                grasp = np.eye(4)

                grasp[:3,0] = base_dirs[i] / np.linalg.norm(base_dirs[i])
                grasp[:3,2] = approach_dirs[i] / np.linalg.norm(approach_dirs[i])
                grasp_y = np.cross( grasp[:3,2],grasp[:3,0])
                grasp[:3,1] = grasp_y / np.linalg.norm(grasp_y)
                # base_gripper xyz = contact + thickness / 2 * baseline_dir - gripper_d * approach_dir
                grasp[:3,3] = contact_pts[i] + thickness[i] / 2 * grasp[:3,0] - gripper_depth * grasp[:3,2]
                # grasp[0,3] = finger_width
                grasps.append(grasp)
            grasps = np.array(grasps)

        return grasps


### POINTNET 2 LAYERS AND HELPERS ###

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
        = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
