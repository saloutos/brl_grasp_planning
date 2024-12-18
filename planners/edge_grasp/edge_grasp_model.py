import numpy as np
import os.path
import os
import sys
import torch.nn.functional as F
from torch_geometric.nn import  PPFConv,knn_graph,global_max_pool,radius
from torch_geometric.nn import PointNetConv as PointNetConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose
from torch.backends import cudnn
from torch.nn import Sequential, Linear, ReLU
import torch
import open3d as o3d

from .edge_grasp_utils import *

# TODO: DUPLICATE FOR VN EDGE GRASP?

### CLASS TO EVALUATE EDGE GRASP, PERFORM PRE AND POST PROCESSING OF DATA ###

# NOTE: structure of original edge grasper seemed really nice for training, testing, saving, loading, etc!
class EdgeGraspNet:
    def __init__(self, cfg) :
        self._edge_grasp_cfg = cfg

        # instantiate model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: these things should be in config file
        sample_num = 32
        position_emd = True
        lr = 1e-5

        # instantiate model
        self.position_emd = position_emd
        self.sample_number = sample_num
        self.model = EdgeGraspModel(device=self.device, sample_num=sample_num, lr=lr)

        # load weights
        # TODO: these filenames can be in config file
        n_iter = 180
        fname1 = 'planners/edge_grasp/checkpoints/local_emd_model-ckpt-%d.pt' % n_iter
        fname2 = 'planners/edge_grasp/checkpoints/global_emd_model-ckpt-%d.pt' % n_iter
        fname3 = 'planners/edge_grasp/checkpoints/classifier_model-ckpt-%d.pt' % n_iter
        self.model.load(fname1,fname2,fname3,)
        print('Loaded params.')

    def predict_scene_grasps(self, pc_full_world):

        # pre-process point clouds
        vertices = np.asarray(pc_full_world.points)
        if len(vertices) < 100:
            print("1. point cloud < 100, should skip scene.")
        # TODO: put filtering parameters into config file?
        pc_full_world, ind = pc_full_world.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        pc_full_world, ind = pc_full_world.remove_radius_outlier(nb_points=30, radius=0.03)
        pc_full_world.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pc_full_world.orient_normals_consistent_tangent_plane(30)
        vertices = np.asarray(pc_full_world.points)
        if len(vertices) < 100:
            print("2. point cloud < 100, should skip scene.")
        # TODO: put voxel size into config file?
        pc_full_world = pc_full_world.voxel_down_sample(voxel_size=0.0045)

        # TODO: is this actually sampling the edge?
        # sample edges
        pos = np.asarray(pc_full_world.points)
        normals = np.asarray(pc_full_world.normals)
        pos = torch.from_numpy(pos).to(torch.float32).to(self.device)
        normals = torch.from_numpy(normals).to(torch.float32).to(self.device)

        fps_sample = FarthestSamplerTorch()
        _, sample = fps_sample(pos,self.sample_number)
        sample = torch.as_tensor(sample).to(torch.long).reshape(-1).to(self.device)
        sample = torch.unique(sample,sorted=True)

        sample_pos = pos[sample, :]
        radius_p_batch_index = radius(pos, sample_pos, r=0.05, max_num_neighbors=1024)
        radius_p_index = radius_p_batch_index[1, :]
        radius_p_batch = radius_p_batch_index[0, :]
        sample_pos = torch.cat(
            [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
            dim=0)
        sample_copy = sample.clone().unsqueeze(dim=-1)
        sample_index = torch.cat(
            [sample_copy[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
        edges = torch.cat((sample_index, radius_p_index.unsqueeze(dim=-1)), dim=1)
        all_edge_index = torch.arange(0,len(edges)).to(self.device)
        des_pos = pos[radius_p_index, :]
        des_normals = normals[radius_p_index, :]
        relative_pos = des_pos - sample_pos
        relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)

        # filter edges
        # only record approach vectors with a angle mask
        x_axis = torch.cross(des_normals, relative_pos_normalized)
        x_axis = F.normalize(x_axis, p=2, dim=1)
        valid_edge_approach = torch.cross(x_axis, des_normals)
        valid_edge_approach = F.normalize(valid_edge_approach, p=2, dim=1)
        valid_edge_approach = -valid_edge_approach
        up_dot_mask = torch.einsum('ik,k->i', valid_edge_approach, torch.tensor([0., 0., 1.]).to(self.device))
        relative_norm = torch.linalg.norm(relative_pos, dim=-1)

        # TODO: put all these filtering parameters into the config file
        depth_proj = -torch.sum(relative_pos * valid_edge_approach, dim=-1)
        geometry_mask = torch.logical_and(up_dot_mask > -0.1, relative_norm > 0.003)
        geometry_mask = torch.logical_and(relative_norm<0.038,geometry_mask)
        depth_proj_mask = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)
        geometry_mask = torch.logical_and(geometry_mask, depth_proj_mask)
        if torch.sum(geometry_mask)<10:
            print('Less than 10 valid edges, should skip scene.')

        # build grasp poses from candidate edges, check table collisions
        pose_candidates = orthogonal_grasps(geometry_mask, depth_proj, valid_edge_approach, des_normals,
                                            sample_pos)
        table_grasp_mask = get_gripper_points_mask(pose_candidates, z_threshold=0.054) # TODO: put table-height into the config file
        geometry_mask[geometry_mask == True] = table_grasp_mask
        edge_sample_index = all_edge_index[geometry_mask]
        print('Number of candidates after checking table collisions: ', len(edge_sample_index))

        # TODO: put the maximum number of sample points in the config file
        MAX_NUM_EDGES = 200

        pred_grasps, grasp_scores, gripper_widths = {}, {}, {}

        # actually evaluate model for these edges
        if len(edge_sample_index) > 0:
            if len(edge_sample_index) > MAX_NUM_EDGES:
                edge_sample_index = edge_sample_index[torch.randperm(len(edge_sample_index))[:MAX_NUM_EDGES]]
            edge_sample_index, _ = torch.sort(edge_sample_index)
            data = Data(pos=pos, normals=normals, sample=sample, radius_p_index=radius_p_index,
                        ball_batch=radius_p_batch,
                        ball_edges=edges, approaches=valid_edge_approach[edge_sample_index, :],
                        reindexes=edge_sample_index,
                        relative_pos=relative_pos[edge_sample_index, :],
                        depth_proj=depth_proj[edge_sample_index])
            data = data.to(self.device)
            score, depth_projection, approaches, sample_pos, des_normals = self.model.act(data)

            # select grasps based on threshold
            # TODO: put this threshold in config file?
            SELECTION_THRESHOLD = 0.0
            all_scores = F.sigmoid(score)
            grasp_mask = (all_scores > SELECTION_THRESHOLD)

            masked_poses = orthogonal_grasps(grasp_mask, depth_projection, approaches, des_normals, sample_pos)
            masked_scores = all_scores[grasp_mask]

            # calculate grasp widths
            widths = torch.abs(torch.sum(data.relative_pos * des_normals, dim=-1)) + 0.016
            widths = widths[grasp_mask].clip(max=0.04)
            widths = (widths * 2)

            # move to CPU
            # NOTE: dict key -1 is to match CGN implementation, in case we feed in segmented point clouds later
            pred_grasps[-1] = masked_poses.detach().cpu().numpy()
            grasp_scores[-1] = masked_scores.detach().cpu().numpy()
            gripper_widths[-1] = widths.detach().cpu().numpy()

            # TODO: why was original code sampling grasp point here?

            # # pull out parameters of best grasp
            # k_score, max_index = torch.topk(score, k=1)
            # selected_edge = edges[edge_sample_index[max_index],:]
            # max_score = score[max_index]
            # max_score = F.sigmoid(max_score).cpu().numpy()
            # print('Best grasp score: ', max_score)
            # if max_score.any() < 0.85:
            #     print('No high score, should skip scene.')
            # grasp_mask = torch.ones(len(depth_projection)) > 2.
            # grasp_mask[max_index] = True
            # trans_matrix = orthogonal_grasps(grasp_mask.to(des_normals.device), depth_projection, approaches,
            #                                 des_normals, sample_pos)
            # trans_matrix = trans_matrix.cpu().numpy()

        else:
            print('No candidates without collisions.')

        return pred_grasps, grasp_scores, gripper_widths
























### EDGE GRASP MODEL ITSELF ###

class EdgeGraspModel:
    def __init__(self, device, sample_num=32, lr=1e-4):
        self.device = device
        self.sample_num = sample_num
        self.local_emd_model = PointNetSimple(out_channels=(32, 64, 128), train_with_norm=True).to(device)
        self.global_emd_model = GlobalEmdModel(input_c=32+64+128, inter_c=(256,512,512),output_c=1024).to(device)
        self.classifier_fail = Classifier(in_channels=1162, hidden_channels=(512, 256, 128)).to(device)
        self.parameter = list(self.local_emd_model.parameters()) + list(self.global_emd_model.parameters()) + list(self.classifier_fail.parameters())
        self.classifier_para = list(self.global_emd_model.parameters()) + list(self.classifier_fail.parameters())
        self.optim = torch.optim.Adam([{'params': self.local_emd_model.parameters(), 'lr': lr}, {'params': self.classifier_para}, ], lr=lr, weight_decay=1e-8)
        print('edge_grasper ball: ', sum(p.numel() for p in self.parameter if p.requires_grad))

    def forward(self, batch, train=True,):
        # Todo get the local emd for every point in the batch
        # balls setup
        ball_batch = batch.ball_batch
        ball_edges = batch.ball_edges
        reindexes = batch.reindexes
        balls = batch.pos[ball_edges[:, 1], :] - batch.pos[ball_edges[:, 0], :]
        ball_normals = batch.normals[ball_edges[:, 1], :]
        sample = batch.sample

        if train:
            self.local_emd_model.train()
            f1, f2, features = self.local_emd_model(pos=balls, normal=ball_normals, batch=ball_batch)
        else:
            self.local_emd_model.eval()
            with torch.no_grad():
                f1, f2, features = self.local_emd_model(pos=balls, normal=ball_normals, batch=ball_batch)

        approaches = batch.approaches
        depth_proj = batch.depth_proj

        des_emd = torch.cat((f1,f2,features),dim=1)
        if train:
            self.global_emd_model.train()
            global_emd = self.global_emd_model(des_emd,ball_batch)
        else:
            self.global_emd_model.eval()
            with torch.no_grad():
                global_emd = self.global_emd_model(des_emd,ball_batch)

        valid_batch = ball_batch[reindexes]
        global_emd_valid = torch.cat([global_emd[i, :].repeat((valid_batch == i).sum(), 1) for i in range(len(sample))],dim=0)
        des_cat = torch.cat((balls[reindexes,:], ball_normals[reindexes,:], features[reindexes,:]), dim=-1)
        edge_attributes = torch.cat((depth_proj.unsqueeze(dim=-1),approaches),dim=-1)
        cat_all_orth_mask = torch.cat((des_cat, global_emd_valid, edge_attributes), dim=-1)

        if train:
            self.classifier_fail.train()
            scores_succ = self.classifier_fail(cat_all_orth_mask)
        else:
            self.classifier_fail.eval()
            with torch.no_grad():
                scores_succ = self.classifier_fail(cat_all_orth_mask)
        return scores_succ, depth_proj

    def act(self, batch, train=False):
        scores_succ, depth_proj = self.forward(batch,train=train)
        approaches = batch.approaches
        sample_pos = batch.pos[batch.ball_edges[:, 0][batch.reindexes], :]
        des_normals = batch.normals[batch.ball_edges[:, 1][batch.reindexes], :]
        scores_succ = scores_succ.squeeze(dim=-1)

        return scores_succ, depth_proj, \
                approaches, sample_pos, \
                des_normals

    def load(self, path1, path2, path3):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier_fail.eval()

        self.local_emd_model.load_state_dict(torch.load(path1, self.device, weights_only=False))
        self.global_emd_model.load_state_dict(torch.load(path2, self.device, weights_only=False))
        self.classifier_fail.load_state_dict(torch.load(path3, self.device, weights_only=False))

class Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=(512,256,128)):
        super().__init__()
        self.head =  Sequential(Linear(in_channels, hidden_channels[0]),
                                ReLU(),
                                Linear(hidden_channels[0], hidden_channels[1]),
                                ReLU(),
                                Linear(hidden_channels[1], hidden_channels[2]),
                                ReLU(),
                                Linear(hidden_channels[2], 1),)
        #self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.head(x)
        return x

class PointNetSimple(torch.nn.Module):
    def __init__(self, out_channels=(64,64,128), train_with_norm=True):
        super().__init__()
        torch.manual_seed(12345)
        if train_with_norm:
            in_channels = 6
        else:
            in_channels = 3
        #out_channels = out_channels
        self.train_with_normal = train_with_norm
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        mlp1 = Sequential(Linear(in_channels + 3, out_channels[0]),
                            ReLU(),
                            Linear(out_channels[0], out_channels[0]))
        self.conv1 = PointNetConv(local_nn=mlp1)

        mlp2 = Sequential(Linear(out_channels[0] + 3, out_channels[1]),
                            ReLU(),
                            Linear(out_channels[1], out_channels[1]))
        self.conv2 = PointNetConv(local_nn=mlp2)

        mlp3 = Sequential(Linear(out_channels[1] + 3, out_channels[2]),
                            ReLU(),
                            Linear(out_channels[2], out_channels[2]))
        self.conv3 = PointNetConv(local_nn=mlp3)

    def forward(self, pos, batch=None, normal=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        if self.train_with_normal:
            assert normal is not None
            h = torch.cat((pos, normal), dim=-1)
        else:
            h = pos
        edge_index = knn_graph(pos, k = 16, batch=batch, loop=True)
        # 3. Start bipartite message passing.
        h1 = self.conv1(x=h, pos=pos, edge_index=edge_index)
        h1 = h1.relu()
        h2 = self.conv2(x=h1, pos=pos, edge_index=edge_index)
        #print('h', h.size())
        h2 = h2.relu()
        h3 = self.conv3(x=h2, pos=pos, edge_index=edge_index)
        h3 = h3.relu()
        # # 5. Classifier.
        return h1, h2, h3

class GlobalEmdModel(torch.nn.Module):
    def __init__(self,input_c = 128, inter_c=(256,512,512), output_c=512):
        super().__init__()
        self.mlp1 = Sequential(Linear(input_c, inter_c[0]), ReLU(), Linear(inter_c[0], inter_c[1]), ReLU(), Linear(inter_c[1], inter_c[2]),)
        self.mlp2 = Sequential(Linear(input_c+inter_c[2], output_c), ReLU(), Linear(output_c, output_c))
    def forward(self,pos_emd,radius_p_batch):
        global_emd = self.mlp1(pos_emd)
        global_emd = global_max_pool(global_emd, radius_p_batch)
        global_emd = torch.cat([global_emd[i,:].repeat((radius_p_batch==i).sum(),1) for i in range(len(global_emd))],dim=0)
        global_emd = torch.cat((pos_emd,global_emd),dim=-1)
        global_emd = self.mlp2(global_emd)
        global_emd = global_max_pool(global_emd, radius_p_batch)
        return global_emd