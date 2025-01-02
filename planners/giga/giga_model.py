import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch_scatter import scatter_mean
from torch.nn import init
import time
from scipy import ndimage
import open3d as o3d
from math import cos, sin


from .giga_utils import *


### CLASS TO EVALUATE GIGA, PERFORM PRE AND POST PROCESSING OF DATA ###

# TODO: what to put in config file?

class GIGANet:
    def __init__(self, cfg):
        self._giga_cfg = cfg

        # some params from giga init
        # TODO: put these in config file
        self.best = False # only return top grasp
        self.always_detect = True # always return at least one grasp, regardless of threshold
        self.qual_th = 0.8 # 0.9
        self.out_th = 0.3 # 0.5
        self.tsdf_resolution = 100 # 40
        # TODO: eventually, increase this
        self.sample_resolution = 100 # 40
        self.size = 0.8 # 0.3

        # instantiate model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GIGAModel(cfg, self.device)

        # load weights and put into model
        # TODO: put some of this in config?
        checkpoint_path = 'planners/giga/checkpoints/giga_pile.pt'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint)
        print('Done loading model params.')

        # TODO: define separate resolution parameter for this?
        # set up positions to use for grasp centers during prediction
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.sample_resolution, steps=self.sample_resolution), \
                                torch.linspace(start=-0.5, end=0.5 - 1.0 / self.sample_resolution, steps=self.sample_resolution), \
                                torch.linspace(start=-0.5, end=0.5 - 1.0 / self.sample_resolution, steps=self.sample_resolution), \
                                indexing='ij')
        # 1, self.sample_resolution, self.sample_resolution, self.sample_resolution, 3
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.sample_pos = pos.view(1, self.sample_resolution * self.sample_resolution * self.sample_resolution, 3)

    # run model for an entire scene
    def predict_scene_grasps(self, depth_image, camera_intrinsics, camera_pose):

        # create scene TSDF
        # TODO: time this, is it ineffecient?
        voxel_size = self.size / self.tsdf_resolution
        sdf_trunc = 4 * voxel_size

        tsdf = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.tsdf_resolution,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_image)),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=camera_intrinsics.width,
            height=camera_intrinsics.height,
            fx=camera_intrinsics.fx,
            fy=camera_intrinsics.fy,
            cx=camera_intrinsics.cx,
            cy=camera_intrinsics.cy,
        )

        # add offset for TSDF origin to camera pose, don't offset z-coords
        camera_pose[0,3] += self.size/2.0
        camera_pose[1,3] += self.size/2.0
        camera_pose[2,3] += self.size/2.0
        tsdf.integrate(rgbd, intrinsic, np.linalg.inv(camera_pose))
        # TODO: is this part too slow?
        # what is this doing? why is it pulling out color?
        shape = (1, self.tsdf_resolution, self.tsdf_resolution, self.tsdf_resolution)
        tsdf_vol = np.zeros(shape, dtype=np.float32)
        voxels = tsdf.extract_voxel_grid().get_voxels()
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_vol[0, i, j, k] = voxel.color[0]

        # Show point cloud
        pcd_new = tsdf.extract_point_cloud()
        # o3d.visualization.draw_geometries([pcd_new])
        # show mesh
        mesh = tsdf.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])

        # run model
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = self.model( torch.from_numpy(tsdf_vol).to(self.device), self.sample_pos)
        # move outputs back to cpu, then reshape
        qual_vol = qual_vol.cpu().squeeze().numpy()
        rot_vol = rot_vol.cpu().squeeze().numpy()
        width_vol = width_vol.cpu().squeeze().numpy()
        qual_vol = qual_vol.reshape((self.sample_resolution, self.sample_resolution, self.sample_resolution))
        rot_vol = rot_vol.reshape((self.sample_resolution, self.sample_resolution, self.sample_resolution, 4))
        width_vol = width_vol.reshape((self.sample_resolution, self.sample_resolution, self.sample_resolution))

        # process outputs
        gaussian_filter_sigma=1.0 # TODO: put these in config file
        min_width=0.033
        max_width=0.233
        out_th=0.5
        tsdf_vol = tsdf_vol.squeeze()
        # smooth quality volume with a Gaussian
        qual_vol = ndimage.gaussian_filter(
            qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
        )
        # mask out voxels too far away from the surface
        # TODO: need to change this if we want higher resolution sampling than TSDF?
        outside_voxels = tsdf_vol > out_th
        inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
        valid_voxels = ndimage.morphology.binary_dilation(
            outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
        )
        qual_vol[valid_voxels == False] = 0.0

        # reject voxels with predicted widths that are too small or too large
        qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

        # avoid grasp out of bounds [0.02  0.02  0.055]
        limit=[0.02, 0.02, 0.055] # TODO: put this in config file
        x_lim = int(limit[0] / voxel_size)
        y_lim = int(limit[1] / voxel_size)
        z_lim = int(limit[2] / voxel_size)
        # TODO: uncomment this once limits are corrected
        # TODO: is this even necessary?
        # qual_vol[:x_lim] = 0.0
        # qual_vol[-x_lim:] = 0.0
        # qual_vol[:, :y_lim] = 0.0
        # qual_vol[:, -y_lim:] = 0.0
        # qual_vol[:, :, :z_lim] = 0.0

        # select grasps to return
        LOW_TH = 0.5 # TODO: put this in config file?
        max_filter_size=4
        center_vol = self.sample_pos.view(self.sample_resolution, self.sample_resolution, self.sample_resolution, 3).cpu()
        best_only = False
        qual_vol[qual_vol < LOW_TH] = 0.0
        if self.always_detect and (qual_vol >= self.qual_th).sum() == 0:
            # worst case, still return the best grasp (even if it has a low score)
            best_only = True
        else:
            # threshold on grasp quality
            qual_vol[qual_vol < self.qual_th] = 0.0
        # non maximum suppression
        max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
        qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
        mask = np.where(qual_vol, 1.0, 0.0)

        # construct grasps
        grasps, scores, widths = [], [], []
        centers = []
        for index in np.argwhere(mask):
            i, j, k = index
            score = qual_vol[i, j, k]
            ori = Rotation.from_quat(rot_vol[i, j, k])
            center = center_vol[i, j, k].numpy()
            width = width_vol[i, j, k]
            # TODO: just save grasp pose, widths
            grasp = Transform(ori, center)
            grasps.append(grasp)
            scores.append(score)
            widths.append(width)
            centers.append(center)

        # TODO: show grasp points and orientations here with voxel point cloud and mesh?
        center_pts = (np.asarray(centers)+0.5)*self.size
        center_colors = np.zeros_like(center_pts)
        center_colors[:,0] = 1.0
        chosen_centers_pcd = o3d.geometry.PointCloud()
        chosen_centers_pcd.points = o3d.utility.Vector3dVector(center_pts)
        chosen_centers_pcd.colors = o3d.utility.Vector3dVector(center_colors)
        base_pts = np.asarray(pcd_new.points)
        base_colors = np.zeros_like(base_pts)
        # TODO: save quality score for each base point to use as affordance map?
        base_colors[:,2] = 1.0
        base_pcd = o3d.geometry.PointCloud()
        base_pcd.points = o3d.utility.Vector3dVector(base_pts)
        base_pcd.colors = o3d.utility.Vector3dVector(base_colors)
        # o3d.visualization.draw_geometries([base_pcd, chosen_centers_pcd])

        # sort grasps by score
        sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
        sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]
        sorted_widths = [widths[i] for i in reversed(np.argsort(scores))]
        if best_only and len(sorted_grasps) > 0:
            sorted_grasps = [sorted_grasps[0]]
            sorted_scores = [sorted_scores[0]]
            sorted_widths = [sorted_widths[0]]

        grasps, scores, widths = np.asarray(sorted_grasps), np.asarray(sorted_scores), np.asarray(sorted_widths)
        grasp_poses = []
        grasp_widths = []
        grasp_scores = []
        if len(grasps) > 0:
            # choose which grasps to return
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                # g.translation[2] += 0.5 # set zero of z-coordinate
                g.translation = g.translation * self.size # scale posiitons to TSDF size
                new_pose = np.eye(4)
                # need to swap some axes here, since grasp frame is different than our baseline CGN parameterization
                new_pose[:3,0] = g.rotation.as_matrix()[:3,1]
                new_pose[:3,1] = -g.rotation.as_matrix()[:3,0]
                new_pose[:3,2] = g.rotation.as_matrix()[:3,2]
                offset_dist = 0.06
                new_pose[:3,3] = g.translation - offset_dist*g.rotation.as_matrix()[:3,2]
                grasp_poses.append(new_pose)
            for w in widths[p]:
                width = w * self.size
                grasp_widths.append(w)
            for s in scores[p]:
                grasp_scores.append(s)

        # convert to dict format used by other models
        grasp_poses_array = np.zeros((len(grasp_scores),4,4))
        for i in range(len(grasp_scores)):
            grasp_poses_array[i,:4,:4] = grasp_poses[i]
        grasp_poses = {-1: grasp_poses_array}
        grasp_scores = {-1: grasp_scores}
        grasp_widths = {-1: grasp_widths}
        # return grasps, scores, widths
        return grasp_poses, grasp_scores, grasp_widths



### GIGA MODEL ITSELF ###

class GIGAModel(nn.Module):
    # Based on Convolutional Occupancy Network class.
    def __init__(self, global_config, device):
        super().__init__()

        # TODO: extact some values from global config?

        self.device = device

        c_dim = 32
        padding = 0
        # if padding is None:
        #     padding = 0.1

        self.decoder_qual = LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=1,
            dim=3, sample_mode='bilinear', hidden_size=32, concat_feat=True
        )
        self.decoder_rot = LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=4,
            dim=3, sample_mode='bilinear', hidden_size=32, concat_feat=True
        )
        self.decoder_width = LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=1,
            dim=3, sample_mode='bilinear', hidden_size=32, concat_feat=True
        )
        self.decoder_tsdf = LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=1,
            dim=3, sample_mode='bilinear', hidden_size=32, concat_feat=True
        )

        self.encoder = LocalVoxelEncoder(
            c_dim=c_dim, padding=padding,
            dim=3,
            plane_type=['xz', 'xy', 'yz'],
            plane_resolution=40,
            unet_kwargs = {
                        'depth': 3,
                        'merge_mode': 'concat',
                        'start_filts': 32
                    }
        )

        # move encoder and decoders to CUDA device
        self.decoder_qual.to(device)
        self.decoder_rot.to(device)
        self.decoder_width.to(device)
        self.decoder_tsdf.to(device)
        self.encoder.to(device)

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        # feature = self.query_feature(p, c)
        # qual, rot, width = self.decode_feature(p, feature)
        qual, rot, width = self.decode(p, c)
        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = - qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            #print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, width_out


### ENCODER AND DECODER MODULES

# TODO: combine with UNet module?
# TODO: these can be less generalized?
class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    # TODO: clean up args based on config
    def __init__(self, dim=3, c_dim=128, unet_kwargs=None,
                    plane_resolution=512, grid_resolution=None, plane_type='xz', kernel_size=3, padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)

        self.c_dim = c_dim

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution

        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)

        # Acquire voxel-wise feature
        x = x.unsqueeze(1)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                    hidden_size=256,
                    n_blocks=5,
                    out_dim=1,
                    leaky=False,
                    sample_mode='bilinear',
                    padding=0.1,
                    concat_feat=False,
                    no_xyz=False):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if 'grid' in plane_type:
                    c = self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xz'], plane='xz'))
                if 'xy' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xy'], plane='xy'))
                if 'yz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['yz'], plane='yz'))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if 'grid' in plane_type:
                    c += self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
                if 'xy' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                if 'yz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c = c.transpose(1, 2)

        p = p.float()

        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


### HELPER MODULES

# UNets
# Code from GIGA and https://github.com/jaxony/unet-pytorch/blob/master/model.py
def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                    merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                    start_filts=64, up_mode='transpose',
                    merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                                "upsampling. Only \"transpose\" and "
                                "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                                "merging up and down paths. "
                                "Only \"concat\" and "
                                "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                                "with merge_mode \"add\" at the moment "
                                "because it doesn't make sense to use "
                                "nearest neighbour to reduce "
                                "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx