import torch
import torch.nn as nn
from torch_scatter import scatter_max
from torchvision.ops import RoIAlign

class Project2D3DROIAlign(nn.Module):
    def __init__(self, scene_size, feature, sampling_ratio, pool_size, spatial_scale=1/4):
        super().__init__()
        self.scene_size = scene_size
        self.pool_size = pool_size
#        self.output_size = (2, 2)
        self.output_size = 1
        self.sampling_ratio = sampling_ratio
        self.img_H = 480
        self.img_W = 640
        self.roi_align = RoIAlign(self.output_size, spatial_scale=spatial_scale, sampling_ratio=self.sampling_ratio, aligned=True)
#        self.resize = nn.Linear(feature * 4, feature)

    def forward(self, x2d, voxel_indices, img_indices, dist_to_cam):
        rois = torch.zeros(img_indices.shape[0], 5).type_as(x2d)
        x = img_indices[:, 1]
        y = img_indices[:, 0]
#        pool_sizes = self.pool_size * torch.round(dist_to_cam + 1)
#        x1 = x - pool_sizes 
#        y1 = y - pool_sizes
#        x2 = x + pool_sizes
#        y2 = y + pool_sizes
        x1 = x - self.pool_size
        y1 = y - self.pool_size
        x2 = x + self.pool_size
        y2 = y + self.pool_size
        rois[:, 1] = x1
        rois[:, 2] = y1
        rois[:, 3] = x2
        rois[:, 4] = y2

        c, h, w = x2d.shape
        x2d = x2d.unsqueeze(0)
        x_roi = self.roi_align(x2d, rois)
        x_roi_flatten = x_roi.reshape(x_roi.shape[0], -1)
        src_feature = x_roi_flatten.T
#        src_feature = self.resize(x_roi_flatten).T

#        src = x2d.view(c, -1) 
#        img_indices = img_indices[:, 0] * w + img_indices[:, 1] 
#        img_indices = img_indices.expand(c, -1).long()
#        print(img_indices.shape)
#        src_feature = torch.gather(src, 1, img_indices) 
#        print("src_feature", src_feature.shape)

        x3d = torch.zeros(c, self.scene_size[0] * self.scene_size[1] * self.scene_size[2], device=x2d.device)
        voxel_indices = voxel_indices[:, 0] * (self.scene_size[1] * self.scene_size[2]) + voxel_indices[:, 1] * self.scene_size[2] + voxel_indices[:, 2] 
        voxel_indices = voxel_indices.expand(c, -1).long()
#        print(voxel_indices.shape)
        assert voxel_indices.shape == src_feature.shape
        x3d.scatter_(1, voxel_indices, src_feature) 
#        print(torch.sum(x3d))
        x3d = x3d.view(c, self.scene_size[0], self.scene_size[1], self.scene_size[2])
#        x3d += self.positional_encodings
        return x3d


    def voxel2pixel(self, inv_cam_pose, vox_origin, voxel_size, cam_k, fliplr, sample_position=0.5):
        scene_size = (int(4.8 / voxel_size), int(2.88/voxel_size), int(4.8/voxel_size)) 
        vox_origin = vox_origin.double()
        voxel_grid = torch.from_numpy(create_voxel_grid(scene_size)).type_as(vox_origin)
        pts_world2 =  torch.ones_like(voxel_grid) * sample_position * voxel_size + voxel_size * voxel_grid
        pts_world = torch.zeros_like(pts_world2)
        pts_world[:, 0] = pts_world2[:, 0]
        pts_world[:, 1] = pts_world2[:, 2]
        pts_world[:, 2] = pts_world2[:, 1]

        vox_origin = vox_origin.reshape(1, 3)
        pts_world += vox_origin

        pts_world_homo = torch.cat([pts_world, torch.ones([pts_world.shape[0], 1]).type_as(pts_world)], dim=1)

        T_world_to_cam = inv_cam_pose
        pts_cam_homo = (T_world_to_cam @ pts_world_homo.T).T

        # remove points with depth < 0
#        keep_idx = pts_cam_homo[:, 2] > 0
#        pts_cam_homo = pts_cam_homo[keep_idx]
#        voxel_grid = voxel_grid[keep_idx]

        pts_cam = pts_cam_homo[:, :3]
        pts_img = (cam_k @ pts_cam.T).T
        pts_img = pts_img[:, :2] / torch.unsqueeze(pts_img[:, 2], dim=1)
        pts_img = torch.round(pts_img) # (x, y)


#        keep_idx = select_points_in_frustum(pts_img, 0, 0, self.img_W , self.img_H)

#        img_indices = pts_img[keep_idx]
##        # fliplr so that indexing is row, col and not col, row
#        img_indices = torch.fliplr(img_indices)
#
#        voxel_indices = voxel_grid[keep_idx]
#
        return img_indices, voxel_indices 
