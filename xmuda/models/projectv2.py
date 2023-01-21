import torch
import torch.nn as nn
from torch_scatter import scatter_max
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum

class Project2D3Dv2(nn.Module):
    def __init__(self, scene_size, feature, 
                 voxel_size, 
                 downsample, 
                 project_min=0.05, 
                 project_max=0.95):
        super().__init__()
        print(project_min, project_max)
        self.scene_size = scene_size
        self.voxel_size = voxel_size
        self.min = project_min
        self.max = project_max
        self.downsample = downsample
        self.img_W = 640
        self.img_H = 480
#        self.positional_encodings = nn.Parameter(torch.rand(feature, 
#                                                            scene_size[0], 
#                                                            scene_size[1], 
#                                                            scene_size[2]), requires_grad=True)
        

    def forward(self, x2d, inv_cam_pose, vox_origin, cam_k, old_voxel_indices, old_image_indices, fliplr):
        c, h, w = x2d.shape
        src = x2d.view(c, -1) 
        img_indices, voxel_indices = self.voxel2pixel(inv_cam_pose, vox_origin, self.voxel_size, cam_k, fliplr)
        if fliplr:
            img_indices[:, 1] = self.img_W - 1 - img_indices[:, 1] 
        img_indices = img_indices // self.downsample

#        diff = torch.sum(img_indices - old_image_indices, dim=1)
#        print(img_indices[diff != 0], old_image_indices[diff != 0], voxel_indices[diff != 0])
        img_indices = img_indices[:, 0] * w + img_indices[:, 1] 
        img_indices = img_indices.expand(c, -1).long()
#        print(img_indices.shape)
        src_feature = torch.gather(src, 1, img_indices) 
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

    def voxel2pixel(self, inv_cam_pose, vox_origin, voxel_size, cam_k, fliplr):
        scene_size = (int(4.8 / voxel_size), int(2.88/voxel_size), int(4.8/voxel_size)) 
#        print(voxel_size)
        # Create voxel grid in cam coords
        vox_origin = vox_origin.double()
        voxel_grid = torch.from_numpy(create_voxel_grid(scene_size)).type_as(vox_origin)
#        deviation = (self.min + (self.max - self.min) * torch.rand(voxel_grid.shape).type_as(vox_origin)) * voxel_size
#        pts_world2 =  deviation + voxel_size * voxel_grid
        pts_world2 =  torch.ones_like(voxel_grid) * 0.5 * voxel_size + voxel_size * voxel_grid
        pts_world = torch.zeros_like(pts_world2)
        pts_world[:, 0] = pts_world2[:, 0]
        pts_world[:, 1] = pts_world2[:, 2]
        pts_world[:, 2] = pts_world2[:, 1]

        vox_origin = vox_origin.reshape(1, 3)
        pts_world += vox_origin

        pts_world_homo = torch.cat([pts_world, torch.ones([pts_world.shape[0], 1]).type_as(pts_world)], dim=1)

#        R_cam = np.identity(4)
#        R_cam[:3, :3] = cam_pose[:3, :3]
#        t_cam = np.identity(4)
#        t_cam[:3, 3] = - cam_pose[:3, 3]
#        T_world_to_cam = R_cam @ t_cam
        T_world_to_cam = inv_cam_pose
#        print(T_world_to_cam.dtype, pts_world_homo.dtype)
        pts_cam_homo = (T_world_to_cam @ pts_world_homo.T).T
#        print("pts_cam", pts_cam_homo.min(0), pts_cam_homo.max(0))

        # remove points with depth < 0
        keep_idx = pts_cam_homo[:, 2] > 0
        pts_cam_homo = pts_cam_homo[keep_idx]
        voxel_grid = voxel_grid[keep_idx]
#        print("pts_cam_filtered", pts_cam_homo.min(0), pts_cam_homo.max(0))

        pts_cam = pts_cam_homo[:, :3]
#        print(pts_cam.min(0), pts_cam.max(0))
        pts_img = (cam_k @ pts_cam.T).T
#        print("pts_img", pts_img.shape)
#        print(np.expand_dims(pts_img[:, 2], axis=1).shape)
        pts_img = pts_img[:, :2] / torch.unsqueeze(pts_img[:, 2], dim=1)
        pts_img = torch.round(pts_img)


        keep_idx = select_points_in_frustum(pts_img, 0, 0, self.img_W , self.img_H)

        img_indices = pts_img[keep_idx]
#        # fliplr so that indexing is row, col and not col, row
        img_indices = torch.fliplr(img_indices)
#
        voxel_indices = voxel_grid[keep_idx]
#
        return img_indices, voxel_indices 
