import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class ProjectROIPool(nn.Module):
    def __init__(self, dataset, scene_size, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, bb, valid_bb, scale):
        d, h, w = x2d.shape
#        print(h, w)

        x3d = torch.zeros(d, self.scene_size[0] * self.scene_size[2] * self.scene_size[1]).type_as(x2d)
        bb = bb[valid_bb, :]
        bb[:, :2] = bb[:, :2] * 1.0 / scale
        bb[:, 2:] = bb[:, 2:] * 1.0 / scale
        x_equal = bb[:, 0] == bb[:, 2]
        y_equal = bb[:, 1] == bb[:, 3]
        bb[x_equal, 0] -= 1
        bb[x_equal, 2] += 1
        bb[y_equal, 1] -= 1
        bb[y_equal, 3] += 1

        bb[bb < 0] = 0
        bb[bb[:, 2] >= w, 2] = w - 1
        bb[bb[:, 3] >= h, 3] = h - 1

#        print("x", torch.sum(bb[:, 0] == bb[:, 2]))
#        print("y", torch.sum(bb[:, 1] == bb[:, 3]))
        
        bb = bb.type_as(x2d)
        roi_feat = roi_align(x2d.unsqueeze(0), [bb], output_size=1, spatial_scale=1, sampling_ratio=2) 
        roi_feat = roi_feat.squeeze().T
#        print(roi_feat.shape, x3d[:, valid_bb].shape)
        x3d[:, valid_bb] = x3d[:, valid_bb] + roi_feat
#        print("roi_feat", torch.sum(roi_feat.sum(0) != 0))
#        print("x3d", torch.sum(x3d.sum(0) != 0))
        if self.dataset == "NYU":
#            x3d = src_feature.reshape(c, self.scene_size[0], self.scene_size[2], self.scene_size[1]) 
            x3d = x3d.reshape(d, self.scene_size[0], self.scene_size[2], self.scene_size[1])
            x3d = x3d.permute(0, 1, 3, 2)
            print("x3d reshape", torch.sum(x3d.sum(0) != 0), torch.sum(valid_bb))
        elif self.dataset == "kitti":
            x3ds = x3ds.reshape(bs, c,
                                self.scene_size[0] // self.project_scale,
                                self.scene_size[1] // self.project_scale,
                                self.scene_size[2] // self.project_scale)
        
        return x3d

class Project2D3D(nn.Module):
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, pix, valid_pix):
        c, h, w = x2d.shape

        src = x2d.view(c, -1) 
        zeros_vec = torch.zeros(c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], 1)

        pix_x, pix_y = pix[:, 0], pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~valid_pix] = h * w
        img_indices = img_indices.expand(c, -1).long() # c, HWD
        
        
        src_feature = torch.gather(src, 1, img_indices) 
        if self.dataset == "NYU":
            x3d = src_feature.reshape(c, self.scene_size[0], self.scene_size[2], self.scene_size[1]) 
            x3d = x3d.permute(0, 1, 3, 2)
        elif self.dataset == "kitti":
#            x3d = src_feature.reshape(c, self.scene_size[0]//2, self.scene_size[1]//2, self.scene_size[2]//2)
#            x3d = src_feature.reshape(c, self.scene_size[0], self.scene_size[1], self.scene_size[2])
            x3d = src_feature.reshape(c,
                                      self.scene_size[0] // self.project_scale,
                                      self.scene_size[1] // self.project_scale,
                                      self.scene_size[2] // self.project_scale)
        
        return x3d

