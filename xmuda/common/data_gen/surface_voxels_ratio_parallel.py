import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.data.semantic_kitti.params import kitti_class_names as classes
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def sample_rays_per_img(
    intr, T_cam2velo,
    img_size,
    sampled_pixels=None,
    sample_distance=0.4,
    n_pts_per_ray=128):
    """
    pix: (n_rays, 2)
    T: (4, 4)
    """        
    device = intr.device
    if sampled_pixels is None:
        # sampled_pixels = torch.rand(n_rays, 2, device=device)
        # sampled_pixels[:, 0] = sampled_pixels[:, 0] * img_size[0]
        # sampled_pixels[:, 1] = sampled_pixels[:, 1] * img_size[1]
        xs = torch.arange(start=0, end=img_size[0], step=1).type_as(intr)
        ys = torch.arange(start=0, end=img_size[1], step=1).type_as(intr)
        grid_x, grid_y = torch.meshgrid(xs, ys)
        sampled_pixels = torch.cat([
            grid_x.unsqueeze(-1),
            grid_y.unsqueeze(-1)
        ], dim=2)
        # print(sampled_pixels[1000, 100])
        sampled_pixels = sampled_pixels.reshape(-1, 1, 2).expand(-1, n_pts_per_ray, -1)
        # print(sampled_pixels.shape)
        # print(sampled_pixels[110, 60])
    
    n_rays = sampled_pixels.shape[0]

    # Unproject pixels into cam coords
    # homo_pix = torch.cat([sampled_pixels, torch.ones_like(sampled_pixels)[:, :1]], dim=1)
    depth = torch.linspace(
        1,  1 + sample_distance * (n_pts_per_ray - 1), 
        steps=n_pts_per_ray, 
        device=device) \
        .reshape(1, n_pts_per_ray, 1) \
        .expand(n_rays, -1, -1)

    cam_pts = torch.cat([sampled_pixels, depth], dim=2)

    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    cam_pts[:, :, 0] = (cam_pts[:, :, 0] - cx) * cam_pts[:, :, 2] / fx
    cam_pts[:, :, 1] = (cam_pts[:, :, 1] - cy) * cam_pts[:, :, 2] / fy

    # Sample the points along the ray
    # unit = 51.2 / n_pts_per_ray
    

    # cam_pts = depth * cam_pts    
     
    ones = torch.ones(n_rays, n_pts_per_ray, 1).type_as(cam_pts)
    homo_cam_pts = torch.cat([cam_pts, ones], dim=2).float()

    # Change to came coord of the other frame    
    homo_pts = (T_cam2velo @ homo_cam_pts.reshape(-1, 4).T).T    
    homo_pts = homo_pts.reshape(n_rays, n_pts_per_ray, 4)
    pts = homo_pts[:, :, :3]

    return pts # n_rays, n_pts_per_ray, 3

class RayProject(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        pass



kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
kitti_preprocess_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti"
depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
kitti_seg2d_pcd_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/seg_2d_pcd/kitti"


data_module = KittiDataModule(root=kitti_root,
                        preprocess_root=kitti_preprocess_root,
                        frustum_size=1,
                        n_relations=1,
                        project_scale=2,
                        batch_size=1, 
                        num_workers=1)
data_module.setup()
val_dataset = data_module.val_ds
val_dataloader = data_module.val_dataloader()
# train_dataset = KittiDataset(split="train", 
#                              root=kitti_root, 
#                              TSDF_root=kitti_tsdf_root,
#                              depth_root=depth_root)
# val_dataset = KittiDataset(split="val",
#                             root=kitti_root,
#                             TSDF_root=kitti_tsdf_root,
#                             depth_root=depth_root)

# test_dataset = KittiDataset(split="test",
#                             root=kitti_root,
#                             TSDF_root=kitti_tsdf_root,
#                             depth_root=depth_root)



    

# class myThread (threading.Thread):
#     def __init__(self, rays, target):
#         threading.Thread.__init__(self)        
#         self.rays = rays
#         self.target = target
#         self.occ = torch.zeros((256, 256, 32))
#     def run(self):
#         for ray in tqdm(range(self.rays.shape[0])):
#             for pts in range(len(self.rays[ray])):             
#                 xyz = torch.round(self.rays[ray, pts] / (0.2)).long()
#                 if (xyz[0] > (self.occ.shape[0] - 1)) or (xyz[1] > (self.occ.shape[1] - 1)) or (xyz[2] > (self.occ.shape[2] - 1)) or \
#                     (xyz[0] < 0) or (xyz[1] < 0) or (xyz[2] < 0):
#                     break
                            
#                 if self.target[xyz[0], xyz[1], xyz[2]] > 0:                
#                     self.occ[xyz[0], xyz[1], xyz[2]] = 1                 
#                     break
def run_process(rays, target):
    occ = torch.zeros((256, 256, 32))
    # print("a")
    for ray in tqdm(range(rays.shape[0])):
        for pts in range(len(rays[ray])):             
            print(pts)
            xyz = torch.round(rays[ray, pts] / (0.2)).long()
            if (xyz[0] > (occ.shape[0] - 1)) or (xyz[1] > (occ.shape[1] - 1)) or (xyz[2] > (occ.shape[2] - 1)) or \
                (xyz[0] < 0) or (xyz[1] < 0) or (xyz[2] < 0):
                break
                        
            if target[xyz[0], xyz[1], xyz[2]] > 0:                
                occ[xyz[0], xyz[1], xyz[2]] = 1                 
                break
    return 2

def run(data_loader, typ):     
    cnt=0
    total_ratio = 0
    scale = 1
    
    for batch in tqdm(data_loader):
        occ = torch.zeros((256, 256, 32))
        batch['T_velo_2_cam'] = torch.from_numpy(batch['T_velo_2_cam']).float()
        batch['cam_k'] = torch.from_numpy(batch['cam_k']).float()
        T_cam_2_velo = torch.inverse(batch['T_velo_2_cam'])
        K = batch['cam_k']
        target = torch.from_numpy(batch['target'].squeeze())
        print(target.shape)
        target[target == 255] = 0
        target[target > 0] = 1
        distance = 0.19
        print(torch.sum(target))
        
        rays = sample_rays_per_img(
            K, T_cam_2_velo,
            (1220, 370),
            # (1220//2, 370//2),
            sampled_pixels=None,
            sample_distance=distance,
            n_pts_per_ray=int(50/distance)) # n_rays, n_pts_per_ray, 3
        rays[:, :, 1] = rays[:, :, 1] + 25.6
        rays[:, :, 2] = rays[:, :, 2] + 2.0
        ray_cnt = 0
        n_rays = rays.shape[0]
        n_threads = 10
        n_rays_per_thread = n_rays / n_threads
        threads = []
        rets = Queue()        
        for i in range(n_threads):
            start_i = int(i * n_rays_per_thread)
            end_i = int(min((i+1)*n_rays_per_thread, n_rays))
            print(start_i, end_i)
            # threads.append(myThread(
            #     rays[start_i: end_i],
            #     target
            # ))
            # threads.append(Process(target=run_process, args=(rays[start_i: end_i], target, rets,)))
            threads.append(
                (rays[start_i: end_i], target)
            )
        with Pool(10) as p:
            print(p.starmap(run_process, threads))
        # print("start")
        # for thread in threads:
        #     thread.start()
        # print("join")
        # for thread in threads:
        #     thread.join()        
        # for occ_slice in q.get():
        #     occ += occ_slice
        # occ[occ > 1] = 1
        
        # for ray in tqdm(range(rays.shape[0])):
        #     for pts in range(int(50/distance)):             

        #         xyz = torch.round(rays[ray, pts] / (0.2 * scale)).long()
        #         if (xyz[0] > (occ.shape[0] - 1)) or (xyz[1] > (occ.shape[1] - 1)) or (xyz[2] > (occ.shape[2] - 1)) or \
        #             (xyz[0] < 0) or (xyz[1] < 0) or (xyz[2] < 0):
        #             break
                           
        #         if target[xyz[0], xyz[1], xyz[2]] > 0:                
        #             occ[xyz[0], xyz[1], xyz[2]] = 1                 
        #             break
        #     ray_cnt += 1
        #     if ray_cnt % 10000 == 0:
        #         current_ratio = occ.sum() / torch.sum(target) 
        #         pred_ratio = current_ratio * (rays.shape[0] / ray_cnt)
        #         print(current_ratio * 100, pred_ratio * 100)
        ratio = occ.sum() / torch.sum(target)
        total_ratio += ratio
        cnt += 1
        print("======")
        print("n_samples=", cnt)
        print("total ratio=", total_ratio * 100 / cnt)            
        if cnt == 100:
            break
    # print("final ratio=", total_ratio / cnt)
            


is_true_depth = True
#run(train_dataset, "train", is_true_depth=is_true_depth, scale=1)
run(val_dataset, "val")

#run(test_dataset, "test", is_true_depth=is_true_depth, is_gen_seg_2d=False, scale=1)

                  

