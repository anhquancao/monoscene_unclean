from xmuda.data.semantic_kitti.kitti_dataset import KittiDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
import os
import cv2
import pickle

kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
TSDF_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
train_dataset = KittiDataset(split="train", 
                             root=kitti_root, 
#                             TSDF_root=TSDF_root,
                             depth_root=depth_root)
val_dataset = KittiDataset(split="val", 
                            root=kitti_root, 
#                            TSDF_root=TSDF_root,
                            depth_root=depth_root)

test_dataset = KittiDataset(split="test", 
                            root=kitti_root, 
#                            TSDF_root=TSDF_root,
                            depth_root=depth_root)

def depth2voxel(depth, cam_pose, vox_origin, cam_k):
    voxel_size = (256, 256, 32)
    unit = 0.2
    # ---- Get point in camera coordinate
    H, W = depth.shape
    gx, gy = np.meshgrid(range(W), range(H))
    pt_cam = np.zeros((H, W, 3), dtype=np.float32)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
    pt_cam[:, :, 2] = depth  # z, in meter
    # ---- Get point in world coordinate
    p = cam_pose
    pt_world = np.zeros((H, W, 3), dtype=np.float32)
    pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
    pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
    pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
    pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
    pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
    pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
    # ---- Aline the coordinates with labeled data (RLE .bin file)
    #	pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
    # pt_world2 = pt_world
    #	pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
    #	pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
    #	pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

    # ---- World coordinate to grid/voxel coordinate
    point_grid = pt_world / unit  # Get point in grid coordinate, each grid is a voxel
    point_grid = np.rint(point_grid).astype(np.int32)  # .reshape((-1, 3))  # (H*W, 3) (H, W, 3)

    # ---- crop depth to grid/voxel
    # binary encoding '01': 0 for empty, 1 for occupancy
    voxel_binary = np.zeros([_ + 1 for _ in voxel_size], dtype=np.float32)  # (W, H, D)
    voxel_xyz = np.zeros(voxel_size + (3,), dtype=np.float32)  # (W, H, D, 3)
    position = np.zeros((H, W), dtype=np.int32).reshape(-1)
    position4 = np.zeros((H, W), dtype=np.int32).reshape(-1)

    voxel_size_lr = (voxel_size[0] // 4, voxel_size[1] // 4, voxel_size[2] // 4)
    point_grid = point_grid.reshape(-1, 3)
    mask = (point_grid[:, 0] < voxel_size[0]) & (point_grid[:, 1] < voxel_size[1]) & (point_grid[:, 2] < voxel_size[2]) &\
           (point_grid[:, 0] >= 0) & (point_grid[:, 1] >= 0) & (point_grid[:, 2] >= 0) 

    position[mask] = np.ravel_multi_index(np.array([point_grid[mask, 0], point_grid[mask, 1], point_grid[mask, 2]]), voxel_size)
    position = position.reshape(H, W)
    point_grid4 = (point_grid / 4).astype(np.int32)
    position4[mask] = np.ravel_multi_index([point_grid4[mask, 0], point_grid4[mask, 1], point_grid4[mask, 2]], voxel_size_lr)
    position4 = position4.reshape(H, W)
#	for h in range(H):
#		for w in range(W):
#			i_x, i_y, i_z = point_grid[h, w, :]
#			if 0 <= i_x < voxel_size[0] and 0 <= i_y < voxel_size[1] and 0 <= i_z < voxel_size[2]:
#				voxel_binary[i_x, i_y, i_z] = 1  # the bin has at least one point (bin is not empty)
#				voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
#				position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_size)
#				position4[h, w] = np.ravel_multi_index((point_grid[h, w, :] / 4).astype(np.int32), voxel_size_lr)
	# output --- 3D Tensor, 240 x 144 x 240
    del depth, gx, gy, pt_cam, pt_world, point_grid  # Release Memory
    return position, position4 # (W, H, D), (W, H, D, 3)

def run(dataset, typ):
    mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/kitti"

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        T_velo_2_cam = batch['T_velo_2_cam']
        P = batch['P']
        name = batch['name']
        sequence = batch['sequence']
        vox_origin = np.array([0, -25.6, -2])
        save_dir = os.path.join(mapping_root, sequence)
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        depth = batch['depth'].squeeze()[:370, :1220]

        mapping_path_1_1 = os.path.join(save_dir, name + "_1_1.npy")
        mapping_path_1_4 = os.path.join(save_dir, name + "_1_4.npy")

        if not os.path.exists(mapping_path_1_4):
#        if True:
            
            position, position4 = depth2voxel(depth, np.linalg.inv(T_velo_2_cam), vox_origin, P[0:3, 0:3])

            np.save(mapping_path_1_1, position)
            print("wrote to", mapping_path_1_1)
            np.save(mapping_path_1_4, position4)
            print("wrote to", mapping_path_1_4)

run(test_dataset, "test")
#run(train_dataset, "train")
                  
