import os
import os.path as osp
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import glob
import yaml
import torch
import open3d as o3d
from xmuda.data.semantic_kitti import splits
from xmuda.data.semantic_kitti.preprocess import DummyDataset
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum


def compute_3d_voxel_to_2d_img_indices(root_dir,
                                       out_dir,
                                       voxel_size=0.8, 
                                       downsample=2):
     
    scene_lower_bound = np.array([0, -25.6, -2]).reshape(1, -1)
    scene_size = np.array([51.2, 51.2, 6.4])

    split_train = getattr(splits, "train")
    split_test = getattr(splits, "test")
    split_val = getattr(splits, "val")
    split_hidden_test = getattr(splits, "hidden_test")
    scenes = split_train + split_test + split_val + split_hidden_test
    voxel_grid = create_voxel_grid(scene_size / voxel_size)

    coords_3d = np.ones((1, 3)) * voxel_size / 2 + voxel_size * voxel_grid + scene_lower_bound
    coords_3d_homo = np.concatenate([coords_3d, np.ones((coords_3d.shape[0], 1), dtype=np.float32)], axis=1)
    data_dict = {}
    for scene in scenes:
        calib = DummyDataset.read_calib(
            osp.join(root_dir, 'dataset', 'sequences', scene, 'calib.txt'))
        P = calib['P2']
        T_velo_2_cam = calib['Tr']
        proj_matrix = P @ T_velo_2_cam 
        proj_matrix = proj_matrix.astype(np.float32)
        
        img_indices = (proj_matrix @ coords_3d_homo.T).T
        img_indices = img_indices[:, :2] / np.expand_dims(img_indices[:, 2], axis=1)
        img_indices = np.round(img_indices).astype(np.int32)
        keep_idx = select_points_in_frustum(img_indices, 0, 0, 1220, 370)

        img_indices = img_indices[keep_idx]
        # fliplr so that indexing is row, col and not col, row
        img_indices = np.fliplr(img_indices)

        voxel_indices = voxel_grid[keep_idx, :3]

        print(img_indices.shape, voxel_indices.shape)
        print(img_indices.min(0), img_indices.max(0))
        
        data_dict[scene] = {
            "voxel_indices": voxel_indices, 
            "img_indices": img_indices.astype(np.int32),
        }

    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'voxel_to_pixel_{}.pkl'.format(voxel_size))
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Wrote unprojected data to ' + save_path)


if __name__ == '__main__':
    # root_dir = '/datasets_master/semantic_kitti'
    # out_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess'
    root_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti'
    out_dir = root_dir + '/preprocess'
    scenes = getattr(splits, "train")
    scenes += getattr(splits, "val")
    scenes += getattr(splits, "test")
#    for voxel_size in [0.2, 0.4, 0.8, 1.6]:
    for voxel_size in [0.8]:
        compute_3d_voxel_to_2d_img_indices(root_dir, out_dir, voxel_size=voxel_size)
