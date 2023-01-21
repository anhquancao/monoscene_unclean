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
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid


def depth_to_3d_indices(img_grid, depth,
                        T_inv, K_inv,
                        scene_lower_bound, output_voxel_size,
                        downsample):
    w, h = img_grid
    img_grid = (w // downsample, h // downsample)
    img_grid_homo = np.hstack(
        [img_grid, np.ones((img_grid.shape[0], 1))])
    points_3d_cam = K_inv @ (img_grid_homo * depth).T
    points_3d_cam = points_3d_cam.T  # (N, 3)

    points_3d_cam_homo = np.hstack(
        [points_3d_cam, np.ones((points_3d_cam.shape[0], 1))])  # (N, 4)
    points_3d_lidar_homo = T_inv @ points_3d_cam_homo.T
    points_3d_lidar_homo = points_3d_lidar_homo.T  # (N, 4)

    points_3d_lidar = points_3d_lidar_homo[:, :3]
    # only keep points inside the interested area
    keep_idx = (points_3d_lidar[:, 0] > 0) & \
        (points_3d_lidar[:, 1] > -25.6) & \
        (points_3d_lidar[:, 2] > -2) & \
        (points_3d_lidar[:, 0] < 51.2) & \
        (points_3d_lidar[:, 1] < 25.6) & \
        (points_3d_lidar[:, 2] < 4.4)

    points_3d_lidar = points_3d_lidar[keep_idx]

    voxel_indices = (points_3d_lidar -
                     scene_lower_bound) / output_voxel_size
    voxel_indices = voxel_indices.astype(np.int32)

    # the the resolution of the image is halved when feed to the network
    img_indices = (img_grid_homo[keep_idx, :2] / downsample).astype(np.int32)

#    # Enforce each voxel is passed by only 1 ray
#    values, indices, counts = np.unique(
#        voxel_indices,
#        return_index=True,
#        return_counts=True,
#        axis=0)
#    # print(voxel_indices.shape, img_indices.shape)
#    voxel_indices = voxel_indices[indices]
#    img_indices = img_indices[indices]
    # print(voxel_indices.shape, img_indices.shape)
    return {
        "voxel_indices": voxel_indices,
        "img_indices": img_indices
    }


def compute_2d_depths_to_3d_voxel_indices(root_dir,
                                          out_dir,
                                          img_size=(1220, 370),
                                          output_voxel_size=0.8,
                                          downsample=8,
                                          max_pixels_per_voxel=64,
                                          num_voxelized_depth_classes=64):
    depth_voxel_size = 51.2 / num_voxelized_depth_classes
    scene_lower_bound = np.array([0, -25.6, -2]).reshape(1, -1)

    split_train = getattr(splits, "train")
    split_test = getattr(splits, "test")
    split_val = getattr(splits, "val")
    split_hidden_test = getattr(splits, "hidden_test")
    scenes = split_train + split_test + split_val + split_hidden_test
    img_grid = create_img_grid(img_size, downsample=downsample)
    data_dict = {}
    for scene in scenes:
        calib = DummyDataset.read_calib(
            osp.join(root_dir, 'dataset', 'sequences', scene, 'calib.txt'))
        P = calib['P2']
        T_velo_2_cam = calib['Tr']
        K_intrinsic = P[0:3, 0:3]
        K_inv = np.linalg.inv(K_intrinsic)
        T_inv = np.linalg.inv(T_velo_2_cam)

        voxel_indices_all = []
        img_indices_all = []

        for depth_class in range(num_voxelized_depth_classes):
            depth = depth_voxel_size / 2.0 + depth_voxel_size * depth_class
            res = depth_to_3d_indices(img_grid, depth,
                                      T_inv, K_inv,
                                      scene_lower_bound, output_voxel_size,
                                      downsample)

            voxel_indices = res["voxel_indices"]
            img_indices = res["img_indices"]

            depth_idx = np.ones((voxel_indices.shape[0], 1)) * depth_class
            voxel_indices = np.hstack([voxel_indices, depth_idx])
            img_indices = np.hstack([img_indices, depth_idx])

            voxel_indices_all.append(voxel_indices)
            img_indices_all.append(img_indices)

        voxel_indices = np.vstack(voxel_indices_all)
        img_indices = np.vstack(img_indices_all)

        # Enforce each voxel is passed by only 1 ray
        voxel_indices, indices, inverse = np.unique(
            voxel_indices,
            return_index=True,
            return_inverse=True,
            axis=0)

        voxel_to_pixel_indices_mapping = []
        for i, index in tqdm(enumerate(indices)):
            pixel_indices = np.where(inverse == i)[0]
            if len(pixel_indices) > max_pixels_per_voxel:
                pixel_indices = pixel_indices[:max_pixels_per_voxel]
            else:
                pad_widths = (0, int(max_pixels_per_voxel - len(pixel_indices)))
                pixel_indices = np.pad(pixel_indices, pad_widths , 'edge')
#                print(len(pixel_indices))
            
            voxel_to_pixel_indices_mapping.append(pixel_indices) 
        voxel_to_pixel_indices_mapping = np.array(voxel_to_pixel_indices_mapping)
        print(voxel_indices.shape, img_indices.shape, voxel_to_pixel_indices_mapping.shape)

        data_dict[scene] = {
            "voxel_indices": voxel_indices,
            "img_indices": img_indices,
            "voxel_to_pixel_indices": voxel_to_pixel_indices_mapping
        }
#            data_dict[scene] = {
#            "voxel_indices": voxel_indices,
#            "img_indices": img_indices
#        }
    # print(data_dict['01']['voxel_incdices'][0].shape, data_dict['01']['img_indices'][0].shape)
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'unproject_{}_{}.pkl'.format(
        num_voxelized_depth_classes, max_pixels_per_voxel))
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
    for num_voxelized_depth_classes in [16, 32]:
        compute_2d_depths_to_3d_voxel_indices(
            root_dir, out_dir, num_voxelized_depth_classes=num_voxelized_depth_classes)
