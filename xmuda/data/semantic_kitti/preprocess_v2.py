import os
import os.path as osp
import numpy as np
import pickle
from PIL import Image
import glob
import yaml
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import open3d as o3d

import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
from xmuda.data.semantic_kitti import splits

# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


class DummyDataset(Dataset):
    """Use torch dataloader for multiprocessing"""

    def __init__(self, root_dir, scenes):
        self.root_dir = root_dir
        self.data = []
        self.glob_frames(scenes)

        yaml_path, _ = os.path.split(os.path.realpath(__file__))
        self.dataset_config = yaml.safe_load(
            open(os.path.join(yaml_path, 'semantic-kitti.yaml'), 'r'))
        self.nbr_classes = self.dataset_config['nbr_classes']
        self.grid_dimensions = self.dataset_config['grid_dims']   # [W, H, D]
        self.remap_lut = self.get_remap_lut()
        # self.VOXEL_DIMS = (256, 256, 32)

        # self.img_size = (610, 185)
        # self.downsample = 2
        # self.img_grid = self._create_img_grid()
        # self.scene_lower_bound = np.array([0, -25.6, -2]).reshape(1, -1)
        # self.output_voxel_size = 0.2
        # self.num_voxelized_depth_classes = num_voxelized_depth_classes
        # self.depth_voxel_size = 51.2 / self.num_voxelized_depth_classes

    def get_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map'].keys())] = list(
            self.dataset_config['learning_map'].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    def glob_frames(self, scenes):
        for scene in scenes:
            # glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2', '*.png')
            # cam_paths = sorted(glob.glob(glob_path))

            glob_path = osp.join(self.root_dir, 'dataset',
                                 'sequences', scene, 'voxels', '*.label')
            voxel_paths = sorted(glob.glob(glob_path))

            # load calibration
            calib = self.read_calib(
                osp.join(self.root_dir, 'dataset', 'sequences', scene, 'calib.txt'))

            P = calib['P2']
            T_velo_2_cam = calib['Tr']
            proj_matrix = P @ T_velo_2_cam
            proj_matrix = proj_matrix.astype(np.float32)
            # print(proj_matrix)
            K_intrinsic = P[0:3, 0:3]

            for voxel_path in voxel_paths:
                basename = osp.basename(voxel_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()
                data = {
                    'scene': scene,

                    'frame_id': frame_id,

                    'camera_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2',
                                            frame_id + '.png'),
                    'edge_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2',
                                            frame_id + '_edge.png'),
                    'lidar_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'velodyne',
                                           frame_id + '.bin'),
                    'label_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'labels',
                                           frame_id + '.label'),

                    'voxel_label_path_1_1': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                     frame_id + '.label'),
                    'voxel_label_path_1_4': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                     frame_id + '.label_1_4'),
                    'voxel_label_path_1_16': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                     frame_id + '.label_1_16'),

                    'voxel_invalid_path_1_1': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                       frame_id + '.invalid'),
                    'voxel_invalid_path_1_4': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                       frame_id + '.invalid_1_4'),
                    'voxel_invalid_path_1_16': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                       frame_id + '.invalid_1_16'),

                    'voxel_occupancy': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'voxels',
                                                frame_id + '.bin'),
                    'proj_matrix': proj_matrix,
                    'K_intrinsic': K_intrinsic,
                    'T_velo_2_cam': T_velo_2_cam
                }
                for k, v in data.items():
                    if isinstance(v, str) and k != "scene" and k != "frame_id":
                        if not osp.exists(v):
                            raise IOError('File not found {}'.format(v))
                self.data.append(data)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def get_label_at_scale(self, scale, idx):

        scale_divide = int(scale[-1])
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(
            self.data[idx]["voxel_invalid_path_" + scale])
        LABEL = SemanticKittiIO._read_label_SemKITTI(
            self.data[idx]["voxel_label_path_" + scale])

        if scale == '1_1':
            LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(
                np.float32)  # Remap 20 classes semanticKITTI SSC
            # unique, counts = np.unique(LABEL, return_counts=True)

        # Setting to unknown all voxels marked on invalid mask...
        LABEL[np.isclose(INVALID, 1)] = 255
        LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                           int(self.grid_dimensions[2] /
                                               scale_divide),
                                           int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])
        # LABEL = LABEL.reshape(self.VOXEL_DIMS)
        return LABEL

    def save_points(path, xyz, colors=None):
        """
        xyz: nx3
        """
        # print(xyz)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            # print(xyz.shape, colors.shape)

        o3d.io.write_point_cloud(path, pcd)

    def __getitem__(self, index):
        data_dict = self.data[index].copy()

        scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        label = label.reshape((-1))
        label = label & 0xFFFF  # get lower half for semantics

        # Keep points inside the completion area
        # min extent: [0, -25.6, -2]
        # max extent: [51.2, 25.6,  4.4]
        # voxel size: 0.2
        keep_idx = (points[:, 0] < 51.2) * \
                   (points[:, 1] < 25.6) * (points[:, 1] > -25.6) * \
                   (points[:, 2] < 4.4) * (points[:, 2] > -2)
        points = points[keep_idx, :]
        label = label[keep_idx]

        # load image
        # image = Image.open(data_dict['camera_path'])
        # image_size = image.size
        image_size = (1220, 370)

        keep_idx = points[:, 0] > 0
        points_3d = points[keep_idx]
        label_3d = label[keep_idx].astype(np.int16)
        label_3d = self.remap_lut[label_3d].astype(np.float32)
        # print(points_3d.min(0))
        # Extract points_2d by projecting points into image
        points_hcoords = np.concatenate(
            [points_3d, np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        # T_velo_2_cam = data_dict['T_velo_2_cam']
        # points_3d_cam = (T_velo_2_cam @ points_hcoords.T).T
        img_points = (data_dict['proj_matrix'] @ points_hcoords.T).T
        depth = img_points[:, 2]
        # print(np.sum(depth < 0) / np.sum(depth))
        img_points = img_points[:, :2] / \
            np.expand_dims(depth, axis=1)  # scale 2D points

        keep_idx_img_pts = self.select_points_in_frustum(
            img_points, 0, 0, *image_size)
        keep_idx[keep_idx] = keep_idx_img_pts
        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)

        points_2d = points[keep_idx]
        label_2d = label[keep_idx].astype(np.int16)
        label_2d = self.remap_lut[label_2d].astype(np.float32)

        data_dict['seg_label_3d'] = label_3d
        data_dict['seg_label_2d'] = label_2d
        # points 3d are points in front of the vehicle
        data_dict['points_3d'] = points_3d
        # points 2d are points in the frustum of the lidar
        data_dict['points_2d'] = points_2d

        points_img = img_points[keep_idx_img_pts]
        data_dict['points_img'] = points_img
        # print("dataset", points_img.shape)


        data_dict['image_size'] = np.array(image_size)
        data_dict['ssc_label_1_1'] = self.get_label_at_scale('1_1', index)
        data_dict['ssc_label_1_4'] = self.get_label_at_scale('1_4', index)
        data_dict['ssc_label_1_16'] = self.get_label_at_scale('1_16', index)

        OCCUPANCY = SemanticKittiIO._read_occupancy_SemKITTI(
            data_dict['voxel_occupancy'])
        OCCUPANCY = np.moveaxis(OCCUPANCY.reshape([self.grid_dimensions[0],
                                                   self.grid_dimensions[2],
                                                   self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
        data_dict['voxel_occupancy'] = OCCUPANCY

        return data_dict

    def __len__(self):
        return len(self.data)


def preprocess(scene, root_dir, out_dir):
    pkl_data = []
    #split = getattr(splits, split_name)
    scenes = [scene]

    dataloader = DataLoader(DummyDataset(
        root_dir, scenes), num_workers=10)

    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        # data error leads to returning empty dict
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        # print(data_dict['scene'])
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))

        # convert to relative path
#        lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')
        cam_path = data_dict['camera_path'].replace(root_dir + '/', '')
        edge_path = data_dict['edge_path'].replace(root_dir + '/', '')

        # print(data_dict['voxel_indices'].shape)
        # append data

        out_dict = {
            'scene': data_dict['scene'],
            'frame_id': data_dict['frame_id'],
#            'points_2d': data_dict['points_2d'].numpy(),
            #'points_3d': data_dict['points_3d'].numpy(),
            # 'seg_label_3d': data_dict['seg_label_3d'].numpy(),
#            'seg_label_2d': data_dict['seg_label_2d'].numpy(),
            # row, col format, shape: (num_points, 2)
#            'points_img': data_dict['points_img'].numpy(),
            # 'lidar_path': lidar_path,
            'camera_path': cam_path,
#            'edge_path': edge_path,
            'image_size': tuple(data_dict['image_size'].numpy()),
#            'ssc_label_1_1': data_dict['ssc_label_1_1'].numpy(),
            #'ssc_label_1_2': data_dict['ssc_label_1_2'].numpy(),
            'proj_matrix': data_dict['proj_matrix'].numpy(),
            'K_intrinsic': data_dict['K_intrinsic'].numpy(),
            'T_velo_2_cam': data['T_velo_2_cam'].numpy(),
            'ssc_label_1_4': data_dict['ssc_label_1_4'].numpy(),
            'ssc_label_1_16': data_dict['ssc_label_1_16'].numpy(),
#            'voxel_occupancy': data_dict['voxel_occupancy'].numpy()
        }
        # pkl_data.append(out_dict)

        # print('Skipped {} files'.format(num_skips))

        # save to pickle file
        save_dir = osp.join(out_dir, 'preprocess', data_dict['scene'])
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, '{}.pkl'.format(data_dict['frame_id']))
        with open(save_path, 'wb') as f:
            pickle.dump(out_dict, f, pickle.HIGHEST_PROTOCOL)
            print('Wrote preprocessed data to ' + save_path)


if __name__ == '__main__':
    # root_dir = '/datasets_master/semantic_kitti'
    # out_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess'
    root_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti'
    out_dir = "/gpfsscratch/rech/xqt/uyl37fq/kitti_preprocess"
    scenes = getattr(splits, "train")
    scenes += getattr(splits, "val")
    scenes += getattr(splits, "test")
    for scene in scenes:
        preprocess(scene, root_dir, out_dir)
#        compute_2d_depths_to_3d_voxel_indices(
#            root_dir, out_dir, num_voxelized_depth_classes=num_voxelized_depth_classes)
