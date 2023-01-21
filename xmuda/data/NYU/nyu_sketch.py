import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import numpy.matlib
from PIL import Image
from torchvision import transforms
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum
from xmuda.models.ssc_loss import construct_ideal_affinity_matrix
import pickle
import imageio
from tqdm import tqdm
from itertools import combinations
import time
import random


seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 
                 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11] 

class NYUDataset(Dataset):
    def __init__(self, 
                 split,
                 root, 
                 preprocess_dir, 
                 color_jitter=None,
                 fliplr=0.0,
                 flipud=0.0,
                 flip3d_x=0.0,
                 flip3d_z=0.0,
                 random_scales=False,
                 extract_data=False):
        self.n_classes = 12
        self.extract_data = extract_data
        self.root = os.path.join(root, "NYU" + split)
        self.sketch_root = os.path.join(root, "sketch3D")
        self.tsdf_root = os.path.join(root, "TSDF")
        self.sketch_mapping_root = os.path.join(root, "Mapping")
        self.preprocess_dir = os.path.join(preprocess_dir, split)
        self.aic_npz_root = os.path.join("/gpfsscratch/rech/xqt/uyl37fq/AIC_dataset/", "NYU" + split + "_npz")
        self.fliplr = fliplr
        self.flipud = flipud
        self.flip3d_x = flip3d_x
        self.flip3d_z = flip3d_z
        self.random_scales = random_scales

        self.voxel_size = 0.02 # 0.02m
        self.scene_size = (240, 144, 240)
        self.img_W = 640
        self.img_H = 480
        self.cam_k = np.array([
            [518.8579, 0, 320],
            [0, 518.8579, 240],
            [0, 0, 1]
        ])

        self.color_jitter = transforms.ColorJitter(*color_jitter) if color_jitter else None
#        self.resize_rgb = transforms.Resize((int(self.img_H), 
#                                             int(self.img_W)))

        self.scan_names = glob.glob(os.path.join(self.root, '*.bin'))
        s = []
        for f in self.scan_names:
#            if 'NYU0223' in f:
            s.append(f)
        self.scan_names = s
        self.downsample = 4
        self.class_map = self.compute_class_relation_map(self.n_classes)

        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]
#        if "NYU0066_0000" not in name:
#            return

        if self.extract_data:
            bin_path = os.path.join(self.root, name + '.bin')
            vox_origin, cam_pose, rle = self._read_rle(bin_path)

#            print("2", time.time() - start_time)
#            img_indices_1_1, voxel_indices_1_1 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size)

            target_1_1 = self._rle2voxel(rle, self.scene_size, bin_path)
#            target_1_2 = self._downsample_label(target_1_1, self.scene_size, 2)
            target_1_4 = self._downsample_label(target_1_1, self.scene_size, 4)
            target_1_8 = self._downsample_label(target_1_1, self.scene_size, 8)
            target_1_16 = self._downsample_label(target_1_1, self.scene_size, 16)

            npz_file_path = os.path.join(self.aic_npz_root, name + "_voxels.npz")
            if os.path.exists(npz_file_path):
                npz_file = np.load(npz_file_path)
                tsdf_1_1 = npz_file['tsdf_hr'] 
                tsdf_1_4 = npz_file['tsdf_lr'] 
            else:
                tsdf_1_1 = None
                tsdf_1_4 = None
#            tsdf_1_4 = self._downsample_tsdf(tsdf_hr, self.downsample)

            data = { 
                "cam_pose": cam_pose,
                "voxel_origin": vox_origin,
                "name": name,
                "tsdf_1_4": tsdf_1_4,
                "tsdf_1_1": tsdf_1_1,
                "target_1_1": target_1_1,
#                "target_1_2": target_1_2,
                "target_1_4": target_1_4,
                "target_1_8": target_1_8,
                "target_1_16": target_1_16,
#                "img_indices_1_1": img_indices_1_1,
#                "voxel_indices_1_1": voxel_indices_1_1,
            }

#            matrix, mask = self.construct_ideal_class_relation_matrix_v2(target_1_16)
#            data['class_relation_mask_1_16'] = mask
#            data['class_relation_matrix_1_16']  = matrix
#            matrix, mask = self.construct_ideal_class_relation_matrix_v2(target_1_8)
#            data['class_relation_mask_1_8'] = mask
#            data['class_relation_matrix_1_8']  = matrix

            CP_matrix_1_16, CP_mask_1_16 = construct_ideal_affinity_matrix(torch.from_numpy(target_1_16), self.n_classes)
            data['CP_matrix_1_16'] = CP_matrix_1_16
            data['CP_mask_1_16'] = CP_mask_1_16
#            CP_matrix_1_8, CP_mask_1_8 = construct_ideal_affinity_matrix(torch.from_numpy(target_1_8), self.n_classes)
#            data['CP_matrix_1_8'] = CP_matrix_1_8
#            data['CP_mask_1_8'] = CP_mask_1_8

            filepath = os.path.join(self.preprocess_dir, name + ".pkl")
            with open(filepath, 'wb') as handle:
                pickle.dump(data, handle)
                print("wrote to", filepath)
        else:
            filepath = os.path.join(self.preprocess_dir, name + ".pkl")
            with open(filepath, 'rb') as handle:
                data = pickle.load(handle)

        cam_pose = data['cam_pose']
        vox_origin = data['voxel_origin']
        img_indices_1_16, voxel_indices_1_16, pts_cam_1_16 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 16 )
        img_indices_1_12, voxel_indices_1_12, pts_cam_1_12 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 12 )
        img_indices_1_8, voxel_indices_1_8, pts_cam_1_8 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 8 )
        img_indices_1_6, voxel_indices_1_6, pts_cam_1_6 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 6 )
        img_indices_1_4, voxel_indices_1_4, pts_cam_1_4 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 4 )
        data["img_indices_1_6"] = img_indices_1_6
        data["voxel_indices_1_6"] = voxel_indices_1_6
        data["pts_cam_1_6"] = pts_cam_1_6
        data["img_indices_1_12"] = img_indices_1_12
        data["voxel_indices_1_12"] = voxel_indices_1_12
        data["pts_cam_1_12"] = pts_cam_1_12
        data["img_indices_1_4"] = img_indices_1_4
        data["voxel_indices_1_4"] = voxel_indices_1_4
        data["pts_cam_1_4"] = pts_cam_1_4
        data["img_indices_1_8"] = img_indices_1_8
        data["voxel_indices_1_8"] = voxel_indices_1_8
        data["pts_cam_1_8"] = pts_cam_1_8
        data["img_indices_1_16"] = img_indices_1_16
        data["voxel_indices_1_16"] = voxel_indices_1_16
        data["pts_cam_1_16"] = pts_cam_1_16
        data['cam_k'] = self.cam_k 

#        class_relation_matrix, mask = self.construct_ideal_class_relation_matrix_v2(data['target_1_16'])
#        print(class_relation_matrix)
#        print(np.min(class_relation_matrix), np.max(class_relation_matrix))
#        data['pairwise_relation_matrix_1_16'] = class_relation_matrix
#        data['pairwise_relation_mask_1_16'] = mask
#        filepath = os.path.join(self.preprocess_dir, data['name'] + ".pkl")
#        with open(filepath, 'wb') as handle:
#            pickle.dump(data, handle)
#            print("wrote to", filepath)

        

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert('RGB') 
        img_indices_1_4 = data['img_indices_1_4'] 
        img_indices_1_6 = data['img_indices_1_6'] 
        img_indices_1_8 = data['img_indices_1_8']
        img_indices_1_12 = data['img_indices_1_12'] 
        img_indices_1_16 = data['img_indices_1_16']

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # resize image
        if self.random_scales:
            scale = 0.9 + np.random.uniform(0.0, 0.2)
        else:
            scale = 1.0
#        print(self.random_scales, scale)
#        scale = random.choice(self.random_scales)
        img_indices_1_16 = img_indices_1_16 * scale
        img_indices_1_12 = img_indices_1_12 * scale
        img_indices_1_8 = img_indices_1_8 * scale
        img_indices_1_6 = img_indices_1_6 * scale
        img_indices_1_4 = img_indices_1_4 * scale
        resize_rgb = transforms.Resize((int(self.img_H * scale), int(self.img_W * scale)))
        img = resize_rgb(img)
#        img = self.resize_rgb(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.
#
        data['fliplr'] = False
#        data['flip3d_x'] = np.random.rand() < self.flip3d_x
#        data['flip3d_z'] = np.random.rand() < self.flip3d_z
#        print(data['flip3d_x'], data['flip3d_z'])
            
        if np.random.rand() < self.fliplr:
            data['fliplr'] = True
            img = np.ascontiguousarray(np.fliplr(img))

            img_indices_1_12[:, 1] = img.shape[1] - 1 - img_indices_1_12[:, 1]
            img_indices_1_4[:, 1] = img.shape[1] - 1 - img_indices_1_4[:, 1]
            img_indices_1_6[:, 1] = img.shape[1] - 1 - img_indices_1_6[:, 1]
            img_indices_1_8[:, 1] = img.shape[1] - 1 - img_indices_1_8[:, 1]
            img_indices_1_16[:, 1] = img.shape[1] - 1 - img_indices_1_16[:, 1]

#            
#        if np.random.rand() < self.flipud:
#            img = np.ascontiguousarray(np.flipud(img))
#            img_indices_1_4[:, 0] = img.shape[0] - 1 - img_indices_1_4[:, 0]

        data['img_indices_1_12'] = img_indices_1_12
        data['img_indices_1_4'] = img_indices_1_4
        data['img_indices_1_6'] = img_indices_1_6
        data['img_indices_1_8'] = img_indices_1_8
        data['img_indices_1_16'] = img_indices_1_16
        
#        data['o_img'] = img
        data['img'] = self.normalize_rgb(img) # (3, 480, 640) 
#        aic_npz_file = np.load(os.path.join(self.aic_npz_root, name + "_voxels.npz"))
#        aic_tsdf_lr = aic_npz_file['tsdf_lr']
#
#        data['tsdf_1_4'] = aic_tsdf_lr
#        data['nonempty'] = self.get_nonempty2(data['tsdf_1_4'], data['target_1_4'], 'TSDF')

        #=============== Load 3D sketch data ================
        tsdf_path = os.path.join(self.tsdf_root, name[3:7] + ".npz")
        sketch_path = os.path.join(self.sketch_root, name[3:7] + ".npy")
        mapping_path = os.path.join(self.sketch_mapping_root, name[3:7] + ".npz")
        sketch_tsdf = np.load(tsdf_path)['arr_0'].reshape(60, 36, 60)
        sketch_mapping = np.load(mapping_path)['arr_0']
#        print(sketch_mapping.shape, np.max(sketch_mapping), np.min(sketch_mapping))
        data['nonempty'] = (np.load(tsdf_path)['arr_1'] > 0).reshape(60, 36, 60)
        sketch = np.load(sketch_path)
        data['sketch'] = sketch
        data['sketch_mapping'] = sketch_mapping
        data['sketch_tsdf'] = sketch_tsdf

#        if data['flip3d_x']:
#            data['target_1_16'] = np.ascontiguousarray(np.flip(data['target_1_16'], 0))
#            data['target_1_4'] = np.ascontiguousarray(np.flip(data['target_1_4'], 0))
#            data['nonempty'] = np.ascontiguousarray(np.flip(data['nonempty'], 0))
#        if data['flip3d_z']:
#            data['target_1_16'] = np.ascontiguousarray(np.flip(data['target_1_16'], 2))
#            data['target_1_4'] = np.ascontiguousarray(np.flip(data['target_1_4'], 2))
#            data['nonempty'] = np.ascontiguousarray(np.flip(data['nonempty'], 2))

#        class_relation_matrix, mask = self.construct_ideal_class_relation_matrix_v2(data['target_1_8'])
#        data['pairwise_relation_matrix_1_16'] = class_relation_matrix
#        data['pairwise_relation_mask_1_16'] = mask

        return data


    @staticmethod
    def compute_class_relation_map(n_classes):
        map = {}
#        count = 0
        L = np.arange(n_classes)
        x, y = np.meshgrid(L, L)
        t = x * n_classes + y
        a = np.concatenate((t[None, :, :], t.T[None, :, :]), axis=0).min(0)

        # merge all relation with empty space to class 0
#        for i in range(n_classes):
#            a[a == i] = 0

        classes  = np.unique(a)
        for i, v in enumerate(classes):
            map[v] = i

        return map 

    def __len__(self):
        return len(self.scan_names)

    @staticmethod
    def construct_ideal_class_relation_matrix_v2(label):
        """
#        same:0
#        diff_non: 1
#        diff_empty: 2
        same_non:0
        same_empty:1
        diff_non: 2
        diff_empty: 3
        """
        label = label.reshape(-1)
        mask = (label != 255)
        label = label[mask]
        N = label.shape[0]
        label_row = np.repeat(label[:, None], N, axis=1)
        label_col = np.repeat(label[None, :], N, axis=0)
        matrix = np.zeros((N, N))
#        matrix[label_row == label_col] = 0
#        matrix[(label_row != label_col) & (label_col !=0)] = 1
#        matrix[(label_row != label_col) & (label_col == 0)] = 2

        matrix[(label_row == label_col) & (label_col != 0)] = 0
        matrix[(label_row == label_col) & (label_col == 0)] = 1
        matrix[(label_row != label_col) & (label_row != 0) & (label_col == 0)] = 2
        matrix[(label_row != label_col) & (label_row != 0) & (label_col != 0)] = 3
        matrix[(label_row != label_col) & (label_row == 0) & (label_col != 0)] = 4
        return matrix, mask

    @staticmethod 
    def construct_ideal_class_relation_matrix(label, n_classes, class_map):
        label = label.reshape(-1)
        mask = (label != 255)
        label = label[mask]
        N = label.shape[0]
        x, y = np.meshgrid(label, label)

        directional_relation_matrix = x * n_classes + y
        indirection_relation_matrix = np.concatenate((directional_relation_matrix[None, :, :], 
                                                      directional_relation_matrix.T[None, :, :]), axis=0).min(0)

        unique_classes = np.unique(indirection_relation_matrix)
        matrix = np.zeros(indirection_relation_matrix.shape)
        for k in class_map:
            matrix[indirection_relation_matrix == k] = class_map[k] 

#        matrix = np.zeros((N, N))
#        for r in range(N):
#            for c in range(N):
#                matrix[r, c] = class_map[label[r] * n_classes + label[c]]
        return matrix, mask

        
    @staticmethod
    def _read_rgb(rgb_filename):  # 0.01s
        r"""Read a RGB image with size H x W
        """
        # rgb = misc.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        rgb = imageio.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        # rgb = np.rollaxis(rgb, 2, 0)  # (H, W, 3)-->(3, H, W)
        return rgb

    @staticmethod
    def _read_depth(depth_filename):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        # depth = misc.imread(depth_filename) / 8000.0  # numpy.float64
        depth = imageio.imread(depth_filename) / 8000.0  # numpy.float64
        # assert depth.shape == (img_h, img_w), 'incorrect default size'
        depth = np.asarray(depth)
        return depth


    def _rle2voxel(cls, rle, voxel_size=(240, 144, 240), rle_filename=''):
        r"""Read voxel label data from file (RLE compression), and convert it to fully occupancy labeled voxels.
        In the data loader of pytorch, only single thread is allowed.
        For multi-threads version and more details, see 'readRLE.py'.
        output: seg_label: 3D numpy array, size 240 x 144 x 240
        """
        # ---- Read RLE
        # vox_origin, cam_pose, rle = cls._read_rle(rle_filename)
        # ---- Uncompress RLE, 0.9s
        seg_label = np.zeros(int(voxel_size[0] * voxel_size[1] * voxel_size[2]), dtype=np.uint8)  # segmentation label
        vox_idx = 0
        for idx in range(int(rle.shape[0] / 2)):
            check_val = rle[idx * 2]
            check_iter = rle[idx * 2 + 1]
            if check_val >= 37 and check_val != 255:  # 37 classes to 12 classes
                print('RLE {} check_val: {}'.format(rle_filename, check_val))
            # seg_label_val = 1 if check_val < 37 else 0  # 37 classes to 2 classes: empty or occupancy
            # seg_label_val = 255 if check_val == 255 else seg_class_map[check_val]
            seg_label_val = seg_class_map[check_val] if check_val != 255 else 255  # 37 classes to 12 classes
            seg_label[vox_idx: vox_idx + check_iter] = np.matlib.repmat(seg_label_val, 1, check_iter)
            vox_idx = vox_idx + check_iter
        seg_label = seg_label.reshape(voxel_size)  # 3D array, size 240 x 144 x 240
        return seg_label

    def fliplr_voxel(self, target, cam_pose, vox_origin, voxel_size):
        scene_size = (int(4.8 / voxel_size), int(2.88/voxel_size), int(4.8/voxel_size)) 
#        print(voxel_size)
        # Create voxel grid in cam coords
        voxel_grid = create_voxel_grid((120, 72, 120))
        pts_world2 = 0.02 + 0.04 * voxel_grid
#        print(pts_world2.min(0), pts_world2.max(0))
        pts_world = np.zeros(pts_world2.shape, dtype=np.float32)
        pts_world[:, 0] = pts_world2[:, 0]
        pts_world[:, 1] = pts_world2[:, 2]
        pts_world[:, 2] = pts_world2[:, 1]

        vox_origin = vox_origin.reshape(1, 3)
        pts_world = pts_world + vox_origin

        pts_world_homo = np.concatenate([pts_world, np.ones([pts_world.shape[0], 1])], axis=1)

        T_world_to_cam = np.linalg.inv(cam_pose)
#        print("cam_pose", cam_pose)
#        print("T_world_to_cam", T_world_to_cam)
        pts_cam_homo = (T_world_to_cam @ pts_world_homo.T).T
#        print(pts_cam_homo.max(0) - pts_cam_homo.min(0))
#        pts_cam = pts_cam_homo[:, :3]
        pts_cam_homo[:, 0] = -1.0 * pts_cam_homo[:, 0]

        pts_world_homo = (cam_pose @ pts_cam_homo.T).T
        pts_world = pts_world_homo[:, :3]

        pts_world = pts_world - vox_origin
        pts_world2[:, 0] = pts_world[:, 0]
        pts_world2[:, 1] = pts_world[:, 2]
        pts_world2[:, 2] = pts_world[:, 1]

        flipped_voxel_grid = (pts_world2 / voxel_size).astype(np.int32)
        flipped_target = np.zeros(target.shape).astype(np.int32)
        voxel_grid = (voxel_grid / 2).astype(np.int32)
#        print(voxel_grid.max(0), flipped_voxel_grid.max(0))
#        print(flipped_target.shape, target.shape)

        mask = (flipped_voxel_grid[:, 0] < scene_size[0]) &  (flipped_voxel_grid[:, 0] >= 0) & \
                (flipped_voxel_grid[:, 1] < scene_size[1]) &  (flipped_voxel_grid[:, 1] >= 0) & \
                (flipped_voxel_grid[:, 2] < scene_size[2]) &  (flipped_voxel_grid[:, 2] >= 0) 

        voxel_grid = voxel_grid[mask]
        flipped_voxel_grid = flipped_voxel_grid[mask]
        flipped_target[flipped_voxel_grid[:, 0],
                       flipped_voxel_grid[:, 1], flipped_voxel_grid[:, 2]] = target[voxel_grid[:, 0], voxel_grid[:, 1], voxel_grid[:, 2]] #        print(torch.sum(flipped_target - target))
        return flipped_target

    def voxel2pixel(self, cam_pose, vox_origin, voxel_size):
        no_mapping_index = 480 * 640
        scene_size = (int(4.8 / voxel_size), int(2.88/voxel_size), int(4.8/voxel_size)) 
        # Create voxel grid in cam coords
        voxel_grid = create_voxel_grid(scene_size)
        pts_world2 = np.ones((1, 3)) * voxel_size / 2 + voxel_size * voxel_grid
        pts_world = np.zeros(pts_world2.shape, dtype=np.float32)
        pts_world[:, 0] = pts_world2[:, 0]
        pts_world[:, 1] = pts_world2[:, 2]
        pts_world[:, 2] = pts_world2[:, 1]

        vox_origin = vox_origin.reshape(1, 3)
        pts_world += vox_origin

        pts_world_homo = np.concatenate([pts_world, np.ones([pts_world.shape[0], 1])], axis=1)

        T_world_to_cam = np.linalg.inv(cam_pose)
        pts_cam_homo = (T_world_to_cam @ pts_world_homo.T).T
        return_pts_cam = pts_cam_homo[:, :3] / np.array([5, 4, 7]).reshape(1, 3)

        # remove points with depth < 0
        negative_depth_idx = (pts_cam_homo[:, 2] <= 0)
#        pts_cam_homo[keep_idx] = no_mapping_index
#        voxel_grid[keep_idx] = no_mapping_index

        pts_cam = pts_cam_homo[:, :3]
        pts_img = (self.cam_k @ pts_cam.T).T

        pts_img = pts_img[:, :2] / np.expand_dims(pts_img[:, 2], axis=1)
        pts_img = np.rint(pts_img).astype(int)


        keep_idx = select_points_in_frustum(pts_img, 0, 0, 640, 480)

        img_indices = pts_img[keep_idx]
#        pts_cam = pts_cam[keep_idx]
#        # fliplr so that indexing is row, col and not col, row
        img_indices = np.fliplr(img_indices)
#
        voxel_indices = voxel_grid[keep_idx]
#
        return np.ascontiguousarray(img_indices), np.ascontiguousarray(voxel_indices), np.ascontiguousarray(return_pts_cam)
    
    @staticmethod
    def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
        r"""downsample the labeled data,
        Shape:
            label, (240, 144, 240)
            label_downscale, if downsample==4, then (60, 36, 60)
        """
        if downscale == 1:
            return label
        ds = downscale
        small_size = (voxel_size[0] // ds, voxel_size[1] // ds, voxel_size[2] // ds)  # small size
        label_downscale = np.zeros(small_size, dtype=np.uint8)
        empty_t = 0.95 * ds * ds * ds  # threshold
        s01 = small_size[0] * small_size[1]
        label_i = np.zeros((ds, ds, ds), dtype=np.int32)

        for i in range(small_size[0]*small_size[1]*small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            # z, y, x = np.unravel_index(i, small_size) 
            # print(x, y, z)

            label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            label_bin = label_i.flatten()  
            # label_bin = label_i.ravel() 

            # zero_count_0 = np.sum(label_bin == 0)
            # zero_count_255 = np.sum(label_bin == 255)
            zero_count_0 = np.array(np.where(label_bin == 0)).size  # 要比sum更快
            zero_count_255 = np.array(np.where(label_bin == 255)).size

            zero_count = zero_count_0 + zero_count_255
            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
                label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        return label_downscale

    def _read_rle(self, rle_filename): # 0.0005s
        """Read RLE compression data
        Return:
            vox_origin,
            cam_pose,
            vox_rle, voxel label data from file
        Shape:
            vox_rle, (240, 144, 240)
        """
        fid = open(rle_filename, 'rb')
        vox_origin = np.fromfile(fid, np.float32, 3).T  # Read voxel origin in world coordinates
        cam_pose = np.fromfile(fid, np.float32, 16).reshape((4, 4))  # Read camera pose
        vox_rle = np.fromfile(fid, np.uint32).reshape((-1, 1)).T  # Read voxel label data from file
        vox_rle = np.squeeze(vox_rle)  # 2d array: (1 x N), to 1d array: (N , )
        fid.close()
        return vox_origin, cam_pose, vox_rle

    @staticmethod
    def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
        r"""downsample the labeled data,
        Shape:
            label, (240, 144, 240)
            label_downscale, if downsample==4, then (60, 36, 60)
        """
        if downscale == 1:
            return label
        ds = downscale
        small_size = (voxel_size[0] // ds, voxel_size[1] // ds, voxel_size[2] // ds)  # small size
        label_downscale = np.zeros(small_size, dtype=np.uint8)
        empty_t = 0.95 * ds * ds * ds  # threshold
        s01 = small_size[0] * small_size[1]
        label_i = np.zeros((ds, ds, ds), dtype=np.int32)

        for i in range(small_size[0]*small_size[1]*small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            # z, y, x = np.unravel_index(i, small_size)
            # print(x, y, z)

            label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            label_bin = label_i.flatten() 
            # label_bin = label_i.ravel()

            # zero_count_0 = np.sum(label_bin == 0)
            # zero_count_255 = np.sum(label_bin == 255)
            zero_count_0 = np.array(np.where(label_bin == 0)).size 
            zero_count_255 = np.array(np.where(label_bin == 255)).size

            zero_count = zero_count_0 + zero_count_255
            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
                label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        return label_downscale

def main():
    NYU_dir = "/gpfswork/rech/xqt/uyl37fq/data/NYU/depthbin"
    write_path = "/gpfsscratch/rech/xqt/uyl37fq/temp"
    preprocess_dir = "/gpfsscratch/rech/xqt/uyl37fq/precompute_data/NYU"
    train_dataset = NYUDataset(split="train",
                               root=NYU_dir,
                               preprocess_dir=preprocess_dir,
                               extract_data=True)
    test_dataset = NYUDataset(split="test",
                              root=NYU_dir,
                              preprocess_dir=preprocess_dir,
                              extract_data=True)
#    t = []
#    for i in tqdm(range(len(train_dataset))):
#        data = train_dataset[i]
#        matrix = data['pairwise_relation_matrix_1_16']
#        t.append(matrix.reshape(-1))
##        break
#    t = np.hstack(t)
#    vals, cnts = np.unique(t, return_counts=True)
#    print(vals, cnts)

    for i in tqdm(range(len(test_dataset))):
        data = test_dataset[i]
#    for i in tqdm(range(len(train_dataset))):
#        data = train_dataset[i]
        

#    out = []
#    for i in tqdm(range(3)):
#        item = {}
#        item['img'] = test_dataset[i]['o_img']
#        item['target_1_4'] = test_dataset[i]['target_1_4']
#        out.append(item)
#
#    write_path = "/gpfsscratch/rech/xqt/uyl37fq/temp/output"
#    filepath = os.path.join(write_path, "data_flip.pkl")
#    filepath = os.path.join(write_path, "data_noflip.pkl")
#    with open(filepath, 'wb') as handle:
#        pickle.dump(out, handle)
#        print("wrote to", filepath)
#    classes = []
#    cnt = 0
##    for i in tqdm(range(len(train_dataset))):
#    for i in tqdm(range(len(test_dataset))):
#        item = train_dataset[i]
#        target_1_4 = item['target_1_4']
#        target_1_4 = target_1_4[target_1_4 != 255]
#        classes.append(target_1_4)
##        cnt += 1
##        if cnt == 4:
##            break
#    classes = np.hstack(classes)
#    vals, counts = np.unique(classes, return_counts=True)
#    print(vals)
#    print(' ,'.join(map(str, counts)))

#
#                                            
#    classes = np.vstack(classes) 
#    classes, counts = np.unique(classes, return_counts=True, axis=0) 
#    print(len(classes))
#    print(' ,'.join(map(str, classes)))
#    print(' ,'.join(map(str, counts)))
#
#                                
#

#        class_relation_matrix_1_16 = train_dataset[i]['class_relation_matrix_1_16']
#        classes.append(class_relation_matrix_1_16.reshape(-1))
#        break
#    classes = np.hstack(classes)
#    classes, counts = np.unique(classes, return_counts=True) 
#    print(classes.shape, counts.shape)
#    print(' ,'.join(map(str, classes)))
#    print(' ,'.join(map(str, counts)))
#        print(class_relation_matrix_1_16.shape, np.min(class_relation_matrix_1_16), np.max(class_relation_matrix_1_16))
#    for i in tqdm(range(len(test_dataset))):
#        test_dataset[i]

if __name__ == '__main__':
    main()
