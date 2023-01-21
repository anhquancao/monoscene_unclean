import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import scipy.stats as scipy_stats
import numpy.matlib
from PIL import Image
from torchvision import transforms
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum, compute_local_frustums, vox2pix, compute_CP_mega_matrix, compute_mega_context
from xmuda.models.ssc_loss import construct_ideal_affinity_matrix
import pickle
import imageio
from tqdm import tqdm
from itertools import combinations
import time
import random
import xmuda.common.utils.fusion as fusion
import torch.nn.functional as F
from xmuda.data.NYU.params import NYU_class_cluster_4, NYU_class_cluster_6


seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 
                 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11] 

class NYUDataset(Dataset):
    def __init__(self, 
                 split,
                 root,
                 preprocess_dir,
                 corenet_proj=None,
                 n_relations=4,
                 color_jitter=None,
                 frustum_size=4,
                 fliplr=0.0,
                 flipud=0.0,
                 flip3d_x=0.0,
                 flip3d_z=0.0,
                 use_predicted_depth=False,
                 random_scales=False,
                 extract_data=False):
        print("nyu_dataset_use_predicted_depth", use_predicted_depth) 
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.n_classes = 12
        self.extract_data = extract_data
        self.root = os.path.join(root, "NYU" + split)
        self.corenet_proj = corenet_proj
        self.sketch_root = os.path.join(root, "sketch3D")
        self.sketch_original_mapping_root = os.path.join(root, "Mapping")
        self.origin_tsdf_root = os.path.join(root, "TSDF")
        self.use_predicted_depth = use_predicted_depth
        self.generated_data_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/NYU"
        self.base_dir = os.path.join(self.generated_data_root, "base", "NYU" + split)

        if self.use_predicted_depth:
            self.tsdf_root = os.path.join(self.generated_data_root, "tsdf_depth_pred", "NYU" + split)
            self.sketch_mapping_root = os.path.join(self.generated_data_root, "mapping_depth_pred", "NYU" + split)
            self.occ_root = os.path.join(self.generated_data_root, "occupancy_depth_pred", "NYU" + split)
            self.depth_root = os.path.join(self.generated_data_root, "depth_pred", "NYU" + split)
        else:
            self.tsdf_root = os.path.join(self.generated_data_root, "tsdf_depth_gt", "NYU" + split)
            self.sketch_mapping_root = os.path.join(self.generated_data_root, "mapping_depth_gt", "NYU" + split)
            self.occ_root = os.path.join(self.generated_data_root, "occupancy_depth_gt", "NYU" + split)
            self.depth_root = self.root
#        self.pred_tsdf_root = os.path.join(self.sketch_pred_root, "TSDF_pred_depth", "NYU" + split)
#        self.pred_sketch_mapping_root = os.path.join(self.sketch_pred_root, "sketch_mapping_pred_depth", "NYU" + split)

        self.preprocess_dir = os.path.join(preprocess_dir, split)
        self.aic_npz_root = os.path.join("/gpfsscratch/rech/kvd/uyl37fq/AIC_dataset/", "NYU" + split + "_npz")
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

        self.scan_names = glob.glob(os.path.join(self.root, '*.bin'))
        s = []
        for f in self.scan_names:
            s.append(f)
        self.scan_names = s
        self.downsample = 4

        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        filepath = os.path.join(self.base_dir, name + ".pkl")
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)

        cam_pose = data['cam_pose']
        vox_origin = data['voxel_origin']
        data['cam_k'] = self.cam_k
        target_1_4 = data['target_1_4']
        target_1_16 = data['target_1_16']
        
        if self.n_relations == 8:
            class_cluster = NYU_class_cluster_4
        elif self.n_relations == 16:
            class_cluster = NYU_class_cluster_6
        else:
            class_cluster = None

        CP_mega_matrix = compute_CP_mega_matrix(target_1_16, self.n_relations, class_cluster)
        data["CP_mega_matrix"] = CP_mega_matrix

        if self.corenet_proj is None:
            scales = [4]
        else:
            scales = [4, 8, 16]
        for scale in scales:
            scene_size = (4.8, 4.8, 2.88)
            pix, valid_pix, pix_z = vox2pix(cam_pose, self.cam_k, vox_origin, self.voxel_size * scale, self.img_W, self.img_H, scene_size)
            data['pix_' + str(scale)] = pix
            data['valid_pix_' + str(scale)] = valid_pix
            data['pix_z_' + str(scale)] = pix_z


            if scale == 4:
                local_frustums, list_cnts = compute_local_frustums(pix, pix_z, target_1_4,
                                                                   self.img_W, self.img_H,
                                                                   dataset="NYU", n_classes=12, size=self.frustum_size)
                data['local_frustums_' + str(scale)] = np.array(local_frustums)
                data['local_frustums_cnt_' + str(scale)] = np.array(list_cnts)


        if self.use_predicted_depth:
            depth_path = os.path.join(self.depth_root, name + "_color.png")  
        else:
            depth_path = os.path.join(self.depth_root, name + ".png")  
        depth = self._read_depth(depth_path)  #
        depth_tensor = depth.reshape((1,) + depth.shape)
        data['depth'] = depth_tensor

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert('RGB') 

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

#        img = resize_rgb(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.
            
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scales:
                key = 'pix_' + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:,0] 

        data['img'] = self.normalize_rgb(img) # (3, 480, 640) 
        data['class_proportion_1_4'] = self.compute_class_proportion(data['target_1_4'], self.n_classes)

        #=============== Load 3D sketch data ================
        sketch_path = os.path.join(self.sketch_root, name[3:7] + ".npy")
        sketch_original_mapping_path = os.path.join(self.sketch_original_mapping_root, name[3:7] + ".npz")
        data['sketch_original_mapping'] = np.load(sketch_original_mapping_path)['arr_0'].astype(np.int64).reshape(60, 36, 60)
        occ_path_1_4 = os.path.join(self.occ_root, name + "_1_4.npy")
        occ_path_1_1 = os.path.join(self.occ_root, name + "_1_1.npy")
        data['occ_1_1'] = np.load(occ_path_1_1)
        data['occ_1_4'] = np.load(occ_path_1_4)
        tsdf_path_1_4 = os.path.join(self.tsdf_root, name + "_1_4.npy")
        tsdf_path_1_1 = os.path.join(self.tsdf_root, name + "_1_1.npy")
        mapping_path_1_1 = os.path.join(self.sketch_mapping_root, name + "_1_1.npy")
        mapping_path_1_4 = os.path.join(self.sketch_mapping_root, name + "_1_4.npy")
        if os.path.exists(tsdf_path_1_1):
            tsdf_1_1 = np.load(tsdf_path_1_1)
            data['tsdf_1_1'] = tsdf_1_1
        if os.path.exists(tsdf_path_1_4):
            tsdf_1_4 = np.load(tsdf_path_1_4)
            data['tsdf_1_4'] = tsdf_1_4
        
        mapping_1_1 = np.load(mapping_path_1_1)
        data['mapping_1_1'] = mapping_1_1
        mapping_1_4 = np.load(mapping_path_1_4)
        data['mapping_1_4'] = mapping_1_4
        nonempty_path = os.path.join(self.origin_tsdf_root, name[3:7] + ".npz")
        data['nonempty'] = (np.load(nonempty_path)['arr_1'] > 0).reshape(60, 36, 60)
        sketch = np.load(sketch_path)
        data['sketch'] = sketch


        return data

    @staticmethod
    def compute_class_proportion(target, n_classes):
        class_proportion = np.zeros(n_classes)
        labels, cnts = np.unique(target[target != 255], return_counts=True)
        class_proportion[labels] = cnts
        class_proportion = class_proportion / np.sum(class_proportion)
#        print(class_proportion, class_proportion.shape)
        return class_proportion

    def __len__(self):
        return len(self.scan_names)




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

    @staticmethod
    def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
        valid_pix = np.logical_and(pix_x >= min_x,
                    np.logical_and(pix_x < max_x,
                    np.logical_and(pix_y >= min_y,
                    np.logical_and(pix_y < max_y,
                    pix_z > 0))))
        return valid_pix

    def voxel2bb(self, cam_pose, vox_origin, voxel_size):
        scene_size = (int(4.8 / voxel_size), int(4.8/voxel_size), int(2.88/voxel_size)) 
        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array([scene_size[0] * voxel_size, scene_size[1] * voxel_size , scene_size[2] * voxel_size])
#        offsets_list = [
#            (0.8, 0.8, 0.8),
#            (0.8, 0.8, 0.2),
#            (0.8, 0.2, 0.8),
#            (0.8, 0.2, 0.2),
#            (0.2, 0.8, 0.8),
#            (0.2, 0.8, 0.2),
#            (0.2, 0.2, 0.8),
#            (0.2, 0.2, 0.2),
#        ]
        offsets_list = [
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        ]

        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(
                range(vol_dim[0]),
                range(vol_dim[1]),
                range(vol_dim[2]),
                indexing='ij'
              )
        vox_coords = np.concatenate([
                xv.reshape(1,-1),
                yv.reshape(1,-1),
                zv.reshape(1,-1)
              ], axis=0).astype(int).T

        pix_xs = []
        pix_ys = []
        for offsets in offsets_list:
            cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size, offsets)
            cam_pts = fusion.rigid_transform(cam_pts, np.linalg.inv(cam_pose))

            pix_z = cam_pts[:, 2]
            pix = fusion.TSDFVolume.cam2pix(cam_pts, self.cam_k)
            pix_x, pix_y = pix[:, 0], pix[:, 1]
#            pix_x[pix_x < 0] = -1 
#            pix_y[pix_y < 0] = -1 
#            pix_x[pix_x > self.img_W] = self.img_W
#            pix_y[pix_y > self.img_H] =  self.img_H

            pix_xs.append(pix_x[np.newaxis, ...])
            pix_ys.append(pix_y[np.newaxis, ...])
        pix_xs = np.concatenate(pix_xs, axis=0)
        pix_ys = np.concatenate(pix_ys, axis=0)

        # Top-left corner
        min_x = pix_xs.min(axis=0)
        min_y = pix_ys.min(axis=0)

        max_x = pix_xs.max(axis=0)
        max_y = pix_ys.max(axis=0)

        bb_width = max_x - min_x
        bb_height = max_y - min_y

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        max_box_size = 32
        min_x[bb_width > max_box_size] = center_x[bb_width > max_box_size] - (max_box_size / 2)
        max_x[bb_width > max_box_size] = center_x[bb_width > max_box_size] + (max_box_size / 2)
        min_y[bb_height > max_box_size] = center_y[bb_height > max_box_size] - (max_box_size / 2)
        max_y[bb_height > max_box_size] = center_y[bb_height > max_box_size] + (max_box_size / 2)


        valid_bbs = ~(
            ((min_x < 0) & (max_x < 0)) | \
            ((min_y < 0) & (max_y < 0)) | \
            ((min_x > self.img_W) & (max_x > self.img_W)) | \
            ((min_y > self.img_H) & (max_y > self.img_H))
        )
        min_x[min_x < 0] = 0
        min_y[min_y < 0] = 0
        max_x[max_x < 0] = 0
        max_y[max_y < 0] = 0

        min_x[min_x >= self.img_W] = self.img_W - 1
        min_y[min_y >= self.img_H] = self.img_H - 1 
        max_x[max_x >= self.img_W] = self.img_W - 1
        max_y[max_y >= self.img_H] = self.img_H - 1 
#        print(np.sum(min_x[valid_bbs] < 0))
#        print(np.sum(min_y[valid_bbs] < 0))
#        print(np.sum(max_x[valid_bbs] >= self.img_W))
#        print(np.sum(max_y[valid_bbs] >= self.img_H))
        min_x = min_x[..., np.newaxis]
        min_y = min_y[..., np.newaxis]
        max_x = max_x[..., np.newaxis]
        max_y = max_y[..., np.newaxis]

#        print("x", np.sum((max_x[valid_bbs] - min_x[valid_bbs]) < 1))
#        print("y", np.sum((max_y[valid_bbs] - min_y[valid_bbs]) < 1))
        bbs = np.concatenate([min_x, min_y, max_x, max_y], axis=1) 
#        print(ret[:100])
        return bbs, valid_bbs


    
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
    NYU_dir = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
    write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp"
    preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
    train_dataset = NYUDataset(split="train",
                               root=NYU_dir,
                               preprocess_dir=preprocess_dir,
                               extract_data=False)
    test_dataset = NYUDataset(split="test",
                              root=NYU_dir,
                              preprocess_dir=preprocess_dir,
                              extract_data=False)
    cnts = np.zeros(12)
    total_ratio = 0
    for i in tqdm(range(len(test_dataset))):
        tsdf_1_1 = test_dataset[i]['tsdf_1_4']
        target_1_4 = test_dataset[i]['target_1_4']
        tsdf_1_1[target_1_4 == 255] = 0
        target_1_4[target_1_4 == 255] = 0
        target_1_4[target_1_4 > 0] = 1
        occ = target_1_4[abs(tsdf_1_1) < 0.08]
        ratio = np.sum(occ) / np.sum(target_1_4)
        print(ratio)
        total_ratio += ratio
    print("total=", total_ratio / len(test_dataset))
#        bbs = train_dataset[i]['bbs_4']
        
        # target_1_4 = target_1_4[target_1_4 != 255]
#        img = train_dataset[i]['img']
#        np.save(os.path.join(write_path, "img.npy"), img)
#        np.save(os.path.join(write_path, "bbs_4.npy"), bbs)
#        np.save(os.path.join(write_path, "target_1_4.npy"), target_1_4)
#        break

#        return
#        cnts += matrix.reshape(4, -1).sum(1)
#        total_cnts += matrix.reshape(4, -1).shape[1]
        # classes, cnt = np.unique(target_1_4, return_counts=True)
        # cnts[classes.astype(int)] += cnt
##    print(total_cnts / np.sum(total_cnts))
    print(cnts)
#    print(cnts[1:] * 100 / np.sum(cnts[1:]))

if __name__ == '__main__':
    main()
