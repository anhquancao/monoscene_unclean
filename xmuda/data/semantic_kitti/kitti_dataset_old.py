import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import scipy.stats as scipy_stats
import numpy.matlib
from PIL import Image
from torchvision import transforms
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum, vox2pix, compute_local_frustums, compute_CP_mega_matrix, compute_local_frustums_enlarge
from xmuda.models.ssc_loss import construct_ideal_affinity_matrix
import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
import pickle
import imageio
from tqdm import tqdm
from itertools import combinations
import time
import random
import xmuda.common.utils.fusion as fusion


class KittiDataset(Dataset):
    def __init__(self, 
                 split,
                 root, 
                 depth_root,
                 TSDF_root=None,
                 project_scale=2,
                 occ_root=None,
                 frustum_size=4,
                 virtual_img=False,
                 label_root=None,
                 mapping_root=None,
                 sketch_root=None,
                 color_jitter=None,
                 fliplr=0.0,
                 use_predicted_depth=True):
        super().__init__()
        self.n_classes = 20
#        splits = {'train': ["00", "01", "02", "03", "04", "05", '06', '07', '09', '10'], 
#                  'val': ['08'], 
#                  'test': ["21"]}
        splits = {'train': ["00", "01", "02", "03", "04", "05", '06', '07', '09', '10'],
                  'val': ['08'],
                  'test': ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]}
        self.split = split
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        self.virtual_img = virtual_img
        self.root = root
#        self.depth_root = depth_root
#        self.label_root = label_root
#        self.occ_root = occ_root
#        self.sketch_root = sketch_root
#        self.mapping_root = mapping_root
#        self.TSDF_root = TSDF_root
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.scene_size = (51.2, 51.2, 6.4)
#        (int(51.2 / voxel_size), int(51.2/voxel_size), int(6.4/voxel_size)) 
        self.use_predicted_depth = use_predicted_depth

        self.fliplr = fliplr

        self.voxel_size = 0.2 # 0.2m
        self.img_W = 1220
        self.img_H = 370

        self.color_jitter = transforms.ColorJitter(*color_jitter) if color_jitter else None
#        self.resize_rgb = transforms.Resize((int(self.img_H), 
#                                             int(self.img_W)))
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(os.path.join(self.root, 'dataset', 'sequences', sequence, 'calib.txt'))
            P = calib['P2']
            T_velo_2_cam = calib['Tr']
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', '*.bin')
            for voxel_path in glob.glob(glob_path):
                self.scans.append({
                    "sequence": sequence,
                    "P": P,
                    "T_velo_2_cam": T_velo_2_cam,
                    "proj_matrix": proj_matrix,
                    "voxel_path": voxel_path
                })
#        self.voxel_paths = sorted(glob.glob(glob_path))
#        self.scan_names = glob.glob(os.path.join(self.root, '*.bin'))

        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        scan = self.scans[index]
        voxel_path = scan['voxel_path']
        sequence = scan['sequence']
        P = scan['P']
        T_velo_2_cam = scan['T_velo_2_cam']
        proj_matrix = scan['proj_matrix']

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        depth_path = os.path.join(self.depth_root, sequence, frame_id + '.png')
        depth = self._read_depth(depth_path)[:self.img_H, :self.img_W]
        depth = depth.reshape((1,) + depth.shape)

        rgb_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'image_2', frame_id + '.png')
#        target_1_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.label')
#        invalid_1_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.invalid')
#        target_4_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.label_1_4')
#        invalid_4_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.invalid_1_4')
#        target_16_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.label_1_16')
#        invalid_16_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.invalid_1_16')

#        target_1_1 = self.get_label_at_scale(invalid_1_path, target_1_path, 1)
#        target_1_4 = self.get_label_at_scale(invalid_4_path, target_4_path, 4)
#        target_1_16 = self.get_label_at_scale(invalid_16_path, target_16_path, 16)

        data = { 
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix, 
#            "target_1_1": target_1_1,
#            "target_1_4": target_1_4,
#            "target_1_16": target_1_16,
            "depth": depth,
        }
        if self.TSDF_root is not None:
#            label_path = os.path.join(self.root, "dataset", "sequences", sequence, "voxels", frame_id + ".label")
#            LABEL = SemanticKittiIO._read_label_SemKITTI(label_path).reshape(256, 256, 32)
#            inv_map = SemanticKittiIO.get_inv_map()
#            print(np.unique(LABEL))
#            print(inv_map)

            tsdf_1_4_path = os.path.join(self.TSDF_root, sequence, frame_id + "_1_4.npy")
            tsdf_1_1_path = os.path.join(self.TSDF_root, sequence, frame_id + "_1_1.npy")
            tsdf_1_1 = np.load(tsdf_1_1_path)
            tsdf_1_1[tsdf_1_1 < -1] = -1
            tsdf_1_1[tsdf_1_1 > 1] = 1
            tsdf_1_4 = np.load(tsdf_1_4_path)
            tsdf_1_4[tsdf_1_4 < -1] = -1
            tsdf_1_4[tsdf_1_4 > 1] = 1
            data["tsdf_1_1"] = tsdf_1_1
            data["tsdf_1_4"] =  tsdf_1_4
        scales = [self.output_scale, self.project_scale]
        data['scales'] = scales
        cam_k = P[0:3, 0:3]
        vox_origin = np.array([0, -25.6, -2])
        data['T_velo_2_cam'] = T_velo_2_cam
        data['cam_k'] = cam_k
        for scale in scales:
            pix, valid_pix, pix_z = vox2pix(np.linalg.inv(T_velo_2_cam), cam_k, vox_origin, self.voxel_size * scale, self.img_W, self.img_H, self.scene_size)

            data['pix_' + str(scale)] = pix
            data['pix_z_' + str(scale)] = pix_z
            data['valid_pix_' + str(scale)] = valid_pix

        if self.label_root is not None:
            target_1_path = os.path.join(self.label_root, sequence, frame_id + '_1_1.npy')
            target_2_path = os.path.join(self.label_root, sequence, frame_id + '_1_2.npy')
            
            if self.project_scale == 2:
                target = np.load(target_1_path)
                data['target'] = target
                target_8_path = os.path.join(self.label_root, sequence, frame_id + '_1_8.npy')
                target_1_8 = np.load(target_8_path)
                CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            elif self.project_scale == 4:
                target = np.load(target_2_path)
                data['target'] = target
                target_16_path = os.path.join(self.label_root, sequence, frame_id + '_1_16.npy')
                target_1_16 = np.load(target_16_path)
                CP_mega_matrix = compute_CP_mega_matrix(target_1_16)
#            data['nonempty'] = np.ascontiguousarray(np.ones(target_1_1.)) # fake nonempty since this one does not make sense in kitti
            data["CP_mega_matrix"] = CP_mega_matrix

            if self.split != "test":
                # local frustum
                pix_output = data['pix_{}'.format(self.output_scale)]
                pix_z_output = data['pix_z_{}'.format(self.output_scale)]
                if not self.virtual_img:
                    local_frustums, list_cnts = compute_local_frustums(pix_output, pix_z_output, target,
                                                                       self.img_W, self.img_H,
                                                                       dataset="kitti", n_classes=20, size=self.frustum_size)
                else:
                    local_frustums, list_cnts = compute_local_frustums_enlarge(pix_output, pix_z_output, target, self.img_W, self.img_H,
                                                                       dataset="kitti", n_classes=20)
            else:
                local_frustums = None
                list_cnts = None
#            data['local_frustums'] = np.array(local_frustums)
#            data['local_frustums_cnt'] = np.array(list_cnts)
            data['local_frustums'] = local_frustums
            data['local_frustums_cnt'] = list_cnts

        if self.occ_root is not None:
            occ_path_1_1 = os.path.join(self.occ_root, sequence, frame_id + "_1_1.npy")
            data['occ_1_1'] = np.load(occ_path_1_1)

        if self.mapping_root is not None:
            mapping_1_4_path = os.path.join(self.mapping_root, sequence, frame_id + "_1_4.npy")
            mapping_1_4 = np.load(mapping_1_4_path)
            mapping_1_1_path = os.path.join(self.mapping_root, sequence, frame_id + "_1_1.npy")
            mapping_1_1 = np.load(mapping_1_1_path)
            data["mapping_1_4"] = mapping_1_4
            data["mapping_1_1"] = mapping_1_1

        if self.sketch_root is not None:
            sketch_path_1_1 = os.path.join(self.sketch_root, sequence, frame_id + "_1_1.npy")
            sketch_path_1_4 = os.path.join(self.sketch_root, sequence, frame_id + ".npy")
            data['sketch_1_1'] = np.load(sketch_path_1_1)
            data['sketch_1_4'] = np.load(sketch_path_1_4)


        img = Image.open(rgb_path).convert('RGB') 

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.
        img = img[:370, :1220, :]

        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scales:
                key = 'pix_' + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:,0] 

        data['img'] = self.normalize_rgb(img) # (3, 480, 640) 
        return data

    def __len__(self):
        return len(self.scans)

    def get_label_at_scale(self, invalid_path, label_path, scale):

#        scale_divide = int(scale[-1])
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
        LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
        remap_lut = SemanticKittiIO.get_remap_lut(os.path.join("/gpfswork/rech/kvd/uyl37fq/code/xmuda-extend/xmuda/data/semantic_kitti", 'semantic-kitti.yaml'))

        if scale == 1:
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
            # unique, counts = np.unique(LABEL, return_counts=True)

        # Setting to unknown all voxels marked on invalid mask...
        LABEL[np.isclose(INVALID, 1)] = 255
        LABEL = LABEL.reshape([int(self.scene_size[0] / (scale * self.voxel_size)), 
                               int(self.scene_size[1] / (scale * self.voxel_size)), 
                               int(self.scene_size[2] / (scale * self.voxel_size))])
        # LABEL = LABEL.reshape(self.VOXEL_DIMS)
        return LABEL

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
        depth = imageio.imread(depth_filename) / 256.0  # numpy.float64
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

    def voxel2pixelv2(self, T_velo_2_cam, cam_k, voxel_size):
#        print(voxel_size)
        vol_bnds = np.zeros((3,2))
        vox_origin = np.array([0, -25.6, -2])
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array([self.scene_size[0], self.scene_size[1], self.scene_size[2]])
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
#        print(vol_dim)
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

#        dev = self.trunc_norm.rvs((vox_coords.shape[0], 3))
        cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = fusion.rigid_transform(cam_pts, T_velo_2_cam)

        pix_z = cam_pts[:, 2]
        pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < self.img_W,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < self.img_H,
                    pix_z > 0))))
        
#        print("pts_cam before", cam_pts[valid_pix].min(0), cam_pts[valid_pix].max(0))
#        cam_pts = cam_pts - np.array([25.6, 0, 1.2]).reshape(1, 3)
#        cam_pts = cam_pts / np.array([25.6, 25.6, 3.2]).reshape(1, 3)
#        print("pts_cam", cam_pts[valid_pix].min(0), cam_pts[valid_pix].max(0))
        cam_pts = cam_pts.reshape((vol_dim[0], vol_dim[1], vol_dim[2], 3))
        cam_pts = cam_pts.reshape(-1, 3)
#        cam_pts = np.moveaxis(cam_pts, [0, 1, 2], [0, 2, 1]).reshape(-1, 3)
#        print(cam_pts.shape)
        return pix, valid_pix, cam_pts

    
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
    kitti_dir = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
    kitti_depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
    label_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/labels/kitti"
    kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
    val_dataset = KittiDataset(split="val", 
                                depth_root=kitti_depth_root,
                                TSDF_root=kitti_tsdf_root,
                                label_root=label_root,
                                root=kitti_dir)
#    train_dataset = KittiDataset(split="train",
#                                 frustum_size=8,
#                                depth_root=kitti_depth_root,
#                                TSDF_root=kitti_tsdf_root,
#                                label_root=label_root,
#                                root=kitti_dir)
    cnts = []
    write_dir = '/gpfsscratch/rech/kvd/uyl37fq/temp/kitti_mask'
    old = None
    for i in tqdm(range(len(val_dataset))):
        item = val_dataset[i]
        valid_pix_1 = item['valid_pix_1']
        frame_id = item['frame_id']
        np.save(os.path.join(write_dir, "valid_pix_seq8.npy"), valid_pix_1)
        break
#        if old is not None:
#            print(np.sum(valid_pix_1 - old))
#        old = valid_pix_1


if __name__ == '__main__':
    main()
