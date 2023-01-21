import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import numpy.matlib
from PIL import Image
from torchvision import transforms
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum
import pickle
import imageio
from tqdm import tqdm


seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 
                 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11] 

class NYUDatasetAIC(Dataset):
    def __init__(self, 
                 split,
                 root, 
                 pred_depth_dir,
                 preprocess_dir, 
                 color_jitter=None,
                 fliplr=0.0,
                 flipud=0.0,
                 extract_data=True):
        self.extract_data = extract_data
        self.root = os.path.join(root, "NYU" + split)
        self.pred_depth_dir = os.path.join(pred_depth_dir, "NYU" + split)
        self.preprocess_dir = os.path.join(preprocess_dir, split)
        self.aic_npz_root = os.path.join("/gpfsscratch/rech/xqt/uyl37fq/AIC_dataset/", "NYU" + split + "_npz")
        self.fliplr = fliplr
        self.flipud = flipud
        self.voxel_size = 0.02 # 0.02m
        self.scene_size = (240, 144, 240)
        self.color_jitter = transforms.ColorJitter(*color_jitter) if color_jitter else None
        self.img_W = 640
        self.img_H = 480
        self.cam_k = np.array([
            [518.8579, 0, 320],
            [0, 518.8579, 240],
            [0, 0, 1]
        ])
        self.scan_names = glob.glob(os.path.join(self.root, '*.bin'))
        s = []
        for f in self.scan_names:
            if 'NYU0223' not in f:
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

        if self.extract_data:
            rgb_path = os.path.join(self.root, name + "_color.jpg")
#            img = Image.open(rgb_path).convert('RGB') 
#            img = self._read_rgb(rgb_path)

            bin_path = os.path.join(self.root, name + '.bin')
            vox_origin, cam_pose, rle = self._read_rle(bin_path)

            img_indices_1_8, voxel_indices_1_8 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 8 )
            img_indices_1_4, voxel_indices_1_4 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 4 )
            img_indices_1_2, voxel_indices_1_2 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size * 2 )
            img_indices_1_1, voxel_indices_1_1 = self.voxel2pixel(cam_pose, vox_origin, self.voxel_size)

            target_1_1 = self._rle2voxel(rle, self.scene_size, bin_path)
            target_1_4 = self._downsample_label(target_1_1, self.scene_size, self.downsample)

            depth_file = os.path.join(self.root, name + '.png')
            depth = self._read_depth(depth_file)

            pred_depth_file = os.path.join(self.pred_depth_dir, name + '.png')
            pred_depth = self._read_depth(pred_depth_file)

            npz_file_path = os.path.join(self.aic_npz_root, name + "_voxels.npz")
            if os.path.exists(npz_file_path):
                npz_file = np.load(npz_file_path)
                tsdf_1_1 = npz_file['tsdf_hr'] 
                tsdf_1_4 = npz_file['tsdf_lr'] 
            else:
                tsdf_1_1 = None
                tsdf_1_4 = None
#            tsdf_1_4 = self._downsample_tsdf(tsdf_hr, self.downsample)
    
            binary_vox, _, position, position4 = self._depth2voxel(depth, cam_pose, vox_origin)
            binary_vox, _, pred_depth_position, pred_depth_position4 = self._depth2voxel(pred_depth, cam_pose, vox_origin)


            data = { 
                "name": name,
                "tsdf_1_4": tsdf_1_4,
                "tsdf_1_1": tsdf_1_1,
                "depth": depth,
                "pred_depth": pred_depth,
                "position": position,
                "pred_depth_position": pred_depth_position,
                "target_1_4": target_1_4,
                "target_1_1": target_1_1,
#                "img": img,
                "img_indices_1_1": img_indices_1_1,
                "voxel_indices_1_1": voxel_indices_1_1,
                "img_indices_1_2": img_indices_1_2,
                "voxel_indices_1_2": voxel_indices_1_2,
                "img_indices_1_4": img_indices_1_4,
                "voxel_indices_1_4": voxel_indices_1_4,
                "img_indices_1_8": img_indices_1_8,
                "voxel_indices_1_8": voxel_indices_1_8
            }


            filepath = os.path.join(self.preprocess_dir, name + ".pkl")
            with open(filepath, 'wb') as handle:
                pickle.dump(data, handle)
                print("wrote to", filepath)
        else:
            filepath = os.path.join(self.preprocess_dir, name + ".pkl")
            with open(filepath, 'rb') as handle:
                data = pickle.load(handle)

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert('RGB') 
        img_indices_1_4 = data['img_indices_1_4']

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.
#
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            img_indices_1_4[:, 1] = img.shape[1] - 1 - img_indices_1_4[:, 1]
#            img_indices_1_1[:, 1] = img.shape[1] - 1 - img_indices_1_1[:, 1]

#            
#        if np.random.rand() < self.flipud:
#            img = np.ascontiguousarray(np.flipud(img))
#            img_indices_1_4[:, 0] = img.shape[0] - 1 - img_indices_1_4[:, 0]


        data['img'] = self.normalize_rgb(img) # (3, 480, 640) 
        aic_npz_file = np.load(os.path.join(self.aic_npz_root, name + "_voxels.npz"))
#        aic_target_lr = aic_npz_file['target_lr']
#        aic_rgb = aic_npz_file['rgb']
#        aic_position = aic_npz_file['position']
#        data['target_1_4'] = aic_target_lr
        aic_tsdf_lr = aic_npz_file['tsdf_lr']
#        data['nonempty'] = self.get_nonempty2(aic_tsdf_lr, data['target_1_4'], 'TSDF')
        data['nonempty'] = self.get_nonempty(aic_tsdf_lr, 'TSDF')
#        data['img'] = torch.from_numpy(aic_rgb)
#        data['depth'] = aic_depth
#        data['target_1_4'] = aic_target_lr.T
#        data['position'] = aic_position

#
#        print(aic_rgb.shape, data['img'].shape)
#        print("aic", aic_rgb[:, :3, :5])
#        print("img", data['img'].numpy()[:, :3, :5])
#        print("rgb", np.sum(aic_rgb - data['img'].numpy()))
#        print("depth", np.sum(aic_depth - data['depth']))
#        print("target_lr", np.sum(aic_target_lr - data['target_1_4']))
#        print("target_lr", np.sum(aic_position - data['position']))
#        print("rgb", np.sum(aic_rgb - data['img']))

        data['target_1_4'] = data['target_1_4']#.T
        data['nonempty'] = data['nonempty']#.T
        data['tsdf_1_4'] = aic_tsdf_lr#.T

#        data['img'] = torch.from_numpy(aic_rgb)
        return data


    def __len__(self):
        return len(self.scan_names)

    @staticmethod
    def get_nonempty(voxels, encoding):  # Get none empty from depth voxels
        data = np.zeros(voxels.shape, dtype=np.float32)  # init 0 for empty
        # if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
        #     data[voxels == 1] = 1
        #     return data
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels != 0] = 1
            surface = np.array(np.where(voxels == 1))  # surface=1
        elif encoding == 'TSDF':
            data[np.where(np.logical_or(voxels <= 0, voxels == 1))] = 1
            surface = np.array(np.where(voxels == 1))  # surface
            # surface = np.array(np.where(np.logical_and(voxels > 0, voxels != np.float32(0.001))))  # surface
        else:
            raise Exception("Encoding error: {} is not validate".format(encoding))

        min_idx = np.amin(surface, axis=1)
        max_idx = np.amax(surface, axis=1)
        # print('min_idx, max_idx', min_idx, max_idx)
        # data[:a], data[a]不包含在内, data[b:], data[b]包含在内
        # min_idx = min_idx
        max_idx = max_idx + 1
        # 本该扩大一圈就够了，但由于GT标注的不是很精确，故在高分辨率情况下，多加大一圈
        # min_idx = min_idx - 1
        # max_idx = max_idx + 2
        min_idx[min_idx < 0] = 0
        max_idx[0] = min(voxels.shape[0], max_idx[0])
        max_idx[1] = min(voxels.shape[1], max_idx[1])
        max_idx[2] = min(voxels.shape[2], max_idx[2])
        data[:min_idx[0], :, :] = 0  # data[:a], data[a]不包含在内
        data[:, :min_idx[1], :] = 0
        data[:, :, :min_idx[2]] = 0
        data[max_idx[0]:, :, :] = 0  # data[b:], data[b]包含在内
        data[:, max_idx[1]:, :] = 0
        data[:, :, max_idx[2]:] = 0
        return data

    @staticmethod
    def get_nonempty2(voxels, target, encoding):  # Get none empty from depth voxels
        data = np.ones(voxels.shape, dtype=np.float32)  # init 1 for none empty
        data[target == 255] = 0
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels == 0] = 0
        elif encoding == 'TSDF':
            # --0
            # data[voxels == np.float32(0.001)] = 0
            # --1
            # data[voxels > 0] = 0
            # --2
            # data[voxels >= np.float32(0.001)] = 0
            # --3
            data[voxels >= np.float32(0.001)] = 0
            data[voxels == 1] = 1

        return data
        
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
    def _downsample_tsdf(tsdf, downscale=4):  # 仅在Get None empty　时会用到
        r"""
        Shape:
            tsdf, (240, 144, 240)
            tsdf_downscale, (60, 36, 60), (stsdf.shape[0]/4, stsdf.shape[1]/4, stsdf.shape[2]/4)
        """
        if downscale == 1:
            return tsdf
        # TSDF_EMPTY = np.float32(0.001)
        # TSDF_SURFACE: 1, sign >= 0
        # TSDF_OCCLUD: sign < 0  np.float32(-0.001)
        ds = downscale
        small_size = (int(tsdf.shape[0] / ds), int(tsdf.shape[1] / ds), int(tsdf.shape[2] / ds))
        tsdf_downscale = np.ones(small_size, dtype=np.float32) * np.float32(0.001)  # init 0.001 for empty
        s01 = small_size[0] * small_size[1]
        tsdf_sr = np.ones((ds, ds, ds), dtype=np.float32)  # search region
        for i in range(small_size[0] * small_size[1] * small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            tsdf_sr[:, :, :] = tsdf[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            tsdf_bin = tsdf_sr.flatten()
            # none_empty_count = np.array(np.where(tsdf_bin != TSDF_EMPTY)).size
            none_empty_count = np.array(np.where(np.logical_or(tsdf_bin <= 0, tsdf_bin == 1))).size
            if none_empty_count > 0:
                # surface_count  = np.array(np.where(stsdf_bin == 1)).size
                # occluded_count = np.array(np.where(stsdf_bin == -2)).size
                # surface_count = np.array(np.where(tsdf_bin > 0)).size  # 这个存在问题
                surface_count  = np.array(np.where(tsdf_bin == 1)).size
                # occluded_count = np.array(np.where(tsdf_bin < 0)).size
                # tsdf_downscale[x, y, z] = 0 if surface_count > occluded_count else np.float32(-0.001)
                tsdf_downscale[x, y, z] = 1 if surface_count > 2 else np.float32(-0.001)  # 1 or 0 ?
            # else:
            #     tsdf_downscale[x, y, z] = empty  # TODO 不应该将所有值均设为0.001
        return tsdf_downscale

    def depth2voxel(self, depth, cam_pose, vox_origin):
        cam_k = self.cam_k
        unit = self.voxel_size  # 0.02

        # ---- Get point in camera coordinate
        H, W = depth.shape
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter

        return pt_cam

    def _depth2voxel(cls, depth, cam_pose, vox_origin):
        cam_k = cls.cam_k
        voxel_size = (240, 144, 240)
        unit = 0.02
        # ---- Get point in camera coordinate
        H, W = depth.shape
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter
#        print("pt_cam", pt_cam.reshape(-1, 3).min(0), pt_cam.reshape(-1, 3).max(0))
        # ---- Get point in world coordinate
        p = cam_pose
        pt_world = np.zeros((H, W, 3), dtype=np.float32)
        pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
        pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
        pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
        pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
        pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
        pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
#        print("pt_world", pt_world.reshape(-1, 3).min(0), pt_world.reshape(-1, 3).max(0))
        # ---- Aline the coordinates with labeled data (RLE .bin file)
        pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
        # pt_world2 = pt_world
        pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
        pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
        pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

        # pt_world2[:, :, 0] = pt_world[:, :, 1]  # x 原始paper方法
        # pt_world2[:, :, 1] = pt_world[:, :, 2]  # y
        # pt_world2[:, :, 2] = pt_world[:, :, 0]  # z

        # ---- World coordinate to grid/voxel coordinate
        point_grid = pt_world2 / unit  # Get point in grid coordinate, each grid is a voxel
        point_grid = np.rint(point_grid).astype(np.int32)  # .reshape((-1, 3))  # (H*W, 3) (H, W, 3)
#        print("point_grid", point_grid.reshape(-1, 3).min(0), point_grid.reshape(-1, 3).max(0))

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        # voxel_binary = np.zeros(voxel_size, dtype=np.uint8)     # (W, H, D)
        voxel_binary = np.zeros([_ + 1 for _ in voxel_size], dtype=np.float32)  # (W, H, D)
        voxel_xyz = np.zeros(voxel_size + (3,), dtype=np.float32)  # (W, H, D, 3)
        position = np.zeros((H, W), dtype=np.int32)
        position4 = np.zeros((H, W), dtype=np.int32)
        # position44 = np.zeros((H/4, W/4), dtype=np.int32)

        voxel_size_lr = (voxel_size[0] // 4, voxel_size[1] // 4, voxel_size[2] // 4)
        for h in range(H):
            for w in range(W):
                i_x, i_y, i_z = point_grid[h, w, :]
                if 0 <= i_x < voxel_size[0] and 0 <= i_y < voxel_size[1] and 0 <= i_z < voxel_size[2]:
                    voxel_binary[i_x, i_y, i_z] = 1  # the bin has at least one point (bin is not empty)
                    voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
                    # position[h, w, :] = point_grid[h, w, :]  # 记录图片上的每个像素对应的voxel位置
                    # 记录图片上的每个像素对应的voxel位置
                    position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_size)
                    # TODO 这个project的方式可以改进
                    position4[h, ] = np.ravel_multi_index((point_grid[h, w, :] / 4).astype(np.int32), voxel_size_lr)
                    # position44[h / 4, w / 4] = np.ravel_multi_index(point_grid[h, w, :] / 4, voxel_size_lr)

        # output --- 3D Tensor, 240 x 144 x 240
        del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid  # Release Memory
        return voxel_binary, voxel_xyz, position, position4  # (W, H, D), (W, H, D, 3)


    def voxel2pixel(self, cam_pose, vox_origin, voxel_size):
        scene_size = (int(4.8 / voxel_size), int(2.88/voxel_size), int(4.8/voxel_size)) 
#        print(voxel_size)
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

#        R_cam = np.identity(4)
#        R_cam[:3, :3] = cam_pose[:3, :3]
#        t_cam = np.identity(4)
#        t_cam[:3, 3] = - cam_pose[:3, 3]
#        T_world_to_cam = R_cam @ t_cam
        T_world_to_cam = np.linalg.inv(cam_pose)
        pts_cam_homo = (T_world_to_cam @ pts_world_homo.T).T
#        print("pts_cam", pts_cam_homo.min(0), pts_cam_homo.max(0))

        # remove points with depth < 0
        keep_idx = pts_cam_homo[:, 2] > 0
        pts_cam_homo = pts_cam_homo[keep_idx]
        voxel_grid = voxel_grid[keep_idx]
#        print("pts_cam_filtered", pts_cam_homo.min(0), pts_cam_homo.max(0))

        pts_cam = pts_cam_homo[:, :3]
        pts_img = (self.cam_k @ pts_cam.T).T
#        print("pts_img", pts_img.shape)
#        print(np.expand_dims(pts_img[:, 2], axis=1).shape)
        pts_img = pts_img[:, :2] / np.expand_dims(pts_img[:, 2], axis=1)
        pts_img = np.rint(pts_img).astype(int)


        keep_idx = select_points_in_frustum(pts_img, 0, 0, 640, 480)

        img_indices = pts_img[keep_idx]
#        # fliplr so that indexing is row, col and not col, row
        img_indices = np.fliplr(img_indices)
#
        voxel_indices = voxel_grid[keep_idx]
#
        return img_indices, voxel_indices 
    
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
    pred_depth_dir = "/gpfsscratch/rech/xqt/uyl37fq/NYU_pred_depth"
    train_dataset = NYUDataset(split="train", 
                               root=NYU_dir, 
                               pred_depth_dir=pred_depth_dir,
                               preprocess_dir=preprocess_dir, 
                               extract_data=True)
    test_dataset = NYUDataset(split="test", 
                              root=NYU_dir, 
                              pred_depth_dir=pred_depth_dir,
                              preprocess_dir=preprocess_dir, 
                              extract_data=True)
#    for i in tqdm(range(len(test_dataset))):
#        test_dataset[i]
    for i in tqdm(range(len(train_dataset))):
        train_dataset[i]
#    print("number of items", len(dataset))
#    for i in tqdm(range(0, 100, 10)):
#        batch = dataset[i]
#        filepath = os.path.join(write_path, batch["name"] + ".pkl")
#        with open(filepath, 'wb') as handle:
#            pickle.dump(batch, handle)
#            print("wrote to", filepath)

if __name__ == '__main__':
    main()


