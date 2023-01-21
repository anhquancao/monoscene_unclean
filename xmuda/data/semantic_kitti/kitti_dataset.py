import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from xmuda.data.utils.preprocess import vox2pix, compute_local_frustums, compute_CP_mega_matrix
from xmuda.data.semantic_kitti.params import kitti_class_cluster_4, kitti_class_cluster_6
import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
import imageio
import xmuda.common.utils.fusion as fusion
from time import time


class KittiDataset(Dataset):
    def __init__(self,
                 split,
                 root,
                 preprocess_root,
                 modalities={
                     "depth": True,
                     "tsdf": True,
                     "occ": True,
                     "label": True,
                     "mapping": True,
                     "sketch": True
                 },
                 n_relations=4,
                 project_scale=2,
                 frustum_size=4,
                 color_jitter=None,
                 fliplr=0.0,
                 use_predicted_depth=True):
        super().__init__()
        self.modalities = modalities
        self.TSDF_root = os.path.join(preprocess_root, "tsdf_depth_pred")
        self.occ_root = os.path.join(preprocess_root, "occupancy_depth_pred")
        self.label_root = os.path.join(preprocess_root, "labels")
        self.sketch_root = os.path.join(preprocess_root, "sketch")
        self.mapping_root = os.path.join(preprocess_root, "mapping_depth_pred")
        self.depth_root = os.path.join(preprocess_root, "depth_pred")
        self.root = root
        self.n_relations = n_relations

        self.n_classes = 20
        splits = {'train': ["00", "01", "02", "03", "04", "05", '06', '07', '09', '10'],
                  'val': ['08'],
                  'test': ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]}
        self.split = split
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.scene_size = (51.2, 51.2, 6.4)
        self.use_predicted_depth = use_predicted_depth
        self.fliplr = fliplr

        self.voxel_size = 0.2 # 0.2m
        self.img_W = 1220
        self.img_H = 370

        self.color_jitter = transforms.ColorJitter(*color_jitter) if color_jitter else None
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

        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        # start = time()
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

        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
            "depth": depth,
        }
        if self.modalities["tsdf"]:
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

        if self.modalities['label']:
            target_1_path = os.path.join(self.label_root, sequence, frame_id + '_1_1.npy')
            # target_2_path = os.path.join(self.label_root, sequence, frame_id + '_1_2.npy')
            # target = np.load(target_2_path)
            # if self.project_scale == 2:
            target = np.load(target_1_path)
            data['target'] = target
            target_8_path = os.path.join(self.label_root, sequence, frame_id + '_1_8.npy')
            target_1_8 = np.load(target_8_path)
            class_cluster = None                
            if self.n_relations == 8:
                class_cluster = kitti_class_cluster_4
            elif self.n_relations == 16:
                class_cluster = kitti_class_cluster_4
            CP_mega_matrix = compute_CP_mega_matrix(target_1_8, self.n_relations, class_cluster)
            # elif self.project_scale == 4:
            #     target = np.load(target_2_path)
            #     data['target'] = target
            #     target_16_path = os.path.join(self.label_root, sequence, frame_id + '_1_16.npy')
            #     target_1_16 = np.load(target_16_path)
            #     CP_mega_matrix = compute_CP_mega_matrix(target_1_16, self.n_relations)
            data["CP_mega_matrix"] = CP_mega_matrix

            if self.split != "test":
            # if self.split != "test" and self.split != "val":
                pix_output = data['pix_{}'.format(self.output_scale)]
                pix_z_output = data['pix_z_{}'.format(self.output_scale)]
#                if not self.virtual_img:
                local_frustums, list_cnts = compute_local_frustums(pix_output, pix_z_output, target,
                                                                   self.img_W, self.img_H,
                                                                   dataset="kitti", n_classes=20, size=self.frustum_size)
#                else:
#                    local_frustums, list_cnts = compute_local_frustums_enlarge(pix_output, pix_z_output, target, self.img_W, self.img_H,
#                                                                       dataset="kitti", n_classes=20)
            else:
                local_frustums = None
                list_cnts = None
            data['local_frustums'] = local_frustums
            data['local_frustums_cnt'] = list_cnts

        if self.modalities['occ']:
            # print(self.use_predicted_depth)
            if self.use_predicted_depth:
                occ_path_1_1 = os.path.join(self.occ_root, sequence, frame_id + "_1_1.npy")
                occ = np.load(occ_path_1_1)
            else:                
                occ_path = os.path.join(self.root, 'dataset', 'sequences', sequence, 'voxels', frame_id + '.bin')
                occ = SemanticKittiIO._read_occupancy_SemKITTI(occ_path)            
                occ = occ.reshape(256, 256, 32)
                # print(occ.shape)

            xs, ys, zs = occ.nonzero()
            occ_noise = np.zeros(occ.shape)
            sigma = 4
            xs_noise = np.clip(xs + sigma * np.random.randn(xs.shape[0]), 0, occ.shape[0] - 1).astype(int)
            ys_noise = np.clip(ys + sigma * np.random.randn(ys.shape[0]), 0, occ.shape[1] - 1).astype(int)
            zs_noise = np.clip(zs + sigma * np.random.randn(zs.shape[0]), 0, occ.shape[2] - 1).astype(int)
            occ_noise[xs_noise, ys_noise, zs_noise] = 1.0

            data['occ_1_1'] = occ_noise
            # data['occ_1_1'] = occ

            # occ_FOV = np.copy(occ)
            # occ_FOV[~data['valid_pix_1'].reshape(256, 256, 32)] = 0
            # data['occ_1_1'] = occ_FOV
            # print(occ_FOV.sum() / occ.sum())

            # monoscene_output_path = os.path.join(
            #     "/gpfsscratch/rech/kvd/uyl37fq/temp/features/kitti",
            #     sequence, "{}.npy".format(frame_id)
            # )
            # occ = np.load(monoscene_output_path).astype(np.float32)
            # # # print(occ.shape)
            # occ[occ > 0] = 1
            # occ[target == 255] = 0
            
            # data['occ_1_1'] = occ
            

        if self.modalities['mapping']:
            mapping_1_4_path = os.path.join(self.mapping_root, sequence, frame_id + "_1_4.npy")
            mapping_1_4 = np.load(mapping_1_4_path)
            mapping_1_1_path = os.path.join(self.mapping_root, sequence, frame_id + "_1_1.npy")
            mapping_1_1 = np.load(mapping_1_1_path)
            data["mapping_1_4"] = mapping_1_4
            data["mapping_1_1"] = mapping_1_1

        if self.modalities['sketch']:
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
        img = img[:370, :1220, :] # crop image

        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scales:
                key = 'pix_' + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:,0] 

        data['img'] = self.normalize_rgb(img)
        # print(f'Time taken to run: {time() - start} seconds')
        return data

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
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
        rgb = imageio.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        return rgb

    @staticmethod
    def _read_depth(depth_filename):
        depth = imageio.imread(depth_filename) / 256.0  # numpy.float64
        depth = np.asarray(depth)
        return depth

