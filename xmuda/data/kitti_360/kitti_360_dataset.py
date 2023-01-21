import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
#from kitti360scripts.helpers.project import CameraPerspective
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum, vox2pix, compute_local_frustums, compute_CP_mega_matrix, compute_local_frustums_enlarge
from PIL import Image
from torchvision import transforms

class Kitti360Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.sequences = ['2013_05_28_drive_0009_sync']
        self.img_H = 376
        self.img_W = 1408
        self.voxel_size = 0.2
        self.scene_size = (51.2, 51.2, 6.4)
        self.T_velo_2_cam = self.get_velo2cam()
        self.cam_k = self.get_cam_k()
        self.scans = []
        for sequence in self.sequences:
            glob_path = os.path.join(self.root, "data_2d_raw", sequence, 'image_00/data_rect', '*.png')
            for img_path in glob.glob(glob_path):
                self.scans.append({
                    "img_path": img_path
                })
#        self.scans = self.scans[::10]
        self.scans = self.scans[:5000]
        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.scans)

    def get_cam_k(self):
        cam_k = np.array([552.554261, 0.000000, 682.049453, 0.000000, 0.000000, 552.554261, 238.769549, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]).reshape(3, 4)
        return cam_k[:3, :3]

    def get_velo2cam(self):
        cam2velo = np.array([0.04307104361, -0.08829286498, 0.995162929, 0.8043914418, -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574, -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824]).reshape(3, 4)
        cam2velo = np.concatenate([cam2velo, np.array([0,0,0,1]).reshape(1,4)], axis=0)
        return np.linalg.inv(cam2velo)
    
    def __getitem__(self, index):
        data = {
            'T_velo_2_cam': self.T_velo_2_cam,
            'cam_k': self.cam_k
        }
        scan = self.scans[index]
        img_path = scan['img_path']
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]
        data['frame_id'] = frame_id
        data['img_path'] = img_path

        img = Image.open(img_path).convert('RGB') 
        img = np.array(img, dtype=np.float32, copy=False) / 255.
        img = self.normalize_rgb(img)
        data['img'] = img

        scales = [1, 2]
        data['scales'] = scales
        vox_origin = np.array([0, -25.6, -2])
        for scale in scales:
            pix, valid_pix, pix_z = vox2pix(np.linalg.inv(self.T_velo_2_cam), self.cam_k, vox_origin, self.voxel_size * scale, self.img_W, self.img_H, self.scene_size)
            data['pix_' + str(scale)] = pix
#            data['pix_z_' + str(scale)] = pix_z
            data['valid_pix_' + str(scale)] = valid_pix
        return data


if __name__ == '__main__':
    root_dir = "/gpfsdswork/dataset/KITTI-360"
    kitti360_ds = Kitti360Dataset(root_dir)
    for i in range(len(kitti360_ds)):
        print(list(kitti360_ds[i].keys()))

