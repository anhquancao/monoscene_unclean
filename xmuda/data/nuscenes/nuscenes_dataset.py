import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from nuscenes.nuscenes import NuScenes
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum, vox2pixCamCoords, compute_local_frustums, compute_CP_mega_matrix, compute_local_frustums_enlarge
from PIL import Image
from torchvision import transforms

class NuscenesDataset(Dataset):
    def __init__(self, root):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
        self.img_H = 900
        self.img_W = 1600
        self.voxel_size = 0.2
        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, index):
        data = {}
        sample = self.nusc.sample[index]
        sample_record = self.nusc.get('sample', sample['token'])
        camera_token = sample_record['data']['CAM_FRONT']
        cam_record = self.nusc.get('sample_data', camera_token)


        img = Image.open(os.path.join(self.nusc.dataroot, cam_record['filename']))
        data['img_path'] = cam_record['filename']
        img = np.array(img, dtype=np.float32, copy=False) / 255.
        img = self.normalize_rgb(img)
        data['img'] = img

        calib_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_k = np.array(calib_record['camera_intrinsic'])
        T_velo_2_cam = np.identity(4) # reconstruct in camera coordinates 

        data['T_velo_2_cam'] = T_velo_2_cam
        data['cam_k'] = cam_k
        data['frame_id'] =  str(index)

        scales = [1, 2]
        data['scales'] = scales
        vox_origin = np.array([-25.6, -4, 0])
        for scale in scales:
            pix, valid_pix = vox2pixCamCoords(cam_k, self.voxel_size, scale, self.img_W, self.img_H)
            data['pix_' + str(scale)] = pix
            data['valid_pix_' + str(scale)] = valid_pix
        return data


if __name__ == '__main__':
    nuscenes_ds = NuscenesDataset('/gpfsscratch/rech/kvd/uyl37fq/dataset/nuscenes')
    for i in range(len(nuscenes_ds)):
        print(list(nuscenes_ds[i].keys()))

