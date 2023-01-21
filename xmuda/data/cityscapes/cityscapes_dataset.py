import torch
import json
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from nuscenes.nuscenes import NuScenes
from xmuda.data.utils.preprocess import create_img_grid, create_voxel_grid, select_points_in_frustum, vox2pixCamCoords, compute_local_frustums, compute_CP_mega_matrix, compute_local_frustums_enlarge
from PIL import Image
from torchvision import transforms

class CityscapesDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.img_root = os.path.join(root, "leftImg8bit", "demoVideo", "stuttgart_01")
        self.scans = []
        for path, subdirs, files in os.walk(self.img_root):
            for name in files:
                if (".png" in name): #and ("berlin" in name):
                    img_path = os.path.join(path, name)
    #                    camera_path = img_path.replace("leftImg8bit", "camera").replace(".png", ".json")
                    camera_path = "/gpfsscratch/rech/kvd/uyl37fq/dataset/cityscapes/camera/test/berlin/berlin_000000_000019_camera.json"
                    self.scans.append(
                        {
                            "img_path": img_path,
                            "camera_path": camera_path
                        }
                    )

        self.img_H = 1024
        self.img_W = 2048
        self.voxel_size = 0.2
        self.scene_size = (51.2, 51.2, 6.4)
        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
#        self.scans = self.scans[::10] # not get too close scan

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        data = {}
        scan = self.scans[index]
        img_path = scan['img_path']
        data['img_path'] = img_path

        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32, copy=False) / 255.
        img = self.normalize_rgb(img)
        data['img'] = img

        camera_path = scan['camera_path']
        with open(camera_path) as f:
            t = json.load(f)
            intrinsic = t['intrinsic']
            fx = intrinsic['fx']
            fy = intrinsic['fy']
            u0 = intrinsic['u0']
            v0 = intrinsic['v0']
            cam_k = np.array([
                    [fx, 0, u0],
                    [0, fy, v0],
                    [0, 0, 1],
                ])

        T_velo_2_cam = np.identity(4) # reconstruct in camera coordinates 

        data['T_velo_2_cam'] = T_velo_2_cam
        data['cam_k'] = cam_k
        data['frame_id'] = os.path.basename(img_path)[:-16] 

        scales = [1, 2]
        data['scales'] = scales
#        vox_origin = np.array([0, -25.6, -2])
        vox_origin = np.array([-25.6, -4, 0])
        for scale in scales:
            pix, valid_pix = vox2pixCamCoords(cam_k, self.voxel_size, scale, self.img_W, self.img_H)
            data['pix_' + str(scale)] = pix
            data['valid_pix_' + str(scale)] = valid_pix
        return data


if __name__ == '__main__':
    ds = CityscapesDataset(root="/gpfsscratch/rech/kvd/uyl37fq/dataset/cityscapes",
                           split="val")
    for i in range(len(ds)):
        print(list(ds[i].keys()))

