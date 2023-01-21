import torch
import torch.nn as nn
from xmuda.data.semantic_kitti.kitti_dataset import KittiDataset
import os
from tqdm import tqdm
import numpy as np


class Sobel3D(nn.Module):
    def __init__(self):
        super(Sobel3D, self).__init__()
        self.kernel_x = [
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1], ],
            [[2, 0, -2],
             [2, 0, -2],
             [2, 0, -2], ],
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1], ]]

        self.kernel_y = [
            [[1, 2, 1],
             [1, 2, 1],
             [1, 2, 1], ],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0], ],
            [[-1, -2, -1],
             [-1, -2, -1],
             [-1, -2, -1], ]]

        self.kernel_z = [
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1], ]]

        self.kernel_x = torch.FloatTensor(self.kernel_x).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor(self.kernel_y).unsqueeze(0).unsqueeze(0)
        self.kernel_z = torch.FloatTensor(self.kernel_z).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=self.kernel_x, requires_grad=False)#.cuda()
        self.weight_y = nn.Parameter(data=self.kernel_y, requires_grad=False)#.cuda()
        self.weight_z = nn.Parameter(data=self.kernel_z, requires_grad=False)#.cuda()

        self.conv_x = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight = self.weight_x
        self.conv_y = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y.weight = self.weight_y
        self.conv_z = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_z.weight = self.weight_z

    def forward(self, data):
        data[data == 255] = 0
        # print(self.conv_x.weight)

        data = data.float()

        dx = self.conv_x(data)
        dy = self.conv_y(data)
        dz = self.conv_z(data)
        dx[abs(dx) > 0] = 1
        dy[abs(dy) > 0] = 1
        dz[abs(dz) > 0] = 1
        cat = torch.cat([dx, dy, dz], dim=1)
        norm = torch.norm(cat, dim=1, keepdim=True)
        thresh = 1
        norm[norm <= thresh] = 0
        norm[norm > thresh] = 1
        return norm

def run(dataset):
    sketch_root = "/gpfsscratch/rech/xqt/uyl37fq/sketch_dataset/sketch_3D/kitti"
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        label = data['target_1_1']
        sequence = data['sequence']
        name = data['name']
#        print(label.shape)
        label = torch.from_numpy(label).view(1, 1, 256, 256, 32)
        sobel = Sobel3D()
        sketch = sobel(label)
        sketch = sketch.squeeze().numpy()
#        print(sketch.shape)
        save_dir = os.path.join(sketch_root, sequence)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "{}_1_1.npy".format(name)), sketch)


kitti_root = "/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti"
depth_root = "/gpfsscratch/rech/xqt/uyl37fq/Adabin/KITTI/"
preprocess_dir = "/gpfsscratch/rech/xqt/uyl37fq/precompute_data/NYU"
tsdf_root = "/gpfsscratch/rech/xqt/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"

train_dataset = KittiDataset(split="train", 
                             root=kitti_root, 
                             TSDF_root=tsdf_root,
                             depth_root=depth_root)
test_dataset = KittiDataset(split="val", 
                            root=kitti_root, 
                            TSDF_root=tsdf_root,
                            depth_root=depth_root)

run(train_dataset)
run(test_dataset)
