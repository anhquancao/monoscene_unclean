from xmuda.data.semantic_kitti.kitti_dataset import KittiDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
from xmuda.models.projection_layer import Project2Dto3D
import os
import cv2
import pickle
import torch

kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
TSDF_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/kitti"
#train_dataset = KittiDataset(split="train", 
#                             root=kitti_root, 
#                             mapping_root=mapping_root,
#                             TSDF_root=TSDF_root,
#                             depth_root=depth_root)
test_dataset = KittiDataset(split="test", 
                            root=kitti_root, 
                            project_scale=2,
                            mapping_root=mapping_root,
                            TSDF_root=TSDF_root,
                            depth_root=depth_root)


def generate_occ_map(mapping, scale):
    project = Project2Dto3D(256//scale, 256//scale, 32//scale)  # w=240, h=144, d=240
    ones = torch.ones(1, 1, 370, 1220)
    occ_map = project(ones, mapping)
    return occ_map

def run(dataset, typ):
    occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_adabin/kitti"

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        mapping_1_1 = torch.from_numpy(batch['mapping_1_1']).long().unsqueeze(0)
        mapping_1_4 = torch.from_numpy(batch['mapping_1_4']).long().unsqueeze(0)
        name = batch['name']
        sequence = batch['sequence']
        save_dir = os.path.join(occ_root, sequence)
        os.makedirs(save_dir, exist_ok=True)

        occ_path_1_1 = os.path.join(save_dir, name + "_1_1.npy")
#        occ_path_1_4 = os.path.join(save_dir, name + "_1_4.npy")

#        if not os.path.exists(occ_path_1_1):
        occ_1_1 = generate_occ_map(mapping_1_1, 1).squeeze().numpy()
#        occ_1_4 = generate_occ_map(mapping_1_4, 4).squeeze().numpy()
#            print(occ_1_1.shape)

        np.save(occ_path_1_1, occ_1_1)
        print("wrote to", occ_path_1_1)
#        np.save(occ_path_1_4, occ_1_4)
#        print("wrote to", occ_path_1_4)

run(test_dataset, "test")
#run(train_dataset, "train")

                  
