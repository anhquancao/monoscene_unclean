from xmuda.data.NYU.nyu_dataset import NYUDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
from xmuda.models.projection_layer import Project2Dto3D
import os
import cv2
import pickle
import torch


is_true_depth = True
NYU_dir = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp"
preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
train_dataset = NYUDataset(split="train",
                           root=NYU_dir,
                           use_predicted_depth=not is_true_depth,
                           preprocess_dir=preprocess_dir,
                           extract_data=False)
test_dataset = NYUDataset(split="test",
                          root=NYU_dir,
                          preprocess_dir=preprocess_dir,
                          use_predicted_depth=not is_true_depth,
                          extract_data=False)


def generate_occ_map(mapping, scale):
    project = Project2Dto3D(240/scale, 144/scale, 240/scale)  # w=240, h=144, d=240
    ones = torch.ones(1, 1, 480, 640) 
    occ_map = project(ones, mapping)
    return occ_map

def run(dataset, typ, is_true_depth=False):
    if is_true_depth:
        occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_gt_depth/NYU" + typ
    else:
        occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_adabin/NYU" + typ

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        mapping_1_1 = torch.from_numpy(batch['mapping_1_1']).long().unsqueeze(0)
        mapping_1_4 = torch.from_numpy(batch['mapping_1_4']).long().unsqueeze(0)
        name = batch['name']
        occ_path_1_4 = os.path.join(occ_root, name + "_1_4.npy")
        occ_path_1_1 = os.path.join(occ_root, name + "_1_1.npy")
        os.makedirs(occ_root, exist_ok=True)
#
##        if not os.path.exists(occ_path_1_1):
        occ_1_4 = generate_occ_map(mapping_1_4, 4).squeeze().numpy()
        occ_1_1 = generate_occ_map(mapping_1_1, 1).squeeze().numpy()
##            print(occ_1_1.shape)
        np.save(occ_path_1_4, occ_1_4)
        print("wrote to", occ_path_1_4)
        np.save(occ_path_1_1, occ_1_1)
        print("wrote to", occ_path_1_1)

run(test_dataset, "test", is_true_depth)
run(train_dataset, "train", is_true_depth)

                  
