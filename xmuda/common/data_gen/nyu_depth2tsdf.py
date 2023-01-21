from xmuda.data.NYU.nyu_dataset import NYUDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
import os
import cv2

is_true_depth=False
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

def run(dataset, typ, is_true_depth=True, scale=4):
    cam_intr = np.array([
        [518.8579, 0, 320],                                                                                                                      [0, 518.8579, 240],                                                                                                                      [0, 0, 1]
    ])
    if is_true_depth:
        depth_root = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
        TSDF_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/NYU/tsdf_depth_gt"
    else:
        depth_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/NYU/depth_pred"
        TSDF_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/NYU/tsdf_depth_pred"
#        depth_root = "/gpfsscratch/rech/kvd/uyl37fq/NYU_pred_depth"
#        TSDF_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth"

    if typ == "train":
        depth_root = os.path.join(depth_root, "NYUtrain")
        TSDF_root = os.path.join(TSDF_root, "NYUtrain")
        rgb_root = os.path.join(NYU_dir, "NYUtrain")
    else:
        depth_root = os.path.join(depth_root, "NYUtest")
        TSDF_root = os.path.join(TSDF_root, "NYUtest")
        rgb_root = os.path.join(NYU_dir, "NYUtest")

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        cam_pose = batch['cam_pose']
        name = batch['name']
        vox_origin = batch['voxel_origin']
        nonempty = batch['nonempty']

        depth_path = os.path.join(depth_root, name + "_color.png") 
        print(depth_path)
#        depth_path = os.path.join(depth_root, name + ".png") 
        print(depth_path)
        depth_im = cv2.imread(depth_path, -1).astype(float)
        depth_im /= 8000.

        rgb_path = os.path.join(rgb_root, name + "_color.jpg")
        color_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        print(depth_im.shape, color_image.shape)

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array([60 * 0.08, 60 * 0.08 , 36 * 0.08])

        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02 * scale)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        
        tsdf_grid, _ = tsdf_vol.get_volume()
        tsdf_grid = np.moveaxis(tsdf_grid, [0, 1, 2], [0, 2, 1])
        print(np.min(tsdf_grid), np.max(tsdf_grid))
        tsdf_grid[tsdf_grid > 1.0] = 1.0
        tsdf_grid[tsdf_grid < -1.0] = -1.0
        tsdf_path = os.path.join(TSDF_root, name + "_1_{}.npy".format(scale))
        np.save(tsdf_path, tsdf_grid)
        print("Saved to", tsdf_path)

#        pred_nonempty = tsdf_grid < 0
#        print(np.sum(pred_nonempty == nonempty) / (60 * 36 * 60))

run(train_dataset, "train", is_true_depth=is_true_depth, scale=4)
run(test_dataset, "test", is_true_depth=is_true_depth, scale=4)


