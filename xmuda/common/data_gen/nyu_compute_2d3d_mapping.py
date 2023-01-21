from xmuda.data.NYU.nyu_dataset import NYUDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
import os
import cv2
import pickle

NYU_dir = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp"
preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
train_dataset = NYUDataset(split="train",
                           root=NYU_dir,
                           preprocess_dir=preprocess_dir,
                           extract_data=False)
test_dataset = NYUDataset(split="test",
                          root=NYU_dir,
                          preprocess_dir=preprocess_dir,
                          extract_data=False)


def depth2voxel(depth, cam_pose, vox_origin, cam_k):
#	cam_k = param['cam_k']
#	voxel_size = param['voxel_size']  # (240, 144, 240)
#	voxel_size = (60, 36, 60)
	voxel_size = (240, 144, 240)
#	unit = param['voxel_unit']  # 0.02
	unit = 0.02
	# ---- Get point in camera coordinate
	H, W = depth.shape
	gx, gy = np.meshgrid(range(W), range(H))
	pt_cam = np.zeros((H, W, 3), dtype=np.float32)
	pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
	pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
	pt_cam[:, :, 2] = depth  # z, in meter
	# ---- Get point in world coordinate
	p = cam_pose
	pt_world = np.zeros((H, W, 3), dtype=np.float32)
	pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
	pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
	pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
	pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
	pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
	pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
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

	# ---- crop depth to grid/voxel
	# binary encoding '01': 0 for empty, 1 for occupancy
	voxel_binary = np.zeros([_ + 1 for _ in voxel_size], dtype=np.float32)  # (W, H, D)
	voxel_xyz = np.zeros(voxel_size + (3,), dtype=np.float32)  # (W, H, D, 3)
	position = np.zeros((H, W), dtype=np.int32)
	position4 = np.zeros((H, W), dtype=np.int32)

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
				position4[h, w] = np.ravel_multi_index((point_grid[h, w, :] / 4).astype(np.int32), voxel_size_lr)
				# position44[h / 4, w / 4] = np.ravel_multi_index(point_grid[h, w, :] / 4, voxel_size_lr)

	# output --- 3D Tensor, 240 x 144 x 240
	del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid  # Release Memory
	return position, position4 # (W, H, D), (W, H, D, 3)

def run(dataset, typ, is_true_depth):
    cam_intr = np.array([
        [518.8579, 0, 320],                                                                                                                           
        [0, 518.8579, 240],                                                                                                                           
        [0, 0, 1]
    ])
    if is_true_depth:
        depth_root = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin/NYU{}".format(typ) 
        sketch_mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_gt_depth/NYU{}".format(typ)
    else:
        depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/NYU/depth/NYU{}".format(typ) 
        sketch_mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/NYU{}".format(typ)
    rgb_root = os.path.join(NYU_dir, "NYU{}".format(typ))

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        cam_pose = batch['cam_pose']
        name = batch['name']
        vox_origin = batch['voxel_origin']
        mapping_path_1_1 = os.path.join(sketch_mapping_root, name + "_1_1.npy")
        mapping_path_1_4 = os.path.join(sketch_mapping_root, name + "_1_4.npy")

        if not os.path.exists(mapping_path_1_1):
#        if True:
#            depth_path = os.path.join(depth_root, name + "_color.png") 
            depth_path = os.path.join(depth_root, name + ".png") 
            depth = cv2.imread(depth_path, -1).astype(float)
            depth /= 8000.
            print(depth.max(), depth.min())
            
            position, position4 = depth2voxel(depth, cam_pose, vox_origin, cam_intr)

#            with open(mapping_path, 'wb') as handle:
#                data = {
#                    "1_1": position,
#                    "1_4": position4
#                }
#                pickle.dump(data, handle)
#                print("wrote to", mapping_path)
            np.save(mapping_path_1_1, position)
            print("wrote to", mapping_path_1_1)
            np.save(mapping_path_1_4, position4)
            print("wrote to", mapping_path_1_4)

is_true_depth = True
run(test_dataset, "test", is_true_depth)
run(train_dataset, "train", is_true_depth)

                  
