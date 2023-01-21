import numpy as np


def to_voxel_coords(points, 
                    voxel_size, 
                    full_scale, 
                    unique=True,
                    lower_bound=np.array([0, -25.6, -2]).reshape(1, 3)):
    """
    3D point cloud augmentation and scaling from points (in meters) to voxels
    :param points: 3D points in meters
    :param voxel_size: voxel size in m, e.g. 0.05 corresponds to 5cm voxels
    :param full_scale: size of the receptive field of SparseConvNet        
    :return coords: the coordinates that are given as input to SparseConvNet
    """
    # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
    points -= lower_bound
    
    coords = (points / voxel_size).astype(np.int64)

    idx = None
    # print(coords.shape)    
    if unique:
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        temp = np.zeros(coords.shape[0])
        temp[unique_indices] = 1

        idx = (coords.min(1) >= 0) * (coords.max(1) < full_scale) * temp.astype(bool)
        coords = coords[idx]
    # print(np.sum(idx))

    return coords, idx
