import numpy as np
import xmuda.common.utils.fusion as fusion
import torch


def construct_voxel_rel_label(label, binary=False):
    """
#        same:0
#        diff_non: 1
#        diff_empty: 2
    same_non:0
    same_empty:1
    diff_non: 2
    diff_empty: 3
    """
    label = label.reshape(-1)
#    mask = (label != 255)
#    label = label[mask]
    N = label.shape[0]
    label_row = label.reshape(N, 1).expand(-1, N)
    label_col = label.reshape(1, N).expand(N, -1)
#    label_row = np.repeat(label[:, None], N, axis=1)
#    label_col = np.repeat(label[None, :], N, axis=0)
    matrix = torch.zeros((N, N))

    if binary:
        matrix[label_row == label_col] = 0
        matrix[label_row != label_col] = 1
    else:
        matrix[(label_row == label_col) & (label_col != 0)] = 0 # non non same
        matrix[(label_row != label_col) & (label_row != 0) & (label_col != 0)] = 1 # non non diff
        matrix[(label_row != label_col) & ((label_row == 0) | (label_col == 0))] = 2 # nonempty empty
        matrix[(label_row == label_col) & (label_col == 0)] = 3 # empty empty
        matrix[(label_row == 255) | (label_col == 255)] = 4 # unknown relation
    return matrix

def compute_mega_context(target, n_classes=12):
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i//2 for i in target.shape]

    mega_context = np.zeros((n_classes, super_voxel_size[0], super_voxel_size[1], super_voxel_size[2]))
    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                children_labels = [
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ]
                mega_context[[i for i in children_labels if i!=255], xx, yy, zz]  = 1

    return mega_context

def compute_CP_mega_matrix(target, n_relations, class_cluster=None):    
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i//2 for i in target.shape]
    
    matrix = np.zeros((n_relations, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = np.array([
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N)  * label_col_mega
                    if class_cluster is not None:
                        group_row = np.copy(label_row)
                        group_col = np.copy(label_col)
                        for k in class_cluster.keys():
                            group_col[label_col == k] = class_cluster[k]
                            group_row[label_row == k] = class_cluster[k]

                    if n_relations == 16:                        
                        # matrix[0, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        # matrix[1, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty

                        # matrix[2, (label_row == 1) & (label_col == 1), col_idx] = 1.0 
                        # matrix[3, (label_row == 2) & (label_col == 2), col_idx] = 1.0 
                        # matrix[4, (label_row == 3) & (label_col == 3), col_idx] = 1.0 
                        # matrix[5, (label_row == 4) & (label_col == 4), col_idx] = 1.0 
                        # matrix[5, (label_row == 5) & (label_col == 5), col_idx] = 1.0 # merge relation 4-4 and 5-5

                        # matrix[6, ((label_row == 1) & (label_col == 2)) | ((label_row == 2) & (label_col == 1)), col_idx] = 1.0 
                        # matrix[7, ((label_row == 1) & (label_col == 3)) | ((label_row == 3) & (label_col == 1)), col_idx] = 1.0 
                        # matrix[8, ((label_row == 1) & (label_col == 4)) | ((label_row == 4) & (label_col == 1)), col_idx] = 1.0
                        # matrix[9, ((label_row == 1) & (label_col == 5)) | ((label_row == 5) & (label_col == 1)), col_idx] = 1.0

                        # matrix[10, ((label_row == 2) & (label_col == 3)) | ((label_row == 3) & (label_col == 2)), col_idx] = 1.0
                        # matrix[11, ((label_row == 2) & (label_col == 4)) | ((label_row == 4) & (label_col == 2)), col_idx] = 1.0
                        # matrix[12, ((label_row == 2) & (label_col == 5)) | ((label_row == 5) & (label_col == 2)), col_idx] = 1.0

                        # matrix[13, ((label_row == 3) & (label_col == 4)) | ((label_row == 4) & (label_col == 3)), col_idx] = 1.0
                        # matrix[14, ((label_row == 3) & (label_col == 5)) | ((label_row == 5) & (label_col == 3)), col_idx] = 1.0

                        # matrix[15, ((label_row == 4) & (label_col == 5)) | ((label_row == 5) & (label_col == 4)), col_idx] = 1.0
                        matrix[0, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[1, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty

                        matrix[2, (label_row != 255) & (label_row == label_col) & (group_row == 1), col_idx] = 1.0 
                        matrix[3, (label_row != 255) & (label_row == label_col) & (group_row == 2), col_idx] = 1.0 
                        matrix[4, (label_row != 255) & (label_row == label_col) & (group_row == 3), col_idx] = 1.0 
                        matrix[5, (label_row != 255) & (label_row == label_col) & (group_row == 4), col_idx] = 1.0 
                        matrix[6, (label_row != 255) & (label_row == label_col) & (group_row == 5), col_idx] = 1.0 
                        matrix[7, (label_row != 255) & (label_row != label_col) & (group_row == group_col), col_idx] = 1.0
            
                        matrix[8, (label_row != 255) & (label_row != label_col) & (((group_row == 1) & (group_col == 2)) | ((group_row == 2) & (group_col == 1))), col_idx] = 1.0 
                        matrix[9, (label_row != 255) & (label_row != label_col) & (((group_row == 1) & (group_col == 3)) | ((group_row == 3) & (group_col == 1))), col_idx] = 1.0 
                        matrix[10, (label_row != 255) & (label_row != label_col) & (((group_row == 1) & (group_col == 4)) | ((group_row == 4) & (group_col == 1))), col_idx] = 1.0 
                        matrix[11, (label_row != 255) & (label_row != label_col) & (((group_row == 1) & (group_col == 5)) | ((group_row == 5) & (group_col == 1))), col_idx] = 1.0 

                        matrix[12, (label_row != 255) & (label_row != label_col) & (((group_row == 2) & (group_col == 3)) | ((group_row == 3) & (group_col == 2))), col_idx] = 1.0
                        matrix[13, (label_row != 255) & (label_row != label_col) & (((group_row == 2) & (group_col == 4)) | ((group_row == 4) & (group_col == 2))), col_idx] = 1.0
                        matrix[14, (label_row != 255) & (label_row != label_col) & (((group_row == 2) & (group_col == 5)) | ((group_row == 5) & (group_col == 2))), col_idx] = 1.0

                        matrix[15, (label_row != 255) & (label_row != label_col) & (((group_row == 3) & (group_col == 4)) | ((group_row == 4) & (group_col == 3))), col_idx] = 1.0
                        matrix[14, (label_row != 255) & (label_row != label_col) & (((group_row == 3) & (group_col == 5)) | ((group_row == 5) & (group_col == 3))), col_idx] = 1.0

                        matrix[15, (label_row != 255) & (label_row != label_col) & (((group_row == 4) & (group_col == 5)) | ((group_row == 5) & (group_col == 4))), col_idx] = 1.0


                    if n_relations == 8:                        
                        matrix[0, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[1, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty

                        matrix[2, (label_row != 255) & (label_row == label_col) & (group_row == 1), col_idx] = 1.0 
                        matrix[3, (label_row != 255) & (label_row == label_col) & (group_row == 2), col_idx] = 1.0 
                        matrix[4, (label_row != 255) & (label_row == label_col) & (group_row == 3), col_idx] = 1.0 
                        matrix[5, (label_row != 255) & (label_row != label_col) & (group_row == group_col), col_idx] = 1.0
            
                        matrix[6, (label_row != 255) & (label_row != label_col) & (((group_row == 1) & (group_col == 2)) | ((group_row == 2) & (group_col == 1))), col_idx] = 1.0 
                        matrix[7, (label_row != 255) & (label_row != label_col) & (((group_row == 1) & (group_col == 3)) | ((group_row == 3) & (group_col == 1))), col_idx] = 1.0 
                        matrix[7, (label_row != 255) & (label_row != label_col) & (((group_row == 2) & (group_col == 3)) | ((group_row == 3) & (group_col == 2))), col_idx] = 1.0
                        
                        
                        
                    if n_relations == 4:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0 # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0 # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty
                    elif n_relations == 2:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0 # non non same
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0 # non non diff
    return matrix

def vox2pixCamCoords(cam_k, voxel_size, scale, img_W, img_H):
    voxel_size = voxel_size * scale
    vol_bnds = np.zeros((3,2))
    vox_origin = np.array([-25.6, -4, 0])
    scene_size = np.array([51.2, 6.4, 51.2])
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + scene_size

    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = cam_pts.reshape(256//scale, 32//scale, 256//scale, 3)
    cam_pts = np.flip(cam_pts, 1)
    cam_pts = np.transpose(cam_pts, (2, 0, 1, 3))
    cam_pts = np.flip(cam_pts, 1)
    cam_pts = cam_pts.reshape(-1, 3)

    pix_z = cam_pts[:, 2]
    pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = pix[:, 0], pix[:, 1]

    # Eliminate pixels outside view frustum
    valid_pix = np.logical_and(pix_x >= 0, 
                               np.logical_and(pix_x < img_W, 
                                              np.logical_and(pix_y >= 0, 
                                                             np.logical_and(pix_y < img_H, pix_z > 0))))

    return pix, valid_pix

def vox2pix(cam_pose, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
#            swap,
            scene_size):
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, np.linalg.inv(cam_pose))

    pix_z = cam_pts[:, 2]
    pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = pix[:, 0], pix[:, 1]

    # Eliminate pixels outside view frustum
    valid_pix = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return pix, valid_pix, pix_z


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                np.logical_and(pix_x < max_x,
                np.logical_and(pix_y >= min_y,
                np.logical_and(pix_y < max_y,
                pix_z > 0))))
    return valid_pix

def compute_local_frustums(pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustums = []
    list_cnts = []
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustums.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            list_cnts.append(class_counts)
    return np.array(local_frustums), np.array(list_cnts)


def compute_local_frustums_enlarge(pix, pix_z, target, img_W, img_H, dataset, n_classes):
    H, W, D = target.shape
    size_w = 24
#    size_w = 16
    size_h = 8
#    size_w = 16
#    size_h = 16
    ranges_w = [(i * 1.0/size_w, (i * 1.0 + 1)/size_w) for i in range(size_w)]
    ranges_h = [(i * 1.0/size_h, (i * 1.0 + 1)/size_h) for i in range(size_h)]
    local_frustums = []
    list_cnts = []
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    for y in ranges_h:
        for x in ranges_w:
#            min_x = -0.5 * img_W + x[0] * (2 * img_W)
#            max_x = -0.5 * img_W + x[1] * (2 * img_W)
#            min_y = -0.5 * img_H + y[0] * (2 * img_H)
#            max_y = -0.5 * img_H + y[1] * (2 * img_H)
            min_x = -1.0 * img_W + x[0] * (3 * img_W)
            max_x = -1.0 * img_W + x[1] * (3 * img_W)
            min_y = -0.0 * img_H + y[0] * (1 * img_H)
            max_y = -0.0 * img_H + y[1] * (1 * img_H)
            local_frustum = np.logical_and(pix_x >= min_x,
                        np.logical_and(pix_x < max_x,
                        np.logical_and(pix_y >= min_y,
                        np.logical_and(pix_y < max_y,
                        pix_z > 0))))
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)
            
            if np.sum(mask) > 80:
                local_frustums.append(mask)
                classes, cnts = np.unique(target[mask], return_counts=True)
                class_counts = np.zeros(n_classes)
                class_counts[classes.astype(int)] = cnts
                list_cnts.append(class_counts)
    return np.array(local_frustums), np.array(list_cnts)

def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] >= x1) * \
                   (points_2d[:, 1] >= y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

def get_pixel_indices_per_voxel(img_indices, voxel_to_pixel_indices):
    voxel_to_pixel_indices = voxel_to_pixel_indices.unsqueeze(-1).expand(-1, -1, 3)
    img_indices = img_indices.unsqueeze(0).expand(voxel_to_pixel_indices.shape[0], -1, -1)
    selected_pixel_indices = torch.gather(img_indices, 1, voxel_to_pixel_indices)
    return selected_pixel_indices

def create_img_grid(img_size, downsample):
    g_xx = np.arange(0, img_size[0])
    g_yy = np.arange(0, img_size[1])
    xx, yy = np.meshgrid(g_xx, g_yy)
    img_grid = np.array([xx.flatten(), yy.flatten()]).T

    # the projection is performed with full resolutions image
    img_grid = img_grid.astype(np.float) * downsample
    # img_grid_homo = np.hstack([img_grid, np.ones((img_grid.shape[0], 1))])

    return img_grid

def create_voxel_position(dx, dy, dz): 
    g_xx = torch.arange(0, dx).reshape(dx, 1, 1).expand(-1, dy, dz)
    g_yy = torch.arange(0, dy).reshape(1, dy, 1).expand(dx, -1, dz)
    g_zz = torch.arange(0, dz).reshape(1, 1, dz).expand(dx, dy, -1)
    grid = torch.stack([g_xx, g_yy, g_zz])
    return grid



def create_voxel_grid(size):
    g_xx = np.arange(0, size[0])
    g_yy = np.arange(0, size[1])
    g_zz = np.arange(0, size[2])
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

    # the projection is performed with full resolutions image
    grid = grid.astype(np.float)
    # img_grid_homo = np.hstack([img_grid, np.ones((img_grid.shape[0], 1))])

    return grid

