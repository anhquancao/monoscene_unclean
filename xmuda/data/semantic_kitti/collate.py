import torch
from functools import partial


def collate_fn(batch):
    # locs_3d = []
    locs_2d = []
    imgs = []
    edges = []
    img_idxs = []
    ssc_label_1_1s = []
    ssc_label_1_4s = []
    voxel_occupancies = []
    seg_label_2ds = []
    scenes = []

    edge_sparse_coords = []
    edge_sparse_feats = []

    list_voxel_indices_1_4s = []
    list_img_indices_1_4s = []

    cam_ks = []
    T_velo_2_cams = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict['cam_k']).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict['T_velo_2_cam']).float())

        list_voxel_indices_1_4s.append(input_dict['voxel_indices_1_4'])
        list_img_indices_1_4s.append(input_dict['img_indices_1_4'])
#        list_pixel_indices_per_voxel.append(input_dict['pixel_indices_per_voxel'])

        # print(input_dict)
        scenes.append(input_dict['scene'])

#        coords_2d = torch.from_numpy(input_dict['coords_2d'])
#        batch_idxs_2d = torch.LongTensor(coords_2d.shape[0], 1).fill_(idx)
#        locs_2d.append(torch.cat([coords_2d, batch_idxs_2d], 1))

        # img = torch.from_numpy(input_dict['img'])
        img = input_dict['img']
        imgs.append(img)

#        edge = input_dict['edge']
#        edges.append(edge)

#        img_indices = torch.from_numpy(input_dict['img_indices'])
#        img_idxs.append(img_indices)

        ssc_label_1_1 = torch.from_numpy(input_dict['ssc_label_1_1'])
        ssc_label_1_1s.append(ssc_label_1_1)

#        seg_label_2d = torch.from_numpy(input_dict['seg_label_2d'])
#        seg_label_2ds.append(seg_label_2d)

#        seg_label_3d = torch.from_numpy(input_dict['seg_label_3d'])
#        seg_label_3ds.append(seg_label_3d)

        # ssc_label_1_2 = torch.from_numpy(input_dict['ssc_label_1_2'])
        # ssc_label_1_2s.append(ssc_label_1_2)

        ssc_label_1_4 = torch.from_numpy(input_dict['ssc_label_1_4'])
        ssc_label_1_4s.append(ssc_label_1_4)

        # ssc_label_1_8 = torch.from_numpy(input_dict['ssc_label_1_8'])
        # ssc_label_1_8s.append(ssc_label_1_8)

        voxel_occupancy = torch.from_numpy(input_dict['voxel_occupancy'])
        voxel_occupancies.append(voxel_occupancy)

#    coords_2d = torch.cat(locs_2d, 0)
#    edge_sparse_coords, edge_sparse_feats = ME.utils.sparse_collate(coords=edge_sparse_coords, feats=edge_sparse_feats)

    return {
#        "voxel_indices": voxel_indices,
#        "pixel_indices_per_voxel": pixel_indices_per_voxel,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "voxel_indices_1_4": list_voxel_indices_1_4s,
        "img_indices_1_4": list_img_indices_1_4s,
        "scene": scenes,
#        "seg_label_2d": torch.cat(seg_label_2ds, 0),
#        "coords_2d": coords_2d,
        "img": torch.stack(imgs),
#        "img_indices": img_idxs,
        "ssc_label_1_1": torch.stack(ssc_label_1_1s),
        "ssc_label_1_4": torch.stack(ssc_label_1_4s),
        "voxel_occupancy": torch.stack(voxel_occupancies),
    }
