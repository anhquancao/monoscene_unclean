import torch
from functools import partial
import numpy as np


def collate_fn(batch):
    data = {}
    imgs = []
#    k_indices = []
    ssc_label_1_4s = []
    ssc_label_1_8s = []
    ssc_label_1_16s = []
    class_proportion_1_4s = []
    names = []

    occ_1_1s = []
    occ_1_4s = []

#    mega_contexts = []

    depths = []
    pred_depths = []
    positions = []
    pred_depth_positions = []
    nonempties = []
    tsdf_1_4s = []
    tsdf_1_1s = []

    cam_poses = []
    vox_origins = []
    cam_ks = []
    sketch_original_mappings = []

#    pairwise_matrices_1_16s = []
    pairwise_mask_1_16s = []

    CP_mega_matrices = []
#    CP_masks = []

    scales = [4, 8, 16]
#    scales = [4]
    for scale in scales: 
#        data['bbs_' + str(scale)] = []
#        data['valid_bbs_' + str(scale)] = []

        data['pix_' + str(scale)] = []
        data['valid_pix_' + str(scale)] = []
        data['pix_z_' + str(scale)] = []
#        data['pts_cam_' + str(scale)] = []
        if scale == 4:
            data['local_frustums_' + str(scale)] = []
            data['local_frustums_cnt_' + str(scale)] = []

#    for aux in range(1): 
#        data['aux_pix_' + str(aux)] = []
#        data['aux_valid_pix_' + str(aux)] = []
#        data['aux_pts_cam_' + str(aux)] = []
    
    sketchs = []
    mapping_1_1s = []
    mapping_1_4s = []
    sketch_tsdfs = []

    for idx, input_dict in enumerate(batch):
        CP_mega_matrices.append(torch.from_numpy(input_dict['CP_mega_matrix']))
#        mega_contexts.append(torch.from_numpy(input_dict['mega_context']))

#        k_indices.append(torch.from_numpy(input_dict['k_indices']))
        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))
#            else:
#                data.pop(key, None)

        if 'occ_1_1' in input_dict:
            occ_1_1s.append(torch.from_numpy(input_dict['occ_1_1']))
        if 'occ_1_4' in input_dict:
            occ_1_4s.append(torch.from_numpy(input_dict['occ_1_4']))

        sketch_original_mappings.append(torch.from_numpy(input_dict['sketch_original_mapping']))
        sketchs.append(torch.from_numpy(input_dict['sketch']))
        mapping_1_1s.append(torch.from_numpy(input_dict['mapping_1_1']))
        mapping_1_4s.append(torch.from_numpy(input_dict['mapping_1_4']))
        tsdf_1_1s.append(torch.from_numpy(input_dict['tsdf_1_1']))
        tsdf_1_4s.append(torch.from_numpy(input_dict['tsdf_1_4']))

        cam_ks.append(torch.from_numpy(input_dict['cam_k']).double())
        cam_poses.append(torch.from_numpy(input_dict['cam_pose']).float())
        vox_origins.append(torch.from_numpy(input_dict['voxel_origin']).double())

        nonempties.append(torch.from_numpy(input_dict['nonempty']))
        depths.append(torch.from_numpy(input_dict['depth']))
        names.append(input_dict['name'])

        img = input_dict['img']
        imgs.append(img)

#        ssc_label_1_1 = torch.from_numpy(input_dict['target_1_1'])
#        ssc_label_1_1s.append(ssc_label_1_1)
        class_proportion_1_4 = torch.from_numpy(input_dict['class_proportion_1_4'])
        class_proportion_1_4s.append(class_proportion_1_4)

        ssc_label_1_4 = torch.from_numpy(input_dict['target_1_4'])
        ssc_label_1_4s.append(ssc_label_1_4)
#        ssc_label_1_8 = torch.from_numpy(input_dict['target_1_8'])
#        ssc_label_1_8s.append(ssc_label_1_8)
        ssc_label_1_16 = torch.from_numpy(input_dict['target_1_16'])
        ssc_label_1_16s.append(ssc_label_1_16)

    ret_data = {
        "CP_mega_matrices": CP_mega_matrices,

        "sketch_1_4": torch.stack(sketchs),
        "sketch_original_mapping": torch.stack(sketch_original_mappings),
        "mapping_1_1": torch.stack(mapping_1_1s),
        "mapping_1_4": torch.stack(mapping_1_4s),
        "tsdf_1_1": torch.stack(tsdf_1_1s),
        "tsdf_1_4": torch.stack(tsdf_1_4s),

        "cam_pose": torch.stack(cam_poses),
        "cam_k": torch.stack(cam_ks),
        "vox_origin": torch.stack(vox_origins),

        "nonempty": torch.stack(nonempties),
        "depth": torch.stack(depths),
        "name": names,
        "img": torch.stack(imgs),
        "ssc_label_1_4": torch.stack(ssc_label_1_4s),
#        "ssc_label_1_8": torch.stack(ssc_label_1_8s),
        "ssc_label_1_16": torch.stack(ssc_label_1_16s),
        "class_proportion_1_4": torch.stack(class_proportion_1_4s)
    }
    if len(occ_1_1s) > 0:
        ret_data['occ_1_1'] = torch.stack(occ_1_1s)
    if len(occ_1_4s) > 0:
        ret_data['occ_1_4'] = torch.stack(occ_1_4s)
    for key in data:
        ret_data[key] = data[key]
    return ret_data
