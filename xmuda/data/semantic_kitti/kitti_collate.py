import torch
from functools import partial
import numpy as np


def collate_fn(batch):
    data = {}
    imgs = []
    depths = []
#    sketch_1_1s = []
    sketch_1_4s = []
    tsdf_1_1s = []
    tsdf_1_4s = []
    targets = []
#    nonempties = []
    occ_1_1s = []
    mapping_1_1s = []
    mapping_1_4s = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    local_frustums = []
    local_frustums_cnt = []

#    scales = [1, 2, 4, 8, 16]
#    scales = [1, 2]
#    scales = [1]
    scales = batch[0]['scales']
    for scale in scales: 
        data['pix_' + str(scale)] = []
        data['valid_pix_' + str(scale)] = []
        data['valid_pix_double'] = []
#        data['pts_cam_' + str(scale)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict['cam_k']).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict['T_velo_2_cam']).float())

        if 'local_frustums' in input_dict:
            local_frustums_cnt.append(torch.from_numpy(input_dict['local_frustums_cnt']).float())
    #        local_frustums.append(torch.from_numpy(input_dict['local_frustums']).float())
    #        local_frustums.append(torch.from_numpy(input_dict['local_frustums']).bool())
            local_frustums.append(torch.from_numpy(input_dict['local_frustums']))

        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))

        if 'mapping_1_1' in input_dict:
            mapping_1_1s.append(torch.from_numpy(input_dict['mapping_1_1']))
        if 'mapping_1_4' in input_dict:
            mapping_1_4s.append(torch.from_numpy(input_dict['mapping_1_4']))

        img = input_dict['img']
        imgs.append(img)

        frame_ids.append(input_dict['frame_id'])
        sequences.append(input_dict['sequence'])

        occ_1_1s.append(torch.from_numpy(input_dict['occ_1_1']))

        tsdf_1_1s.append(torch.from_numpy(input_dict['tsdf_1_1']))
        tsdf_1_4s.append(torch.from_numpy(input_dict['tsdf_1_4']))

        depths.append(torch.from_numpy(input_dict['depth']))

        if 'target' in input_dict:
            targets.append(torch.from_numpy(input_dict['target']))
            CP_mega_matrices.append(torch.from_numpy(input_dict['CP_mega_matrix']))
#        nonempties.append(torch.from_numpy(input_dict['nonempty']))

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,

        "local_frustums_cnt": local_frustums_cnt,
        "local_frustums": local_frustums,

        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,

        "img": torch.stack(imgs),
        "depth": torch.stack(depths),
        "occ_1_1": torch.stack(occ_1_1s),
        "tsdf_1_1": torch.stack(tsdf_1_1s),
        "tsdf_1_4": torch.stack(tsdf_1_4s),
#            "nonempty": torch.stack(nonempties)
    }
    if 'target' in input_dict:
        ret_data['target'] = torch.stack(targets)                
        ret_data["CP_mega_matrices"] = torch.stack(CP_mega_matrices)

        sketch_1_4s.append(torch.from_numpy(input_dict['sketch_1_4']))

        ret_data["sketch_1_4"] = torch.stack(sketch_1_4s)
        ret_data["target"] = torch.stack(targets)


    if len(mapping_1_1s) > 0:
        ret_data["mapping_1_1"] = torch.stack(mapping_1_1s)
    if len(mapping_1_4s) > 0:
        ret_data["mapping_1_4"] = torch.stack(mapping_1_4s)

    for key in data:
        ret_data[key] = data[key]
    return ret_data
