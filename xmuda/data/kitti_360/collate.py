import torch
from functools import partial
import numpy as np


def collate_fn(batch):
    data = {}
    imgs = []
    frame_ids = []
    img_paths = []

    cam_ks = []
    T_velo_2_cams = []

    scales = batch[0]['scales']
    for scale in scales: 
        data['pix_' + str(scale)] = []
        data['valid_pix_' + str(scale)] = []

    for idx, input_dict in enumerate(batch):
        if 'img_path' in input_dict:
            img_paths.append(input_dict['img_path'])

        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict['cam_k']).float())
        T_velo_2_cams.append(torch.from_numpy(input_dict['T_velo_2_cam']).float())


        img = input_dict['img']
        imgs.append(img)

        frame_ids.append(input_dict['frame_id'])

        ret_data = {
            "frame_id": frame_ids,
            "cam_k": cam_ks,
            "T_velo_2_cam": T_velo_2_cams,
            "img": torch.stack(imgs),
            "img_path": img_paths
        }

    for key in data:
        ret_data[key] = data[key]
    return ret_data
