from xmuda.models.quan_trainer_lmsc import RecNetLMSC
from xmuda.models.quan_trainer import RecNet
from xmuda.models.quan_trainer_seg import RecNetSeg
from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
from torch.utils.data.dataloader import DataLoader
import pickle
from xmuda.data.semantic_kitti.collate import collate_fn
from tqdm import tqdm
import torch
import numpy as np

# lmsc_model = RecNetLMSC.load_from_checkpoint("/home/docker_user/workspace/xmuda/tb_logs/depth/version_0/checkpoints/checkpoint-epoch=11-step=10835.ckpt")
# lmsc_model.cuda()
# lmsc_model.eval()

# quan_model = RecNet.load_from_checkpoint("/home/docker_user/workspace/xmuda/tb_logs/input_point_in_front_data_aug/version_0/checkpoints/checkpoint-epoch=59-step=54179.ckpt")
# quan_model.cuda()
# quan_model.eval()

quan_model = RecNetSeg.load_from_checkpoint("/home/docker_user/workspace/xmuda/tb_logs/depth/version_0/checkpoints/checkpoint-epoch=11-step=10835.ckpt")
quan_model.cuda()
quan_model.eval()

preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
semantic_kitti_dir = '/datasets_master/semantic_kitti'
val_dataset = SemanticKITTISCN(split=('val',),
                               preprocess_dir=preprocess_dir,
                               semantic_kitti_dir=semantic_kitti_dir,                               
                               noisy_rot=0,
                               flip_y=0,
                               rot_z=0,
                               transl=False,
                               bottom_crop=None,
                               fliplr=0,
                               color_jitter=None
                               )

val_dataloader = DataLoader(
    val_dataset,
    batch_size=4, 
    drop_last=False,
    num_workers=6,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn
)


for i, batch in tqdm(enumerate(val_dataloader)):    
    depth_logit = quan_model(batch)['depth_logit']
    depth = torch.argmax(depth_logit, dim=1) * 0.2
    img_indices = batch['img_indices']
    coords_2d = batch['coords_2d']
    
    depth_logit = depth_logit[coords_2d[:, 3] == 0].detach().cpu().numpy()
    img_indices = img_indices[0].detach().cpu().numpy()

    print(depth_logit.shape, img_indices.shape)
    
    homo_points_img = np.hstack([img_indices, np.ones([img_indices.shape[0], 1])])            
    T_velo_2_cam = batch['T_velo_2_cam'][0][0].detach().cpu().numpy()
    K_inv = batch['K_inv'][0][0].detach().cpu().numpy()
    T_velo_2_cam_inv = np.linalg.inv(T_velo_2_cam)
    print(T_velo_2_cam)
    print(K_inv)
    print(batch['proj_matrix'][0])
    depth = depth.detach().cpu().numpy()
    t =  K_inv @ (homo_points_img * depth).T            
    t = t.T
    t_homo = np.hstack([t, np.ones([t.shape[0], 1])])        
    t_homo = T_velo_2_cam_inv @ t_homo.T
    t_homo = t_homo.T[:, 0:3]  
    # lmsc_pred = lmsc_model(batch)    
    # data = {
    #     "quan_pred": quan_pred.detach().cpu().numpy(),
    #     "lmsc_pred": lmsc_pred.detach().cpu().numpy(),
    #     "gt": batch['ssc_label_1_4'].numpy()
    # }    
    
    # with open('results/val_pred_' + str(i) + '.pkl', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    break