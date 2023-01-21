from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.data.semantic_kitti.params import semantic_kitti_class_frequencies, kitti_class_names, class_weights as kitti_class_weights
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, NYU_class_names
import numpy as np
import torch
import torch.nn.functional as F
from xmuda.models.ssc_loss import get_class_weights
from tqdm import tqdm
import pickle
import os


# model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_new/v22_full_NYU_3_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.27023.ckpt"
#model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.10886.ckpt"
model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=029-val/mIoU=0.11596.ckpt"

class_weights = {
    '1_4': get_class_weights(NYU_class_freq_1_4).cuda(),
    '1_8': get_class_weights(NYU_class_freq_1_8).cuda(),
    '1_16': get_class_weights(NYU_class_freq_1_16).cuda(),
}


# dataset = "NYU"
dataset = "kitti"
if dataset == "NYU":
    AIC_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines/baseline_1_1_1divlogLabelWeights_FixOptimizer_AICNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=197-val/mIoU=0.175.ckpt"
    sketch_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines/baseline_1_1_1divlogLabelWeights_FixOptimizer_3DSketch_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=241-val/mIoU=0.226.ckpt"
    lmscnet_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=48-val/mIoU=0.157.ckpt"
    lmsc_channels = 144
    NYU_root = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
    NYU_preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
    full_scene_size = (240, 144, 240)
    output_scene_size = (60, 36, 60)
    class_names = NYU_class_names 
    class_weights = NYU_class_weights

    NYUdm = NYUDataModule(NYU_root, NYU_preprocess_dir, batch_size=4, 
                          use_predicted_depth=False,
                          num_workers=3)
    NYUdm.setup()
    n_classes = 12
    data_loader = NYUdm.val_dataloader()

    class_relation_weights = get_class_weights(NYU_class_relation_freqs)

else:
    AIC_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_kitti/baseline_1_1_AICNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=64-val/mIoU=0.081.ckpt"
    sketch_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_kitti/baseline_1_1_3DSketch_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=66-val/mIoU=0.074.ckpt"
    lmscnet_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_kitti/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=84-val/mIoU=0.082.ckpt"
    lmsc_channels = 32
    kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
    preprocess_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti"

    kitti_depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
        
    full_scene_size = (256, 256, 32)
    output_scene_size = full_scene_size
    class_names = kitti_class_names
    epsilon_w = 0.001  # eps to avoid zero division
    class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05, 6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05, 2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07, 2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08, 2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
    class_weights = torch.from_numpy(1 / np.log(class_frequencies + epsilon_w))
    KITTIdm = KittiDataModule(root=kitti_root,
                              preprocess_root=preprocess_root,
                              frustum_size=1,
                              project_scale=2,
                              n_relations=1,
                              batch_size=1, 
                              num_workers=10)
    KITTIdm.setup()
    n_classes = 20
    data_loader = KITTIdm.val_dataloader()
    # data_loader = KITTIdm.train_dataloader()

# train_dataloader         output_scene_size=output_scene_size,
#                                         class_weights=class_weights,
#                                         class_names=class_names)
#aicnet.cuda()
#aicnet.eval()
#sketch = SketchTrainer.load_from_checkpoint(sketch_path, 
#                                          n_classes=n_classes, 
#                                          dataset=dataset, 
#                                          full_scene_size=full_scene_size, 
#                                          output_scene_size=output_scene_size, 
#                                          class_names=class_names)
#sketch.cuda()
#sketch.eval()
#lmscnet = LMSCTrainer.load_from_checkpoint(lmscnet_path, 
#                                           n_classes=n_classes, 
#                                           dataset=dataset, 
#                                           in_channels=lmsc_channels,
#                                           full_scene_size=full_scene_size, 
#                                           output_scene_size=output_scene_size, 
#                                           class_names=class_names)
#lmscnet.cuda()
#lmscnet.eval()
#others = [
#    lmscnet, aicnet, sketch
#]


our = SSC2dProj3d2d.load_from_checkpoint(model_path)
our.cuda()
our.eval()

count = 0

count = 0
write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/features/{}".format(dataset)
cnt = 0


with torch.no_grad():
    for batch in tqdm(data_loader):   
        out_features = []
        out_labels = []     

        frame_id = batch['frame_id'][0] # 1 batch only
        sequence = batch['sequence'][0]

        out_dir = os.path.join(write_path, sequence)
        os.makedirs(out_dir, exist_ok=True)
        
        filepath = os.path.join(out_dir, "{}.npy".format(frame_id))     
        if os.path.exists(filepath):
            print(filepath, " existed")
            continue

        y_true = batch['target'].detach().cpu().numpy()            

        for key in ['img', 'mapping_1_1','depth', 'mapping_1_4', 'tsdf_1_4']:
            batch[key] = batch[key].cuda()
        pred = np.argmax(our(batch)['ssc'].detach().cpu().numpy(), axis=1)
        features = our(batch)['features'].detach().cpu().numpy()
        for i in range(y_true.shape[0]):
            frame_id = batch['frame_id'][i]
            sequence = batch['sequence'][i]
            our_pred_i = pred[i].astype(np.uint16)
            # y_true_i = y_true[i].astype(np.uint16)            
            # correct_voxes = ((our_pred_i == y_true_i) & (y_true_i != 0)).reshape(-1)
            # features_i = features[i].reshape(32, -1)
            # print(correct_voxes.shape, features_i.shape)            
            # correct_features_i = features_i[:, correct_voxes]
            # correct_labels_i = y_true_i.reshape(-1)[correct_voxes]
            # print(correct_features_i.shape, correct_labels_i.shape)
            # out_features.append(correct_features_i)
            # out_labels.append(correct_labels_i)
        # cnt += 1
        # if cnt == 50:
        #     break
            
            
        # out_dict = {
        #     "features": np.concatenate(out_features, axis=1),
        #     "labels": np.concatenate(out_labels, axis=0)
        # }
        # print(out_dict['features'].shape, out_dict['labels'].shape)
        # os.makedirs(write_path, exist_ok=True)
            # with open(filepath, 'wb') as handle:
            print(our_pred_i.shape)
            np.save(filepath, our_pred_i)
                # print(list(out_dict.keys()))
                # pickle.dump(out_dict, handle)
            print("wrote to", filepath)
        
        # cnt += 1

