from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.data.kitti_360.kitti_360_dm import Kitti360DataModule
from xmuda.data.nuscenes.nuscenes_dm import NuscenesDataModule
from xmuda.data.cityscapes.cityscapes_dm import CityscapesDataModule
from xmuda.common.utils.sscMetrics import SSCMetrics
import numpy as np
import torch
import torch.nn.functional as F
from xmuda.models.ssc_loss import get_class_weights
from tqdm import tqdm
import pickle
import os


model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=029-val/mIoU=0.11596.ckpt"
our = SSC2dProj3d2d.load_from_checkpoint(model_path)
our.cuda()
our.eval()

dataset = "cityscapes"

if dataset == "kitti_360":
    kitti360_dm = Kitti360DataModule(root="/gpfsdswork/dataset/KITTI-360",
                                 batch_size=1,
                                 num_workers=3)
    kitti360_dm.setup()
    dataloader = kitti360_dm.val_dataloader()

elif dataset == "nuscenes":
    nuscene_dm = NuscenesDataModule(root='/gpfsscratch/rech/kvd/uyl37fq/dataset/nuscenes',
                                    batch_size=1)
    nuscene_dm.setup()
    dataloader = nuscene_dm.val_dataloader()

elif dataset == "cityscapes":
    cityscapes_dm = CityscapesDataModule(root="/gpfsscratch/rech/kvd/uyl37fq/dataset/cityscapes",
                                         batch_size=1)
    cityscapes_dm.setup()
    dataloader = cityscapes_dm.test_dataloader()
else:
    raise "Dataset not supported"


n_classes = 20


count = 0
out_dict = {}
write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/draw_output/{}".format(dataset)
#write_path = "/gpfswork/rech/kvd/uyl37fq/code/temp/draw_output/{}".format(dataset)

with torch.no_grad():
    for batch in tqdm(dataloader):
        valid_pix_1 = batch['valid_pix_1']
        frame_ids = batch['frame_id']
        for key in ['img']:
            batch[key] = batch[key].cuda()
        pred = np.argmax(our(batch)['ssc'].detach().cpu().numpy(), axis=1)
        for i in range(pred.shape[0]):
            classes = np.unique(pred[i])
            classes_in_scene = len(classes)
            out_dict = {
                "our_pred": pred[i].astype(np.uint16),
            }
            if 'img_path' in batch:
                out_dict['img_path'] = batch['img_path'][i]
            filepath = os.path.join(write_path, frame_ids[i] + "_nclasses={}.pkl".format(classes_in_scene))
            out_dict["valid_pix_1"] = batch['valid_pix_1'][i].detach().cpu().numpy()
            out_dict["cam_k"] = batch['cam_k'][i].detach().cpu().numpy()
            out_dict["T_velo_2_cam"] = batch['T_velo_2_cam'][i].detach().cpu().numpy()
            os.makedirs(write_path, exist_ok=True)
            with open(filepath, 'wb') as handle:
                print(list(out_dict.keys()))
                pickle.dump(out_dict, handle)
                print("wrote to", filepath)

