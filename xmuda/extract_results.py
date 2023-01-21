from xmuda.models.quan_trainer_lmsc import RecNetLMSC
from xmuda.models.quan_trainer import RecNet
# from xmuda.models.quan_trainer_seg import RecNetSeg
from xmuda.models.SSC2d import SSC2d
from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
from torch.utils.data.dataloader import DataLoader
import pickle
from xmuda.data.semantic_kitti.collate import collate_fn
from tqdm import tqdm

# lmsc_model = RecNetLMSC.load_from_checkpoint("/home/docker_user/workspace/xmuda/tb_logs/lmsc_1_4/version_0/checkpoints/checkpoint-epoch=99-step=90399.ckpt")
# lmsc_model.cuda()
# lmsc_model.eval()

#preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
semantic_kitti_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti'
#semantic_kitti_dir = '/datasets_master/semantic_kitti'
preprocess_dir = semantic_kitti_dir + '/preprocess/preprocess'

model = SSC2d.load_from_checkpoint("/gpfswork/rech/xqt/uyl37fq/code/xmuda-extend/tb_logs/last_layer,dim_feat_depth=16/version_0/checkpoints/checkpoint-epoch=59-step=57479.ckpt", num_depth_classes=16, preprocess_dir=preprocess_dir)
model.cuda()
model.eval()


val_dataset = SemanticKITTISCN(split=('val',),
                               preprocess_dir=preprocess_dir,
                               semantic_kitti_dir=semantic_kitti_dir,                               
                               noisy_rot=0,
                               flip_y=0,
                               rot_z=0,
                               transl=False,
                               bottom_crop=None,
                               fliplr=0,
                               color_jitter=None,
                            #    normalize_image=False
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
    pred = model(batch)
    # lmsc_pred = lmsc_model(batch)    
    ssc = pred['ssc_logit'].detach().cpu().numpy()
    img = batch['img'].detach().cpu().numpy()
    data = {
        "pred_ssc": ssc,
        "img": img,
        # "lmsc_pred": lmsc_pred.detach().cpu().numpy(),
        "gt": batch['ssc_label_1_4'].numpy()
    }    
    
    with open('/gpfswork/rech/xqt/uyl37fq/code/xmuda-extend/tb_logs/draws/val_pred_' + str(i) + '.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
