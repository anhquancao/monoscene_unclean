from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.data.semantic_kitti.params import semantic_kitti_class_frequencies, kitti_class_names, class_weights as kitti_class_weights
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, NYU_class_names, invfreq_class_relation_weights as NYU_invfreq_class_relation_weights
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.common.utils.torch_util import worker_init_fn
from xmuda.data.semantic_kitti.collate import collate_fn
from torch.utils.data.dataloader import DataLoader
from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.models.SSC2d_LMSCNetProj import SSC2dLMSCNetProj
from xmuda.models.SSC2d_corenet_heavy import SSC2dCorenetHeavy
#from xmuda.models.SSC2d_v2 import SSC2dProj3d2d
from xmuda.models.ssc_loss import get_class_weights
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

seed_everything(42)
hydra.output_subdir = None

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

@hydra.main(config_name="config/project3d2d.yaml")
def main(config: DictConfig):
    preprocess_dir = config.preprocess_dir

    exp_name = config.exp_prefix 
    exp_name += "_{}_{}".format(config.dataset, config.run)
    exp_name += "_FrusSize_{}".format(config.frustum_size)
    exp_name += "_nRelations{}".format(config.n_relations)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)
    exp_name += "_optimizeIoU{}_virtualImg{}".format(config.optimize_iou, config.virtual_img)

    if config.CE_ssc_loss:
        exp_name += "_CEssc"
    if config.MCA_ssc_loss:
        exp_name += "_MCAssc"
    if config.class_proportion_loss:
        exp_name += "_ProportionLoss"

    if config.CE_relation_loss:
        exp_name += "_CERel"
    if config.context_prior is not None:
        exp_name += "_{}".format(config.context_prior)


    if config.dataset == "kitti":
        class_names = kitti_class_names
        max_epochs = 30
        eval_every = 1
        logdir=config.kitti_logdir
        full_scene_size = (256, 256, 32)
        output_scene_size = (int(full_scene_size[0]/(config.project_scale/2)),
                             int(full_scene_size[1]/(config.project_scale/2)),
                             int(full_scene_size[2]/(config.project_scale/2)))
        project_scale = config.project_scale 
        if project_scale == 4:
#            f = 50
            f = 50
#            f = 96
        else:
            f = 32
        features = [f]
#        features = [32, f*2, f*4, f*8, f*16]
#        f = 16
#        features = [f, f*2, f*4, f*8, f*16]
        ###########
        scene_sizes = [
            tuple(np.ceil(i / 1).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 2).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 4).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 8).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 16).astype(int) for i in full_scene_size)
        ]
        n_classes=20
#        class_weights = kitti_class_weights
#        class_weights = kitti_class_weights
        epsilon_w = 0.001  # eps to avoid zero division
        class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05, 6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05, 2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07, 2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08, 2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
        class_weights = torch.from_numpy(1 / np.log(class_frequencies + epsilon_w))
        data_module = KittiDataModule(root=config.kitti_root,
                                      preprocess_root=config.kitti_preprocess_root,
                                      frustum_size=config.frustum_size,
                                      project_scale=project_scale,
                                      n_relations=config.n_relations,
                                      batch_size=int(config.batch_size / config.n_gpus), 
                                      num_workers=int(config.num_workers_per_gpu * config.n_gpus))

    elif config.dataset == "NYU":
        class_names = NYU_class_names 
        max_epochs = 30
        eval_every = 1
        logdir=config.logdir
        full_scene_size = (240, 144, 240)
        output_scene_size = (60, 36, 60)
        project_scale = 4
#        features = [128, 256, 512]
        f = 200
#        f = 32
        features = [f, f * 2, f * 4]
        scene_sizes = [
            tuple(np.ceil(i / 4).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 8).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 16).astype(int) for i in full_scene_size)
        ]
        n_classes=12
        class_weights = NYU_class_weights
#        pred_depth_dir = "/gpfsscratch/rech/xqt/uyl37fq/NYU_pred_depth"
        data_module = NYUDataModule(config.NYU_root,
                                    config.NYU_preprocess_dir,
                                    n_relations=config.n_relations,
                                    frustum_size=config.frustum_size,
                                    corenet_proj=config.corenet_proj,
                                    data_aug=True,
                                    batch_size=int(config.batch_size / config.n_gpus),
                                    num_workers=int(config.num_workers_per_gpu * config.n_gpus))


    project_res = ['1']
    if config.project_1_2:
        exp_name += "_Proj_2"
        project_res.append('2')
    if config.project_1_4:
        exp_name += "_4"
        project_res.append('4')
    if config.project_1_8:
        exp_name += "_8"
        project_res.append('8')
    if config.project_1_16:
        exp_name += "_16"
        project_res.append('16')

    print(exp_name)
    print(class_weights)
    if config.corenet_proj == "heavy":
        Model = SSC2dCorenetHeavy
    elif config.corenet_proj == "lmscnet":
        Model = SSC2dLMSCNetProj
    else:
        Model = SSC2dProj3d2d

    model = Model(preprocess_dir,
                          dataset=config.dataset,
                          corenet_proj=config.corenet_proj,
                          frustum_size=config.frustum_size,
                          lovasz=config.lovasz,
                          optimize_iou=config.optimize_iou,
                          project_scale=project_scale,
                          n_relations=config.n_relations,
                          scene_sizes=scene_sizes,
                          class_proportion_loss=config.class_proportion_loss,
                          features=features,
                          full_scene_size=full_scene_size,
                          output_scene_size=output_scene_size,
                          project_res=project_res,
                          n_classes=n_classes,
                          class_names=class_names,
                          rgb_encoder=config.rgb_encoder,
                          context_prior=config.context_prior,
                          CE_relation_loss=config.CE_relation_loss,
                          MCA_relation_loss=config.MCA_relation_loss,
                          MCA_ssc_loss_type=config.MCA_ssc_loss_type,
                          CE_ssc_loss=config.CE_ssc_loss,
                          MCA_ssc_loss=config.MCA_ssc_loss,
                          lr=config.lr,
                          weight_decay=config.weight_decay,
                          class_weights=class_weights)


    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir,
                                   name=exp_name,
                                   version=''
                                   )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callbacks = [ModelCheckpoint(save_last=True,
                                                monitor='val/mIoU',
                                                save_top_k=1,
                                                mode='max',
                                                filename='{epoch:03d}-{val/mIoU:.5f}'),
                               lr_monitor]
#                                lr_monitor]
    else:
        logger = False
        checkpoint_callbacks = False


    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        trainer = Trainer(callbacks=checkpoint_callbacks,
                          resume_from_checkpoint=model_path,
                          sync_batchnorm=True,
                          deterministic=True,
                          max_epochs=max_epochs, gpus=config.n_gpus, logger=logger,
                          check_val_every_n_epoch=eval_every, log_every_n_steps=10,
                          flush_logs_every_n_steps=100, accelerator='ddp')
    else:
        trainer = Trainer(callbacks=checkpoint_callbacks,
                          sync_batchnorm=True,
                          deterministic=True,
                          max_epochs=max_epochs, gpus=config.n_gpus, logger=logger,
                          check_val_every_n_epoch=eval_every, log_every_n_steps=10,
                          flush_logs_every_n_steps=100, accelerator='ddp')

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
