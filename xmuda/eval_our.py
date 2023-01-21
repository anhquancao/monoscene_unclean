from pytorch_lightning import Trainer, seed_everything
from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, NYU_class_names
from xmuda.data.semantic_kitti.params import kitti_class_names, semantic_kitti_class_frequencies, class_weights as kitti_class_weights
from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.models.SSC2d_LMSCNetProj import SSC2dLMSCNetProj
from xmuda.models.SSC2d_corenet_heavy import SSC2dCorenetHeavy
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from xmuda.models.ssc_loss import get_class_weights
import numpy as np

seed_everything(42)

@hydra.main(config_name="config/project3d2d.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    if config.dataset == "kitti":
        config.batch_size = 1
        class_names = kitti_class_names
        logdir=config.kitti_logdir
        full_scene_size = (256, 256, 32)
        output_scene_size = (int(full_scene_size[0]/(config.project_scale/2)),
                             int(full_scene_size[1]/(config.project_scale/2)),
                             int(full_scene_size[2]/(config.project_scale/2)))
#        features = [8, 16, 32, 64, 128]
        project_scale = config.project_scale
        if project_scale == 4:
            f = 50
        else:
            f = 32
        features = [f]
#        f = 50
#        features = [32, f*2, f*4, f*8, f*16]
        scene_sizes = [
            tuple(np.ceil(i / 1).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 2).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 4).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 8).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 16).astype(int) for i in full_scene_size)
        ]
        n_classes=20
        class_weights = kitti_class_weights
        data_module = KittiDataModule(root=config.kitti_root,
                                      preprocess_root=config.kitti_preprocess_root,
                                      frustum_size=config.frustum_size,
                                      n_relations=config.n_relations,
                                      project_scale=project_scale,
                                      batch_size=int(config.batch_size / config.n_gpus), 
                                      num_workers=int(config.num_workers_per_gpu * config.n_gpus))

    elif config.dataset == "NYU":
        config.batch_size = 2
        class_names = NYU_class_names 
        logdir=config.logdir
        full_scene_size = (240, 144, 240)
        output_scene_size = (60, 36, 60)
        net_2d_num_features = 2048
        features = [128, 256, 512]
        scene_sizes = [
            tuple(np.ceil(i / 4).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 8).astype(int) for i in full_scene_size),
            tuple(np.ceil(i / 16).astype(int) for i in full_scene_size)
        ]
        n_classes=12
#        class_weights = NYU_class_weights 
        class_weights = NYU_class_weights
        project_scale = 4
#        pred_depth_dir = "/gpfsscratch/rech/xqt/uyl37fq/NYU_pred_depth"
        data_module = NYUDataModule(config.NYU_root,
                                    config.NYU_preprocess_dir,
                                    frustum_size=config.frustum_size,
                                    corenet_proj=config.corenet_proj,
                                    n_relations=config.n_relations,
                                    data_aug=True,
                                    batch_size=int(config.batch_size / config.n_gpus),
                                    num_workers=int(config.num_workers_per_gpu * config.n_gpus))

    trainer = Trainer(sync_batchnorm=True,
                      deterministic=True,
                     gpus=config.n_gpus, accelerator='ddp')
#    model_path_1 = "/gpfswork/rech/kvd/uyl37fq/logs/NYU/CorenetTune_NYU_1_FrusSize_8_nRelations4_WD1e-05_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_Proj_2_4/checkpoints/epoch=022-val/mIoU=0.18759.ckpt"
#    model_path_2 = "/gpfswork/rech/kvd/uyl37fq/logs/NYU/CorenetTune_NYU_2_FrusSize_8_nRelations4_WD1e-05_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_Proj_2_4/checkpoints/epoch=020-val/mIoU=0.18223.ckpt"
#    model_path_3 = "/gpfswork/rech/kvd/uyl37fq/logs/NYU/CorenetTune_NYU_3_FrusSize_8_nRelations4_WD1e-05_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_Proj_2_4/checkpoints/epoch=020-val/mIoU=0.18702.ckpt"

#    model_path_1 = "/gpfswork/rech/kvd/uyl37fq/logs/NYU/CorenetRerun_NYU_2_FrusSize_8_nRelations4_WD0.001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel/checkpoints/epoch=017-val/mIoU=0.17065.ckpt"
#    model_path_2 = "/gpfswork/rech/kvd/uyl37fq/logs/NYU/CorenetRerun_NYU_1_FrusSize_8_nRelations4_WD0.001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel/checkpoints/epoch=014-val/mIoU=0.17496.ckpt"
#    model_path_3 = "/gpfswork/rech/kvd/uyl37fq/logs/NYU/CorenetRerun_NYU_3_FrusSize_8_nRelations4_WD0.001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel/checkpoints/epoch=014-val/mIoU=0.17396.ckpt"

    # model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_nRelations/Group8_kitti_1_FrusSize_8_nRelations8_WD0_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=025-val/mIoU=0.11297.ckpt"
    # model_path_2 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_nRelations/Group8_kitti_2_FrusSize_8_nRelations8_WD0_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.11324.ckpt"
    # model_path_3 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_nRelations/Group8_kitti_3_FrusSize_8_nRelations8_WD0_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.11286.ckpt"

    model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_rerun/v2_NoRelLoss_NYU_1_FrusSize_8_nRelations8_WD0.0001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CRCP_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.26747.ckpt"
    model_path_2 = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_rerun/MoreConv_add16_NYU_3_FrusSize_8_nRelations8_WD0.0001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.26023.ckpt"
    model_path_3 = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_rerun/v2_NoRelLoss_NYU_3_FrusSize_8_nRelations8_WD0.0001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CRCP_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.26459.ckpt"

    # model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_rerun/RelSupervision_NYU_1_FrusSize_8_nRelations16_WD0.0001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=021-val/mIoU=0.26948.ckpt"
    # model_path_2 = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_rerun/RelSupervision_NYU_2_FrusSize_8_nRelations16_WD0.0001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=021-val/mIoU=0.26552.ckpt"
    # model_path_3 = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_rerun/RelSupervision_NYU_3_FrusSize_8_nRelations16_WD0.0001_lr0.0001_optimizeIoUTrue_virtualImgFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.26338.ckpt"

#    model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_1_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.11444.ckpt"
#    model_path_2 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=029-val/mIoU=0.11596.ckpt"
#    model_path_2_save = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=029-val/mIoU=0.11596_test.ckpt"
#    model_path_3 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_3_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=026-val/mIoU=0.11318.ckpt"

    # model_paths = [model_path_1, model_path_2, model_path_3]
    # model_paths = [model_path_2, model_path_3]
    model_paths = [model_path_2]
#    model_paths = [model_path_1]
    if config.corenet_proj == "heavy":
        Model = SSC2dCorenetHeavy
    elif config.corenet_proj == "lmscnet":
        Model = SSC2dLMSCNetProj
    else:
        Model = SSC2dProj3d2d
    for model_path in model_paths:
#        ckpt = torch.load(model_path)
#        for key in ckpt:
#            if key != "state_dict":
#                print(key, ckpt[key])
#        ckpt['state_dict'].pop("net_3d_decoder.CP_mega_voxels.mega_context_logit.0.weight")
#        ckpt['state_dict'].pop("net_3d_decoder.CP_mega_voxels.mega_context_logit.0.bias")
#        torch.save(ckpt, model_path_2_save) 
#        print(list(ckpt['state_dict'].keys()))
        model = Model.load_from_checkpoint(model_path,
                                           save_data_for_submission=False, 
                                           corenet_proj=config.corenet_proj, 
                                           project_scale=project_scale)
        model.eval()
#        print(model_path)
#        trainer.test(model, datamodule=data_module)
        data_module.setup()
        val_dataloader = data_module.val_dataloader()
        trainer.test(model, test_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
