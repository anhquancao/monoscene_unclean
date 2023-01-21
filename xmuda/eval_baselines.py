from pytorch_lightning import Trainer, seed_everything
from xmuda.models.SketchTrainer import SketchTrainer
from xmuda.models.LMSC_trainer import LMSCTrainer
from xmuda.models.AIC_trainer import AICTrainer
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.models.ssc_loss import get_class_weights
from xmuda.data.semantic_kitti.params import semantic_kitti_class_frequencies, kitti_class_names, class_weights as kitti_class_weights
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, NYU_class_names
import hydra
from omegaconf import DictConfig, OmegaConf

seed_everything(42)

@hydra.main(config_name="config/baseline.yaml")
def main(config: DictConfig):

    if config.dataset == "kitti":
        class_names = kitti_class_names
        logdir=config.kitti_logdir
        full_scene_size = (256, 256, 32)
#        output_scene_size = (256 // 4, 256 // 4, 32 // 4)
        output_scene_size = (256, 256, 32)
        n_classes=20
        class_weights = kitti_class_weights
        data_module = KittiDataModule(root=config.kitti_root,
                                      data_aug=True,
                                      TSDF_root=config.kitti_tsdf_root,
                                      project_scale=2,
                                      depth_root=config.kitti_depth_root,
                                      label_root=config.kitti_label_root,
                                      mapping_root=config.kitti_mapping_root,
                                      occ_root=config.kitti_occ_root,
                                      sketch_root=config.kitti_sketch_root,
                                      batch_size=int(config.batch_size / config.n_gpus), 
                                      num_workers=int(config.num_workers_per_gpu * config.n_gpus))

    elif config.dataset == "NYU":
        class_names = NYU_class_names 
        logdir=config.logdir
        full_scene_size = (240, 144, 240)
        output_scene_size = (60, 36, 60)
        n_classes=12
        class_weights = NYU_class_weights
        data_module = NYUDataModule(config.NYU_root,
                                    config.NYU_preprocess_dir,
                                    data_aug=True,
                                    use_predicted_depth=config.use_predicted_depth,
                                    batch_size=int(config.batch_size / config.n_gpus),
                                    num_workers=int(config.num_workers_per_gpu * config.n_gpus))

    trainer = Trainer(sync_batchnorm=True,
                      deterministic=True,
                      gpus=config.n_gpus, accelerator='ddp')
    if config.method == "3DSketch":
#        model_path = "/gpfswork/rech/kvd/uyl37fq/models/kitti/res_1_1/baseline_1_1_3DSketch_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=76-val/mIoU=0.075.ckpt"
        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_3DSketch_NYU_PredDepthFalse_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=221-val/mIoU=0.292.ckpt"
#        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_1divlogLabelWeights_FixOptimizer_3DSketch_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=241-val/mIoU=0.226.ckpt"
        model = SketchTrainer.load_from_checkpoint(model_path, 
                                                   dataset=config.dataset)

    elif config.method == "AICNet":
        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_1divlogLabelWeights_FixOptimizer_AICNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=197-val/mIoU=0.175.ckpt"
#        model_path = "/gpfswork/rech/kvd/uyl37fq/models/kitti/res_1_1/baseline_1_1_AICNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=99-val/mIoU=0.083.ckpt"
#        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_AICNet_NYU_PredDepthFalse_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=249-val/mIoU=0.238.ckpt"

        model = AICTrainer.load_from_checkpoint(model_path, 
                                                n_classes=n_classes, 
                                                full_scene_size=full_scene_size,
                                                output_scene_size=output_scene_size,
                                                class_names=class_names, 
                                                dataset=config.dataset,
                                                class_weights=class_weights)
    elif config.method == "LMSCNet":
        if config.dataset == "NYU":
            in_channels = 144
        elif config.dataset == "kitti":
            in_channels = 32
        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=48-val/mIoU=0.157.ckpt"
#        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti/baseline_1_1_LMSCNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=63-val/mIoU=0.056.ckpt"
#        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=84-val/mIoU=0.082.ckpt"
#        model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_LMSCNet_NYU_PredDepthFalse_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=43-val/mIoU=0.205.ckpt"


        model = LMSCTrainer.load_from_checkpoint(model_path, 
                                                 n_classes=n_classes,
                                                 dataset=config.dataset,
                                                 full_scene_size=full_scene_size,
                                                 output_scene_size=output_scene_size,
                                                 class_names=class_names,
                                                 in_channels=in_channels)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
