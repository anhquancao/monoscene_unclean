from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.data.semantic_kitti.params import semantic_kitti_class_frequencies, kitti_class_names, class_weights as kitti_class_weights
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, NYU_class_names
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.models.AIC_trainer import AICTrainer
from xmuda.common.utils.torch_util import worker_init_fn
from torch.utils.data.dataloader import DataLoader
from xmuda.models.SketchTrainer import SketchTrainer
from xmuda.models.LMSC_trainer import LMSCTrainer
from xmuda.models.ssc_loss import get_class_weights
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from xmuda.auto_requeue import init_signal_handler
from time import time

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
seed_everything(42)
hydra.output_subdir = None

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

@hydra.main(config_name="config/baseline.yaml")
def main(config: DictConfig):
#    init_signal_handler()
    exp_name = config.exp_prefix
    exp_name += "_{}_{}".format(config.method, config.dataset)
    exp_name += "_PredDepth{}".format(config.use_predicted_depth)
    exp_name += "_OptimizeEverywhere{}".format(config.optimize_everywhere)
    exp_name += "_3DSketchNonempty{}".format(config.use_3DSketch_nonempty_mask)
    exp_name += "_EmptyFromDepth{}".format(config.predict_empty_from_depth)

    print(exp_name)

    if config.dataset == "kitti":
        class_names = kitti_class_names
        logdir=config.kitti_logdir
        max_epochs=100
        output_scene_size = (256, 256, 32) 
        full_scene_size = (256, 256, 32)
        n_classes=20
        class_weights = kitti_class_weights
#        class_weights = {
##            '1_1' : get_class_weights(semantic_kitti_class_frequencies)
#            '1_1' : kitti_class_weights
#        }
        class_relation_weights = NYU_class_relation_weights # TODO: tune this one latter 
        data_module = KittiDataModule(root=config.kitti_root,
                                      preprocess_root=config.kitti_preprocess_root,
                                      use_predicted_depth=config.use_predicted_depth,
                                      frustum_size=1, # does not matter here
                                      project_scale=2, # does not matter here                
                                      n_relations=1, # does not matter here
                                      batch_size=int(config.batch_size / config.n_gpus), 
                                      num_workers=int(config.num_workers_per_gpu * config.n_gpus))
    elif config.dataset == "NYU":
        class_names = NYU_class_names 
        logdir=config.logdir
        max_epochs=250
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
    data_module.setup()

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
                                                filename='{epoch:02d}-{val/mIoU:.3f}'), 
                                lr_monitor]
    else:
        logger = False
        checkpoint_callbacks = False


    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    print("checkpoint", model_path)
    if not os.path.isfile(model_path):
        model_path = None

    if config.method == "LMSCNet":
        if config.dataset == "NYU":
            in_channels = 144
        elif config.dataset == "kitti":
            in_channels = 32
        model = LMSCTrainer(n_classes=n_classes,
#                            class_weights=class_weights,
                            full_scene_size=full_scene_size,
                            output_scene_size=output_scene_size,
                            dataset=config.dataset,
                            class_names=class_names, 
                            in_channels=in_channels)

    elif config.method == "3DSketch":
        model = SketchTrainer(predict_empty_from_depth=config.predict_empty_from_depth,
                              n_training_items=len(data_module.train_ds),
                              optimize_everywhere=config.optimize_everywhere,
                              full_scene_size=full_scene_size,
                              output_scene_size=output_scene_size,
                              use_3DSketch_nonempty_mask=config.use_3DSketch_nonempty_mask,
                              n_classes=n_classes,
                              class_names=class_names,
                              class_weights=class_weights)

    elif config.method == "AICNet":
        model = AICTrainer(n_classes=n_classes,
                           output_scene_size=output_scene_size,
                           full_scene_size=full_scene_size,
                           class_names=NYU_class_names,
                           class_weights=class_weights)

    trainer = Trainer(callbacks=checkpoint_callbacks,
                      resume_from_checkpoint=model_path,
                      sync_batchnorm=True,
                      deterministic=True,
                      max_epochs=max_epochs, gpus=config.n_gpus, logger=logger,
                      check_val_every_n_epoch=1, log_every_n_steps=10,
                      flush_logs_every_n_steps=100, accelerator='ddp')

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
