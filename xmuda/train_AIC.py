from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
from xmuda.data.semantic_kitti.semantic_kitti_dm import SemanticKittiDataModule
from xmuda.data.semantic_kitti.params import semantic_kitti_class_frequencies
from xmuda.data.NYU.params import class_weights as NYU_class_weights, NYU_class_names
from xmuda.common.utils.torch_util import worker_init_fn
from xmuda.data.semantic_kitti.collate import collate_fn
from torch.utils.data.dataloader import DataLoader
from xmuda.models.AIC_trainer import AICTrainer
from xmuda.models.ssc_loss import get_class_weights
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xmuda.data.NYU.nyu_dm import NYUDataModule
import os
import hydra
from omegaconf import DictConfig, OmegaConf

hydra.output_subdir = None

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

@hydra.main(config_name="config/sketch.yaml")
def main(config: DictConfig):
    exp_name = "{}".format(config.exp_prefix)
    exp_name += "_{}".format(config.dataset)
    exp_name += "_PredDepth{}".format(config.use_predicted_depth)

    print(exp_name)

    if config.dataset == "semantic_kitti":
        scene_size = (64, 8, 64)
        n_classes=20
        class_weights = get_class_weights(semantic_kitti_class_frequencies)
        data_module = SemanticKittiDataModule(batch_size=config.batch_size, 
                                                 n_pixels_per_voxel=64,
                                                 num_workers=config.num_workers)
    elif config.dataset == "NYU":
        class_names = NYU_class_names 
        logdir=config.logdir
        full_scene_size = (240, 144, 240)
        n_classes=12
#        class_weights = NYU_class_weights 
        class_weights = {
            '1_4': NYU_class_weights#.cuda(),
#            '1_4': get_class_weights(NYU_class_freq_1_4),#.cuda(),
#            '1_8': get_class_weights(NYU_class_freq_1_8),#.cuda(),
#            '1_16': get_class_weights(NYU_class_freq_1_16)#.cuda(),
        }
        data_module = NYUDataModule(config.NYU_root,
                                    config.NYU_preprocess_dir,
                                    data_aug=True,
                                    use_predicted_depth=config.use_predicted_depth,
                                    batch_size=int(config.batch_size / config.n_gpus),
                                    num_workers=int(config.num_workers_per_gpu * config.n_gpus))

    
    model = AICTrainer(n_classes=n_classes,
                       class_names=NYU_class_names,
                       class_weights=class_weights)

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=config.logdir,
                                   name=exp_name,
                                   version=''
                                   )

        checkpoint_callbacks = [ModelCheckpoint(save_last=True,
                                                monitor='val/mIoU',
                                                save_top_k=1,
                                                mode='max',
                                                filename='{epoch:03d}-{val_nonempty/mIoU:.5f}')]
    else:
        logger = False
        checkpoint_callbacks = False


    model_path = os.path.join(config.logdir, exp_name, "checkpoints/last.ckpt")
    if not os.path.isfile(model_path):
        model_path = None
    trainer = Trainer(callbacks=checkpoint_callbacks,
                      resume_from_checkpoint=model_path,
                      max_epochs=300, gpus=1, logger=logger,
                      check_val_every_n_epoch=1, log_every_n_steps=10,
                      flush_logs_every_n_steps=100)

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
