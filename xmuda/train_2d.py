from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
from xmuda.data.semantic_kitti.semantic_kitti_dm import SemanticKittiDataModule
from xmuda.common.utils.torch_util import worker_init_fn
from xmuda.data.semantic_kitti.collate import collate_fn
from torch.utils.data.dataloader import DataLoader
from xmuda.models.SSC2d import SSC2d
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import hydra
from omegaconf import DictConfig, OmegaConf

hydra.output_subdir = None

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

@hydra.main(config_name="config/common.yaml")
def main(config: DictConfig):
    preprocess_dir = config.preprocess_dir

    exp_name = config.exp_prefix 
    if config.seg_2d:
        exp_name += "_Seg2d"
    else:
        exp_name += "_NoSeg2d"
    if config.edge_rgb_post_process:
        exp_name += "_EdgeRGBPostProcess_" + config.edge_rgb_post_process
    else:
        exp_name += "_NoEdgeRGBPostProcess" 
    exp_name += "_EdgeExtractor_" + str(config.edge_extractor)
    exp_name += "_{}LMSCNet".format(config.n_lmscnet_encoders)
    exp_name += "_{}PixelsPerVoxel".format(config.n_pixels_per_voxel)
    exp_name += "_{}".format(config.branch)
    if config.shared_lmsc_encoder:
        exp_name += "_SharedLMSC"

    print(exp_name)

    model = SSC2d(num_depth_classes=16, 
                  preprocess_dir=preprocess_dir,
                  seg_2d=config.seg_2d,
                  edge_rgb_post_process=config.edge_rgb_post_process,
                  edge_extractor=config.edge_extractor, 
                  shared_lmsc_encoder=config.shared_lmsc_encoder,
                  branch=config.branch,
                  n_lmscnet_encoders=config.n_lmscnet_encoders)

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=config.logdir,
                                   name=exp_name,
                                   version=''
                                   )

        checkpoint_callbacks = [ModelCheckpoint(save_last=True)]
    else:
        logger = False
        checkpoint_callbacks = False

    semantic_kitti = SemanticKittiDataModule(batch_size=config.batch_size, 
                                             n_pixels_per_voxel=config.n_pixels_per_voxel,
                                             num_workers=config.num_workers)

    model_path = os.path.join(config.logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        trainer = Trainer(callbacks=checkpoint_callbacks,
                          max_epochs=60, gpus=1, logger=logger,
                          check_val_every_n_epoch=1, log_every_n_steps=10,
                          flush_logs_every_n_steps=100)
    else:
        trainer = Trainer(callbacks=checkpoint_callbacks,
                          resume_from_checkpoint=model_path,
                          max_epochs=60, gpus=1, logger=logger,
                          check_val_every_n_epoch=1, log_every_n_steps=10,
                          flush_logs_every_n_steps=100)

    trainer.fit(model, semantic_kitti)

if __name__ == "__main__":
    main()
