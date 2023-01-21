from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
from xmuda.data.semantic_kitti.semantic_kitti_dm import SemanticKittiDataModule
from xmuda.common.utils.torch_util import worker_init_fn
from xmuda.data.semantic_kitti.collate import collate_fn
from torch.utils.data.dataloader import DataLoader
from xmuda.models.quan_trainer_lmsc import RecNetLMSC
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
preprocess_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti/preprocess/preprocess' 
model = RecNetLMSC(preprocess_dir)

logdir = '/gpfswork/rech/xqt/uyl37fq/code/xmuda-extend/tb_logs'
# /home/docker_user/workspace/xmuda/tb_logs',
logger = TensorBoardLogger(logdir,
                           name='lmsc_1_4',
                           version=0
                           )


checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    filename='checkpoint-{epoch}-{step}',
    save_top_k=-1,
    mode='min',
)

semantic_kitti = SemanticKittiDataModule(batch_size=4, num_workers=10)

trainer = Trainer(callbacks=[checkpoint_callback],
                  max_epochs=60, gpus=1, logger=logger,
                  check_val_every_n_epoch=1, log_every_n_steps=10, 
                  flush_logs_every_n_steps=100)
# trainer = Trainer(callbacks=[checkpoint_callback],
#                   max_epochs=1, gpus=1, check_val_every_n_epoch=1)
# trainer = Trainer(max_epochs=1, gpus=1, automatic_optimization=True)
# trainer.fit(model, train_dataloader)
trainer.fit(model, semantic_kitti)
