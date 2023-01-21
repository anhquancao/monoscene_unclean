from torch.utils.data.dataloader import DataLoader
from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
import pytorch_lightning as pl
from xmuda.data.semantic_kitti.collate import collate_fn
from xmuda.common.utils.torch_util import worker_init_fn
from torchvision import transforms


class SemanticKittiDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=4,
                 num_workers=6,
                 n_pixels_per_voxel=1,
                 normalize_image=True):
        super().__init__()
#        self.preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
#        self.semantic_kitti_dir = '/datasets_master/semantic_kitti'
        self.preprocess_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti/preprocess/preprocess'
        self.semantic_kitti_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_h = 370
        self.img_w = 1220
        self.normalize_image = normalize_image
        self.n_pixels_per_voxel = n_pixels_per_voxel

    def setup(self, stage=None):
        self.train_ds = SemanticKITTISCN(split=('train',),
                                         preprocess_dir=self.preprocess_dir,
                                         semantic_kitti_dir=self.semantic_kitti_dir,
                                         img_h=self.img_h,
                                         img_w=self.img_w,
                                         noisy_rot=0.0,
                                         rot_z=0,
                                         transl=False,
                                         bottom_crop=None,
                                         fliplr=0.5,
                                         color_jitter=(0.4, 0.4, 0.4),
                                         normalize_image=self.normalize_image
                                         )
        self.val_ds = SemanticKITTISCN(split=('val',),
                                       preprocess_dir=self.preprocess_dir,
                                       semantic_kitti_dir=self.semantic_kitti_dir,
                                       img_h=self.img_h,
                                       img_w=self.img_w,
                                       noisy_rot=0,
                                       flip_y=0,
                                       rot_z=0,
                                       transl=False,
                                       bottom_crop=None,
                                       fliplr=0,
                                       color_jitter=None,
                                       normalize_image=self.normalize_image
                                       )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_ds,
    #         pin_memory=True,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         collate_fn=semantic_kitti_collate_fn,
    #         shuffle=False
    #     )
