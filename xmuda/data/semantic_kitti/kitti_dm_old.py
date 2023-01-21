from torch.utils.data.dataloader import DataLoader
from xmuda.data.semantic_kitti.kitti_dataset import KittiDataset
import pytorch_lightning as pl
from xmuda.data.semantic_kitti.kitti_collate import collate_fn
from xmuda.common.utils.torch_util import worker_init_fn
from torchvision import transforms


class KittiDataModule(pl.LightningDataModule):
    def __init__(self,
                 root,
                 TSDF_root,
                 depth_root,
                 label_root,
                 sketch_root,
                 occ_root,
                 mapping_root,
                 virtual_img=False,
                 project_scale=2,
                 data_aug=True,
                 frustum_size=4,
                 batch_size=4,
                 num_workers=6,
                 use_predicted_depth=True):
        super().__init__()
        self.root = root
        self.TSDF_root = TSDF_root
        self.virtual_img = virtual_img
        self.depth_root = depth_root
        self.sketch_root = sketch_root
        self.mapping_root = mapping_root
        self.project_scale = project_scale
        self.occ_root = occ_root
        self.label_root = label_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.img_h = 370
        self.img_w = 1220

    def setup(self, stage=None):
        self.train_ds = KittiDataset(split='train',
                                     root=self.root,
                                     TSDF_root=self.TSDF_root,
                                     virtual_img=self.virtual_img,
                                     project_scale=self.project_scale,
                                     mapping_root=self.mapping_root,
                                     occ_root=self.occ_root,
                                     depth_root=self.depth_root,
                                     label_root=self.label_root,
                                     sketch_root=self.sketch_root,
                                     frustum_size=self.frustum_size,
                                     fliplr=0.5,
                                     color_jitter=(0.4, 0.4, 0.4))

        self.val_ds = KittiDataset(split='val',
                                   root=self.root,
                                   TSDF_root=self.TSDF_root,
                                   virtual_img=self.virtual_img,
                                   project_scale=self.project_scale,
                                   mapping_root=self.mapping_root,
                                   occ_root=self.occ_root,
                                   depth_root=self.depth_root,
                                   label_root=self.label_root,
                                   sketch_root=self.sketch_root,
                                   frustum_size=self.frustum_size,
                                   fliplr=0,
                                   color_jitter=None)
        self.test_ds = KittiDataset(split='test',
                                   root=self.root,
                                   TSDF_root=self.TSDF_root,
                                   project_scale=self.project_scale,
                                   mapping_root=self.mapping_root,
                                   occ_root=self.occ_root,
                                   depth_root=self.depth_root,
                                   label_root=None,
                                   sketch_root=None,
                                   frustum_size=self.frustum_size,
                                   fliplr=0,
                                   color_jitter=None)

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

    def test_dataloader(self):
        return DataLoader(
#            self.test_ds,
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )
