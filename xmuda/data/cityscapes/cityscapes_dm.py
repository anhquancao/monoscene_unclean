from torch.utils.data.dataloader import DataLoader
from xmuda.data.cityscapes.cityscapes_dataset import CityscapesDataset
import pytorch_lightning as pl
from xmuda.data.kitti_360.collate import collate_fn
from xmuda.common.utils.torch_util import worker_init_fn
from torchvision import transforms


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=4, num_workers=3):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.val_ds = CityscapesDataset(root=self.root, split="val")
        self.train_ds = CityscapesDataset(root=self.root, split="train")
        self.test_ds = CityscapesDataset(root=self.root, split="test")

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
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
