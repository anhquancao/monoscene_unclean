from torch.utils.data.dataloader import DataLoader
from xmuda.data.NYU.nyu_dataset_AIC import NYUDatasetAIC as NYUDataset 
from xmuda.data.NYU.collate import collate_fn
import pytorch_lightning as pl
from xmuda.common.utils.torch_util import worker_init_fn
from torchvision import transforms
from xmuda.data.NYU.AIC_dataloader import NYUDataset as AICNYUDataset


class NYUDataModuleAIC(pl.LightningDataModule):
    def __init__(self,
                 root,
                 preprocess_dir,
                 pred_depth_dir,
                 batch_size=4,
                 data_aug=True,
                 num_workers=3):
        super().__init__()
        self.preprocess_dir = preprocess_dir
        self.pred_depth_dir = pred_depth_dir
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug

    def setup(self, stage=None):
        if self.data_aug:
            fliplr = 0.5
            color_jitter = (0.4, 0.4, 0.4)
        else:
            fliplr = 0.0
            color_jitter = None
        self.train_ds = NYUDataset(split='train',
                                   preprocess_dir=self.preprocess_dir, 
                                   pred_depth_dir=self.pred_depth_dir,
                                   root=self.root,
                                   fliplr=fliplr,
                                   flipud=0.0,
                                   color_jitter=color_jitter,
                                   extract_data=False)
#        self.test_ds = AICNYUDataset("/gpfsscratch/rech/xqt/uyl37fq/AIC_dataset/NYUtest_npz", True) 
        self.test_ds = NYUDataset(split='test',
                                  preprocess_dir=self.preprocess_dir, 
                                  pred_depth_dir=self.pred_depth_dir,
                                  root=self.root,
                                  fliplr=0.0,
                                  flipud=0.0,
                                  color_jitter=None,
                                  extract_data=False)

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
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
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
