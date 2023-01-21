from torch.utils.data.dataloader import DataLoader
from xmuda.data.NYU.nyu_dataset import NYUDataset
from xmuda.data.NYU.collate import collate_fn
import pytorch_lightning as pl
from xmuda.common.utils.torch_util import worker_init_fn
from torchvision import transforms
from xmuda.data.NYU.AIC_dataloader import NYUDataset as AICNYUDataset


class NYUDataModule(pl.LightningDataModule):
    def __init__(self,
                 root,
                 preprocess_dir,
                 corenet_proj=False,
                 n_relations=4,
                 batch_size=4,
                 frustum_size=4,
                 data_aug=True,
                 use_predicted_depth=False,
                 num_workers=6):
        super().__init__()
        self.n_relations = n_relations
        self.use_predicted_depth = use_predicted_depth
        self.preprocess_dir = preprocess_dir
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug
        self.frustum_size = frustum_size
        self.corenet_proj = corenet_proj

    def setup(self, stage=None):
        if self.data_aug:
            fliplr = 0.5
            flip3d_x = 0.0
            flip3d_z = 0.0
            color_jitter = (0.4, 0.4, 0.4)
            random_scales = False
        else:
            fliplr = 0.0
            flip3d_x = 0.0
            flip3d_z = 0.0
            color_jitter = None
            random_scales = False 
        self.train_ds = NYUDataset(split='train',
                                   preprocess_dir=self.preprocess_dir,
                                   n_relations=self.n_relations,
                                   root=self.root,
                                   fliplr=fliplr,
                                   corenet_proj=self.corenet_proj,
                                   frustum_size=self.frustum_size,
                                   flip3d_x=flip3d_x,
                                   flip3d_z=flip3d_z,
                                   flipud=0.0,
                                   random_scales=random_scales,
                                   color_jitter=color_jitter,
                                   use_predicted_depth=self.use_predicted_depth,
                                   extract_data=False)
        self.test_ds = NYUDataset(split='test',
                                  preprocess_dir=self.preprocess_dir,
                                  n_relations=self.n_relations,
                                  root=self.root,
                                  corenet_proj=self.corenet_proj,
                                  frustum_size=self.frustum_size,
                                  fliplr=0.0,
                                  flipud=0.0,
                                  flip3d_x=0.0,
                                  flip3d_z=0.0,
                                  random_scales=False,
                                  color_jitter=None,
                                  use_predicted_depth=self.use_predicted_depth,
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
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

