import torch
import os.path as osp
import pickle
from PIL import Image
import numpy as np
from numpy.core.numeric import full
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms as T
import glob
import cv2

from xmuda.data.utils.preprocess import get_pixel_indices_per_voxel
from xmuda.data.utils.refine_pseudo_labels import refine_pseudo_labels
from xmuda.data.utils.augmentation_3d import augment_and_scale_3d
from xmuda.data.utils.data_preprocess import to_voxel_coords
from xmuda.data.semantic_kitti import splits 


class SemanticKITTIBase(Dataset):
    """SemanticKITTI dataset"""

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # use those categories if merge_classes == True (common with A2D2)
    categories = {
        'car': ['car', 'moving-car'],
        'truck': ['truck', 'moving-truck'],
        'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
                 'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
        'person': ['person', 'moving-person'],
        'road': ['road', 'lane-marking'],
        'parking': ['parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'nature': ['vegetation', 'trunk', 'terrain'],
        'other-objects': ['fence', 'pole', 'traffic-sign', 'other-object'],
    }

    def __init__(self,
                 split,
                 preprocess_dir
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize SemanticKITTI dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            scenes = getattr(splits, curr_split)
            for scene in scenes:
                print("Load scene: " + scene)
                glob_path = osp.join(self.preprocess_dir, scene, '*.pkl') 
                pkl_paths = sorted(glob.glob(glob_path))
                self.data.extend(pkl_paths)
#                for pkl_path in pkl_paths:
#                    with open(osp.join(pkl_path), 'rb') as f:
#                        self.data.append(pickle.load(f))
   
            #with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
            #    self.data.extend(pickle.load(f))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticKITTISCN(SemanticKITTIBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 semantic_kitti_dir='',
                 scale=1 / 0.2,
                 full_scale=512,
                 img_h=370,
                 img_w=1220,
                 normalize_image=True,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 bottom_crop=tuple(),  # 2D augmentation (also effects 3D)
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False,
                 down_sample=2,
                 num_depth_classes=16,
                 n_pixels_per_voxel=64,
                 ):
        super().__init__(split,
                         preprocess_dir)

        with open(osp.join(preprocess_dir, "voxel_to_pixel_0.8.pkl"), 'rb') as f:
            self.mapping_2d_3d_1_4 = pickle.load(f)

        self.n_pixels_per_voxel = n_pixels_per_voxel
        self.semantic_kitti_dir = semantic_kitti_dir
        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl
        self.normalize_image = normalize_image

        self.img_h = img_h
        self.img_w = img_w
        self.down_sample = down_sample
        # image parameters
        self.common_2d_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((int(self.img_h/self.down_sample),
                      int(self.img_w / self.down_sample))),
            T.ToTensor(),
        ])
        self.normalize_image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 2D augmentation
        self.bottom_crop = bottom_crop
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(
            *color_jitter) if color_jitter else None

    def __getitem__(self, index):
        pkl_path = self.data[index]
        with open(osp.join(pkl_path), 'rb') as f:
            data_dict = pickle.load(f)

        points_2d = data_dict['points_2d'].copy()
        # points_3d = data_dict['points_3d'].copy()

#        seg_label_3d = data_dict['seg_label_3d'].astype(np.int64)
        seg_label_2d = data_dict['seg_label_2d'].astype(np.int64)

        points_img = data_dict['points_img'].copy()
        ssc_label_1_1 = data_dict['ssc_label_1_1'].copy()
        ssc_label_1_4 = data_dict['ssc_label_1_4'].copy()
        voxel_occupancy = data_dict['voxel_occupancy'].copy()

        out_dict = {}

        img_path = osp.join(self.semantic_kitti_dir, data_dict['camera_path'])
        image = Image.open(img_path).convert('RGB')

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # PIL to numpy
        image = np.array(image)
        image = image[:370, :1220, :]
        
        # compute the mapping voxel to pixels
        scene = data_dict['scene']
        voxel_indices_1_4 = torch.tensor(self.mapping_2d_3d_1_4[scene]['voxel_indices'].astype(int))
        img_indices_1_4 = torch.tensor(self.mapping_2d_3d_1_4[scene]['img_indices'].astype(int))

        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices_1_4[:, 1] = image.shape[1] - 1 - img_indices_1_4[:, 1]

        out_dict['voxel_indices_1_4'] = voxel_indices_1_4
        out_dict['img_indices_1_4'] = img_indices_1_4 // self.down_sample

        image = self.common_2d_transforms(image)
        if self.normalize_image:
            image = self.normalize_image(image)

        out_dict['img'] = image
        out_dict['ssc_label_1_1'] = ssc_label_1_1
        out_dict['ssc_label_1_4'] = np.transpose(ssc_label_1_4, (0, 2, 1))
        out_dict['voxel_occupancy'] = voxel_occupancy
        out_dict['scene'] = data_dict['scene']


        return out_dict


def test_SemanticKITTISCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    semantic_kitti_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti'
    preprocess_dir = semantic_kitti_dir + '/preprocess/preprocess'
#    preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
#    semantic_kitti_dir = '/datasets_master/semantic_kitti'
    # pselab_paths = ("/home/docker_user/workspace/outputs/xmuda/a2d2_semantic_kitti/xmuda_crop_resize/pselab_data/train.npy",)
    # split = ('train',)
    split = ('val',)
    dataset = SemanticKITTISCN(split=split,
                               preprocess_dir=preprocess_dir,
                               semantic_kitti_dir=semantic_kitti_dir,
                               noisy_rot=0,
                               flip_y=0,
                               rot_z=0,
                               transl=False,
                               bottom_crop=None,
                               normalize_image=False,
                               fliplr=1.0,
                               color_jitter=None
                               )
    for i in np.arange(0, 200, 20):
        print("===============")
        data = dataset[i]
        for k, v in data.items():
            if 'scene' == k:
                print(k)
            else:
                print(k, v.shape)
#        coords_3d = data['coords_3d']
        coords_2d = data['coords_2d']
        seg_label_2d = data['seg_label_2d']
        idx = seg_label_2d != 255
        # print(np.sum(idx))
        seg_label_2d = seg_label_2d[idx]
        img_indices = data['img_indices'][idx]
        # print(np.max(seg_label_2d))
        # print(data['img'].shape)
        img = data['img'].numpy()
        img = np.moveaxis(img, 0, 2)
        
        # pseudo_label_2d = data['pseudo_label_2d']
#        draw_points_image_labels(img, img_indices, "images/img_point_" + str(
#            i) + ".jpg", seg_label_2d, color_palette_type='SemanticKITTI_long', point_size=0.5)
#        # draw_points_image_labels(img,df img_indices, pseudo_label_2d, color_palette_type='SemanticKITTI', point_size=1)
#        # assert len(pseudo_label_2d) == len(seg_label)
#        draw_bird_eye_view(coords_2d, "images/bird_eye_2d_" +
#                           str(i) + ".jpg", full_scale=300)
#        draw_bird_eye_view(coords_3d, "images/bird_eye_3d_" +
#                           str(i) + ".jpg", full_scale=300)
#
#        with open('images/data_' + str(i) + '.pkl', 'wb') as handle:
#            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_class_weights():
    # preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
    preprocess_dir = "/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti/preprocess/preprocess"
    split = ('train',)
    dataset = SemanticKITTIBase(split,
                                preprocess_dir
                                )
    # compute points per class over whole dataset
    num_classes = 20
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        # labels = dataset.label_mapping[data['seg_labels']]
        seg_label_3d = data['seg_label_3d']
        seg_label_2d = data['seg_label_2d']
        points_per_class += np.bincount(seg_label_2d[seg_label_2d != 255].astype(int),
                                        minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    test_SemanticKITTISCN()
    # compute_class_weights()
