from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, classes as NYU_class_names

def get_dataset(config):
    if config.dataset == "kitti":
        class_names = ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 
                       'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
        logdir=config.kitti_logdir
        full_scene_size = (256, 256, 32)
        n_classes=20
        class_weights = {
            '1_4' : get_class_weights(semantic_kitti_class_frequencies)
#            '1_4': get_class_weights(NYU_class_freq_1_4),#.cuda(),
#            '1_8': get_class_weights(NYU_class_freq_1_8),#.cuda(),
#            '1_16': get_class_weights(NYU_class_freq_1_16)#.cuda(),
        }
        class_relation_weights = NYU_class_relation_weights # TODO: tune this one latter 
        data_module = KittiDataModule(root=config.kitti_root,
                                      data_aug=True,
                                      batch_size=int(config.batch_size / config.n_gpus), 
                                      num_workers=int(config.num_workers_per_gpu * config.n_gpus))

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
#        class_relation_weights = get_class_weights(NYU_class_relation_freqs) # best with 67 relation classes
        class_relation_weights = NYU_class_relation_weights 
#        pred_depth_dir = "/gpfsscratch/rech/xqt/uyl37fq/NYU_pred_depth"
        data_module = NYUDataModule(config.NYU_root,
                                    config.NYU_preprocess_dir,
                                    data_aug=True,
                                    batch_size=int(config.batch_size / config.n_gpus),
                                    num_workers=int(config.num_workers_per_gpu * config.n_gpus))
    return data_module, class_names
