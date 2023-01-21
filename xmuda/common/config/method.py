from xmuda.models.SketchTrainer import SketchTrainer
from xmuda.models.LMSC_trainer import LMSCTrainer

def select_method(config):
    if config.method == "LMSCNet":
        if config.dataset == "NYU":
            in_channels = 144
        elif config.dataset == "kitti":
            in_channels = 32
        model = LMSCTrainer(n_classes=n_classes,
                            class_weights=class_weights,
                            dataset=config.dataset,
                            class_names=class_names, 
                            in_channels=in_channels)

    elif config.method == "3DSketch":
        model = SketchTrainer(predict_empty_from_depth=config.predict_empty_from_depth,
                              optimize_everywhere=config.optimize_everywhere,
                              use_3DSketch_nonempty_mask=config.use_3DSketch_nonempty_mask,
                              n_classes=n_classes,
                              class_names = class_names,
                              class_weights=class_weights['1_4'])

    elif config.method == "AICNet":
    model = AICTrainer(n_classes=n_classes,
                       class_names=NYU_class_names,
                       class_weights=class_weights)
