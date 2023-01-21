# Some sections of this code reused code from SemanticKITTI development kit
# https://github.com/PRBonn/semantic-kitti-api

import numpy as np
import torch
import copy


class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets

    assert x.shape == y.shape

    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == x_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.conf_matrix, idxs, 1)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    print("LMSC", tp, fp, fn)
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    print(iou_mean, iou)
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"

  def get_confusion(self):
    return self.conf_matrix.copy()


class LossesTrackEpoch:
  def __init__(self, num_iterations):
    # classes
    self.num_iterations = num_iterations
    self.validation_losses = {}
    self.train_losses = {}
    self.train_iteration_counts = 0
    self.validation_iteration_counts = 0

  def set_validation_losses(self, keys):
    for key in keys:
      self.validation_losses[key] = 0
    return

  def set_train_losses(self, keys):
    for key in keys:
      self.train_losses[key] = 0
    return

  def update_train_losses(self, loss):
    for key in loss:
      self.train_losses[key] += loss[key]
    self.train_iteration_counts += 1
    return

  def update_validaiton_losses(self, loss):
    for key in loss:
      self.validation_losses[key] += loss[key]
    self.validation_iteration_counts += 1
    return

  def restart_train_losses(self):
    for key in self.train_losses.keys():
      self.train_losses[key] = 0
    self.train_iteration_counts = 0
    return

  def restart_validation_losses(self):
    for key in self.validation_losses.keys():
      self.validation_losses[key] = 0
    self.validation_iteration_counts = 0
    return


class Metrics:

  def __init__(self, nbr_classes):

    self.nbr_classes = nbr_classes
    self.evaluator = iouEval(self.nbr_classes, [])
    self.best_metric_record = {'mIoU': 0, 'IoU':0, 'epoch': 0, 'loss': 99999999}


  def add_batch(self, prediction, target, scenes=None, invisible_data_dict=None):

    prediction = torch.argmax(prediction, dim=1).data.cpu().numpy()
    # target = target.cpu().numpy()

    # print(target.shape, prediction.shape)
    
    if invisible_data_dict is not None:
      for i in range(prediction.shape[0]):
          invisible_idx = invisible_data_dict[scenes[i]].astype(np.int32)
          target[i, invisible_idx[:, 0], invisible_idx[:, 2], invisible_idx[:, 1]] = 255

    prediction = prediction.reshape(-1).astype('int64')
    target = target.reshape(-1).astype('int64')
    lidar_mask = self.get_eval_mask_Lidar(target)
    
    self.evaluator.addBatch(prediction[lidar_mask], target[lidar_mask])

    return

  def get_eval_mask_Lidar(self, target):
    '''
    eval_mask_lidar is only to ingore unknown voxels in groundtruth
    '''
    mask = (target != 255)
    return mask

  def get_occupancy_IoU(self):
    conf = self.evaluator.get_confusion()
    tp_occupancy = np.sum(conf[1:, 1:])
    fp_occupancy = np.sum(conf[1:, 0])
    fn_occupancy = np.sum(conf[0, 1:])
    intersection = tp_occupancy
    union = tp_occupancy + fp_occupancy + fn_occupancy + 1e-15
    iou_occupancy = intersection / union
    return iou_occupancy  # returns iou occupancy

  def get_occupancy_Precision(self):
    conf = self.evaluator.get_confusion()
    tp_occupancy = np.sum(conf[1:, 1:])
    fp_occupancy = np.sum(conf[1:, 0])
    precision = tp_occupancy / (tp_occupancy + fp_occupancy + 1e-15)
    return precision  # returns precision occupancy

  def get_occupancy_Recall(self):
    conf = self.evaluator.get_confusion()
    tp_occupancy = np.sum(conf[1:, 1:])
    fn_occupancy = np.sum(conf[0, 1:])
    recall = tp_occupancy/(tp_occupancy + fn_occupancy + 1e-15)
    return recall  # returns recall occupancy

  def get_occupancy_F1(self):
    conf = self.evaluator.get_confusion()
    tp_occupancy = np.sum(conf[1:, 1:])
    fn_occupancy = np.sum(conf[0, 1:])
    fp_occupancy = np.sum(conf[1:, 0])
    precision = tp_occupancy/(tp_occupancy + fp_occupancy + 1e-15)
    recall = tp_occupancy/(tp_occupancy + fn_occupancy + 1e-15)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    return F1  # returns recall occupancy

  def get_semantics_mIoU(self):
    _, class_jaccard = self.evaluator.getIoU()
    mIoU_semantics = class_jaccard[1:].mean()  # Ignore on free voxels (0 excluded)
    return mIoU_semantics  # returns mIoU semantics

  def reset_evaluator(self):
    # for key in self.evaluator:
    self.evaluator.reset()

  def update_best_metric_record(self, mIoU, IoU, loss, epoch):
    self.best_metric_record['mIoU'] = mIoU
    self.best_metric_record['IoU'] = IoU
    self.best_metric_record['loss'] = loss
    self.best_metric_record['epoch'] = epoch
    return

