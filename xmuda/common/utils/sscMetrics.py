#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import precision_score, recall_score

"""
---- Input:
    predict:
        type, numpy.ndarray 
        shape, (BS=batch_size, C=class_num, W, H, D), onehot encoding
    target:
        type, numpy.ndarray 
        shape, (batch_size, W, H, D)
---- Return
    iou, Intersection over Union
    precision,
    recall
"""
####################################################################################################
#      Channel first, process in CPU with numpy
#  (BS, C, W, H, D) or (BS, C, D, H, W)均可，注意predict和target两者统一即可
####################################################################################################


def get_iou(iou_sum, cnt_class):
    # iou = np.divide(iou_sum, cnt_class)  # what if cnt_class[i]==0, 当测试集中，某些样本类别缺失
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=np.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx]/cnt_class[idx] if cnt_class[idx] else 0

    # mean_iou = np.mean(iou, dtype=np.float32)  # what if cnt_class[i]==0
    # mean_iou = np.sum(iou) / np.count_nonzero(cnt_class)
    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])  # 去掉第一类empty
    # mean_iou = np.mean(iou[1:]) 
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]   # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = np.argmax(predict, axis=1)  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = (predict == target)  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
                # weight_k[i, n] = weight[target[i, n]]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc



class SSCMetrics():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert (pred.shape == gt.shape)
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                           minlength=n_cl ** 2).reshape(n_cl,
                                                        n_cl), correct, labeled
    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0 

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        #precision, recall, iou = self.get_score_completion(y_pred, y_true, nonempty)
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn
        
#        self.precision += precision
#        self.recall += recall
#        self.iou += iou
#        self.iou_ssc = np.add(self.iou_ssc, iou_sum)
#        self.cnt_class = np.add(self.cnt_class, cnt_class)
#        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(y_pred, y_true, mask)
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

#        nonefree = y_true != 255
#        if nonempty is not None:
#            nonefree = nonefree & nonempty
#        nonefree_pred = y_pred[nonefree]
#        nonefree_label = y_true[nonefree]
#
#        h_ssc, c_ssc, l_ssc = self.hist_info(self.n_classes, nonefree_pred, nonefree_label)
#        self.hist_ssc += h_ssc
#        self.correct_ssc += c_ssc
#        self.labeled_ssc += l_ssc
            
    def get_stats(self):
#        iou_ssc, iou_ssc_mean = get_iou(self.iou_ssc, self.cnt_class)
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (self.completion_tp + self.completion_fp + self.completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
#        score_ssc = self.compute_score(self.hist_ssc, self.correct_ssc, self.labeled_ssc)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,

            'iou_ssc': iou_ssc,
            'iou_ssc_mean': np.mean(iou_ssc[1:]),

#            'iou_ssc': score_ssc[0].tolist(),
#            'iou_ssc_mean': score_ssc[2]

#            'precision': self.precision / self.count,
#            'recall': self.recall / self.count,
#            'iou': self.iou / self.count,
#            'iou_ssc': iou_ssc,
#            'iou_ssc_mean': iou_ssc_mean
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes)) 
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0 
        self.iou = 0 
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32) 
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32) 


    def get_score_completion(self, predict, target, nonempty=None):  # on both observed and occluded voxels
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # _C = predict.shape[1]  # _C = 12
        # ---- one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
#        predict = np.argmax(predict, axis=1)
        # ---- check empty
#        if nonempty is not None:
#            predict[nonempty == 0] = 0     # 0 empty
#            nonempty = nonempty.reshape(_bs, -1)
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)    # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum
#            _p, _r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')  # labels=[0, 1] pos_label=1,
#            _iou = 1 / (1 / _p + 1 / _r - 1) if _p else 0  # 1/iou = (tp+fp+fn)/tp = (tp+fp)/tp + (tp+fn)/tp - 1
#            p += _p
#            r += _r
#            iou += _iou
#        p = p / _bs
#        r = r / _bs
#        iou = iou / _bs
#        return p, r, iou


    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
#        predict = np.argmax(predict, axis=1)
        # ---- check empty
#        if nonempty is not None:
#            predict[nonempty == 0] = 0     # 0 empty
#            nonempty = nonempty.reshape(_bs, -1)
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)    # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]  # 去掉需ignore的点
                y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size
#                u_j = np.array(np.where(y_true == j)).size
#                cnt_class[j] += 1 if u_j else 0
                # iou = 1.0 * tp/(tp+fp+fn) if u_j else 0
                # iou_sum[j] += iou
#                iou_sum[j] += 1.0*tp/(tp+fp+fn) if u_j else 0  # iou = tp/(tp+fp+fn)

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        # return acc, iou_sum, cnt_class
#        return iou_sum, cnt_class, tp_sum, fp_sum, fn_sum
        return tp_sum, fp_sum, fn_sum

            
