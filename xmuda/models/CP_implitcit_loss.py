import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.utils.one_hot import one_hot
from xmuda.data.utils.preprocess import construct_voxel_rel_label


def compute_CP_sliced_loss(pred_logits, CP_mega_matrices):
    logits = []
    labels = []
    bs = pred_logits.shape[0]
    n_relations = 5
    loss = 0
    loss_aff = 0
    cnt_aff = 0
    for i in range(bs):
        pred_logit = pred_logits[i, :, :, :].permute(0, 2, 1) # n_relations, N, n_mega_voxels
        CP_mega_matrix = CP_mega_matrices[i] # n_relations, N, n_mega_voxels
#        print(pred_logit.shape, CP_mega_matrix.shape)
        logits.append(pred_logit.reshape(n_relations, -1))
        labels.append(CP_mega_matrix.reshape(-1))

    logits = torch.cat(logits, dim=1).T # M, 5  
    labels = torch.cat(labels, dim=0).T # M
    cw = torch.ones(5).type_as(logits)
    classes, cnts = torch.unique(labels, return_counts=True)
    cw[classes.long()] = cnts.type_as(logits)
    cw = 1. / torch.log(cw)
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=cw)
#    print(torch.unique(labels))
    loss_bce = criterion(logits, labels.long())
    return loss_bce

def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
    logits = []
    labels = []
    bs, n_relations, _, _ = pred_logits.shape

    for i in range(bs):
##        print(masks[i].shape, masks[i].dtype)
##        pred_logit = pred_logits[i, :, :, masks[i]].permute(0, 2, 1) # n_relations, N, n_mega_voxels
        pred_logit = pred_logits[i, :, :, :].permute(0, 2, 1) # n_relations, N, n_mega_voxels
        CP_mega_matrix = CP_mega_matrices[i] # n_relations, N, n_mega_voxels
        logits.append(pred_logit.reshape(n_relations, -1))
        labels.append(CP_mega_matrix.reshape(n_relations, -1))
#
        reduction = "mean"


    logits = torch.cat(logits, dim=1).T # M, 4
    labels = torch.cat(labels, dim=1).T # M, 4
    
    
    cnt_neg = (labels == 0).sum(0)
    cnt_pos = labels.sum(0)    
    

    pos_weight =  cnt_neg / cnt_pos
    pos_weight[cnt_pos == 0] = 1.0
    # print(pos_weight)
    # print(pos_weight)
    # pos_mask = cnt_pos != 0
    # neg_mask = cnt_pos == 0
    
    # logits = logits[:, pos_mask]
    # labels = labels[:, pos_mask]
    # pos_weight = pos_weight[pos_mask]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_bce = criterion(logits, labels.float())        
    
    return loss_bce

#    if global_loss:
#    P = torch.sigmoid(logits)
#    loss_rel_global = 0
#    for i in range(n_relations):
#        pos_prob = P[:, i]
#        target = labels[:, i]
#        intersection = (target * pos_prob).sum()
#        if pos_prob.sum() > 0:
#            precision = intersection / pos_prob.sum()
#            loss_rel_global += F.binary_cross_entropy(precision, torch.ones_like(precision))
#        if target.sum() > 0:
#            recall = intersection / target.sum()
#            loss_rel_global += F.binary_cross_entropy(precision, torch.ones_like(precision))
#        if (1 - pos_prob).sum() > 0:
#            spec = ((1-target) * (1 - pos_prob)).sum() / (1 - pos_prob).sum()
#            loss_rel_global += F.binary_cross_entropy(spec, torch.ones_like(spec))
#    loss_rel_global /= n_relations
#    return loss_bce, loss_rel_global

def compute_mega_CP_loss(pred_logit, CP_mega_matrix, mask):
    known_logit = pred_logit[:, mask, :]
#    unknown_logit = pred_logit[:, ~mask, :]
#    unknown_logit = unknown_logit.reshape(-1)
    mega_voxels_mask_255 = (CP_mega_matrix.sum(0) != 0).reshape(-1)
    known_logit = known_logit.reshape(4, -1)[:, mega_voxels_mask_255]
    CP_mega_matrix = CP_mega_matrix.reshape(4, -1)[:, mega_voxels_mask_255]
#    known_prob = F.softmax(known_logit, dim=0)
#    CP_mega_matrix = CP_mega_matrix / CP_mega_matrix.sum(0, keepdim=True)
#    print(CP_mega_matrix[:, :10].T, known_prob[:, :10].T)
#    loss_known = torch.mean(1.0 - (known_prob * CP_mega_matrix).sum(0))
#    loss_known = torch.mean((CP_mega_matrix * (torch.log(CP_mega_matrix + 1e-30) - torch.log(known_prob + 1e-30))).sum(0))

    known_logit = known_logit.reshape(-1)
    CP_mega_matrix = CP_mega_matrix.reshape(-1)
    criterion = nn.BCEWithLogitsLoss()
    loss_known = criterion(known_logit, CP_mega_matrix)

#    if unknown_logit.shape[0] > 0:
#        loss_unknown = criterion(unknown_logit, torch.zeros_like(unknown_logit))
#    else:
#        loss_unknown = 0
#    print(loss_unknown)
    return loss_known
#    return loss_known + loss_unknown
#    CP_mega_matrix += 1e-8 # smooth zero-probs
#    CP_mega_matrix = CP_mega_matrix / CP_mega_matrix.sum(0, keepdim=True)    
##    print(CP_mega_matrix)
#    pred_prob = F.softmax(pred_logit, dim=0)
##    print(pred_prob.shape, CP_mega_matrix.shape)
#    p = CP_mega_matrix
#    q = pred_prob 
#    jsd =  torch.sum(p * torch.log(p/q), dim=0) + torch.sum(q * torch.log(q/p), dim=0)
#    return torch.mean(jsd)
#    pass


def focal_loss(pred_logit, target, gamma=2, alpha=0.5, reduction='mean', eps=1e-8):
    input_soft: torch.Tensor = F.softmax(pred_logit, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target.long(), num_classes=pred_logit.shape[1], device=pred_logit.device, dtype=pred_logit.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    loss_tmp[mask] = 0.0

    loss = torch.mean(loss_tmp)
    return loss


def compute_voxel_pairwise_rel_loss(P_logit, target, CE_relation_loss=True, global_loss=True):
    ret = {}
    bs = target.shape[0]
    gt_matrices = []
    for i in range(bs):
        gt_matrix = construct_voxel_rel_label(target[i])
        gt_matrices.append(gt_matrix)
    gt_matrices = torch.stack(gt_matrices).type_as(P_logit).long()
    if CE_relation_loss:
        rel, cnts = torch.unique(gt_matrix, return_counts=True)
        freqs = torch.zeros(5)
        freqs[rel.long()] = cnts.float()
        relation_weights = torch.zeros(5)
#        relation_weights[freqs != 0] = 1/torch.log(freqs[freqs != 0])
        relation_weights[freqs != 0] = 1/torch.sqrt(freqs[freqs != 0])
#        print(rel, cnts)
        criterion = nn.CrossEntropyLoss(reduction='mean', weight=relation_weights.type_as(P_logit))
#        print(P_logit.shape, gt_matrices.shape)
        loss_rel_ce = criterion(P_logit, gt_matrices)
        ret['loss_rel_ce'] = loss_rel_ce

    if global_loss:
        P = F.softmax(P_logit, dim=1)
        loss_rel_global = 0
        for i in range(P_logit.shape[1]):
            pos_prob = P[:, i, :, :]
            target = (gt_matrices == i).float()
            intersection = (target * pos_prob).sum()
            if pos_prob.sum() > 0:
                precision = intersection / pos_prob.sum()
                loss_rel_global += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if target.sum() > 0:
                recall = intersection / target.sum()
                loss_rel_global += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if (1 - pos_prob).sum() > 0:
                spec = ((1-target) * (1 - pos_prob)).sum() / (1 - pos_prob).sum()
                loss_rel_global += F.binary_cross_entropy(spec, torch.ones_like(spec))

        ret['loss_rel_global'] = loss_rel_global / 5.0

    return ret


def compute_CP_implicit_loss(P_logit, CP_matrix, topk_indices, CE_relation_loss=True, global_loss=True):
    """
    ssc_target: (N,)
    P_logit: N, k
    topk_indices: k, 1
    CP_matrix: N, N
    """
    loss = 0
    N, k = P_logit.shape
    label = torch.gather(CP_matrix, 1, topk_indices.T.expand(N, -1))
    if CE_relation_loss:
        loss += F.binary_cross_entropy_with_logits(P_logit, label)
    if MCA_relation_loss:
        cls_criterion = F.binary_cross_entropy
        reduction = 'mean'
        label = label.unsqueeze(0)
        vtarget = label
        cls_score = torch.sigmoid(P_logit).unsqueeze(0)
        recall_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)

        recall_loss = cls_criterion(
            recall_part,
            recall_label,
            reduction=reduction)

        spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
        denominator = torch.sum(1 - label, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)

        spec_loss = cls_criterion(
            spec_part,
            spec_label,
            reduction=reduction)

        precision_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(cls_score, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        

        precision_loss = cls_criterion(
            precision_part,
            precision_label,
            reduction=reduction)

    return loss



