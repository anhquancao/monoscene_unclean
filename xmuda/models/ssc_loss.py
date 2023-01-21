import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from kornia.utils.one_hot import one_hot

def construct_ideal_affinity_matrix(scaled_labels, num_classes):
    scaled_labels = scaled_labels.squeeze_().long()
    scaled_labels = scaled_labels.view(-1)

#    mask = (scaled_labels != 255) & (scaled_labels != 0)
    mask_255 = (scaled_labels != 255)
#    mask_0 = (scaled_labels != 0)
#    scaled_labels = scaled_labels[(mask_255 & mask_0)]
    scaled_labels = scaled_labels[mask_255]

    one_hot_labels = F.one_hot(scaled_labels, num_classes)
    one_hot_labels = one_hot_labels.view(-1, num_classes).float()
    ideal_affinity_matrix = torch.mm(one_hot_labels, one_hot_labels.permute(1, 0))
#    ideal_affinity_matrix[mask_v & mask_h] = 3 
#    return ideal_affinity_matrix, mask_0, mask_255
    return ideal_affinity_matrix, mask_255


def get_class_weights(class_frequencies):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.0001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(class_frequencies + epsilon_w))

#    weights = torch.from_numpy(1 / class_frequencies)

    return weights.float()

def mca_conved_loss(p_ori, completion_target_ori, ssc_target, kernel_size=(10, 6, 10), stride=(5, 3, 5)):
    conv_weight = torch.ones((1, 1,
                              kernel_size[0],
                              kernel_size[1],
                              kernel_size[2])).type_as(p_ori)
    nominator_conved = F.conv3d((p_ori * completion_target_ori).unsqueeze(1),
                                conv_weight, stride=stride)
    sum_p_conved = F.conv3d(p_ori.unsqueeze(1),
                            conv_weight,
                            stride=stride)  + 1e-4
    sum_completion_target_conved = F.conv3d(completion_target_ori.unsqueeze(1), conv_weight, stride=stride)  + 1e-4
    t = 1 - completion_target_ori
    t[ssc_target == 255] = 0.0
    sum_spec_nominator_conved = F.conv3d(((1-p_ori)*t).unsqueeze(1), conv_weight, stride=stride)  + 1e-4
    sum_1_minus_completion_target_conved = F.conv3d(t.unsqueeze(1), conv_weight, stride=stride)  + 1e-4
    precision_conved = nominator_conved / sum_p_conved
    recall_conved = nominator_conved / sum_completion_target_conved
    specificity_conved = sum_spec_nominator_conved / sum_1_minus_completion_target_conved
    loss_class_conved = (1 - torch.mean(precision_conved)) + (1 - torch.mean(recall_conved)) + (1 - torch.mean(specificity_conved))
    return loss_class_conved

def compute_shanon_entropy(p, target):
    """
    p: softmax output of the network (bs, n_classes, W, H, D)
    shanon entropy: - sum(p * log(p))
    """
    p = F.softmax(p, dim=1)
    mask = target != 255
    entropy = -1 * (p * torch.log(p + 1e-5)).sum(dim=1)
    masked_entropy = entropy * mask
    return torch.mean(masked_entropy)

def JSD_v2(p, target):
    m = 0.5 * (p + target)
    kl_p = torch.sum(p * (torch.log(p) - torch.log(m)))
    kl_target = torch.sum(target * (torch.log(target) - torch.log(m)))
    return 0.5 * (kl_p + kl_target)

def JSD_smooth(p, target, eps):
    p = p + eps
    p = p / torch.sum(p)
    target = target + eps
    target = target / torch.sum(target)
    return JSD_v2(p, target)

def JSD_nonzeros(p, target):
    nonzeros = target != 0
    p_nonzeros = p[nonzeros] 
    target_nonzeros = target[nonzeros]
    return JSD_v2(p_nonzeros, target_nonzeros)

def JSD_sep(p, target):
    nonzeros = (target != 0)
    nonzero_p = p[nonzeros]
    nonzero_target = target[nonzeros]
    JSD_term = JSD_v2(nonzero_p, nonzero_target)
    zero_p = p[~nonzeros]
    force_empty_term = F.binary_cross_entropy(zero_p, torch.zeros_like(zero_p), reduction='sum')
    return JSD_term + force_empty_term



#def JSD(p, q, eps=1e-5):
##    return torch.sum(p * torch.log(p/(q + 1e-5) + 1e-5)) + torch.sum(q * torch.log(q/(p + 1e-5) + 1e-5))
##    log_p = torch.log(p)
##    log_q = torch.log(q) 
#    return torch.sum(p * torch.log(p/q)) + torch.sum(q * torch.log(q/p))

def KL(p, target):
    kl_term = F.kl_div(torch.log(p), target, reduction='sum') #+ torch.sum(nonzero_p * (torch.log(nonzero_p) - torch.log(target[nonzeros])))
    return kl_term

def cosine_dissimilarity(p, target):
    return 1.0 - torch.dot(p, target)

def KL_sep(p, target, is_nonzeros=True, is_force_empty=False):
    nonzeros = (target != 0)
    if is_nonzeros:
        nonzero_p = p[nonzeros]
        kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction='sum') 
    else:
        kl_term = F.kl_div(torch.log(p), target, reduction='sum') 
    if not is_force_empty:
        return kl_term
    zero_p = p[~nonzeros]
    force_empty_term = F.binary_cross_entropy(zero_p, torch.zeros_like(zero_p), reduction='sum')
    return kl_term + force_empty_term



def compute_class_proportion_from_2d(pred, target_proportion, weight):
    bs = pred.shape[0]
    total_loss = 0
    for i in range(bs):
#        l = KL(target_proportion[i], pred[i], weight)
        l = KL(pred[i], target_proportion[i])
        total_loss += l
    return total_loss / bs


def compute_class_proportion_klmax_loss(pred, target, class_weight=None, frustum_mask=None):
    bs, n_classes, _, _, _ = pred.shape
    
    target_proportion = torch.zeros(n_classes).type_as(pred)
    labels, cnts = torch.unique(target[target != 255], return_counts=True)
    target_proportion[labels.long()] = cnts.float()
    target_proportion = target_proportion / torch.sum(target_proportion)

    pred_probs = F.softmax(pred, dim=1)
    pred_probs = pred_probs.permute(1, 0, 2, 3, 4).reshape(n_classes, -1)
    mask = (target != 255).reshape(-1)
    pred_probs = pred_probs[:, mask]

    max_probs, max_idx = torch.max(pred_probs, dim=0)
#    print("2", max_probs.shape, max_idx.shape)
    pred_proportion = torch.zeros(n_classes).type_as(pred)
    for c in range(n_classes):
        c_mask = (max_idx == c) 
        if torch.sum(c_mask) > 0:
            pred_proportion[c] += torch.sum(max_probs[c_mask])
#    pred_proportion =  pred_proportion / torch.sum(mask)
    cum_prob = pred_proportion / torch.sum(pred_proportion)
#    print(pred_proportion, target_proportion)

#    kl_term, force_empty_term = KL_sep(pred_proportion, target_proportion)
#    kl_term, force_empty_term = KL_sep_max(pred_proportion, target_proportion)
#    return kl_term, force_empty_term
    eps = 1e-8
    target_proportion = target_proportion + eps
    target_proportion = target_proportion / torch.sum(target_proportion)
    cum_prob = cum_prob + eps
    cum_prob = cum_prob / torch.sum(cum_prob)
    kl_term = JSD(cum_prob, target_proportion)
    return kl_term, 0

def compute_class_proportion_loss(pred, target, class_weight=None, frustum_mask=None):
    bs, n_classes, _, _, _ = pred.shape
    
    target_proportion = torch.zeros(n_classes).type_as(pred)
    labels, cnts = torch.unique(target[target != 255], return_counts=True)
#    target_proportion[labels.long()] = cnts.float()
    target_proportion[labels.long()] = cnts.type_as(target_proportion)
    target_proportion = target_proportion / torch.sum(target_proportion)

    pred_probs = F.softmax(pred, dim=1)
    
    pred_probs = pred_probs.permute(1, 0, 2, 3, 4).reshape(n_classes, -1)
    if frustum_mask is not None:
        mask = ((target != 255) & frustum_mask).unsqueeze(1).expand(-1, n_classes, -1, -1, -1)
    else:
        mask = (target != 255).reshape(-1)
    pred_probs = pred_probs[:, mask]
    cum_prob = pred_probs.sum(dim=1)
    cum_prob = cum_prob / cum_prob.sum()
#    cum_prob = cum_prob / torch.sum(target != 255) 
#
#    kl_term, force_empty_term = KL_sep(cum_prob, target_proportion)
#    return kl_term, force_empty_term

    eps = 1e-5
    target_proportion = target_proportion + eps
    target_proportion = target_proportion / torch.sum(target_proportion)
#    cum_prob = cum_prob + eps
#    cum_prob = cum_prob / torch.sum(cum_prob)
    kl_term = JSD(cum_prob, target_proportion)
#    kl_term = KL(cum_prob, target_proportion)
#    kl_term = cosine_dissimilarity(cum_prob, target_proportion)
    return kl_term, 0

#    l_logexpsum = 3.0 * JSD(log_probs, log_target)
#    return loss

#    l = KL(cum_prob, target_proportion)
#    for i in range(bs):
#        pred_prob = pred_probs[i]
#        mask = target[i].unsqueeze(0).expand(n_classes, -1, -1, -1) != 255
#        pred_prob = pred_prob * mask.reshape(n_classes, -1)
#        cum_prob = pred_prob.sum(dim=1)
#        cum_prob = cum_prob / torch.sum(cum_prob) + 1e-5
#
#        l = KL(target_proportion[i], cum_prob)
#        total_loss += l
#    return total_loss / bs 


def multiscale_mca_loss(pred_logit, full_target, target, scale):
#    size = (full_target.shape[1],full_target.shape[2],full_target.shape[3])
    pred_prob = F.softmax(pred_logit, dim=1)
    upsampled_target = F.interpolate(target.unsqueeze(1), 
                                     scale_factor=scale, 
                                     mode='nearest').squeeze()
#    print(target.unsqueeze(1).shape, upsampled_target.shape)
    mask = (upsampled_target == full_target).unsqueeze(1)
    total_loss = 0
    for c in range(pred_prob.shape[1]):
        completion_target = torch.ones_like(target)
        completion_target[target != c] = 0

        prob_c = pred_prob[:, c, :, :, :].unsqueeze(1) * mask
#        prob_c[mask] = 0.0

#        print(prob_c.shape, mask.shape)
        p = F.avg_pool3d(prob_c, kernel_size=scale).squeeze()
        nominator = torch.sum(p * completion_target) 
        precision = nominator / (torch.sum(p) + 1e-4)
        recall = nominator / (torch.sum(completion_target) + 1e-4)
        specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-4)
        loss_class = (1 - precision) + (1 - recall) + (1 - specificity)
        total_loss += loss_class
    return total_loss / pred_prob.shape[1]    
#        print(prob_c.shape, mask.shape)

    

def IoU_loss(pred, ssc_target, is_lovasz=False):
    bs = pred.shape[0]
    n_classes = pred.shape[1]

    pred = F.softmax(pred, dim=1)
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]
#    print(nonempty_target.shape, nonempty_probs.shape)
    if is_lovasz:
#        print("lovasz")
        intersection = (nonempty_target * nonempty_probs).sum()
        union = (nonempty_target).sum() + (nonempty_probs).sum() - intersection.sum()
        return 1 - intersection / union
    else:
        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1-nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
#
    return F.binary_cross_entropy(precision, torch.ones_like(precision)) \
            + F.binary_cross_entropy(recall, torch.ones_like(recall)) \
            + F.binary_cross_entropy(spec, torch.ones_like(spec))


def MCA_ssc_loss(pred, ssc_target, agg, frustum_mask=None):
    bs = pred.shape[0]
    assert agg in ["l2", 'minus_sum', "minus_log", "one_minus", "iou_log", "iou_minus", "sum_product"]

    pred = F.softmax(pred, dim=1)
#    pred_class = torch.argmax(pred, dim=1)

    loss = 0
    count = 0
    mask = ssc_target != 255
    if frustum_mask is not None:
        mask = mask & frustum_mask
#    class_metrics = torch.zeros(12).type_as(pred)
    n_classes = pred.shape[1]
    for i in range(0, n_classes):
#        p = 1 - pred[:, 0, :, :, :]
        p = pred[:, i, :, :, :]
#        p_ori = p
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]
#       print(p.shape, target.shape)

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            if agg == 'minus_sum':
                nominator = torch.sum(p * completion_target) 
                precision = nominator / (torch.sum(p) + 1e-4)
                recall = nominator / (torch.sum(completion_target) + 1e-4)
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-4)
                loss_class = - (precision + recall + specificity)
            elif agg == 'sum_product':
                nominator = torch.sum(p * completion_target) 
                precision = nominator / (torch.sum(p) + 1e-4)
                recall = nominator / (torch.sum(completion_target) + 1e-4)
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-4)
                loss_class = (1 - precision * recall * specificity)
            elif agg == 'minus_log':
                nominator = torch.sum(p * completion_target) 
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target))
                    loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                    loss_class += loss_specificity
            elif agg == 'l2':
                nominator = torch.sum(p * completion_target) 
                precision = nominator / (torch.sum(p) + 1e-4)
                recall = nominator / (torch.sum(completion_target) + 1e-4)
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-4)
                loss_class = (1 - precision) ** 2 + (1 - recall) ** 2 + (1 - specificity) ** 2
            elif agg == 'one_minus':
                nominator = torch.sum(p * completion_target) 
                precision = nominator / (torch.sum(p) + 1e-4)
                recall = nominator / (torch.sum(completion_target) + 1e-4)
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-4)
#                f1_score = 2*((precision*recall)/(precision+recall))
#                loss_class = (1 - f1_score)
                loss_class = (1 - precision) + (1 - recall) + (1 - specificity)
                
            elif agg == 'iou_log':
                intersection = torch.sum(p * completion_target)
                union = torch.sum(p) + torch.sum(completion_target) - intersection
                iou = intersection / union 
                loss_class = F.binary_cross_entropy(iou, torch.ones_like(iou))
            elif agg == 'iou_minus':
                intersection = torch.sum(p * completion_target)
                union = torch.sum(p) + torch.sum(completion_target) - intersection
                loss_class = 1 - intersection / union
            loss += loss_class
#    print(class_metrics)
#    mca_loss = loss / count
    return loss / count
#    metric_aware_ssc_loss = compute_metric_aware_ssc_loss(pred_logit, ssc_target, class_metrics)
#    return loss / count  + metric_aware_ssc_loss

def compute_metric_aware_ssc_loss(pred_logit, target, class_metrics, gamma=2, alpha=0.5, reduction='mean', eps=1e-8):
    mask = target == 255
    target[mask] = 0
    input_soft: torch.Tensor = F.softmax(pred_logit, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target.long(), num_classes=pred_logit.shape[1], device=pred_logit.device, dtype=pred_logit.dtype)

    # compute the actual focal loss
    class_metrics = class_metrics.reshape(1, class_metrics.shape[0], 1, 1, 1) 
#    weight = torch.pow(-input_soft + 1., gamma)
    weight = torch.pow(-class_metrics + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    loss_tmp[mask] = 0.0

    loss = torch.mean(loss_tmp)
    return loss

def focal_ssc_loss(pred_logit, target, gamma=2, alpha=0.5, reduction='mean', eps=1e-8):
    mask = (target == 255)
    target[mask] = 0
#    target = target[mask]
#    pred_logit = pred_logit[mask]

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


def CE_ssc_loss(pred, target, class_weights):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''

#    if use_prob:
#        criterion = nn.NLLLoss(weight=class_weights, ignore_index=255, reduction='mean')
#        loss = criterion(torch.log(pred + 1e-5), target.long())
#    else:
    device, dtype = target.device, target.dtype
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean')

    bs, n_classes, _, _, _ = pred.shape
#    pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, n_classes)
#    target = target.reshape(-1)

    loss = criterion(pred, target.long())

    return loss

def binary_cross_entropy(pred,
                         label,
                         use_sigmoid=False,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
#    if pred.dim() != label.dim():
#        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
#    if weight is not None:
#        weight = weight.float()
#    if use_sigmoid:
#     loss = F.binary_cross_entropy_with_logits(
#            pred, label.float(), weight=class_weight, reduction='none')
#    else:
    loss = F.binary_cross_entropy(pred, label.float(), weight=class_weight, reduction=reduction)
    # do the reduction for the weighted loss
#    loss = weight_reduce_loss(
#        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

class ClassRelationLoss(nn.Module):
    def __init__(self, class_weights, CE_relation_loss, MCA_relation_loss):
        super(ClassRelationLoss, self).__init__()
        self.class_weights = class_weights
        self.cls_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=self.class_weights, reduction='mean')
        self.CE_relation_loss = CE_relation_loss   
        self.MCA_relation_loss = MCA_relation_loss   
        self.bce = binary_cross_entropy

    def forward(self, cls_logit, ssc_label):
        """Forward function."""
#        print(cls_logit.shape, label.shape)
        reduction = "mean"
        avg_factor = None

        loss = 0
        if self.CE_relation_loss:
            loss  += self.cls_criterion(cls_logit, ssc_label)

        if self.MCA_relation_loss:    
            pred = F.softmax(cls_logit, dim=1)

            loss_aff = 0
            count = 0
            for i in range(1, pred.shape[1]):
                cls_score = pred[:, i, :, :]
                cls_label = torch.zeros_like(ssc_label)
                cls_label[ssc_label == i] = 1.0

                if torch.sum(cls_label) > 0:

                    diagonal_matrix = (1 - torch.eye(cls_label.size(1))).type_as(pred)
                    vtarget = diagonal_matrix * cls_label

                    recall_part = torch.sum(cls_score * vtarget, dim=2)
                    denominator = torch.sum(vtarget, dim=2)
                    denominator = denominator.masked_fill_(~(denominator > 0), 1)
                    recall_part = recall_part.div_(denominator)
                    recall_label = torch.ones_like(recall_part)
                    recall_loss = self.bce(
                        recall_part,
                        recall_label,
                        reduction=reduction,
                        avg_factor=avg_factor)

                    spec_part = torch.sum((1 - cls_score) * (1 - cls_label), dim=2)
                    denominator = torch.sum(1 - cls_label, dim=2)
                    denominator = denominator.masked_fill_(~(denominator > 0), 1)
                    spec_part = spec_part.div_(denominator)
                    spec_label = torch.ones_like(spec_part)
                    spec_loss = self.bce(
                        spec_part,
                        spec_label,
                        reduction=reduction,
                        avg_factor=avg_factor)

                    precision_part = torch.sum(cls_score * vtarget, dim=2)
                    denominator = torch.sum(cls_score, dim=2)
                    denominator = denominator.masked_fill_(~(denominator > 0), 1)
                    precision_part = precision_part.div_(denominator)
                    precision_label = torch.ones_like(precision_part)
                    precision_loss = self.bce(
                        precision_part,
                        precision_label,
                        reduction=reduction,
                        avg_factor=avg_factor)
#                    print(precision_loss, recall_loss, spec_loss)
                    loss_aff += (precision_loss + recall_loss + spec_loss)
                    count += 1
#            print(loss_aff, count)
            loss += loss_aff / count 


#                completion_target = torch.ones_like(label)
#                completion_target[label != i] = 0
#
#                diagonal_matrix = (1 - torch.eye(completion_target.size(1))).type_as(p)
#                completion_target = completion_target * diagonal_matrix
#
#                if torch.sum(completion_target) > 0:
#                    count += 1
#                    precision = torch.sum(p * completion_target) / (torch.sum(p) + 1e-8)
#                    recall = torch.sum(p * completion_target) / (torch.sum(completion_target) + 1e-8)
#                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-8)
#                    loss_class = - (torch.log(precision + 1e-8) + torch.log(recall + 1e-8) + torch.log(specificity + 1e-8))
#                    loss_aff += loss_class 

        return loss


class AffinityLoss(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, 
                 CE_relation_loss=True, MCA_relation_loss=True, 
                 reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.CE_relation_loss = CE_relation_loss
        self.MCA_relation_loss = MCA_relation_loss
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = binary_cross_entropy

    def forward(self,
                cls_score,
                label, 
                avg_factor=None):
        reduction = self.reduction
        loss = 0
        if self.CE_relation_loss:
            unary_term = self.cls_criterion(
                cls_score,
                label,
                reduction="mean")
            loss += unary_term

        if self.MCA_relation_loss:
#            p = cls_score
#            diagonal_matrix = (1 - torch.eye(label.size(1))).type_as(p)
#            completion_target = diagonal_matrix * label
#            if torch.sum(completion_target) > 0:
#                precision = torch.sum(p * completion_target) / (torch.sum(p) + 1e-8)
#                recall = torch.sum(p * completion_target) / (torch.sum(completion_target) + 1e-8)
#                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-8)
#                loss_class = (1 - precision) + (1 - recall) + (1 - specificity)
#                loss += loss_class

            diagonal_matrix = (1 - torch.eye(label.size(1))).type_as(cls_score)
            vtarget = diagonal_matrix * label

            recall_part = torch.sum(cls_score * vtarget, dim=2)
            denominator = torch.sum(vtarget, dim=2)
            denominator = denominator.masked_fill_(~(denominator > 0), 1)
            recall_part = recall_part.div_(denominator)
            recall_label = torch.ones_like(recall_part)

#            recall_loss = torch.mean(recall_label - recall_part)

            recall_loss = self.cls_criterion(
                recall_part,
                recall_label,
                reduction=reduction,
                avg_factor=avg_factor)

            spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
            denominator = torch.sum(1 - label, dim=2)
            denominator = denominator.masked_fill_(~(denominator > 0), 1)
            spec_part = spec_part.div_(denominator)
            spec_label = torch.ones_like(spec_part)

#            spec_loss = torch.mean(spec_label - spec_part)

            spec_loss = self.cls_criterion(
                spec_part,
                spec_label,
                reduction=reduction,
                avg_factor=avg_factor)

            precision_part = torch.sum(cls_score * vtarget, dim=2)
            denominator = torch.sum(cls_score, dim=2)
            denominator = denominator.masked_fill_(~(denominator > 0), 1)
            precision_part = precision_part.div_(denominator)
            precision_label = torch.ones_like(precision_part)
            
#            precision_loss = torch.mean(precision_label - precision_part)

            precision_loss = self.cls_criterion(
                precision_part,
                precision_label,
                reduction=reduction,
                avg_factor=avg_factor)

#            per_voxel_losses = (recall_label - recall_part) + (precision_label - precision_part) + (spec_label - spec_part)
#            per_voxel_losses /= 3.0
#            loss += self.cls_criterion(
#                per_voxel_losses,
#                torch.zeros_like(per_voxel_losses),
#                reduction=reduction,
#                avg_factor=avg_factor)
#
            loss  += recall_loss + spec_loss + precision_loss

        return loss
