#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d, nn.SyncBatchNorm, nn.GroupNorm)):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.BatchNorm1d,  nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)
        # else:
        #     print(m, norm_layer)
    # print(module.modules)
    # print( len(list(module.parameters())) , 'HHHHHHHHHHHHHHHHH',  len(group_decay) + len(
    #    group_no_decay))
#    print(len(list(module.parameters())),  (len(group_decay) + len(group_no_decay)))
#    print(set(group_decay).intersection(set(group_no_decay)))
    assert len(list(module.parameters())) == (len(group_decay) + len(group_no_decay))
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
