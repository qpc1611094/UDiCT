# -*- coding: UTF-8 -*-
import random
import os
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import numpy as np
import cv2
from skimage import io
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import scipy
import torch.nn as nn
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
# import surface_distance as surfdist
from scipy.spatial.distance import directed_hausdorff as hausdorff

def norm_white(np_array):
    mean = np_array.mean()
    std = np_array.std()
    np_array = (np_array-mean)/std
    return np_array

def norm(np_array):
    np_array = np_array - np_array.min()
    np_array = np_array / (np_array.max() + 1e-7)
    return np_array


def sharp(tensor):
    tensor/=0.5
    return tensor

def binary_poss(poss):
    poss[poss>=0.5]=1
    poss[poss<0.5]=0
    return poss

def get_lam_mask(img,alpha=1.0):
    img = img[:,0][:,None]
    mask=torch.ones(img.shape)
    lam=np.random.beta(alpha,alpha)
    lam_mask=mask*lam
    return lam_mask


def get_lam(img,alpha=1.0,mask=False):
    if mask:
        lam=torch.randn_like(img[:,0])[:,None]
    else:
        lam=np.random.beta(alpha,alpha)
    return lam


def soft_mse_loss(input_logits, target_logits, T=1, sigmoid = False, reduction="mean"):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_logits, target_logits, reduction=reduction) / num_classes


def soft_kl_loss(input_logits, target_logits, T=1, sigmoid=False, reduction="mean"):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_logits = torch.log(input_logits)

    kl_div=F.kl_div(input_log_logits, target_logits, reduction=reduction)
    # print (kl_div.mean())
    return kl_div


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def _l2_normalize(d):
    d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape(
        (-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)



# def _disable_tracking_bn_stats(model):
#     def switch_attr(m):
#         if hasattr(m, 'track_running_stats'):
#             m.track_running_stats = False
#     model.apply(switch_attr)


# def _see_bn_stats(model):
#     def switch_attr(m):
#         if hasattr(m, 'track_running_stats'):
#             print ("bn_stats",m.running_mean, m.running_var)
#     model.apply(switch_attr)


# def _note_bn_stats(model):
#     bn_lis_mean = []
#     bn_lis_var = []
#     def switch_attr(m):
#         if hasattr(m, 'track_running_stats'):
#             # print ("bn_stats",m.running_mean, m.running_var)
#             bn_lis_mean.append(m.running_mean)
#             bn_lis_var.append(m.running_var)
#     model.apply(switch_attr)
#     return bn_lis_mean, bn_lis_var


# def _change_bn_stats(model, model_teacher):
#     alpha = 0.0
#     model_dict = model.state_dict()
#     model_teacher_dict = model_teacher.state_dict()
#     for (name, param), (name_teacher, param_teacher) in zip(model_dict.items(), model_teacher_dict.items()):
#         # print (name,"---------------")
#         if "running_mean" in name:
#             # print(name, name_teacher)
#             # param_teacher.copy_(param)
#             param_teacher.data.mul_(alpha).add_((1 - alpha) * param.data)
#         elif "running_var" in name:
#             # print(name, name_teacher)
#             # param_teacher.copy_(param)
#             param_teacher.data.mul_(alpha).add_((1 - alpha) * param.data)




# def see_tracking_bn_stats(model):
#     def switch_attr(m):
#         if hasattr(m, 'track_running_stats'):
#             if m.track_running_stats:
#                 print ("bn")

#     # model.apply(switch_attr)
#     # yield
#     model.apply(switch_attr)


def compute_pixel_level_metrics_mean1(pred_a, target_a):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """
    # print (pred_a.shape,target_a.shape)
    # pred_a = pred_a.cpu().data
    # target_a = target_a.cpu().data
    b=pred_a.shape[0]
    c=pred_a.shape[1]
    w=pred_a.shape[2]
    h=pred_a.shape[3]
    # print (b,c,w,h)
    pred_b = pred_a[None,:].repeat(256,1,1,1,1).float()
    target_b = target_a[None,:].repeat(256,1,1,1,1).float()
    threshold = np.ones(256)
    threshold[:255] = np.arange(0,1,1/255)
    # print(threshold)
    threshold = torch.from_numpy(threshold).cuda().float().view(-1,1,1,1,1).repeat(1,b,c,w,h)

    pred_b = ((pred_b-threshold)>=0).float()

    dice_mat = torch.zeros((b,256))
    precision_mat = torch.zeros((b,256))
    sensitivity_mat = torch.zeros((b,256))
    specificity_mat = torch.zeros((b,256))
    mae_mat = torch.zeros((b,256))
    iou_mat = torch.zeros((b,256))
    performance_mat = torch.zeros((b,256))

    for i in range(b):
        pred=pred_b[:,i]
        target=target_b[:,i]
        # print (pred.shape,target.shape)

        tp = torch.sum(pred * target,dim=[1,2,3])  # true postives
        # print("tp: ", tp)
        tn = torch.sum((1-pred) * (1-target),dim=[1,2,3])  # true negatives
        # print("tn: ", tn)
        fp = torch.sum(pred * (1-target),dim=[1,2,3])  # false postives
        # print("fp: ", fp)
        fn = torch.sum((1-pred) * target,dim=[1,2,3])  # false negatives
        # print("fn: ", fn)
        # print (tp,tn,fp,fn)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        specificity = tn / (fp + tn + 1e-10)
        mae = torch.sum(abs(target - pred),dim=[1,2,3])/(c*w*h)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)
        performance = (recall + tn/(tn+fp+1e-10)) / 2
        iou = tp / (tp+fp+fn+1e-10)

        precision_mat[i] = precision
        sensitivity_mat[i] = recall
        specificity_mat[i] = specificity
        mae_mat[i] = mae
        performance_mat[i] = performance
        iou_mat[i] = iou
        dice_mat[i]=F1

    # print (dice_mat.shape, dice_mat.mean(), dice_mat.mean(1).mean())
    dice_mat = dice_mat.mean()
    precision_mat = precision_mat.mean()
    sensitivity_mat = sensitivity_mat.mean()
    specificity_mat = specificity_mat.mean()
    mae_mat = mae_mat.mean()
    iou_mat = iou_mat.mean()
    performance_mat = performance_mat.mean()

    evaluation_metric = np.array([precision_mat,sensitivity_mat,specificity_mat,performance_mat,mae_mat,iou_mat,dice_mat])

    return [dice_mat, evaluation_metric]
    # return [f1.mean(),f1.max()]


def compute_pixel_level_metrics_mean1_over(pred_a, target_a):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    b=pred_a.shape[0]
    c=pred_a.shape[1]
    w=pred_a.shape[2]
    h=pred_a.shape[3]
    
    pred_b = ((pred_a - 0.5)>=0).float()
    target_b = target_a

    dice_mat = []
    precision_mat = []
    sensitivity_mat = []
    specificity_mat = []
    mae_mat = []
    iou_mat = []
    performance_mat = []
    
    for i in range(b):
        pred=pred_b[i]
        target=target_b[i]
        if (pred.sum()==0 and target.sum()==0):
            return [-1, -1]

        tp = torch.sum(pred * target,dim=[0,1,2])  # true postives
        # print("tp: ", tp)
        tn = torch.sum((1-pred) * (1-target),dim=[0,1,2])  # true negatives
        # print("tn: ", tn)
        fp = torch.sum(pred * (1-target),dim=[0,1,2])  # false postives
        # print("fp: ", fp)
        fn = torch.sum((1-pred) * target,dim=[0,1,2])  # false negatives
        # print("fn: ", fn)
        # print (tp,tn,fp,fn)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        specificity = tn / (fp + tn + 1e-10)
        mae = torch.sum(abs(target - pred),dim=[0,1,2])/(c*w*h)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)
        performance = (recall + tn/(tn+fp+1e-10)) / 2
        iou = tp / (tp+fp+fn+1e-10)

        precision_mat.append(precision.cpu().data.numpy())
        sensitivity_mat.append(recall.cpu().data.numpy())
        specificity_mat.append(specificity.cpu().data.numpy())
        mae_mat.append(mae.cpu().data.numpy())
        performance_mat.append(performance.cpu().data.numpy())
        iou_mat.append(iou.cpu().data.numpy())
        dice_mat.append(F1.cpu().data.numpy())
    
    
    dice_mat = np.array(dice_mat).mean()
    precision_mat = np.array(precision_mat).mean()
    sensitivity_mat = np.array(sensitivity_mat).mean()
    specificity_mat = np.array(specificity_mat).mean()
    mae_mat = np.array(mae_mat).mean()
    iou_mat = np.array(iou_mat).mean()
    performance_mat = np.array(performance_mat).mean()

    evaluation_metric = np.array([precision_mat,sensitivity_mat,specificity_mat,performance_mat,mae_mat,iou_mat,dice_mat])

    return [dice_mat, evaluation_metric]
    
    
    
def compute_pixel_level_metrics_mean1_dis(pred_a, target_a):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """
    # print (pred_a.shape,target_a.shape)
    # pred_a = pred_a.cpu().data
    # target_a = target_a.cpu().data
    b=pred_a.shape[0]
    c=pred_a.shape[1]
    w=pred_a.shape[2]
    h=pred_a.shape[3]
    # print (b,c,w,h)
    pred_b = pred_a[None,:].repeat(256,1,1,1,1).float()
    target_b = target_a[None,:].repeat(256,1,1,1,1).float()
    threshold = np.ones(256)
    threshold[:255] = np.arange(0,1,1/255)
    # print(threshold)
    threshold = torch.from_numpy(threshold).cuda().float().view(-1,1,1,1,1).repeat(1,b,c,w,h)

    pred_b = ((pred_b-threshold)>=0).float()
    
    hd_dist_95_mat = torch.zeros(b)
    dice_mat = torch.zeros((b,256))
    precision_mat = torch.zeros((b,256))
    sensitivity_mat = torch.zeros((b,256))
    specificity_mat = torch.zeros((b,256))
    mae_mat = torch.zeros((b,256))
    iou_mat = torch.zeros((b,256))
    performance_mat = torch.zeros((b,256))
    

    result_a = (pred_a.cpu().data.numpy()>=0.5).astype(np.float32)[0,0]
    label_a = target_a.cpu().data.numpy()[0,0]
    if target_a.max()==0 and result_a.max()==0:
        haus=0
    elif target_a.max()==0:
        haus=76
    elif result_a.max()==0:
        haus=76
    else:
        haus = metric.binary.hd95(result_a, target_a.cpu().data.numpy()[0,0])

    for i in range(b):
        pred=pred_b[:,i]
        target=target_b[:,i]
        # print (pred.shape,target.shape)

        tp = torch.sum(pred * target,dim=[1,2,3])  # true postives
        # print("tp: ", tp)
        tn = torch.sum((1-pred) * (1-target),dim=[1,2,3])  # true negatives
        # print("tn: ", tn)
        fp = torch.sum(pred * (1-target),dim=[1,2,3])  # false postives
        # print("fp: ", fp)
        fn = torch.sum((1-pred) * target,dim=[1,2,3])  # false negatives
        # print("fn: ", fn)
        # print (tp,tn,fp,fn)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        specificity = tn / (fp + tn + 1e-10)
        mae = torch.sum(abs(target - pred),dim=[1,2,3])/(c*w*h)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)
        performance = (recall + tn/(tn+fp+1e-10)) / 2
        iou = tp / (tp+fp+fn+1e-10)

        precision_mat[i] = precision
        sensitivity_mat[i] = recall
        specificity_mat[i] = specificity
        mae_mat[i] = mae
        performance_mat[i] = performance
        iou_mat[i] = iou
        dice_mat[i]=F1

    # print (dice_mat.shape, dice_mat.mean(), dice_mat.mean(1).mean())
    hd_dist_95_mat = haus
    dice_mat = dice_mat.mean()
    precision_mat = precision_mat.mean()
    sensitivity_mat = sensitivity_mat.mean()
    specificity_mat = specificity_mat.mean()
    mae_mat = mae_mat.mean()
    iou_mat = iou_mat.mean()
    performance_mat = performance_mat.mean()

    evaluation_metric = np.array([precision_mat,sensitivity_mat,specificity_mat,performance_mat,mae_mat,iou_mat,dice_mat, haus])

    return [dice_mat, evaluation_metric]