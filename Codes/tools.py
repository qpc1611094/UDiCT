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



def get_lam_mask(img,alpha=1.0):
    # lam_lis=[]
    # for i in range(num):
    #     lam=np.random.beta(alpha,alpha)
    #     # lam=max(lam,1-lam)
    #     lam_lis.append(lam)
    # lam_lis=torch.from_numpy(np.array(lam_lis).astype(np.float32))
    img = img[:,0][:,None]
    mask=torch.ones(img.shape)
    lam=np.random.beta(alpha,alpha)
    lam_mask=mask*lam

#     lam=max(lam,1-lam)
    return lam_mask


def get_lam(img,alpha=1.0,mask=False):
    # lam_lis=[]
    # for i in range(num):
    #     lam=np.random.beta(alpha,alpha)
    #     # lam=max(lam,1-lam)
    #     lam_lis.append(lam)
    # lam_lis=torch.from_numpy(np.array(lam_lis).astype(np.float32))
    if mask:
        lam=torch.randn_like(img[:,0])[:,None]
    else:
        lam=np.random.beta(alpha,alpha)
#     lam=max(lam,1-lam)
    return lam


def data_mix(img_l,target_l,img_u,alpha=1,mask=False):
    lam=get_lam(img_l,alpha,mask)
    # print (lam.shape,img_l.shape)
#     lam=lam.expand(img_l.shape)
#     print (img_l.shape,img_u.shape,lam.shape)
    img_m=lam*img_l+(1-lam)*img_u
    return img_l,target_l,img_u,img_m,lam
    

def softmax_kl_loss(input_logits, target_logits, T=1, sigmoid=False, reduction="mean"):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits/T, dim=1)
    target_softmax = F.softmax(target_logits/T, dim=1)

    # target_softmax = torch.sigmoid(target_logits)
    # input_log_softmax = input_logits
#     if sigmoid:
#         # input_log_softmax = F.log_softmax(input_logits,dim=1)
#         input_log_softmax = F.logsigmoid(input_logits/Ti)
#         target_softmax = torch.sigmoid(target_logits/Tt)
#     else:
#         input_log_softmax = torch.log(input_logits/Ti)
#         target_softmax = target_logits/Tt
    # print (input_log_softmax.min() ,input_log_softmax.max(), target_softmax.min(),target_logits.max())
    kl_div=F.kl_div(input_log_softmax, target_softmax, reduction=reduction)
    # print (kl_div.mean())
    return kl_div


def softmax_mse_loss(input_logits, target_logits, T=1, sigmoid = False, reduction="mean"):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits/T, dim=1)
    target_softmax = F.softmax(target_logits/T, dim=1)
#     if sigmoid:
#         input_softmax = input_logits.sigmoid()
#         target_softmax = target_logits.sigmoid()
#     else:
#         input_softmax = input_logits
#         target_softmax = target_logits
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction=reduction) / num_classes


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



def soft_mix_loss(pred_m,pred_1,pred_2,alpha,sigmoid=False):
    loss_func = nn.MSELoss(reduction='none')
    # loss_func= nn.L1Loss(reduction='none')
    if sigmoid:
        pred_m=pred_m.sigmoid()
        pred_1=pred_1.sigmoid()
        pred_2=pred_2.sigmoid()
    return loss_func(pred_m,(pred_1*alpha+pred_2*(1-alpha)))
    # return -pred_m*(alpha1*pred_1+alpha2*pred_2)


def soft_mix_loss1(pred_m,pred_1,pred_2,alpha,sigmoid=False):
    # loss_func = nn.MSELoss(reduction='none')
    # loss_func= nn.L1Loss(reduction='none')
    if sigmoid:
        pred_m=pred_m.sigmoid()
        pred_1=pred_1.sigmoid()
        pred_2=pred_2.sigmoid()
    return softmax_kl_loss(pred_m,alpha*pred_1+(1-alpha)*pred_2,sigmoid=False,reduction='none')
    # return -pred_m*(alpha1*pred_1+alpha2*pred_2)


def entrophy_loss(p,sigmoid=False):
    if sigmoid:
        p=p.sigmoid()

    return -1 * torch.sum(p * torch.log(p)) / p.size()[0]


def entropy(p,sigmoid=False,softmax=False):
    if sigmoid:
        p=p.sigmoid()
    if softmax:
        p=p.softmax(dim=1)

    return torch.sum(p * torch.log(p)) / p.size()[0]


def softmax_eudist_loss(input_logits, target_logits, Ti=1, Tt=1,sigmoid = False, reduction="mean"):
    assert input_logits.size() == target_logits.size()

    if sigmoid:
        input_softmax = input_logits.sigmoid()
        target_softmax = target_logits.sigmoid()
    else:
        input_softmax = input_logits
        target_softmax = target_logits
    return torch.pow(input_softmax-target_softmax, 2)


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
    

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def dice_loss(score, target, ce):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    if ce:
        loss_ce=nn.BCELoss()
        loss+=loss_ce(score,target)
    return loss

def dice_loss_mix(score, target, ce, alpha):
    target = target.float()
    smooth = 1e-5
    intersect = score * target
    y_sum = target * target
    z_sum = score * score
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth) * alpha
    loss = 1 - loss.mean()
    if ce:
        loss_ce=nn.BCELoss(reduction='none')
        loss+=(loss_ce(score,target)*alpha).mean()
    return loss


def nrloss(score, target):
    score = score.softmax(dim=1)[:,1][:,None]
    target = target.float()
    # print (score.shape,target.shape)
    smooth = 1e-8
    intersect = (score-target)**2
    intersect = (intersect+smooth)**0.75
    # print (intersect.sum())
    # intersect = (torch.pow((target-score).abs(), 2)+smooth).pow(0.75)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (intersect) / (z_sum + y_sum + smooth)
    return loss.sum()



def _l2_normalize(d):
    d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape(
        (-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def _entropy(logits):
    p = logits
    return -torch.mean(torch.sum(p * torch.log(logits+1e-16), dim=1))


def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats = False
    model.apply(switch_attr)


def _see_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            print ("bn_stats",m.running_mean, m.running_var)
    model.apply(switch_attr)


def _note_bn_stats(model):
    bn_lis_mean = []
    bn_lis_var = []
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            # print ("bn_stats",m.running_mean, m.running_var)
            bn_lis_mean.append(m.running_mean)
            bn_lis_var.append(m.running_var)
    model.apply(switch_attr)
    return bn_lis_mean, bn_lis_var


def _change_bn_stats(model, model_teacher):
    alpha = 0.0
    model_dict = model.state_dict()
    model_teacher_dict = model_teacher.state_dict()
    for (name, param), (name_teacher, param_teacher) in zip(model_dict.items(), model_teacher_dict.items()):
        # print (name,"---------------")
        if "running_mean" in name:
            # print(name, name_teacher)
            # param_teacher.copy_(param)
            param_teacher.data.mul_(alpha).add_((1 - alpha) * param.data)
        elif "running_var" in name:
            # print(name, name_teacher)
            # param_teacher.copy_(param)
            param_teacher.data.mul_(alpha).add_((1 - alpha) * param.data)



def _entropy_sig(logits):
    p = logits
    return -torch.mean(torch.sum(p * torch.log(logits), dim=1))

            

# bn_param.get('running_mean', dtype= )
# torch.nn.batchnormalization2d(in_ch, eps, mom, tracking_bn_stats, affin, )
# mom:running_mean = mom*running_mean + (1-mom)*x_mean


def see_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            if m.track_running_stats:
                print ("bn")

    # model.apply(switch_attr)
    # yield
    model.apply(switch_attr)


class VAT(object):
    def __init__(self, eps=1, xi=10, k=1, use_entmin=False):
        self.xi = xi
        self.eps = eps
        self.k = k
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False).cuda()
        self.use_entmin = use_entmin

    def __call__(self, model, X):
        _disable_tracking_bn_stats(model)

        logits = model(X)
        # logits = torch.cat((1-logits,logits),dim=1)

        prob_logits = logits.detach()
        d = _l2_normalize(torch.randn(X.size())).cuda()

        for ip in range(self.k):
            X_hat = X + d * self.xi
            X_hat.requires_grad = True
            # see_tracking_bn_stats(model)
            logits_hat = model(X_hat)
            # logits_hat = torch.cat((1 - logits_hat, logits_hat), dim=1)

            adv_distance = torch.mean(self.kl_div(
                torch.log(logits_hat+1e-16), prob_logits).sum(dim=1))
            adv_distance.backward()
            d = _l2_normalize(X_hat.grad).cuda()

        logits_hat = model(X + self.eps * d)
        # logits_hat = torch.cat((1 - logits_hat, logits_hat), dim=1)
        LDS = torch.mean(self.kl_div(
            torch.log(logits_hat+1e-16), prob_logits).sum(dim=1))

        if self.use_entmin:
            LDS += _entropy(logits_hat)

        return LDS


class VAT_noise(object):
    def __init__(self, eps=1, xi=10, k=1):
        self.xi = xi
        self.eps = eps
        self.k = k
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False).cuda()

    def __call__(self, model, X, index=0):
        _disable_tracking_bn_stats(model)
        # model.eval()

        logits = model(X)[index]
        prob_logits = logits.detach()
        # prob_logits = F.softmax(logits.detach(), dim=1)
        # prob_logits = F.sigmoid(logits.detach())
        d = _l2_normalize(torch.randn(X.size())).cuda()

        for ip in range(self.k):
            X_hat = X + d * self.xi
            X_hat.requires_grad = True
            # see_tracking_bn_stats(model)
            logits_hat = model(X_hat)[index]

            adv_distance = torch.mean(self.kl_div(
                torch.log(logits_hat), prob_logits).sum(dim=1))
            adv_distance.backward()
            d = _l2_normalize(X_hat.grad).cuda()
        # model.train()

        return d


def binary_poss(poss,thre=0.5, sigmoid=False):
    if sigmoid:
        poss = torch.sigmoid(poss)
    poss[poss>=thre]=1
    poss[poss<thre]=0
    # poss=torch.argmax(poss,dim=1)
    return poss


def _entropy_sig(logits):
    p = logits
    return -torch.mean(torch.sum(p * torch.log(logits), dim=1))



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



def compute_pixel_level_metrics(pred_a, target_a):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """
    # f1 = 0.0
    dice_mat = np.zeros((pred_a.shape[0]))
    precision_mat = np.zeros((pred_a.shape[0]))
    sensitivity_mat = np.zeros((pred_a.shape[0]))
    specificity_mat = np.zeros((pred_a.shape[0]))
    mae_mat = np.zeros((pred_a.shape[0]))
    iou_mat = np.zeros((pred_a.shape[0]))
    performance_mat = np.zeros((pred_a.shape[0]))
    # print (pred_a.shape,target_a.shape)
    for i in range(pred_a.shape[0]):
        pred = binary_poss(pred_a[i].clone().float(), 0.5)
        target = target_a[i].float()

        tp = torch.sum(pred * target)  # true postives
        # print("tp: ", tp)
        tn = torch.sum((1-pred) * (1-target))  # true negatives
        # print("tn: ", tn)
        fp = torch.sum(pred * (1-target))  # false postives
        # print("fp: ", fp)
        fn = torch.sum((1-pred) * target)  # false negatives
        # print("fn: ", fn)
        # print (tp,tn,fp,fn)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        specificity = tn / (fp + tn + 1e-10)
        mae = abs(target - pred).mean()
        F1 = 2 * precision * recall / (precision + recall + 1e-10)
        performance = (recall + tn / (tn + fp + 1e-10)) / 2
        iou = tp / (tp + fp + fn + 1e-10)

        precision_mat[i] = precision
        sensitivity_mat[i] = recall
        specificity_mat[i] = specificity
        mae_mat[i] = mae
        performance_mat[i] = performance
        iou_mat[i] = iou
        dice_mat[i] = F1
        # f1 +=F1

    dice_mat = dice_mat.mean()
    precision_mat = precision_mat.mean()
    sensitivity_mat = sensitivity_mat.mean()
    specificity_mat = specificity_mat.mean()
    mae_mat = mae_mat.mean()
    iou_mat = iou_mat.mean()
    performance_mat = performance_mat.mean()

    evaluation_metric = np.array(
        [precision_mat, sensitivity_mat, specificity_mat, performance_mat, mae_mat, iou_mat, dice_mat])


    return [dice_mat, evaluation_metric]
    # return f1/pred_a.shape[0], f1/pred_a.shape[0]


    
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
    
#     for i in range(256):
#     for j in range(b):
#     print ("s",pred_a.shape,target_a.shape)
    result_a = (pred_a.cpu().data.numpy()>=0.5).astype(np.float32)[0,0]
    label_a = target_a.cpu().data.numpy()[0,0]
    if target_a.max()==0 and result_a.max()==0:
        haus=0
    elif target_a.max()==0:
        haus=76 #mean lesion size
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