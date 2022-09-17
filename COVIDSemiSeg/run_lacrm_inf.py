import os, sys
sys.path.append('..')
import torch
from torch.autograd import Variable
import model_unet
import torch.nn as nn
import random
import dataload_inf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tools
import argparse
import inf_net
import other_models
import torch.nn.functional as F
import copy
import math
import time
from skimage import measure,color,morphology


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic=True
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.benchmark = True


argparser = argparse.ArgumentParser()
argparser.add_argument('--use_pretrained', type=bool, help='use_pretrained', default=True)
argparser.add_argument('--backbone_name', type=str, help='backbone_name', default="vgg16_bn")
argparser.add_argument('--model_name', type=str, help='model_name', default="unet")
argparser.add_argument('--LACRM', type=bool, help='LACRM', default=True)
argparser.add_argument('--rank', type=str, help='rank', default='ul')
argparser.add_argument('--soft', type=str, help='soft', default='none')
argparser.add_argument('--mix_beta', type=float, help='mix_beta', default=0.5)
argparser.add_argument('--main_model_mode', type=str, help='main_model_mode', default="classification")
argparser.add_argument('--category_nums', type=int, help='category_nums', default=1)
argparser.add_argument('--lr', type=float, help='lr', default=1e-4)
argparser.add_argument('--aug', type=bool, help='aug', default=False)
argparser.add_argument('--output_type', type=str, help='output_type', default="none")
argparser.add_argument('--u_thre', type=float, help='u_thre', default=0.5)
argparser.add_argument('--beta', type=float, help='beta', default=1.0)
argparser.add_argument('--ram_beta', type=str, help='ram_beta', default="down")
argparser.add_argument('--begin_epoch', type=int,  default=10, help='begin_epoch')
argparser.add_argument('--save_root', type=str, help='save_root', default='')
args = argparser.parse_args()

savename="inf_lacrm_"+args.model_name+"_"+str(args.begin_epoch)+"_"+str(args.mix_beta)+"_"+args.rank+"_"
    
if not args.ram_beta=="none":
    savename = savename+"_"+args.ram_beta+str(args.beta)
else:
    savename = savename+"_"+str(args.beta)

if args.rank:
    savename+="_rank"

if not args.soft=='none':
    savename+=("_"+args.soft)

if not args.LACRM:
    savename+="_baseline"

print (args.LACRM,savename)

batch_size=8
labeled_bs=batch_size//2
labeled_idxs = list(range(50))
unlabeled_idxs = list(range(50,1600))
val_labeled_idxs = [1,9,17,26,35,44,50,14]
train_size=352
orisize=train_size
batch_sampler = dataload_inf.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
trainset=dataload_inf.COVIDDataset(mode="train",trainsize=train_size,use_edge=True,aug=args.aug)
train_loader=DataLoader(trainset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

val_sampler = dataload_inf.TwoStreamBatchSampler(val_labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
valset=dataload_inf.COVIDDataset(mode="train",trainsize=train_size,use_edge=True,aug=False)
val_loader=DataLoader(valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

testset=dataload_inf.COVIDDataset_test(mode="test",testsize=train_size)
test_loader=DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


if args.model_name=="infnet":
    model=torch.nn.DataParallel(inf_net.Inf_Net())
    model.cuda().float()

elif args.model_name=="unet":
    model=torch.nn.DataParallel(model_unet.Unet(args))
    model.cuda().float()

elif args.model_name=="nested_unet":
    model=torch.nn.DataParallel(other_models.NestedUNet(args))
    model.cuda().float()


def lacrm_flip(lesion1,lesion_pred1,lesion_target1):
    p0 = np.random.rand()
    p1 = np.random.rand()
    if p0>0.5:
        lesion1 = lesion1.flip(dims=[-1])
        lesion_pred1 = lesion_pred1.flip(dims=[-1])
        lesion_target1 = lesion_target1.flip(dims=[-1])
    if p1>0.5:
        lesion1 = lesion1.flip(dims=[-2])
        lesion_pred1 = lesion_pred1.flip(dims=[-2])
        lesion_target1 = lesion_target1.flip(dims=[-2])
    return lesion1,lesion_pred1,lesion_target1
    
    
    
def lesion_paste_one(alpha,img1,img2,pred1,pred2,target1,target2,lung1,lung2,mask1,mask2):
    pixel_num = float(img1.shape[-2]*img1.shape[-1])
    back1 = img1.sum(dim=(1,2),keepdim=True)/pixel_num
    back2 = img2.sum(dim=(1,2),keepdim=True)/pixel_num
    mask1 = mask1.cpu().data.numpy()
    mask2 = mask2.cpu().data.numpy()
    
    label1 = measure.label(mask1)
    num1 = min(label1.max(),10)
    label2 = measure.label(mask2)
    num2 = min(label2.max(),10)
    
    img_patches1 = []
    wdes1 = []
    locs1=[]
    target_patches1=[]
    pred_patches1 = []
    
    img_patches2 = []
    wdes2 = []
    locs2=[]
    target_patches2=[]
    pred_patches2 = []
    for i in range(1,num1+1):
        loc = arr_loc(label1[0],i)
        w = loc[1]-loc[0]
        d = loc[3]-loc[2]
        if (w==0 or d==0):
            continue
        img_patch1 = img1[:,loc[0]:loc[1],loc[2]:loc[3]]
        img_patches1.append(img_patch1)
        wdes1.append((w,d))
        locs1.append(loc)
        target_patches1.append(target1[:,loc[0]:loc[1],loc[2]:loc[3]])
        pred_patches1.append(pred1[:,loc[0]:loc[1],loc[2]:loc[3]])
        
    for i in range(1,num2+1):
        loc = arr_loc(label2[0],i)
        w = loc[1]-loc[0]
        d = loc[3]-loc[2]
        if (w==0 or d==0):
            continue
        img_patch2 = img2[:,loc[0]:loc[1],loc[2]:loc[3]]
        img_patches2.append(img_patch2)
        wdes2.append((w,d))
        locs2.append(loc)
        target_patches2.append(target2[:,loc[0]:loc[1],loc[2]:loc[3]])
        pred_patches2.append(pred2[:,loc[0]:loc[1],loc[2]:loc[3]])

    loc_lung2 = arr_loc(lung2[0].cpu(),1)
    for i in range(len(img_patches1)):
        lesion1 = img_patches1[i]
        lesion_pred1 = pred_patches1[i]
        lesion_target1 = target_patches1[i]
        w,d = wdes1[i]
        
        scale_size = random.uniform(0.5,2)
        new_w = int(round(w*scale_size))
        new_d = int(round(d*scale_size))
        lesion1 = F.upsample(lesion1[None,:], size=(new_w, new_d), mode='bilinear', align_corners=True)[0]
        lesion_pred1 = F.upsample(lesion_pred1[None,:], size=(new_w, new_d), mode='bilinear', align_corners=True)[0]
        lesion_target1 = F.upsample(lesion_target1[None,:], size=(new_w, new_d), mode='nearest')[0]
        w = new_w
        d = new_d
        lesion1,lesion_pred1,lesion_target1 = lacrm_flip(lesion1,lesion_pred1,lesion_target1)
        
        
        paste_0 = loc_lung2[0]
        paste_1 = loc_lung2[1]-w
        paste_2 = loc_lung2[2]
        paste_3 = loc_lung2[3]-d
        if paste_1<=paste_0 or paste_3<=paste_2:
            continue

        random_index_w = random.randint(paste_0,paste_1)
        random_index_d = random.randint(paste_2,paste_3)
        img2, pred2, target2 = paste_b2a(alpha,img2,lung2,pred2,target2,lesion1,lesion_pred1,lesion_target1,random_index_w,random_index_d,w,d,back2,back1)

    loc_lung1 = arr_loc(lung1[0].cpu(),1)
    for i in range(len(img_patches2)):
        lesion2 = img_patches2[i]
        lesion_pred2 = pred_patches2[i]
        lesion_target2 = target_patches2[i]
        w,d = wdes2[i]
        
        scale_size = random.uniform(0.5,2)
        new_w = int(round(w*scale_size))
        new_d = int(round(d*scale_size))
        lesion2 = F.upsample(lesion2[None,:], size=(new_w, new_d), mode='bilinear', align_corners=True)[0]
        lesion_pred2 = F.upsample(lesion_pred2[None,:], size=(new_w, new_d), mode='bilinear', align_corners=True)[0]
        lesion_target2 = F.upsample(lesion_target2[None,:], size=(new_w, new_d), mode='nearest')[0]
        w = new_w
        d = new_d
        lesion2,lesion_pred2,lesion_target2 = lacrm_flip(lesion2,lesion_pred2,lesion_target2)
        
        paste_0 = loc_lung1[0]
        paste_1 = loc_lung1[1]-w
        paste_2 = loc_lung1[2]
        paste_3 = loc_lung1[3]-d
        if paste_1<=paste_0 or paste_3<=paste_2:
            continue
        
        random_index_w = random.randint(paste_0,paste_1)
        random_index_d = random.randint(paste_2,paste_3)
        img1, pred1, target1 = paste_b2a(alpha,img1,lung1,pred1,target1,lesion2,lesion_pred2,lesion_target2,random_index_w,random_index_d,w,d,back1,back2)
        
    return img1, pred1, target1, img2, pred2, target2



def paste_b2a(alpha,img,lung,pred,target,lesion,lesion_pred,lesion_target,random_index_w,random_index_d,w2,d2,back1,back2):
    lesion = lesion-back2+back1
    
    w2 = lesion.shape[-2]
    d2 = lesion.shape[-1]
    
    new_img_patch = lesion*alpha+img[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2]*(1-alpha)
    new_pred_patch = lesion_pred*alpha+pred[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2]*(1-alpha)
    new_target_patch = lesion_target*alpha+target[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2]*(1-alpha)
    lung_patch = lung[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2]
    lung_target_patch = lung_patch*lesion_target
    
    img[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2][lung_target_patch.repeat(img.shape[0],1,1)==1] = new_img_patch[lung_target_patch.repeat(img.shape[0],1,1)==1]
    pred[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2][lung_target_patch.repeat(pred.shape[0],1,1)==1] = new_pred_patch[lung_target_patch.repeat(pred.shape[0],1,1)==1]
    target[:,random_index_w:random_index_w+w2,random_index_d:random_index_d+d2][lung_target_patch.repeat(target.shape[0],1,1)==1] = new_target_patch[lung_target_patch.repeat(target.shape[0],1,1)==1]
    
    return img, pred, target



def LACRM(img_sure, img_hesi, pred_sure, pred_hesi, target_sure, target_hesi, lung_sure, lung_hesi,mask_sure,mask_hesi):
    
    for i in range(img_sure.shape[0]):
        if args.mix_beta==-1:
            alpha = 1
        else:
            beta = args.mix_beta
            alpha = np.random.uniform(0, 2*beta)
        img_si = img_sure[i]
        img_hi = img_hesi[i]
        pred_si = pred_sure[i]
        pred_hi = pred_hesi[i]
        lung_si = lung_sure[i]
        lung_hi = lung_hesi[i]
        target_si = target_sure[i]
        target_hi = target_hesi[i]
        num_si = (target_si==1).sum()
        num_hi = (target_hi==1).sum()
        mask_si = mask_sure[i]
        mask_hi = mask_hesi[i]
        num_si = (mask_si==1).sum()
        num_hi = (mask_hi==1).sum()
        
        if num_si>=1 and num_hi>=1 and (lung_si==1).sum()>=1 and (lung_hi==1).sum()>=1:
            img_si, pred_si, target_si, img_hi, pred_hi, target_hi = lesion_paste_one(alpha,img_si,img_hi,pred_si,pred_hi,target_si,target_hi,lung_si,lung_hi,mask_si,mask_hi)
            
        img_si = img_si[None,:]
        img_hi = img_hi[None,:]
        pred_si = pred_si[None,:]
        pred_hi = pred_hi[None,:]
        target_si = target_si[None,:]
        target_hi = target_hi[None,:]
        
        if i==0:
            img_ref = torch.cat((img_si,img_hi),dim=0)
            pred_ref = torch.cat((pred_si,pred_hi),dim=0)
            target_ref = torch.cat((target_si,target_hi),dim=0)
        else:
            img_ref = torch.cat((img_ref,img_si,img_hi),dim=0)
            pred_ref = torch.cat((pred_ref,pred_si,pred_hi),dim=0)
            target_ref = torch.cat((target_ref,target_si,target_hi),dim=0)
        
    return img_ref,pred_ref,target_ref


def arr_loc(arr,i):
    loc = np.where(arr==i)
    x_min,x_max = loc[0].min(),loc[0].max()
    y_min,y_max = loc[1].min(),loc[1].max()
    return x_min,x_max,y_min,y_max


def norm(x):
    x_min = x.min()
    x_max = x.max()
    return (x-x_min)/(x_max-x_min)


lr=args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
BCE=nn.BCELoss()
BCE_mask = nn.BCELoss(reduction='none')
lr_num=0
def adjust_learning_rate(optimizer,epoch):
    global lr_num
    temp_num=epoch//(EPOCH//2)
    if (temp_num>lr_num):
        lr_num=temp_num
        for param_group in optimizer.param_groups:
            lr=param_group['lr']
            param_group['lr'] = lr*0.1
            print ("decay",param_group['lr'])
        print ("-----------------------------"+'\n')


def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def iteration(out_ppd, out_d4, out_d3, out_d2, out_edge, gt,edge):
    loss5 = joint_loss(out_d2, gt)
    loss4 = joint_loss(out_d3, gt)
    loss3 = joint_loss(out_d4, gt)
    loss2 = joint_loss(out_ppd, gt)
    loss1 = BCE(torch.sigmoid(out_edge), edge)
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    return loss


def cal_uncertainty(img,model,batchsize, T=4, get_index=0):
    img_r = img
    stride = img_r.shape[0] // batchsize
    pred_ema = torch.zeros([T * batchsize, args.category_nums, img.shape[-2], img.shape[-1]]).cuda()
    for t in range(T // stride):
        ema_inputs = img_r
        with torch.no_grad():
            if args.model_name == "infnet":
                pred_ema[batchsize * stride * t: batchsize * stride * (t + 1)] = torch.sigmoid(
                    model(ema_inputs, noise=True)[get_index])
            else:
                pred_ema[batchsize * stride * t: batchsize * stride * (t + 1)] = model(ema_inputs, noise = True)
    pred_ema = pred_ema.reshape(T // stride, batchsize, stride, img.shape[-2],
                                img.shape[-1])
    pred_ema = pred_ema.mean(0)
    pred_ema_0 = 1-pred_ema
    uncertainty_1 = -1.0 * torch.sum(pred_ema * torch.log(pred_ema + 1e-6), dim=1, keepdim=True)
    uncertainty_0 = -1.0 * torch.sum(pred_ema_0 * torch.log(pred_ema_0 + 1e-6), dim=1, keepdim=True)
    uncertainty = uncertainty_0 + uncertainty_1
    return uncertainty,pred_ema,uncertainty_1


def get_edge(pred_mask):
    conv = nn.Conv2d(1,1,3,padding=1,bias=False)
    sobel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((1,1,3,3))
    conv.weight.data = torch.from_numpy(sobel).cuda().float()
    edge = conv(pred_mask)
    return edge
        
        
def train_u(epoch):
    loss_accu=0.0
    dice=0.0
    num=0.0
    size_rates = [0.75, 1, 1.25]
    # size_rates = [1]
    model.train()
    u_thre = args.u_thre

    if args.ram_beta == "up":
        beta = tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5) * args.beta
    elif args.ram_beta == "down":
        beta = max((1-tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5))*args.beta, 0.01)
    else:
        beta = args.beta


    for i, train_datas in enumerate(train_loader):
        for r in range(len(size_rates)):
            rate=size_rates[r]
            img, lung, gt, edge, name = train_datas
            img = img.cuda().float()
            lung = lung.cuda().float()
            gt = gt.cuda().float()
            edge = edge.cuda().float()
            trainsize = int(round(train_size * rate / 32) * 32)
            if rate != 1:
                img = F.upsample(img, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                lung = F.upsample(lung, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt = F.upsample(gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edge = F.upsample(edge, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            img_l, lung_l, gt_l, edge_l, name_l = img[:batch_size//2], lung[:batch_size//2], gt[:batch_size//2], edge[:batch_size//2], name[:batch_size//2]
            img_u, lung_u, gt_u, edge_u, name_u = img[-batch_size//2:], lung[-batch_size//2:], gt[-batch_size//2:], edge[-batch_size//2:], name[-batch_size//2:]
            out_d2_l = model(img_l)
            pred_l = out_d2_l.clone().sigmoid().detach()
            ol = out_d2_l.clone().detach()
            loss= 0
            loss_l=joint_loss(out_d2_l, gt_l)
            loss += loss_l

            if (args.LACRM and epoch >= args.begin_epoch):
                with torch.no_grad():
                    pred_u = model(img_u).sigmoid()
                target_l = gt_l
                mask_u = tools.binary_poss(pred_u.clone().detach())
                mask = torch.cat((target_l, mask_u), dim=0)
                target_u = tools.binary_poss(pred_u.clone().detach())

                target = torch.cat((target_l, target_u), dim=0)
                img = torch.cat((img_l, img_u), dim=0)
                
                pred = torch.cat((pred_l, pred_u), dim=0).detach()
                lung = torch.cat((lung_l, lung_u), dim=0)

                if args.rank == 'ul':
                    unc_rank_l = list(np.arange(labeled_bs))
                    random.shuffle(unc_rank_l)
                    unc_rank_u = list(np.arange(labeled_bs, 2 * labeled_bs))
                    random.shuffle(unc_rank_u)
                    unc_rank = unc_rank_l + unc_rank_u
                elif args.rank == "uu":
                    unc_rank_l = list(np.arange(labeled_bs))
                    random.shuffle(unc_rank_l)
                    unc_rank_u = list(np.arange(labeled_bs, 2 * labeled_bs))
                    random.shuffle(unc_rank_u)
                    unc_rank = unc_rank_l[:len(unc_rank_l) // 2] + unc_rank_u + unc_rank_l[-len(unc_rank_l) // 2:]
                elif args.rank == 'unc':
                    uncertainty_l, pred_ema_l, uncertainty_l1 = cal_uncertainty(img_l, model, labeled_bs)
                    uncertainty_u, pred_ema_u, uncertainty_u1 = cal_uncertainty(img_u, model, labeled_bs)
                    uncertainty = torch.cat((uncertainty_l, uncertainty_u), dim=0)
                    unc_list_l = []
                    for u in range(uncertainty_l.shape[0]):
                        u_mean = uncertainty_l[u].mean()
                        unc_list_l.append(u_mean)
                    unc_list_u = []
                    for u in range(uncertainty_u.shape[0]):
                        u_mean = uncertainty_u[u].mean()
                        unc_list_u.append(u_mean)
                    unc_rank = list(np.argsort(unc_list_l)) + [i + 4 for i in np.argsort(unc_list_u)]
                else:
                    unc_rank = list(np.arange(2 * labeled_bs))
                    random.shuffle(unc_rank)
                img_sure = img[unc_rank][:len(unc_rank) // 2]
                img_hesi = img[unc_rank][-len(unc_rank) // 2:].flip(dims=[0])
                pred_sure = pred[unc_rank][:len(unc_rank) // 2]
                pred_hesi = pred[unc_rank][-len(unc_rank) // 2:].flip(dims=[0])
                lung_sure = lung[unc_rank][:len(unc_rank) // 2]
                lung_hesi = lung[unc_rank][-len(unc_rank) // 2:].flip(dims=[0])
                mask_sure = mask[unc_rank][:len(unc_rank) // 2]
                mask_hesi = mask[unc_rank][-len(unc_rank) // 2:].flip(dims=[0])
                target_sure = target[unc_rank][:len(unc_rank) // 2]
                target_hesi = target[unc_rank][-len(unc_rank) // 2:].flip(dims=[0])

                img_ref, pred_ref, target_ref = LACRM(img_sure.clone(), img_hesi.clone(), pred_sure.clone(),
                                                      pred_hesi.clone(), target_sure.clone(), target_hesi.clone(),
                                                      lung_sure.clone(), lung_hesi.clone(), mask_sure.clone(),
                                                      mask_hesi.clone())
                pred_ref_new = model(img_ref)
                if args.soft == 'mse':
                    loss_m = tools.soft_mse_loss(pred_ref_new, pred_ref)
                elif args.soft == 'kl':
                    loss_m = tools.soft_kl_loss(pred_ref_new, pred_ref)
                else:
                    loss_m = joint_loss(pred_ref_new, target_ref)
                loss = loss + loss_m * beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dice_i = tools.compute_pixel_level_metrics_mean1(ol.sigmoid(), gt_l)[0]
            loss_accu+=loss_l.item()
            dice+=dice_i
            num+=1
    print ("trainloss",loss_accu/num,dice/num)


def test_u(epoch):
    dice=0.0
    num=0.0
    dices=0.0
    model.eval()
    for i, test_datas in enumerate(test_loader):
        img, gt, name = test_datas
        img = img.cuda().float()
        gt = gt.cuda().float()
        with torch.no_grad():
            out_d2 = model(img)
        metric = tools.compute_pixel_level_metrics_mean1(out_d2.sigmoid(), gt)
        dice += metric[0]
        if i == 0:
            metric_mat = metric[1]
        else:
            metric_mat += metric[1]

        num += 1
    print ("dice",dice/num,metric_mat/num)
    return dice/num, metric_mat/num


def validation_u(epoch):
    dice=0.0
    num=0.0
    dices=0.0
    model.eval()
    for i, val_datas in enumerate(val_loader):
        img, lung, gt, edge, name = val_datas
        img_l, gt_l, edge_l, name_l = img[:batch_size//2], gt[:batch_size//2], edge[:batch_size//2], name[:batch_size//2]
        img_l = img_l.cuda().float()
        gt_l = gt_l.cuda().float()
        with torch.no_grad():
            out_d2_s = model(img_l)
        dice_is = tools.compute_pixel_level_metrics_mean1(out_d2_s.sigmoid(), gt_l)[0]
        dices += dice_is
        num += 1
    print ("valdice",dices/num)
    return (dice/num)



if __name__ == '__main__':
    best_index = 0
    saveroot = os.path.join(args.save_root, savename)
    EPOCH=400
    train_loss_list=[]
    test_loss_list=[]
    val_dice_list=[]
    test_metric_list=[]
    test_metric_list=[]
    val_dice_best=0.0
    test_dice_best=0
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    for i in range(EPOCH):
        print(optimizer.state_dict()["param_groups"][0]["lr"])
        print("epoch", i)
        train_u(i)

        val_dice = validation_u(i)
        test_dice, test_metric = test_u(i)
        val_dice_list.append(val_dice)
        if (val_dice >= val_dice_best):
            torch.save(model.state_dict(), os.path.join(saveroot, 'best' + savename + '.pkl'))
            val_dice_best = val_dice
            best_index = i
        torch.save(model.state_dict(), os.path.join(saveroot, str(i)+'_' + savename + '.pkl'))
        test_metric_list.append(test_metric)

        test_dice_best = test_metric_list[best_index][-1]
        print("val_dice_best", val_dice_best)
        print("test_dice_best", test_dice_best, best_index)
        adjust_learning_rate(optimizer, i)
        
        print(test_metric_list[best_index])
        test_metric_np = np.array(test_metric_list)
        val_dice_np = np.array(val_dice_list)
        np.save(os.path.join(saveroot, "test_metric" + savename), test_metric_np)
        np.save(os.path.join(saveroot, "val_np" + savename), val_dice_np)
        train_loss_np = np.array(train_loss_list)
        test_loss_np = np.array(test_loss_list)
        np.save(os.path.join(saveroot, "train_loss" + savename), train_loss_np)
        np.save(os.path.join(saveroot, "test_loss" + savename), test_loss_np)

