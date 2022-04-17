import os
import sys, os
sys.path.append("..")
import torch
from torch.autograd import Variable
import model_unet
import torch.nn as nn
import random
import dataload
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
from itertools import cycle
from skimage import measure,color,morphology

print(torch.__version__)
print(torch.version.cuda)



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
argparser.add_argument('--reflect_shuffle', type=bool, help='reflect_shuffle', default=False)
argparser.add_argument('--back_norm', type=bool, help='back_norm', default=True)
argparser.add_argument('--LACRM', type=bool, help='LACRM', default=False)
argparser.add_argument('--smooth', type=bool, help='smooth', default=False)
argparser.add_argument('--open', type=bool, help='open', default=False)
argparser.add_argument('--rank', type=str, help='rank', default='unc')
argparser.add_argument('--soft', type=str, help='soft', default='none')
argparser.add_argument('--measure', type=bool, help='measure', default=True)
argparser.add_argument('--mix_beta', type=float, help='mix_beta', default=-1)
argparser.add_argument('--uamt', type=bool, help='uamt', default=False)
argparser.add_argument('--main_model_mode', type=str, help='main_model_mode', default="classification")
argparser.add_argument('--category_nums', type=int, help='category_nums', default=1)
argparser.add_argument('--ema_decay', type=float, help='ema_decay', default=0.99)
argparser.add_argument('--lr', type=float, help='lr', default=1e-4)
argparser.add_argument('--datas', type=float, help='datas', default=0.1)
argparser.add_argument('--lacrm_num', type=int, help='lacrm_num', default=1)
argparser.add_argument('--output_type', type=str, help='output_type', default="sigmoid")
argparser.add_argument('--u_thre', type=float, help='u_thre', default=0.5)
argparser.add_argument('--mask_thre', type=float, help='mask_thre', default=0.5)
argparser.add_argument('--beta', type=float, help='beta', default=0.1)
argparser.add_argument('--ram_mask_thre', type=str, help='ram_mask_thre', default="none")
argparser.add_argument('--ram_u_thre', type=str, help='ram_u_thre', default="none")
argparser.add_argument('--ram_beta', type=str, help='ram_beta', default="none")
argparser.add_argument('--datanum', type=int, help='datanum', default=2000)
argparser.add_argument('--batchsize', type=int, help='batchsize', default=4)
argparser.add_argument('--max_iterations', type=int,  default=9000, help='maximum epoch number to train')
argparser.add_argument('--devices', type=str, help='devices', default="6,7")
argparser.add_argument('--begin_epoch', type=int,  default=0, help='begin_epoch')
argparser.add_argument('--epoch', type=int,  default=400, help='epoch')
argparser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
argparser.add_argument('-i','--start_iter', type=str,  default=1000, help='ema_model_weigt_dir')
argparser.add_argument('--if_scale', type=bool, help='if_scale', default=False)
args = argparser.parse_args()
iter_num = args.start_iter

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

savename="lidc_lacrm_tmi_"+args.model_name+"_"+str(args.datas)+"_"+str(args.begin_epoch)+"_"+str(args.mix_beta)+"_"+str(args.lacrm_num)+"_"+args.rank+"_"+str(args.datanum)+"_"+str(args.lr)+"_"+str(args.batchsize)+"_"+str(args.epoch)+"_"

if not args.ram_u_thre=="none":
    savename+=args.ram_u_thre
else:
    savename+=str(args.u_thre)
    
if not args.ram_mask_thre=="none":
    savename+="_"+args.ram_mask_thre
else:
    savename+="_"+str(args.mask_thre)
    
if not args.ram_beta=="none":
    savename = savename+"_"+args.ram_beta+str(args.beta)
else:
    savename = savename+"_"+str(args.beta)

if args.reflect_shuffle:
    savename+="_reflect_shuffle"
if args.back_norm:
    savename+="_back_norm"
if args.rank:
    savename+="_rank"
if args.measure:
    savename+="_measure"
if not args.soft=='none':
    savename+=("_"+args.soft)
if args.smooth:
    savename+="_smooth"
if args.open:
    savename+="_open"
if args.LACRM:
    savename+="_lacrm"
if not args.if_scale:
    savename+="_no_scale"
    
print (savename)




def adjust_learning_rate(optimizer,epoch):
    if epoch==EPOCH//2:
        for param_group in optimizer.param_groups:
            lr=param_group['lr']
            param_group['lr'] = lr*0.1
            print ("decay",param_group['lr'])
        print ("-----------------------------"+'\n')



batchsize=args.batchsize
annset = dataload.funcset(mode="ann",num=args.datanum)
ann_loader = DataLoader(annset, batch_size=batchsize, shuffle=True, num_workers=8,drop_last=True)
unannset = dataload.funcset(mode="unann",num=args.datanum)
unann_loader = DataLoader(unannset, batch_size=batchsize, shuffle=True, num_workers=8,drop_last=True)
print (len(ann_loader))

valset=dataload.funcset(mode="val")
val_loader=DataLoader(valset, batch_size=1, shuffle=False, drop_last=False)
testset=dataload.funcset(mode="test")
test_loader=DataLoader(testset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

if args.model_name=="infnet":
    model=torch.nn.DataParallel(inf_net.Inf_Net())
    model.cuda().float()
    if args.uamt:
        model_teacher = torch.nn.DataParallel(inf_net.Inf_Net())
        for param in model_teacher.parameters():
            param.detach_()  # teacher不更新
elif args.model_name=="unet":
    model=torch.nn.DataParallel(model_unet.Unet(args))
    model.cuda().float()
    if args.uamt:
        model_teacher = torch.nn.DataParallel(model_unet.Unet(args))
        for param in model_teacher.parameters():
            param.detach_()  # teacher不更新
        model_teacher.cuda().float()
elif args.model_name=="attention_unet":
    model=torch.nn.DataParallel(other_models.AttU_Net(args))
    model.cuda().float()
    if args.uamt:
        model_teacher = torch.nn.DataParallel(other_models.AttU_Net(args))
        for param in model_teacher.parameters():
            param.detach_()  # teacher不更新
        model_teacher.cuda().float()
elif args.model_name=="nested_unet":
    model=torch.nn.DataParallel(other_models.NestedUNet(args))
    model.cuda().float()
    if args.uamt:
        model_teacher = torch.nn.DataParallel(other_models.NestedUNet(args))
        for param in model_teacher.parameters():
            param.detach_()  # teacher不更新
        model_teacher.cuda().float()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2        
        

        
def cal_uncertainty(img,model,batchsize, T=4, get_index=0):
    model.eval()
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
    # print (pred_ema.max(),pred_ema.min())
    pred_ema_0 = 1-pred_ema
    uncertainty_1 = -1.0 * torch.sum(pred_ema * torch.log(pred_ema + 1e-6), dim=1, keepdim=True)
    uncertainty_0 = -1.0 * torch.sum(pred_ema_0 * torch.log(pred_ema_0 + 1e-6), dim=1, keepdim=True)
    uncertainty = uncertainty_0 + uncertainty_1
    model.train()
    return uncertainty,pred_ema,uncertainty_1


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
    
    
    
def lesion_paste_one(alpha,img1,img2,pred1,pred2,target1,target2,lung1,lung2,mask1,mask2, if_scale=False):
    pixel_num = float(img1.shape[-2]*img1.shape[-1])
    back1 = img1.sum(dim=(1,2),keepdim=True)/pixel_num
    back2 = img2.sum(dim=(1,2),keepdim=True)/pixel_num
    mask1 = mask1.cpu().data.numpy()
    mask2 = mask2.cpu().data.numpy()
    if args.open:
        mask1 = morphology.erosion(mask1[0], np.ones([4, 4]))[None,:]
        mask1 = morphology.dilation(mask1[0], np.ones([10, 10]))[None,:]

        mask2 = morphology.erosion(mask2[0], np.ones([4, 4]))[None,:]
        mask2 = morphology.dilation(mask2[0], np.ones([10, 10]))[None,:]
    
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
#     print ("num",num1,num2)
    for i in range(1,num1+1):
        loc = arr_loc(label1[0],i)
#         print (loc)
        w = loc[1]-loc[0]
        d = loc[3]-loc[2]
        if (w==0 or d==0):
            continue
        img_patch1 = img1[:,loc[0]:loc[1],loc[2]:loc[3]]
        img_patches1.append(img_patch1)
        wdes1.append((w,d))
        locs1.append(loc)
#         print ('t1',target1.shape,target1[:,loc[0]:loc[1],loc[2]:loc[3]].shape)
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
#     print (lung2.shape)
    loc_lung2 = arr_loc(lung2[0].cpu(),1)
    for i in range(len(img_patches1)):
        lesion1 = img_patches1[i]
        lesion_pred1 = pred_patches1[i]
        lesion_target1 = target_patches1[i]
        w,d = wdes1[i]
        if if_scale:
            scale_size = random.uniform(0.8,1.2)
        else:
            scale_size = 1.0
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
#         print (paste_0,paste_1,paste_2,paste_3)
        random_index_w = random.randint(paste_0,paste_1)
        random_index_d = random.randint(paste_2,paste_3)
        img2, pred2, target2 = paste_b2a(alpha,img2,lung2,pred2,target2,lesion1,lesion_pred1,lesion_target1,random_index_w,random_index_d,w,d,back2,back1)

    loc_lung1 = arr_loc(lung1[0].cpu(),1)
    for i in range(len(img_patches2)):
        lesion2 = img_patches2[i]
        lesion_pred2 = pred_patches2[i]
        lesion_target2 = target_patches2[i]
        w,d = wdes2[i]
        if if_scale:
            scale_size = random.uniform(0.8,1.2)
        else:
            scale_size = 1.0
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
#     print ("li",lesion.shape, lesion_pred.shape, lesion_target.shape)
    
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
            alpha = np.random.beta(beta,beta)
        img_si = img_sure[i] #3,512,512
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
#             print ("img_si",img_si.shape)
            img_si, pred_si, target_si, img_hi, pred_hi, target_hi = lesion_paste_one(alpha,img_si,img_hi,pred_si,pred_hi,target_si,target_hi,lung_si,lung_hi,mask_si,mask_hi,if_scale = args.if_scale)
            
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

    

def tensor_loc(tensor):
    arr = tensor.cpu().data.numpy()
    loc = np.where(arr>0)
    x_min,x_max = loc[0].min(),loc[0].max()
    y_min,y_max = loc[1].min(),loc[1].max()
    return x_min,x_max,y_min,y_max


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
optimizer_l = torch.optim.Adam(model.parameters(), lr=lr)
loss_func_seg=nn.BCELoss()
BCE=nn.BCELoss()
BCEWL=nn.BCEWithLogitsLoss()
BCE_mask=nn.BCELoss(reduction='none')
train_loss_list=[]
test_loss_list=[]
val_dice_list=[]
test_metric_list=[]
val_dice_best=0.0
test_dice_best=0
flag=0
EPOCH=args.epoch

trainsize = 192

def train(epoch=0):
    
    if not args.ram_u_thre=="none":
        if args.ram_u_thre=="down":
            u_thre = (1-tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5))*0.3 + 0.5 #0.8~0.5
        else:
            u_thre = tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5)*0.3 + 0.5 #0.5~0.8
    else:
        u_thre = args.u_thre
        
    if not args.ram_mask_thre=="none":
        if args.ram_mask_thre=="down":
            mask_thre = (1-tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5))*0.5 + 0.3 #0.8~0.3
        else:
            mask_thre = tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5)*0.5 + 0.3 #0.3~0.8
    else:
        mask_thre = args.mask_thre
        
    if args.ram_beta=="up":
        beta = tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5)*args.beta
    elif args.ram_beta=="down":
#         beta = (1-tools.sigmoid_rampup(epoch + 0.1, round(EPOCH)*0.5))*args.beta
        beta = max((1-tools.sigmoid_rampup(epoch + 0.1, round(EPOCH) * 0.5))*args.beta, 0.01)
    else:
        beta = args.beta
    print ("beta",beta)
    
    
    size_rates = [1]
    loss_accu = 0.0
    num = 0
    global iter_num, trainsize
    train_loader = iter(zip(cycle(ann_loader), unann_loader))
    model.train()
    for i, train_datas in enumerate(train_loader):
        iter_num += 1
        ((img_l, lung_l, target_l), (img_u, lung_u, target_u)) = train_datas
        img_l = img_l.cuda().float()
        img_u = img_u.cuda().float()
        target_l = target_l.cuda().float()[:,None,:,:]
#         target_u = target_u.cuda().float()[:,None,:,:]
        
        img_l = F.upsample(img_l, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        img_u = F.upsample(img_u, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        target_l = F.upsample(target_l, size=(trainsize, trainsize), mode='nearest')
        
        lung_l = lung_l.cuda().float()[:,None,:,:]
        lung_u = lung_u.cuda().float()[:,None,:,:]
#         print (lung_l.shape)


        pred_l = model(img_l)
        ol = pred_l.clone().detach()
        loss_l = BCE(pred_l, target_l)
        loss = 0
        loss += loss_l
        
        if (args.LACRM and epoch>=args.begin_epoch):
            pred_u = model(img_u)
            mask_u = tools.binary_poss(pred_u.clone().detach(), mask_thre)
            mask = torch.cat((target_l,mask_u), dim=0)
            target_u = tools.binary_poss(pred_u.clone().detach(), u_thre)
            if args.smooth:
                target_u[target_u==1]=0.9
            target = torch.cat((target_l,target_u), dim=0)
            img = torch.cat((img_l,img_u),dim=0)
            pred = torch.cat((pred_l,pred_u),dim=0)
            lung = torch.cat((lung_l,lung_u),dim=0)
                
            for lacrm_num in range(args.lacrm_num):
                if args.rank=='ul':
                    unc_rank_l = list(np.arange(batchsize))
                    random.shuffle(unc_rank_l)
                    unc_rank_u = list(np.arange(batchsize,2*batchsize))
                    random.shuffle(unc_rank_u)
                    unc_rank = unc_rank_l + unc_rank_u
                elif args.rank=="uu":
                    unc_rank_l = list(np.arange(batchsize))
                    random.shuffle(unc_rank_l)
                    unc_rank_u = list(np.arange(batchsize,2*batchsize))
                    random.shuffle(unc_rank_u)
                    unc_rank = unc_rank_l[:len(unc_rank_l) // 2] + unc_rank_u + unc_rank_l[-len(unc_rank_l) // 2:]
                elif args.rank=='unc':
                    uncertainty_l, pred_ema_l, uncertainty_l1 = cal_uncertainty(img_l, model, batchsize)
                    uncertainty_u, pred_ema_u, uncertainty_u1 = cal_uncertainty(img_u, model, batchsize)
                    target_u = tools.binary_poss(pred_ema_u.clone().detach(), u_thre)
                    target = torch.cat((target_l,target_u), dim=0)
                    uncertainty = torch.cat((uncertainty_l,uncertainty_u),dim=0)
                    unc_list_l = []
                    for u in range(uncertainty_l.shape[0]):
                        u_mean = uncertainty_l[u].mean().cpu().data.numpy()
                        unc_list_l.append(u_mean)
                    unc_list_u = []
                    for u in range(uncertainty_u.shape[0]):
                        u_mean = uncertainty_u[u].mean().cpu().data.numpy()
                        unc_list_u.append(u_mean)
                    unc_rank = list(np.argsort(unc_list_l))+[i+4 for i in np.argsort(unc_list_u)]
                else:
                    unc_rank = list(np.arange(2*batchsize))
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

                img_ref,pred_ref,target_ref = LACRM(img_sure.clone(), img_hesi.clone(), pred_sure.clone(), pred_hesi.clone(), target_sure.clone(), target_hesi.clone(), lung_sure.clone(), lung_hesi.clone(), mask_sure.clone(),mask_hesi.clone())
                
#                 plt.imshow(img_sure[3,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(img_hesi[3,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(lung_sure[3,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(lung_hesi[3,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(target_sure[3,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(target_hesi[3,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(img_ref[6,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(img_ref[7,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(target_ref[6,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
#                 plt.imshow(target_ref[7,0].cpu().data.numpy(),cmap="gray")
#                 plt.show()
                
                pred_ref_new = model(img_ref)
                target_ref_new = tools.binary_poss(pred_ref_new.clone().detach(), u_thre)
        
                if args.soft=='mse':
                    loss_m = tools.soft_mse_loss(pred_ref_new, pred_ref)*beta
                elif args.soft=='kl':
                    loss_m = tools.soft_kl_loss(pred_ref_new, pred_ref)*beta
                elif args.soft=='both':
                    loss_m = tools.soft_mse_loss(pred_ref_new, pred_ref)*(args.beta-beta) + BCE(pred_ref_new, target_ref)*beta
                elif args.soft=='both2':
                    loss_m = tools.soft_mse_loss(pred_ref_new, pred_ref)*0.1 + BCE(pred_ref_new, target_ref)*beta
                else:
                    loss_m = BCE(pred_ref_new, target_ref)*beta
#                     loss_m = BCE(pred_ref, target_ref_new)*beta
                loss = loss + loss_m/float(args.lacrm_num)

        loss_accu += loss.item()
        num+=1
        optimizer_l.zero_grad()
        loss.backward()
        optimizer_l.step()
        if args.uamt:
            update_ema_variables(model, model_teacher, args.ema_decay)
        if (i % 100 == 0):
            dice_train = tools.compute_pixel_level_metrics_mean1(ol, target_l)[0]
            print("loss_l:", loss_l.item(), "dice:", dice_train)
            if args.LACRM and epoch>=args.begin_epoch:
                print ("loss_m",loss_m)
                del loss_m
            print("loss", loss)
        del img_l,img_u,target_l,loss,ol
    train_loss_list.append(loss_accu/num)
    print ("train_loss_accu",loss_accu/num)


def val(epoch=0):
    global flag, val_dice_best, trainsize
    dice = 0.0
    num = 0.0
    model.eval()
    for i, test_datas in enumerate(val_loader):
        img_l, target_l = test_datas

        img_l = img_l.cuda().float()
        target_l = target_l.cuda().float()[:,None,:,:]
        
        img_l = F.upsample(img_l, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        target_l = F.upsample(target_l, size=(trainsize, trainsize), mode='nearest')
        
        if args.uamt:
            model_teacher.eval()
            with torch.no_grad():
                pred_l = model_teacher(img_l)
        else:
            with torch.no_grad():
                pred_l = model(img_l)

        metric = tools.compute_pixel_level_metrics_mean1(pred_l, target_l)
        dice += metric[0]
        if i == 0:
            metric_mat = metric[1]
        else:
            metric_mat += metric[1]

        num += 1
        del img_l, target_l
    return dice/num


def test(epoch, save_path="", save=False):
    global flag, test_dice_best, trainsize
    dice = 0.0
    num = 0.0
    loss_accu=0.0
    model.eval()
    for i, test_datas in enumerate(test_loader):
        img_l, target_l = test_datas

        img_l = img_l.cuda().float()
        target_l = target_l.cuda().float()[:,None,:,:]
        
        img_l = F.upsample(img_l, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        target_l = F.upsample(target_l, size=(trainsize, trainsize), mode='nearest')
        
        if args.uamt:
            model_teacher.eval()
            with torch.no_grad():
                pred_l = model_teacher(img_l)
        else:
            with torch.no_grad():
                pred_l = model(img_l)
        test_loss = loss_func_seg(pred_l,target_l)
        loss_accu += test_loss.item()
        metric = tools.compute_pixel_level_metrics_mean1(pred_l, target_l)
        dice += metric[0]
        if i == 0:
            metric_mat = metric[1]
        else:
            metric_mat += metric[1]

        num += 1
        del img_l, target_l
    test_loss_list.append(loss_accu/num)
    print("test_loss_accu", loss_accu / num)
    print("testdice", dice / num, metric_mat/num)
    return dice/num, metric_mat/num



if __name__ == '__main__':
    best_index = 0
    saveroot = "lidc_result/" + savename
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    for i in range(EPOCH):
        print(optimizer_l.state_dict()["param_groups"][0]["lr"])
        print("epoch", i)
        train(i)
        # print (len(grad_block),len(fmap_block))
        # print (grad_block[0].shape,fmap_block[0].shape)
        #
        # plt.subplot(1,2,1)
        # plt.imshow(fmap_block[0][0,0].cpu().data.numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(grad_block[0][0, 0].cpu().data.numpy())
        # plt.savefig("im_gt")
        val_dice = val(i)
        test_dice, test_metric = test(i)
        val_dice_list.append(val_dice)
        if (val_dice >= val_dice_best):
            torch.save(model.state_dict(), os.path.join(saveroot, 'model_vgg16_best' + savename + '.pkl'))
            if (args.uamt):
                torch.save(model_teacher.state_dict(), os.path.join(saveroot, 'model_vgg16_teacher_best' + savename + '.pkl'))
            val_dice_best = val_dice
            best_index = i
        torch.save(model.state_dict(), os.path.join(saveroot, 'model_vgg16' + savename + str(i) + '.pkl'))
        if args.uamt:
            torch.save(model_teacher.state_dict(), os.path.join(saveroot, 'model_vgg16_teacher' + savename + '.pkl'))
        test_metric_list.append(test_metric)

        test_dice_best = test_metric_list[best_index][-1]
        print("val_dice_best", val_dice_best)
        print("test_dice_best", test_dice_best, best_index)
        adjust_learning_rate(optimizer_l, i)
    print(test_metric_list[best_index])
    test_metric_np = np.array(test_metric_list)
    val_dice_np = np.array(val_dice_list)
    np.save(os.path.join(saveroot, "test_metric" + savename), test_metric_np)
    np.save(os.path.join(saveroot, "val_np" + savename), val_dice_np)
    train_loss_np = np.array(train_loss_list)
    test_loss_np = np.array(test_loss_list)
    np.save(os.path.join(saveroot, "train_loss" + savename), train_loss_np)
    np.save(os.path.join(saveroot, "test_loss" + savename), test_loss_np)