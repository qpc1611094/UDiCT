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

# f = open('data_sample4000.pkl', 'rb')
# l_list = pickle.load(f)["ann"]
# f = open('data_sample4000.pkl', 'rb')
# u_list = pickle.load(f)["unann"]
# new_l_list = []
# new_u_list = []
# for item in (l_list):
#     loc = item.rfind('images')
#     new_l_list.append(item[loc+7:])
# for item in (u_list):
#     loc = item.rfind('images')
#     new_u_list.append(item[loc+7:])

# print (len(new_l_list), len(new_u_list))
    
# new_data = {'ann': new_l_list, 'unann': new_u_list}
# with open("data_sample4000_1.pkl", "wb") as f:
#     pickle.dump(new_data, f)
    
# print (l_list[0], u_list[0])
# print (new_l_list[0], new_u_list[0])




def get_list(mode,num):
    dataset_root = ""#lidc data root
    
    if mode=="ann":
        img_root = os.path.join(dataset_root, "train/images")
        f = open('data_sample'+str(num)+'.pkl', 'rb')
        l_list = pickle.load(f)["ann"]
        print (l_list[0])
        for i in range(len(l_list)):
            item = l_list[i]
            l_list[i] = os.path.join(img_root, item)
        return l_list
    elif mode=="unann":
        img_root = os.path.join(dataset_root, "train/images")
        f = open('data_sample'+str(num)+'.pkl', 'rb')
        u_list = pickle.load(f)["unann"]
        for i in range(len(u_list)):
            item = u_list[i]
            u_list[i] = os.path.join(img_root, item)
        return u_list
    elif mode=="val":
        img_root = os.path.join(dataset_root, "val/images")
    elif mode=="test":
        img_root = os.path.join(dataset_root, "test/images")
    elif mode=="all_train":
        img_root = os.path.join(dataset_root, "train/images")
    
    res_list = []
    root_list = os.listdir(img_root)
    for i in range(len(root_list)):
        case_name = root_list[i]
        if not case_name[:4]=="LIDC":
            continue
        case_root = os.path.join(img_root,case_name)
        case_list = os.listdir(case_root)
        for j in range(len(case_list)):
            slice_name = case_list[j]
            if not slice_name[-4:]==".png":
                continue
                
            path = os.path.join(case_root,slice_name)
            res_list.append(path)
    return res_list
            

def norm(np_array):
    np_array = np_array - np.min(np_array)
    np_array = np_array / np.max(np_array + 1e-7)
    return np_array



def move(img,lung,target,rate=0.3):
    #w,d,3
    #w,d
    flag = random.randint(0,4)
    img_n = img.copy()
    lung_n = lung.copy()
    target_n = target.copy()
    rand_float = np.random.uniform(0.8,1.2)
    px = int(img.shape[0]*rate*rand_float)
    if flag==0:
        return img_n, lung_n, target_n
    elif flag==1:
        img_n[:-px]=img[px:]
        img_n[-px:]=0
        lung_n[:-px]=lung[px:]
        lung_n[-px:]=0
        target_n[:-px]=target[px:]
        target_n[-px:]=0
        return img_n, lung_n, target_n
    elif flag==2:
        img_n[px:]=img[:-px]
        img_n[:px]=0
        lung_n[px:]=lung[:-px]
        lung_n[:px]=0
        target_n[px:]=target[:-px]
        target_n[:px]=0
        return img_n, lung_n, target_n
    elif flag==3:
        img_n[:,:-px]=img[:,px:]
        img_n[:,-px:]=0
        lung_n[:,:-px]=lung[:,px:]
        lung_n[:,-px:]=0
        target_n[:,:-px]=target[:,px:]
        target_n[:,-px:]=0
        return img_n, lung_n, target_n
    elif flag==4:
        img_n[:,px:]=img[:,:-px]
        img_n[:,:px]=0
        lung_n[:,px:]=lung[:,:-px]
        lung_n[:,:px]=0
        target_n[:,px:]=target[:,:-px]
        target_n[:,:px]=0
        return img_n, lung_n, target_n
        




def augment_lung(glbimage,lung,glblabel):
    theta = (np.random.rand()-0.5) * 60.0
    p0=np.random.rand()
    p1 = np.random.rand()
    p2=0
    if p0>0.5:
        glbimage = np.flip(glbimage, axis=0)
        lung = np.flip(lung, axis=0)
        glblabel = np.flip(glblabel, axis=0)
    if p1>0.5:
        glbimage = np.flip(glbimage, axis=1)
        lung = np.flip(lung, axis=1)
        glblabel = np.flip(glblabel, axis=1)
    glbimage = np.ascontiguousarray(glbimage)
    lung = np.ascontiguousarray(lung)
    glblabel = np.ascontiguousarray(glblabel)
    if p2 > 0.5:
        glbimage = scipy.ndimage.rotate(glbimage, theta, axes=(0, 1), reshape=False, mode="reflect")
        lung = scipy.ndimage.rotate(lung, theta, axes=(0, 1), reshape=False, mode="reflect",order=0)
        glblabel = scipy.ndimage.rotate(glblabel, theta, axes=(0, 1), reshape=False, mode="reflect",order=0)
    glblabel[glblabel > 1] = 1
    glblabel[glblabel < 0] = 0
#     glbimage=random_bright(glbimage)
    glblabel = np.round(glblabel)

    return glbimage,lung,glblabel



def augment(glbimage,glblabel):
    theta = (np.random.rand()-0.5) * 60.0
    p0 = np.random.rand()
    p1 = np.random.rand()
    p2=0
    if p0>0.5:
        glbimage = np.flip(glbimage, axis=0)
        glblabel = np.flip(glblabel, axis=0)
    if p1>0.5:
        glbimage = np.flip(glbimage, axis=1)
        glblabel = np.flip(glblabel, axis=1)
    glbimage = np.ascontiguousarray(glbimage)
    glblabel = np.ascontiguousarray(glblabel)
    if p2 > 0.5:
        glbimage = scipy.ndimage.rotate(glbimage, theta, axes=(0, 1), reshape=False, mode="reflect")
        glblabel = scipy.ndimage.rotate(glblabel, theta, axes=(0, 1), reshape=False, mode="reflect",order=0)
    glblabel[glblabel > 1] = 1
    glblabel[glblabel < 0] = 0
    glblabel = np.round(glblabel)
    return glbimage,glblabel



def load_data(img_path, mode="test"):
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img /= 255.
    
    gt_path = img_path.replace('images', 'gt')
    lung_path = img_path.replace('images', 'lung')[:-4]+".npy"
    for i in range(4):
        tail="_l"+str(i)
        gt_path_i = gt_path[:-4]+tail+".png"
        tmp_target_i = (io.imread(gt_path_i) > 0) * 1.
        if i==0:
            tmp_target = tmp_target_i
        else:
            tmp_target += tmp_target_i
        
    target = np.asarray(tmp_target).astype(np.float32)
    target = target/4.0
    target[target>=0.5]=1
    target[target<=0.5]=0
    target=norm(target).astype(np.float32)
    target=np.round(target)
    
    if os.path.exists(lung_path):
        lung = np.load(lung_path)[0]
    
    if (mode=="ann") or (mode=="unann"):
        img,lung,target = augment_lung(img,lung,target)
        img,lung,target = move(img,lung,target,rate=0.3)
        return img,lung,target
    else:
        return img,target,img_path
    


class funcset(Dataset):
    def __init__(self,mode,num=0):
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        self.data_list = get_list(mode,num)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        img_path = self.data_list[index]
        if (self.mode=="ann") or (self.mode=="unann"):
            img_l, lung_l, target_l = load_data(img_path, mode=self.mode)
            img_l = self.transform(img_l)
            return img_l,lung_l,target_l
        else:
            img_l, target_l,path_l = load_data(img_path, mode=self.mode)
            img_l = self.transform(img_l)
            return img_l,target_l,path_l