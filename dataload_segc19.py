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
from skimage import exposure
import skimage.transform as tsf


def data_sample_res(num, datalis, png_path):
    f = open('data_sample.pkl', 'rb+')
    d_list = pickle.load(f)
    if num>0.5:
        num=int(num)

    datauselis = d_list["data"+str(num)]
    print ("-----",datauselis[0])
    for i in range(len(datauselis)):
        name = datauselis[i]
        loc = name.rfind("/")
        name = png_path+name[loc:]
#         print (name)
        datauselis[i] = name
    
    return datauselis



def norm(np_array):
    np_array = np_array - np.min(np_array)
    np_array = np_array / np.max(np_array + 1e-7)
    return np_array


def get_fold(mode,fold=2,train_num=1):
    dataset_root="" # dataroot here
    f = open(os.path.join(dataset_root, 'splits_final.pkl'),'rb')
    data = pickle.load(f)
    train_lis_name=data[fold]["train"]
    val_lis_name=data[fold]["val"]
    png_path = os.path.join(dataset_root,"_PNGImages")
    png_lis=os.listdir(png_path)
    train_lis=[]
    val_lis=[]
    test_lis=[]
    for i in range(len(png_lis)):
        loc=png_lis[i].rfind("_")
        tempname=png_lis[i][:loc]
        if (tempname in train_lis_name):
            train_lis.append(os.path.join(png_path,png_lis[i]))
        elif (tempname in val_lis_name):
            val_lis.append(os.path.join(png_path,png_lis[i]))
        elif not png_lis[i].startswith('.'):
            test_lis.append(os.path.join(png_path,png_lis[i]))     
    
    if (mode=="train"):
        reslist = data_sample_res(train_num, train_lis, png_path)
        return reslist
    elif mode=="val":
        return val_lis
    elif mode=="test":
        return test_lis
    elif mode=="all":
        return train_lis+val_lis+test_lis


def random_bright(im,delta=0.125):
    if random.random() < 0.5:
        delta = random.uniform(-delta, delta)
        im += delta
    return im


def augment1(glbimage, glblabel):
    theta = (np.random.rand() - 0.5) * 60.0
    p0 = np.random.rand()
    p1 = np.random.rand()
    p2 = np.random.rand()
    p3 = np.random.rand()

    if p0 > 0.5:
        glbimage = np.flip(glbimage, axis=1)
    if p1 > 0.5:
        glbimage = np.flip(glbimage, axis=2)
    if p2 > 0.5:
        glbimage = np.flip(glbimage, axis=3)
    glbimage = np.ascontiguousarray(glbimage)
    if p3 > 0.5:
        glbimage = scipy.ndimage.rotate(glbimage, theta, axes=(2, 3), reshape=False, mode="reflect")

    glbimage = random_bright(glbimage)
    return glbimage, glblabel




def augment2(glbimage,glblabel):
    theta = (np.random.rand()-0.5) * 60.0
    p0 = np.random.rand()
    p1 = np.random.rand()
    p2 = 0
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
    
    glbimage = random_bright(glbimage)
    glblabel = np.round(glblabel)

    return glbimage,glblabel


def augment(image,p0,p1,if_label=False):
    if p0>0.5:
        image = np.flip(image, axis=0)
    if p1>0.5:
        image = np.flip(image, axis=1)
    if not if_label:
        image = random_bright(image)
    image = np.ascontiguousarray(image)

    return image
    
    
def load_data_l(img_path,mode):
    lung_path = img_path.replace('_PNGImages', '_MASK_Lung')+".npy"
    gt_path = img_path.replace('_PNGImages', '_MaskImages')
    img = cv2.imread(img_path)

    tmp_target = (io.imread(gt_path) > 0) * 1.
    img = img.astype(np.float32)
    img /= 255.
    target=np.asarray(tmp_target).astype(np.float32)
    target=norm(target).astype(np.float32)
    target=np.round(target)
    target[target>1]=1
    
    lung = np.load(lung_path).astype(np.float32)
    lung = np.round(lung)[0]
    
    if (mode=="train"):
        p0 = np.random.rand()
        p1 = np.random.rand()
        img=augment(img,p0,p1)
        target = augment(target,p0,p1,if_label=True)
        lung = augment(lung,p0,p1,if_label=True)
        
    return img.copy(),lung.copy(),target.copy()


def windowing(im, win):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= (win[1] - win[0])
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1.astype(np.uint8)


def multi_windowing(im):
    windows = [[-174,274], [-1493,484], [-534,1425]]
    im_win1 = windowing(im, windows[2])
    im_win2 = windowing(im, windows[1])
    im_win3 = windowing(im, windows[0])
    im = np.stack((im_win1, im_win2, im_win3), axis=2)
    return im


def load_data_u(img_path,mode):
    if not os.path.exists(img_path):
        print (img_path)
    lung_path = img_path.replace('PCL_data_png', 'lung_mask_PCL_data_png')[:-4]+".npy"
    img = cv2.imread(img_path,-1).astype(np.float32)
    img-=32768.0
    img=multi_windowing(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.
    lung = np.load(lung_path).astype(np.float32)
    lung = np.round(lung)
    
    if (mode=="train"):
        p0 = np.random.rand()
        p1 = np.random.rand()
        img = augment(img,p0,p1)
        lung = augment(lung,p0,p1,if_label=True)
        
    return img.copy(), lung.copy()

    
def get_u(filename="data_u.txt"):
    dataset_root="" # dataroot here
    filepath = os.path.join(dataset_root, filename)
    lis=open(filepath,"r")
    lis_r=lis.readlines()
    return lis_r


def get_patient_name(img_path):
    loc1=img_path.rfind("/")
    loc2=img_path.rfind("_")
    return img_path[loc1+1:loc2]


def get_unlabeled_path(img_path):
    unlabeled_root = "" #unlabeled dataroot
    name=get_patient_name(img_path)
    num_flod=0
    fold_root = os.path.join(unlabeled_root,name)
    for item in os.listdir(fold_root):
        if "0"<=str(item) and str(item)<="9":
            num_flod+=1
    rint=random.randint(0,num_flod-1)
    onepath=os.path.join(unlabeled_root,name+"/"+str(rint))
    onelist=os.listdir(onepath)
    num=random.randint(len(onelist)//3,2*len(onelist)//3)
    samplename=onelist[0][:8]+str(num)+".png"
    return os.path.join(onepath,samplename)


class funcset(Dataset):
    def __init__(self,mode,train_num):
        self.mode = mode
        self.train_num = train_num
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        self.data_list_l=get_fold(mode=self.mode,train_num=self.train_num)
        print (len(self.data_list_l))
        
        self.data_list_u=get_u()
        
    def __len__(self):
        return len(self.data_list_l)

    def __getitem__(self, index):

        index_l=index
        img_path_l = self.data_list_l[index_l]
        img_l, lung_l, target_l = load_data_l(img_path_l, mode=self.mode)
        
        img_path_u=get_unlabeled_path(img_path_l)
        img_u, lung_u = load_data_u(img_path_u,mode=self.mode)

        img_l = self.transform(img_l)
        img_u = self.transform(img_u)
        
        loc=img_path_l.rfind("/")
        tempname=img_path_l[loc+1:-4]
        locu=img_path_u.rfind("/")
        u_name = img_path_u[locu+1:-4]
        return img_l,lung_l,target_l,img_u,lung_u,[img_path_l, img_path_u]


if __name__ == '__main__':
    print ('1')