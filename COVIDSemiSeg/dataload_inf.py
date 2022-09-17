# -*- coding: UTF-8 -*-
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import itertools
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import scipy
import cv2
import torch
from skimage import io
from skimage import transform as tsf

covid_semiseg_root = ""

def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = COVIDDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader


class COVIDDataset(data.Dataset):
    def __init__(self, mode, trainsize, use_edge=True, aug=False):
        self.mode=mode
        self.trainsize=trainsize
        self.use_edge=use_edge
        self.aug=aug
        if mode=="train":
            self.root = os.path.join(covid_semiseg_root, 'TrainingSet')
        elif mode=="test":
            self.root = os.path.join(covid_semiseg_root, 'TestingSet')

        self.unlabeled_img_path = os.path.join(self.root, "LungInfection-Train/Pseudo-label/Imgs")
        self.unlabeled_edge_path = os.path.join(self.root, "LungInfection-Train/Pseudo-label/Edge")
        self.unlabeled_gt_path = os.path.join(self.root, "LungInfection-Train/Pseudo-label/GT")
        self.unlabeled_lung_path = os.path.join(self.root, "LungInfection-Train/Pseudo-label/lung_mask")
        
        self.labeled_img_path = os.path.join(self.root,"LungInfection-Train/Doctor-label/Imgs")
        self.labeled_edge_path = os.path.join(self.root, "LungInfection-Train/Doctor-label/Edge")
        self.labeled_gt_path = os.path.join(self.root, "LungInfection-Train/Doctor-label/GT")
        self.labeled_lung_path = os.path.join(self.root, "LungInfection-Train/Doctor-label/lung_mask")


        self.labeled_img_list = self.sort_listdir(self.labeled_img_path)
        self.unlabeled_img_list = self.sort_listdir(self.unlabeled_img_path)
        self.labeled_edge_list = self.sort_listdir(self.labeled_edge_path)
        self.unlabeled_edge_list = self.sort_listdir(self.unlabeled_edge_path)
        self.labeled_gt_list = self.sort_listdir(self.labeled_gt_path)
        self.unlabeled_gt_list = self.sort_listdir(self.unlabeled_gt_path)
        self.labeled_lung_list = self.sort_listdir(self.labeled_lung_path)
        self.unlabeled_lung_list = self.sort_listdir(self.unlabeled_lung_path)

        self.img_list = self.labeled_img_list + self.unlabeled_img_list
        self.edge_list = self.labeled_edge_list + self.unlabeled_edge_list
        self.gt_list = self.labeled_gt_list + self.unlabeled_gt_list
        self.lung_list = self.labeled_lung_list + self.unlabeled_lung_list

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # print (index)
        # print (self.img_list[index])
        img=self.rgb_loader(self.img_list[index])
        gt=self.binary_loader(self.gt_list[index])
        img = self.img_transform(img)
        gt = self.gt_transform(gt)
        img = self.norm(img)
        lung = Image.fromarray(np.load(self.lung_list[index])[0])
        lung = self.gt_transform(lung)
#         print ("lung", lung.shape)
        if self.aug:
            img,lung,gt=self.augment(img,lung,gt)

        if self.use_edge:
            edge = self.binary_loader(self.edge_list[index])
            edge = self.gt_transform(edge)
            return img, lung, gt, edge, self.img_list[index]
        else:
            return img, lung, gt, self.img_list[index]

    def __len__(self):
        return len(self.img_list)
        # return 20

    def sort_listdir(self, path):
        lis=os.listdir(path)
        lis.sort()
        rlis = []
        for i in range(len(lis)):
            if os.path.isdir(lis[i]):
                continue
            rlis.append(os.path.join(path,lis[i]))

        return rlis

    def gray2rgb(self, img):
        return np.concatenate((img,img,img),axis=2)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def norm(self,np_array):
        np_array = np_array - np_array.min()
        np_array = np_array / (np_array.max() + 1e-7)
        return np_array

    def random_bright(self, im, delta=0.125):
        if random.random() < 0.5:
            delta = random.uniform(-delta, delta)
            im += delta
        return im

    def non_neg(self, np_array):
        np_array = np_array - np_array.min()

        return np_array

    def augment(self, glbimage, lung, glblabel):
        theta = (np.random.rand() - 0.5) * 60.0
        p0 = np.random.rand()
        p1 = np.random.rand()
        p2 = np.random.rand()
        glbimage = np.ascontiguousarray(glbimage).astype(np.float32)
        lung = np.ascontiguousarray(lung).astype(np.float32)
        glblabel = np.ascontiguousarray(glblabel).astype(np.float32)
        if p0 > 0.5:
            glbimage = np.flip(glbimage, axis=1)
            lung = np.flip(lung, axis=1)
            glblabel = np.flip(glblabel, axis=1)
        if p1 > 0.5:
            glbimage = np.flip(glbimage, axis=2)
            lung = np.flip(lung, axis=2)
            glblabel = np.flip(glblabel, axis=2)
        
        if p2 > 0.5:
            glbimage = scipy.ndimage.rotate(glbimage, theta, axes=(1, 2), reshape=False, mode="reflect")
            lung = scipy.ndimage.rotate(lung, theta, axes=(1, 2), reshape=False, mode="reflect", order=0)
            glblabel = scipy.ndimage.rotate(glblabel, theta, axes=(1, 2), reshape=False, mode="reflect", order=0)
        
        glbimage = np.ascontiguousarray(glbimage)
        lung = np.ascontiguousarray(lung)
        glblabel = np.ascontiguousarray(glblabel)
        
        glblabel[glblabel > 1] = 1
        glblabel[glblabel < 0] = 0

        glbimage = self.random_bright(glbimage)
        glblabel = np.round(glblabel)

        return glbimage, lung, glblabel


class COVIDDataset_test(data.Dataset):
    def __init__(self, mode, testsize):
        self.mode=mode
        self.trainsize=testsize

        self.root = os.path.join(covid_semiseg_root, 'TestingSet')
        self.labeled_img_path = os.path.join(self.root, "LungInfection-Test/Imgs")
        self.labeled_gt_path = os.path.join(self.root, "LungInfection-Test/GT")

        self.labeled_img_list = self.sort_listdir(self.labeled_img_path)
        self.labeled_gt_list = self.sort_listdir(self.labeled_gt_path)

        self.img_list = self.labeled_img_list
        self.gt_list = self.labeled_gt_list

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img=self.rgb_loader(self.img_list[index])
        img = self.img_transform(img)
        img = self.norm(img)
        gt=self.binary_loader(self.gt_list[index])
        gt = self.gt_transform(gt)

        return img, gt, self.img_list[index]

    def __len__(self):
        return len(self.img_list)

    def sort_listdir(self, path):
        lis=os.listdir(path)
        lis.sort()
        rlis = []
        for i in range(len(lis)):
            if os.path.isdir(lis[i]):
                continue
            rlis.append(os.path.join(path,lis[i]))

        return rlis

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def norm(self,np_array):
        np_array = np_array - np_array.min()
        np_array = np_array / (np_array.max() + 1e-7)
        return np_array

    def non_neg(self,np_array):
        np_array = np_array - np_array.min()
        return np_array


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )
    #每次调出两个。

    def __len__(self): #以有label的图的数量除batch size 当成主要的迭代数
        # print (len(self.primary_indices) // self.primary_batch_size)
        return len(self.primary_indices) // self.primary_batch_size
        # return len(self.secondary_indices) // self.secondary_batch_size
        # return 1



class indexBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, indices, batch_size):
        self.primary_indices = indices
        self.primary_batch_size = batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        return (primary_batch for (primary_batch) in grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self): #以有label的图的数量除batch size 当成主要的迭代数
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)
#这个函数只对list的第一维重排，所以对一维的比较有效


def iterate_eternally(indices):
    def infinite_shuffles():
        while True: #反复调用，上面的有label图是只调用一次
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)