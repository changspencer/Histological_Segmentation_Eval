# -*- coding: utf-8 -*-
"""
This pytorch custom dataset was modified from code in this repository:
https://github.com/jeromerony/survey_wsl_histology. Please cite their work:
    
@article{rony2019weak-loc-histo-survey,
  title={Deep weakly-supervised learning methods for classification and localization in histology images: a survey},
  author={Rony, J. and Belharbi, S. and Dolz, J. and Ben Ayed, I. and McCaffrey, L. and Granger, E.},
  journal={coRR},
  volume={abs/1909.03354},
  year={2019}
}
@author: jpeeples 
"""

from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from .utils import check_files
from torchvision.transforms import functional as F
from random import random, randint, setstate, getstate
from os.path import splitext
from os import listdir
import os
import pdb
import pathlib
from glob import glob
import itertools
import numpy as np
import torch

#Update load_data function
def load_data(samples, resize=None, min_resize=None):
    images = {}
    for image_path in samples:
        image = Image.open(image_path)
        if resize is not None:
            image = image.resize(resize, resample=Image.LANCZOS)
        elif min_resize:
            image = F.resize(image, min_resize, interpolation=Image.LANCZOS)
        images[image_path] = image.copy()
    return images

class PhotoDataset(Dataset):
    def __init__(self, data_path, files, transform, mask_transform, augment=False, 
                 patch_size=None, rotate=False,
                 preload=False, resize=None, min_resize=None,class_label=True,
                 label_path=None,img_ext='.jpg',mask_ext='.png'):
        self.transform = transform
        self.mask_transform = mask_transform
        self.resize = resize
        self.min_resize = min_resize
        self.augment = augment
        self.patch_size = patch_size
        self.rotate = rotate
        self.masks_dir = label_path
        self.class_label = class_label
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        
        #Get rgb mapping for CSAS data
        self.rgb = [(86, 115, 181), (132, 167, 77), (77, 77, 77), (141, 202, 207),
                    (211, 150, 202), (209, 205, 75), (255, 196, 52), (51, 69, 83),
                    (145, 58, 219), (58, 69, 219), (80, 139, 48), (164, 38, 41)]
        self.mapping = {tuple(c): t for c, t in zip(self.rgb, range(len(self.rgb)))}

       #Updated from previous loader
        if class_label:
            self.samples = [(os.path.join(data_path,file[0]), os.path.join(data_path,file[1]), 
                         file[2]) for file in files]
        else:
            self.samples = [(os.path.join(data_path,file[0].replace(" ", "")), os.path.join(label_path,file[0].replace(" ", "")), 
                          file[2]) for file in files]
            
            self.ids = [splitext(file)[0] for file in listdir(data_path)
                        if (not file.startswith('.') and os.path.isfile(os.path.join(data_path,file)))]
                   
        self.n = len(self.samples)
            

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, _, _ in self.samples],
                                    resize=resize, min_resize=min_resize)
            self.masks = load_data([mask_path for _, mask_path, _ in self.samples if mask_path != ''],
                                   resize=resize, min_resize=min_resize)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded with {} images'.format(len(self.images.keys())))

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        img_name = image_path.rsplit('/',1)[-1].rsplit('.',1)[0]
        if self.preloaded:
                image = self.images[image_path].convert('RGB')
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                image = Image.open(image_path+self.img_ext).convert('RGB')
            
            if self.resize is not None:
                image = image.resize(self.resize, resample=Image.LANCZOS)
            elif self.min_resize is not None:
                image = F.resize(image, self.min_resize, interpolation=Image.LANCZOS)
        image_size = image.size # to generate the mask if there is no file

        if mask_path == '':
            mask = Image.new('L', image_size)
        else:
            if self.preloaded:
                mask = self.masks[mask_path].convert('L')
            else:
                try:
                    mask = Image.open(mask_path).convert('L')
                except:
                    mask = Image.open(mask_path+self.mask_ext).convert('RGB')
                    
                if self.resize is not None:
                    mask = mask.resize(self.resize, resample=Image.LANCZOS)
                elif self.min_resize is not None:
                    mask = F.resize(mask, self.min_resize, interpolation=Image.LANCZOS)

        if self.augment:

            # extract patch
            if self.patch_size is not None:
                left = randint(0, image_size[0] - self.patch_size)
                up = randint(0, image_size[1] - self.patch_size)
                image = image.crop(box=(left, up, left + self.patch_size, up + self.patch_size))
                mask = mask.crop(box=(left, up, left + self.patch_size, up + self.patch_size))

            # rotate
            if self.rotate:
                angle = randint(0, 3) * 90
                image = image.rotate(angle)
                mask = mask.rotate(angle)
                    

            # flip
            if random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            image = self.transform(image)
         
            mask = self.mask_transform(mask)
            if self.class_label:
                #Preprocess to be binary
                mask = (mask != 0).long()
            else:
                #Clean up masks, software leaves two labels (1,2) for fat
                mask[mask>=.5] = 1
                mask[mask<.5] = 0

        return {'image':image,'mask': mask, 'index': img_name, 'label': label}


class RootsDataset(Dataset):
    """
    Created on Mon May 21 14:55:40 2018

    @author: weihuang.xu

    Sourced from the GatorSense/EZUNET GitHub Repository for working with PRMI data.
    Some changes have been made to adapt it to Josh's Histological Segmentation code.
    """
    def __init__(self, root, mode='RGB', img_transform=None, label_transform=None,
                 subset:list=None, class_count_lim:int=None):

        self.class_list = [
            'Cotton',
            'Papaya',
            'Peanut',
            'Sesame',
            'Sunflower',
            'Switchgrass'
        ]
        if subset is not None:
            self.class_list = subset

        self.root = root
        self.mode = mode
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []

        self.class_count = np.zeros(len(self.class_list), dtype=np.int)
        imgdir = os.path.join(self.root, 'images')

        for os_root, dirs, files in os.walk(imgdir):
            if os_root.find("_noMask") > -1 or os_root.find("- Copy") > -1 or os_root.find("_balanced") > -1:
                continue

            curr_class_idx = None
            for class_idx in range(len(self.class_list)):
                if self.class_list[class_idx] in os_root:
                    curr_class_idx = class_idx
                    break

            # The subset of classes does not include 'os_root'
            if curr_class_idx is None:
                continue

            for name in files:
                if name == "placeholder.file" or ".db" in name:
                    continue

                imgpath = os.path.join(os_root, name)

                rootp = pathlib.Path(os_root)
                index = rootp.parts.index('images')
                labelpath = os.path.join(self.root, 'masks_pixel_gt', *rootp.parts[index+1:])
                labelpath = os.path.join(labelpath, f"GT_{name}")
                # Label name validation
                if labelpath.endswith('.png'):
                    self.files.append({
                        "img": imgpath,
                        "label": labelpath
                    })
                else:
                    ending = '.' + labelpath.split('.')[-1]
                    self.files.append({
                        "img": imgpath,
                        "label": labelpath.replace(ending, '.png')
                    })
                self.class_count[curr_class_idx] += 1

                # Cut-off the number of exemplars to use per class
                if class_count_lim is not None and self.class_count[curr_class_idx] >= class_count_lim:
                    break

        # Increase the number of times an underrepresented class is sampled
        file_idx = 0
        self.sample_weights = np.zeros(self.class_count.sum())
        for count in self.class_count:
            class_weight = 0 if count == 0 else self.class_count.max() / count

            self.sample_weights[file_idx:file_idx + count] = class_weight
            file_idx += count

    def __len__(self):     
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Get specific filename (not filepath)
        img_file = datafiles["img"]
        img_name = pathlib.PurePath(img_file).name.rsplit('.', 1)[0]

        if self.mode == 'RGB':
            img = Image.open(img_file).convert('RGB')
        if self.mode == 'gray':
            img = Image.open(img_file).convert('L')
            img = img.convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("1")

        # Have the longer side always put first
        if img.size[0] < img.size[1]:
            img = img.transpose(method=Image.ROTATE_90)
            label = label.transpose(method=Image.ROTATE_90)

        torch_state = torch.get_rng_state()
        python_state = getstate()  # For QuantizedRotation
        if self.img_transform is not None:
            img = self.img_transform(img)

        torch.set_rng_state(torch_state)
        setstate(python_state)  # For QuantizedRotation
        if self.label_transform is not None:
            label = self.label_transform(label)
        label = np.array(label)

        # Need to flip PRMI roots labeling (make 0 be soil)
        # label = -label + label.max()    # This was used for Wei's old PRMI dataset location
        return {'image':img, 'mask': label, 'index': img_name, 'label': label_file}


class SitsDataset(Dataset):
    """
    Created on Tue Nov 8 09:12:40 2022

    @author: chang.spencer

    Adapted from the RootsDataset class above.
    """
    def __init__(self, root, mode='RGB', img_transform=None, label_transform=None,
                 subset:list=None):

        self.class_list = [
            'Peanut',
            'SweetCorn',
            'Coffee'
        ]
        if subset is not None:
            self.class_list = subset

        self.root = root
        self.mode = mode
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []

        self.class_count = np.zeros(len(self.class_list), dtype=np.int)
        imgdir = os.path.join(self.root, 'images')
    
        for os_root, dirs, files in os.walk(imgdir):
            if os_root.find("_noMask") > -1:
                continue

            curr_class_idx = None
            for class_idx in range(len(self.class_list)):
                if self.class_list[class_idx] in os_root:
                    curr_class_idx = class_idx
                    break

            # The subset of classes does not include 'os_root'
            if curr_class_idx is None:
                continue

            for name in files:
                imgpath = os.path.join(os_root, name)

                rootp = pathlib.Path(os_root)
                index = rootp.parts.index('images')
                labelpath = os.path.join(self.root, 'masks_pixel_gt', *rootp.parts[index+1:])
                labelpath = os.path.join(labelpath, f"{name.rsplit('.', 1)[0]}_mask.png")

                # Label name validation
                if labelpath.endswith('.png'):
                    self.files.append({
                        "img": imgpath,
                        "label": labelpath
                    })
                else:
                    ending = '.' + labelpath.split('.')[-1]
                    self.files.append({
                        "img": imgpath,
                        "label": labelpath.replace(ending, '.png')
                    })
                self.class_count[curr_class_idx] += 1

        # Increase the number of times an underrepresented class is sampled
        file_idx = 0
        self.sample_weights = np.zeros(self.class_count.sum())
        for count in self.class_count:
            class_weight = 0 if count == 0 else self.class_count.max() / count

            self.sample_weights[file_idx:file_idx + count] = class_weight
            file_idx += count

    def __len__(self):     
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Get specific filename (not filepath)
        img_file = datafiles["img"]
        img_name = pathlib.PurePath(img_file).name.rsplit('.', 1)[0]

        if self.mode == 'RGB':
            img = Image.open(img_file).convert('RGB')
        if self.mode == 'gray':
            img = Image.open(img_file).convert('L')
            img = img.convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("1")

        # print(img.size)
        if img.size[0] < img.size[1]:
            img = img.transpose(method=Image.ROTATE_90)
            label = label.transpose(method=Image.ROTATE_90)
            # print("ROTATED")

        state = torch.get_rng_state()
        if self.img_transform is not None:
            img = self.img_transform(img)

        torch.set_rng_state(state)
        if self.label_transform is not None:
            label = self.label_transform(label)
        label = np.array(label)

        return {'image':img, 'mask': label, 'index': img_name, 'label': label_file}