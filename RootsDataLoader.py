# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:55:40 2018

@author: weihuang.xu

Sourced from the GatorSense/EZNET GitHub Repository for working with PRMI data
"""
import os
import numpy as np
from PIL import Image

import logging
import pathlib
import torch
from torch.utils.data import Dataset


class RootsDataset(Dataset):
    def __init__(self, root, mode='RGB', img_transform=None, label_transform=None):
        self.root = root
        self.mode = mode
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []
        
        imgdir = os.path.join(self.root, 'images')

        for root, dirs, files in os.walk(imgdir):
            for name in files:
                if name != "placeholder.file":
                    imgpath = os.path.join(root, name)

                    rootp = pathlib.Path(root)
                    index = rootp.parts.index('images')
                    labelpth = pathlib.Path(os.path.join(root, 'masks_pixel_gt')).joinpath(*rootp.parts[index+1:])
                    labelpth = labelpth.joinpath(f"GT_{name}") 
                    self.files.append({
                            "img": imgpath, 
                            "label": labelpth.with_suffix('.png')
                    }) 
        
            for name in dirs:
                logging.info(f"Directory Walked: {name}")
                
    def __len__(self):     
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        
        img_file = datafiles["img"]
        
        if self.mode == 'RGB':
            img = Image.open(img_file).convert('RGB')
        if self.mode == 'gray':
            img = Image.open(img_file).convert('L')
            img = img.convert('RGB')
                    
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("1")
        
       
        state = torch.get_rng_state()
        if self.img_transform is not None:
            img = self.img_transform(img)

        
        torch.set_rng_state(state)
        if self.label_transform is not None:
            label = self.label_transform(label)  
       
        label = np.array(label)
        return img, label