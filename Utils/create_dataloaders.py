# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:35:49 2021
Generate Dataloaders
@author: jpeeples
@editor: changspencer
"""
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from .utils import ExpandedRandomSampler
from .dataset import PhotoDataset, RootsDataset, SitsDataset
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class QuantizedRotation:
    """Rotate by one of the given angles.
    Originally presented as example custom transforms at
        https://pytorch.org/vision/stable/transforms.html.
    """
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):

        angle = random.choice(self.angles)
        return TF.rotate(x, angle, fill=0)


def Get_Dataloaders(split,indices,Network_parameters,batch_size):
    
    
    if Network_parameters['Dataset'] == 'GlaS':
        train_loader, val_loader, test_loader = load_glas(Network_parameters['imgs_dir'],
                                                          indices,batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'])
        pos_wt = 1
        
    elif Network_parameters['Dataset'] == 'SFBHI':
        train_loader, val_loader, test_loader = load_SFBHI(Network_parameters['imgs_dir'],
                                                          indices,batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'],
                                                          patch_size=Network_parameters['center_size'],
                                                          label_path=Network_parameters['masks_dir'])
       
        #Get postive weight (for histological fat images only)
        pos_wt = 3
        
    elif Network_parameters['Dataset'] == 'PRMI':
        train_loader, val_loader, test_loader = load_PRMI(Network_parameters['imgs_dir'],
                                                          batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'],
                                                          patch_size=Network_parameters['patch_size'],
                                                          train_class_lim=Network_parameters['train_class_lim'])
       
        #Get postive weight (for histological fat images only)
        pos_wt = 1
        
    elif Network_parameters['Dataset'] == 'Peanut_PRMI':
        train_loader, val_loader, test_loader = load_PRMI(Network_parameters['imgs_dir'],
                                                          batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'],
                                                          patch_size=Network_parameters['patch_size'],
                                                          data_subset=['Peanut'],
                                                          train_class_lim=Network_parameters['train_class_lim'])
       
        #Get postive weight (for histological fat images only)
        pos_wt = 1
        
    elif Network_parameters['Dataset'] == 'SiTS':
        train_loader, val_loader, test_loader = load_sits(Network_parameters['imgs_dir'],
                                                          batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'])
       
        #Get postive weight (for histological fat images only)
        pos_wt = 1
        
    elif Network_parameters['Dataset'] == 'SiTS_crop':
        train_loader, val_loader, test_loader = load_sits(Network_parameters['imgs_dir'],
                                                          batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'])
       
        #Get postive weight (for histological fat images only)
        pos_wt = 1
       
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    return dataloaders, pos_wt
    
def load_glas(data_path,indices, batch_size, num_workers, pin_memory=True,
              split=0, patch_size=416,sampler_mul=8, augment=False, rotate=False):
  
    test_transform = transforms.ToTensor()

    train_loader = DataLoader(
        PhotoDataset(
            data_path= data_path,
            files=indices['train'][split],
            patch_size=patch_size,
            # augment=augment,
            # rotate=rotate,
            transform=transforms.Compose([
                #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor(),
            ]),
            mask_transform=transforms.ToTensor(),
            preload=False,
        ),
        batch_size=batch_size['train'],
        num_workers=num_workers,
        #sampler=ExpandedRandomSampler(len(indices['train'][split]), sampler_mul),
        pin_memory=pin_memory,
        drop_last=True,worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['val'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False),
        batch_size=1, num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['test'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False),
        batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    
    return train_loader, valid_loader, test_loader

def load_SFBHI(data_path,indices, batch_size, num_workers, pin_memory=True,
              split=0, patch_size=416,sampler_mul=8, augment=False, rotate=False,
              label_path=None):
    test_transform = transforms.ToTensor()
    
    train_loader = DataLoader(
        PhotoDataset(
            data_path= data_path,
            files=indices['train'][split],
            patch_size=patch_size,
            # augment=augment,
            # rotate=rotate,
            transform=transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor(),
            ]),
            mask_transform=transforms.ToTensor(),
            preload=False,class_label=False,label_path=label_path
        ),
        batch_size=batch_size['train'],
        num_workers=num_workers,
        # sampler=ExpandedRandomSampler(len(indices['train'][split]), sampler_mul),
        pin_memory=pin_memory,
        drop_last=True,worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['val'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False,
                     class_label=False,label_path=label_path),
        batch_size=batch_size['val'], num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['test'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False,
                     class_label=False,label_path=label_path),
        batch_size=batch_size['test'], num_workers=num_workers,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    
    return train_loader, valid_loader, test_loader


def load_PRMI(data_path, batch_size, num_workers, pin_memory=True,
              split=0, patch_size=640, sampler_mul=8, augment=False, rotate=False,
              data_subset=None, train_class_lim:int=None):

    # Resize to some 4:3 ratio because PRMI data is in 4:3 ratio.
    # center_crop = [transforms.CenterCrop((patch_size * 3 // 4, patch_size))]
    resize_transform = [transforms.Resize((patch_size, patch_size))]
    crop_transform = [transforms.RandomResizedCrop((patch_size, patch_size),
                                                   scale=(0.1, 1.0),
                                                   ratio=(1, 1))]
                                                #    ratio=(0.75, 0.75))]
    misc_transform = [
        QuantizedRotation(angles=[0, 90, 180, 270])
    ]
    test_crop = [transforms.RandomCrop((480, 480))]

    # Normalizing values taken from manual image analysis of images
    if data_subset == ["Peanut"]:
        prmi_mean = (0.5073, 0.4775, 0.4381)
        prmi_dev = (0.1463, 0.1448, 0.1424)
    elif data_subset == ["Peanut", "Switchgrass"]:
        prmi_mean = (0.4910, 0.4621, 0.4246)
        prmi_dev = (0.1338, 0.1321, 0.1297)
    else:  # data_subset is None
        prmi_mean = (0.5075, 0.4687, 0.4296)
        prmi_dev = (0.1302, 0.1275, 0.1245)

    # Train data transforms: Resizing and maybe some data augmentation
    if augment:
        train_transform = crop_transform + misc_transform + [
            transforms.ToTensor(),
            transforms.Normalize(prmi_mean, prmi_dev)
        ]
        # Mask transforms: resizing only
        gt_transforms = transforms.Compose(crop_transform + misc_transform +
                                           [transforms.ToTensor()])
    else:
        random_crop = [transforms.RandomCrop((patch_size * 3 // 4, patch_size))]
        train_transform = random_crop + [transforms.ToTensor(),
                                         transforms.Normalize(prmi_mean, prmi_dev)]
        gt_transforms = transforms.Compose(random_crop +
                                           [transforms.ToTensor()])
    # Test data transforms: resizing only
    test_transform = transforms.Compose(test_crop +
                                        [transforms.ToTensor(),
                                         transforms.Normalize(prmi_mean, prmi_dev)])
    test_gt_transform = transforms.Compose(test_crop +
                                        [transforms.ToTensor()])

    # Have a uniform sampling of classes for each batch
    train_dataset = RootsDataset(
        root=data_path + "/train",
        img_transform=transforms.Compose(train_transform),
        label_transform=gt_transforms,
        subset=data_subset,
        class_count_lim=train_class_lim
    )
    train_sampler = WeightedRandomSampler(train_dataset.sample_weights,
                                          len(train_dataset.files),
                                          replacement=False)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size['train'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        RootsDataset(root=data_path + "/val",
                     img_transform=test_transform,
                     label_transform=test_gt_transform,
                     subset=data_subset),
        batch_size=batch_size['val'],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        RootsDataset(root=data_path + "/test",
                     img_transform=test_transform,
                     label_transform=test_gt_transform,
                     subset=data_subset),
        batch_size=batch_size['test'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    
    print("Dataloader results: {}, {}, {}".format(len(train_loader),
                                                  len(valid_loader),
                                                  len(test_loader)))
    return train_loader, valid_loader, test_loader


def load_sits(data_path, batch_size, num_workers, pin_memory=True,
              split=0, patch_size:int=None, sampler_mul=8, augment=False, rotate=False,
              data_subset=None):
    resize_transform = [transforms.Resize((patch_size, patch_size))]
    center_crop = [transforms.CenterCrop((60, 96))]

    # Normalizations found manually for the uncropped image size
    sits_mean = (0.2870, 0.2238, 0.1639)
    sits_dev = (0.0975, 0.0914, 0.0713)

    # Train data transforms: Resizing and maybe some data augmentation
    if augment:
        crop_transform = [
            transforms.RandomCrop((patch_size, patch_size))
        ]
        train_transform = transforms.Compose(
            crop_transform + [
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(sits_mean, sits_dev)
        ])
        # Mask transforms: 
        gt_transforms = transforms.Compose(crop_transform +
                                           [transforms.ToTensor()])
    else:
        train_transform = transforms.Compose(center_crop +
                                             [transforms.ToTensor(),
                                              transforms.Normalize(sits_mean, sits_dev)])
        gt_transforms = transforms.Compose(center_crop + 
                                           [transforms.ToTensor()])
    # Test data transforms: resizing only - Commented out for now; all images same size
    test_transform = transforms.Compose(# center_crop +
                                        [transforms.ToTensor(),
                                         transforms.Normalize(sits_mean, sits_dev)])
    test_gt_transform = transforms.Compose(# center_crop +
                                        [transforms.ToTensor()])

    # Have a uniform sampling of classes for each batch
    train_dataset = SitsDataset(
        root=data_path + "/train",
        img_transform=train_transform,
        label_transform=gt_transforms,
        subset=data_subset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size['train'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        SitsDataset(root=data_path + "/val",
                     img_transform=test_transform,
                     label_transform=test_gt_transform,
                     subset=data_subset),
        batch_size=batch_size['val'],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        SitsDataset(root=data_path + "/test",
                     img_transform=test_transform,
                     label_transform=test_gt_transform,
                     subset=data_subset),
        batch_size=batch_size['test'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    
    print("Dataloader results: {}, {}, {}".format(len(train_loader),
                                                  len(valid_loader),
                                                  len(test_loader)))
    return train_loader, valid_loader, test_loader
