# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:57:56 2020

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division

import torch

## Local external libraries
from .models.Histogram_Model import JOSHUA
from .models.unet_model import UNet
from .models.attention_unet_model import AttUNet
from .models.prmi_unet import PrmiUNet
from .models.fully_conv import myFCN

       
def initialize_model(model_name, num_classes, Network_parameters,
                    analyze=False, comet_exp=None):
 
    #Generate segmentation model 
    if model_name in ['JOSHUA', 'JOSHUA+', 'JOSHUAres', 'BNH']:
            model = JOSHUA(Network_parameters['channels'],num_classes,
                             skip=Network_parameters['histogram_skips'],
                             pool=Network_parameters['histogram_pools'],
                             bilinear=Network_parameters['bilinear'],
                             num_bins=Network_parameters['numBins'],
                             normalize_count=Network_parameters['normalize_count'],
                             normalize_bins=Network_parameters['normalize_bins'],
                             skip_locations=Network_parameters['skip_locations'],
                             pool_locations=Network_parameters['pool_locations'],
                             use_attention=Network_parameters['use_attention'],
                             feature_extraction=Network_parameters['feature_extraction'],
                             kernels=Network_parameters['hist_size'],
                             add_bn=Network_parameters['add_bn'],
                             analyze=analyze,
                             parallel=Network_parameters['parallel_skips'])

    # PRMI UNET model for the roots segmentation
    #!! Commented out to retain comparability with JOSHUA models
    elif (model_name == 'XuNET') and Network_parameters['Dataset'] in ['PRMI', 'Peanut_PRMI', 'PS_PRMI']: 
        model = PrmiUNet(num_classes, Network_parameters['channels'],
                         depth=5)

        if Network_parameters['use_pretrained']:
            state_dict = torch.load("P-EnDe-model.pth", map_location='cpu')
            model.load_state_dict(state_dict)
            print("Loaded pretrained P-EnDe-model.pth model successfully...")

    #Base UNET model or UNET+ (our version of attention)
    elif (model_name == 'UNET') or (model_name == 'UNET+'): 
        model = UNet(Network_parameters['channels'],num_classes,
                     bilinear = Network_parameters['bilinear'],
                     feature_extraction = Network_parameters['feature_extraction'],
                     use_attention=Network_parameters['use_attention'],
                     analyze=analyze)
    
    #Attetion UNET model introduced in 2018
    elif model_name == 'Attention_UNET':
            model = AttUNet(Network_parameters['channels'],num_classes,
                          bilinear = Network_parameters['bilinear'],
                          feature_extraction = Network_parameters['feature_extraction'],
                          use_attention=Network_parameters['use_attention'])

    #mini-FCn model introduced in 2016
    elif model_name == 'FCN':
            model = myFCN(Network_parameters['channels'], num_classes)
   
    else: #Show error that segmentation model is not available
        raise RuntimeError('Invalid model')


    return model
