# -*- coding: utf-8 -*-
"""
Main demo script for histological segmentation models. 
Used to train all models (modify line 204 to select certain models)
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division

# Comet ML logging package
try:
    from comet_ml import Experiment
except:
    Experiment = None

import numpy as np
import os
import argparse
import logging
import sys
import random
import matplotlib.pyplot as plt

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Initialize_Model import initialize_model
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
from Utils.Save_Results import save_params

#UNET functions
from Utils.train import train_net

import pdb

#Turn off plotting
plt.ioff()

def main(Params, args):
    # Reproducibility and option for cross-validation runs (no initial seed)
    if Params['random_state'] > 0:
        torch.manual_seed(Params['random_state'])
        np.random.seed(Params['random_state'])
        random.seed(Params['random_state'])
        torch.cuda.manual_seed(Params['random_state'])
        torch.cuda.manual_seed_all(Params['random_state'])
    else:
        print(f"Initial Torch seed: {torch.seed()}")
        
    #Name of dataset
    Dataset = Params['Dataset']
    
    #Model(s) to be used
    model_name = Params['Model_name']
    
    #Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
                                     
    #Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]

    # Detect if we have a GPU available
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print()
    print('Starting Experiments...')
    
    # Create training and validation dataloaders
    print("Initializing Datasets and Dataloaders...")
    
    #Return indices of training/validation/test data
    Params['imgs_dir'] = os.path.join(os.path.dirname(__file__),
                                      Params['imgs_dir'])
    indices = Prepare_DataLoaders(Params,numRuns)
    
    #Loop counter
    split = 0
    
    for split in range(0, numRuns):
        
        if Experiment is None:  # Comet-ML Logging initialization
            experiment = None
        else:  # Not yet sure whether to also use this for logging HistRes inits
            proj_name = 'segmentation'
            experiment = Experiment(
                api_key="cf2AdIgBb4jLjQZHyCyWoo2k2",
                project_name=proj_name,
                workspace="changspencer",
            )
            experiment.set_name(f"{Dataset}-{model_name}-{split+1}")
        
        print('Starting Experiments...')
        if experiment is not None:
            experiment.log_parameters(Params)
            # save_params(Params, split)
        else:
            # save_params(Params, split)
            print("NOTE: No Comet Experiment import found...")

        # Initialize the segmentation model for this run
        model = initialize_model(model_name, num_classes, Params)
        if experiment is not None:
            experiment.set_model_graph(model)
        
        # Send the model to GPU if available, use multiple if available
        if use_cuda and torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        # Send the model to GPU if available
        model = model.to(device)
        
        #Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        # Train and evaluate
        try:
            if torch.cuda.device_count() > 1:
                n_channels = model.module.n_channels
                n_classes = model.module.n_classes
                bilinear = model.module.bilinear
            else:
                n_channels = model.n_channels
                n_classes = model.n_classes
                bilinear = model.bilinear
                
            logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
            logging.info(f'Using device {device}')
            logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if bilinear else "Transposed conv"} upscaling\n'
                 f'\tTotal number of trainable parameters: {num_params}')
               
            train_net(net=model,
                      device=device,
                      indices=indices,
                      split=split,
                      Network_parameters=Params,
                      epochs=Params['num_epochs'],
                      batch_size=Params['batch_size'],
                      lr=Params['lr_rate'],
                      save_cp=Params['save_cp'],
                      save_results=Params['save_results'],
                      save_epoch=Params['save_epoch'],
                      comet_exp=experiment)
        
        except KeyboardInterrupt:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
      
        torch.cuda.empty_cache()
            
        if Params['hist_model'] is not None:
            print('**********Run ' + str(split + 1) + ' For ' +
                  Params['hist_model'] + ' Finished**********') 
        else:
            print('**********Run ' + str(split + 1) + ' For ' + model_name + 
                  ' Finished**********') 
            
        #Iterate counter
        split += 1
       
def parse_args():
    # 'UNET'
    # 'Attention UNET'
    # 'UNET+'
    # 'JOSHUA'
    # 'JOSHUA+'
    parser = argparse.ArgumentParser(description='Run segmentation models for dataset')
    parser.add_argument('--save_results', type=bool, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--save_cp', type=bool, default=False,
                        help='Save results of experiments at each checkpoint (default: False)')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='Epoch for checkpoint (default: 5')
    parser.add_argument('--folder', type=str, default='Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, nargs="+", default=['UNET'],
                        help='Select models to train with (UNET, UNET+, Attention_UNET, JOSHUA, JOSHUA+, JOSHUAres, XuNET, FCN, BNH) default: [UNET]')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection:  1: SFBHI, 2: GlaS, 3: PRMI, 4: Peanut_PRMI, 5: Peanut-Switchgrass (PS_PRMI), 6: SitS')
    parser.add_argument('--channels', type=int, default=3,
                        help='Input channels of network (default: 3, RGB images)')
    parser.add_argument('--bilinear', type=bool, default=False,
                        help='Upsampling feature maps, set to True - bilinear interpolation. False - learn transpose convolution (consume more memory) (default: False)')
    parser.add_argument('--augment', type=bool, default=False,
                        help='Data augmentation (default: True)')
    parser.add_argument('--rotate', type=bool, default=True,
                        help='Training data will be rotated, random flip (p=.5), random patch extraction (default:True')
    parser.add_argument('--numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', type=bool, default=False,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', type=bool, default=False,
                        help='Flag to use pretrained model from ImageNet or train from scratch, only works for XuNET (default: False)')
    parser.add_argument('--data_split', type=str, default='Normal',
                        help='whether to split SFBHI at time split (default: Normal)')
    parser.add_argument('--train_class_lim', type=int, default=None,
                        help='maximum number of exemplars per class for training (default: None)')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val_batch_size', type=int, default=10,
                        help='input batch size for validation (default: 10)')
    parser.add_argument('--test_batch_size', type=int, default=10,
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs to train each model for (default: 150)')
    parser.add_argument('--random_state', type=int, default=1,
                        help='Set random state for K fold CV for repeatability of data/model initialization (default: 1)')
    parser.add_argument('--hist_size', type=int, default=0,
                        help="Option for different histogram bin averaging sizes (0 == [2,2,2,2], 1 == [3,3,2,2], default: 0)")
    parser.add_argument('--add_bn', type=bool, default=False,
                        help='Add batch normalization before histogram layer(s) (default: False)')
    parser.add_argument('--padding', type=int, default=0,
                        help='If padding is desired, enter amount of zero padding to add to each side of image  (default: 0)')
    parser.add_argument('--normalize_count',type=bool, default=True,
                        help='Set whether to use sum (unnormalized count) or average pooling (normalized count) (default: True)')
    parser.add_argument('--normalize_bins',type=bool, default=True,
                        help='Set whether to enforce sum to one constraint across bins (default: True)')
    parser.add_argument('--patch_size', type=int, default=640,
                        help='Patch size of random image crops; aspect is 3:4. (default: 640)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.8,
                        help='SGD momentum (default: 0.8)')
    parser.add_argument('--wgt_decay', type=float, default=1e-5,
                        help='optimizer weight decay (default: 1e-5)')
    parser.add_argument('--optim', type=str, default='adam',
                        help='learning optimizer - sgd, adamax, adam (default: Adam)')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='number of epochs to wait for early stopping (default: 10)')
    parser.add_argument('--parallelize_model', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of random weight initializations and training runs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    #Trains all models
    args = parse_args()
    model_list = args.model
    args.folder = os.path.join(os.path.dirname(__file__), args.folder)

    model_count = 0
    for model in model_list:
        setattr(args, 'model', model)
        params = Parameters(args)
        main(params, args)
        model_count += 1
        print('Finished Model {} of {}'.format(model_count,len(model_list)))
        
