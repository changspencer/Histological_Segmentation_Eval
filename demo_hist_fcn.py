## Python standard libraries
from __future__ import print_function
from __future__ import division

# Comet ML logging package
try:
    from comet_ml import Experiment
except:
    Experiment = None

from os import path
import numpy as np
import os
import logging
import sys
import random
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

## Local external libraries
from Prepare_Data import Prepare_DataLoaders

# Training functions
from Utils.train import train_net
from Utils.models.hist_fcn import HistFCN

#Turn off plotting
plt.ioff()


def fcn_main(Params):
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
    num_classes = Params['num_classes']

    #Number of runs and/or splits for dataset
    numRuns = Params['Splits']

    # Detect if we have a GPU available
    use_cuda = Params['use_cuda'] and torch.cuda.is_available()
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
        model = HistFCN(
            Params['channels'],
            num_classes,
            n_bins=Params['numBins'],
            norm_count=Params['normalize_count'],
            norm_bins=Params['normalize_bins']
        )
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

if __name__ == "__main__":
    true_dir = path.abspath(path.dirname(__file__)) + "/"
    train_params = {'save_results': True,
                    'folder': true_dir + "Saved_Models/",
                    'Dataset': "Peanut_PRMI",
                    'imgs_dir': true_dir +'Datasets/PRMI/PRMI_official',
                    'masks_dir': true_dir + 'Datasets/PRMI/PRMI_official',
                    'num_workers': 1,
                    'mode': "Peanut_PRMI_Split_RandSeed_1",
                    'lr_rate': 0.005,
                    'optim': 'adam',
                    'momentum': 0.9,
                    'wgt_decay': 0,
                    'early_stop': 25,
                    'batch_size' : {'train': 8, 'val': 8, 'test': 8},
                    'train_class_lim': 100000,
                    'num_epochs': 200,
                    'patch_size': 640,
                    'padding': 0,
                    'normalize_count': True, 
                    'normalize_bins': True,
                    'numBins': 4,
                    'Model_name': "HistFCN",
                    'num_classes': 1,
                    'Splits': 1,
                    'feature_extraction': False,
                    'hist_model': "HistFCN",
                    'add_bn': False,
                    'pin_memory': True,
                    'folds': 1,
                    'fig_size': 12,
                    'font_size': 16,
                    'channels': 3,
                    'random_state': 1,
                    'save_cp': False,
                    'save_epoch': 10,
                    'use_attention': False,
                    'augment': False,
                    'rotate': False,
                    'use_pretrained': False,
                    'use_cuda': True}
    print("Testing the Histogram-Layer-infused mini-FCN")
    fcn_main(train_params)
    print("Finished training the HistFCN model...")