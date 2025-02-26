# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:46:58 2021

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import jaccard_score as jsc
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Demo_Parameters import Parameters
from Utils.Initialize_Model import initialize_model
from Utils.functional import *
from Utils.decode_segmentation import decode_segmap
from Utils.metrics import eval_metrics

def Generate_Dir_Name(split,Network_parameters):
    
    if Network_parameters['hist_model'] is not None:
        dir_name = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' 
                    + Network_parameters['hist_model'] + '/Run_' 
                    + str(split + 1) + '/')
    #Baseline model
    else:
        dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' +
                    Network_parameters['Model_name'] 
                    + '/Run_' + str(split + 1) + '/')  
    
    #Location to save figures
    fig_dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/')
        
    return dir_name, fig_dir_name

def inverse_normalize(tensor, mean=(0,0,0), std=(1,1,1)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def Generate_Images(dataloaders,mask_type,seg_models,device,split,
                    num_classes,fat_df,args,show_fat=False,alpha=.35,class_name='Fat'):

    model_names = []
    
    #Set names of models
    for eval_model in seg_models:
        model_names.append(eval_model)
    
    hausdorff_pytorch = HausdorffDistance()
    for phase in ['val','test']:
        print(f"{phase} Image Phase: {len(dataloaders[phase])} images")
        img_count = 0
        for batch in dataloaders[phase]:
           
            imgs, true_masks, idx = (batch['image'], batch['mask'],
                                              batch['index'])
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
           
            # for img in range(0,imgs.size(0)):
            for img in range(0, 1):  # Limit number of outputs to expedite eval
        
                #Create figure for each image
                temp_fig, temp_ax = plt.subplots(nrows=1,ncols=len(seg_models)+2,figsize=(16,8))
                
                #Initialize fat array
                if show_fat:
                    temp_fat = np.zeros(len(seg_models)+1)
                
                #Get conversion rate from pixels to fat
                if show_fat:
                    temp_org_size = fat_df.loc[fat_df['Image Name (.tif)'].replace(" ", "")==idx[img].replace(" ", "")]['# of Pixels'].iloc[-1]
                    temp_ds_size = fat_df.loc[fat_df['Image Name (.tif)'].replace(" ", "")==idx[img].replace(" ", "")]['Down sampled # of Pixels'].iloc[-1]
                    temp_org_rate = fat_df.loc[fat_df['Image Name (.tif)'].replace(" ", "")==idx[img].replace(" ", "")]['Reference Length (um/px)'].iloc[-1]
        
                #Plot images, hand labels, and masks
                temp_ax[0].imshow(imgs[img].cpu().permute(1, 2, 0))
                temp_ax[0].tick_params(axis='both', labelsize=0, length = 0)
                
                if num_classes == 1:
                    temp_ax[1].imshow(imgs[img].cpu().permute(1,2,0))
                    M, N = true_masks[img][0].shape
                    temp_overlap = np.zeros((M,N,3))
                    gt_mask = true_masks[img][0].cpu().numpy().astype(dtype=bool)
                    temp_overlap[gt_mask,:] = [5/255, 133/255, 176/255]
                    temp_ax[1].imshow(temp_overlap,'jet',interpolation=None,alpha=alpha)
                    temp_ax[1].tick_params(axis='both', labelsize=0, length = 0)
                else:
                    temp_ax[0,1].imshow(imgs[img].cpu().permute(1,2,0))
                    temp_true = decode_segmap(true_masks[img].cpu().numpy())
                    temp_ax[0,1].imshow(temp_true,interpolation=None,alpha=alpha)
                    temp_ax[0,1].tick_params(axis='both', labelsize=0, length = 0)
                    temp_mask = decode_segmap(true_masks[img].cpu().numpy(),nc=num_classes)
                    temp_ax[1,1].imshow(temp_mask)
                    temp_ax[1,1].tick_params(axis='both', labelsize=0, length = 0)
                axes = temp_ax
                        
                #Compute percentage of fat from ground truth
                if show_fat:
                    temp_fat[0] = true_masks[img][0].count_nonzero().item() * (temp_org_size/temp_ds_size) * (temp_org_rate)**2
                 
                #Labels Rows and columns
                if num_classes == 1:
                    col_names = [idx[img], 'Ground Truth'] + model_names
                else:
                    col_names = ['Input Image', 'Ground Truth'] + model_names
                cols = ['{}'.format(col) for col in col_names]
                
                for ax, col in zip(axes, cols):
                    ax.set_title(col)
            
                
                # Initialize the segmentation model for this run
                for key_idx, eval_model in enumerate(seg_models):
                    
                    setattr(args, 'model', eval_model)
                    temp_params = Parameters(args)
                    model = initialize_model(eval_model, num_classes,temp_params)
                    sub_dir, fig_dir = Generate_Dir_Name(split, temp_params)

                    print(" -- Evaluating {} model...".format(eval_model))

                    #If parallelized, need to set model
                      # Send the model to GPU if available
                    try:
                        model = nn.DataParallel(model)
                        model = model.to(device)
                        best_weights = torch.load(sub_dir + 'best_wts.pt',
                                                  map_location=torch.device(device))

                        # Reloaded weights - data validation
                        best_wts_dict = OrderedDict()
                        for key in best_weights.keys():
                            if not key.startswith('module.'):
                                best_wts_dict['module.' + key] = best_weights[key]
                            else:
                                best_wts_dict[key] = best_weights[key]

                        model.load_state_dict(torch.load(sub_dir + 'best_wts.pt', 
                                                map_location=torch.device(device)))
                    except:
                        model = model.to(device)
                        best_weights = torch.load(sub_dir + 'best_wts.pt', 
                                                  map_location=torch.device(device))

                        # Reloaded weights - data validation
                        best_wts_dict = OrderedDict()
                        for key in best_weights.keys():
                            if not key.startswith('module.'):
                                best_wts_dict['module.' + key] = best_weights[key]
                            else:
                                best_wts_dict[key] = best_weights[key]

                        model.load_state_dict(best_wts_dict)
                    
                    #Get location of best weights
                    sub_dir, fig_dir = Generate_Dir_Name(split, temp_params)
                    
                    #Get output and plot
                    model.eval()
                    
                    with torch.no_grad():
                        preds = model(imgs[img].unsqueeze(0))
                    
                    if num_classes == 1:
                        preds = (torch.sigmoid(preds) > .5).float()
                        
                        #Plot masks only
                        M, N = true_masks[img][0].shape
                        temp_overlap = np.zeros((M,N,3))
                        preds_mask = preds[0].cpu().permute(1,2,0)[:,:,0].numpy().astype(dtype=bool)
                        gt_mask = true_masks[img][0].cpu().numpy().astype(dtype=bool)
                        temp_overlap[:,:,0] = preds_mask
                        temp_overlap[:,:,1] = gt_mask
                        
                        #Convert to color blind
                        #Output
                        temp_overlap[preds_mask,:] = [202/255, 0/255, 32/255]
                        temp_overlap[gt_mask, :] = [5/255, 133/255, 176/255]
                        agreement = preds_mask * gt_mask
                        temp_overlap[agreement, :] = [155/255, 191/255, 133/255]
                        
                        temp_ax[key_idx+2].imshow(imgs[img].cpu().permute(1,2,0))
                        temp_ax[key_idx+2].imshow(temp_overlap,alpha=alpha)
                        temp_ax[key_idx+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                    else:
                        temp_pred = torch.argmax(preds[0], dim=0).detach().cpu().numpy()
                        temp_pred = decode_segmap(temp_pred,nc=num_classes)
                        temp_ax[0, key_idx+2].imshow(temp_pred,
                                                interpolation=None,alpha=alpha)
                        temp_ax[0, key_idx+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                        #Plot masks only
                        temp_ax[1, key_idx+2].imshow(temp_pred)
                        temp_ax[1, key_idx+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                        #Computed weighted IOU (account for class imbalance)
                        _, _, avg_jacc, avg_dice, avg_mAP = eval_metrics(true_masks[img].unsqueeze(0),
                                                                         preds,num_classes)
                        temp_ax[1, key_idx+2].set_title('IOU: {:.2f}, \n F1 Score: {:.2f}, \n mAP: {:.2f}'.format(avg_jacc, avg_dice, avg_mAP))
                    del model
                    torch.cuda.empty_cache()

                    #Compute estimated fat
                    if show_fat:
                        temp_fat[key_idx+1] = preds[0].count_nonzero().item() * (temp_org_size/temp_ds_size) * (temp_org_rate)**2
                
                folder = fig_dir + '{}_Segmentation_Maps/Run_{}/'.format(phase.capitalize(),split+1)
                
                #Create Training and Validation folder
                if not os.path.exists(folder):
                    os.makedirs(folder)
                 
                img_name = folder + idx[img] + '.png'
                
                temp_fig.savefig(img_name,dpi=temp_fig.dpi)
                plt.close(fig=temp_fig)
            
                img_count += 1
                print('Finished image {} of {} for {} dataset'.format(img_count,len(dataloaders[phase].sampler),phase))
        
    
    
    
    
    
    
    
