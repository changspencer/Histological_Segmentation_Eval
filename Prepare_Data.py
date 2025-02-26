# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Load datasets for models
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import csv
import pdb
from os.path import join, abspath, dirname
import pdb

from Utils.dataset import RootsDataset

#From WSL repo
def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out

def get_files(folds_dir, split, fold,data_type='time'):
    splits = ['train', 'valid', 'test']
    if data_type == 'Time_Folds':
        csv_dir = join(folds_dir, 'Time_Folds_split_1', 'fold_{}'.format(fold))
        csv_files = [join(csv_dir,  '{}_s_time_f_{}.csv'.format(s, fold)) for s in splits]
    elif data_type == 'Val_Week_8':
        csv_dir = join(folds_dir, 'Time_Folds_split_1', 'fold_0')
        csv_files = [join(csv_dir,  '{}_s_time_f_0.csv'.format(s)) for s in splits]
    elif data_type == 'GlaS':
        csv_dir = join(folds_dir, 'split_{}'.format(split), 'fold_{}'.format(fold))
        csv_files = [join(csv_dir,  '{}_s_{}_f_{}.csv'.format(s, split, fold)) for s in splits]
    else:
        csv_dir = join(folds_dir, '{}_split_{}'.format(data_type,split), 'fold_{}'.format(fold))
        csv_files = [join(csv_dir,  '{}_s_{}_f_{}.csv'.format(s, split, fold)) for s in splits]
    split_files = [csv_reader(csv) for csv in csv_files]
    return split_files


def decode_classes(files: list,class_label=True) -> list:
    if class_label:
        classes = {'benign': 0, 'malignant': 1}
        files_decoded_classes = []
        for f in files:
            class_name = f[2]
            files_decoded_classes.append((f[0], f[1], classes[class_name]))
    else:
        files_decoded_classes = []
        for f in files:
            try:
                files_decoded_classes.append((f[0]+'.jpeg', f[1], f[2]))
            except:
                files_decoded_classes.append((f[0]+'.jpeg',f[0]+'.jpeg',f[0]+'.jpeg')) 
    return files_decoded_classes

def Prepare_DataLoaders(Network_parameters, splits, data_type='time'):
    Dataset = Network_parameters['Dataset']
    imgs_dir = Network_parameters['imgs_dir']

    # Load datasets
    #Histologial images
    if (Dataset=='SFBHI'):
        #Get files for each fold
        train_indices = []
        val_indices = []
        test_indices = []
        for fold in range(0,splits):
            files = get_files(imgs_dir+'folds',1,fold,data_type=data_type)
            temp_train, temp_val, temp_test = [decode_classes(f,class_label=False) for f in files]
            train_indices.append(temp_train)
            val_indices.append(temp_val)
            test_indices.append(temp_test)

           
   #Glas Dataset
    elif Dataset == 'GlaS':
        #Get files for each fold
        train_indices = []
        val_indices = []
        test_indices = []
        for fold in range(0,splits):
            files = get_files(imgs_dir+'folds',0,fold,data_type='GlaS')
            temp_train, temp_val, temp_test = [decode_classes(f) for f in files]
            train_indices.append(temp_train)
            val_indices.append(temp_val)
            test_indices.append(temp_test)


   #Glas Dataset
    elif Dataset in ['PRMI', 'Peanut_PRMI', 'PS_PRMI']:
        #Get files for each fold - For now, I'm not using any folds here.
        train_indices = []
        val_indices = []
        test_indices = []
        # I'd like to avoid "double loading" this dataset in the future
        imgs_dir = abspath(dirname(__file__)) + "/" + imgs_dir
        train_set = RootsDataset(root=imgs_dir + "/train")
        val_set = RootsDataset(root=imgs_dir + "/val")
        test_set = RootsDataset(root=imgs_dir + "/test")
        train_indices.append([i for i in range(len(train_set))])
        val_indices.append([i for i in range(len(val_set))])
        test_indices.append([i for i in range(len(test_set))])
        print("*** PRMI Dataset acknowledged...")

           
   #SiTS_crop Dataset
    elif Dataset == 'SiTS_crop':
        #Get files for each fold - For now, I'm not using any folds here.
        train_indices = []
        val_indices = []
        test_indices = []
        # I'd like to avoid "double loading" this dataset in the future
        imgs_dir = abspath(dirname(__file__)) + "/" + imgs_dir
        train_set = RootsDataset(root=imgs_dir + "/train")
        val_set = RootsDataset(root=imgs_dir + "/val")
        test_set = RootsDataset(root=imgs_dir + "/test")
        train_indices.append([i for i in range(len(train_set))])
        val_indices.append([i for i in range(len(val_set))])
        test_indices.append([i for i in range(len(test_set))])
        print("*** SitS_crop Dataset acknowledged...")
    
    #Generate indices (img files) for training, validation, and test
    indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}
    
    return indices
