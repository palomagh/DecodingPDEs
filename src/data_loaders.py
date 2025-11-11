import numpy as np
import os
import pickle
import gzip
import requests
import glob
import h5py
import math as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Data Loaders"""

from PIL import Image

def make_dataset(image_list_path, domain):
    image_list = open(image_list_path).readlines()
    images = [(val.split()[0], int(val.split()[1]), int(domain)) for val in image_list]
    return images


import scipy.io
from sklearn.model_selection import train_test_split

def load_text(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'text_xs.npy')

    try:
        train_data = np.load(os.path.join(path, 'text_xs_32.npy'), allow_pickle =True)
        train_labels = np.load(os.path.join(path, 'text_ys_32.npy'), allow_pickle =True)
    except:
        cwd = os.getcwd()
        os.chdir(path)
        os.system("wget -O text_xs.npy \"https://www.dropbox.com/s/yhlf25n8rzmdrtp/text_xs.npy?dl=1\"")
        os.system("wget -O text_ys.npy \"https://www.dropbox.com/s/16lj1vprg1pzckt/text_ys.npy?dl=1\"")
        train_data = np.load(os.path.join(path, 'text_xs.npy'), allow_pickle =True)
        train_labels = np.load(os.path.join(path, 'text_ys.npy'), allow_pickle =True)
        os.chdir(cwd)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader
 
def load_pythia_14m(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_pythia14m.npy')

    train_data = np.load(os.path.join(path, 'x_pythia14m.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'y_pythia14m.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_pythia_70m(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_pythia70m.npy')
    
    train_data = np.load(os.path.join(path, 'x_pythia70m.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'y_pythia70m.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_pythia_160m(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_pythia160m.npy')
    
    train_data = np.load(os.path.join(path, 'x_pythia160m.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'y_pythia160m.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader  

def load_pythia_410m(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_pythia410m.npy')
    
    train_data = np.load(os.path.join(path, 'x_pythia410m.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'y_pythia410m.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_pythia_1b(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_pythia1b.npy')
    
    train_data = np.load(os.path.join(path, 'x_pythia1b.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'y_pythia1b.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_pythia_14b(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_pythia14b.npy')
    
    train_data = np.load(os.path.join(path, 'x_pythia1,4b.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'y_pythia1,4b.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader 

def load_gpt(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_gpt.npy')
    
    train_data = np.load(os.path.join(path, 'x_gpt.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'gpt-y.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()#.mean(1)
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_gptm(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_gptm.npy')
    
    train_data = np.load(os.path.join(path, 'x_gpt_medium.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'gpt-m-y.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()#.mean(1)
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_gptl(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'x_gptl.npy')
    
    train_data = np.load(os.path.join(path, 'x_gpt_large.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'gpt-l-y.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()#.mean(1)
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

def load_gptxl(root, batch_size, valid_split=-1, maxsize=None):
   path = root

   file_path = os.path.join(path, 'x_gptxl.npy') 
   
   train_data = np.load(os.path.join(path, 'x_gpt_xl.npy'), allow_pickle =True)
   train_labels = np.load(os.path.join(path, 'gpt-xl-y.npy'), allow_pickle =True)

   maxsize = len(train_data) if maxsize is None else maxsize
   train_data = torch.from_numpy(train_data[:maxsize]).float()#.mean(1)
   train_labels = torch.from_numpy(train_labels[:maxsize]).long()

   train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
   return train_loader, None, train_loader

def load_pde(root, batch_size, dataset='1DCFD', flip=False, double=False, valid_split=-1, num_workers=4):
    large = False

    if dataset == 'Burgers':
        filename = '1D_Burgers_Sols_Nu1.0.hdf5' 

        reduced_resolution = 1 
        reduced_resolution_t = 5
        reduced_batch = 1
        initial_step = 10
        t_train = 200
        single_file = True

    elif dataset == '1DCFD':
        root = '/run/determined/workdir/shared_fs/data/PDEBench'
        filename = '1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5'
        reduced_resolution = 1
        reduced_resolution_t = 5
        reduced_batch = 1
        initial_step = 10
        t_train = 100
        single_file = True

    elif dataset == 'ADV':
        root = '/run/determined/workdir/shared_fs/data/PDEBench'
        filename = '1D_Advection_Sols_beta0.4.hdf5'
        reduced_resolution = 4
        reduced_resolution_t = 5
        reduced_batch = 1
        initial_step = 10
        t_train = 200
        single_file = True 

    elif dataset == 'DS':
        root = '/run/determined/workdir/shared_fs/data/PDEBench'
        filename = '1D_diff-sorp_NA_NA.h5'
        reduced_resolution = 1
        reduced_resolution_t = 1
        reduced_batch = 1
        initial_step = 10
        t_train = 101
        single_file = False 

    elif dataset == 'RD':
        root = '/run/determined/workdir/shared_fs/data/PDEBench'
        filename = 'ReacDiff_Nu0.5_Rho1.0.hdf5'
        reduced_resolution = 1
        reduced_resolution_t = 1
        reduced_batch = 1
        initial_step = 5
        t_train = 30
        single_file = True 


    if single_file:
        if large:
            train_data = UNetDatasetSingleLarge(filename,
                                    saved_folder=root,
                                    reduced_resolution=reduced_resolution,
                                    reduced_resolution_t=reduced_resolution_t,
                                    reduced_batch=reduced_batch,
                                    initial_step=initial_step, t_train=t_train)

            val_data = UNetDatasetSingleLarge(filename,
                                  saved_folder=root,
                                  reduced_resolution=reduced_resolution,
                                  reduced_resolution_t=reduced_resolution_t,
                                  reduced_batch=reduced_batch,
                                  initial_step=initial_step,
                                  if_test=True, x_normalizer=train_data.x_normalizer)

        else:
            train_data = UNetDatasetSingle(filename,
                                        saved_folder=root,
                                        reduced_resolution=reduced_resolution,
                                        reduced_resolution_t=reduced_resolution_t,
                                        reduced_batch=reduced_batch,
                                        initial_step=initial_step, t_train=t_train)

            val_data = UNetDatasetSingle(filename,
                                      saved_folder=root,
                                      reduced_resolution=reduced_resolution,
                                      reduced_resolution_t=reduced_resolution_t,
                                      reduced_batch=reduced_batch,
                                      initial_step=initial_step,
                                      if_test=True)
    else:
        train_data = UNetDatasetMult(filename,
                                    saved_folder=root,
                                    reduced_resolution=reduced_resolution,
                                    reduced_resolution_t=reduced_resolution_t,
                                    reduced_batch=reduced_batch,
                                    initial_step=initial_step, t_train=t_train)

        val_data = UNetDatasetMult(filename,
                                  saved_folder=root,
                                  reduced_resolution=reduced_resolution,
                                  reduced_resolution_t=reduced_resolution_t,
                                  reduced_batch=reduced_batch,
                                  initial_step=initial_step,
                                  if_test=True)

    
    
    if dataset == '1DCFD':
        x_normalizer = UnitGaussianNormalizer(train_data.x)
        train_data.x = x_normalizer.encode(train_data.x)
        val_data.x = x_normalizer.encode(val_data.x)

        y_normalizer = x_normalizer
        train_data.y = y_normalizer.encode(train_data.y)
        val_data.y = y_normalizer.encode(val_data.y)
        

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=False)
    return train_loader, None, test_loader



from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder




# +
# class DataGenerator(object):
#     """
#     A class of functions for generating jsonl datasets for classification tasks.
#     """
#     def __init__(self, did, seed = 123):
#         self.seed = seed
#         self.did = did
#         self.fname = f'{did}'
#         self.scaler = StandardScaler()

#     def preprocess_data(self, data,  normalized=False, corruption_level=0, outliers=None):
#         X, y = data['data'], data['target']
#         if normalized:
#             X = self.scaler.fit_transform(X)
        
#         X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(X, y)
#         if outliers is not None:
#             X_out, y_out = outliers
#             X_train = np.concatenate([X_train, X_out], axis = 0)
#             y_train = np.concatenate([y_train, y_out], axis = 0)
#         if corruption_level > 0:
#             # corrupt here
#             n = len(y_train)
#             m = int(n * corruption_level)
#             inds = random.sample(range(1, n), m)
#             for i in inds:
#                 y_train[i] = 1 - y_train[i] #binary
        
#         train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
#         train_df['y'], val_df['y'], test_df['y'] = y_train, y_valid, y_test   

#         return train_df, val_df, test_df
 

def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping

def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')

            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([min(1.0, float(tokens[2]))])

    return feature, np.array(label)

def prepare_train_data(root):

    train_file = root + "/drugcell_all.txt"
    cell2id_mapping_file = root + "/cell2ind.txt"
    drug2id_mapping_file = root + "/drug2ind.txt"

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)

    cell_features = np.genfromtxt(root + "/cell2mutation.txt", delimiter=',')
    drug_features = np.genfromtxt(root + '/drug2fingerprint.txt', delimiter=',')

    train_feature = build_input_vector(train_feature, cell_features, drug_features)

    shuffle_pid = np.random.permutation(train_feature.shape[0])
    train_feature = train_feature[shuffle_pid]
    train_label = train_label[shuffle_pid]
    test_size = int(0.2 * train_feature.shape[0])
    train_feature, test_feature = train_feature[:-test_size], train_feature[-test_size:]
    train_label, test_label = train_label[:-test_size], train_label[-test_size:]
    train_feature = torch.from_numpy(train_feature).float().unsqueeze(1)
    test_feature = torch.from_numpy(test_feature).float().unsqueeze(1)

    return train_feature, torch.FloatTensor(train_label), test_feature, torch.FloatTensor(test_label)

def build_input_vector(input_data, cell_features, drug_features):
    genedim = len(cell_features[0,:])
    drugdim = len(drug_features[0,:])
    feature = np.zeros((len(input_data), (genedim+drugdim)))

    for i in range(len(input_data)):
        feature[i] = np.concatenate([drug_features[int(input_data[i][1])], cell_features[int(input_data[i][0])]], axis=None)

    feature = np.array(feature)
    return feature



class UNetDatasetSingle(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1, t_train=100):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """
        
        # Define path to files
        root_path = os.path.abspath(saved_folder + "/" + filename)
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd)==3:  # 1D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          3],
                                         dtype=np.float32)
                    #density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,2] = _data   # batch, x, t, ch

                if len(idx_cfd)==4:  # 2D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,3] = _data   # batch, x, t, ch
                    
                if len(idx_cfd)==5:  # 3D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          idx_cfd[4]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          5],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,3] = _data   # batch, x, t, ch
                    # Vz
                    _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,4] = _data   # batch, x, t, ch
                

            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    #if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        if t_train > self.data.shape[-2]:
            t_train = self.data.shape[-2]
        self.initial_step = initial_step
        self.t_train = t_train
        
        self.data = torch.tensor(self.data)

        if len(self.data[..., 0,:].shape) == 3:
            self.x = self.data[..., 0,:].transpose(-1, -2)
            self.y = self.data[..., t_train-1:t_train, :].squeeze(-2).transpose(-1, -2)
        else:
            self.x = self.data[..., 0,:].permute(0, 3, 1, 2)
            self.y = self.data[..., t_train-1:t_train, :].squeeze(-2).permute(0, 3, 1, 2)

        if self.x.shape[1] != 1:
            self.x = self.x.reshape(self.x.shape[0], 1, -1)
            self.y = self.y.reshape(self.y.shape[0], 1, -1)
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class UNetDatasetSingleLarge(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1, t_train=100, x_normalizer=None
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(saved_folder +  '/' + filename)
        
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as f:
            data_list = np.arange(len(f['density']))
            if x_normalizer is None:
                samples = np.concatenate((np.expand_dims(np.transpose(np.array(f['density'][:100], dtype=np.float32)[:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution],(0,2,3,1)), 4),
                    np.expand_dims(np.transpose(np.array(f['pressure'][:100], dtype=np.float32)[:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution],(0,2,3,1)), 4),
                    np.expand_dims(np.transpose(np.array(f['Vx'][:100], dtype=np.float32)[:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution],(0,2,3,1)), 4),
                    np.expand_dims(np.transpose(np.array(f['Vy'][:100], dtype=np.float32)[:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution],(0,2,3,1)), 4)), axis=-1)
                samples = torch.tensor(samples)
                self.x_normalizer = UnitGaussianNormalizer(samples)
            else:
                self.x_normalizer = x_normalizer

        test_idx = int(len(data_list) * (1-test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
        
        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.t_train = t_train
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        # Open file and read data
        with h5py.File(self.file_path, 'r') as f:
            _data = np.array(f['density'][self.data_list[idx]], dtype=np.float32)  # batch, time, x,...
            idx_cfd = _data.shape

            self.data = np.zeros([
                                          idx_cfd[1]//self.reduced_resolution,
                                          idx_cfd[2]//self.reduced_resolution,
                                          mt.ceil(idx_cfd[0]/self.reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
            # density
            _data = _data[::self.reduced_resolution_t,::self.reduced_resolution,::self.reduced_resolution]
            ## convert to [x1, ..., xd, t, v]
            _data = np.transpose(_data, (1, 2, 0))
            self.data[...,0] = _data   # batch, x, t, ch
            # pressure
            _data = np.array(f['pressure'][self.data_list[idx]], dtype=np.float32)  # batch, time, x,...
            _data = _data[::self.reduced_resolution_t,::self.reduced_resolution,::self.reduced_resolution]
            ## convert to [x1, ..., xd, t, v]
            _data = np.transpose(_data, (1, 2, 0))
            self.data[...,1] = _data   # batch, x, t, ch
            # Vx
            _data = np.array(f['Vx'][self.data_list[idx]], dtype=np.float32)  # batch, time, x,...
            _data = _data[::self.reduced_resolution_t,::self.reduced_resolution,::self.reduced_resolution]
            ## convert to [x1, ..., xd, t, v]
            _data = np.transpose(_data, (1, 2, 0))
            self.data[...,2] = _data   # batch, x, t, ch
            # Vy
            _data = np.array(f['Vy'][self.data_list[idx]], dtype=np.float32)  # batch, time, x,...
            _data = _data[::self.reduced_resolution_t,::self.reduced_resolution,::self.reduced_resolution]
            ## convert to [x1, ..., xd, t, v]
            _data = np.transpose(_data, (1, 2, 0))
            self.data[...,3] = _data   # batch, x, t, ch


            if self.t_train > self.data.shape[-2]:
                self.t_train = self.data.shape[-2]

            self.data = torch.tensor(self.data)
            self.data  = self.x_normalizer.encode(self.data)

            x, y = self.data[...,0,:].permute(2, 0, 1), self.data[..., self.t_train-1:self.t_train, :].squeeze(-2).permute(2, 0, 1) 
            
        return x, y

class UNetDatasetMult(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1, t_train=100
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(saved_folder +  '/' + filename)
        
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1-test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
        
        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.t_train = t_train
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
        
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)
            
            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1,len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            if self.t_train > data.shape[-2]:
                self.t_train = data.shape[-2]


        if len(data[...,0,:].shape) == 3:
            data = data[::self.reduced_resolution,::self.reduced_resolution, ::self.reduced_resolution_t, :]
            if data.shape[-1] == 2:
                x, y = data[...,0,:].permute(2, 0, 1)[:1,...], data[..., self.t_train-1:self.t_train, :].squeeze(-2).permute(2, 0, 1)[1:2,...]
            else:
                x, y = data[...,0,:].permute(2, 0, 1), data[..., self.t_train-1:self.t_train, :].squeeze(-2).permute(2, 0, 1)
            return x, y
            
        else:
            data = data[::self.reduced_resolution, ::self.reduced_resolution_t, :]
            return data[...,0,:].permute(1, 0), data[..., self.t_train-1:self.t_train, :].squeeze(-2).permute(1, 0)

        return data[...,:self.initial_step,:], data

"""Hepler Funcs"""

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def split_dataset(train_dataset, valid_split):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train)) if valid_split <= 1 else num_train - valid_split

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler

# def bitreversal_permutation(n):
#     """Return the bit reversal permutation used in FFT.
#     Parameter:
#         n: integer, must be a power of 2.
#     Return:
#         perm: bit reversal permutation, numpy array of size n
#     """
#     m = int(math.log2(n))
#     assert n == 1 << m, 'n must be a power of 2'
#     perm = np.arange(n).reshape(n, 1)
#     for i in range(m):
#         n1 = perm.shape[0] // 2
#         perm = np.hstack((perm[:n1], perm[n1:]))
#     return torch.tensor(perm.squeeze(0))

# class Permute2D(nn.Module):

#     def __init__(self, row, col):

#         super().__init__()
#         self.rowperm = torch.LongTensor(bitreversal_permutation(row))
#         self.colperm = torch.LongTensor(bitreversal_permutation(col))

#     def forward(self, tensor):

#         return tensor[:,self.rowperm][:,:,self.colperm]

class Permute1D(nn.Module):

    def __init__(self, length):

        super().__init__()
        self.permute = torch.Tensor(np.random.permutation(length).astype(np.float64)).long()

    def forward(self, tensor):

        return tensor[:,self.permute]


class dataset_wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, normvalx, normvaly, clip=None, get_tensors=True):
        self.dataset = dataset
        self.normvalx = normvalx
        self.normvaly = normvaly
        self.clip = clip
        self.transform = False
        if get_tensors:
            self.tensors = self.get_tensors()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, i):

        item = self.dataset.__getitem__(i)
        if len(item) == 3:
            self.transform = True
            newx = item[0] / self.normvalx
            newy = item[1] / self.normvaly

            if self.clip is not None:
                newx[newx > self.clip] = self.clip
                newx[newx < -self.clip] = -self.clip
                newy[newy > self.clip] = self.clip
                newy[newy < -self.clip] = -self.clip
            return newx, newy, item[2]

        elif len(item) == 2:
            newx = item[0] / self.normvalx
            newy = item[1] / self.normvaly
            
            if self.clip is not None:
                newx[newx > self.clip] = self.clip
                newx[newx < -self.clip] = -self.clip
                newy[newy > self.clip] = self.clip
                newy[newy < -self.clip] = -self.clip
            return newx, newy
        else:
            return item

    def get_tensors(self):
        xs, ys, zs = [], [], []
        for i in range(self.dataset.__len__()):
            data = self.__getitem__(i)
            xs.append(np.expand_dims(data[0],0))
            ys.append(np.expand_dims(data[1],0))
            if len(data) == 3:
                zs.append(np.expand_dims(data[2],0))
        xs = torch.from_numpy(np.array(xs)).squeeze(1)
        ys = torch.from_numpy(np.array(ys)).squeeze(1)
        if len(zs) > 0:
            zs = torch.from_numpy(np.array(zs)).squeeze(1)
            self.transform = True

        return xs, ys, zs


class PairedDatasetImagePath(torch.utils.data.Dataset):
    def __init__(self, root, paths, skyaug_min=0, skyaug_max=0, part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param paths: (list) list of file paths to (3, W, H) images: image, cr, ignore.
        :param skyaug_min: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param skyaug_min: float. subtract maximum amount of abs(skyaug_min) * sky_level as data augmentation
        :param skyaug_max: float. add maximum amount of skyaug_max * sky_level as data augmentation
        :param part: either 'train' or 'val'.
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """
        assert 0 < f_val < 1
        np.random.seed(seed)
        n_total = len(paths)
        n_train = int(n_total * (1 - f_val)) #int(len * (1 - f_val)) JK
        f_test = f_val
        n_search = int(n_total * (1 - f_val - f_test))

        if part == 'train':
            s = np.s_[:max(1, n_train)]
        elif part == 'test':
            s = np.s_[min(n_total - 1, n_train):]
        else:
            s = np.s_[0:]

        self.paths = paths[s]
        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max
        self.root = root

    def __len__(self):
        return len(self.paths)

    def get_skyaug(self, i):
        """
        Return the amount of background flux to be added to image
        The original sky background should be saved in sky.npy in each sub-directory
        Otherwise always return 0
        :param i: index of file
        :return: amount of flux to add to image
        """
        path = os.path.split(self.paths[i])[0]
        sky_path = os.path.join(self.root, path[2:], 'sky.npy') #JK
        if os.path.isfile(sky_path):
            f_img = self.paths[i].split('/')[-1]
            sky_idx = int(f_img.split('_')[0])
            sky = np.load(sky_path)[sky_idx-1]
            return sky * np.random.uniform(self.skyaug_min, self.skyaug_max)
        else:
            return 0

    def __getitem__(self, i):
        data = np.load(os.path.join(self.root, self.paths[i][2:]))
        image = data[0]
        mask = data[1]
        if data.shape[0] == 3:
            ignore = data[2]
        else:
            ignore = np.zeros_like(data[0])
        # try:#JK
        skyaug = self.get_skyaug(i)
        #crop to 128*128
        image, mask, ignore = get_fixed_crop([image, mask, ignore], 128, 128)

        return np.expand_dims(image + skyaug, 0).astype(np.float32), mask.astype(np.float32), ignore.astype(np.float32)

def get_random_crop(images, crop_height, crop_width):

    max_x = images[0].shape[1] - crop_width
    max_y = images[0].shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crops = []
    for image in images:
        crop = image[y: y + crop_height, x: x + crop_width]
        crops.append(crop)

    return crops

def get_fixed_crop(images, crop_height, crop_width):

    x = 64
    y = 64

    crops = []
    for image in images:
        crop = image[y: y + crop_height, x: x + crop_width]
        crops.append(crop)

    return crops


import operator
from functools import reduce
from functools import partial

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class DistGenerator():
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, batch_size, expected_n_channels, label_engineering = None):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.distmap_path = distmap_path
        self.dim = dim
        self.pad_size = pad_size
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.label_engineering = label_engineering

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.pdb_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_list = self.pdb_id_list[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = get_input_output_dist(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim, self.expected_n_channels)
        if self.label_engineering is None:
            return X, Y
        if self.label_engineering == '100/d':
            return X, 100.0 / Y
        try:
            t = float(self.label_engineering)
            Y[Y > t] = t
        except ValueError:
            print('ERROR!! Unknown label_engineering parameter!!')
            return 
        return X, Y

class PDNetDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, 
        pad_size, batch_size, expected_n_channels, label_engineering = None):

        self.distGenerator = DistGenerator(
            pdb_id_list, features_path, distmap_path, dim, pad_size, 
            1, expected_n_channels, label_engineering) # Don't use batch_size

    def __len__(self):
        return self.distGenerator.__len__()

    def __getitem__(self, index):
        X, Y = self.distGenerator.__getitem__(index)
        X = X[0, :, :, :].transpose(2, 0, 1)
        Y = Y[0, :, :, :].transpose(2, 0, 1)
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        return X, Y

import pickle
import random

def load_list(file_lst, max_items = 1000000):
    if max_items < 0:
        max_items = 1000000
    protein_list = []
    f = open(file_lst, 'r')
    for l in f.readlines():
        protein_list.append(l.strip().split()[0])
    if (max_items < len(protein_list)):
        protein_list = protein_list[:max_items]
    return protein_list

def summarize_channels(x, y):
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i+1, a, m, s))

def get_bulk_output_contact_maps(pdb_id_list, all_dist_paths, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, all_dist_paths)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    if np.any(np.isnan(Y)):
        print('')
        print('WARNING:')
        print('Some pdbs in the following list have NaNs in their distances:', pdb_id_list)
        np.seterr(invalid='ignore')
    YY[ YY < 8.0 ] = 1.0
    YY[ YY >= 8.0 ] = 0.0
    return YY.astype(np.float32)

def get_bulk_output_dist_maps(pdb_id_list, all_dist_paths, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), np.inf)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, all_dist_paths)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    return YY.astype(np.float32)

def get_input_output_dist(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y0 = get_map(pdb, all_dist_paths, len(X[:, 0, 0]))
        assert len(X[:, 0, 0]) >= len(Y0[:, 0])
        if len(X[:, 0, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        l = len(X[:, 0, 0])
        Y = np.full((l, l), np.nan)
        Y[:len(Y0[:, 0]), :len(Y0[:, 0])] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])), dtype=np.float32)
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size), 100.0, dtype=np.float32)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2)] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, 0] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, 0] = Ypadded[rx:rx+OUTL, ry:ry+OUTL]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_input_output_bins(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels, bins):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, len(bins)), 0.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y0 = dist_map_to_bins(get_map(pdb, all_dist_paths, len(X[:, 0, 0])), bins)
        assert len(X[:, 0, 0]) >= len(Y0[:, 0])
        if len(X[:, 0, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        l = len(X[:, 0, 0])
        Y = np.full((l, l, len(Y0[0, 0, :])), np.nan)
        Y[:len(Y0[:, 0]), :len(Y0[:, 0]), :] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])))
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size, len(bins)), 0.0)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, :] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, :] = Ypadded[rx:rx+OUTL, ry:ry+OUTL, :]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_sequence(pdb, feature_file):
    features = pickle.load(open(feature_file, 'rb'))
    return features['seq']

def get_feature(pdb, all_feat_paths, expected_n_channels):
    features = None
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features == None:
        print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    for j in range(3):
        a = np.repeat(ss[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (l, 22)
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add SA
    sa = features['sa']
    assert sa.shape == (l, )
    a = np.repeat(sa.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

def get_map(pdb, all_dist_paths, expected_l = -1):
    seqy = None
    mypath = ''
    for path in all_dist_paths:
        if os.path.exists(path + pdb + '-cb.npy'):
            mypath = path + pdb + '-cb.npy'
            (ly, seqy, cb_map) = np.load(path + pdb + '-cb.npy', allow_pickle = True)
    if seqy == None:
        print('Expected distance map file for', pdb, 'not found at', all_dist_paths)
        exit(1)
    if 'cameo' not in mypath and expected_l > 0:
        assert expected_l == ly
        assert cb_map.shape == ((expected_l, expected_l))
    Y = cb_map
    # Only CAMEO dataset has this issue
    if 'cameo' not in mypath:
        assert not np.any(np.isnan(Y))
    if np.any(np.isnan(Y)):
        np.seterr(invalid='ignore')
        print('')
        print('WARNING!! Some values in the pdb structure of', pdb, 'l = ', ly, 'are missing or nan! Indices are: ', np.where(np.isnan(np.diagonal(Y))))
    Y[Y < 1.0] = 1.0
    Y[0, 0] = Y[0, 1]
    Y[ly-1, ly-1] = Y[ly-1, ly-2]
    for q in range(1, ly-1):
        if np.isnan(Y[q, q]):
            continue
        if np.isnan(Y[q, q-1]) and np.isnan(Y[q, q+1]):
            Y[q, q] = 1.0
        elif np.isnan(Y[q, q-1]):
            Y[q, q] = Y[q, q+1]
        elif np.isnan(Y[q, q+1]):
            Y[q, q] = Y[q, q-1]
        else:
            Y[q, q] = (Y[q, q-1] + Y[q, q+1]) / 2.0
    assert np.nanmax(Y) <= 500.0
    assert np.nanmin(Y) >= 1.0
    return Y

def save_dist_rr(pdb, pred_matrix, feature_file, file_rr):
    sequence = get_sequence(pdb, feature_file)
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i %0.3f %.3f 1\n" %(j+1, k+1, P[j][k], P[j][k]) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def save_contacts_rr(pdb, all_feat_paths, pred_matrix, file_rr):
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features == None:
        print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    sequence = features['seq']
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(j+1, k+1, (P[j][k])) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def dist_map_to_bins(Y, bins):
    L = len(Y[:, 0])
    B = np.full((L, L, len(bins)), 0)
    for i in range(L):
        for j in range(L):
            for bin_i, bin_range in bins.items():
                min_max = [float(x) for x in bin_range.split()]
                if Y[i, j] > min_max[0] and Y[i, j] <= min_max[1]:
                    B[i, j, bin_i] = 1
    return B


def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_text(root, batch_size, valid_split=-1, maxsize=None):
    path = root

    file_path = os.path.join(path, 'text_xs_32.npy')
    load_success = False

    while not load_success:
        try:
            train_data = np.load(os.path.join(path, 'text_xs_32.npy'), allow_pickle =True)
            train_labels = np.load(os.path.join(path, 'text_ys_32.npy'), allow_pickle =True)
            load_success = True
        except:
            cwd = os.getcwd()
            os.chdir(path)
            try:
                os.system("wget -O text_xs_32.npy \"https://www.dropbox.com/s/yhlf25n8rzmdrtp/text_xs_32.npy?dl=1\"")
                os.system("wget -O text_ys_32.npy \"https://www.dropbox.com/s/16lj1vprg1pzckt/text_ys_32.npy?dl=1\"")
                # os.system("gdown 1YgAXhRzcmhkjNqvE5c4MUFIx7T08KmJf")#1P4xjZjyx2WKVOnZ7hocUq-ldA6vfKn2x")
                # os.system("gdown 1lYHpj9hRd0yXTNcgs4mJDnmkdM7vJj2H")#1A4YH1TdYt9xYtWBEpliBTo7ut-jwOAPl")
            except:
                pass
            # os.system("wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1P4xjZjyx2WKVOnZ7hocUq-ldA6vfKn2x' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=1P4xjZjyx2WKVOnZ7hocUq-ldA6vfKn2x\" -O text_xs_16.npy && rm -rf /tmp/cookies.txt")
            # os.system("wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A4YH1TdYt9xYtWBEpliBTo7ut-jwOAPl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=1A4YH1TdYt9xYtWBEpliBTo7ut-jwOAPl\" -O text_ys_16.npy && rm -rf /tmp/cookies.txt")
            os.chdir(cwd)
    
    # train_data = np.load(os.path.join(path, 'x_16.npy'), allow_pickle =True)
    # train_labels = np.load(os.path.join(path, 'text_ys_16.npy'), allow_pickle =True)

    maxsize = len(train_data) if maxsize is None else maxsize
    train_data = torch.from_numpy(train_data[:maxsize]).float()#.mean(1)
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None, train_loader

