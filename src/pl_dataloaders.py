import os
import pickle
import random

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader

import utils.my_functions as fu
from data.data_loader import (CustomDataset, CustomDatasetTarget, collate_fn,
                              collate_fn_target)
from data.procesamiento import preprocessing
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, args, data_args):
        super().__init__()
        
        self.args = args
        self.data_args = data_args
        self.split = False
        self.idx_tr =[]
        self.idx_val = []
        self.idx_tst = []
        self.train, self.valid, self.test = self.get_data()
        self._save_information()

    def train_dataloader(self):
        return self.train
    
    def val_dataloader(self):
        return self.valid
    
    def test_dataloader(self):
        return self.test
        
    def _get_data(self, test=False):
        """
        test: if is False it will return the train dataloader and the validation dataloader
        """

        #get the data from confirguration
        configurations = {'carga': [0.0, 1.0], 'velocidad': [0.0, 1.0], 
            'lado': [0, 1], 'direct': [0, 1], 'corte': [0, 1]}
        grid_conf = list(ParameterGrid(configurations))

        conf = grid_conf[0]
        data, target, features, _ = fu.preprocess_pickle_configuration(self.data_args.path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
        target = target.values
        features = features.values
        if self.data_args.several_conf: #we add more configurations
            for conf in grid_conf[1:]: 
                x_ws1, y_ws1, feat_ws1, _ = fu.preprocess_pickle_configuration(self.data_args.path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
                data = np.concatenate((data, x_ws1), axis = 0)
                features = np.concatenate((features, feat_ws1.values), axis=0)
                target = np.concatenate((target, y_ws1.values), axis=0)

        n_signals, _ = data.shape 

        if not self.split and not test: 
            self._split_data(n_signals=n_signals)
            print(f'Train: {len(self.idx_tr)} \nValidation: {len(self.idx_val)} \nTest: {len(self.idx_tst)} \n')

        #do the preprocessing (min-max scaler)
        data_new = preprocessing(n_signals, data, feature_range = self.args.feature_range)

        #traspose the signal to have --> [sequence len, number of signals]. Map the labels and transform features to tensor
        data_t = torch.tensor(np.array(data_new).T, dtype=torch.float32)#.permute((1,2,0))
        target_t = torch.tensor(list(map(lambda x: {'eje sano':0, 'd1':1, 'd2':2, 'd3':3}[x], target)))
        features_t = torch.tensor(features, dtype=torch.float32)

        
        if not test:
            tr = data_t[:, self.idx_tr]
            val = data_t[:, self.idx_val]
            feat_tr = features_t[self.idx_tr, :]
            feat_val = features_t[self.idx_val, :]

            if self.data_args.get_class: #we add the target so we can use it 
                tr_class = target_t[self.idx_tr]
                val_class = target_t[self.idx_val]
                dataset = CustomDatasetTarget(tr, tr_class, feat_tr, self.idx_tr)
                dataset2 = CustomDatasetTarget(val, val_class, feat_val, self.idx_val)
                tr_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn_target)
                val_loader = DataLoader(dataset2, batch_size = self.args.batch_size, collate_fn=collate_fn_target)
            else:
                dataset = CustomDataset(tr)
                dataset2 = CustomDataset(val)
                tr_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn)
                val_loader = DataLoader(dataset2, batch_size = self.args.batch_size, collate_fn=collate_fn)

            return tr_loader, val_loader

        else:
            self.idx_tst = np.load(f'./results/{self.args.name_folder}/test_index.npy')
            tst = data_t[:, self.idx_tst]
            feat_tst = features_t[self.idx_tst,:]
            if self.data_args.get_class:
                tst_class = target_t[self.idx_tst]
                dataset = CustomDatasetTarget(tst, tst_class, feat_tst, self.idx_tst)
                ts_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn_target, shuffle=False)
            else:
                dataset = CustomDataset(tst)
                ts_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn, shuffle=False)

            return ts_loader

    def get_data(self):
        save_dir = "DATASETS/dataloaders.pkl"
        if not os.path.exists(save_dir):
            train_dataloader, val_dataloader = self._get_data(test=False)
            test_dataloader = self._get_data(test=True)
            with open(save_dir, "wb") as file:
                pickle.dump((train_dataloader, val_dataloader, test_dataloader), file)
        else:
            with open(save_dir, "rb") as file:
                print("Loading DataLoaders from existing pickle file.")
                train_dataloader, val_dataloader, test_dataloader = pickle.load(file)

                # Now set the appropiate batch size
                train_dataloader = DataLoader(
                    train_dataloader.dataset,
                    batch_size=self.args.batch_size,
                    collate_fn=train_dataloader.collate_fn,
                )
                val_dataloader = DataLoader(
                    val_dataloader.dataset,
                    batch_size=self.args.batch_size,
                    collate_fn=val_dataloader.collate_fn,
                )
                test_dataloader = DataLoader(
                    test_dataloader.dataset,
                    batch_size=self.args.batch_size,
                    collate_fn=test_dataloader.collate_fn,
                    shuffle=False,
                )
        return train_dataloader, val_dataloader, test_dataloader

    def _save_information(self):
        if not os.path.exists(f'./results/{self.args.name_folder}'):
            os.makedirs(f'./results/{self.args.name_folder}')

        f = open(f'./results/{self.args.name_folder}/arguments.txt', "w")
        for a in self.args:
            f.write(f'{a}: {self.args[a]} \n')
        f.close()

    ## OTHER FUNCTIONS FOR THE DATA
    def _split_data(self, n_signals, train_split = 0.7, test_split = 0.1):
        if not self.split :
            #Knowing the number of signals we hace, we split between train / validation / test
            n_train = int(n_signals* train_split) # 0 to n_train
            n_test = int(n_signals * test_split) #n_train to n_train+n_test
            n_val = n_signals - (n_train + n_test) #the last n_val

            range_list = list(range(n_signals))
            random.shuffle(range_list) #shuffle the list (this is just for when we use several configurations, so we take different configurations for train / test / validation)

        
            self.idx_tr = range_list[:n_train]
            self.idx_val = range_list[-n_val:]
            self.idx_tst = range_list[n_train:n_train+n_test]

            #save the indexs of the test
            np.save(f'./results/{self.args.name_folder}/test_index.npy', self.idx_tst)
            self.split = True

