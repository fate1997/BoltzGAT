import json
import logging
import logging.config
import os
import random
import sys
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd
import torch
import yaml
from easydict import EasyDict
from sklearn import metrics
from torch import nn
from torch_geometric.loader import DataLoader

from data.data import MolPropDataset
from data.mixture_data import MixtureDataset


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, patience=100, verbose=False,
                 path='../results/trained_models/', model_type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.model_type = model_type

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model) 
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation score doesn't improve in patience
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, self.model_type + '.pt'))
        self.val_loss_min = val_loss


def evaluate_score(model, 
                   data_loader, 
                   loss_fn,
                   logarithm=True, 
                   normalizer=None, 
                   train_mixture=False,):
    if isinstance(model, nn.Module):
        model.eval()

    if train_mixture:
        y_true, y_pred, _, _, _ = mixture_batch_flatten(model, data_loader, logarithm)
    else:
        y_true, y_pred, _, _ = batch_flatten(model, data_loader, normalizer, logarithm)

    metric_dict = {}
    metric_dict['AARD'] = np.abs((y_true - y_pred) / y_true).mean()
    metric_dict['R2'] = metrics.r2_score(y_true, y_pred)
    metric_dict['loss'] = loss_fn(torch.from_numpy(y_true), 
                                  torch.from_numpy(y_pred)).item()
    metric_dict['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    metric_dict['RMSE'] = metrics.root_mean_squared_error(y_true, y_pred)
    return metric_dict


def get_logger(name, log_dir, config_path='./configs/log_config.json'):
    config_dict = json.load(open(config_path))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, name.replace('/', '-'))
    config_dict['handlers']['file_handler']['filename'] = log_path
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    std_out_format = '%(asctime)s - %(levelname)s - %(message)s'
    fomatter = logging.Formatter(std_out_format, datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(fomatter)
    logger.addHandler(consoleHandler)
    return logger



def data2iter(name, batch_size, train_ratio, additional_features, replace, logarithm, seed=42):
    def flatten(data_list):
        flatten_dataset = []
        for j, data in enumerate(data_list):
            for i in range(data.temps.size(1)):
                flatten_data = deepcopy(data)
                flatten_data.temps = data.temps[0][i].unsqueeze(0).unsqueeze(0)
                flatten_data.y = data.y[0][i].unsqueeze(0).unsqueeze(0)
                flatten_dataset.append(flatten_data)
        return flatten_dataset

    dataset = MolPropDataset(name, additional_features, replace=replace)
    if logarithm:
        dataset.logarithm()

    random.seed(seed)
    num_points = len(dataset)
    ids = list(range(num_points))
    random.shuffle(ids)

    split_1 = int(train_ratio * num_points)
    split_2 = int((train_ratio+(1-train_ratio)/2) * num_points)
    train_ids = ids[: split_1]
    val_ids = ids[split_1:split_2]
    test_ids = ids[split_2:]

    train_data = flatten([dataset.data_list[i] for i in train_ids])
    val_data = flatten([dataset.data_list[i] for i in val_ids])
    test_data = flatten([dataset.data_list[i] for i in test_ids])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader,test_loader


def mixture2iter(name, batch_size, train_ratio, additional_features, replace, logarithm, label_col, split_by='entry'):
    
    dataset = MixtureDataset(name, additional_features, replace=replace, label_col=label_col)
    if logarithm:
        dataset.logarithm()

    random.seed(42)
    if split_by == 'entry':
        num_points = len(dataset)
        ids = list(range(num_points))
        random.shuffle(ids)

        split_1 = int(train_ratio * num_points)
        split_2 = int((train_ratio+(1-train_ratio)/2) * num_points)
        train_ids = ids[: split_1]
        val_ids = ids[split_1:split_2]
        test_ids = ids[split_2:]
        print(f'Splitting by entry, train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}')
    elif split_by == 'mol':
        smiles_data = dataset.groupby_smiles()
        all_smiles = list(smiles_data.keys())
        random.shuffle(all_smiles)
        split_1 = int(train_ratio * len(all_smiles))
        split_2 = int((train_ratio+(1-train_ratio)/2) * len(all_smiles))
        train_smiles = all_smiles[:split_1]
        val_smiles = all_smiles[split_1:split_2]
        test_smiles = all_smiles[split_2:]
        train_ids = list(chain(*[smiles_data[smiles] for smiles in train_smiles]))
        val_ids = list(chain(*[smiles_data[smiles] for smiles in val_smiles]))
        test_ids = list(chain(*[smiles_data[smiles] for smiles in test_smiles]))
        print(f'Splitting by molecule, train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}')
    else:
        raise ValueError(f"Invalid split_by: {split_by}")

    train_data1 = [dataset.data_list1[i] for i in train_ids]
    val_data1 = [dataset.data_list1[i] for i in val_ids]
    test_data1 = [dataset.data_list1[i] for i in test_ids]

    train_data2 = [dataset.data_list2[i] for i in train_ids]
    val_data2 = [dataset.data_list2[i] for i in val_ids]
    test_data2 = [dataset.data_list2[i] for i in test_ids]

    train_loader1 = DataLoader(train_data1, batch_size=batch_size)
    val_loader1 = DataLoader(val_data1, batch_size=batch_size)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size)

    train_loader2 = DataLoader(train_data2, batch_size=batch_size)
    val_loader2 = DataLoader(val_data2, batch_size=batch_size)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size)

    return (train_loader1, train_loader2), (val_loader1, val_loader2), (test_loader1, test_loader2)


class Normalizer:
    def __init__(self, train_loader, batch_size, do_normalize=True):
        self.mean = None
        self.std = None
        self.batch_size = batch_size
        self.do_normalize = do_normalize
        if self.do_normalize:
            self.get_mean_std(train_loader)
    
    def get_mean_std(self, train_loader):

        labels = []
        for data in train_loader:
            labels.append(data.y.reshape(-1, 1))
        labels = torch.cat(labels)
        self.mean = labels.mean(0).reshape((-1, 1))
        self.std = labels.std(0).reshape((-1, 1))

    def normalize(self, y, device='cuda:0'):
        y = y.to(device)
        if self.do_normalize:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            y = (y.reshape(-1, 1) - self.mean)/self.std
        return y
    
    def denormalize(self, y, device='cpu'):
        y = y.to(device)
        if self.do_normalize:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            y = y.reshape(-1, 1) * self.std + self.mean
        return y


def batch_flatten(model, data_loader, normalizer=None, logarithm=True):
    model.eval()
    model.to('cpu')
    y_true = []
    y_predict = []
    smiles_list = []
    temp_list = []
    for batch in data_loader:
        batch = batch.to('cpu')
        y_hat = model(batch).detach()
        y_hat = normalizer.denormalize(y_hat).flatten().tolist()
        y_true += batch.y.flatten().tolist()
        temp_list += batch.temps.flatten().tolist()
        y_predict += y_hat
        smiles_list += [smiles for smiles in batch.smiles for i in range(batch.temps.size(1))]
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)   

    if logarithm:
        y_true = np.exp(y_true)
        y_predict = np.exp(y_predict)
    return y_true, y_predict, smiles_list, temp_list


def mixture_batch_flatten(model, data_loader, logarithm=True):
    model.eval()
    model.to('cpu')
    y_true = []
    y_predict = []
    smiles_list1 = []
    smiles_list2 = []
    temp_list = []
    for batch1, batch2 in zip(data_loader[0], data_loader[1]):
        batch1 = batch1.to('cpu')
        batch2 = batch2.to('cpu')
        y_hat = model(batch1, batch2).detach().flatten().tolist()
        y_true += batch1.y.flatten().tolist()
        temp_list += batch1.temps.flatten().tolist()
        y_predict += y_hat
        smiles_list1 += [smiles for smiles in batch1.smiles for i in range(batch1.temps.size(1))]
        smiles_list2 += [smiles for smiles in batch2.smiles for i in range(batch1.temps.size(1))]
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)  
    
    if logarithm:
        y_true = np.exp(y_true)
        y_predict = np.exp(y_predict)
    return y_true, y_predict, smiles_list1, smiles_list2, temp_list


def save_results(model, data_loader, save_folder, model_type, logarithm=False, normalizer=None, train_mixture=False):
    if train_mixture:
        y_true, y_pred, smiles1, smiles2, temp_list = mixture_batch_flatten(model, data_loader, logarithm)
        smiles = [smiles1[i] + '.' + smiles2[i] for i in range(len(smiles1))]
    else:
        y_true, y_pred, smiles, temp_list = batch_flatten(model, data_loader, normalizer, logarithm)
    results = pd.DataFrame(np.array([temp_list, y_true, y_pred]).T, 
                           columns=['T', 'y_true', 'y_pred'], index=smiles)
    results.to_csv(os.path.join(save_folder, model_type + '.csv'))


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
