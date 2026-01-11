import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import pathlib

from data.data import AVAILABLE_MIXTURE_DATASETS
from model.BoltzGAT_mixture import BoltzGAT4Mixture
from model.GATv2_concat_mixture import GATv2Concat4Mixture
from train_utils import (EarlyStopping, Normalizer, mixture2iter, evaluate_score,
                         get_logger, load_config, save_results)

THIS_DIR = pathlib.Path(__file__).parent

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, logger):

    seed = config.train.seed
    save_folder = config.train.save_folder
    ckpt_folder = os.path.join(save_folder, 'checkpoints')
    result_folder = os.path.join(save_folder, 'pred_results')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    desc = config.desc
    model_type = config.model_type
    
    logger.info("-"*60)
    logger.info(desc.center(60, '-'))
    logger.info("-"*60)
    logger.info(vars(config))
    
    # Load train config
    num_epochs = config.train.num_epochs
    lr = config.train.lr 
    patience = config.train.patience
    device = config.train.device
    log_interval = config.train.log_interval

    # Load data
    logger.info("-----------Dataset Loading-----------")
    batch_size = config.data.batch_size
    name = config.data.name
    logarithm = config.data.logarithm
    additional_features = config.data.additional_features
    replace = config.data.replace
    train_ratio = config.data.train_ratio
    label_col = config.data.label_col
    split_by = config.data.split_by
    train_loader, val_loader, test_loader = mixture2iter(name, 
                                                      batch_size, 
                                                      train_ratio,
                                                      additional_features, 
                                                      replace,
                                                      logarithm,
                                                      label_col,
                                                      split_by)

    # Training init
    seed_all(seed)
    early_stopping = EarlyStopping(patience=patience, 
                                   path=os.path.join(save_folder, 'checkpoints/'),
                                   model_type=model_type)
    criterion = nn.L1Loss()

    logger.info("------------Model Creating-----------")
    model = BoltzGAT4Mixture(config)
    # model = GATv2Concat4Mixture(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("------------Train Running------------")
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        num_examples = 0
        for batch1, batch2 in zip(train_loader[0], train_loader[1]):
            model.to(device)
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            y = batch2.y
            outputs = model(batch1, batch2)
            if torch.isnan(outputs).any():
                print(f'outputs is nan: {batch1.smiles}')
                assert False
            if torch.isnan(y).any():
                print(f'y: {y}')
                assert False
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_examples += y.shape[0]
            loss_sum += loss.item() * y.shape[0]
        val_metric = evaluate_score(model, val_loader, criterion, logarithm, train_mixture=True)

        if epoch % log_interval == 0:
            logger.info(f'epoch:{epoch}, loss = {loss_sum / num_examples: .4f}, '
                        f'val loss = {val_metric["loss"]:.4f}, '
                        f'val AARD = {np.round(val_metric["AARD"], decimals=4)}, '
                        f'val R2 = {np.round(val_metric["R2"], decimals=4)}')
    
        # Early stopping
        min_metrics = np.array(val_metric["AARD"]).mean()
        early_stopping(min_metrics, model)
        if early_stopping.early_stop:
            logger.info('------------Early stopping------------')
            break

    model.load_state_dict(torch.load(os.path.join(ckpt_folder, model_type + '.pt')))
    test_metric = evaluate_score(model, test_loader, criterion, logarithm, train_mixture=True)
    val_metric = evaluate_score(model, val_loader, criterion, logarithm, train_mixture=True)

    logger.info(f'test RMSE = {np.round(test_metric["RMSE"], decimals=4)}, '
                f'test MAE = {np.round(test_metric["MAE"], decimals=4)}, '
                f'test AARD = {np.round(test_metric["AARD"], decimals=4)}, '
                f'test R2 = {np.round(test_metric["R2"], decimals=4)}, ')

    save_results(model, train_loader, result_folder, model_type+'_train', logarithm, train_mixture=True)
    save_results(model, val_loader, result_folder, model_type+'_val', logarithm, train_mixture=True)
    save_results(model, test_loader, result_folder, model_type+'_test', logarithm, train_mixture=True)
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', default='Testing', type=str)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', default='BoltzGAT', type=str, 
                        choices=['BoltzGAT', 'GATv2Concat'])
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--label_col', default='viscosity', type=str)
    parser.add_argument('--add_pos_enc', action='store_true')
    parser.add_argument('--split_by', default='entry', type=str, choices=['entry', 'mol'])
    args = parser.parse_args()
    
    config_dir = THIS_DIR / 'config'
    default_config = config_dir / 'train_mixture.yml'
    log_config_path = config_dir / 'log_config.json'
    
    assert args.dataset in AVAILABLE_MIXTURE_DATASETS, f"Dataset {args.dataset} not found"
        
    config = load_config(default_config)
    config.desc = args.desc
    config.data.name = args.dataset
    config.data.replace = args.replace
    config.model.name = args.model
    config.train.pretrained_path = args.pretrained_path
    config.train.lr = args.lr
    config.data.label_col = args.label_col
    if args.dataset != 'viscosity_mixture':
        config.data.logarithm = False
    config.model.add_pos_enc = args.add_pos_enc
    config.data.split_by = args.split_by
    save_folder = config.train.save_folder
    model_type = args.model + '_' + args.dataset + '_' + config.desc
    config.model_type = model_type
    log_dir = os.path.join(save_folder, 'logs')
    logger = get_logger(model_type+'.log', log_dir, log_config_path)
    
    train(config, logger)
    