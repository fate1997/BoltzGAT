import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import pathlib

from data.data import AVAILABLE_DATASETS
from model.BoltzGAT import BoltzGAT
from model.GATv2_equation import GATv2EquationModel
from model.GATv2_concat import GATv2ConcatModel
from train_utils import (EarlyStopping, Normalizer, data2iter, evaluate_score,
                         get_logger, load_config, save_results)

THIS_DIR = pathlib.Path(__file__).parent

def seed_everything(seed: int):
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
    
    logger.info("#"*60)
    logger.info(desc.center(60, '#'))
    logger.info("#"*60)
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
    data_name = config.data.name
    logarithm = config.data.logarithm
    additional_features = config.data.additional_features
    replace = config.data.replace
    train_ratio = config.data.train_ratio
    train_loader, val_loader, test_loader = data2iter(
        data_name, 
        batch_size, 
        train_ratio,
        additional_features, 
        replace, 
        logarithm
    )

    # Training init
    seed_everything(seed)
    early_stopping = EarlyStopping(patience=patience, 
                                   path=os.path.join(save_folder, 'checkpoints/'),
                                   model_type=model_type)
    criterion = nn.HuberLoss()
    normalizer = Normalizer(train_loader, batch_size, config.train.normalize)


    logger.info("------------Model Creating-----------")
    model_name = config.model.name
    if model_name.lower() == 'GATv2Concat'.lower():
        model = GATv2ConcatModel(config.model)
    elif model_name.lower() == 'BoltzGAT'.lower():
        model = BoltzGAT(config.model)
    elif model_name.lower() == 'GATv2Equation'.lower():
        if data_name != 'viscosity_L.pickle':
            raise ValueError("GATv2Equation model is only for liquid viscosity prediction")
        model = GATv2EquationModel(config.model)
    else:
        raise ValueError(f"Model {model_name} not found")
    
    if config.train.pretrained_path:
        model.load_state_dict(torch.load(config.train.pretrained_path))
        for _, param in model.message_passing.named_parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        model.parameters()), lr=lr)

    logger.info("------------Train Running------------")
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        num_examples = 0
        for i, batch in enumerate(train_loader):
            model.to(device)
            batch = batch.to(device)
            y = batch.y
            y = normalizer.normalize(y)
            outputs = model(batch)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_examples += y.shape[0]
            loss_sum += loss.item() * y.shape[0]
        val_metric = evaluate_score(model, val_loader, criterion, logarithm, normalizer)

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
    test_metric = evaluate_score(model, test_loader, criterion, logarithm, normalizer)

    logger.info(f'test MAE = {np.round(test_metric["MAE"], decimals=4)}, '
                f'test RMSE = {np.round(test_metric["RMSE"], decimals=4)}, '
                f'test AARD = {np.round(test_metric["AARD"], decimals=4)}, '
                f'test R2 = {np.round(test_metric["R2"], decimals=4)}, ')

    save_results(model, train_loader, result_folder, model_type+'_train', logarithm, normalizer)
    save_results(model, val_loader, result_folder, model_type+'_val', logarithm, normalizer)
    save_results(model, test_loader, result_folder, model_type+'_test', logarithm, normalizer)
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', default='Testing', type=str)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--model', default='BoltzGAT', type=str, 
                        choices=['BoltzGAT', 'GATv2Concat', 'GATv2Equation'])
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    
    if args.datasets == ['all']:
        args.datasets = AVAILABLE_DATASETS
    if args.pretrained_path:
        assert len(args.datasets) == 1, "Pretrained path can only be used for single dataset"
    
    config_dir = THIS_DIR / 'config'
    default_config = config_dir / 'train.yml'
    log_config_path = config_dir / 'log_config.json'
    non_logarithm_datasets = ['delta_Hf', 'delta_Uf', 'delta_Sf', 
                              'delta_Gf', 'delta_Af']
    for dataset in args.datasets:
        assert dataset in AVAILABLE_DATASETS, f"Dataset {dataset} not found"
        if dataset == 'viscosity_mixture':
            continue
        config = load_config(default_config)
        config.desc = args.desc
        config.data.name = dataset
        config.data.replace = args.replace
        config.model.name = args.model
        config.train.pretrained_path = args.pretrained_path
        config.train.lr = args.lr
        
        if dataset in non_logarithm_datasets:
            config.data.logarithm = False

        save_folder = config.train.save_folder
        model_type = args.model + '_' + dataset + '_' + config.desc
        config.model_type = model_type
        log_dir = os.path.join(save_folder, 'logs')
        logger = get_logger(model_type+'.log', log_dir, log_config_path)
        
        train(config, logger)