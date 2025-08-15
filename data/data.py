import os
import pathlib
import pickle
import random
from typing import List

import pandas as pd
import torch
from rdkit import Chem
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from .featurizer import MoleculeFeaturizer

RAW_DIR = pathlib.Path(__file__).parent.parent / 'database/raw'
PROCESSED_DIR = pathlib.Path(__file__).parent.parent / 'database/processed'
AVAILABLE_PURE_DATASETS = [
 'thermal_cond_L', 'thermal_cond_G',
 'diffusion_coef_air', 'diffusion_coef_water',
 'delta_Sf', 'delta_Hf', 'delta_Uf', 'delta_Gf', 'delta_Af',
 'viscosity_G', 'viscosity_L',
 'Cp_S', 'Cp_L', 'Cp_G',
 'gas_entropy',
 'density',
 'H_vap',
 'surface_tension',
]
AVAILABLE_MIXTURE_DATASETS = [
     'viscosity_mixture', 'BigSolDBv2'
]

class MolPropData(Data):
    def __init__(self, smiles: str=None):
        super().__init__()
        self.smiles: str = smiles
        
        # Temperature and corresponding property value
        self.temps: torch.FloatTensor = None
        self.y: torch.FloatTensor = None
    
    def normalize(self, mean: Tensor, std: Tensor) -> Tensor:
        mean = mean.to(self.y.device)
        std = std.to(self.y.device)
        self.y = (self.y - mean)/std
        return self.y
    
    def denomarlzie(self, mean: Tensor, std: Tensor) -> Tensor:
        mean = mean.to(self.y.device)
        std = std.to(self.y.device)
        self.y = self.y * std + mean
        return self.y
    
    def __repr__(self):
        return f'MolPropData(SMILES={self.smiles}, num_atomse={self.num_nodes})'    
            
 
class MolPropDataset(Dataset):
    def __init__(
        self, 
        name: str,
        additional_features: List[str]=None,
        replace: bool=False, 
        raw_dir: pathlib.Path=RAW_DIR,
        save_dir: pathlib.Path=PROCESSED_DIR
    ):
        
        if name not in AVAILABLE_PURE_DATASETS:
            raise ValueError(f'{name} is not available in {raw_dir}')
        
        self.data_list = []
        self.raw_file_path = str(raw_dir / f'{name}.pickle')
        self.additional_features = additional_features
        self.processed_path = save_dir / f'{name}.pt'
        if replace or not os.path.exists(self.processed_path):
            self._process()
        else:
            self.data_list = torch.load(self.processed_path)
    
    def _load_data(self):
        if self.raw_file_path.endswith('.pickle'):
            with open(self.raw_file_path, 'rb') as f:
                raw_dataset = pickle.load(f)
        elif self.raw_file_path.endswith('.csv'):
            df = pd.read_csv(self.raw_file_path)
            smiles_group = df.groupby('SMILES').groups
            raw_dataset = []
            for smiles, indices in smiles_group.items():
                raw_data = {
                    'SMILES': smiles,
                    'y': df.iloc[indices]['y'].values,
                    'Temperature': df.iloc[indices]['Temperature'].values,
                }
                raw_dataset.append(raw_data)
        else:
            raise ValueError(f'Invalid format: {self.format}')
        return raw_dataset
    
    def _process(self):
        raw_dataset = self._load_data()
        for raw_data in tqdm(raw_dataset):
            mol = Chem.MolFromSmiles(raw_data['SMILES'])
            if mol is None:
                print(f"{raw_data['SMILES']} can not be loaded by RDKit")
                continue
            
            data = MolPropData(smiles=raw_data['SMILES'])
            data.y = torch.FloatTensor([raw_data['y']])
            data.temps = torch.FloatTensor([raw_data['Temperature']])
            assert data.temps.size() == data.y.size()

            featurizer = MoleculeFeaturizer(additional_features=self.additional_features)
            feature_dict = featurizer(mol)
            
            data.edge_index = feature_dict['edge_index']
            data.x = torch.tensor(feature_dict['x'], dtype=torch.float32)
            data.num_nodes = data.x.size(0)

            # Additional features if needed
            if 'pos' in self.additional_features:
                if feature_dict['pos'] is None:
                    print(f'Warning: No positions found for smiles {data.smiles}')
                    continue
                data.pos = torch.tensor(feature_dict['pos'], dtype=torch.float32)
            if 'subgraph_index' in self.additional_features:
                data.triple_index = torch.LongTensor(feature_dict['triple_index'])
                data.quadra_index = torch.LongTensor(feature_dict['quadra_index'])
            if 'mol_desc' in self.additional_features:
                data.mol_desc = torch.FloatTensor(feature_dict['mol_desc']).unsqueeze(0)

            self.data_list.append(data)
        
        torch.save(self.data_list, self.processed_path)
    
    def logarithm(self):
        for data in self.data_list:
            data.y = torch.log(data.y)

    def shuffle(self, seed):
        r = random.random
        random.seed(seed)
        random.shuffle(self.data_list, random=r)

    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)
    
    def __repr__(self):
        return f'MolPropDataset(num_mols={self.__len__()})'