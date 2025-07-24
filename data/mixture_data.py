import os

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Dataset
from tqdm import tqdm
from collections import defaultdict

from .data import MolPropData, PROCESSED_DIR, RAW_DIR
from .featurizer import MoleculeFeaturizer


class MixtureDataset(Dataset):
    def __init__(
        self, 
        name, 
        additional_features, 
        save_folder=PROCESSED_DIR, 
        replace=False,
        label_col='viscosity'
    ):
        super().__init__()
        self.processed_path = os.path.join(save_folder, f'{name}.pt')
        self.label_col = label_col
        self.additional_features = additional_features

        self.data_list1 = []
        self.data_list2 = []
        if replace or not os.path.exists(self.processed_path):
            self.df = pd.read_csv(RAW_DIR / f'{name}.csv')
            assert 'smiles1' in self.df.columns
            assert 'smiles2' in self.df.columns
            assert 'molar_ratio' in self.df.columns
            assert 'temperature' in self.df.columns
            assert label_col in self.df.columns
            self._process()
        else:
            self.data_list1, self.data_list2 = torch.load(self.processed_path)
     
    def _process(self):

        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[i, :]
            mol1 = Chem.MolFromSmiles(row.smiles1)
            mol2 = Chem.MolFromSmiles(row.smiles2)
            if mol1 is None:
                print(f"{row.smiles1} can not generate valid molecule")
                continue
            if mol2 is None:
                print(f"{row.smiles2} can not generate valid molecule")
                continue

            data1 = MolPropData(row.smiles1)
            data2 = MolPropData(row.smiles2)
            data1.y = torch.tensor([row[self.label_col]], dtype=torch.float32).unsqueeze(0)
            data2.y = torch.tensor([row[self.label_col]], dtype=torch.float32).unsqueeze(0)
            data1.temps = torch.tensor([row.temperature], dtype=torch.float32).unsqueeze(0)
            data2.temps = torch.tensor([row.temperature], dtype=torch.float32).unsqueeze(0)

            featurizer = MoleculeFeaturizer(additional_features=self.additional_features)
            
            ##########################
            ## molecule component 1 ##
            ##########################
            feature_dict1 = featurizer(mol1)
            
            # edge index and edges
            data1.edge_index = feature_dict1['edge_index']
            data1.x = torch.tensor(feature_dict1['x'], dtype=torch.float32)

            # pos, subgraph_index
            if 'pos' in self.additional_features:
                if feature_dict1['pos'] is None:
                    print('Warning: No positions found for smiles {}'.format(data1.smiles))
                    continue
                data1.pos = torch.tensor(feature_dict1['pos'], dtype=torch.float32)
            if 'subgraph_index' in self.additional_features:
                data1.triple_index = torch.LongTensor(feature_dict1['triple_index'])
                data1.quadra_index = torch.LongTensor(feature_dict1['quadra_index'])
            if 'mol_desc' in self.additional_features:
                data1.mol_desc = torch.FloatTensor(feature_dict1['mol_desc']).unsqueeze(0)
            
            ##########################
            ## molecule component 2 ##
            ##########################
            feature_dict2 = featurizer(mol2)
            
            # edge index and edges
            data2.edge_index = feature_dict2['edge_index']
            data2.x = torch.tensor(feature_dict2['x'], dtype=torch.float32)

            # pos, subgraph_index
            if 'pos' in self.additional_features:
                if feature_dict2['pos'] is None:
                    print('Warning: No positions found for smiles {}'.format(data2.smiles))
                    continue
                data2.pos = torch.tensor(feature_dict2['pos'], dtype=torch.float32)
            if 'subgraph_index' in self.additional_features:
                data2.triple_index = torch.LongTensor(feature_dict2['triple_index'])
                data2.quadra_index = torch.LongTensor(feature_dict2['quadra_index'])
            if 'mol_desc' in self.additional_features:
                data2.mol_desc = torch.FloatTensor(feature_dict2['mol_desc']).unsqueeze(0)
            
            data1.num_nodes = data1.x.size(0)
            data1.molar_ratio = row.molar_ratio
            data2.num_nodes = data2.x.size(0)
            
            self.data_list1.append(data1)
            self.data_list2.append(data2)
        torch.save((self.data_list1, self.data_list2), self.processed_path)
    
    def len(self):
        assert len(self.data_list1) == len(self.data_list2)
        return len(self.data_list1)
    
    def __repr__(self):
        return f'MixtureDataset(num_points: {len(self.data_list1)})'
    
    def groupby_smiles(self):
        smiles_data = defaultdict(list)
        for i, (data1, data2) in enumerate(zip(self.data_list1, self.data_list2)):
            smiles = data1.smiles + '.' + data2.smiles
            smiles_data[smiles].append(i)
        return smiles_data
    
    def logarithm(self):
        for data in self.data_list1:
            data.y = torch.log(data.y)
        for data in self.data_list2:
            data.y = torch.log(data.y)