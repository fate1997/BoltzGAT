from typing import List

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.sparse import coo_matrix

from .feature_utils import one_hot_encoding, subgraph_index

Atom = Chem.rdchem.Atom
Mol = Chem.Mol

ATOM_POOL =  [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20,
              21, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 
              49, 50, 51, 52, 53, 82]

class MoleculeFeaturizer(object):
    def __init__(self, 
                 additional_features: List[str] = None):
        if additional_features is None:
            additional_features = []
        self.additional_features = additional_features

    def _atom_featurizer(self, atom: Atom) -> List[float]:
        atom_type_value = atom.GetAtomicNum()
        atom_type_features = one_hot_encoding(atom_type_value, ATOM_POOL)

        # atom degree one-hot encoding
        degree_choices = list(range(5))
        degree = atom.GetTotalDegree()
        degree_features = one_hot_encoding(degree, degree_choices)

        # atom hybridization one-hot encoding
        hybridization_choices = list(range(len(Chem.HybridizationType.names)-1))
        hybridization = int(atom.GetHybridization())
        hybrid_features = one_hot_encoding(hybridization, hybridization_choices)

        # chiral tag one-hot encoding
        chiral_tag_choices = list(range(len(Chem.ChiralType.names)-1))
        chiral_tag = atom.GetChiralTag()
        chiral_tag_features = one_hot_encoding(chiral_tag, chiral_tag_choices)

        # number of hydrogens one-hot encoding
        num_Hs_choices = list(range(5))
        num_Hs = atom.GetTotalNumHs()
        num_Hs_features = one_hot_encoding(num_Hs, num_Hs_choices)

        # aromatic
        aromatic_features = [1 if atom.GetIsAromatic() else 0]
        
        # concat features
        atom_features = atom_type_features + degree_features + hybrid_features + \
                        chiral_tag_features + num_Hs_features + aromatic_features
        return atom_features
    
    def _get_postions(self, mol: Mol):
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        num_atoms = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        if success == -1:
            return None
        elif success == 0:
            return mol.GetConformer().GetPositions()[:num_atoms, :]

    def _bond_featurizer(self, bond: Chem.rdchem.Bond):
        # bond type one-hot encoding
        bond_type = bond.GetBondType()
        bond_type_one_hot_encoding = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC)
        ]
        addition_bond_features = [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())]

        return bond_type_one_hot_encoding + addition_bond_features

    def _rdkit_desc(self, mol):
        # get calculator method for descriptor
        calc = MoleculeDescriptors.MolecularDescriptorCalculator\
            ([x[0] for x in Descriptors._descList])
        ds = calc.CalcDescriptors(mol)
        return list(ds)

    def _morgan_desc(self, mol, radius=2, num_bits=1024):
        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, num_bits, useChirality=True)
        morgan_fingerprint_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_array)
        return morgan_fingerprint_array
    
    @property
    def feature_keys(self) -> List:
        return self.feature_dict.keys()

    def __call__(self, mol: Mol):
        self.feature_dict = {}
        
        # atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self._atom_featurizer(atom)) 
        self.feature_dict['x'] = atom_features

        # edge index and edges
        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        edge_index = np.array([coo_adj.row, coo_adj.col], dtype=int)
        self.feature_dict['edge_index'] = torch.from_numpy(edge_index).long()

        # position
        if 'pos' in self.additional_features:
            self.feature_dict['pos'] = self._get_postions(mol)
        # subgraph_index
        if 'subgraph_index' in self.additional_features:
            adj = Chem.GetAdjacencyMatrix(mol)
            G = nx.from_numpy_matrix(adj)
            self.feature_dict['triple_index'] = subgraph_index(G, 2)
            self.feature_dict['quadra_index'] = subgraph_index(G, 3)
        # molecular desc
        if 'mol_desc' in self.additional_features:
            # self.feature_dict['mol_desc'] = self._rdkit_desc(mol)
            self.feature_dict['mol_desc'] = self._morgan_desc(mol)
        return self.feature_dict

    