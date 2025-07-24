import torch
from torch import nn

from .modules import MLP, BoltzmannLayer, GATv2Layer, ReadoutPhase


class BoltzGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers  
        self.num_heads = config.num_heads
        self.dropout = config.dropout 
        self.num_energies = config.num_energies
        self.return_repr = config.return_repr
        
        self.message_passing = nn.ModuleList()
        for i in range(self.num_layers):
            self.message_passing.append(
                GATv2Layer(
                    self.input_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim // self.num_heads,
                    self.num_heads,
                    dropout=self.dropout
                ))
        self.readout = ReadoutPhase(self.hidden_dim)
        
        self.pred_boltzmann = BoltzmannLayer(2*self.hidden_dim, self.num_energies)
        self.pred_head = MLP(self.num_energies, 64, 1, 1, 0.1, nn.ELU())
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        
        mol_repr = self.readout(x, data.batch)
        energy_distribution = self.pred_boltzmann(mol_repr, data.temps)
        output = self.pred_head(energy_distribution)

        if self.return_repr:
            return output, mol_repr
        else:
            return output

    def get_mol_repr(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        mol_repr = self.readout(x, data.batch)
        return mol_repr
    
    def get_energy_dist(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        mol_repr = self.readout(x, data.batch)
        _, prob = self.pred_boltzmann(mol_repr, data.temps, return_prob=True)
        return _, prob