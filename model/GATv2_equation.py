# Note: This model is only for liquid viscosity prediction
import torch
from torch import nn

from .modules import MLP, GATv2Layer, ReadoutPhase


class GATv2EquationModel(nn.Module):
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
        self.pred_head = MLP(2*self.hidden_dim, 256, 4, 3, 0.2, nn.ELU())
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        mol_repr = self.readout(x, data.batch)
        params = self.pred_head(mol_repr)
        temps = self.temperature_basis(data.temps/100)
        viscosity = torch.matmul(temps, params.unsqueeze(-1))
        return viscosity.squeeze(-1)
    
    @staticmethod
    def temperature_basis(temp):
        return torch.stack([temp**0, 1/temp, temp, temp**2], axis=-1)