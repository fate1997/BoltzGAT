import torch
from torch import nn

from .BoltzGAT import BoltzGAT
from .GATv2_concat import GATv2ConcatModel
from .modules import (MLP, BoltzmannLayer, GATv2Layer, ReadoutPhase,
                      TransformerEncoder, LearnablePositionalEncoding)


class Interaction(nn.Module):
    def __init__(self, in_dim, dim, heads, dropout, num_layers, add_pos_enc=False):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.repr_token = nn.Parameter(torch.FloatTensor(1, in_dim))
        self.encoders = nn.ModuleList([TransformerEncoder(in_dim, dim, heads, dropout) if i == 0 else TransformerEncoder(dim, dim, heads, dropout) for i in range(num_layers)])
        self.init_params()
        
        self.add_pos_enc = add_pos_enc
        if self.add_pos_enc:
            self.pos_enc = LearnablePositionalEncoding(in_dim, 3)
    
    def init_params(self):
        self.repr_token.data.normal_(0, 0.02)
    
    def forward(self, x1, x2):
        in_set = torch.stack([self.repr_token.expand(x1.size(0), self.in_dim), x1, x2], dim=1)
        if self.add_pos_enc:
            in_set = self.pos_enc(in_set)
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                output = encoder(in_set)
            else:
                output = encoder(output)
        return output[:, 0, :]
    
    def get_attn(self, x1, x2):
        in_set = torch.stack([self.repr_token.expand(x1.size(0), self.in_dim), x1, x2], dim=1)
        attn_list = []
        output = in_set
        for i, encoder in enumerate(self.encoders):
            attn = encoder.get_attn(output)
            attn_list.append(attn)
            output = encoder(output)
        return attn_list

class BoltzGAT4Mixture(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers  
        self.num_heads = config.model.num_heads
        self.dropout = config.model.dropout 
        self.num_energies = config.model.num_energies
        self.return_repr = config.model.return_repr
        self.pretrained_path = config.train.pretrained_path
        self.add_pos_enc = config.model.add_pos_enc
        
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
        self.pred_boltzmann = BoltzmannLayer(self.hidden_dim*2 + 2, self.num_energies)
        self.pred_head = MLP(self.num_energies, 128, 1, 2, 0.1, nn.ELU())

        self.inter_module = Interaction(2*self.hidden_dim, 128, 8, 0.1, 3, self.add_pos_enc)

        # pretrain
        if self.pretrained_path is not None:
            if 'GATv2Concat' in self.pretrained_path:
                self.mol_repr_model = GATv2ConcatModel(config.model)
            elif 'BoltzGAT' in self.pretrained_path:
                self.mol_repr_model = BoltzGAT(config.model)
            else:
                raise ValueError(f'Pretrained model {self.pretrained_path} not found')
            self.mol_repr_model.eval()
            self.mol_repr_model.load_state_dict(torch.load(config.train.pretrained_path))
            print(f'Pretrained model loaded from {config.train.pretrained_path}')
            for _, param in self.mol_repr_model.named_parameters():
                param.requires_grad = False
    
    def forward_pure(self, data):
        if self.pretrained_path is not None:
            _, mol_repr = self.mol_repr_model(data)
        else:
            x, edge_index = data.x, data.edge_index
            for layer in self.message_passing:
                x = layer(x, edge_index)
            mol_repr = self.readout(x, data.batch)
        return mol_repr

    def forward(self, data1, data2):
        mol_repr1 = self.forward_pure(data1)
        mol_repr2 = self.forward_pure(data2)

        x = data1.molar_ratio.reshape(-1, 1).float()
 
        inter_repr = self.inter_module(mol_repr1, mol_repr2)
        inter_repr = torch.cat([inter_repr, x, 1-x], dim=1)
        energy_distribution = self.pred_boltzmann(inter_repr, data1.temps)
        delta_visc = self.pred_head(energy_distribution)

        return delta_visc
    
    def get_energy_dist(self, data1, data2):
        mol_repr1 = self.forward_pure(data1)
        mol_repr2 = self.forward_pure(data2)

        x = data1.molar_ratio.reshape(-1, 1).float()
        
        inter_repr = self.inter_module(mol_repr1, mol_repr2)
        inter_repr = torch.cat([inter_repr, x, 1-x], dim=1)
        _, prob = self.pred_boltzmann(inter_repr, data1.temps, return_prob=True)
        return prob
