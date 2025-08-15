import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GINConv, PNAConv, TransformerConv, GCNConv, GATConv
from torch_geometric.utils import degree

from .modules import MLP, BoltzmannLayer, ReadoutPhase


class TemperatureModel(nn.Module):
    def __init__(
        self,
        mol_repr_model: nn.Module,
        repr_dim: int,
        use_boltzmann: bool=False,
        num_energies: int=24,
    ):
        super(TemperatureModel, self).__init__()
        self.mol_repr_model = mol_repr_model
        self.use_boltzmann = use_boltzmann
        
        if use_boltzmann:
            self.pred_boltzmann = BoltzmannLayer(repr_dim, num_energies)
            self.pred_head = MLP(num_energies, 64, 1, 1, 0.1, nn.ELU())
        else:
            self.pred_head = MLP(repr_dim + 1, 256, 1, 3, 0.2, nn.ELU())
    
    def forward(self, batch: Data):
        mol_repr = self.mol_repr_model(batch)
        if self.use_boltzmann:
            energy_dist = self.pred_boltzmann(mol_repr, batch.temps)
            return self.pred_head(energy_dist)
        else:
            return self.pred_head(torch.cat([mol_repr, batch.temps], dim=-1))


class GIN(nn.Module):
    def __init__(
        self,
        num_atom_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float=0.2,
        n_mlp_layers: int=2,
    ):
        super(GIN, self).__init__()

        # GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GINConv(
                nn=MLP(
                    input_dim=num_atom_features if i==0 else hidden_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=hidden_dim,
                    layers=n_mlp_layers,
                    dropout=dropout,
                    activation=nn.ELU()
                )
            )
            self.layers.append(layer)
        
        # Readout phase
        self.readout = ReadoutPhase(hidden_dim)

    def forward(self, batch: Data):
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)

        mol_repr_all = self.readout(x, batch.batch)
        
        return mol_repr_all


class PNA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        deg: torch.Tensor,
        towers: int = 1,
        edge_dim: int = None,
    ):
        super().__init__()
        
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            conv = PNAConv(
                in_channels=in_channels if i == 0 else hidden_dim, 
                out_channels=hidden_dim,
                aggregators=aggregators, 
                scalers=scalers,
                deg=deg,
                edge_dim=edge_dim, 
                towers=towers, 
                pre_layers=1, 
                post_layers=1,
                divide_input=False
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.readout = ReadoutPhase(hidden_dim)

    def forward(self, batch: Data):
        x = batch.x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, batch.edge_index, batch.edge_attr)))
        return self.readout(x, batch.batch)
    
    @classmethod
    def compute_deg(
        cls,
        train_loader: DataLoader,
    ) -> torch.Tensor:
        max_degree = -1
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float=0.2,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerConv(
                in_channels=in_channels if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
            self.layers.append(layer)
        
        self.readout = ReadoutPhase(hidden_dim)
    
    def forward(self, batch: Data):
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        return self.readout(x, batch.batch)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float=0.2,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GCNConv(
                in_channels=in_channels if i == 0 else hidden_dim,
                out_channels=hidden_dim,
            )
            self.layers.append(layer)
        
        self.readout = ReadoutPhase(hidden_dim)
    
    def forward(self, batch: Data):
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        return self.readout(x, batch.batch)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float=0.2,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GATConv(
                in_channels=in_channels if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
            self.layers.append(layer)
        
        self.readout = ReadoutPhase(hidden_dim)
    
    def forward(self, batch: Data):
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        return self.readout(x, batch.batch)