import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import global_add_pool, global_max_pool


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout, activation):
        super().__init__()
        if layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = []
            for i in range(layers - 1):
                self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                self.layers.append(activation)
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*self.layers)
        
        self.layers.apply(init_weight)
        
    def forward(self, x):
        output = self.layers(x)
        return output


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias != None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class RBFExpansion(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(RBFExpansion, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

        
class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, d_model: int, max_len):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]
    

class GATv2Layer(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int, num_heads: int,
                 activation=nn.PReLU(), concat: bool = True, residual: bool = True,
                 bias: bool = True, dropout: float = 0.1, share_weights: bool = False):
        super(GATv2Layer, self).__init__()

        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights

        # Embedding by linear projection
        self.linear_src = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
        if self.share_weights:
            self.linear_dst = self.linear_src
        else:
            self.linear_dst = nn.Linear(num_node_features, output_dim * num_heads, bias=False)


        # The learnable parameters to compute attention coefficients
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(num_node_features, num_heads * output_dim, bias=False)
        else:
            self.register_parameter('residual_linear', None)

        # Some fixed function
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)

        nn.init.xavier_uniform_(self.double_attn)
        if self.residual:
            if self.num_node_features != self.num_heads * self.output_dim:
                nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index):

        # Input preprocessing
        edge_src_index, edge_dst_index = edge_index

        # Projection on the new space
        src_projected = self.linear_src(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        dst_projected = self.linear_dst(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        

        #######################################
        ############## Edge Attn ##############
        #######################################

        # Edge attention coefficients
        edge_attn = self.leakyReLU((src_projected.index_select(0, edge_src_index)
                                    + dst_projected.index_select(0, edge_dst_index)))
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        num_nodes = x.shape[0]
        edge_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                          dtype=exp_edge_attn.dtype,
                                          device=exp_edge_attn.device)
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        # edge_attn shape = [num_edges, num_heads, 1]
        exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = src_projected.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                  dtype=exp_edge_attn.dtype,
                                  device=exp_edge_attn.device)
        edge_dst_index_broadcast = (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        output = edge_output
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(num_nodes, -1, self.output_dim)
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output


class ReadoutPhase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # readout phase
        self.weighting = nn.Linear(dim, 1) 
        self.score = nn.Sigmoid() 
        
        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)
    
    def forward(self, x, batch):
        weighted = self.weighting(x)
        score = self.score(weighted)
        output1 = global_add_pool(score * x, batch)
        output2 = global_max_pool(x, batch)
        
        output = torch.cat([output1, output2], dim=1)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(input_dim, d_model)
        self.wk = nn.Linear(input_dim, d_model)
        self.wv = nn.Linear(input_dim, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if input_dim == d_model:
            self.residual_linear = nn.Identity()
        else:
            self.residual_linear = nn.Linear(input_dim, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x (input features): [num_nodes, input_dim]
            mask (mask matrix): [num_heads, num_nodes, num_nodes]
        """
        bs = x.shape[0]
        residual = self.residual_linear(x)
        q = self.wq(x).view(bs, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(bs, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(bs, -1, self.num_heads, self.depth).transpose(1, 2)
        
        # shape of attn = [num_heads, num_nodes, num_nodes]
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)  
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # [num_nodes, num_heads * depth]
        output = torch.matmul(attn, v).transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.dense(output)
        output  = self.ln(residual + output)
        
        return output, attn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(model_dim, d_ff)
        self.w2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(model_dim)
        self.act = nn.ELU()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        
    def forward(self, x):
        output = self.act(self.w1(x))
        output = self.dropout(output)

        output = self.ln(x + self.dropout(self.w2(output)))
        return output



class BoltzmannLayer(nn.Module):
    def __init__(self, dim=128, num_levels=16):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.cal_energies = MLP(self.dim, 256, self.num_levels * 2, 3, 0.2, nn.ELU())
        self.cal_level_prob = nn.Softmax(dim=-1)
    
    def init_params(self):
        self.cal_energies.apply(init_weight)
    
    def forward(self, x, temperature, return_prob=False):
        energies = self.cal_energies(x)

        energies_double = energies.view(-1, 2, self.num_levels)
        energies = energies_double[:, 0, :]
        energies_value = energies_double[:, 1, :]
        coef = (1 / (8.314 * 0.001 * temperature)) # [batch_size, num_temperature]
        exponent = -1 *  torch.matmul(coef.unsqueeze(-1), energies.unsqueeze(1))
        exponent = exponent.view(-1, temperature.size(1), self.num_levels)
        prob = self.cal_level_prob(exponent)
        
        energy = (prob *energies_value.view(-1, 1, self.num_levels))
        if return_prob:
            return energy.squeeze(1), prob
        else:
            return energy.squeeze(1)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.attn = MultiHeadAttention(input_dim, d_model, heads, dropout=0.1)
        self.ff = PositionWiseFeedForward(d_model, d_model, dropout=dropout)

    def forward(self, x):
        if self.input_dim == self.d_model:
            x1 = self.attn(x)[0] + x
        else:
            x1 = self.attn(x)[0]
        x2 = x1 + self.ff(x1)
        return x2

    def get_attn(self, x):
        if self.input_dim == self.d_model:
            _, attn = self.attn(x)
        else:
            _, attn = self.attn(x)
        return attn

