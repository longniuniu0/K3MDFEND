import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)#双线

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):#全局 正样本 负样本
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)#得到一个相似性分数
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)#
        logits = torch.unsqueeze(logits, 0)
        return logits

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        assert k >= 2
        self.k = k
        self.skip = skip
        
        if not self.skip:
            self.linear = nn.ModuleList([nn.Linear(in_channels if i == 0 else 2 * out_channels, 
                                                    2 * out_channels) for i in range(k - 1)])
            self.linear.append(nn.Linear(2 * out_channels, out_channels))
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.linear = nn.ModuleList([nn.Linear(in_channels if i == 0 else out_channels, 
                                                    out_channels) for i in range(k)])
            self.activation = activation

    def forward(self, x: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.linear[i](x))
            return x
        else:
            h = self.activation(self.linear[0](x))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.linear[i](u)))
            return hs[-1]

def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fine_tune_weights(domain_emb_all, domain_embedding, attention_weights):
    num_samples = attention_weights.shape[0]
    num_classes = attention_weights.shape[2]
    modified_attention = attention_weights.clone().to(device)  

    d = 0.0005 
    for i in range(num_samples):
        sorted_indices = torch.argsort(modified_attention[i, 0, :], descending=True).cpu().numpy() 
        for j, idx in enumerate(sorted_indices):
            if j < num_classes // 2:
                modified_attention[i, 0, idx] = modified_attention[i, 0, idx] - d * (num_classes // 2 - j)
            else:
                modified_attention[i, 0, idx] = modified_attention[i, 0, idx] + (j - num_classes // 2) * d

    general_domain_embedding = torch.mm(modified_attention.squeeze(1), domain_emb_all)
    gate_input_combined = torch.cat([domain_embedding, general_domain_embedding], dim=-1)
    return gate_input_combined

def adjust_weights(domain_emb_all, domain_embedding, attention_weights, category_indices):
    num_samples = attention_weights.shape[0]
    adjusted_attention = attention_weights.clone().to(device)
    
    for i in range(num_samples):
        domain_idx = category_indices[i]
        adjusted_attention[i, 0, domain_idx] = 0 
        remaining_weight = torch.sum(adjusted_attention[i, 0, :])
        
        if remaining_weight > 0:
            adjusted_attention[i, 0, :] = adjusted_attention[i, 0, :].clone() / remaining_weight 

    general_domain_embedding = torch.mm(adjusted_attention.squeeze(1), domain_emb_all)
    
    shuffled_indices = category_indices.clone().detach().view(-1, 1).to(device)
    shuffled_indices = shuffled_indices[torch.randperm(category_indices.shape[0])].squeeze(1)
    gate_input_combined = torch.cat([domain_embedding, general_domain_embedding], dim=-1)

    return gate_input_combined

