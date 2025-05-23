import torch
from torch_geometric.utils import degree, to_undirected


def drop_feature(x, drop_prob):#随机丢弃一些特征
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):#根据权重丢弃一些特征
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):#根据权重w丢弃一些特征

    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)

    mean = w.mean().item()  # 将 mean 转换为标量
    std = w.std().item()
    drop_prob = torch.normal(mean=mean, std=std, size=(x.size(1),))
    drop_prob = torch.clamp(drop_prob, min=0, max=1)
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool) # 扩展掩码形状以匹配 x 的形状

    x = x.clone()
    x[:,drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):#计算丢弃特征权重
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):#根据权重丢弃一些边
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):#计算边的丢弃权重
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights
