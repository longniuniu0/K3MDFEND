import random
import numpy as np
import torch
from Process.augmentation import *

class DictClass:
    def __init__(self, edge_index, BU_edge_index, aug1_x, aug2_x, aug3_x, aug2_edge_index, aug3_edge_index, ori_edge_index, batch):
        self.edge_index = torch.LongTensor(edge_index)
        self.BU_edge_index = torch.LongTensor(BU_edge_index)
        self.aug1_x = aug1_x
        self.aug2_x = aug2_x
        self.aug3_x = aug3_x
        self.aug2_edge_index = aug2_edge_index
        self.aug3_edge_index = aug3_edge_index
        self.ori_edge_index = ori_edge_index
        self.batch = batch

def process_loader(shared_feature, tddroprate=0, budroprate=0, drop_edge_rate_1=0.3, drop_edge_rate_2=0.4):
    col = []
    row = []
    batch = []
    tag_e = 0
    tag_b = 0
    for i in range(shared_feature.shape[0]):
        if (i + 1) % shared_feature.shape[0] == 0:
            tag_e += shared_feature.shape[0]
            tag_b += 1
        col.append(tag_e)
        row.append(i)
        batch.append(tag_b)
    batch = torch.tensor(batch)
    edge_index = [row, col]

    edgeindex = torch.LongTensor(edge_index)

    drop_weights = degree_drop_weights(edgeindex)
    edge_index_ = to_undirected(edgeindex)
    node_deg = degree(edge_index_[1])
    shared_feature = shared_feature.clone().detach().float()

    idx = np.random.permutation(shared_feature.shape[0])
    aug1_x = shared_feature[idx, :]
    def drop_edge(drop_weights, idx: int):
        if idx == 1:
            return drop_edge_weighted(edgeindex, drop_weights, p=drop_edge_rate_1, threshold=0.7)
        elif idx == 2:
            return drop_edge_weighted(edgeindex, drop_weights, p=drop_edge_rate_2, threshold=0.7)

    aug2_edge_index = drop_edge(drop_weights, 1)
    aug3_edge_index = drop_edge(drop_weights, 2)
    aug2_x = drop_feature_weighted_2(shared_feature, node_deg, drop_edge_rate_1)
    aug3_x = drop_feature_weighted_2(shared_feature, node_deg, drop_edge_rate_2)

    if (tddroprate > 0):
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        length = len(row)
        poslist = random.sample(range(length), int(length * (1 - tddroprate)))
        poslist = sorted(poslist)
        row = list(np.array(row)[poslist])
        col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]
    else:
        new_edgeindex = edgeindex

    burow = list(edgeindex[1])
    bucol = list(edgeindex[0])
    if (budroprate > 0):
        length = len(burow)
        poslist = random.sample(range(length), int(length * (1 - budroprate)))
        poslist = sorted(poslist)
        row = list(np.array(burow)[poslist])
        col = list(np.array(bucol)[poslist])
        bunew_edgeindex = [row, col]
    else:
        bunew_edgeindex = [burow, bucol]

    return DictClass(new_edgeindex, bunew_edgeindex, aug1_x, aug2_x, aug3_x, aug2_edge_index, aug3_edge_index, edgeindex, batch)