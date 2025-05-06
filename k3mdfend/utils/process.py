from torch_geometric.nn import GCNConv
import torch as th
from transformers import BertModel
from transformers import RobertaModel
from models.layers import *
from torch_geometric.data import Data
from torch_scatter import scatter_mean
import copy
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
# dic_prom = [
#             'expert_comment', 'user_comment', 'reporter_comment',
#             'user_to_expert','expert_to_user',
#             'expert_to_reporter', 'user_to_reporter'
#         ]
dic_prom = [
            'expert_comment', 'user_comment',
        ]
import torch.nn as nn

class BertGCN(nn.Module):
    def __init__(self, dataset, input_dim, hidden_dim, out_dim, emb_dim):
        super(BertGCN, self).__init__()
        self.ROGCN = ROGCN(input_dim, hidden_dim, out_dim)
        self.EOGCN = EOGCN(input_dim, hidden_dim, out_dim)
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
        self.attention = MaskAttention(emb_dim)
        self.fc = nn.Linear(2 * (out_dim + hidden_dim), 768)
    def forward(self, data):

        content_feature = self.bert(data['content'], attention_mask=data['content_masks'])[0]
        gate_content_feature, _ = self.attention(content_feature, data['content_masks'])

        comments_feature = self.bert(data['comments'], attention_mask=data['comments_masks'])[0]
        gate_comments_feature, _ = self.attention(comments_feature, data['comments_masks'])

        expert_feature = self.bert(data['expert_comment'], attention_mask=data['expert_comment_masks'])[0]
        gate_expert_feature, _ = self.attention(expert_feature, data['expert_comment_masks'])

        user_feature = self.bert(data['user_comment'], attention_mask=data['user_comment_masks'])[0]
        gate_user_feature, _ = self.attention(user_feature, data['user_comment_masks'])

        reporter_feature = self.bert(data['reporter_comment'], attention_mask=data['reporter_comment_masks'])[0]
        gate_reporter_feature, _ = self.attention(reporter_feature, data['reporter_comment_masks'])

        user_to_expert_feature = self.bert(data['user_to_expert'], attention_mask=data['user_to_expert_masks'])[0]
        gate_user_to_expert_feature, _ = self.attention(user_to_expert_feature, data['user_to_expert_masks'])

        expert_to_user_feature = self.bert(data['expert_to_user'], attention_mask=data['expert_to_user_masks'])[0]
        gate_expert_to_user_feature, _ = self.attention(expert_to_user_feature, data['expert_to_user_masks'])

        expert_to_reporter_feature = self.bert(data['expert_to_reporter'], attention_mask=data['expert_to_reporter_masks'])[0]
        gate_expert_to_reporter_feature, _ = self.attention(expert_to_reporter_feature, data['expert_to_reporter_masks'])

        user_to_reporter_feature = self.bert(data['user_to_reporter'], attention_mask=data['user_to_reporter_masks'])[0]
        gate_user_to_reporter_feature, _ = self.attention(user_to_reporter_feature, data['user_to_reporter_masks'])

        combined_feature = feature_gather(gate_content_feature, gate_comments_feature, gate_expert_feature, gate_user_feature, 
                                          gate_reporter_feature, gate_user_to_expert_feature, gate_user_to_reporter_feature, gate_expert_to_user_feature, gate_expert_to_reporter_feature, )
        #combined_feature = feature_gather(gate_content_feature, gate_comments_feature)
        graph_data = Graph_Create(combined_feature)

        num_nodes = combined_feature.size(0)  # 获取节点总数
        batch_size = num_nodes // 9  # 计算图的数量（每 9 个节点为一个图）
        batch = th.repeat_interleave(th.arange(batch_size),9).to(device)
        TD_t = self.ROGCN(graph_data.x, graph_data.edge_index, batch)#[512,768]
        TD_t = scatter_mean(TD_t, batch, dim=0)#[64,768]
        BU_t = self.EOGCN(graph_data.x, graph_data.T_edge_index, batch)
        BU_t = scatter_mean(BU_t, batch, dim=0)
        t_ = th.cat((BU_t, TD_t), 1)#
        t = self.fc(t_)

        return t


class ROGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(ROGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, edge_index, batch):
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        expert_extend = th.zeros(len(batch), x1.size(1)).to(device)
        for i in range(0, x.size(0), 9):
            expert_extend[i:i + 9] = x1[i].unsqueeze(0).expand(9, -1)
        x = th.cat((x, expert_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(batch), x2.size(1)).to(device)
        for i in range(0, x.size(0), 9):
            root_extend[i:i + 9] = x2[i].unsqueeze(0).expand(9, -1)#512
        x = th.cat((x, root_extend), 1)#512+256
        return x
class EOGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(EOGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, edge_index, batch):
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        expert_extend = th.zeros(len(batch), x1.size(1)).to(device)
        for i in range(0, x.size(0), 9):
            expert_extend[i:i + 9] = x1[i].unsqueeze(0).expand(9, -1)
        x = th.cat((x, expert_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(batch), x2.size(1)).to(device)
        for i in range(0, x.size(0), 9):
            root_extend[i:i + 9] = x2[i].unsqueeze(0).expand(9, -1)
        x = th.cat((x, root_extend), 1)
        return x

def feature_gather(content, comments, expert, user, reporter, user_to_expert, expert_to_user, expert_to_reporter,  user_to_reporter):
    return th.cat((content, comments, expert, user, reporter, user_to_expert, expert_to_user, expert_to_reporter,  user_to_reporter), 0)
# def feature_gather(content, comments):
#     return th.cat((content, comments), 0)
def Graph_Create(combined_feature):
    edge_index = th.tensor([[0, 0, 0, 0, 2, 3, 4, 4],
                            [1, 2, 3, 4, 3, 2, 2, 3]], dtype=th.long, device=device)
    T_edge_index = th.tensor([[1, 2, 3, 4, 3, 2, 2, 3],
                            [0, 0, 0, 0, 2, 3, 4, 4]], dtype=th.long,device=device)
    # edge_index = th.tensor([[0],
    #                         [1]], dtype=th.long, device=device)
    # T_edge_index = th.tensor([[1],
    #                         [0]], dtype=th.long,device=device)
    data = Data(x=combined_feature, edge_index=edge_index, T_edge_index=T_edge_index)
    return data