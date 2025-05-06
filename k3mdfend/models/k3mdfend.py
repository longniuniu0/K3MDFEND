from Process.Graph import SCL
from .layers import *
from torch import nn
import torch as th
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel
from utils.process import BertGCN
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
# node_types = ['content','comments', 'expert_comment', 'user_comment', 'reporter_comment',
#         'user_to_expert', 'expert_to_user', 
#         'expert_to_reporter', 'user_to_reporter']
node_types = ['content','comments']
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict

class QualityAwareAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 质量评估模块（所有评论类型共享）
        self.quality_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 注意力计算模块
        self.query_proj = nn.Linear(feature_dim, feature_dim)  # 将content映射为query
        self.key_proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, feature_dim)
            for name in node_types if name != 'content'  # 排除content自身的key投影
        })
        
    def forward(self, content_feat, comment_features):
        """
        输入:
        - content_feat: [B, D] 主内容特征
        - comment_features: {
            'expert_comment': [B, D],
            'user_comment': [B, D],
            ... (其他评论类型)
        }
        """
        quality_scores = {
            name: self.quality_predictor(feat).squeeze(1)  # [B]
            for name, feat in comment_features.items()
        }
        
        q = self.query_proj(content_feat)  # 用content生成query [B, D]
        
        attn_scores = []
        for name, feat in comment_features.items():
            k = self.key_proj[name](feat)  # 各评论节点的key [B, D]
            raw_score = torch.sum(q * k, dim=1)  # [B]
            adjusted_score = raw_score * quality_scores[name]
            attn_scores.append(adjusted_score)
        
        attn_weights = F.softmax(torch.stack(attn_scores, dim=1), dim=1)  # [B, N_comments]
        
        fused = sum(
            attn_weights[:, i].unsqueeze(1) * feat 
            for i, (name, feat) in enumerate(comment_features.items())
        )
        
        return fused, quality_scores


class KairosModel(nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, dataset):
        super().__init__()
        # 初始化BERT和CNN
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
        
        # 解冻最后层
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        
        # 特征提取
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64}
        self.convs1 = cnn_extractor(feature_kernel, emb_dim)
        
        # 改进的注意力融合
        self.fusion = QualityAwareAttentionFusion(256)
        
        # 分类头
        self.cla_head = nn.Sequential(
            MLP(256, mlp_dims, dropout, False),
            nn.Linear(mlp_dims[-1], 2))
        self.classifier = nn.Linear(2, 1)

        self.dropout = nn.Dropout(p=0.5)
        self.scl = SCL(temperature=0.1)

    def forward(self, data, flag):
        lossfun = nn.BCELoss()
        if flag == 'train':
            label_1 = data['label'].float()
            domain_label = data['category'].float()
            # 特征提取
            all_features = {
                name: self.convs1(self.bert(data[name],attention_mask=data[name+'_masks']).last_hidden_state)
                for name in node_types
            }
            # 分离主内容和评论
            content_feat = all_features['content']  # [B, D]
            comment_features = {k:v for k,v in all_features.items() if k != 'content'}
            # 质量感知融合
            fused_comments, quality_scores = self.fusion(content_feat, comment_features)
            # 残差连接分类特征
            final_feature = content_feat + fused_comments  # [B, D]
            # 对抗目标：让融合评论逼近主内容
            adv_loss = (F.mse_loss(fused_comments, content_feat.detach()) * 0.1)
            #contrastive learning
            fake_scloss = self.scl(final_feature, final_feature, domain_label, None, label_1, None)
            logits = self.cla_head(final_feature)
            output = self.classifier(logits)
            output = th.sigmoid(output.squeeze(1))


            fake_CEloss = lossfun(output, label_1)

            alp = 0.1
            main_loss = (1 - alp) * fake_CEloss

            total_loss = main_loss + adv_loss + 0.07 * fake_scloss
            out = []
            out.append(logits)
            out.append(output)
            out.append(total_loss)
            out.append(final_feature)
            return out
        else:
            # 特征提取
            all_features = {
                name: self.convs1(self.bert(data[name],attention_mask=data[name+'_masks']).last_hidden_state)
                for name in node_types
            }
            # 分离主内容和评论
            content_feat = all_features['content']  # [B, D]
            comment_features = {k:v for k,v in all_features.items() if k != 'content'}
            # 质量感知融合
            fused_comments, quality_scores = self.fusion(content_feat, comment_features)
            # 残差连接分类特征
            final_feature = content_feat + fused_comments  # [B, D]
            logits = self.cla_head(final_feature)
            output = self.classifier(logits)
            output = th.sigmoid(output.squeeze(1))
            out = []
            out.append(logits)
            out.append(output)
            out.append(final_feature)
            return out
