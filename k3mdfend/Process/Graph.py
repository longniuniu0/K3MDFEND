import torch.nn as nn
import random
from models.layers import *
from Process.contrastive import AvgReadout, Discriminator, Encoder
import torch as th
import torch.nn.functional as F
import numpy as np

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class SCL(th.nn.Module):
    def __init__(self, temperature=0.1, total_epochs=50, ema_decay=0.9):
        super(SCL, self).__init__()
        self.temperature = temperature
        self.total_epochs = total_epochs
        self.ema_decay = ema_decay
        self.dynamic_threshold = DynamicThreshold(total_epochs)
        self.current_epoch = 0  # 需要从外部更新
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    def forward(self, inrep_1, inrep_2, domain_labels_1, domain_labels_2, label_1, label_2=None):
        bs_1 = int(inrep_1.shape[0])  # 获取批次大小
        bs_2 = int(inrep_2.shape[0])

        if label_2 == None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)  # 输入特征向量归一化处理/-
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = th.diag(cosine_similarity)  # 对角线元素
            cos_diag = th.diag_embed(diag)  # bs,bs

            label = th.unsqueeze(label_1, -1)
            domain_label = th.unsqueeze(domain_labels_1,-1)

            # 标签矩阵
            if label.shape[0] == 1:
                cos_loss = th.zeros(1)
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = th.cat((label, label), -1)
                    else:
                        label_mat = th.cat((label_mat, label), -1)  # bs, bs
                domain_mask = (domain_labels_1.unsqueeze(1) == domain_labels_1.unsqueeze(0)).float()
                mid_mat_ = (label_mat.eq(label_mat.t()))
                mid_mat = mid_mat_.float()  # 标签矩阵的相等性矩阵
                cosine_similarity = (cosine_similarity - cos_diag)
                cosine_similarity[(domain_mask == 0) & (cosine_similarity > 0)] = cosine_similarity[(domain_mask == 0) & (cosine_similarity > 0)] / 3
                cosine_similarity[(domain_mask == 0) & (cosine_similarity < 0)] = cosine_similarity[(domain_mask == 0) & (cosine_similarity < 0)] / 3
                
                cosine_similarity = cosine_similarity / self.temperature  # the diag is 0计算对比学习损失
                mid_diag = th.diag_embed(th.diag(mid_mat))
                mid_mat = mid_mat - mid_diag
                
                
                # 削弱跨域低相似度样本
                # con_cosine_similarity = torch.where(
                #     cross_domain_mask & pos_domain_mask,
                #     torch.sqrt(cosine_similarity+1e-6),
                #     cosine_similarity  # 其他样本保持不变
                # )
                # cosine_similarity[(domain_mask == 0) & (cosine_similarity > 0)] = th.pow(cosine_similarity[(domain_mask == 0) & (cosine_similarity > 0)],1/4)
                # cosine_similarity[(domain_mask == 0) & (cosine_similarity < 0)] = cosine_similarity[(domain_mask == 0) & (cosine_similarity < 0)] * 2
                #con_cosine_similarity = th.where((domain_mask == 0) & (cosine_similarity < ret_std), th.sqrt(cosine_similarity), cosine_similarity)
                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.bool(), -float(
                    'inf'))  # mask the diag将对角线元素掩盖为无穷大 确保softmax中对角线元素不会产生影响

                cos_loss = th.log(
                    th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1
                # softmax计算每个样本和其他样本的相似度
                cos_loss = cos_loss * mid_mat

                cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)  # bs
                # 与标签矩阵相乘，确保只计算相同标签样本之间的相似度损失，并通过求和和归一化得到最终的对比学习损失。
        else:  # 同训练集批次
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    domain_labels_2 = domain_labels_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    do_lbl_pad = domain_labels_2[ra].unsqueeze(0)
                    inrep_2 = th.cat((inrep_2, pad), 0)
                    label_2 = th.cat((label_2, lbl_pad), 0)
                    domain_labels_2 = th.cat((domain_labels_2, do_lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = th.unsqueeze(label_1, -1)
            label_1_mat = th.cat((label_1, label_1), -1)
            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = label_1_mat
                else:
                    label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = th.unsqueeze(label_2, -1)
            label_2_mat = th.cat((label_2, label_2), -1)
            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = label_2_mat
                else:
                    label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()  # 两组数据相等标签矩阵
            cosine_similarity = cosine_similarity / self.temperature
            domain_mask = (domain_labels_1.unsqueeze(1) == domain_labels_2.unsqueeze(0)).float()
            threshold = self.dynamic_threshold.compute_threshold(
                self.current_epoch, 
                cosine_similarity, 
                domain_mask
            )
            
            # 改进的跨域相似度调整
            cross_domain_mask = (domain_mask == 0)  # 跨域样本对
            pos_domain_mask = (cosine_similarity > 0)
            high_sim_mask = (cosine_similarity > threshold)

            # cosine_similarity[(domain_mask == 0) & (cosine_similarity > 0)] = th.pow(cosine_similarity[(domain_mask == 0) & (cosine_similarity > 0)],1/4)
            # cosine_similarity[(domain_mask == 0) & (cosine_similarity < 0)] = cosine_similarity[(domain_mask == 0) & (cosine_similarity < 0)] * 2
            #con_cosine_similarity = th.where((domain_mask == 0) & (cosine_similarity < ret_std), th.sqrt(cosine_similarity), cosine_similarity)
            #con_cosine_similarity = th.where((domain_mask == 1)&(cosine_similarity > ret_std), (cosine_similarity + ret_std), con_cosine_similarity)
            cosine_similarity = cosine_similarity * domain_mask
            cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat  # find the sample with the same label
            cos_loss = th.sum(cos_loss, dim=1) / th.sum(mid_mat + 1e-10, dim=1)  # bs

        cos_loss = -th.mean(cos_loss, dim=0)
        return cos_loss

#得到%的值
def compute_adaptive_threshold(cosine_sim, domain_mask, quantile):
    # 仅保留跨域样本对的相似度（domain_mask=0的位置）
    cross_domain_sim = cosine_sim[domain_mask == 0]
    pos_domain_sim = cross_domain_sim[cross_domain_sim > 0]
    # 处理无跨域样本的情况
    if pos_domain_sim.numel() == 0:
        return torch.tensor(0.0, device=cosine_sim.device)
    
    # 确保分位数在合理范围内
    valid_quantile = max(min(quantile, 1.0), 0.0)
    return torch.quantile(pos_domain_sim, valid_quantile)

class DynamicThreshold:
    def __init__(self, total_epochs, base_quantile=0.6):  # 提高base_quantile
        self.total_epochs = max(total_epochs, 1)
        self.base_quantile = base_quantile  
    
    def compute_threshold(self, epoch, cosine_sim, domain_mask):
        progress = min(epoch / self.total_epochs, 1.0)
        current_quantile = self.base_quantile * (1 + progress)  
        current_quantile = max(current_quantile, 0.6)
        current_threshold = compute_adaptive_threshold(
            cosine_sim, domain_mask, quantile=current_quantile
        )
        return current_threshold
class GlobalContrastiveLearning(th.nn.Module):
    def __init__(self, n_h, act):
        super(GlobalContrastiveLearning, self).__init__()
        self.read = AvgReadout()
        self.sigm = th.nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.pre = nn.PReLU() if act == 'prelu' else act
        self.fc = nn.Linear(256, 128, bias=False)

    def forward(self, raw_view, aug1_view):
        raw_view.to(device)
        aug1_view.to(device)
        nb_nodes = raw_view.shape[0]
        msk, samp_bias1, samp_bias2 = None, None, None
        lbl_1 = th.ones(1, nb_nodes)
        lbl_2 = th.zeros(1, nb_nodes)
        lbl = th.cat((lbl_1, lbl_2), 1)
        if th.cuda.is_available():
            raw_view = raw_view.to(device)
            lbl = lbl.to(device)
        h_1 = self.fc(raw_view)
        h_1 = self.pre(h_1)
        c = self.read(h_1, msk)
        c = self.sigm(c)#激活
        h_2 = self.fc(aug1_view)
        h_2 = self.pre(h_2) #
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        b_xent = th.nn.BCEWithLogitsLoss()
        loss = b_xent(ret, lbl)
        return loss

# import torch.nn as nn
# import random
# from models.layers import *
# from Process.contrastive import AvgReadout, Discriminator, Encoder
# import torch as th
# import torch.nn.functional as F
# import numpy as np

# device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

# class SCL(th.nn.Module):
#     def __init__(self, temperature=0.1, total_epochs=50, ema_decay=0.9):
#         super(SCL, self).__init__()
#         self.temperature = temperature
#         self.total_epochs = total_epochs
#         self.ema_decay = ema_decay
#         self.dynamic_threshold = DynamicThreshold(total_epochs)
#         self.current_epoch = 0  # 需要从外部更新
#     def set_epoch(self, epoch):
#         self.current_epoch = epoch
#     def forward(self, inrep_1, inrep_2, domain_labels_1, domain_labels_2, label_1, label_2=None):
#         bs_1 = int(inrep_1.shape[0])  # 获取批次大小
#         bs_2 = int(inrep_2.shape[0])

#         if label_2 == None:
#             normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
#             normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)  # 输入特征向量归一化处理/-
#             cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

#             diag = th.diag(cosine_similarity)  # 对角线元素
#             cos_diag = th.diag_embed(diag)  # bs,bs

#             label = th.unsqueeze(label_1, -1)
#             domain_label = th.unsqueeze(domain_labels_1,-1)

#             # 标签矩阵
#             if label.shape[0] == 1:
#                 cos_loss = th.zeros(1)
#             else:
#                 for i in range(label.shape[0] - 1):
#                     if i == 0:
#                         label_mat = th.cat((label, label), -1)
#                     else:
#                         label_mat = th.cat((label_mat, label), -1)  # bs, bs

#                 mid_mat_ = (label_mat.eq(label_mat.t()))
#                 mid_mat = mid_mat_.float()  # 标签矩阵的相等性矩阵

#                 cosine_similarity = (cosine_similarity - cos_diag) / self.temperature  # the diag is 0计算对比学习损失
#                 print(torch.max(cosine_similarity))
#                 mid_diag = th.diag_embed(th.diag(mid_mat))
#                 mid_mat = mid_mat - mid_diag

#                 domain_mask = (domain_labels_1.unsqueeze(1) == domain_labels_1.unsqueeze(0)).float()
#                 threshold = self.dynamic_threshold.compute_threshold(
#                     self.current_epoch, 
#                     cosine_similarity, 
#                     domain_mask
#                 )
                
#                 # 改进的跨域相似度调整
#                 cross_domain_mask = (domain_mask == 0)  # 跨域样本对
#                 pos_domain_mask = (cosine_similarity > 0)
#                 high_sim_mask = (cosine_similarity > threshold)
#                 # 削弱跨域低相似度样本
#                 boost_factor=1.5
#                 con_cosine_similarity = torch.where(
#                     cross_domain_mask & high_sim_mask & pos_domain_mask,
#                     torch.clamp(cosine_similarity * boost_factor, max=1.0),  # 确保不超过1
#                     cosine_similarity  # 其他样本保持不变
#                 )
#                 #con_cosine_similarity = th.where((domain_mask == 0) & (cosine_similarity < ret_std), th.sqrt(cosine_similarity), cosine_similarity)
#                 cosine_similarity = con_cosine_similarity.masked_fill_(mid_diag.bool(), -float(
#                     'inf'))  # mask the diag将对角线元素掩盖为无穷大 确保softmax中对角线元素不会产生影响

#                 cos_loss = th.log(
#                     th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1
#                 # softmax计算每个样本和其他样本的相似度
#                 cos_loss = cos_loss * mid_mat

#                 cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)  # bs
#                 # 与标签矩阵相乘，确保只计算相同标签样本之间的相似度损失，并通过求和和归一化得到最终的对比学习损失。
#         else:  # 同训练集批次
#             if bs_1 != bs_2:
#                 while bs_1 < bs_2:
#                     inrep_2 = inrep_2[:bs_1]
#                     label_2 = label_2[:bs_1]
#                     domain_labels_2 = domain_labels_2[:bs_1]
#                     break
#                 while bs_2 < bs_1:
#                     inrep_2_ = inrep_2
#                     ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
#                     pad = inrep_2_[ra].unsqueeze(0)
#                     lbl_pad = label_2[ra].unsqueeze(0)
#                     do_lbl_pad = domain_labels_2[ra].unsqueeze(0)
#                     inrep_2 = th.cat((inrep_2, pad), 0)
#                     label_2 = th.cat((label_2, lbl_pad), 0)
#                     domain_labels_2 = th.cat((domain_labels_2, do_lbl_pad), 0)
#                     bs_2 = int(inrep_2.shape[0])

#             normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
#             normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
#             cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

#             label_1 = th.unsqueeze(label_1, -1)
#             label_1_mat = th.cat((label_1, label_1), -1)
#             for i in range(label_1.shape[0] - 1):
#                 if i == 0:
#                     label_1_mat = label_1_mat
#                 else:
#                     label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

#             label_2 = th.unsqueeze(label_2, -1)
#             label_2_mat = th.cat((label_2, label_2), -1)
#             for i in range(label_2.shape[0] - 1):
#                 if i == 0:
#                     label_2_mat = label_2_mat
#                 else:
#                     label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

#             mid_mat_ = (label_1_mat.t().eq(label_2_mat))
#             mid_mat = mid_mat_.float()  # 两组数据相等标签矩阵
#             cosine_similarity = cosine_similarity / self.temperature
#             print(torch.max(cosine_similarity))
#             domain_mask = (domain_labels_1.unsqueeze(1) == domain_labels_2.unsqueeze(0)).float()
#             threshold = self.dynamic_threshold.compute_threshold(
#                 self.current_epoch, 
#                 cosine_similarity, 
#                 domain_mask
#             )
            
#             # 改进的跨域相似度调整
#             cross_domain_mask = (domain_mask == 0)  # 跨域样本对
#             pos_domain_mask = (cosine_similarity > 0)
#             high_sim_mask = (cosine_similarity > threshold)
#             # 削弱跨域低相似度样本
#             boost_factor=1.5
#             boost_factor = boost_factor - 0.5 * (self.current_epoch / self.total_epochs)
#             con_cosine_similarity = torch.where(
#                 cross_domain_mask & high_sim_mask & pos_domain_mask,
#                 torch.clamp(cosine_similarity * boost_factor, max=1.0),  # 确保不超过1
#                 cosine_similarity  # 其他样本保持不变
#             )
#             #con_cosine_similarity = th.where((domain_mask == 0) & (cosine_similarity < ret_std), th.sqrt(cosine_similarity), cosine_similarity)
#             #con_cosine_similarity = th.where((domain_mask == 1)&(cosine_similarity > ret_std), (cosine_similarity + ret_std), con_cosine_similarity)
#             cos_loss = th.log(th.clamp(F.softmax(con_cosine_similarity, dim=1), 1e-10, 1e10))
#             cos_loss = cos_loss * mid_mat  # find the sample with the same label
#             cos_loss = th.sum(cos_loss, dim=1) / th.sum(mid_mat + 1e-10, dim=1)  # bs

#         cos_loss = -th.mean(cos_loss, dim=0)
#         return cos_loss

# #得到%的值
# def compute_adaptive_threshold(cosine_sim, domain_mask, quantile):
#     # 仅保留跨域样本对的相似度（domain_mask=0的位置）
#     cross_domain_sim = cosine_sim[domain_mask == 0]
#     pos_domain_sim = cross_domain_sim[cross_domain_sim > 0]
#     # 处理无跨域样本的情况
#     if pos_domain_sim.numel() == 0:
#         return torch.tensor(0.0, device=cosine_sim.device)
    
#     # 确保分位数在合理范围内
#     valid_quantile = max(min(quantile, 1.0), 0.0)
#     return torch.quantile(pos_domain_sim, valid_quantile)

# class DynamicThreshold:
#     def __init__(self, total_epochs, base_quantile=0.4):  # 提高base_quantile
#         self.total_epochs = max(total_epochs, 1)
#         self.base_quantile = base_quantile  
    
#     def compute_threshold(self, epoch, cosine_sim, domain_mask):
#         progress = min(epoch / self.total_epochs, 1.0)
#         current_quantile = self.base_quantile * (1 + progress)  
#         current_quantile = max(current_quantile, 0.4)
#         current_threshold = compute_adaptive_threshold(
#             cosine_sim, domain_mask, quantile=current_quantile
#         )
#         return current_threshold