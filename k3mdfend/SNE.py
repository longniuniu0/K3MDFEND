import os
import torch
import tqdm
import torch.nn as nn
import datetime
import numpy as np
from models.layers import *
from models.mdfend import MultiDomainFENDModel as MDFENDModel
from models.bigru import BiGRUModel
from models.bert import BertFNModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from models.kairos import KairosModel as K3MDFENDModel
#from models.student_ad import KairosModel as StudentModel
from models.student import StudentModel
from utils.utils import plot
from torch.nn import functional as F

class Trainer():
    def __init__(self,
                 modelname1,
                 modelname2,
                 studentname,
                 emb_dim,
                 mlp_dims,
                 usemul,
                 logits_shape,
                 use_cuda,
                 dataset,
                 lr,
                 dropout,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 early_stop,
                 epoches,
                 train_loader,
                 val_loader,
                 test_loader,
                 path1,
                 path2,
                 Momentum=0.99,
                 ):
        self.modelname1=modelname1
        self.modelname2=modelname2
        self.studentname=studentname
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.path1=path1
        self.path2=path2
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.usemul = usemul
        self.logits_shape = logits_shape

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset
        self.Momentum=Momentum
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
    def train(self):
        print('modelname',self.modelname1,self.modelname2)
        # if self.modelname1 == 'mdfend':
        #     self.teacher0=MDFENDModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset,logits_shape=self.logits_shape)
        # elif self.modelname1 == 'm3fend':
        #     self.teacher0 = M3FENDModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
        #                            self.style_num, self.lnn_dim, len(self.category_dict), dataset=self.dataset,logits_shape=self.logits_shape)
        # self.model = KairosModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, dataset=self.dataset, logits_shape=self.logits_shape)
        self.model = K3MDFENDModel(self.emb_dim, self.mlp_dims, self.dropout, dataset=self.dataset, alp=0.9, beta=0.03, gamma=0.07)

        if self.use_cuda:
            self.model = self.model.cuda()
        # for epoch in range(self.epoches):
        #     self.model.train()
        #     train_data_iter = tqdm.tqdm(self.graph_loader)
        #     avg_loss = Averager()
        #     for step_np, batch in enumerate(train_data_iter):
        #         flag = 'train'
        #         data = data2gpu(batch, self.use_cuda)
        #         optimizer.zero_grad()
        #         out = self.model(data, flag)
        #         loss = out[2]
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         avg_loss.add(loss.item())
        #         if scheduler is not None:
        #             scheduler.step()
        #     print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
        #     results = self.test(self.val_loader, 0)
        #     mark = recorder.add(results)
        #     if mark == 'save':
        #         torch.save(self.model.state_dict(),
        #                    os.path.join(self.save_param_dir+'/k3MDFEND/', 'parameter' +'StudentKDfrom_'+
        #                                 'k3MDFEND_test3'+'_'+
        #                                 str(self.early_stop)+'_'+
        #                                 str(self.logits_shape)+'_'+
        #                                 str(self.usemul)+self.dataset+
        #                                 '.pkl'))
        #     elif mark == 'esc':
        #         break
        #     else:
        #         continue
        self.model = torch.load(self.path2)
        results = self.test(self.test_loader, 1)
        print(results)
        # file_path = "k3MDFEND_test3.txt"  # 文件名和路径
        # with open(file_path, "w") as file:  # 打开文件，'w' 表示写入模式
        #     file.write(str(results))
        # return results, os.path.join(self.save_param_dir+'/k3MDFEND/', 'parameter' +'StudentKDfrom_'+
        #                                 'k3MDFEND_test3'+'_'+
        #                                 str(self.early_stop)+'_'+
        #                                 str(self.logits_shape)+'_'+
        #                                 str(self.usemul)+self.dataset+
        #                                 '.pkl')

    def test(self, dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.model.eval()
        if testorval==0:
            flag='test-time'
        else:
            flag='test'
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.model(batch_data)
                batch_label_pred = out[1]
                feature = out[3]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        plot(shared_feature,category)
        result = metrics(label, pred, category, self.category_dict)
        return result

    def testteacher0(self,dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.teacher0.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.teacher0(**batch_data)
                batch_label_pred = out[1]
                feature = out[3]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        resultlog={}
        mainresultlog={}
        result = metrics(label, pred, category, self.category_dict)
        return result
    def testteacher1(self,dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.teacher1.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.teacher1(**batch_data, alpha=-1)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        resultlog={}
        mainresultlog={}
        result = metrics(label, pred, category, self.category_dict)
        return result