import os
import torch
import tqdm
import torch.nn as nn
import datetime
import numpy as np
from models.layers import *
from models.kairos import KairosModel as StudentADModel
from models.bigru import BiGRUModel
from utils.utils import data2gpu, Averager, metrics, Recorder, GridRecorder
from torch.nn import functional as F
import itertools
def euclidean_dist(shared_feature):
    trans=shared_feature.T
    dist_matrix=torch.cdist(trans,trans)
    dist_matrix=dist_matrix.T
    return dist_matrix

def distillation(student_scores,teacher_scores,temp):
    loss_soft=F.kl_div(F.log_softmax(student_scores/temp,dim=1),F.softmax(teacher_scores/temp,dim=1),reduction="batchmean")
    return loss_soft*temp*temp
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
        self.val_loader = val_loader
        self.test_loader = test_loader
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
        self.param_grid = {
            'beta': [0.03, 0.05, 0.07, 0.1],
            'gamma': [0.03, 0.05, 0.07, 0.1]}
    def grid_train(self):
        best_score = 0
        best_params = {}
        param_combinations = list(itertools.product(*self.param_grid.values()))
        recorder = GridRecorder(self.early_stop)
        for params in tqdm.tqdm(param_combinations, desc="Grid Searching"):
            beta, gamma = params
            print("params training:", params)
            # 初始化带当前参数的模型
            self.model = StudentADModel(
                self.emb_dim, self.mlp_dims, self.dropout, 
                self.dataset, alp=1-beta-gamma, beta=beta, gamma=gamma
            ).cuda() 
            
            # 执行训练流程
            results = self.train(1-beta-gamma, beta, gamma)
            file_path = "grid_en.txt"
            with open(file_path, "a") as file:  
                file.write(str(params) + '\n')
                file.write(str(results) + '\n')
            
            mark = recorder.add(results)
            if mark == 'save':
                best_params = params
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter' +'StudentKDfrom_'+
                                        'K3MDFEND_en'+'_'+str(beta)+str(gamma)+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl'))
        print(f"Best Params: {best_params}")
        beta, gamma = best_params
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter'+'StudentKDfrom_'+
                                            'K3MDFEND_en'+'_'+str(beta)+str(gamma)+
                                            str(self.early_stop)+'_'+
                                            str(self.logits_shape)+'_'+
                                            str(self.usemul)+self.dataset+
                                            '.pkl')))
        results = self.test(self.test_loader, 1, 1-beta-gamma, beta, gamma)
        print(results)
        file_path = "K3MDFEND_en.txt"  # 文件名和路径
        print('The opotimized resuts:')
        with open(file_path, "w")as file:  # 打开文件，'w' 表示写入模式
            file.write(str(params) + '\n')
            file.write(str(results))
        return results, os.path.join(self.save_param_dir, 'parameter' +'StudentKDfrom_'+
                                        'K3MDFEND_en'+'_'+str(beta)+str(gamma)+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl')

    def train(self, alp, beta, gamma):
        # print('modelname',self.modelname1,self.modelname2)
        # if self.modelname1 == 'mdfend':
        #     self.teacher0=MDFENDModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset,logits_shape=self.logits_shape)
        # elif self.modelname1 == 'm3fend':
        #     self.teacher0 = M3FENDModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
        #                            self.style_num, self.lnn_dim, len(self.category_dict), dataset=self.dataset,logits_shape=self.logits_shape)
        # self.model =StudentADModel(self.emb_dim, self.mlp_dims, self.dropout, dataset=self.dataset)
        

        if self.use_cuda:
            self.model = self.model.cuda()
        lossfun = torch.nn.BCELoss()
        loss_fn2=torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        f1_me=[0,0]
        fd_me=[0,0]
        m=self.Momentum
        a=0.4
        best_results = []
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch_data in enumerate(train_data_iter):
                batch_data = data2gpu(batch_data, self.use_cuda)
                flag = 'train'
                optimizer.zero_grad()
                out = self.model(batch_data)
                loss = out[2]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
                if scheduler is not None:
                    scheduler.step()
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results = self.test(self.val_loader, 0, 1 - beta-gamma, beta, gamma)
            mark = recorder.add(results)
            if mark == 'save':
                best_results = results
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter' +'StudentKDfrom_'+
                                        'K3MDFEND_en'+'_'+str(beta)+str(gamma)+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        return results

    def test(self, dataloader, testorval, alp, beta, gamma):
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
        #plot(shared_feature,category)
        result = metrics(label, pred, category, self.category_dict)
        if testorval == 1:
            torch.save(self.model,
                       'recodertestpkl/' + 'K3MDFEND_en' + str(beta)+str(gamma)+ self.dataset + '.pkl')

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
                feature = out[2]
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