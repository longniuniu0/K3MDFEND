import torch
import random
import pandas as pd
import tqdm
import numpy as np
import pickle
import re
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader

def _init_fn(worker_id):
    np.random.seed(2021)

def read_data(path):
    if path.endswith('csv'):
        t=pd.read_csv(path)
    if path.endswith('pkl'):
        t=pd.read_pickle(path)
    if path.endswith('json'):
        t = pd.read_json(path)
    return t

def df_filter(df_data, category_dict):
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data

#process context
def word2input(texts, max_len, dataset):
    if dataset == 'ch1':
        tokenizer = BertTokenizer.from_pretrained("./pretrained_model/chinese-bert-wwm-ext/")
    elif dataset == 'en':
        tokenizer = RobertaTokenizer.from_pretrained('./pretrained_model/roberta-base/')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def process(x):
    x['content_emotion'] = x['content_emotion'].astype(float)
    return x
    

class bert_data():
    def __init__(self, max_len, batch_size, category_dict, dataset, domain_num, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset = dataset
        self.domain_num = domain_num
    
    def load_data(self, path, shuffle,drop_last=True):
        print(self.category_dict)
        self.data = df_filter(read_data(path), self.category_dict)
        with open('case.txt', 'a', encoding='utf-8') as f:
            for idx, row in self.data.iterrows():
                content = row['content']
                category = row['category']
                label = row['label']
                # 写入格式：内容 + 制表符 + 标签 + 换行
                f.write(f"{content}\t{label}\t{category}\n")
        content = self.data['content'].to_numpy()
        
        comments = self.data['comments'].to_numpy()
        content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
        comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        categoryonehot = torch.nn.functional.one_hot(category, num_classes=self.domain_num)
        content_token_ids, content_masks = word2input(content, self.max_len, self.dataset)

        expert_comment = self.data['expert'].to_numpy()
        user_comment = self.data['user'].to_numpy()
        reporter_comment = self.data['reporter'].to_numpy()
        expert_to_user = self.data['expert_to_user'].to_numpy()
        expert_to_reporter = self.data['expert_to_reporter'].to_numpy()
        user_to_expert = self.data['user_to_expert'].to_numpy()
        user_to_reporter = self.data['user_to_reporter'].to_numpy()
        node_num = content_token_ids.size(0)
        print('The LLM data\'s num is:',node_num)
        comments_token_ids, comments_masks = word2input(comments, self.max_len, self.dataset)
        expert_comment, expert_comment_masks = word2input(expert_comment, self.max_len, self.dataset)
        user_comment, user_comment_masks = word2input(user_comment, self.max_len, self.dataset)
        reporter_comment, reporter_comment_masks = word2input(reporter_comment, self.max_len, self.dataset)
        expert_to_user, expert_to_user_comment_masks = word2input(expert_to_user, self.max_len, self.dataset)
        expert_to_reporter,expert_to_reporter_comment_masks = word2input(expert_to_reporter, self.max_len, self.dataset)
        user_to_expert,user_to_expert_comment_masks = word2input(user_to_expert, self.max_len, self.dataset)
        user_to_reporter,user_to_reporter_comment_masks = word2input(user_to_reporter, self.max_len, self.dataset)
        dataset = TensorDataset(content_token_ids,content_masks,comments_token_ids,
                                comments_masks,content_emotion,comments_emotion,emotion_gap,
                                style_feature,label,category,categoryonehot,expert_comment,
                                expert_comment_masks,user_comment,user_comment_masks,reporter_comment,
                                reporter_comment_masks,expert_to_user,expert_to_user_comment_masks,
                                expert_to_reporter,expert_to_reporter_comment_masks,
                                user_to_expert,user_to_expert_comment_masks,
                                user_to_reporter,user_to_reporter_comment_masks
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader
# import torch
# import pandas as pd
# import numpy as np
# import pickle
# from transformers import BertTokenizer
# from transformers import RobertaTokenizer
# import torch as th
# from torch.utils.data import TensorDataset, DataLoader
# device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
# def _init_fn(worker_id=2021):
#     np.random.seed(worker_id)
# def _init_fn_(worker_id=6):
#     np.random.seed(worker_id)

# def read_data(path):
#     # 用于存储所有提取的数据
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     return data


# def df_filter(df_data, category_dict):
#     df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
#     return df_data


# # def df_filter(df_data, category_dict):
# #     df_data['category'] = 0
# #     return df_data

# # process context

# def word2inputs(texts, max_len, dataset):
#     if dataset == 'ch1':
#         tokenizer = BertTokenizer.from_pretrained("./pretrained_model/chinese-bert-wwm-ext/")
#     elif dataset == 'en':
#         tokenizer = RobertaTokenizer.from_pretrained('./pretrained_model/roberta-base/')
#     token_ids = []
#     for i, text in enumerate(texts):
#         token_ids.append(
#             tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
#                              truncation=True))
#     token_ids = torch.tensor(token_ids)
#     masks = torch.zeros(token_ids.shape)
#     mask_token_id = tokenizer.pad_token_id
#     for i, tokens in enumerate(token_ids):
#         masks[i] = (tokens != mask_token_id)
#     return token_ids, masks

# def process(x):
#     x['content_emotion'] = x['content_emotion'].astype(float)
#     return x

# # def Graph_Create(content, comments, expert, user, reporter, user_to_expert, expert_to_user, expert_to_reporter, user_to_reporter,label):
# #     node_features = [
# #         content.unsqueeze(0),    
# #         comments.unsqueeze(0),  
# #         expert.unsqueeze(0),     
# #         user.unsqueeze(0),      
# #         reporter.unsqueeze(0),
# #         user_to_expert.unsqueeze(0),
# #         expert_to_user.unsqueeze(0),
# #         expert_to_reporter.unsqueeze(0),
# #         user_to_reporter.unsqueeze(0)
# #     ]
# #     x = torch.cat(node_features, dim=0)

# #     edge_index = th.tensor([[0, 0, 0, 0, 2, 3, 4, 4],
# #                             [1, 2, 3, 4, 5, 6, 7, 8]], dtype=th.long, device=device)
# #     T_edge_index = th.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
# #                             [0, 0, 0, 0, 2, 3, 4, 4]], dtype=th.long,device=device)
# #     data = Data(x=x, edge_index=edge_index, T_edge_index=T_edge_index, label=label)
# #     return data

# class bert_data():
#     def __init__(self, max_len, batch_size, dataset,num_workers):
#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.dataset = dataset
#         self.category_dict = {}
#         self.domain_num = 0
#     def load_data(self, path, shuffle):
#         # self.category_dict = {
#         #     "科技": 0,
#         #     "军事": 1,
#         #     "教育考试": 2,
#         #     "灾难事故": 3,
#         #     "政治": 4,
#         #     "医药健康": 5,
#         #     "财经商业": 6,
#         #     "文体娱乐": 7,
#         #     "社会生活": 8,
#         # }
#         # self.domain_num = 9
#         self.category_dict = {
#             "gossipcop": 0,
#             "politifact": 1,
#             "COVID": 2,
#         }
#         self.domain_num = 3
#         self.data = df_filter(read_data(path), self.category_dict)
#         content = self.data['content'].to_numpy()
#         comments = self.data['comments'].to_numpy()
#         content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
#         comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
#         emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
#         style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
#         label = torch.tensor(self.data['label'].astype(int).to_numpy())
#         category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
#         categoryonehot = torch.nn.functional.one_hot(category, num_classes=self.domain_num)
#         content_token_ids, content_masks = word2inputs(content, self.max_len, self.dataset)

#         # expert_comment = self.data['expert'].to_numpy()
#         # user_comment = self.data['user'].to_numpy()
#         # reporter_comment = self.data['reporter'].to_numpy()
#         # expert_to_user = self.data['expert_to_user'].to_numpy()
#         # expert_to_reporter = self.data['expert_to_reporter'].to_numpy()
#         # user_to_expert = self.data['user_to_expert'].to_numpy()
#         # user_to_reporter = self.data['user_to_reporter'].to_numpy()
#         expert_comment = content
#         user_comment = content
#         reporter_comment = content
#         expert_to_user = content
#         expert_to_reporter = content
#         user_to_expert = content
#         user_to_reporter = content
#         node_num = content_token_ids.size(0)
#         print('The LLM data\'s num is:',node_num)
#         comments_token_ids, comments_masks = word2inputs(comments, self.max_len, self.dataset)
#         expert_comment, expert_comment_masks = word2inputs(expert_comment, self.max_len, self.dataset)
#         user_comment, user_comment_masks = word2inputs(user_comment, self.max_len, self.dataset)
#         reporter_comment, reporter_comment_masks = word2inputs(reporter_comment, self.max_len, self.dataset)
#         expert_to_user, expert_to_user_comment_masks = word2inputs(expert_to_user, self.max_len, self.dataset)
#         expert_to_reporter,expert_to_reporter_comment_masks = word2inputs(expert_to_reporter, self.max_len, self.dataset)
#         user_to_expert,user_to_expert_comment_masks = word2inputs(user_to_expert, self.max_len, self.dataset)
#         user_to_reporter,user_to_reporter_comment_masks = word2inputs(user_to_reporter, self.max_len, self.dataset)
#         dataset = TensorDataset(content_token_ids,content_masks,comments_token_ids,
#                                 comments_masks,content_emotion,comments_emotion,emotion_gap,
#                                 style_feature,label,category,categoryonehot,expert_comment,
#                                 expert_comment_masks,user_comment,user_comment_masks,reporter_comment,
#                                 reporter_comment_masks,expert_to_user,expert_to_user_comment_masks,
#                                 expert_to_reporter,expert_to_reporter_comment_masks,
#                                 user_to_expert,user_to_expert_comment_masks,
#                                 user_to_reporter,user_to_reporter_comment_masks
#                                 )
#         
        # dataloader = DataLoader(
        #     dataset=dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=shuffle,
        #     worker_init_fn=_init_fn
        # )
        
        # return dataloader