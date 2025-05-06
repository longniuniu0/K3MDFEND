from torch_geometric.data import Data
import torch
import numpy as np
from transformers import BertTokenizer
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
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

class Bidataset(Dataset):
    def __init__(self, data, dataset, max_len, batch_size, category_dict, domain_num, num_workers, shuffle):
        self.data = data
        self.dataset = dataset
        self.max_len = max_len
        self.batch_size = batch_size
        self.category_dict = category_dict
        self.domain_num = domain_num
        self.num_workers = num_workers
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        content = self.data['content'].to_numpy()
        comments = self.data['comments'].to_numpy()
        content_emotion = torch.LongTensor(np.vstack(self.data['content_emotion']).astype('float32'))
        comments_emotion = torch.LongTensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        emotion_gap = torch.LongTensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        style_feature = torch.LongTensor(np.vstack(self.data['style_feature']).astype('float32'))
        label = torch.LongTensor(self.data['label'].astype(int).to_numpy())
        category = torch.LongTensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        categoryonehot = torch.nn.functional.one_hot(category, num_classes=self.domain_num)
        content_token_ids, content_masks = word2input(content, self.max_len, self.dataset)
        comments_token_ids, comments_masks = word2input(comments, self.max_len, self.dataset)
        num_nodes = 64
        num_features = 256
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.LongTensor([[0],[1]])
        dataset = Data(content_token_ids=content_token_ids, content_masks=content_masks, comments_token_ids=comments_token_ids,
                    comments_masks=comments_masks,  content_emotion=content_emotion,comments_emotion=comments_emotion,
                    emotion_gap=emotion_gap, style_feature=style_feature,
                    label=label, category=category, categoryonehot=categoryonehot,
                    x = x, edge_index=edge_index
                    )
        return dataset