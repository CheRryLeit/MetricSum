import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class TSUM(nn.Module):
    def __init__(self, opts, word2vec=None):
        super(TSUM,self).__init__()
        # 嵌入词向量
        self.word_embedding = nn.Embedding(opts.vocab_size,opts.embedding_dim)
        if word2vec is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(word2vec))
            self.word_embedding.weight.requires_grad = opts.embed_train
        # 句编码得到句向量
        self.se = SentenceEncoder(opts.embedding_dim,opts.hidden_size)
        # 文档编码
        self.de = DocumentEncoder(opts.hidden_size, opts.hidden_size)
        # 对句子分类
        self.fc = nn.Linear(opts.hidden_size,1)


    def forward(self, docs):
        logit = []
        for i in range(len(docs)):
            doc = docs[i]
            d = self.word_embedding(doc)
            d = self.se(d)
            d = self.de(d)
            d = self.fc(d)
            d = d.squeeze(1)
            d = F.sigmoid(d)
            logit.append(d)

        
        return logit

    def loss(self, x, y):
        loss = 0.0
        for i in range(len(x)):
            xx = x[i]
            yy = y[i]
            temp = torch.zeros((len(xx)), device=xx.device)
            temp[yy] = 1.0
            loss += F.binary_cross_entropy(xx, temp)
        return loss


class SentenceEncoder(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(SentenceEncoder,self).__init__()
        self.cnn = nn.Conv1d(in_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.drop = nn.Dropout()

    def forward(self,sentence):
        s = sentence.transpose(1,2)
        s = self.cnn(s)
        s = self.drop(s)
        s = F.relu(s)
        s = F.max_pool1d(s,100)
        s = s.squeeze(2)
        return s


class DocumentEncoder(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(DocumentEncoder,self).__init__()
        self.cnn = nn.Conv1d(in_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.drop = nn.Dropout()

    def forward(self,document):
        document = document.unsqueeze(0)
        s = document.transpose(1,2)
        s = self.cnn(s)
        s = self.drop(s)
        s = F.relu(s)
        s = s.transpose(1,2)
        s = s.squeeze(0)
        return s
