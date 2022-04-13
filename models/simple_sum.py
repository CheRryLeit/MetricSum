import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import Transformer

class sentEncoder(nn.Modeule):
      def __init__(self, in_channels, out_channels, max_len = 100):
        super(CNNEncoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, in_channels))
        self.cnn = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1)
        self.cnn1 = nn.Conv1d(out_channels, out_channels, 3, 1, padding=1)
        self.dropout = nn.Dropout(0.2)
        
      def forward(self, x):
        x += self.pos_embedding[:, :x.size(1),:]
        # x = self.dropout(x)
        x = x.transpose(1,2)
        x = self.cnn(x)
        x = F.relu(x)
        # x = self.dropout(x)
        # x = self.cnn1(x)
        # x = F.relu(x)
        x = torch.max(x, dim=2)[0]
        return x



class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, max_len = 100):
        super(CNNEncoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, in_channels))
        self.cnn = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1)
        self.cnn1 = nn.Conv1d(out_channels, out_channels, 3, 1, padding=1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x += self.pos_embedding[:, :x.size(1),:]
        # x = self.dropout(x)
        x = x.transpose(1,2)
        x = self.cnn(x)
        x = F.relu(x)
        # x = self.dropout(x)
        # x = self.cnn1(x)
        # x = F.relu(x)
        x = torch.max(x, dim=2)[0]
        return x
class DocumentEncoder(nn.Module):
    def __init__(self, max_len, dim, depth, heads, mlp_dim, dropout = 0., emb_dropout = 0., speaker_num=2):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, dim))
        self.fc = nn.Linear(2*dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_cls_token = nn.Identity()

    def forward(self, x):
        orix = x
        x = x.unsqueeze(0) if x.dim()==2 else x
        # x - > batch d_len seq_len or d_len seq_len
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :x.size(1),:]

        # x = self.norm(x)
        # x = self.dropout(x)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 1:])
        x = x.squeeze(0) if orix.dim() == 2 else x
        # c = self.to_cls_token(x[:, 0])
        # x = torch.cat([x, c.unsqueeze(1).expand_as(x)], dim=-1)
        return x

class MetricSum(nn.Module):

    def __init__(self, opts, word2vec=None):
        super(MetricSum, self).__init__()
        self.embedding = nn.Embedding(opts.vocab_size, opts.embedding_dim)
        if word2vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(word2vec))
            self.embedding.weight.requires_grad = opts.embed_train
        self.sentence_encoder = CNNEncoder(opts.embedding_dim, 300)
        self.document_encoder = DocumentEncoder(150, 300, 3, 6, 300*4, dropout=0.5)
        # dim = 300
        # mlp_dim = 2*dim
        self.classifier = nn.Linear(300, 1)
        # self.classifier = nn.Sequential(
        #     # nn.LayerNorm(dim),
        #     nn.Linear(dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(mlp_dim, 1),
        #     # nn.Dropout(dropout)
        # )

    def forward(self, docs):
        batch_len = len(docs)
        # Check size of tensors
        logits = []
        for i in range(batch_len):
            s = self.embedding(docs[i])
            s = self.sentence_encoder(s)
            s = self.document_encoder(s)
            # s = torch.tanh(s)
            s = self.classifier(s).squeeze(1)
            s = s.sigmoid()
            logits.append(s)
        return logits
    def loss(self, x, y):
        loss = 0.0
        total = len(x)
        for i in range(total):
            xx = x[i]
            yy = y[i]
            temp = torch.zeros((len(xx)), device=xx.device)
            temp[yy] = 1.0
            loss += F.binary_cross_entropy(xx, temp)
        return loss / total

