import torch
from torch import nn
import math
#from torch_cluster import knn as tc_knn

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    '''
    q: (bs, num_p, 1, fea_dim)
    k: (bs, num_p, 9, fea_dim)
    v: (bs, num_p, 9, fea_dim)
    '''
    def forward(self, q, k, v):
        '''
        q: (bs, num_p, 1, fea_dim)
        k: (bs, num_p, 9, fea_dim)
        v: (bs, num_p, 9, fea_dim)
        '''
        d = q.shape[-1]
        scores = torch.einsum('ijkl,ijlm->ijkm', q, k.transpose(2, 3)) / math.sqrt(d)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        output = torch.einsum('ijkm,ijmn->ijkn', attention_weights, v)
        return output

class SubAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.num_heads = num_heads

    def transpose_qkv(self, X):
        # input X (batch_size, no. ,num_neighbor, num_hiddens)
        # output (batch_size*num_heads, no. , num_nei, num_hiddens/num_heads)
        bs = X.shape[0]
        num_p = X.shape[1]
        num_neighbor = X.shape[2]

        X = X.reshape(bs, num_p, num_neighbor, self.num_heads, -1) # (batch_size, no. , num_nei, num_heads, num_hiddens / num_heads)
        X = X.permute(0, 3, 1, 2, 4) # (bs, num_heads, num_p, num_nei,  num_hiddens/num_heads)
        output = X.reshape(-1, X.shape[2], X.shape[3], X.shape[4]) # (batch_size*num_heads, no. , num_nei, num_hiddens/num_heads)
        return output

    def transpose_output(self, X):
        '''
        input: (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
        output: (batch_size, no. of queries, num_hiddens)
        '''
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, q, k, v, pos_encoding):
        '''
        q: (bs, num_p, 1, fea_dim)
        k: (bs, num_p, 9, fea_dim)
        v: (bs, num_p, 9, fea_dim)
        '''

        q = self.transpose_qkv(q)
        k = self.transpose_qkv(k)
        v = self.transpose_qkv(v)
        pos_encoding = self.transpose_qkv(pos_encoding)

        scores = self.attention_mlp( q - k + pos_encoding) # (bs,256,9,32)
        attention_weights = nn.functional.softmax(scores, dim=-2)
        v = v + pos_encoding # (bs, 256, 9, 32)
        output = torch.einsum('bijd,bijd->bid', attention_weights, v) # (bs, 256, 32)

        output = self.transpose_output(output)

        return output

