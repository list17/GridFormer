import torch
from torch import nn
import math

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, num_heads=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        #print('### queries size:', queries.shape)
        #print('### keys size:', keys.shape)
        #print('### values size:', values.shape)
        #scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        scores = torch.einsum('ijk,ikl->ijl', queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        #print('### at weight size:', self.attention_weights.shape)
        output = torch.bmm(self.dropout(self.attention_weights), values)
        #print('###output', output.shape)
        return output

class MultiHeadAttention(nn.Module):
  def __init__(self, q_size, k_size, v_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
      super().__init__()
      self.num_heads = num_heads
      self.attention = DotProductAttention(dropout, num_heads)
      self.W_q = nn.Linear(q_size, num_hiddens, bias=bias)
      self.W_k = nn.Linear(k_size, num_hiddens, bias=bias)
      self.W_v = nn.Linear(v_size, num_hiddens, bias=bias)
      self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

  def forward(self, q, k, v, valid_lens):
    # transpose to: (batch_size*num_heads, no. , num_hiddens/num_heads)
    Q = self.W_q(q)
    K = self.W_k(k)
    V = self.W_v(v)

    queries = self.transpose_qkv(Q)
    keys = self.transpose_qkv(K)
    values = self.transpose_qkv(V)

    if valid_lens is not None:
      valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

    output = self.attention(queries, keys, values, valid_lens) # (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
    output_concat = self.transpose_output(output) # (batch_size, no. of queries, num_hiddens)
    return self.W_o(output_concat)

  def transpose_qkv(self, X):
    # input X (batch_size, no. , num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1) # (batch_size, no. , num_heads, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3) #(batch_size, num_heads, no. , num_hiddens/num_heads)
    # output (batch_size*num_heads, no. , num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

  def transpose_output(self, X):
    '''
    input: (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
    output: (batch_size, no. of queries, num_hiddens)
    '''
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class DecoderBlock(nn.Module):
    # The i-th block in the transformer decoder without self-attention
    def __init__(self, q_size, k_size, v_size, num_hiddens, ffn_num_input, ffn_num_hiddens, num_heads, dropout): #, i):
        super().__init__()
        #self.i = i
        self.attention = MultiHeadAttention(q_size, k_size, v_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, q, k, v, valid_lens=None):
        X = self.attention(q, k, v, valid_lens)
        Y = self.addnorm1(q, X)
        return self.addnorm2(Y, self.ffn(Y))

class TransformerDecoder(nn.Module):
    # Transformer decoder without self-attention
    def __init__(self, q_size, k_size, v_size, output_size, num_hiddens, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        # self.q_embedding = nn.Embedding(q_size, num_hiddens)
        # self.k_embedding = nn.Embedding(k_size, num_hiddens)
        # self.v_embedding = nn.Embedding(v_size, num_hiddens)
        self.q_embedding = nn.Linear(q_size, num_hiddens)
        self.k_embedding = nn.Linear(k_size, num_hiddens)
        self.v_embedding = nn.Linear(v_size, num_hiddens)

        self.Q_size = num_hiddens
        self.K_size = num_hiddens
        self.V_size = num_hiddens

        #self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), DecoderBlock(
                self.Q_size, self.K_size, self.V_size, num_hiddens, ffn_num_input, ffn_num_hiddens, num_heads, dropout))
        self.dense = nn.Linear(num_hiddens, output_size)

    def forward(self, q, k, v, valid_lens=None):
        #X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        q = self.q_embedding(q)
        k = self.k_embedding(k)
        v = self.v_embedding(v)
        X = q

        for i, blk in enumerate(self.blks):
            X = blk(X, k, v, valid_lens)
        return self.dense(X)