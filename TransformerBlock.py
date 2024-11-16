from SelfAttention import SelfAttention
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n, k, heads):
        super().__init__()
        self.embedding_size = k
        self.heads = heads

        self.attention = SelfAttention(k,heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.fc = nn.Sequential(
            nn.Linear(k,n*k),
            nn.ReLU(),
            nn.Linear(n*k,k)
        )
        self.dropout = nn.Dropout()

    def forward(self, queries, keys, vals):

        x = queries + keys + vals
        y = self.attention(queries,keys,vals,None)

        y = self.norm1(x+queries)
        y = self.fc(y)
        y = self.norm2(x+y)
        return y