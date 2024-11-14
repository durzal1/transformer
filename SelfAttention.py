import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, k, heads):
        super(SelfAttention,self).__init__()
        self.embedding_size = k
        self.heads = heads
        self.head_dim = k // heads
        assert(self.head_dim * heads == k)

        self.queries = nn.Linear(k,k,False)
        self.keys = nn.Linear(k,k,False)
        self.vals = nn.Linear(k,k,False)

        self.fc = nn.Linear(k,k)


    def forward(self, queries, keys, vals):
        batches = queries.shape[0]
        tokens = queries.shape[1]

        # first we're going to run a linear opretation through the 3 vectors
        queries = self.queries(queries)
        keys = self.keys(keys)
        vals = self.vals(vals)

        # we're now going to resize them for the different heads
        queries = queries.reshape(batches, tokens, self.heads, self.head_dim)
        keys = queries.reshape(batches, tokens, self.heads, self.head_dim)
        vals = queries.reshape(batches, tokens, self.heads, self.head_dim)

        # matrix multiplication between queries and keys
        # we want the output vector be to {batches,tokens,head_dim,head_dim} this is our weights matrix

        # we have to resize to {batches*tokens, heads, head_dim}
        queries = queries.reshape(batches*self.heads, tokens, self.head_dim)
        keys = keys.reshape(batches*self.heads, tokens, self.head_dim)
        vals = vals.reshape(batches*self.heads, tokens, self.head_dim)
        # 8 2 4
        x = torch.bmm(queries, keys.transpose(-1,-2))

        x = x / self.head_dim ** .5

        x = torch.softmax(x,dim=-1)

        x = torch.bmm(x,vals) # make vals part of it

        # concat all the heads together

        x = x.reshape(batches,tokens,self.embedding_size)

        x = self.fc(x)

        return x

q = torch.rand(2,16,24)
k = torch.rand(2,16,24)
v = torch.rand(2,16,24)
attention = SelfAttention(24,4)
output = attention(q,k,v)
print(output)