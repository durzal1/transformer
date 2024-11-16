from TransformerBlock import TransformerBlock
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n, k, heads, vocab_size, seq_length, layers):
        super().__init__()
        self.n = n
        self.embedding_size = k
        self.heads = heads
        self.layers = layers
        self.sequence_length = seq_length
        self.vs = vocab_size

        self.embed_meaning = nn.Embedding(vocab_size, self.embedding_size)
        self.embed_pos = nn.Embedding(seq_length, self.embedding_size)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(n,k,heads) for _ in range(layers)])

        self.key_linear = nn.Linear(k,k)
        self.val_linear = nn.Linear(k,k)



    def forward(self, inputs):
        b,t = inputs.shape
        # assuming inputs is of {b,t} so it is already pre-split into batches and t is the number of words
        # each entry in the tensor is an integer representation of the word

        meaning = self.embed_meaning(inputs)
        position_indices = torch.arange(self.sequence_length).unsqueeze(0).repeat(b, 1)  # Shape (5, 100)
        pos = self.embed_pos(position_indices)

        x = meaning + pos



        for block in self.encoder_blocks:
            x = block(x)

        key = self.key_linear(x)
        vals = self.val_linear(x)

        return key,vals


class EncoderBlock(nn.Module):
    def __init__(self, n,k,heads):
        super().__init__()

        self.embedding_size = k

        self.query_linear = nn.Linear(k, k)
        self.key_linear = nn.Linear(k, k)
        self.value_linear = nn.Linear(k, k)

        self.internal = TransformerBlock(n,k,heads)


    def forward(self, x):
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.key_linear(x)

        x = self.internal(q, k, v)

        return x

# x = torch.randint(1,100,(5,100))
# encoder = Encoder(1,512, 6, 10000, 4,6)
# output = encoder(x)
# print(output)