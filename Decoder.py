from Encoder import Encoder
from SelfAttention import SelfAttention
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n, k, heads, vocab_size, seq_length, layers):
        super().__init__()

        self. n = n
        self.embedding_size = k
        self.heads = heads
        self.layers = layers
        self.seq_length = seq_length

        self.seq_embed = nn.Embedding(vocab_size, k)
        self.pos_mbed = nn.Embedding(seq_length, k)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(n,k,vocab_size,seq_length,heads,layers) for _ in range(layers)])



    def forward(self,outputs, enc_key, enc_val):
        # outputs of {b,t} note that this t can be diff than t for inputs
        b = outputs.shape[0]

        seq = self.seq_embed(outputs)
        position_indices = torch.arange(self.seq_length).unsqueeze(0).repeat(b, 1)  # Shape (5, 100)
        pos = self.pos_mbed(position_indices)

        x = seq + pos

        for block in self.decoder_blocks:
            x = block(x,enc_key,enc_val)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, n, k, vocab_size, seq_length, heads, layers):
        super().__init__()

        self. n = n
        self.embedding_size = k
        self.heads = heads
        self.layers = layers
        self.seq_length = seq_length
        self.sequence_length = seq_length

        self.masked_attention = SelfAttention(k,heads)
        self.attention = SelfAttention(k,heads)
        self.fc = nn.Sequential(
            nn.Linear(k, n*k),
            nn.ReLU(),
            nn.Linear(n*k,k)
        )

        self.query_linear = nn.Linear(k, k)
        self.key_linear = nn.Linear(k, k)
        self.value_linear = nn.Linear(k, k)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.norm3 = nn.LayerNorm(k)

    def forward(self,x,enc_key,enc_val):
        b = x.shape[0]

        # outputs is {b,t,k}

        mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device=x.device)).bool()

        keys = self.key_linear(x)
        queries = self.query_linear(x)
        vals = self.value_linear(x)

        result = self.masked_attention(keys,queries,vals,mask)

        x = self.norm1(x+result)

        result = self.attention(x,enc_key,enc_val,None)
        x = self.norm2(result + x)

        result = self.fc(x)
        x = self.norm3(result + x)

        return x


