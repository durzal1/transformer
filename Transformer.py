from Encoder import Encoder
from Decoder import Decoder
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, n, k, heads, vocab_size, seq_length1, seq_length2, layers):
        super().__init__()
        self.encoder = Encoder(n,k,heads,vocab_size,seq_length1,layers)
        self.decoder = Decoder(n,k,heads,vocab_size,seq_length2,layers)
        self.fc = nn.Linear(k,vocab_size)

    def forward(self, inputs, outputs):

        enc_key, enc_val = self.encoder(inputs)
        output = self.decoder(outputs,enc_key,enc_val)

        output = self.fc(output)
        output = torch.softmax(output, dim = -1)
        return output

b=5
n = 4
k = 64
heads = 4
vocab_size = 10000
sl1 = 100
sl2 = 120
layer = 4

inputs = torch.randint(1,vocab_size,(b,sl1))
outputs = torch.randint(1,vocab_size,(b,sl2))
transformer = Transformer(n,k,heads,vocab_size,sl1,sl2,layer)
output = transformer(inputs,outputs)
print(output)
