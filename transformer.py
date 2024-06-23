import torch
import lightning as L
import torch.nn as nn
from torch.nn import functional as F
from data import train, test, nanoset

from config import Config


class Head(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass



class SelfAttention(L.LightningModule):
    def __init__(self, config, head_size=16, token_embedding_size=256, pos_embedding_size=256):
        super().__init__()
        self.config = config

        self.head_size = head_size
        self.token_embedding_size = token_embedding_size
        self.pos_embedding_size = pos_embedding_size

        self.token_embedding = nn.Embedding(self.config.vocab_size, self.token_embedding_size)
        self.positional_embedding = nn.Embedding(self.config.block_size, self.token_embedding_size)

        self.key = nn.Linear(self.token_embedding_size, self.head_size, bias=False)
        self.query = nn.Linear(self.token_embedding_size, self.head_size, bias=False)
        self.value = nn.Linear(self.token_embedding_size, self.head_size, bias=False)

        # Do not understand, if its constant -- should we register **-0.5 for instance?
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size,
                                                  self.config.block_size)))

    def forward(self, x: torch.Tensor):
        emb = self.token_embedding(x)  # b, t, voc
        position = self.positional_embedding(torch.arange(self.config.block_size))

        emb = emb + position

        key = self.key(emb)  # b, t, head
        query = self.query(emb) # b, t, head
        value = self.value(emb)  # b, t, head

        wei = key @ query.transpose(-2, -1) # (b, t, head) x (b, head, t) -> (b, t, t)
        wei /= self.config.block_size ** -0.5

        wei = wei.masked_fill(self.tril == 0, value=-torch.inf)
        wei = F.softmax(wei, dim=2) # or -1 how it was in karpaty's video

        out = wei @ value # (b, t, t) x (b, t, head) --> (b, t, head)

        return out

one_shot = next(iter(train))
one_shot_x, one_shot_y = one_shot

config = Config()
config.vocab_size = nanoset.vocab_size

print(config)
sa = SelfAttention(config)

print(sa(one_shot_x))

