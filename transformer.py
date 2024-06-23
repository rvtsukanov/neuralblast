from typing import Any

import torch
import lightning as L
import torch.nn as nn
from torch.nn import functional as F
from data import train, test, nanoset
import pprint

from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Head(L.LightningModule):
    def __init__(self, config, _head_size):
        super().__init__()
        self.config = config
        self.key = nn.Linear(self.config.token_embedding_size, _head_size, bias=False)
        self.query = nn.Linear(self.config.token_embedding_size, _head_size, bias=False)
        self.value = nn.Linear(self.config.token_embedding_size, _head_size, bias=False)

        # Do not understand, if its constant -- should we register **-0.5 for instance?
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size,
                                                           self.config.block_size)))

    def forward(self, x):
        key = self.key(x)  # b, t, head
        query = self.query(x)  # b, t, head
        value = self.value(x)  # b, t, head
        wei = key @ query.transpose(-2, -1)  # (b, t, head) x (b, head, t) -> (b, t, t)
        wei /= self.config.block_size ** -0.5
        wei = wei.masked_fill(self.tril == 0, value=-torch.inf)
        wei = F.softmax(wei, dim=2)  # or -1 how it was in karpaty's video
        out = wei @ value  # (b, t, t) x (b, t, head) --> (b, t, head)
        return out


class MultiHead(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.config = config
        self.heads = nn.ModuleList([Head(config, config.head_size // self.num_heads) for _ in range(self.num_heads)])
        self.project = nn.Linear(self.config.head_size, self.config.pos_embedding_size)

    def forward(self, x):
        # x: (b, t, emb)
        # cat(x): (b, t, head)
        return self.project(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(self.config.token_embedding_size, 4 * self.config.token_embedding_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * self.config.token_embedding_size, self.config.token_embedding_size)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class Block(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.sa_heads = MultiHead(self.config)
        self.ff = FeedForward(self.config)
        self.ln1 = nn.LayerNorm(self.config.token_embedding_size)
        self.ln2 = nn.LayerNorm(self.config.token_embedding_size)

    def forward(self, x):
        # x: (b, t, emb)
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SelfAttention(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding_size = self.config.token_embedding_size
        self.pos_embedding_size = self.config.pos_embedding_size

        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size

        self.repeat_blocks = self.config.repeat_blocks

        self.token_embedding = nn.Embedding(self.vocab_size, self.token_embedding_size)
        self.positional_embedding = nn.Embedding(self.block_size, self.token_embedding_size)

        self.blocks = nn.Sequential(*nn.ModuleList([Block(self.config) for _ in range(self.repeat_blocks)] + \
                                                   [nn.LayerNorm(self.token_embedding_size)]))

        self.last_linear = nn.Linear(self.token_embedding_size, self.vocab_size)
        # self.last_linear = nn.Linear(self.config.token_embedding_size * self.config.block_size, self.vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        emb = self.token_embedding(x)  # b, t, emb
        position = self.positional_embedding(torch.arange(self.block_size, device=device))
        # emb + position -> b, t, emb
        concat = self.blocks(emb + position)
        # logits = self.last_linear(concat.view(-1, self.config.token_embedding_size * self.config.block_size))
        logits = self.last_linear(concat)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).transpose(-2, -1)
        # print(f'logits shape: {logits.shape}')
        # print(f'target shape: {y.shape}')
        loss = self.ce_loss(logits, y)
        self.log(f'train loss', loss)
        return loss

    def generate_text(self, max_tokens=20):
        x = torch.zeros(1, 8, device=device).long()
        sample_list = []
        for i in range(max_tokens):
            ls = self(x)[0]
            probs = F.softmax(ls[-1])
            samples = torch.multinomial(probs, num_samples=1)
            sample_list.append(samples)
            x = torch.cat((x[:, 1:], samples.unsqueeze(dim=0)), dim=1)

        decoded_row = nanoset.decode(torch.cat(sample_list).to_dense())
        return decoded_row

    def validation_step(self, batch, batch_idx):
        x, y = batch
        max_len_dataloader = len(test)
        tensorboard = self.logger.experiment
        _batch_size = x.shape[0]
        logits = self(x).transpose(-2, -1)
        loss = self.ce_loss(logits, y)
        self.log(f'val loss', loss)

        if batch_idx == max_len_dataloader - 1:
            decoded_rows = []
            for _ in range(self.config.num_generated_phrases):
                decoded_row = self.generate_text(self.config.max_tokens_generation)
                decoded_rows.append(decoded_row)
            tensorboard.add_text('validation_corpus', ' '.join(decoded_rows), self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.optimizer.lr)
        return optimizer


config = Config()
config.vocab_size = nanoset.vocab_size

pprint.pp(config)

sa = SelfAttention(config).to(device)

trainer = L.Trainer(max_epochs=config.max_epochs, enable_progress_bar=True)
trainer.fit(sa, train, test)
# trainer.fit(sa, train)

