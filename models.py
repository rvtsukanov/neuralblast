import torch
import lightning as L
import torch.nn as nn
from torch.nn import functional as F

from data import train, test
from config import Config

class SimpleModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.embedding_output_dim = 1024
        self.embedding = nn.Embedding(config.vocab_size, self.embedding_output_dim)
        self.lin1 = nn.Linear(self.embedding_output_dim * config.block_size, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.logits = nn.Linear(1024, config.vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.flatten(start_dim=1)
        x = F.relu(self.lin1(emb))
        x = F.relu(self.lin2(x))
        logits = self.logits(x)
        return logits

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _batch_size = x.shape[0]
        logits = self(x)
        loss = self.ce_loss(logits, y[:, 0])
        self.log(f'val loss', loss)

        for i in range(20):
            ls = sa(x)
            probs = F.softmax(ls, dim=1)
            samples = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x[:, :-1], samples), dim=1)

        random_idx = torch.randint(_batch_size, (20,))

        tensorboard = self.logger.experiment

        strings = []
        for row in x[random_idx].to_dense():
            decoded_row = config.decode(row)
            strings.append(decoded_row)

        tensorboard.add_text('validation_corpus', ' '.join(strings), self.current_epoch)

        self.log('words_of_vocab',
                 len(set(config.vocab) & set(' '.join(strings).split(' '))))
        return loss


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce_loss(logits, y[:, 0])
        self.log(f'train loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.optimizer.lr)
        return optimizer


config = Config()
print(config)

# sa = SimpleModel(config)

# trainer = L.Trainer(max_epochs=config.max_epochs)
# trainer.fit(sa, train, test)




