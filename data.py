import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

config = Config()

class PushkinSet(Dataset):
    path = 'input.txt'
    def __init__(self, type='train'):
        super(PushkinSet).__init__()
        with open(self.path) as f:
            self.data = f.read()

        self.vocab = sorted(set(self.data))
        self.vocab_size = len(self.vocab)

        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        if type == 'train':
            self.data = self.data[:int(len(self.data) * config.train_test_split_ratio)]
        elif type == 'test':
            self.data = self.data[int(len(self.data) * config.train_test_split_ratio):]

        self.datatensor = torch.tensor(self.encode(self.data),
                                       dtype=torch.long)

    def encode(self, string):
        ans = []
        for ch in string:
            ans.append(self.stoi[ch])
        return ans

    def decode(self, index):
        ans = []
        for idx in index:
            ans.append(self.itos[int(idx)])
        return ''.join(ans)

    def __len__(self):
        return len(self.data) - config.block_size - 1

    def __getitem__(self, idx):
        data = self.datatensor[idx:idx + config.block_size]
        targets = self.datatensor[idx + 1:idx + 1 + config.block_size]
        return data, targets

nanoset = PushkinSet(type='train')
nanoset_test = PushkinSet(type='test')

train = DataLoader(nanoset, batch_size=config.batch_size, shuffle=True, num_workers=4)
test = DataLoader(nanoset_test, batch_size=config.batch_size, shuffle=False, num_workers=4)
