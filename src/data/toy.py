import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch.nn.utils.rnn import pack_sequence
import re
import random

class ToyData(torch.utils.data.Dataset):

    def __init__(self, num_observations, num_classes = 4, min_length=2,max_length=10, error_prob=0) -> None:

        self.num_classes = num_classes
        self.min_length = min_length
        self.max_length = max_length

        self.EOS = num_classes + 1

        values = []

        for _ in range(num_observations):
            length = random.randint(self.min_length, self.max_length)
            class_ = random.randint(1, self.num_classes)

            sequence = [class_]*length
            for i in range(length):
                if random.random() < error_prob:
                    sequence[i] = 0

            values.append( torch.tensor( sequence + [self.EOS] ))

        self.values = values


    def __len__(self):
        return len(self.values)

    def __getitem__(self, indices):

        values = [self.values[i] for i in indices]
        return pack_sequence(sorted(values, key=lambda x: -len(x)))    


class Continuous(torch.utils.data.Dataset):

    def __init__(self, num_observations, min_length=2, max_length=10, error_scale=0) -> None:

        self.min_length = min_length
        self.max_length = max_length

        self.EOS = -1.

        values = []

        for _ in range(num_observations):

            value = random.random()
            length = random.randint(self.min_length, self.max_length)

            sequence = [value]*length

            for i in range(length):
                sequence[i] += random.gauss(0, error_scale)

            sequence = torch.tensor( sequence + [self.EOS] )
            sequence = sequence.view(-1, 1)
            values.append( sequence )

        self.values = values


    def __len__(self):
        return len(self.values)

    def __getitem__(self, indices):

        values = [self.values[i] for i in indices]
        return pack_sequence(sorted(values, key=lambda x: -len(x)))        
        
if __name__ ==  "__main__":

    ds = ToyData(1000, error_prob=0.05)
    print(ds.values[:3])

    print(ds[[1, 2, 3]])