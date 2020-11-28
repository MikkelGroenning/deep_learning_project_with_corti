import torch
from torch.nn.utils.rnn import pack_sequence

class TwitterDataWords(torch.utils.data.Dataset):

    def __init__(self, embedded_tweets):
        self.embedded_tweets = embedded_tweets

    def __len__(self):
        return len(self.embedded_tweets)

    def __getitem__(self, indices):

        values = [self.embedded_tweets[i] for i in indices]
        return pack_sequence(sorted(values, key=lambda x: -len(x)))
