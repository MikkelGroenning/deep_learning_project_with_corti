from torch.nn.utils.rnn import pad_packed_sequence
from src.models.rvae import RecurrentVariationalAutoencoder
from src.models.rae import RecurrentAutoencoder
from src.models.rvae_words import RVAEWords
from src.models.common import *
from src.data.words import *
from src.data.characters import *

import torch

rvae_w, rvae_w_tinfo = get_trained_model(RVAEWords, training_info=True)

data = torch.load("data/processed/200316_embedding.pkl")

num_samples = 10

ds = TwitterDataWords(data[-10:])
loader = get_loader(ds, batch_size=num_samples)

x = next(iter(loader))

output = rvae_w(x)

# Sample from observation model and pack, then pad
sample_packed = PackedSequence(output['px'].sample(), x.batch_sizes)
sample_padded, _ = pad_packed_sequence(sample_packed)

# word x batch x embedding_dim
print(sample_padded.shape)

for i in range(num_samples):

    decoded_tweet = sample_padded[:, i, :]